#!/usr/bin/env python3
"""Collect canonical HUSKY skateboard expert raw rollouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from skate_husky import (
    DEFAULT_FALL_CONFIRM_TIME,
    DEFAULT_FALL_ORIENTATION_LIMIT_DEG,
    DEFAULT_FALL_ROOT_HEIGHT_MIN,
    LiveFallDetector,
    fall_confirmation_steps,
    randomize_husky_play_physics,
)
from tqdm import tqdm

PHASE_LABEL_TO_ID = {
    "push": 0,
    "push2steer": 1,
    "steer_left": 2,
    "steer_right": 3,
    "steer_forward": 4,
    "steer2push": 5,
    "fall": 6,
}
PHASE_ID_TO_LABEL = {value: key for key, value in PHASE_LABEL_TO_ID.items()}
DATASET_SPLITS = ("train", "validation", "test")
COMMAND_V_RANGE = (0.0, 1.5)
COMMAND_H_RANGE = (-math.pi / 4, math.pi / 4)
BFM_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
FIXED_BFM_JOINTS = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def unqualified_name(name: str) -> str:
    return str(name).split("/")[-1]


class OfficialPhaseClock:
    """The fixed phase schedule used by the official HUSKY policy."""

    phase_ratios = (0.0, 0.4, 0.5, 0.95, 1.0)

    def __init__(self, policy_frequency: int, cycle_time: float) -> None:
        self.cycle_frames = round(policy_frequency * cycle_time)
        if self.cycle_frames <= 0 or not math.isclose(
            self.cycle_frames, policy_frequency * cycle_time, abs_tol=1e-9
        ):
            raise ValueError("policy_frequency * cycle_time must be an integer")
        self.boundaries = tuple(round(ratio * self.cycle_frames) for ratio in self.phase_ratios)
        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0

    def next(self) -> tuple[str, float]:
        frame = self.step_count % self.cycle_frames
        self.step_count += 1
        value = frame / self.cycle_frames
        p0, p1, p2, p3, _ = self.boundaries
        if frame < p1:
            return "push", value
        if frame < p2:
            return "push2steer", value
        if frame < p3:
            return "steer", value
        if frame < self.cycle_frames:
            return "steer2push", value
        return "push", value


class BoardSteerDiagnostics:
    def __init__(self, model: Any) -> None:
        self.body_id = model.body("skateboard/skateboard_deck").id
        self.previous_yaw: float | None = None
        self.heading_delta = 0.0

    def reset(self) -> None:
        self.previous_yaw = None
        self.heading_delta = 0.0

    @staticmethod
    def wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update(self, data: Any, command_h: float) -> tuple[str, float]:
        matrix = np.asarray(data.xmat[self.body_id]).reshape(3, 3)
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        if self.previous_yaw is not None:
            self.heading_delta += self.wrap(yaw - self.previous_yaw)
        self.previous_yaw = yaw
        if command_h > 0:
            direction = "left"
        elif command_h < 0:
            direction = "right"
        else:
            direction = "forward"
        return direction, -self.heading_delta


class LiveRolloutRecorder:
    """Write one complete policy-rate robot-board trajectory."""

    def __init__(
        self,
        model: Any,
        args: argparse.Namespace,
        physics_randomization: Mapping[str, Any],
    ) -> None:
        import mujoco

        self.model = model
        self.args = args
        self.physics_randomization = dict(physics_randomization)
        self.frames: dict[str, list[Any]] = defaultdict(list)
        self.phase_frame_counts: Counter[str] = Counter()
        self.phase_run_counts: Counter[str] = Counter()
        self.last_phase: str | None = None
        self.last_sim_time = 0.0
        self.terminal_reason = "viewer_closed"
        self.active = True
        self.progress_path = args.progress_file.resolve() if args.progress_file else None
        self.robot_joint_ids = [
            joint_id
            for joint_id in range(model.njnt)
            if (model.joint(joint_id).name or "").startswith("robot/")
            and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.joint_order = [model.joint(i).name for i in self.robot_joint_ids]
        self.qpos_ids = np.asarray(
            [model.jnt_qposadr[i] for i in self.robot_joint_ids], dtype=np.int64
        )
        self.qvel_ids = np.asarray(
            [model.jnt_dofadr[i] for i in self.robot_joint_ids], dtype=np.int64
        )
        self.robot_body_ids = [
            body_id
            for body_id in range(model.nbody)
            if (model.body(body_id).name or "").startswith("robot/")
        ]
        self.body_order = [model.body(i).name for i in self.robot_body_ids]
        board_root = model.joint("skateboard/floating_base_joint_skateboard")
        if board_root.type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("skateboard root joint must be free")
        self.board_qpos_root = int(model.jnt_qposadr[board_root.id])
        self.board_qvel_root = int(model.jnt_dofadr[board_root.id])
        self.board_joint_ids = [
            joint_id
            for joint_id in range(model.njnt)
            if (model.joint(joint_id).name or "").startswith("skateboard/")
            and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.board_joint_order = [model.joint(i).name for i in self.board_joint_ids]
        self.board_qpos_ids = np.asarray(
            [model.jnt_qposadr[i] for i in self.board_joint_ids], dtype=np.int64
        )
        self.board_qvel_ids = np.asarray(
            [model.jnt_dofadr[i] for i in self.board_joint_ids], dtype=np.int64
        )
        source_names = {unqualified_name(name) for name in self.joint_order}
        missing = [name for name in BFM_JOINT_ORDER if name not in source_names]
        if set(missing) != set(FIXED_BFM_JOINTS):
            raise ValueError(f"HUSKY/BFM joint audit failed: {missing}")
        self.max_policy_frames = args.max_policy_frames
        self.write_progress("collecting", "initializing", 0.0)

    @property
    def num_frames(self) -> int:
        return len(self.frames["sim_time"])

    def write_progress(
        self, status: str, phase: str | None = None, sim_time: float | None = None
    ) -> None:
        if self.progress_path is None:
            return
        current_phase = phase or self.last_phase or "initializing"
        current_time = self.last_sim_time if sim_time is None else sim_time
        write_json(
            self.progress_path,
            {
                "status": status,
                "round_id": self.args.round_id,
                "rollout_id": self.args.rollout_id,
                "episode_id": self.args.episode_id,
                "collected_frames": self.num_frames,
                "max_policy_frames": self.max_policy_frames,
                "sim_time": float(current_time),
                "phase": current_phase,
                "command_v": self.args.initial_v,
                "command_h": self.args.initial_h,
                "physics_seed": self.physics_randomization.get("seed"),
                "terminal_reason": self.terminal_reason if not self.active else None,
                "phase_frames": dict(sorted(self.phase_frame_counts.items())),
                "phase_runs": dict(sorted(self.phase_run_counts.items())),
            },
        )

    def capture(
        self,
        data: Any,
        action: np.ndarray,
        phase: str,
        phase_value: float,
        command_v: float,
        command_h: float,
        heading_delta: float,
    ) -> None:
        if not self.active:
            return
        import mujoco

        body_lin_vel = np.empty((len(self.robot_body_ids), 3), dtype=np.float32)
        body_ang_vel = np.empty((len(self.robot_body_ids), 3), dtype=np.float32)
        velocity = np.empty(6, dtype=np.float64)
        for index, body_id in enumerate(self.robot_body_ids):
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
            )
            body_ang_vel[index] = velocity[:3]
            body_lin_vel[index] = velocity[3:]
        root_joint = next(
            joint_id
            for joint_id in range(self.model.njnt)
            if (self.model.joint(joint_id).name or "").startswith("robot/")
            and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        )
        root_qpos = int(self.model.jnt_qposadr[root_joint])
        self.frames["qpos"].append(np.asarray(data.qpos, dtype=np.float32).copy())
        self.frames["qvel"].append(np.asarray(data.qvel, dtype=np.float32).copy())
        self.frames["action"].append(np.asarray(action, dtype=np.float32).copy())
        self.frames["sim_time"].append(float(data.time))
        self.frames["frame_idx"].append(self.num_frames - 1)
        self.frames["phase_id"].append(PHASE_LABEL_TO_ID[phase])
        self.frames["phase_value"].append(float(phase_value))
        self.frames["fall"].append(phase == "fall")
        self.frames["reset"].append(False)
        self.frames["root_pos"].append(
            np.asarray(data.qpos[root_qpos : root_qpos + 3], dtype=np.float32).copy()
        )
        self.frames["root_quat"].append(
            np.asarray(data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float32).copy()
        )
        self.frames["dof_pos"].append(np.asarray(data.qpos[self.qpos_ids], dtype=np.float32).copy())
        self.frames["dof_vel"].append(np.asarray(data.qvel[self.qvel_ids], dtype=np.float32).copy())
        self.frames["body_pos"].append(
            np.asarray(data.xpos[self.robot_body_ids], dtype=np.float32).copy()
        )
        self.frames["body_quat"].append(
            np.asarray(data.xquat[self.robot_body_ids], dtype=np.float32).copy()
        )
        self.frames["body_lin_vel"].append(body_lin_vel)
        self.frames["body_ang_vel"].append(body_ang_vel)
        self.frames["board_heading_delta"].append(float(heading_delta))
        self.frames["board_root_pos"].append(
            np.asarray(
                data.qpos[self.board_qpos_root : self.board_qpos_root + 3], dtype=np.float32
            ).copy()
        )
        self.frames["board_root_quat"].append(
            np.asarray(
                data.qpos[self.board_qpos_root + 3 : self.board_qpos_root + 7], dtype=np.float32
            ).copy()
        )
        self.frames["board_root_lin_vel"].append(
            np.asarray(
                data.qvel[self.board_qvel_root : self.board_qvel_root + 3], dtype=np.float32
            ).copy()
        )
        self.frames["board_root_ang_vel"].append(
            np.asarray(
                data.qvel[self.board_qvel_root + 3 : self.board_qvel_root + 6], dtype=np.float32
            ).copy()
        )
        self.frames["board_dof_pos"].append(
            np.asarray(data.qpos[self.board_qpos_ids], dtype=np.float32).copy()
        )
        self.frames["board_dof_vel"].append(
            np.asarray(data.qvel[self.board_qvel_ids], dtype=np.float32).copy()
        )
        self.frames["command_v"].append(float(command_v))
        self.frames["command_h"].append(float(command_h))
        self.phase_frame_counts[phase] += 1
        if phase != self.last_phase:
            self.phase_run_counts[phase] += 1
        self.last_phase = phase
        self.last_sim_time = float(data.time)
        if self.max_policy_frames is not None and self.num_frames >= self.max_policy_frames:
            self.terminal_reason = "max_policy_frames"
            self.active = False
        if self.num_frames % 10 == 0 or not self.active:
            self.write_progress(
                "collecting" if self.active else "collection_complete", phase, data.time
            )

    def mark_fall_and_stop(self, confirm_frames: int) -> None:
        start = max(0, self.num_frames - max(1, confirm_frames))
        for index in range(start, self.num_frames):
            self.frames["phase_id"][index] = PHASE_LABEL_TO_ID["fall"]
            self.frames["fall"][index] = True
        self.terminal_reason = "fall"
        self.active = False
        self.write_progress("collection_complete", "fall")

    def mark_reset_and_stop(self) -> None:
        if self.num_frames:
            self.frames["reset"][-1] = True
        self.terminal_reason = "reset"
        self.active = False
        self.write_progress("collection_complete", "reset")

    def arrays(self) -> dict[str, np.ndarray]:
        dtypes = {
            "qpos": np.float32,
            "qvel": np.float32,
            "action": np.float32,
            "sim_time": np.float64,
            "frame_idx": np.int64,
            "phase_id": np.int16,
            "phase_value": np.float32,
            "fall": np.bool_,
            "reset": np.bool_,
            "root_pos": np.float32,
            "root_quat": np.float32,
            "dof_pos": np.float32,
            "dof_vel": np.float32,
            "body_pos": np.float32,
            "body_quat": np.float32,
            "body_lin_vel": np.float32,
            "body_ang_vel": np.float32,
            "board_heading_delta": np.float32,
            "board_root_pos": np.float32,
            "board_root_quat": np.float32,
            "board_root_lin_vel": np.float32,
            "board_root_ang_vel": np.float32,
            "board_dof_pos": np.float32,
            "board_dof_vel": np.float32,
            "command_v": np.float32,
            "command_h": np.float32,
        }
        arrays = {
            name: np.asarray(values, dtype=dtypes[name]) for name, values in self.frames.items()
        }
        frame_count = self.num_frames
        if any(value.shape[0] != frame_count for value in arrays.values()):
            raise ValueError("raw rollout arrays are not frame aligned")
        for name, value in arrays.items():
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"raw field {name} contains NaN or Inf")
        return arrays

    def finalize(self) -> Path | None:
        if not self.num_frames:
            self.write_progress("failed")
            return None
        args = self.args
        arrays = self.arrays()
        split = args.dataset_split
        dt_values = np.diff(arrays["sim_time"])
        positive_dt = dt_values[dt_values > 0]
        dt = float(np.median(positive_dt)) if positive_dt.size else 1.0 / args.policy_frequency
        fps = 1.0 / dt
        if args.round_id is None:
            output_root = args.output_dir.resolve() / f"rollout_{args.rollout_id}"
        else:
            output_root = (
                args.output_dir.resolve() / f"round_{args.round_id}" / f"rollout_{args.rollout_id}"
            )
        raw_dir = output_root / "raw_rollout"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{args.episode_id}.npz"
        metadata_path = raw_path.with_suffix(".json")
        if (raw_path.exists() or metadata_path.exists()) and not args.overwrite:
            raise FileExistsError(f"raw rollout exists: {raw_path}")
        np.savez_compressed(raw_path, **arrays)
        phase_runs = []
        start = 0
        phase_ids = arrays["phase_id"]
        for end in range(1, len(phase_ids) + 1):
            if end == len(phase_ids) or phase_ids[end] != phase_ids[start]:
                label = PHASE_ID_TO_LABEL[int(phase_ids[start])]
                phase_runs.append((label, start, end))
                start = end
        phase_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"raw_runs": 0, "raw_frames": 0}
        )
        for label, start, end in phase_runs:
            phase_stats[label]["raw_runs"] += 1
            phase_stats[label]["raw_frames"] += end - start
        metadata = {
            "episode_id": args.episode_id,
            "round_id": args.round_id,
            "rollout_id": args.rollout_id,
            "dataset_split": split,
            "rollout_dir": str(output_root.resolve()),
            "fps": fps,
            "dt": dt,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "num_frames": self.num_frames,
            "max_policy_frames": self.max_policy_frames,
            "command_v": float(arrays["command_v"][0]),
            "command_h": float(arrays["command_h"][0]),
            "joint_order": self.joint_order,
            "body_order": self.body_order,
            "board_joint_order": self.board_joint_order,
            "qpos_quaternion_order": "wxyz",
            "body_quaternion_order": "wxyz",
            "board_quaternion_order": "wxyz",
            "phase_mapping": {str(key): value for key, value in PHASE_ID_TO_LABEL.items()},
            "robot_xml": str(args.robot_xml.resolve()),
            "policy_checkpoint": str(args.policy.resolve()),
            "physics_randomization": self.physics_randomization,
            "failure_margin_s": float(args.failure_margin),
            "fall_confirmation_s": float(args.fall_confirm_time),
            "fall_orientation_limit_deg": float(args.fall_orientation_deg),
            "fall_root_height_min": float(args.fall_root_height_min),
            "action_alignment": "action[t] is the previous policy output before state[t]",
            "body_velocity_frame": "world",
            "board_heading_delta_unit": "radian",
            "terminal_reason": self.terminal_reason,
            "fixed_bfm_joints": list(FIXED_BFM_JOINTS),
            "phase_statistics": dict(sorted(phase_stats.items())),
            "fields": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        }
        write_json(metadata_path, metadata)
        self.write_progress("completed")
        print(f"\nRaw rollout: {raw_path}")
        print(f"Raw frames: {self.num_frames}; terminal: {self.terminal_reason}")
        return raw_path


def load_upstream_sim(robot_xml: Path, headless: bool) -> Any:
    sim_path = robot_xml.resolve().parent / "sim.py"
    spec = importlib.util.spec_from_file_location("skate_bfm_upstream_sim", sim_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load HUSKY sim: {sim_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if not headless:
        spec.loader.exec_module(module)
        return module

    class NoopListener:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> NoopListener:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def join(self) -> None:
            return None

    fake_keyboard = types.ModuleType("pynput.keyboard")
    fake_keyboard.Listener = NoopListener
    fake_keyboard.Key = types.SimpleNamespace()
    fake_pynput = types.ModuleType("pynput")
    fake_pynput.keyboard = fake_keyboard
    saved = {name: sys.modules.get(name) for name in ("pynput", "pynput.keyboard")}
    sys.modules["pynput"] = fake_pynput
    sys.modules["pynput.keyboard"] = fake_keyboard
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


class HeadlessViewer:
    class Camera:
        def __init__(self) -> None:
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0
            self.lookat = np.zeros(3, dtype=np.float64)

    def __init__(self) -> None:
        self.cam = self.Camera()
        self.running = True

    def is_running(self) -> bool:
        return self.running

    def sync(self) -> None:
        return

    def close(self) -> None:
        self.running = False


class HeadlessViewerModule:
    @staticmethod
    def launch_passive(*_args: Any, **_kwargs: Any) -> HeadlessViewer:
        return HeadlessViewer()


def clamp_commands(sim_module: Any) -> tuple[float, float]:
    command_v = float(np.clip(float(sim_module.v), *COMMAND_V_RANGE))
    command_h = float(np.clip(float(sim_module.h), *COMMAND_H_RANGE))
    sim_module.v, sim_module.h = command_v, command_h
    return command_v, command_h


def run_live(args: argparse.Namespace) -> int:
    robot_xml = args.robot_xml.resolve()
    policy = args.policy.resolve()
    if not robot_xml.is_file() or not policy.is_file():
        raise FileNotFoundError(f"missing robot XML or policy: {robot_xml}, {policy}")
    sim_module = load_upstream_sim(robot_xml, args.headless)
    if args.headless:
        sim_module.mjv = HeadlessViewerModule
    if args.initial_v is not None:
        sim_module.v = args.initial_v
    if args.initial_h is not None:
        sim_module.h = args.initial_h

    class LiveController(sim_module.RealTimePolicyController):
        def __init__(self, *controller_args: Any, **controller_kwargs: Any) -> None:
            super().__init__(*controller_args, **controller_kwargs)
            if args.randomize_physics:
                self.physics_randomization, self.initial_joint_offsets = (
                    randomize_husky_play_physics(self.model, args.rollout_id, args.physics_seed)
                )
                import mujoco

                mujoco.mj_setConst(self.model, self.data)
            else:
                self.physics_randomization = {"enabled": False, "mode": "nominal_test_scene_xml"}
                self.initial_joint_offsets = {}
            self.phase_clock = OfficialPhaseClock(args.policy_frequency, args.cycle_time)
            self.steer = BoardSteerDiagnostics(self.model)
            self.fall_detector = LiveFallDetector(
                self.model,
                args.fall_orientation_deg,
                args.fall_root_height_min,
                fall_confirmation_steps(args.fall_confirm_time, 1.0 / args.policy_frequency),
            )
            self.recorder = (
                LiveRolloutRecorder(self.model, args, self.physics_randomization)
                if args.record
                else None
            )

        def reset(self, init_pos: np.ndarray) -> None:
            if self.recorder is not None and self.recorder.num_frames:
                self.recorder.mark_reset_and_stop()
            randomized = np.asarray(init_pos, dtype=np.float64).copy()
            for joint_name, offset in self.initial_joint_offsets.items():
                joint = self.model.joint(joint_name)
                randomized[self.model.jnt_qposadr[joint.id]] += offset
            super().reset(randomized)
            self.phase_clock.reset()
            self.steer.reset()
            self.fall_detector.reset()

        def extract_data(self) -> Any:
            command_v, command_h = clamp_commands(sim_module)
            values = super().extract_data()
            phase, phase_value = self.phase_clock.next()
            fallen, _, _ = self.fall_detector.check(self.data)
            direction, heading_delta = self.steer.update(self.data, command_h)
            if fallen:
                phase = "fall"
            elif phase == "steer":
                phase = f"steer_{direction}"
            if self.recorder is not None:
                self.recorder.capture(
                    self.data,
                    self.last_action,
                    phase,
                    phase_value,
                    command_v,
                    command_h,
                    heading_delta,
                )
                if fallen:
                    self.recorder.mark_fall_and_stop(self.fall_detector.confirm_frames)
                elif not self.recorder.active and self.viewer is not None:
                    self.viewer.close()
            if fallen and self.viewer is not None:
                self.viewer.close()
            return values

    controller = LiveController(
        xml_file=str(robot_xml),
        policy_path=str(policy),
        device=args.device,
        policy_frequency=args.policy_frequency,
    )
    controller.run()
    if controller.recorder is not None:
        controller.recorder.finalize()
    return 0


def round_grid_assignments(
    headings: Sequence[float],
    velocities: Sequence[float],
    round_count: int,
    rollouts_per_round: int,
    seed: int,
    round_offset: int = 0,
) -> list[tuple[float, float]]:
    if not headings or not velocities:
        return []
    rng = np.random.default_rng(seed)
    assignments: list[tuple[float, float]] = []
    for local_round in range(round_count):
        round_index = round_offset + local_round
        batch = [
            (
                velocities[
                    (slot % len(headings) + slot // len(headings) + round_index) % len(velocities)
                ],
                headings[slot % len(headings)],
            )
            for slot in range(rollouts_per_round)
        ]
        assignments.extend(batch[int(index)] for index in rng.permutation(len(batch)))
    return assignments


def collection_job_summary(job: Mapping[str, Any], policy_frequency: int) -> dict[str, Any] | None:
    raw_dir = job["rollout_root"] / "raw_rollout"
    metadata_files = sorted(raw_dir.glob("*.json"))
    if len(metadata_files) != 1:
        return None
    try:
        payload = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        raw_path = metadata_files[0].with_suffix(".npz")
        if not raw_path.is_file() or int(payload["num_frames"]) <= 0:
            return None
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    return {
        "round_id": job["round_id"],
        "rollout_id": job["rollout_id"],
        "episode_id": job["episode_id"],
        "command_v": job["velocity"],
        "command_h": job["heading"],
        "physics_seed": job["physics_seed"],
        "terminal_reason": payload.get("terminal_reason", "unknown"),
        "raw_frames": int(payload["num_frames"]),
        "raw_duration_seconds": int(payload["num_frames"]) / policy_frequency,
        "metadata_path": str(metadata_files[0]),
        "phase_statistics": payload.get("phase_statistics", {}),
    }


def run_parallel(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    output_root = args.output_dir.resolve()
    baseline_count = args.round_count * args.rollouts_per_round
    assignments = round_grid_assignments(
        args.parallel_heading_values,
        args.parallel_velocity_values,
        args.round_count + args.max_extra_rounds,
        args.rollouts_per_round,
        args.plan_seed,
    )
    round_start = int(args.round_id)
    jobs = []
    for index, (velocity, heading) in enumerate(assignments):
        round_id = str(round_start + index // args.rollouts_per_round).zfill(3)
        rollout_id = str(args.parallel_rollout_start + index % args.rollouts_per_round).zfill(3)
        episode_id = (
            f"round{round_id}_rollout{rollout_id}_"
            f"v{round(velocity * 100):03d}_h_{heading_name(heading)}"
        )
        seed = (
            args.physics_seed + index
            if args.physics_seed is not None
            else int.from_bytes(
                hashlib.sha256(f"parallel:{round_id}:{rollout_id}".encode()).digest()[:4], "big"
            )
        )
        root = output_root / f"round_{round_id}" / f"rollout_{rollout_id}"
        jobs.append(
            {
                "round_id": round_id,
                "rollout_id": rollout_id,
                "episode_id": episode_id,
                "heading": heading,
                "velocity": velocity,
                "physics_seed": seed,
                "baseline": index < baseline_count,
                "rollout_root": root,
                "progress_path": root / "collection_progress.json",
                "displayed_frames": 0,
                "latest_progress": {},
            }
        )
    target_frames = (
        round(args.target_raw_minutes * 60 * args.policy_frequency)
        if args.target_raw_minutes is not None
        else baseline_count * args.max_policy_frames
    )
    plan_path = output_root / "collection_plan.json"
    write_json(
        plan_path,
        {
            "target_raw_minutes": args.target_raw_minutes,
            "target_raw_frames": target_frames,
            "policy_frequency": args.policy_frequency,
            "max_policy_frames_per_rollout": args.max_policy_frames,
            "planned_rollout_seconds": args.max_policy_frames / args.policy_frequency,
            "round_start": str(round_start).zfill(3),
            "baseline_rounds": args.round_count,
            "max_extra_rounds": args.max_extra_rounds,
            "extra_rollout_capacity": len(jobs) - baseline_count,
            "rollouts_per_round": args.rollouts_per_round,
            "baseline_rollouts": baseline_count,
            "parallel_workers": args.parallel_workers,
            "headings": args.parallel_heading_values,
            "velocities": args.parallel_velocity_values,
            "plan_seed": args.plan_seed,
            "policy_checkpoint": str(args.policy.resolve()),
            "robot_xml": str(args.robot_xml.resolve()),
            "jobs": [
                {
                    "round_id": job["round_id"],
                    "rollout_id": job["rollout_id"],
                    "episode_id": job["episode_id"],
                    "command_v": job["velocity"],
                    "command_h": job["heading"],
                    "physics_seed": job["physics_seed"],
                    "baseline": job["baseline"],
                    "output": str(job["rollout_root"]),
                }
                for index, job in enumerate(jobs)
            ],
        },
    )
    command_cells = len(args.parallel_heading_values) * len(args.parallel_velocity_values)
    extra_capacity = len(jobs) - baseline_count
    target_minutes = target_frames / args.policy_frequency / 60
    maximum_minutes = len(jobs) * args.max_policy_frames / args.policy_frequency / 60
    print("[Collection Plan]")
    print(f"Raw root: {output_root}")
    print(f"Velocity grid: {args.parallel_velocity_values}")
    print(f"Heading grid: {args.parallel_heading_values}")
    print(f"Command cells: {command_cells}")
    print(f"Baseline rollouts: {baseline_count}")
    print(f"Extra replacement capacity: {extra_capacity}")
    print(f"Parallel workers: {args.parallel_workers}")
    print(f"Frames / rollout: {args.max_policy_frames}")
    print(f"FPS: {args.policy_frequency}")
    print(f"Target raw: {target_minutes:.2f} min")
    print(f"Maximum nominal capacity: {maximum_minutes:.2f} min")
    print(f"Collection plan: {plan_path}")
    completed: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    if not args.overwrite:
        for job in jobs:
            record = collection_job_summary(job, args.policy_frequency)
            if record is not None:
                record["replacement"] = not job["baseline"]
                completed[job["episode_id"]] = record
                job["displayed_frames"] = record["raw_frames"]
    progress = tqdm(
        total=target_frames,
        initial=min(target_frames, sum(r["raw_frames"] for r in completed.values())),
        desc="M2.5c-P raw collection",
        unit="frame",
        dynamic_ncols=True,
    )

    def summary() -> dict[str, Any]:
        records = list(completed.values())
        raw_frames = sum(r["raw_frames"] for r in records)
        phase_statistics: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            for label, stats in record.get("phase_statistics", {}).items():
                for key, value in stats.items():
                    phase_statistics[label][key] += int(value)
        return {
            "target_raw_minutes": args.target_raw_minutes,
            "target_raw_frames": target_frames,
            "target_achieved": raw_frames >= target_frames,
            "baseline_planned": baseline_count,
            "baseline_completed": sum(
                1 for record in records if not record.get("replacement", False)
            ),
            "extra_rollout_capacity": extra_capacity,
            "replacement_completed": sum(
                1 for record in records if record.get("replacement", False)
            ),
            "planned_rollouts": len(jobs),
            "completed_rollouts": len(records),
            "failed_rollouts": sorted(failed),
            "failed_rollout_count": len(failed),
            "replacement_rollouts": sum(
                1 for record in records if record.get("replacement", False)
            ),
            "raw_frames": raw_frames,
            "raw_seconds": raw_frames / args.policy_frequency,
            "raw_minutes": raw_frames / args.policy_frequency / 60,
            "raw_duration_seconds": raw_frames / args.policy_frequency,
            "raw_duration_minutes": raw_frames / args.policy_frequency / 60,
            "terminal_reasons": dict(Counter(r["terminal_reason"] for r in records)),
            "phase_statistics": {
                label: dict(sorted(stats.items()))
                for label, stats in sorted(phase_statistics.items())
            },
            "records": sorted(records, key=lambda r: (r["round_id"], r["rollout_id"])),
        }

    def write_summary() -> dict[str, Any]:
        payload = summary()
        write_json(output_root / "collection_summary.json", payload)
        return payload

    write_summary()

    def launch(job: dict[str, Any]) -> None:
        job["rollout_root"].mkdir(parents=True, exist_ok=True)
        write_json(
            job["progress_path"],
            {
                "status": "starting",
                "round_id": job["round_id"],
                "rollout_id": job["rollout_id"],
                "episode_id": job["episode_id"],
                "collected_frames": 0,
                "max_policy_frames": args.max_policy_frames,
                "phase": "initializing",
                "command_v": job["velocity"],
                "command_h": job["heading"],
                "physics_seed": job["physics_seed"],
            },
        )
        log = (job["rollout_root"] / "collection.log").open("w", encoding="utf-8")
        log.write(
            "[Worker Startup]\n"
            f"episode_id={job['episode_id']}\n"
            f"round={job['round_id']}\n"
            f"rollout={job['rollout_id']}\n"
            f"command_v={job['velocity']}\n"
            f"command_h={job['heading']}\n"
            f"physics_seed={job['physics_seed']}\n"
            f"policy={args.policy.resolve()}\n"
            f"robot_xml={args.robot_xml.resolve()}\n\n"
        )
        log.flush()
        command = [
            sys.executable,
            str(script),
            "--live",
            "--record",
            "--headless",
            "--robot-xml",
            str(args.robot_xml),
            "--policy",
            str(args.policy),
            "--device",
            args.device,
            "--round-id",
            job["round_id"],
            "--rollout-id",
            job["rollout_id"],
            "--episode-id",
            job["episode_id"],
            "--dataset-split",
            args.dataset_split,
            "--randomize-physics",
            "--physics-seed",
            str(job["physics_seed"]),
            "--initial-v",
            str(job["velocity"]),
            "--initial-h",
            str(job["heading"]),
            "--max-policy-frames",
            str(args.max_policy_frames),
            "--policy-frequency",
            str(args.policy_frequency),
            "--cycle-time",
            str(args.cycle_time),
            "--fall-orientation-deg",
            str(args.fall_orientation_deg),
            "--fall-root-height-min",
            str(args.fall_root_height_min),
            "--fall-confirm-time",
            str(args.fall_confirm_time),
            "--failure-margin",
            str(args.failure_margin),
            "--output-dir",
            str(args.output_dir),
            "--progress-file",
            str(job["progress_path"]),
        ]
        if args.overwrite:
            command.append("--overwrite")
        handle = log
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
        job["process"], job["log_handle"] = process, handle
        tqdm.write(
            f"Started round_{job['round_id']}/"
            f"rollout_{job['rollout_id']}: "
            f"v={job['velocity']:.2f}, h={job['heading']:+.2f}, "
            f"seed={job['physics_seed']}, pid={process.pid}"
        )

    def run_batch(batch: Sequence[dict[str, Any]]) -> None:
        queue = [job for job in batch if args.overwrite or job["episode_id"] not in completed]
        active: list[dict[str, Any]] = []
        next_index = 0
        while next_index < len(queue) or active:
            while next_index < len(queue) and len(active) < args.parallel_workers:
                launch(queue[next_index])
                active.append(queue[next_index])
                next_index += 1
            worker_states = []
            for job in list(active):
                try:
                    payload = json.loads(job["progress_path"].read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = job["latest_progress"]
                job["latest_progress"] = payload
                captured = min(int(payload.get("collected_frames", 0)), args.max_policy_frames)
                increment = max(0, captured - job["displayed_frames"])
                progress.update(min(increment, max(0, target_frames - progress.n)))
                job["displayed_frames"] = max(job["displayed_frames"], captured)
                worker_states.append(
                    f"{job['round_id']}/{job['rollout_id']}:"
                    f"{captured}/{args.max_policy_frames} "
                    f"{payload.get('phase', 'starting')}"
                )
                process = job["process"]
                if process.poll() is None:
                    continue
                process.wait()
                active.remove(job)
                record = collection_job_summary(job, args.policy_frequency)
                if process.returncode != 0 or record is None:
                    failure = f"round_{job['round_id']}/rollout_{job['rollout_id']}"
                    failed.append(failure)
                    job["log_handle"].write(
                        f"\n[Worker Complete]\nreturncode={process.returncode}\n"
                        "terminal_reason=failed_or_missing_raw\n"
                    )
                    tqdm.write(f"Failed {failure}: returncode={process.returncode}")
                else:
                    record["replacement"] = not job["baseline"]
                    completed[job["episode_id"]] = record
                    job["log_handle"].write(
                        f"\n[Worker Complete]\nreturncode={process.returncode}\n"
                        f"terminal_reason={record['terminal_reason']}\n"
                    )
                    tqdm.write(
                        f"Finished round_{job['round_id']}/"
                        f"rollout_{job['rollout_id']}: "
                        f"raw={record['raw_duration_seconds']:.2f}s"
                    )
                    write_summary()
                job["log_handle"].close()
            now = summary()
            current_minutes = progress.n / args.policy_frequency / 60
            worker_text = ", ".join(worker_states)
            progress.set_postfix_str(
                f"raw={current_minutes:.1f}/{target_minutes:.1f}min | "
                f"done={now['baseline_completed']}/{baseline_count} | "
                f"failed={now['failed_rollout_count']} | "
                f"replacement={now['replacement_completed']} | "
                f"active={len(active)}" + (f" | {worker_text}" if worker_text else ""),
                refresh=True,
            )
            if active:
                time.sleep(0.2)

    try:
        run_batch(jobs[:baseline_count])
        result = write_summary()
        extra_index = baseline_count
        while result["raw_frames"] < target_frames and extra_index < len(jobs):
            needed = max(
                1,
                math.ceil((target_frames - result["raw_frames"]) / args.max_policy_frames),
            )
            run_batch(jobs[extra_index : extra_index + needed])
            extra_index += needed
            result = write_summary()
        progress.close()
        print(f"Collection summary: {output_root / 'collection_summary.json'}")
        print(
            f"Raw duration: {result['raw_duration_minutes']:.3f} min "
            f"({result['raw_frames']} frames)"
        )
        print(f"Completed rollouts: {result['completed_rollouts']}/{result['planned_rollouts']}")
        print(f"Failed rollouts: {result['failed_rollout_count']}")
        print(f"Replacement rollouts: {result['replacement_completed']}")
        print(f"Target achieved: {result['target_achieved']}")
        return 0 if result["target_achieved"] else 2
    except KeyboardInterrupt:
        for job in jobs:
            process = job.get("process")
            if process is not None and process.poll() is None:
                process.terminate()
        progress.close()
        write_summary()
        return 130


def heading_name(value: float) -> str:
    direction = "pos" if value > 0 else "neg" if value < 0 else "zero"
    return f"{direction}{round(abs(value) * 100):03d}"


def apply_parallel_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.parallel_config is None:
        return
    path = args.parallel_config.resolve()
    if not path.is_file():
        parser.error(f"--parallel-config does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"unable to load --parallel-config: {error}")
    fields = {
        "round_start": "round_id",
        "round_count": "round_count",
        "rollouts_per_round": "rollouts_per_round",
        "parallel_workers": "parallel_workers",
        "rollout_start": "parallel_rollout_start",
        "headings": "parallel_headings",
        "velocities": "parallel_velocities",
        "target_raw_minutes": "target_raw_minutes",
        "max_extra_rounds": "max_extra_rounds",
        "plan_seed": "plan_seed",
        "max_policy_frames": "max_policy_frames",
        "physics_seed_start": "physics_seed",
        "device": "device",
        "dataset_split": "dataset_split",
        "output_dir": "output_dir",
        "robot_xml": "robot_xml",
        "policy": "policy",
        "overwrite": "overwrite",
    }
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        parser.error(f"unknown --parallel-config fields: {unknown}")
    for name, attribute in fields.items():
        if name not in payload:
            continue
        value = payload[name]
        if name in {"headings", "velocities"}:
            value = ",".join(str(item) for item in value)
        elif name in {"output_dir", "robot_xml", "policy"}:
            value = Path(value)
            if not value.is_absolute():
                value = path.parent / value
        elif name == "round_start":
            value = str(value)
        setattr(args, attribute, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--parallel-config",
        nargs="?",
        const=Path(__file__).with_name("rollout_config.json"),
        type=Path,
    )
    parser.add_argument("--parallel-headings")
    parser.add_argument("--parallel-velocities")
    parser.add_argument("--parallel-rollout-start", type=int, default=1)
    parser.add_argument("--round-id")
    parser.add_argument("--round-count", type=int, default=1)
    parser.add_argument("--rollouts-per-round", type=int)
    parser.add_argument("--parallel-workers", type=int)
    parser.add_argument("--target-raw-minutes", type=float)
    parser.add_argument("--max-extra-rounds", type=int, default=0)
    parser.add_argument("--plan-seed", type=int, default=20260804)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--rollout-id")
    parser.add_argument("--dataset-split", choices=("auto", *DATASET_SPLITS), default="auto")
    parser.add_argument("--randomize-physics", action="store_true")
    parser.add_argument("--physics-seed", type=int)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--progress-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-frequency", type=int, default=50)
    parser.add_argument("--cycle-time", type=float, default=6.0)
    parser.add_argument("--initial-v", type=float)
    parser.add_argument("--initial-h", type=float)
    parser.add_argument(
        "--fall-orientation-deg", type=float, default=DEFAULT_FALL_ORIENTATION_LIMIT_DEG
    )
    parser.add_argument("--fall-root-height-min", type=float, default=DEFAULT_FALL_ROOT_HEIGHT_MIN)
    parser.add_argument("--fall-confirm-time", type=float, default=DEFAULT_FALL_CONFIRM_TIME)
    parser.add_argument("--failure-margin", type=float, default=0.15)
    parser.add_argument("--max-policy-frames", type=int, default=3000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    apply_parallel_config(args, parser)
    if args.parallel_config is not None and args.parallel_headings is None:
        args.parallel_headings = ",".join(
            str(value) for value in json.loads(args.parallel_config.read_text())["headings"]
        )
        args.parallel_velocities = ",".join(
            str(value) for value in json.loads(args.parallel_config.read_text())["velocities"]
        )
    if args.parallel_headings is not None:
        args.parallel_heading_values = [
            float(value) for value in args.parallel_headings.split(",") if value.strip()
        ]
        args.parallel_velocity_values = [
            float(value)
            for value in (args.parallel_velocities or "1.0").split(",")
            if value.strip()
        ]
        args.rollouts_per_round = args.rollouts_per_round or len(args.parallel_heading_values)
        args.parallel_workers = args.parallel_workers or min(
            args.rollouts_per_round, len(args.parallel_heading_values)
        )
        args.max_extra_rounds = args.max_extra_rounds or 0
        if args.round_id is None:
            args.round_id = "001"
        args.live = args.record = args.headless = args.randomize_physics = True
        repo_root = Path(__file__).resolve().parents[3]
        args.output_dir = (
            args.output_dir
            or repo_root / "train" / "dataset" / "sim_collected" / "raw"
        )
        args.robot_xml = (
            args.robot_xml or repo_root / "husky_sim/upstream/test_scene/mjlab_scene.xml"
        )
        args.policy = args.policy or repo_root / "husky_sim/upstream/ckpts/test.onnx"
        args.dataset_split = "train" if args.dataset_split == "auto" else args.dataset_split
        args.initial_v = 1.0 if args.initial_v is None else args.initial_v
        args.episode_id = "parallel_parent"
        args.rollout_id = str(args.parallel_rollout_start)
        if args.target_raw_minutes is not None:
            max_minutes = (
                (args.round_count + args.max_extra_rounds)
                * args.rollouts_per_round
                * args.max_policy_frames
                / args.policy_frequency
                / 60
            )
            if args.target_raw_minutes > max_minutes:
                parser.error(f"collection plan can provide at most {max_minutes:.2f} raw minutes")
    if args.live:
        if args.robot_xml is None or args.policy is None:
            parser.error("--live requires --robot-xml and --policy")
        if args.record and (args.output_dir is None or args.episode_id is None):
            parser.error("--live --record requires --output-dir and --episode-id")
        args.rollout_id = str(
            args.rollout_id or re.search(r"(\d+)$", args.episode_id or "").group(1)
            if re.search(r"(\d+)$", args.episode_id or "")
            else "001"
        ).zfill(3)
        if args.round_id is not None:
            args.round_id = str(args.round_id).zfill(3)
        if args.initial_v is None:
            args.initial_v = 1.0
        if (
            not COMMAND_V_RANGE[0] <= args.initial_v <= COMMAND_V_RANGE[1]
            or not COMMAND_H_RANGE[0] <= (args.initial_h or 0.0) <= COMMAND_H_RANGE[1]
        ):
            parser.error("initial commands are outside HUSKY command ranges")
        if args.physics_seed is not None and not args.randomize_physics:
            parser.error("--physics-seed requires --randomize-physics")
        if args.max_policy_frames <= 0 or args.policy_frequency <= 0:
            parser.error("max-policy-frames and policy-frequency must be positive")
    elif args.parallel_headings is None:
        parser.error("use --parallel-config or --live --record")
    return args


def main() -> int:
    args = parse_args()
    if args.parallel_headings is not None:
        return run_parallel(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
