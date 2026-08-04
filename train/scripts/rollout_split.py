#!/usr/bin/env python3
"""Split one HUSKY rollout into command-labelled motion clips.

The script preserves the source rollout, slices only existing frame-aligned
fields, and refuses expert export when failure detection is not reliable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

UNKNOWN_FAILURE_DETECTION = "UNKNOWN_FAILURE_DETECTION"
NATIVE_FAILURE_FIELDS = (
    "terminated",
    "reset_buf",
    "reset_terminated",
    "fall",
    "fallen",
    "termination",
    "termination_condition",
    "forbidden_contact",
    "illegal_contact",
)
RESET_FIELDS = ("reset_buf", "reset", "reset_flag", "resets")
TIME_FIELDS = ("sim_time", "simulation_time", "time", "timestamps")
COMMAND_FIELDS = ("high_level_command", "command", "commands", "command_history")
QPOS_FIELDS = ("qpos", "joint_qpos")
ROOT_POS_FIELDS = ("root_pos", "root_position", "base_pos", "base_position")
ROOT_QUAT_FIELDS = ("root_quat", "root_quaternion", "base_quat", "base_quaternion")

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
HUSKY_ROBOT_COM_RANGES = (
    (-0.025, 0.025),
    (-0.025, 0.025),
    (-0.03, 0.03),
)
HUSKY_SKATEBOARD_COM_RANGES = (
    (-0.02, 0.02),
    (-0.02, 0.02),
    (-0.01, 0.01),
)
HUSKY_ROBOT_FRICTION_SCALE_RANGE = (0.3, 1.6)
HUSKY_DECK_FRICTION_SCALE_RANGE = (0.8, 2.0)
HUSKY_FOOT_FRICTION_RANGE = (0.3, 1.8)
HUSKY_WHEEL_FRICTION_SCALE_RANGE = (0.8, 1.6)
HUSKY_JOINT_POSITION_OFFSET_RANGE = (-0.01, 0.01)

BFM_29_JOINT_ORDER = (
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
BFM_FIXED_WRIST_JOINTS = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

SEGMENT_FIELDS = (
    "segment_id",
    "episode_id",
    "motion_label",
    "key",
    "command",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration",
    "num_frames",
    "pose_path",
    "video_path",
    "status",
    "failure_detected",
    "failure_frame",
    "failure_reason",
    "truncated_by_failure",
    "original_end_frame",
    "valid_end_frame",
    "reset_detected",
    "discard_reason",
    "notes",
)


@dataclass
class Rollout:
    path: Path
    suffix: str
    payload: Any
    top_type: str
    keys: list[str]
    shape: list[int] | None
    dtype: str | None
    num_frames: int


@dataclass
class KeyEvent:
    frame_idx: int
    sim_time: float | None
    key: str
    event_type: str
    command: str | None


@dataclass
class Span:
    command: str
    key: str
    start: int
    end: int


@dataclass
class Segment:
    segment_id: str
    episode_id: str
    motion_label: str
    key: str
    command: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    num_frames: int
    pose_path: str = ""
    video_path: str = ""
    status: str = "valid"
    failure_detected: bool = False
    failure_frame: int | None = None
    failure_reason: str = ""
    truncated_by_failure: bool = False
    original_end_frame: int | None = None
    valid_end_frame: int | None = None
    reset_detected: bool = False
    discard_reason: str = ""
    notes: str = ""


class KeySegmentRecorder:
    """Minimal event recorder suitable for calling from a keyboard callback."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def on_key_event(
        self,
        key: str,
        frame_idx: int,
        sim_time: float,
        event_type: str,
        command: str | None = None,
    ) -> None:
        event_type = normalize_event_type(event_type)
        if event_type not in {"key_down", "key_up"}:
            raise ValueError(f"unsupported event_type: {event_type}")
        self.events.append(
            {
                "frame_idx": int(frame_idx),
                "sim_time": float(sim_time),
                "key": normalize_key(key),
                "event_type": event_type,
                "command": command or "",
            }
        )

    def finalize(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        events = sorted(self.events, key=lambda item: item["frame_idx"])
        if path.suffix.lower() == ".jsonl":
            with path.open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
            return
        if path.suffix.lower() != ".csv":
            raise ValueError("KeySegmentRecorder output must be .csv or .jsonl")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("frame_idx", "sim_time", "key", "event_type", "command"),
            )
            writer.writeheader()
            writer.writerows(events)


class OfficialPhaseClock:
    """The fixed HUSKY phase schedule used by the official environment."""

    phase_ratios = (0.0, 0.4, 0.5, 0.95, 1.0)

    def __init__(self, policy_frequency: int, cycle_time: float = 6.0) -> None:
        self.policy_frequency = policy_frequency
        self.cycle_time = cycle_time
        self.cycle_frames = round(policy_frequency * cycle_time)
        if not math.isclose(
            self.cycle_frames,
            policy_frequency * cycle_time,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "policy_frequency * cycle_time must be an integer number of frames"
            )
        self.boundaries = tuple(
            round(ratio * self.cycle_frames) for ratio in self.phase_ratios
        )
        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0

    def next(self) -> tuple[str, float]:
        frame_in_cycle = self.step_count % self.cycle_frames
        self.step_count += 1
        phase_value = frame_in_cycle / self.cycle_frames
        p0, p1, p2, p3, p4 = self.boundaries
        if p0 <= frame_in_cycle < p1:
            label = "push"
        elif p1 <= frame_in_cycle < p2:
            label = "push2steer"
        elif p2 <= frame_in_cycle < p3:
            label = "steer"
        elif p3 <= frame_in_cycle < p4:
            label = "steer2push"
        else:
            label = "push"
        return label, phase_value


class BoardSteerDirection:
    """Track board heading while deriving steering direction only from h."""

    def __init__(self, model: Any) -> None:
        self.previous_board_yaw: float | None = None
        self.board_heading_delta = 0.0
        self.board_body_id = model.body("skateboard/skateboard_deck").id

    def reset(self) -> None:
        self.previous_board_yaw = None
        self.board_heading_delta = 0.0

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def classify(self, data: Any, command_h: float = 0.0) -> tuple[str, dict[str, Any]]:
        board_mat = data.xmat[self.board_body_id].reshape(3, 3)
        board_yaw = math.atan2(board_mat[1, 0], board_mat[0, 0])
        if self.previous_board_yaw is not None:
            self.board_heading_delta += self.wrap_angle(board_yaw - self.previous_board_yaw)
        self.previous_board_yaw = board_yaw

        if command_h > 0.0:
            direction = "left"
        elif command_h < 0.0:
            direction = "right"
        else:
            direction = "forward"

        heading_delta_deg = -math.degrees(self.board_heading_delta)
        if abs(heading_delta_deg) < 0.005:
            heading_delta_deg = 0.0
        return direction, {
            "board_heading_delta_deg": heading_delta_deg,
            "board_heading_delta_rad": -self.board_heading_delta,
        }


class LiveFallDetector:
    """Detect a persistent fall without treating foot lift-off as a fall."""

    _illegal_geom = re.compile(
        r"(left|right)_(shin|linkage_brace|shoulder_yaw|elbow_yaw|wrist|hand)_collision"
        r"|robot/pelvis_collision$"
    )

    def __init__(
        self,
        model: Any,
        orientation_limit_deg: float,
        root_height_min: float,
        confirm_frames: int,
    ) -> None:
        self.model = model
        self.orientation_limit_deg = orientation_limit_deg
        self.root_height_min = root_height_min
        self.confirm_frames = max(1, confirm_frames)
        self.bad_frames = 0
        self.foot_geoms: set[int] = set()
        self.board_geoms: set[int] = set()
        self.illegal_geoms: set[int] = set()
        import mujoco

        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            if re.search(r"robot/(left|right)_foot[0-9]+_collision$", name):
                self.foot_geoms.add(geom_id)
            if name == "skateboard/skateboard_deck_collision":
                self.board_geoms.add(geom_id)
            if self._illegal_geom.search(name):
                self.illegal_geoms.add(geom_id)

    def reset(self) -> None:
        self.bad_frames = 0

    def check(self, data: Any) -> tuple[bool, list[str], dict[str, Any]]:
        qw, qx, qy, qz = np.asarray(data.qpos[3:7], dtype=float)
        norm = np.linalg.norm((qw, qx, qy, qz))
        if norm <= 0.0:
            tilt_deg = 180.0
        else:
            qw, qx, qy, qz = np.asarray((qw, qx, qy, qz)) / norm
            gravity_z = 1.0 - 2.0 * (qx * qx + qy * qy)
            tilt_deg = math.degrees(math.acos(np.clip(gravity_z, -1.0, 1.0)))

        feet_on_board = False
        illegal_contact = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom_a, geom_b = contact.geom1, contact.geom2
            if (geom_a in self.foot_geoms and geom_b in self.board_geoms) or (
                geom_b in self.foot_geoms and geom_a in self.board_geoms
            ):
                feet_on_board = True
            if geom_a in self.illegal_geoms or geom_b in self.illegal_geoms:
                illegal_contact = True
        root_height = float(data.qpos[2])
        severe_tilt = tilt_deg > self.orientation_limit_deg
        low_contact_fall = root_height < self.root_height_min and illegal_contact
        candidate = severe_tilt or low_contact_fall
        self.bad_frames = self.bad_frames + 1 if candidate else 0
        reasons = []
        if severe_tilt:
            reasons.append(f"tilt>{self.orientation_limit_deg:.0f}deg")
        if low_contact_fall:
            reasons.append(f"height<{self.root_height_min:.2f}+illegal_contact")
        diagnostics = {
            "tilt_deg": tilt_deg,
            "root_height": root_height,
            "feet_on_board": feet_on_board,
            "illegal_contact": illegal_contact,
            "fall_candidate": candidate,
            "confirm_frames": self.bad_frames,
        }
        return self.bad_frames >= self.confirm_frames, reasons, diagnostics


def unqualified_name(name: str) -> str:
    return str(name).split("/")[-1]


class LiveRolloutRecorder:
    """Capture policy-rate MuJoCo state and export phase-aligned segments."""

    def __init__(
        self,
        model: Any,
        args: argparse.Namespace,
        physics_randomization: Mapping[str, Any] | None = None,
    ) -> None:
        import mujoco

        self.model = model
        self.args = args
        self.physics_randomization = dict(
            physics_randomization
            or {
                "enabled": False,
                "mode": "nominal_test_scene_xml",
            }
        )
        self.frames: dict[str, list[np.ndarray | float | int | bool]] = defaultdict(list)
        self.terminal_reason = "viewer_closed"
        self.active = True
        self.phase_frame_counts: Counter[str] = Counter()
        self.phase_run_counts: Counter[str] = Counter()
        self.last_phase: str | None = None
        self.last_sim_time = 0.0
        self.progress_path = args.progress_file.resolve() if args.progress_file else None
        self.robot_joint_ids = [
            joint_id
            for joint_id in range(model.njnt)
            if (model.joint(joint_id).name or "").startswith("robot/")
            and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.joint_order = [model.joint(joint_id).name for joint_id in self.robot_joint_ids]
        self.qpos_ids = np.asarray(
            [model.jnt_qposadr[joint_id] for joint_id in self.robot_joint_ids],
            dtype=np.int64,
        )
        self.qvel_ids = np.asarray(
            [model.jnt_dofadr[joint_id] for joint_id in self.robot_joint_ids],
            dtype=np.int64,
        )
        self.robot_body_ids = [
            body_id
            for body_id in range(model.nbody)
            if (model.body(body_id).name or "").startswith("robot/")
        ]
        self.body_order = [model.body(body_id).name for body_id in self.robot_body_ids]
        board_root_joint = model.joint("skateboard/floating_base_joint_skateboard")
        if board_root_joint.type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("skateboard root joint must be a MuJoCo free joint")
        self.board_root_qpos_adr = int(model.jnt_qposadr[board_root_joint.id])
        self.board_root_qvel_adr = int(model.jnt_dofadr[board_root_joint.id])
        self.board_joint_ids = [
            joint_id
            for joint_id in range(model.njnt)
            if (model.joint(joint_id).name or "").startswith("skateboard/")
            and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.board_joint_order = [
            model.joint(joint_id).name for joint_id in self.board_joint_ids
        ]
        self.board_qpos_ids = np.asarray(
            [model.jnt_qposadr[joint_id] for joint_id in self.board_joint_ids],
            dtype=np.int64,
        )
        self.board_qvel_ids = np.asarray(
            [model.jnt_dofadr[joint_id] for joint_id in self.board_joint_ids],
            dtype=np.int64,
        )
        self.max_policy_frames = args.max_policy_frames

        source_names = {unqualified_name(name) for name in self.joint_order}
        missing = [name for name in BFM_29_JOINT_ORDER if name not in source_names]
        if set(missing) != set(BFM_FIXED_WRIST_JOINTS):
            raise ValueError(
                "HUSKY/BFM joint audit failed; expected only six fixed wrist joints "
                f"to be absent, found: {missing}"
            )
        self.write_progress("collecting", "initializing", 0.0)

    @property
    def num_frames(self) -> int:
        return len(self.frames["sim_time"])

    def mark_reset_and_stop(self) -> None:
        if self.num_frames:
            self.frames["reset"][-1] = True
        self.terminal_reason = "reset"
        self.active = False
        self.write_progress("collection_complete")

    def write_progress(
        self,
        status: str,
        phase: str | None = None,
        sim_time: float | None = None,
    ) -> None:
        if self.progress_path is None:
            return
        current_phase = phase or self.last_phase or "initializing"
        current_time = self.last_sim_time if sim_time is None else sim_time
        payload = {
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
            "device": self.args.device,
            "terminal_reason": self.terminal_reason if not self.active else None,
            "phase_frames": dict(sorted(self.phase_frame_counts.items())),
            "phase_runs": dict(sorted(self.phase_run_counts.items())),
        }
        write_json_atomic(self.progress_path, payload)

    def capture(
        self,
        data: Any,
        action: np.ndarray,
        phase: str,
        phase_value: float,
        diagnostics: Mapping[str, Any],
        command_v: float,
        command_h: float,
    ) -> None:
        if not self.active:
            return
        import mujoco

        body_ang_vel = np.empty((len(self.robot_body_ids), 3), dtype=np.float32)
        body_lin_vel = np.empty((len(self.robot_body_ids), 3), dtype=np.float32)
        object_velocity = np.empty(6, dtype=np.float64)
        for index, body_id in enumerate(self.robot_body_ids):
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                object_velocity,
                0,
            )
            body_ang_vel[index] = object_velocity[:3]
            body_lin_vel[index] = object_velocity[3:]

        self.frames["qpos"].append(np.asarray(data.qpos, dtype=np.float32).copy())
        self.frames["qvel"].append(np.asarray(data.qvel, dtype=np.float32).copy())
        self.frames["action"].append(np.asarray(action, dtype=np.float32).copy())
        self.frames["sim_time"].append(float(data.time))
        self.frames["frame_idx"].append(self.num_frames)
        self.frames["phase_id"].append(PHASE_LABEL_TO_ID[phase])
        self.frames["phase_value"].append(float(phase_value))
        self.frames["fall"].append(phase == "fall")
        self.frames["reset"].append(False)
        self.frames["root_pos"].append(np.asarray(data.qpos[:3], dtype=np.float32).copy())
        self.frames["root_quat"].append(np.asarray(data.qpos[3:7], dtype=np.float32).copy())
        self.frames["dof_pos"].append(
            np.asarray(data.qpos[self.qpos_ids], dtype=np.float32).copy()
        )
        self.frames["dof_vel"].append(
            np.asarray(data.qvel[self.qvel_ids], dtype=np.float32).copy()
        )
        self.frames["body_pos"].append(
            np.asarray(data.xpos[self.robot_body_ids], dtype=np.float32).copy()
        )
        self.frames["body_quat"].append(
            np.asarray(data.xquat[self.robot_body_ids], dtype=np.float32).copy()
        )
        self.frames["body_lin_vel"].append(body_lin_vel)
        self.frames["body_ang_vel"].append(body_ang_vel)
        self.frames["board_heading_delta"].append(
            float(diagnostics.get("board_heading_delta_rad", 0.0))
        )
        board_qpos = self.board_root_qpos_adr
        board_qvel = self.board_root_qvel_adr
        self.frames["board_root_pos"].append(
            np.asarray(data.qpos[board_qpos : board_qpos + 3], dtype=np.float32).copy()
        )
        self.frames["board_root_quat"].append(
            np.asarray(
                data.qpos[board_qpos + 3 : board_qpos + 7], dtype=np.float32
            ).copy()
        )
        self.frames["board_root_lin_vel"].append(
            np.asarray(data.qvel[board_qvel : board_qvel + 3], dtype=np.float32).copy()
        )
        self.frames["board_root_ang_vel"].append(
            np.asarray(
                data.qvel[board_qvel + 3 : board_qvel + 6], dtype=np.float32
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
        if (
            self.max_policy_frames is not None
            and self.num_frames >= self.max_policy_frames
        ):
            self.terminal_reason = "max_policy_frames"
            self.active = False
        if self.num_frames % 10 == 0 or not self.active:
            status = "collecting" if self.active else "collection_complete"
            self.write_progress(status, phase, float(data.time))

    def mark_fall_and_stop(self, confirm_frames: int) -> None:
        start = max(0, self.num_frames - max(1, confirm_frames))
        for index in range(start, self.num_frames):
            self.frames["phase_id"][index] = PHASE_LABEL_TO_ID["fall"]
            self.frames["fall"][index] = True
        self.terminal_reason = "fall"
        self.active = False
        self.write_progress("collection_complete", "fall")

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
        return {
            name: np.asarray(values, dtype=dtypes[name])
            for name, values in self.frames.items()
        }

    def bfm_pose(self, arrays: Mapping[str, np.ndarray], start: int, end: int) -> np.ndarray:
        source_order = [unqualified_name(name) for name in self.joint_order]
        source_index = {name: index for index, name in enumerate(source_order)}
        dof = np.zeros((end - start, len(BFM_29_JOINT_ORDER)), dtype=np.float32)
        source_dof = arrays["dof_pos"][start:end]
        for target_index, name in enumerate(BFM_29_JOINT_ORDER):
            if name in source_index:
                dof[:, target_index] = source_dof[:, source_index[name]]
        return np.concatenate(
            [
                arrays["root_pos"][start:end],
                arrays["root_quat"][start:end],
                dof,
            ],
            axis=1,
        ).astype(np.float32)

    def phase_runs(self, arrays: Mapping[str, np.ndarray]) -> list[tuple[str, int, int]]:
        phase_ids = arrays["phase_id"]
        if not len(phase_ids):
            return []
        result: list[tuple[str, int, int]] = []
        start = 0
        for end in range(1, len(phase_ids) + 1):
            if end == len(phase_ids) or phase_ids[end] != phase_ids[start]:
                result.append((PHASE_ID_TO_LABEL[int(phase_ids[start])], start, end))
                start = end
        return result

    def finalize(self) -> None:
        if not self.num_frames:
            print("No live rollout frames were captured.", file=sys.stderr)
            self.write_progress("failed")
            return
        self.write_progress("finalizing")

        args = self.args
        dataset_split = resolve_dataset_split(
            args.rollout_id, args.dataset_split, args.split_seed
        )
        if args.round_id is None:
            output_root = (
                args.output_dir.resolve()
                / dataset_split
                / f"rollout_{args.rollout_id}"
            )
        else:
            output_root = (
                args.output_dir.resolve()
                / f"round_{args.round_id}"
                / f"rollout_{args.rollout_id}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        raw_dir = output_root / "raw_rollout"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_npz = raw_dir / f"{args.episode_id}.npz"
        raw_json = raw_dir / f"{args.episode_id}.json"
        if (raw_npz.exists() or raw_json.exists()) and not args.overwrite:
            raise FileExistsError(
                f"raw rollout {args.episode_id} exists; pass --overwrite to replace it"
            )

        arrays = self.arrays()
        for name, value in arrays.items():
            if value.shape[0] != self.num_frames:
                raise AssertionError(f"live field {name} is not frame-aligned")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"live field {name} contains NaN or Inf")

        np.savez_compressed(raw_npz, **arrays)
        dt_values = np.diff(arrays["sim_time"])
        positive_dt = dt_values[dt_values > 0]
        dt = (
            float(np.median(positive_dt))
            if positive_dt.size
            else 1.0 / args.policy_frequency
        )
        fps = 1.0 / dt
        raw_metadata = {
            "episode_id": args.episode_id,
            "round_id": args.round_id,
            "rollout_id": args.rollout_id,
            "dataset_split": dataset_split,
            "rollout_dir": str(output_root),
            "command_v": args.initial_v,
            "command_h": args.initial_h,
            "fps": fps,
            "dt": dt,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "num_frames": self.num_frames,
            "max_policy_frames": self.max_policy_frames,
            "joint_order": self.joint_order,
            "body_order": self.body_order,
            "board_joint_order": self.board_joint_order,
            "qpos_quaternion_order": "wxyz",
            "body_quaternion_order": "wxyz",
            "phase_mapping": {str(key): value for key, value in PHASE_ID_TO_LABEL.items()},
            "robot_xml": str(args.robot_xml.resolve()),
            "policy_checkpoint": str(args.policy.resolve()),
            "physics_randomization": self.physics_randomization,
            "action_alignment": (
                "action[t] is the previous policy output applied before state[t]"
            ),
            "body_velocity_frame": "world",
            "board_heading_delta_unit": "radian",
            "terminal_reason": self.terminal_reason,
            "fixed_bfm_joints": list(BFM_FIXED_WRIST_JOINTS),
            "fields": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        }
        write_json(raw_json, raw_metadata)

        dynamic_root = output_root / "dynamic_motion"
        margin_frames = round(args.failure_margin * fps)
        minimum_frames = max(8, math.ceil(args.min_duration * fps - 1e-9))
        counters: Counter[str] = Counter()
        exported_frames: Counter[str] = Counter()
        written: list[dict[str, Any]] = []
        state_fields = tuple(arrays)

        runs = self.phase_runs(arrays)
        raw_run_counts = Counter(label for label, _, _ in runs)
        raw_frame_counts = Counter()
        for label, start, end in runs:
            raw_frame_counts[label] += end - start
        self.phase_run_counts = raw_run_counts
        self.phase_frame_counts = raw_frame_counts
        fall_start = next((start for label, start, _ in runs if label == "fall"), None)
        for motion_type, start, end in runs:
            if fall_start is not None and motion_type != "fall" and end == fall_start:
                end = max(start, end - margin_frames)
            required_frames = 8 if motion_type == "fall" else minimum_frames
            if end - start < required_frames:
                continue

            segment_id = (
                f"{args.episode_id}_{motion_type}_{counters[motion_type]:03d}"
            )
            counters[motion_type] += 1
            exported_frames[motion_type] += end - start
            segment_dir = dynamic_root / motion_type / segment_id
            if segment_dir.exists():
                if not args.overwrite:
                    raise FileExistsError(
                        f"{segment_dir} exists; pass --overwrite to replace it"
                    )
                shutil.rmtree(segment_dir)
            segment_dir.mkdir(parents=True)

            pose = self.bfm_pose(arrays, start, end)
            np.save(segment_dir / "pose.npy", pose, allow_pickle=False)
            np.savez_compressed(
                segment_dir / "state.npz",
                **{name: arrays[name][start:end] for name in state_fields},
            )
            metadata = {
                "segment_id": segment_id,
                "source_episode": args.episode_id,
                "round_id": args.round_id,
                "dataset_split": dataset_split,
                "motion_type": motion_type,
                "start_frame": start,
                "end_frame": end,
                "num_frames": end - start,
                "fps": fps,
                "dt": dt,
                "joint_order": self.joint_order,
                "body_order": self.body_order,
                "board_joint_order": self.board_joint_order,
                "quaternion_order": "wxyz",
                "robot_xml": str(args.robot_xml.resolve()),
                "policy_checkpoint": str(args.policy.resolve()),
                "physics_randomization": self.physics_randomization,
                "failure_filtered": motion_type != "fall",
                "terminal_failure": motion_type == "fall",
                "source_state_dof": 23,
                "pose_dof": 29,
                "fixed_bfm_joints": list(BFM_FIXED_WRIST_JOINTS),
                "pose_schema": (
                    "root_pos[3], root_quat_wxyz[4], "
                    "BFM_29_joint_positions[29]"
                ),
                "source_raw_npz": str(raw_npz),
                "state_fields": list(state_fields),
                "preview": None,
            }
            preview_path = segment_dir / "preview.mp4"
            if args.preview_robot_xml is not None:
                render_pose_only_video(
                    args.preview_robot_xml.resolve(),
                    pose,
                    preview_path,
                    fps,
                )
                metadata["preview"] = str(preview_path)
            write_json(segment_dir / "metadata.json", metadata)
            written.append(metadata)

        summary = {
            "episode_id": args.episode_id,
            "round_id": args.round_id,
            "rollout_id": args.rollout_id,
            "dataset_split": dataset_split,
            "rollout_dir": str(output_root),
            "command_v": args.initial_v,
            "command_h": args.initial_h,
            "raw_rollout": str(raw_npz),
            "num_frames": self.num_frames,
            "max_policy_frames": self.max_policy_frames,
            "terminal_reason": self.terminal_reason,
            "physics_randomization": self.physics_randomization,
            "phase_statistics": {
                label: {
                    "raw_runs": raw_run_counts[label],
                    "raw_frames": raw_frame_counts[label],
                    "exported_segments": counters[label],
                    "exported_frames": exported_frames[label],
                }
                for label in PHASE_LABEL_TO_ID
                if raw_frame_counts[label] or counters[label]
            },
            "segments": [
                {
                    "segment_id": item["segment_id"],
                    "motion_type": item["motion_type"],
                    "num_frames": item["num_frames"],
                }
                for item in written
            ],
        }
        write_json(raw_dir / f"{args.episode_id}_segments.json", summary)
        print(f"\nDataset split: {dataset_split}")
        print(f"\nRaw rollout: {raw_npz}")
        print(f"Exported segments: {len(written)}")
        self.write_progress("completed")


def load_upstream_sim(robot_xml: Path, headless: bool = False) -> Any:
    sim_path = robot_xml.resolve().parent / "sim.py"
    if not sim_path.is_file():
        raise FileNotFoundError(f"official HUSKY sim.py not found beside {robot_xml}")
    spec = importlib.util.spec_from_file_location("skate_bfm_upstream_sim", sim_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load HUSKY sim.py from {sim_path}")
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
    sentinel = object()
    previous_pynput = sys.modules.get("pynput", sentinel)
    previous_keyboard = sys.modules.get("pynput.keyboard", sentinel)
    sys.modules["pynput"] = fake_pynput
    sys.modules["pynput.keyboard"] = fake_keyboard
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_pynput is sentinel:
            sys.modules.pop("pynput", None)
        else:
            sys.modules["pynput"] = previous_pynput
        if previous_keyboard is sentinel:
            sys.modules.pop("pynput.keyboard", None)
        else:
            sys.modules["pynput.keyboard"] = previous_keyboard
    return module


class HeadlessViewer:
    """Minimal passive-viewer interface for the unchanged HUSKY run loop."""

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


def resolve_dataset_split(
    rollout_id: str, requested_split: str, split_seed: str
) -> str:
    if requested_split != "auto":
        return requested_split
    digest = hashlib.sha256(f"{split_seed}:{rollout_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if bucket < 0.8:
        return "train"
    if bucket < 0.9:
        return "validation"
    return "test"


def clamp_live_commands(sim_module: Any) -> tuple[float, float, bool]:
    raw_v = float(sim_module.v)
    raw_h = float(sim_module.h)
    command_v = float(np.clip(raw_v, *COMMAND_V_RANGE))
    command_h = float(np.clip(raw_h, *COMMAND_H_RANGE))
    sim_module.v = command_v
    sim_module.h = command_h
    return command_v, command_h, command_v != raw_v or command_h != raw_h


def resolve_physics_seed(rollout_id: str, requested_seed: int | None) -> int:
    if requested_seed is not None:
        return requested_seed
    digest = hashlib.sha256(f"husky-play-dr-v1:{rollout_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def randomize_husky_play_physics(
    model: Any,
    rollout_id: str,
    requested_seed: int | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Apply the official HUSKY play-time DR ranges once per rollout."""
    import mujoco

    seed = resolve_physics_seed(rollout_id, requested_seed)
    rng = np.random.default_rng(seed)

    def uniform(ranges: tuple[float, float]) -> float:
        return float(rng.uniform(*ranges))

    robot_torso = model.body("robot/torso_link").id
    skateboard_deck = model.body("skateboard/skateboard_deck").id
    robot_com_offset = np.asarray(
        [uniform(ranges) for ranges in HUSKY_ROBOT_COM_RANGES],
        dtype=np.float64,
    )
    skateboard_com_offset = np.asarray(
        [uniform(ranges) for ranges in HUSKY_SKATEBOARD_COM_RANGES],
        dtype=np.float64,
    )
    model.body_ipos[robot_torso] += robot_com_offset
    model.body_ipos[skateboard_deck] += skateboard_com_offset

    robot_friction_scales: dict[str, float] = {}
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if name.startswith("robot/"):
            scale = uniform(HUSKY_ROBOT_FRICTION_SCALE_RANGE)
            model.geom_friction[geom_id, 0] *= scale
            robot_friction_scales[name] = scale

    deck_friction_scales: dict[str, float] = {}
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if name == "skateboard/skateboard_deck_collision":
            scale = uniform(HUSKY_DECK_FRICTION_SCALE_RANGE)
            model.geom_friction[geom_id, 0] *= scale
            deck_friction_scales[name] = scale

    foot_friction: dict[str, float] = {}
    foot_pattern = re.compile(r"robot/(left|right)_foot[1-7]_collision$")
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if foot_pattern.fullmatch(name):
            value = uniform(HUSKY_FOOT_FRICTION_RANGE)
            model.geom_friction[geom_id, 0] = value
            foot_friction[name] = value

    wheel_friction_scales: dict[str, float] = {}
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if name.startswith("skateboard/") and name.endswith("_wheel_collision"):
            scale = uniform(HUSKY_WHEEL_FRICTION_SCALE_RANGE)
            model.geom_friction[geom_id, 2] *= scale
            wheel_friction_scales[name] = scale

    joint_offsets: dict[str, float] = {}
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name or ""
        if (
            name.startswith("robot/")
            and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ):
            joint_offsets[name] = uniform(HUSKY_JOINT_POSITION_OFFSET_RANGE)

    report: dict[str, Any] = {
        "enabled": True,
        "mode": "official_husky_play_startup_and_reset",
        "lifecycle": "sampled_once_per_rollout",
        "seed": seed,
        "ranges": {
            "robot_torso_com_offset_m": HUSKY_ROBOT_COM_RANGES,
            "skateboard_com_offset_m": HUSKY_SKATEBOARD_COM_RANGES,
            "robot_sliding_friction_scale": HUSKY_ROBOT_FRICTION_SCALE_RANGE,
            "deck_sliding_friction_scale": HUSKY_DECK_FRICTION_SCALE_RANGE,
            "foot_sliding_friction": HUSKY_FOOT_FRICTION_RANGE,
            "wheel_rolling_friction_scale": HUSKY_WHEEL_FRICTION_SCALE_RANGE,
            "joint_position_offset_rad": HUSKY_JOINT_POSITION_OFFSET_RANGE,
        },
        "robot_torso_com_offset_m": robot_com_offset.tolist(),
        "skateboard_com_offset_m": skateboard_com_offset.tolist(),
        "robot_sliding_friction_scale": robot_friction_scales,
        "deck_sliding_friction_scale": deck_friction_scales,
        "foot_sliding_friction": foot_friction,
        "wheel_rolling_friction_scale": wheel_friction_scales,
        "joint_position_offset_rad": joint_offsets,
        "external_push": False,
        "observation_corruption": False,
    }
    return report, joint_offsets


def run_live(args: argparse.Namespace) -> int:
    """Run the existing HUSKY controller with the official phase clock."""
    robot_xml = args.robot_xml.resolve()
    policy = args.policy.resolve()
    if not robot_xml.is_file():
        raise FileNotFoundError(robot_xml)
    if not policy.is_file():
        raise FileNotFoundError(policy)

    sim_module = load_upstream_sim(robot_xml, headless=args.headless)
    if args.headless:
        sim_module.mjv = HeadlessViewerModule
    if args.initial_v is not None:
        sim_module.v = float(args.initial_v)
    if args.initial_h is not None:
        sim_module.h = float(args.initial_h)

    class LiveController(sim_module.RealTimePolicyController):
        def __init__(self, *controller_args: Any, **controller_kwargs: Any) -> None:
            super().__init__(*controller_args, **controller_kwargs)
            if args.randomize_physics:
                (
                    self.physics_randomization,
                    self.initial_joint_offsets,
                ) = randomize_husky_play_physics(
                    self.model,
                    args.rollout_id,
                    args.physics_seed,
                )
                print(
                    "[DR] Applied official HUSKY play randomization "
                    f"with seed={self.physics_randomization['seed']}"
                )
                import mujoco

                mujoco.mj_setConst(self.model, self.data)
            else:
                self.physics_randomization = {
                    "enabled": False,
                    "mode": "nominal_test_scene_xml",
                }
                self.initial_joint_offsets = {}
            self.phase_clock = OfficialPhaseClock(args.policy_frequency, args.cycle_time)
            self.steer_direction = BoardSteerDirection(self.model)
            confirm_frames = max(1, round(args.fall_confirm_time * args.policy_frequency))
            self.fall_detector = LiveFallDetector(
                self.model,
                args.fall_orientation_deg,
                args.fall_root_height_min,
                confirm_frames,
            )
            self.last_reported_phase: str | None = None
            self.last_status_time = -math.inf
            self.last_output_heading: float | None = None
            self.steer_start_heading: float | None = None
            self.in_steer_phase = False
            self.recorder = (
                LiveRolloutRecorder(self.model, args, self.physics_randomization)
                if args.record
                else None
            )

        def report_phase(
            self,
            phase: str,
            diagnostics: Mapping[str, Any] | None = None,
            force: bool = False,
        ) -> None:
            sim_time = float(self.data.time)
            changed = phase != self.last_reported_phase
            if (
                not changed
                and not force
                and sim_time - self.last_status_time < args.status_interval
            ):
                return
            self.last_status_time = sim_time
            self.last_reported_phase = phase
            details = diagnostics or {}
            current_heading = float(details.get("board_heading_delta_deg", 0.0))
            previous_delta = (
                0.0
                if self.last_output_heading is None
                else current_heading - self.last_output_heading
            )
            self.last_output_heading = current_heading
            steer_delta = (
                None
                if self.steer_start_heading is None
                else current_heading - self.steer_start_heading
            )
            steer_delta_text = "--" if steer_delta is None else f"{steer_delta:+.2f}deg"
            line = (
                f"t={sim_time:.2f}s phase={phase} "
                f"delta_prev={previous_delta:+.2f}deg "
                f"delta_steer={steer_delta_text}"
            )
            if changed or force:
                print(f"\n[PHASE] {line}", flush=True)
            else:
                sys.stdout.write(f"\r[STATUS] {line.ljust(80)[:80]}")
                sys.stdout.flush()

        def reset_fall_state(self) -> None:
            self.fall_detector.reset()
            self.phase_clock.reset()
            self.steer_direction.reset()
            self.last_reported_phase = None
            self.last_status_time = -math.inf
            self.last_output_heading = None
            self.steer_start_heading = None
            self.in_steer_phase = False

        def reset(self, init_pos: np.ndarray) -> None:
            terminate_rollout = self.recorder is not None and self.recorder.num_frames > 0
            if terminate_rollout:
                self.recorder.mark_reset_and_stop()
            randomized_init_pos = np.asarray(init_pos, dtype=np.float64).copy()
            for joint_name, offset in self.initial_joint_offsets.items():
                joint = self.model.joint(joint_name)
                randomized_init_pos[self.model.jnt_qposadr[joint.id]] += offset
            super().reset(randomized_init_pos)
            self.reset_fall_state()
            _, _, diagnostics = self.fall_detector.check(self.data)
            steer_direction, steer_diagnostics = self.steer_direction.classify(
                self.data, float(sim_module.h)
            )
            diagnostics.update(steer_diagnostics)
            diagnostics["steer_direction"] = steer_direction
            diagnostics["phase_value"] = 0.0
            self.report_phase("push", diagnostics=diagnostics, force=True)
            if terminate_rollout and self.viewer is not None:
                self.viewer.close()

        def extract_data(self) -> Any:
            command_v, command_h, command_clamped = clamp_live_commands(sim_module)
            if command_clamped:
                print(
                    f"\n[COMMAND] clamped to v={command_v:.2f}, h={command_h:.3f}rad",
                    flush=True,
                )
            values = super().extract_data()
            phase, phase_value = self.phase_clock.next()
            fallen, _, diagnostics = self.fall_detector.check(self.data)
            steer_direction, steer_diagnostics = self.steer_direction.classify(
                self.data, command_h
            )
            diagnostics.update(steer_diagnostics)
            in_steer_phase = phase == "steer"
            current_heading = float(diagnostics["board_heading_delta_deg"])
            if in_steer_phase and not self.in_steer_phase:
                self.steer_start_heading = current_heading
            elif not in_steer_phase:
                self.steer_start_heading = None
            self.in_steer_phase = in_steer_phase
            if fallen:
                phase = "fall"
            elif phase == "steer":
                phase = f"steer_{steer_direction}"
            diagnostics["steer_direction"] = steer_direction
            diagnostics["phase_value"] = phase_value
            if self.recorder is not None:
                self.recorder.capture(
                    self.data,
                    self.last_action,
                    phase,
                    phase_value,
                    diagnostics,
                    command_v,
                    command_h,
                )
                if fallen:
                    self.recorder.mark_fall_and_stop(
                        self.fall_detector.confirm_frames
                    )
                if not self.recorder.active and not fallen and self.viewer is not None:
                    self.viewer.close()
            self.report_phase(phase, diagnostics)
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


def heading_name(value: float) -> str:
    if value > 0:
        direction = "pos"
    elif value < 0:
        direction = "neg"
    else:
        direction = "zero"
    return f"{direction}{round(abs(value) * 100):03d}"


def velocity_name(value: float) -> str:
    return f"{round(value * 100):03d}"


def round_grid_assignments(
    headings: Sequence[float],
    velocities: Sequence[float],
    round_count: int,
    rollouts_per_round: int,
    seed: int,
    round_offset: int = 0,
) -> list[tuple[float, float]]:
    """Give each round balanced velocities and deterministic heading coverage."""
    if (
        not headings
        or not velocities
        or round_count <= 0
        or rollouts_per_round <= 0
    ):
        return []
    rng = np.random.default_rng(seed)
    assignments: list[tuple[float, float]] = []
    for local_round in range(round_count):
        round_index = round_offset + local_round
        batch = []
        for slot in range(rollouts_per_round):
            heading_index = slot % len(headings)
            heading_cycle = slot // len(headings)
            velocity_index = (
                heading_index + round_index + heading_cycle
            ) % len(velocities)
            batch.append(
                (velocities[velocity_index], headings[heading_index])
            )
        order = rng.permutation(len(batch))
        assignments.extend(batch[int(index)] for index in order)
    return assignments


def collection_job_summary(
    job: Mapping[str, Any],
    policy_frequency: int,
) -> dict[str, Any] | None:
    summary_path = (
        job["rollout_root"]
        / "raw_rollout"
        / f"{job['episode_id']}_segments.json"
    )
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_frames = int(payload.get("num_frames", 0))
    expert_frames = sum(
        int(segment.get("num_frames", 0))
        for segment in payload.get("segments", [])
        if segment.get("motion_type") != "fall"
    )
    return {
        "round_id": job["round_id"],
        "rollout_id": job["rollout_id"],
        "episode_id": job["episode_id"],
        "command_v": job["velocity"],
        "command_h": job["heading"],
        "physics_seed": job["physics_seed"],
        "terminal_reason": payload.get("terminal_reason", "unknown"),
        "raw_frames": raw_frames,
        "raw_duration_seconds": raw_frames / policy_frequency,
        "expert_frames": expert_frames,
        "expert_duration_seconds": expert_frames / policy_frequency,
        "summary_path": str(summary_path),
        "phase_statistics": payload.get("phase_statistics", {}),
    }


def run_parallel(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    output_root = args.output_dir.resolve()
    baseline_count = args.round_count * args.rollouts_per_round
    baseline_assignments = round_grid_assignments(
        args.parallel_heading_values,
        args.parallel_velocity_values,
        args.round_count,
        args.rollouts_per_round,
        args.plan_seed,
    )
    extra_assignments = round_grid_assignments(
        args.parallel_heading_values,
        args.parallel_velocity_values,
        args.max_extra_rounds,
        args.rollouts_per_round,
        args.plan_seed + 1,
        round_offset=args.round_count,
    )
    assignments = baseline_assignments + extra_assignments
    jobs: list[dict[str, Any]] = []
    round_start = int(args.round_id)
    for index, (velocity, heading) in enumerate(assignments):
        round_offset = index // args.rollouts_per_round
        round_id = str(round_start + round_offset).zfill(3)
        rollout_index = index % args.rollouts_per_round
        rollout_id = str(args.parallel_rollout_start + rollout_index).zfill(3)
        episode_id = (
            f"round{round_id}_rollout{rollout_id}_"
            f"v{velocity_name(velocity)}_h_{heading_name(heading)}"
        )
        seed = (
            args.physics_seed + index
            if args.physics_seed is not None
            else int.from_bytes(
                hashlib.sha256(
                    f"parallel:{round_id}:{rollout_id}".encode()
                ).digest()[:4],
                "big",
            )
        )
        rollout_root = output_root / f"round_{round_id}" / f"rollout_{rollout_id}"
        jobs.append(
            {
                "round_id": round_id,
                "rollout_id": rollout_id,
                "episode_id": episode_id,
                "heading": heading,
                "velocity": velocity,
                "physics_seed": seed,
                "rollout_root": rollout_root,
                "progress_path": rollout_root / "collection_progress.json",
                "displayed_frames": 0,
                "latest_progress": {},
                "reported_finished": False,
            }
        )

    target_frames = (
        round(args.target_raw_minutes * 60.0 * args.policy_frequency)
        if args.target_raw_minutes is not None
        else baseline_count * args.max_policy_frames
    )
    plan_path = output_root / "collection_plan.json"
    plan = {
        "target_raw_minutes": args.target_raw_minutes,
        "target_raw_frames": target_frames,
        "policy_frequency": args.policy_frequency,
        "max_policy_frames_per_rollout": args.max_policy_frames,
        "planned_rollout_seconds": args.max_policy_frames / args.policy_frequency,
        "round_start": str(round_start).zfill(3),
        "baseline_rounds": args.round_count,
        "max_extra_rounds": args.max_extra_rounds,
        "rollouts_per_round": args.rollouts_per_round,
        "baseline_rollouts": baseline_count,
        "parallel_workers": args.parallel_workers,
        "headings": args.parallel_heading_values,
        "velocities": args.parallel_velocity_values,
        "plan_seed": args.plan_seed,
        "render_previews": args.render_previews,
        "jobs": [
            {
                "round_id": job["round_id"],
                "rollout_id": job["rollout_id"],
                "episode_id": job["episode_id"],
                "command_v": job["velocity"],
                "command_h": job["heading"],
                "physics_seed": job["physics_seed"],
                "baseline": index < baseline_count,
                "output": str(job["rollout_root"]),
            }
            for index, job in enumerate(jobs)
        ],
    }
    write_json_atomic(plan_path, plan)

    print("Parallel collection parameters")
    print(
        f"  rounds: {str(round_start).zfill(3)}-"
        f"{str(round_start + args.round_count - 1).zfill(3)} baseline, "
        f"up to {args.max_extra_rounds} replacement rounds"
    )
    print(
        f"  rollouts: {args.rollouts_per_round}/round, "
        f"{baseline_count} baseline, {args.parallel_workers} concurrent workers"
    )
    print(
        f"  target: {target_frames} raw frames "
        f"({target_frames / args.policy_frequency / 60.0:.2f}min actual), "
        f"{args.max_policy_frames} frames/rollout"
    )
    print(f"  h grid: {args.parallel_heading_values}")
    print(f"  v grid: {args.parallel_velocity_values}")
    phase_clock = OfficialPhaseClock(args.policy_frequency, args.cycle_time)
    p0, p1, p2, p3, p4 = phase_clock.boundaries
    print(
        f"  phase cycle: {phase_clock.cycle_frames} frames "
        f"({args.cycle_time:g}s at {args.policy_frequency}Hz), "
        f"push={p1 - p0}, push2steer={p2 - p1}, "
        f"steer={p3 - p2}, steer2push={p4 - p3}"
    )
    print(
        f"  device={args.device}; dataset_split={args.dataset_split}; "
        f"HUSKY_DR=enabled; previews={'on' if args.render_previews else 'off'}"
    )
    print(f"  policy: {args.policy.resolve()}")
    print(f"  output: {output_root}")
    print(f"  plan: {plan_path}")

    completed_records: dict[str, dict[str, Any]] = {}
    if not args.overwrite:
        for job in jobs:
            record = collection_job_summary(job, args.policy_frequency)
            if record is not None:
                completed_records[job["episode_id"]] = record
                job["displayed_frames"] = record["raw_frames"]

    progress_bar = tqdm(
        total=target_frames,
        initial=min(
            target_frames,
            sum(record["raw_frames"] for record in completed_records.values()),
        ),
        desc="Raw rollout target",
        unit="frame",
        dynamic_ncols=True,
    )

    def aggregate() -> dict[str, Any]:
        records = list(completed_records.values())
        raw_frames = sum(record["raw_frames"] for record in records)
        expert_frames = sum(record["expert_frames"] for record in records)
        terminal_counts = Counter(
            record["terminal_reason"] for record in records
        )
        phase_totals: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            for label, statistics in record["phase_statistics"].items():
                for name in (
                    "raw_runs",
                    "raw_frames",
                    "exported_segments",
                    "exported_frames",
                ):
                    phase_totals[label][name] += int(statistics.get(name, 0))
        return {
            "target_raw_minutes": args.target_raw_minutes,
            "target_raw_frames": target_frames,
            "target_achieved": raw_frames >= target_frames,
            "completed_rollouts": len(records),
            "raw_frames": raw_frames,
            "raw_duration_seconds": raw_frames / args.policy_frequency,
            "raw_duration_minutes": raw_frames / args.policy_frequency / 60.0,
            "expert_frames": expert_frames,
            "expert_duration_seconds": expert_frames / args.policy_frequency,
            "expert_duration_minutes": expert_frames / args.policy_frequency / 60.0,
            "terminal_reasons": dict(sorted(terminal_counts.items())),
            "phase_statistics": {
                label: dict(sorted(statistics.items()))
                for label, statistics in sorted(phase_totals.items())
            },
            "records": sorted(
                records,
                key=lambda item: (item["round_id"], item["rollout_id"]),
            ),
        }

    def update_summary() -> dict[str, Any]:
        summary = aggregate()
        write_json_atomic(output_root / "collection_summary.json", summary)
        return summary

    def progress_update(job: dict[str, Any], captured: int) -> None:
        increment = max(0, captured - job["displayed_frames"])
        remaining = max(0, target_frames - progress_bar.n)
        if increment and remaining:
            progress_bar.update(min(increment, remaining))
        job["displayed_frames"] = max(job["displayed_frames"], captured)

    def launch(job: dict[str, Any]) -> None:
        rollout_id = job["rollout_id"]
        rollout_root = job["rollout_root"]
        rollout_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            job["progress_path"],
            {
                "status": "starting",
                "round_id": job["round_id"],
                "rollout_id": rollout_id,
                "episode_id": job["episode_id"],
                "collected_frames": 0,
                "max_policy_frames": args.max_policy_frames,
                "phase": "initializing",
                "command_v": job["velocity"],
                "command_h": job["heading"],
                "physics_seed": job["physics_seed"],
                "device": args.device,
            },
        )
        log_handle = (rollout_root / "collection.log").open("w", encoding="utf-8")
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
            rollout_id,
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
            "--status-interval",
            str(args.status_interval),
            "--min-duration",
            str(args.min_duration),
            "--failure-margin",
            str(args.failure_margin),
            "--output-dir",
            str(args.output_dir),
            "--progress-file",
            str(job["progress_path"]),
        ]
        if args.render_previews and args.preview_robot_xml is not None:
            command.extend(["--preview-robot-xml", str(args.preview_robot_xml)])
        else:
            command.append("--no-render-previews")
        if args.overwrite:
            command.append("--overwrite")
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        job["process"] = process
        job["log_handle"] = log_handle
        tqdm.write(
            f"Started round_{job['round_id']}/rollout_{rollout_id}: "
            f"v={job['velocity']:.2f}, h={job['heading']:+.2f}, "
            f"dr_seed={job['physics_seed']}, pid={process.pid}"
        )

    def run_job_batch(batch: Sequence[dict[str, Any]]) -> list[str]:
        queue = [
            job
            for job in batch
            if args.overwrite or job["episode_id"] not in completed_records
        ]
        active: list[dict[str, Any]] = []
        failed: list[str] = []
        next_index = 0
        while next_index < len(queue) or active:
            while (
                next_index < len(queue)
                and len(active) < args.parallel_workers
            ):
                job = queue[next_index]
                next_index += 1
                launch(job)
                active.append(job)

            postfix: dict[str, str] = {}
            for job in list(active):
                try:
                    payload = json.loads(
                        job["progress_path"].read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    payload = job["latest_progress"]
                job["latest_progress"] = payload
                captured = min(
                    int(payload.get("collected_frames", 0)),
                    args.max_policy_frames,
                )
                progress_update(job, captured)
                status = payload.get("status", "starting")
                detail = (
                    payload.get("phase", "starting")
                    if status == "collecting"
                    else status
                )
                key = f"{job['round_id']}/{job['rollout_id']}"
                postfix[key] = f"{captured}/{args.max_policy_frames}:{detail}"
                process = job["process"]
                if process.poll() is None:
                    continue
                process.wait()
                job["log_handle"].close()
                active.remove(job)
                if process.returncode != 0:
                    failed.append(
                        f"round_{job['round_id']}/rollout_{job['rollout_id']}"
                    )
                    tqdm.write(
                        f"Failed round_{job['round_id']}/"
                        f"rollout_{job['rollout_id']}: status={process.returncode}"
                    )
                    continue
                record = collection_job_summary(job, args.policy_frequency)
                if record is None:
                    failed.append(
                        f"round_{job['round_id']}/rollout_{job['rollout_id']}"
                    )
                    continue
                completed_records[job["episode_id"]] = record
                update_summary()
                tqdm.write(
                    f"Finished round_{job['round_id']}/"
                    f"rollout_{job['rollout_id']}: "
                    f"raw={record['raw_duration_seconds']:.2f}s, "
                    f"expert={record['expert_duration_seconds']:.2f}s, "
                    f"terminal={record['terminal_reason']}"
                )
            aggregate_now = aggregate()
            postfix["total"] = (
                f"raw={aggregate_now['raw_duration_minutes']:.2f}min,"
                f"expert={aggregate_now['expert_duration_minutes']:.2f}min"
            )
            progress_bar.set_postfix(postfix, refresh=True)
            if active:
                time.sleep(0.2)
        return failed

    try:
        failed = run_job_batch(jobs[:baseline_count])
        progress_bar.close()
        if failed:
            print(f"Parallel collection failed: {', '.join(failed)}", file=sys.stderr)
            return 1

        summary = update_summary()
        extra_index = baseline_count
        while (
            summary["raw_frames"] < target_frames
            and extra_index < len(jobs)
        ):
            missing = target_frames - summary["raw_frames"]
            needed = max(1, math.ceil(missing / args.max_policy_frames))
            batch = jobs[extra_index : extra_index + needed]
            extra_index += len(batch)
            progress_bar = tqdm(
                total=target_frames,
                initial=min(target_frames, summary["raw_frames"]),
                desc="Raw rollout target",
                unit="frame",
                dynamic_ncols=True,
            )
            failed = run_job_batch(batch)
            progress_bar.close()
            if failed:
                print(
                    f"Replacement collection failed: {', '.join(failed)}",
                    file=sys.stderr,
                )
                return 1
            summary = update_summary()

        print("Collection summary")
        print(
            f"  raw: {summary['raw_duration_minutes']:.3f}min "
            f"({summary['raw_frames']} frames)"
        )
        print(
            f"  cleaned expert: {summary['expert_duration_minutes']:.3f}min "
            f"({summary['expert_frames']} frames)"
        )
        print(f"  completed rollouts: {summary['completed_rollouts']}")
        print(f"  terminal reasons: {summary['terminal_reasons']}")
        print(f"  target achieved: {summary['target_achieved']}")
        print(f"  summary: {output_root / 'collection_summary.json'}")
        return 0 if summary["target_achieved"] else 2
    except KeyboardInterrupt:
        for job in jobs:
            process = job.get("process")
            log_handle = job.get("log_handle")
            if process is None:
                continue
            if process.poll() is None:
                process.terminate()
            if log_handle is not None and not log_handle.closed:
                log_handle.close()
        for job in jobs:
            process = job.get("process")
            if process is not None:
                process.wait()
        progress_bar.close()
        update_summary()
        return 130


def apply_parallel_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.parallel_config is None:
        return
    config_path = args.parallel_config.resolve()
    if not config_path.is_file():
        parser.error(f"--parallel-config does not exist: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"unable to load --parallel-config: {error}")
    if not isinstance(payload, dict):
        parser.error("--parallel-config must contain a JSON object")

    fields = {
        "round_id": ("round_id", "--round-id"),
        "round_start": ("round_id", "--round-id"),
        "round_count": ("round_count", "--round-count"),
        "rollouts_per_round": ("rollouts_per_round", "--rollouts-per-round"),
        "parallel_workers": ("parallel_workers", "--parallel-workers"),
        "rollout_start": ("parallel_rollout_start", "--parallel-rollout-start"),
        "headings": ("parallel_headings", "--parallel-headings"),
        "velocities": ("parallel_velocities", "--parallel-velocities"),
        "target_raw_minutes": ("target_raw_minutes", "--target-raw-minutes"),
        "max_extra_rounds": ("max_extra_rounds", "--max-extra-rounds"),
        "plan_seed": ("plan_seed", "--plan-seed"),
        "render_previews": ("render_previews", "--render-previews"),
        "max_policy_frames": ("max_policy_frames", "--max-policy-frames"),
        "initial_v": ("initial_v", "--initial-v"),
        "physics_seed_start": ("physics_seed", "--physics-seed"),
        "device": ("device", "--device"),
        "dataset_split": ("dataset_split", "--dataset-split"),
        "output_dir": ("output_dir", "--output-dir"),
        "robot_xml": ("robot_xml", "--robot-xml"),
        "policy": ("policy", "--policy"),
        "overwrite": ("overwrite", "--overwrite"),
    }
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        parser.error(f"unknown --parallel-config fields: {unknown}")

    def supplied(option: str) -> bool:
        return any(
            token == option or token.startswith(f"{option}=") for token in sys.argv[1:]
        )

    path_fields = {"output_dir", "robot_xml", "policy"}
    for config_name, (attribute, option) in fields.items():
        if config_name not in payload or supplied(option):
            continue
        value = payload[config_name]
        if value is None:
            continue
        if config_name in {"headings", "velocities"}:
            if not isinstance(value, list):
                parser.error(f"parallel config {config_name} must be a JSON array")
            value = ",".join(str(item) for item in value)
        elif config_name in path_fields:
            value = Path(value)
            if not value.is_absolute():
                value = config_path.parent / value
        elif config_name in {"round_id", "round_start"}:
            value = str(value)
        setattr(args, attribute, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the official HUSKY controller with real-time phase output.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record and split one interactive --live rollout.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without the real-time MuJoCo viewer. Recorded phase preview "
            "videos are still rendered during finalization."
        ),
    )
    parser.add_argument(
        "--parallel-headings",
        help=(
            "Comma-separated fixed headings to collect in parallel, for "
            "example 0.2,-0.2. This enables live headless test collection."
        ),
    )
    parser.add_argument(
        "--parallel-config",
        nargs="?",
        const=Path(__file__).resolve().with_name("rollout_config.json"),
        type=Path,
        help=(
            "Load parallel test settings from JSON. With no path, use "
            "rollout_config.json beside this script."
        ),
    )
    parser.add_argument(
        "--parallel-rollout-start",
        type=int,
        default=1,
        help="First sequential rollout ID used by --parallel-headings.",
    )
    parser.add_argument(
        "--parallel-velocities",
        help="Comma-separated forward commands paired with the heading grid.",
    )
    parser.add_argument("--round-count", type=int, default=1)
    parser.add_argument("--rollouts-per-round", type=int)
    parser.add_argument("--parallel-workers", type=int)
    parser.add_argument("--target-raw-minutes", type=float)
    parser.add_argument("--max-extra-rounds", type=int, default=0)
    parser.add_argument("--plan-seed", type=int, default=20260804)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--key-events", type=Path)
    parser.add_argument("--key-map", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument(
        "--round-id",
        help=(
            "Optional numeric test-batch suffix. When set, output is written "
            "as round_<id>/rollout_<id>."
        ),
    )
    parser.add_argument(
        "--rollout-id",
        help="Numeric rollout directory suffix, written as rollout_<id>.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=("auto", *DATASET_SPLITS),
        default="auto",
        help=(
            "Assign the complete rollout to one dataset split. 'auto' uses a "
            "stable 80/10/10 hash split."
        ),
    )
    parser.add_argument(
        "--split-seed",
        default="skate-bfm-v1",
        help="Stable seed used only by --dataset-split auto.",
    )
    parser.add_argument(
        "--randomize-physics",
        action="store_true",
        help=(
            "Apply official HUSKY play-time startup physics DR and reset joint "
            "noise once for this rollout."
        ),
    )
    parser.add_argument(
        "--physics-seed",
        type=int,
        help=(
            "Optional HUSKY physics randomization seed. By default it is "
            "derived reproducibly from --rollout-id."
        ),
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument(
        "--preview-robot-xml",
        type=Path,
        help="BFM G1 29DoF XML used for pose-only segment videos.",
    )
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--progress-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--render-previews",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render pose-only phase videos after each recorded rollout.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-frequency", type=int, default=50)
    parser.add_argument("--cycle-time", type=float, default=6.0)
    parser.add_argument(
        "--initial-v",
        type=float,
        help="Optional reproducible initial forward command for --live.",
    )
    parser.add_argument(
        "--initial-h",
        type=float,
        help="Optional reproducible initial steering command for --live.",
    )
    parser.add_argument("--steer-confirm-time", type=float, default=0.1)
    parser.add_argument("--fall-orientation-deg", type=float, default=70.0)
    parser.add_argument("--fall-root-height-min", type=float, default=0.45)
    parser.add_argument("--fall-confirm-time", type=float, default=0.2)
    parser.add_argument("--status-interval", type=float, default=0.2)
    parser.add_argument(
        "--max-policy-frames",
        type=int,
        default=3000,
        help="Stop a recorded live rollout after this many policy-rate frames.",
    )
    parser.add_argument("--fps", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--pre-padding", type=float, default=0.1)
    parser.add_argument("--post-padding", type=float, default=0.1)
    parser.add_argument("--merge-gap", type=float, default=0.1)
    parser.add_argument("--failure-margin", type=float, default=0.15)
    parser.add_argument("--post-reset-ignore", type=float, default=0.2)
    parser.add_argument("--root-height-min", type=float)
    parser.add_argument("--root-tilt-max-deg", type=float)
    args = parser.parse_args()
    apply_parallel_config(args, parser)
    if args.parallel_headings is not None:
        try:
            args.parallel_heading_values = [
                float(value.strip())
                for value in args.parallel_headings.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--parallel-headings must contain comma-separated numbers")
        if not args.parallel_heading_values:
            parser.error("--parallel-headings requires at least one value")
        if args.parallel_velocities is None:
            args.parallel_velocity_values = [
                1.0 if args.initial_v is None else args.initial_v
            ]
        else:
            try:
                args.parallel_velocity_values = [
                    float(value.strip())
                    for value in args.parallel_velocities.split(",")
                    if value.strip()
                ]
            except ValueError:
                parser.error(
                    "--parallel-velocities must contain comma-separated numbers"
                )
            if not args.parallel_velocity_values:
                parser.error("--parallel-velocities requires at least one value")
        if args.round_id is None:
            parser.error("--parallel-headings requires --round-id")
        if args.parallel_rollout_start < 0:
            parser.error("--parallel-rollout-start must be non-negative")
        for heading in args.parallel_heading_values:
            if not COMMAND_H_RANGE[0] <= heading <= COMMAND_H_RANGE[1]:
                parser.error(
                    "parallel heading values must be within [-pi/4, pi/4]"
                )
        for velocity in args.parallel_velocity_values:
            if not COMMAND_V_RANGE[0] <= velocity <= COMMAND_V_RANGE[1]:
                parser.error("parallel velocities must be within [0.0, 1.5]")
        if args.round_count <= 0:
            parser.error("--round-count must be positive")
        if args.rollouts_per_round is None:
            args.rollouts_per_round = len(args.parallel_heading_values)
        if args.rollouts_per_round <= 0:
            parser.error("--rollouts-per-round must be positive")
        if args.parallel_workers is None:
            args.parallel_workers = min(
                args.rollouts_per_round,
                len(args.parallel_heading_values),
            )
        if args.parallel_workers <= 0:
            parser.error("--parallel-workers must be positive")
        if args.max_extra_rounds < 0:
            parser.error("--max-extra-rounds must be non-negative")
        if args.target_raw_minutes is not None and args.target_raw_minutes <= 0:
            parser.error("--target-raw-minutes must be positive")
        max_rollouts = (
            args.round_count + args.max_extra_rounds
        ) * args.rollouts_per_round
        max_raw_minutes = (
            max_rollouts
            * args.max_policy_frames
            / args.policy_frequency
            / 60.0
        )
        if (
            args.target_raw_minutes is not None
            and args.target_raw_minutes > max_raw_minutes
        ):
            parser.error(
                f"collection plan can provide at most {max_raw_minutes:.2f} "
                "raw minutes before early termination"
            )
        repo_root = Path(__file__).resolve().parents[2]
        args.live = True
        args.record = True
        args.headless = True
        args.randomize_physics = True
        if args.dataset_split == "auto":
            args.dataset_split = "train"
        args.output_dir = args.output_dir or repo_root / "train" / "scripts" / "temp"
        args.robot_xml = (
            args.robot_xml
            or repo_root / "husky_sim" / "upstream" / "test_scene" / "mjlab_scene.xml"
        )
        args.policy = (
            args.policy
            or repo_root / "husky_sim" / "upstream" / "ckpts" / "test.onnx"
        )
        args.initial_v = 1.0 if args.initial_v is None else args.initial_v
        args.episode_id = "parallel_parent"
        args.rollout_id = str(args.parallel_rollout_start)
    else:
        args.parallel_heading_values = []
        args.parallel_velocity_values = []
    if args.live:
        if args.robot_xml is None or args.policy is None:
            parser.error("--live requires --robot-xml and --policy")
        if args.record and (args.output_dir is None or args.episode_id is None):
            parser.error("--live --record requires --output-dir and --episode-id")
        if args.rollout_id is None:
            match = re.search(r"(\d+)$", args.episode_id or "")
            args.rollout_id = match.group(1) if match else "001"
        if not re.fullmatch(r"\d+", args.rollout_id):
            parser.error("--rollout-id must contain only digits")
        args.rollout_id = args.rollout_id.zfill(3)
        if args.round_id is not None and not re.fullmatch(r"\d+", args.round_id):
            parser.error("--round-id must contain only digits")
        if args.round_id is not None:
            args.round_id = args.round_id.zfill(3)
        if args.physics_seed is not None and not args.randomize_physics:
            parser.error("--physics-seed requires --randomize-physics")
        if args.physics_seed is not None and args.physics_seed < 0:
            parser.error("--physics-seed must be non-negative")
        if args.record and args.render_previews and args.preview_robot_xml is None:
            candidate = (
                Path(__file__).resolve().parents[2]
                / "model"
                / "bfm-zero-source"
                / "humanoidverse"
                / "data"
                / "robots"
                / "g1"
                / "g1_29dof.xml"
            )
            if candidate.is_file():
                args.preview_robot_xml = candidate
            else:
                parser.error(
                    "--live --record requires --preview-robot-xml when the "
                    "repository BFM G1 XML is unavailable"
                )
        if (
            args.render_previews
            and args.preview_robot_xml is not None
            and not args.preview_robot_xml.is_file()
        ):
            parser.error(f"--preview-robot-xml does not exist: {args.preview_robot_xml}")
        if args.policy_frequency <= 0:
            parser.error("--policy-frequency must be positive")
        if args.cycle_time <= 0:
            parser.error("--cycle-time must be positive")
        if (
            args.initial_v is not None
            and not COMMAND_V_RANGE[0] <= args.initial_v <= COMMAND_V_RANGE[1]
        ):
            parser.error("--initial-v must be in [0.0, 1.5]")
        if (
            args.initial_h is not None
            and not COMMAND_H_RANGE[0] <= args.initial_h <= COMMAND_H_RANGE[1]
        ):
            parser.error("--initial-h must be in [-pi/4, pi/4]")
        if args.steer_confirm_time <= 0:
            parser.error("--steer-confirm-time must be positive")
        if args.fall_root_height_min <= 0:
            parser.error("--fall-root-height-min must be positive")
        if args.fall_confirm_time <= 0:
            parser.error("--fall-confirm-time must be positive")
        if args.status_interval <= 0:
            parser.error("--status-interval must be positive")
        if args.max_policy_frames is not None and args.max_policy_frames <= 0:
            parser.error("--max-policy-frames must be positive")
    else:
        missing = [
            option
            for option, value in (
                ("--rollout", args.rollout),
                ("--output-dir", args.output_dir),
                ("--episode-id", args.episode_id),
            )
            if value is None
        ]
        if missing:
            parser.error(f"split mode requires {', '.join(missing)}")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def flatten_mapping(payload: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        result[name] = value
        if isinstance(value, Mapping):
            result.update(flatten_mapping(value, name))
    return result


def field_by_name(payload: Any, names: Sequence[str]) -> tuple[str | None, Any]:
    flat = flatten_mapping(payload)
    for wanted in names:
        for name, value in flat.items():
            if name.split(".")[-1].lower() == wanted:
                return name, value
    return None, None


def unwrap_object_array(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def load_rollout(path: Path) -> Rollout:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload = unwrap_object_array(np.load(path, allow_pickle=True))
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            payload = {key: unwrap_object_array(archive[key]) for key in archive.files}
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        raise ValueError("rollout must use .npy, .npz, .pkl, or .pickle")

    keys = [str(key) for key in payload.keys()] if isinstance(payload, Mapping) else []
    shape = list(payload.shape) if isinstance(payload, np.ndarray) else None
    dtype = str(payload.dtype) if isinstance(payload, np.ndarray) else None
    num_frames = infer_num_frames(payload)
    return Rollout(path, suffix, payload, type(payload).__name__, keys, shape, dtype, num_frames)


def infer_num_frames(payload: Any) -> int:
    if isinstance(payload, np.ndarray):
        if payload.ndim == 0:
            raise ValueError("scalar ndarray does not define rollout frames")
        return int(payload.shape[0])
    if not isinstance(payload, Mapping):
        raise ValueError(f"unsupported rollout top-level type: {type(payload).__name__}")

    flat = flatten_mapping(payload)
    preferred: list[tuple[str, int]] = []
    for name, value in flat.items():
        value = np.asarray(value) if isinstance(value, (list, tuple)) else value
        if not isinstance(value, np.ndarray) or value.ndim == 0:
            continue
        leaf = name.split(".")[-1].lower()
        if leaf in {*TIME_FIELDS, *COMMAND_FIELDS, *QPOS_FIELDS, "frame_idx", "frame_index"}:
            preferred.append((name, int(value.shape[0])))
    if preferred:
        lengths = {length for _, length in preferred}
        if len(lengths) != 1:
            raise ValueError(f"conflicting frame counts in confirmed fields: {preferred}")
        return preferred[0][1]

    lengths = Counter()
    for value in flat.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            lengths[int(value.shape[0])] += 1
    if not lengths:
        raise ValueError("no frame-aligned arrays found")
    length, count = lengths.most_common(1)[0]
    if count < 2:
        raise ValueError(
            "frame count is ambiguous; provide a recognized time, command, qpos, "
            "or frame index field"
        )
    return length


def numeric_integrity(payload: Any, num_frames: int) -> dict[str, Any]:
    arrays: list[tuple[str, np.ndarray]] = []
    if isinstance(payload, np.ndarray):
        arrays.append(("<array>", payload))
    else:
        for name, value in flatten_mapping(payload).items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == num_frames:
                arrays.append((name, value))
    checked, nan_fields, inf_fields = [], [], []
    for name, value in arrays:
        if not np.issubdtype(value.dtype, np.number):
            continue
        checked.append(name)
        if np.isnan(value).any():
            nan_fields.append(name)
        if np.isinf(value).any():
            inf_fields.append(name)
    return {
        "checked_numeric_fields": checked,
        "has_nan": bool(nan_fields),
        "nan_fields": nan_fields,
        "has_inf": bool(inf_fields),
        "inf_fields": inf_fields,
    }


def video_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 0.0))
        frame_count = metadata.get("nframes")
        if not frame_count or not math.isfinite(float(frame_count)):
            frame_count = reader.count_frames()
        return {
            "path": str(path.resolve()),
            "fps": fps,
            "num_frames": int(frame_count),
            "duration": float(frame_count) / fps if fps > 0 else None,
        }
    finally:
        reader.close()


def infer_fps(
    rollout: Rollout, explicit_fps: float | None, video_info: Mapping[str, Any] | None
) -> tuple[float, str]:
    if explicit_fps is not None:
        if explicit_fps <= 0:
            raise ValueError("--fps must be positive")
        return float(explicit_fps), "command_line"

    _, time_value = field_by_name(rollout.payload, TIME_FIELDS)
    if time_value is not None:
        times = np.asarray(time_value, dtype=float).reshape(-1)
        if len(times) == rollout.num_frames:
            differences = np.diff(times)
            differences = differences[np.isfinite(differences) & (differences > 0)]
            if differences.size:
                return float(1.0 / np.median(differences)), "rollout_sim_time"

    _, scalar_fps = field_by_name(rollout.payload, ("fps", "control_fps", "policy_fps"))
    if scalar_fps is not None and np.asarray(scalar_fps).size == 1:
        fps = float(np.asarray(scalar_fps).reshape(-1)[0])
        if fps > 0:
            return fps, "rollout_scalar"

    if video_info and float(video_info["fps"]) > 0:
        return float(video_info["fps"]), "video_metadata"
    raise ValueError("unable to infer fps; pass --fps or provide simulation time/video")


def boolean_series(value: Any, num_frames: int) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != num_frames:
        return None
    if array.ndim > 1:
        array = np.any(array.astype(bool), axis=tuple(range(1, array.ndim)))
    return array.astype(bool)


def first_true(value: Any, num_frames: int) -> int | None:
    series = boolean_series(value, num_frames)
    if series is None:
        return None
    indices = np.flatnonzero(series)
    return int(indices[0]) if indices.size else None


def find_reset_frame(payload: Any, num_frames: int) -> tuple[int | None, str | None]:
    flat = flatten_mapping(payload)
    candidates = []
    for wanted in RESET_FIELDS:
        for name, value in flat.items():
            if name.split(".")[-1].lower() == wanted:
                frame = first_true(value, num_frames)
                if frame is not None:
                    candidates.append((frame, name))
    return min(candidates) if candidates else (None, None)


def quaternion_tilt_deg(quaternions: np.ndarray) -> np.ndarray:
    """Return root tilt from MuJoCo wxyz quaternions."""
    quat = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.where(norms > 0, norms, 1.0)
    w, x, y, z = (quat[:, index] for index in range(4))
    gravity_z = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(gravity_z, -1.0, 1.0)))


def detect_failure_frame(
    rollout: Rollout, root_height_min: float | None, root_tilt_max_deg: float | None
) -> tuple[int | None, str | None, str]:
    flat = flatten_mapping(rollout.payload)
    available_native = []
    detected = []
    for wanted in NATIVE_FAILURE_FIELDS:
        for name, value in flat.items():
            if name.split(".")[-1].lower() != wanted:
                continue
            series = boolean_series(value, rollout.num_frames)
            if series is None:
                continue
            available_native.append(name)
            indices = np.flatnonzero(series)
            if indices.size:
                detected.append((int(indices[0]), name))
    if detected:
        frame, reason = min(detected)
        return frame, reason, "native"
    if available_native:
        return None, None, "native_clear"

    fallback_checks = []
    if root_height_min is not None:
        name, value = field_by_name(rollout.payload, ROOT_POS_FIELDS)
        if value is None:
            raise RuntimeError(f"{UNKNOWN_FAILURE_DETECTION}: root position field missing")
        positions = np.asarray(value)
        if (
            positions.ndim != 2
            or positions.shape[0] != rollout.num_frames
            or positions.shape[1] < 3
        ):
            raise RuntimeError(f"{UNKNOWN_FAILURE_DETECTION}: invalid {name} shape")
        indices = np.flatnonzero(positions[:, 2] < root_height_min)
        if indices.size:
            fallback_checks.append((int(indices[0]), f"{name}<root_height_min"))

    if root_tilt_max_deg is not None:
        name, value = field_by_name(rollout.payload, ROOT_QUAT_FIELDS)
        if value is None:
            raise RuntimeError(f"{UNKNOWN_FAILURE_DETECTION}: root quaternion field missing")
        quaternions = np.asarray(value)
        if (
            quaternions.ndim != 2
            or quaternions.shape[0] != rollout.num_frames
            or quaternions.shape[1] != 4
        ):
            raise RuntimeError(f"{UNKNOWN_FAILURE_DETECTION}: invalid {name} shape")
        indices = np.flatnonzero(quaternion_tilt_deg(quaternions) > root_tilt_max_deg)
        if indices.size:
            fallback_checks.append((int(indices[0]), f"{name}>root_tilt_max_deg"))

    if root_height_min is None and root_tilt_max_deg is None:
        raise RuntimeError(
            f"{UNKNOWN_FAILURE_DETECTION}: no native failure field and no "
            "explicit fallback threshold"
        )
    if fallback_checks:
        frame, reason = min(fallback_checks)
        return frame, reason, "fallback"
    return None, None, "fallback_clear"


def normalize_key(key: str) -> str:
    value = str(key).strip().lower()
    aliases = {"key.space": "space", "key.enter": "enter", "key.return": "enter"}
    return aliases.get(value, value.strip("'\""))


def normalize_event_type(event_type: str) -> str:
    value = str(event_type).strip().lower().replace("-", "_")
    return {
        "down": "key_down",
        "press": "key_down",
        "pressed": "key_down",
        "up": "key_up",
        "release": "key_up",
        "released": "key_up",
    }.get(value, value)


def parse_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_key_event_rows(
    rows: Sequence[Mapping[str, Any]], fps: float, num_frames: int
) -> list[KeyEvent]:
    events = []
    for row_number, row in enumerate(rows, start=2):
        frame_value = row.get("frame_idx", row.get("frame_index"))
        sim_time = parse_optional_float(row.get("sim_time", row.get("time")))
        if frame_value is None or str(frame_value).strip() == "":
            if sim_time is None:
                raise ValueError(f"key event row {row_number} has no frame_idx or sim_time")
            frame_idx = int(round(sim_time * fps))
        else:
            frame_idx = int(frame_value)
        if not 0 <= frame_idx <= num_frames:
            raise ValueError(f"key event row {row_number} frame {frame_idx} is out of range")
        event_type = normalize_event_type(str(row.get("event_type", "")))
        if event_type not in {"key_down", "key_up", "command"}:
            raise ValueError(f"key event row {row_number} has invalid event_type: {event_type}")
        command = str(row.get("command", "")).strip() or None
        events.append(
            KeyEvent(
                frame_idx=frame_idx,
                sim_time=sim_time,
                key=normalize_key(str(row.get("key", ""))),
                event_type=event_type,
                command=command,
            )
        )
    return sorted(events, key=lambda event: event.frame_idx)


def load_key_events(path: Path, fps: float, num_frames: int) -> list[KeyEvent]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[Mapping[str, Any]]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError("key events must use .csv or .jsonl")
    return parse_key_event_rows(rows, fps, num_frames)


def key_events_from_rollout(
    rollout: Rollout, fps: float
) -> tuple[str | None, list[KeyEvent] | None]:
    name, value = field_by_name(
        rollout.payload, ("key_events", "keyboard_events", "keyboard_event")
    )
    if value is None:
        return None, None
    value = unwrap_object_array(value)
    rows: list[Mapping[str, Any]]
    if isinstance(value, Mapping):
        lengths = {
            len(column)
            for column in value.values()
            if isinstance(column, (list, tuple, np.ndarray))
        }
        if len(lengths) != 1:
            raise ValueError(f"{name} column lengths are inconsistent")
        length = lengths.pop()
        rows = [
            {
                key: column[index]
                for key, column in value.items()
                if isinstance(column, (list, tuple, np.ndarray))
            }
            for index in range(length)
        ]
    elif isinstance(value, np.ndarray) and value.dtype.names:
        rows = [{field: row[field].item() for field in value.dtype.names} for row in value]
    elif isinstance(value, np.ndarray) and value.dtype == object:
        rows = list(value.tolist())
    elif isinstance(value, (list, tuple)):
        rows = list(value)
    else:
        raise ValueError(f"{name} has unsupported type {type(value).__name__}")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} must contain event records")
    return name, parse_key_event_rows(rows, fps, rollout.num_frames)


def load_key_map(path: Path | None) -> tuple[dict[str, str], list[str]]:
    if path is None:
        return {}, []
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("key-map JSON must contain an object")
    if "mapping" in config:
        mapping_obj = config["mapping"]
        priority_obj = config.get("priority", [])
    else:
        mapping_obj = {key: value for key, value in config.items() if key != "priority"}
        priority_obj = config.get("priority", [])
    if not isinstance(mapping_obj, Mapping):
        raise ValueError("key-map 'mapping' must contain an object")
    mapping = {}
    for key, value in mapping_obj.items():
        if isinstance(value, Mapping):
            command = value.get("command")
        else:
            command = value
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"invalid command mapping for key {key!r}")
        mapping[normalize_key(str(key))] = command.strip()
    if not isinstance(priority_obj, list) or not all(
        isinstance(item, str) for item in priority_obj
    ):
        raise ValueError("key-map priority must be a list of keys")
    return mapping, [normalize_key(item) for item in priority_obj]


def event_rows(events: Sequence[KeyEvent]) -> list[dict[str, Any]]:
    return [
        {
            "frame_idx": event.frame_idx,
            "sim_time": "" if event.sim_time is None else event.sim_time,
            "key": event.key,
            "event_type": event.event_type,
            "command": event.command or "",
        }
        for event in events
    ]


def write_key_events(path: Path, events: Sequence[KeyEvent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("frame_idx", "sim_time", "key", "event_type", "command")
        )
        writer.writeheader()
        writer.writerows(event_rows(events))


def active_command(
    active_keys: set[str], mapping: Mapping[str, str], priority: Sequence[str]
) -> tuple[str, str]:
    mapped = [(key, mapping[key]) for key in sorted(active_keys) if key in mapping]
    if not mapped:
        return "neutral", "+".join(sorted(active_keys))
    commands = {command for _, command in mapped}
    if len(commands) == 1:
        return mapped[0][1], "+".join(key for key, _ in mapped)
    if priority:
        for key in priority:
            if key in active_keys and key in mapping:
                return mapping[key], "+".join(sorted(active_keys))
    return "combined_command", "+".join(sorted(active_keys))


def command_history_from_events(
    events: Sequence[KeyEvent],
    mapping: Mapping[str, str],
    priority: Sequence[str],
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    commands = np.full(num_frames, "neutral", dtype=object)
    keys = np.full(num_frames, "", dtype=object)
    by_frame: dict[int, list[KeyEvent]] = defaultdict(list)
    for event in events:
        by_frame[event.frame_idx].append(event)

    active_keys: set[str] = set()
    direct_command: str | None = None
    for frame in range(num_frames):
        for event in by_frame.get(frame, []):
            if event.event_type == "key_down":
                if event.key:
                    active_keys.add(event.key)
            elif event.event_type == "key_up":
                active_keys.discard(event.key)
            if event.command:
                direct_command = event.command
            elif event.event_type == "command":
                raise ValueError(f"command event at frame {frame} has no command value")

        mapped_command, key = active_command(active_keys, mapping, priority)
        if mapping:
            command = mapped_command
        elif direct_command is not None:
            command = direct_command
        elif active_keys:
            raise ValueError("key events have no commands and no --key-map was supplied")
        else:
            command = "neutral"
        commands[frame] = command
        keys[frame] = key
    return commands, keys


def command_history_from_rollout(rollout: Rollout) -> tuple[np.ndarray, np.ndarray]:
    name, value = field_by_name(rollout.payload, COMMAND_FIELDS)
    if value is None:
        raise ValueError("no --key-events and no rollout command history were found")
    commands = np.asarray(value).reshape(-1)
    if len(commands) != rollout.num_frames:
        raise ValueError(f"{name} is not frame-aligned")
    if commands.dtype.kind not in {"U", "S", "O"}:
        raise ValueError(
            f"{name} is numeric; semantic segmentation requires key events or string commands"
        )
    values = np.asarray(
        [item.decode() if isinstance(item, (bytes, np.bytes_)) else str(item) for item in commands],
        dtype=object,
    )
    return values, np.full(rollout.num_frames, "", dtype=object)


def runs(commands: np.ndarray, keys: np.ndarray) -> list[Span]:
    if not len(commands):
        return []
    spans = []
    start = 0
    for frame in range(1, len(commands) + 1):
        if frame == len(commands) or commands[frame] != commands[start]:
            key_values = [str(item) for item in keys[start:frame] if str(item)]
            key = Counter(key_values).most_common(1)[0][0] if key_values else ""
            spans.append(Span(str(commands[start]), key, start, frame))
            start = frame
    return spans


def merge_spans(spans: Sequence[Span], max_gap_frames: int) -> list[Span]:
    if max_gap_frames <= 0:
        return list(spans)
    merged: list[Span] = []
    index = 0
    while index < len(spans):
        current = spans[index]
        if (
            index + 2 < len(spans)
            and current.command != "neutral"
            and spans[index + 1].command == "neutral"
            and spans[index + 1].end - spans[index + 1].start <= max_gap_frames
            and spans[index + 2].command == current.command
        ):
            next_span = spans[index + 2]
            merged.append(
                Span(
                    current.command,
                    current.key or next_span.key,
                    current.start,
                    next_span.end,
                )
            )
            index += 3
            continue
        merged.append(current)
        index += 1
    return merge_spans(merged, max_gap_frames) if len(merged) < len(spans) else merged


def safe_label(command: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", command.strip().lower()).strip("_")
    return label or "unlabelled"


def build_segments(
    spans: Sequence[Span],
    episode_id: str,
    fps: float,
    num_frames: int,
    failure_frame: int | None,
    failure_reason: str | None,
    valid_end_frame: int,
    reset_frame: int | None,
    min_duration: float,
    pre_padding: float,
    post_padding: float,
    include_neutral: bool,
) -> list[Segment]:
    pre_frames = round(pre_padding * fps)
    post_frames = round(post_padding * fps)
    minimum_frames = math.ceil(min_duration * fps)
    counts: Counter[str] = Counter()
    segments = []

    for span in spans:
        label = safe_label(span.command)
        identifier = f"{label}_{counts[label]:03d}"
        counts[label] += 1
        original_end = span.end
        start = max(0, span.start - pre_frames)
        end = min(num_frames, span.end + post_frames)
        truncated = failure_frame is not None and end > valid_end_frame and start < valid_end_frame
        status = "valid"
        discard_reason = ""

        if reset_frame is not None and start >= reset_frame:
            status, discard_reason = "discarded", "after_reset"
        elif failure_frame is not None and start >= failure_frame:
            status, discard_reason = "discarded", "after_failure"
        elif start >= valid_end_frame:
            status, discard_reason = "discarded", "failure_margin"
        else:
            end = min(end, valid_end_frame)

        if span.command == "neutral" and not include_neutral:
            status, discard_reason = "discarded", "neutral_excluded"
        if end <= start:
            status, discard_reason = "discarded", discard_reason or "empty_after_clipping"
        elif end - start < minimum_frames:
            status, discard_reason = "discarded", discard_reason or "shorter_than_min_duration"

        num_segment_frames = max(0, end - start)
        segments.append(
            Segment(
                segment_id=identifier,
                episode_id=episode_id,
                motion_label=label,
                key=span.key,
                command=span.command,
                start_frame=start,
                end_frame=end,
                start_time=start / fps,
                end_time=end / fps,
                duration=num_segment_frames / fps,
                num_frames=num_segment_frames,
                status=status,
                failure_detected=failure_frame is not None,
                failure_frame=failure_frame,
                failure_reason=failure_reason or "",
                truncated_by_failure=truncated,
                original_end_frame=original_end,
                valid_end_frame=valid_end_frame,
                reset_detected=reset_frame is not None,
                discard_reason=discard_reason,
                notes="end_frame is exclusive",
            )
        )
    return segments


def slice_payload(value: Any, start: int, end: int, num_frames: int) -> Any:
    if isinstance(value, np.ndarray):
        return value[start:end] if value.ndim > 0 and value.shape[0] == num_frames else value
    if isinstance(value, Mapping):
        return {key: slice_payload(item, start, end, num_frames) for key, item in value.items()}
    if isinstance(value, list) and len(value) == num_frames:
        return value[start:end]
    if isinstance(value, tuple) and len(value) == num_frames:
        return value[start:end]
    return value


def save_segment_payload(rollout: Rollout, segment: Segment, destination_stem: Path) -> Path:
    sliced = slice_payload(
        rollout.payload, segment.start_frame, segment.end_frame, rollout.num_frames
    )
    if rollout.suffix == ".npy":
        path = destination_stem.with_suffix(".npy")
        np.save(path, sliced, allow_pickle=True)
    elif rollout.suffix == ".npz":
        path = destination_stem.with_suffix(".npz")
        if not isinstance(sliced, Mapping):
            raise ValueError("NPZ rollout did not load as a mapping")
        np.savez(path, **sliced)
    else:
        path = destination_stem.with_suffix(".pkl")
        with path.open("wb") as handle:
            pickle.dump(sliced, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def write_segments_csv(path: Path, segments: Sequence[Segment]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        for segment in segments:
            writer.writerow(asdict(segment))


def relative_to_episode(path: Path, episode_dir: Path) -> str:
    return str(path.relative_to(episode_dir))


def validate_segments(segments: Sequence[Segment], num_frames: int, fps: float) -> None:
    for segment in segments:
        if segment.status != "valid":
            continue
        if not (0 <= segment.start_frame < segment.end_frame <= num_frames):
            raise AssertionError(f"invalid frame range for {segment.segment_id}")
        if segment.num_frames != segment.end_frame - segment.start_frame:
            raise AssertionError(f"frame count mismatch for {segment.segment_id}")
        if not math.isclose(segment.duration, segment.num_frames / fps, abs_tol=1e-9):
            raise AssertionError(f"duration mismatch for {segment.segment_id}")
        if segment.failure_frame is not None and segment.end_frame > segment.valid_end_frame:
            raise AssertionError(f"failure boundary crossed by {segment.segment_id}")


def overlay_frame(frame: np.ndarray, lines: Iterable[str], invalid: bool = False) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame).astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    text_lines = list(lines)
    line_height = 14
    width = min(image.width, max(280, max((len(line) for line in text_lines), default=1) * 7))
    draw.rectangle((0, 0, width, 10 + line_height * len(text_lines)), fill=(0, 0, 0, 175))
    for index, line in enumerate(text_lines):
        draw.text((6, 5 + index * line_height), line, fill=(255, 255, 255, 255), font=font)
    if invalid:
        draw.rectangle((0, image.height - 30, image.width, image.height), fill=(150, 0, 0, 190))
        draw.text((8, image.height - 23), "INVALID: failure/reset region", fill="white", font=font)
    return np.asarray(image)


def segment_at_frame(segments: Sequence[Segment], frame: int) -> Segment | None:
    for segment in segments:
        if segment.status == "valid" and segment.start_frame <= frame < segment.end_frame:
            return segment
    return None


def render_from_synchronized_video(
    video_path: Path,
    video_info: Mapping[str, Any],
    episode_dir: Path,
    episode_id: str,
    segments: Sequence[Segment],
    commands: np.ndarray,
    keys: np.ndarray,
    rollout_fps: float,
    num_frames: int,
    failure_frame: int | None,
    valid_end_frame: int,
    reset_frame: int | None,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    video_fps = float(video_info["fps"])
    video_frames = int(video_info["num_frames"])
    rollout_duration = num_frames / rollout_fps
    video_duration = video_frames / video_fps
    tolerance = max(0.25, 2.0 / rollout_fps)
    if abs(rollout_duration - video_duration) > tolerance:
        raise ValueError(
            "video/rollout synchronization check failed: "
            f"{video_duration:.3f}s vs {rollout_duration:.3f}s"
        )

    preview_dir = episode_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    preview_path = preview_dir / "full_rollout_with_segments.mp4"
    preview_writer = imageio.get_writer(preview_path, fps=video_fps, macro_block_size=1)
    writers: dict[str, Any] = {}
    segment_paths: dict[str, Path] = {}
    for segment in segments:
        if segment.status != "valid":
            continue
        path = episode_dir / segment.motion_label / f"{segment.segment_id}.mp4"
        writers[segment.segment_id] = imageio.get_writer(path, fps=video_fps, macro_block_size=1)
        segment_paths[segment.segment_id] = path

    failure_writer = None
    failure_path = None
    failure_video_start = None
    if failure_frame is not None:
        failure_dir = episode_dir / "failures"
        failure_dir.mkdir(exist_ok=True)
        failure_path = failure_dir / "failure_preview.mp4"
        failure_writer = imageio.get_writer(failure_path, fps=video_fps, macro_block_size=1)
        failure_video_start = max(0, failure_frame - round(rollout_fps))

    reader = imageio.get_reader(video_path)
    decoded = 0
    try:
        for video_index, frame in enumerate(reader):
            decoded += 1
            rollout_frame = min(num_frames - 1, int(video_index * rollout_fps / video_fps))
            segment = segment_at_frame(segments, rollout_frame)
            markers = []
            if rollout_frame == failure_frame:
                markers.append("FAILURE")
            if rollout_frame == valid_end_frame:
                markers.append("VALID_END")
            if rollout_frame == reset_frame:
                markers.append("RESET")
            lines = [
                f"episode: {episode_id}",
                f"frame/time: {rollout_frame} / {rollout_frame / rollout_fps:.3f}s",
                f"key: {keys[rollout_frame] or '-'}",
                f"command: {commands[rollout_frame]}",
                f"segment: {segment.segment_id if segment else '-'}",
                f"boundary: {', '.join(markers) if markers else '-'}",
            ]
            invalid = rollout_frame >= valid_end_frame
            preview_writer.append_data(overlay_frame(frame, lines, invalid))

            if segment is not None:
                segment_lines = [
                    f"motion: {segment.motion_label}",
                    f"key/command: {segment.key or '-'} / {segment.command}",
                    f"episode: {episode_id}",
                    f"time: {segment.start_time:.3f}-{segment.end_time:.3f}s",
                ]
                if segment.truncated_by_failure:
                    segment_lines.append("truncated_before_failure")
                writers[segment.segment_id].append_data(overlay_frame(frame, segment_lines))

            if (
                failure_writer is not None
                and failure_video_start is not None
                and rollout_frame >= failure_video_start
            ):
                failure_writer.append_data(overlay_frame(frame, lines, invalid))
    finally:
        reader.close()
        preview_writer.close()
        for writer in writers.values():
            writer.close()
        if failure_writer is not None:
            failure_writer.close()

    for segment in segments:
        if segment.segment_id in segment_paths:
            segment.video_path = relative_to_episode(segment_paths[segment.segment_id], episode_dir)
    return {
        "method": "synchronized_video",
        "source_video": str(video_path.resolve()),
        "video_fps": video_fps,
        "decoded_frames": decoded,
        "preview_path": relative_to_episode(preview_path, episode_dir),
        "failure_preview_path": (
            relative_to_episode(failure_path, episode_dir) if failure_path else None
        ),
    }


def find_qpos(rollout: Rollout) -> tuple[str | None, np.ndarray | None]:
    name, value = field_by_name(rollout.payload, QPOS_FIELDS)
    if value is None:
        return None, None
    qpos = np.asarray(value)
    if qpos.ndim != 2 or qpos.shape[0] != rollout.num_frames:
        return name, None
    return name, qpos


def render_pose_only_video(
    robot_xml: Path,
    pose: np.ndarray,
    output_path: Path,
    fps: float,
) -> None:
    """Render a 36-D BFM G1 pose sequence without the skateboard scene."""
    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if model.nq != 36:
        raise ValueError(f"pose preview model must have nq=36, got {model.nq}")
    sequence = np.asarray(pose, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != model.nq:
        raise ValueError(f"pose preview requires [T, 36], got {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError("pose preview contains NaN or Inf")

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.5
    camera.azimuth = 145.0
    camera.elevation = -15.0
    writer = imageio.get_writer(
        output_path,
        fps=float(fps),
        macro_block_size=1,
        codec="libx264",
    )
    try:
        for frame in sequence:
            quaternion = frame[3:7]
            norm = np.linalg.norm(quaternion)
            if norm <= 0:
                raise ValueError("pose preview has an invalid root quaternion")
            data.qpos[:] = frame
            data.qpos[3:7] = quaternion / norm
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, -0.1])
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


def render_from_qpos(
    robot_xml: Path,
    rollout: Rollout,
    episode_dir: Path,
    episode_id: str,
    segments: Sequence[Segment],
    commands: np.ndarray,
    keys: np.ndarray,
    fps: float,
    failure_frame: int | None,
    valid_end_frame: int,
    reset_frame: int | None,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco

    qpos_name, qpos = find_qpos(rollout)
    if qpos is None:
        raise ValueError("pose replay requires a frame-aligned qpos field")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"qpos width {qpos.shape[1]} != model.nq {model.nq}; joint mapping is UNKNOWN"
        )

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -20.0

    preview_dir = episode_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    preview_path = preview_dir / "full_rollout_with_segments.mp4"
    preview_writer = imageio.get_writer(preview_path, fps=fps, macro_block_size=1)
    writers: dict[str, Any] = {}
    segment_paths: dict[str, Path] = {}
    for segment in segments:
        if segment.status == "valid":
            path = episode_dir / segment.motion_label / f"{segment.segment_id}.mp4"
            writers[segment.segment_id] = imageio.get_writer(path, fps=fps, macro_block_size=1)
            segment_paths[segment.segment_id] = path

    failure_writer = None
    failure_path = None
    failure_start = None
    if failure_frame is not None:
        failure_dir = episode_dir / "failures"
        failure_dir.mkdir(exist_ok=True)
        failure_path = failure_dir / "failure_preview.mp4"
        failure_writer = imageio.get_writer(failure_path, fps=fps, macro_block_size=1)
        failure_start = max(0, failure_frame - round(fps))

    try:
        for frame_index in range(rollout.num_frames):
            data.qpos[:] = qpos[frame_index]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            segment = segment_at_frame(segments, frame_index)
            markers = []
            if frame_index == failure_frame:
                markers.append("FAILURE")
            if frame_index == valid_end_frame:
                markers.append("VALID_END")
            if frame_index == reset_frame:
                markers.append("RESET")
            lines = [
                f"episode: {episode_id}",
                f"frame/time: {frame_index} / {frame_index / fps:.3f}s",
                f"key: {keys[frame_index] or '-'}",
                f"command: {commands[frame_index]}",
                f"segment: {segment.segment_id if segment else '-'}",
                f"boundary: {', '.join(markers) if markers else '-'}",
            ]
            invalid = frame_index >= valid_end_frame
            preview_writer.append_data(overlay_frame(frame, lines, invalid))
            if segment is not None:
                segment_lines = [
                    f"motion: {segment.motion_label}",
                    f"key/command: {segment.key or '-'} / {segment.command}",
                    f"episode: {episode_id}",
                    f"time: {segment.start_time:.3f}-{segment.end_time:.3f}s",
                ]
                if segment.truncated_by_failure:
                    segment_lines.append("truncated_before_failure")
                writers[segment.segment_id].append_data(overlay_frame(frame, segment_lines))
            if (
                failure_writer is not None
                and failure_start is not None
                and frame_index >= failure_start
            ):
                failure_writer.append_data(overlay_frame(frame, lines, invalid))
    finally:
        renderer.close()
        preview_writer.close()
        for writer in writers.values():
            writer.close()
        if failure_writer is not None:
            failure_writer.close()

    for segment in segments:
        if segment.segment_id in segment_paths:
            segment.video_path = relative_to_episode(segment_paths[segment.segment_id], episode_dir)
    return {
        "method": "mujoco_qpos_replay",
        "robot_xml": str(robot_xml.resolve()),
        "qpos_field": qpos_name,
        "model_nq": model.nq,
        "quaternion_convention": "MuJoCo wxyz",
        "preview_path": relative_to_episode(preview_path, episode_dir),
        "failure_preview_path": (
            relative_to_episode(failure_path, episode_dir) if failure_path else None
        ),
    }


def rollout_summary(
    rollout: Rollout,
    fps: float,
    fps_source: str,
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    flat = flatten_mapping(rollout.payload)
    fields = {}
    for name, value in flat.items():
        if isinstance(value, np.ndarray):
            fields[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif not isinstance(value, Mapping):
            fields[name] = {"type": type(value).__name__}
    return {
        "file_type": rollout.suffix,
        "top_level_type": rollout.top_type,
        "top_level_keys": rollout.keys,
        "shape": rollout.shape,
        "dtype": rollout.dtype,
        "num_frames": rollout.num_frames,
        "fps": fps,
        "fps_source": fps_source,
        "fields": fields,
        "numeric_integrity": dict(integrity),
    }


def prepare_output(
    output_dir: Path,
    episode_id: str,
    overwrite: bool,
    protected_paths: Sequence[Path | None],
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
        raise ValueError("--episode-id may contain only letters, numbers, underscores, and hyphens")
    episode_dir = output_dir / episode_id
    resolved_episode_dir = episode_dir.resolve()
    for protected_path in protected_paths:
        if protected_path is None:
            continue
        try:
            protected_path.resolve().relative_to(resolved_episode_dir)
        except ValueError:
            continue
        raise ValueError(f"refusing output directory containing source input: {protected_path}")
    if episode_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{episode_dir} already exists; pass --overwrite to replace this test output"
            )
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True)
    return episode_dir


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.parallel_heading_values:
        return run_parallel(args)
    if args.live:
        return run_live(args)
    source_hash_before = sha256(args.rollout)
    rollout = load_rollout(args.rollout)
    video_info = video_metadata(args.video)
    fps, fps_source = infer_fps(rollout, args.fps, video_info)
    integrity = numeric_integrity(rollout.payload, rollout.num_frames)
    schema = rollout_summary(rollout, fps, fps_source, integrity)
    print(json.dumps(schema, indent=2))

    episode_dir = prepare_output(
        args.output_dir,
        args.episode_id,
        args.overwrite,
        (args.rollout, args.key_events, args.key_map, args.video, args.robot_xml),
    )
    raw_reference = {
        "episode_id": args.episode_id,
        "source_rollout_path": str(args.rollout.resolve()),
        "source_rollout_sha256": source_hash_before,
        "source_rollout_size_bytes": args.rollout.stat().st_size,
        "source_video": str(args.video.resolve()) if args.video else None,
        "source_key_events": str(args.key_events.resolve()) if args.key_events else None,
        "source_key_map": str(args.key_map.resolve()) if args.key_map else None,
        "rollout_schema": schema,
    }
    write_json(episode_dir / "raw_reference.json", raw_reference)

    report: dict[str, Any] = {
        "status": "processing",
        "episode_id": args.episode_id,
        "rollout_schema": schema,
        "video_input": video_info,
        "unknown_fields": [],
    }
    try:
        failure_frame, failure_reason, failure_source = detect_failure_frame(
            rollout, args.root_height_min, args.root_tilt_max_deg
        )
        reset_frame, reset_reason = find_reset_frame(rollout.payload, rollout.num_frames)
        margin_frames = round(args.failure_margin * fps)
        valid_end_frame = rollout.num_frames
        if failure_frame is not None:
            valid_end_frame = max(0, failure_frame - margin_frames)
        if reset_frame is not None:
            valid_end_frame = min(valid_end_frame, reset_frame)

        embedded_event_field = None
        if args.key_events:
            events = load_key_events(args.key_events, fps, rollout.num_frames)
            mapping, priority = load_key_map(args.key_map)
            commands, keys = command_history_from_events(
                events, mapping, priority, rollout.num_frames
            )
            write_key_events(episode_dir / "key_events.csv", events)
            command_source = "key_events"
        else:
            embedded_event_field, embedded_events = key_events_from_rollout(rollout, fps)
            mapping, priority = load_key_map(args.key_map)
            if embedded_events is not None:
                events = embedded_events
                commands, keys = command_history_from_events(
                    events, mapping, priority, rollout.num_frames
                )
                command_source = f"rollout:{embedded_event_field}"
            else:
                events = []
                commands, keys = command_history_from_rollout(rollout)
                command_source = "rollout_command_history"
            write_key_events(episode_dir / "key_events.csv", events)

        spans = merge_spans(runs(commands, keys), round(args.merge_gap * fps))
        segments = build_segments(
            spans=spans,
            episode_id=args.episode_id,
            fps=fps,
            num_frames=rollout.num_frames,
            failure_frame=failure_frame,
            failure_reason=failure_reason,
            valid_end_frame=valid_end_frame,
            reset_frame=reset_frame,
            min_duration=args.min_duration,
            pre_padding=args.pre_padding,
            post_padding=args.post_padding,
            include_neutral=args.include_neutral,
        )
        validate_segments(segments, rollout.num_frames, fps)

        for segment in segments:
            if segment.status != "valid":
                continue
            directory = episode_dir / segment.motion_label
            directory.mkdir(exist_ok=True)
            pose_path = save_segment_payload(rollout, segment, directory / segment.segment_id)
            segment.pose_path = relative_to_episode(pose_path, episode_dir)
            metadata = asdict(segment)
            metadata.update(
                {
                    "source_rollout_path": str(args.rollout.resolve()),
                    "source_rollout_sha256": source_hash_before,
                    "source_format": rollout.suffix,
                    "sliced_existing_fields_only": True,
                }
            )
            write_json(directory / f"{segment.segment_id}.json", metadata)

        video_result: dict[str, Any]
        try:
            if args.video:
                assert video_info is not None
                video_result = render_from_synchronized_video(
                    args.video,
                    video_info,
                    episode_dir,
                    args.episode_id,
                    segments,
                    commands,
                    keys,
                    fps,
                    rollout.num_frames,
                    failure_frame,
                    valid_end_frame,
                    reset_frame,
                )
            elif args.robot_xml:
                video_result = render_from_qpos(
                    args.robot_xml,
                    rollout,
                    episode_dir,
                    args.episode_id,
                    segments,
                    commands,
                    keys,
                    fps,
                    failure_frame,
                    valid_end_frame,
                    reset_frame,
                )
            else:
                qpos_name, qpos = find_qpos(rollout)
                missing = ["--video"]
                if qpos is None:
                    missing.append("frame-aligned qpos")
                missing.append("--robot-xml")
                video_result = {
                    "method": "unavailable",
                    "missing": missing,
                    "qpos_field": qpos_name,
                    "note": "pose segments retained; no fake video generated",
                }
                report["unknown_fields"].append("video_generation_capability")
        except Exception as error:
            video_result = {
                "method": "failed",
                "error": f"{type(error).__name__}: {error}",
                "note": "pose segments retained; no fake video generated",
            }
            report["unknown_fields"].append("video_generation_failed")

        if failure_frame is not None:
            failure_dir = episode_dir / "failures"
            failure_dir.mkdir(exist_ok=True)
            write_json(
                failure_dir / "failure_info.json",
                {
                    "failure_frame": failure_frame,
                    "failure_time": failure_frame / fps,
                    "failure_reason": failure_reason,
                    "failure_source": failure_source,
                    "failure_margin_frames": margin_frames,
                    "valid_end_frame": valid_end_frame,
                    "valid_end_time": valid_end_frame / fps,
                },
            )

        source_hash_after = sha256(args.rollout)
        if source_hash_after != source_hash_before:
            raise RuntimeError("source rollout changed during processing")
        write_segments_csv(episode_dir / "segments.csv", segments)

        report.update(
            {
                "status": "completed",
                "command_source": command_source,
                "key_mapping": mapping,
                "key_priority": priority or "UNKNOWN",
                "failure_detected": failure_frame is not None,
                "failure_frame": failure_frame,
                "failure_reason": failure_reason,
                "failure_detection_source": failure_source,
                "failure_margin_frames": margin_frames,
                "valid_end_frame": valid_end_frame,
                "reset_detected": reset_frame is not None,
                "reset_frame": reset_frame,
                "reset_reason": reset_reason,
                "post_reset_ignore_frames": round(args.post_reset_ignore * fps),
                "first_reset_ends_processing": True,
                "segments": [asdict(segment) for segment in segments],
                "valid_segment_count": sum(segment.status == "valid" for segment in segments),
                "discarded_segment_count": sum(segment.status != "valid" for segment in segments),
                "video": video_result,
                "source_sha256_before": source_hash_before,
                "source_sha256_after": source_hash_after,
                "source_unchanged": True,
            }
        )
        write_json(episode_dir / "processing_report.json", report)
        # Rewrite after video paths have been assigned.
        write_segments_csv(episode_dir / "segments.csv", segments)
        for segment in segments:
            if segment.status == "valid":
                metadata_path = episode_dir / segment.motion_label / f"{segment.segment_id}.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata.update(asdict(segment))
                write_json(metadata_path, metadata)
        print(f"Output: {episode_dir}")
        print(
            f"Valid segments: {report['valid_segment_count']}; "
            f"discarded: {report['discarded_segment_count']}"
        )
        if video_result["method"] in {"unavailable", "failed"}:
            video_detail = video_result.get("missing") or video_result.get("error")
            print(f"Video: {video_result['method']} ({video_detail})")
        return 0
    except RuntimeError as error:
        if UNKNOWN_FAILURE_DETECTION not in str(error):
            raise
        report.update(
            {
                "status": UNKNOWN_FAILURE_DETECTION,
                "error": str(error),
                "export_stopped": True,
                "source_unchanged": sha256(args.rollout) == source_hash_before,
            }
        )
        write_json(episode_dir / "processing_report.json", report)
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
