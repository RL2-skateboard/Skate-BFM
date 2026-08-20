#!/usr/bin/env python3
"""Build the fixed-window continuous BFM MotionLib from canonical HUSKY raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

PHASES = ("push", "push2steer", "steer_left", "steer_forward", "steer_right", "steer2push")
PHASE_IDS = {
    "push": 0,
    "push2steer": 1,
    "steer_left": 2,
    "steer_right": 3,
    "steer_forward": 4,
    "steer2push": 5,
    "fall": 6,
}
PHASE_LABELS = {value: key for key, value in PHASE_IDS.items()}
BFM_REQUIRED = {
    "root_trans_offset": (3,),
    "pose_aa": (30, 3),
    "dof": (29,),
    "root_rot": (4,),
    "smpl_joints": (24, 3),
}
SKATE_REQUIRED = {
    "action": None,
    "board_root_pos": (3,),
    "board_root_quat": (4,),
    "board_root_lin_vel": (3,),
    "board_root_ang_vel": (3,),
    "board_dof_pos": None,
    "board_dof_vel": None,
    "phase_id": (),
    "phase_value": (),
    "board_heading_delta": (),
}
RAW_REQUIRED = (
    "qpos",
    "qvel",
    "action",
    "root_pos",
    "root_quat",
    "dof_pos",
    "dof_vel",
    "body_pos",
    "body_quat",
    "body_lin_vel",
    "body_ang_vel",
    "board_root_pos",
    "board_root_quat",
    "board_root_lin_vel",
    "board_root_ang_vel",
    "board_dof_pos",
    "board_dof_vel",
    "board_heading_delta",
    "frame_idx",
    "sim_time",
    "phase_id",
    "phase_value",
    "command_v",
    "command_h",
    "fall",
    "reset",
)
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate-continuous", action="store_true")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--dataset-split", choices=("train", "validation", "test"), required=True)
    p.add_argument("--bfm-repo", type=Path, required=True)
    p.add_argument("--bfm-reference", type=Path, required=True)
    p.add_argument("--robot-xml", type=Path, required=True)
    p.add_argument("--husky-xml", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--qc-root", type=Path)
    p.add_argument("--qc-seed", type=int, default=20260813)
    p.add_argument("--seq-length", type=int, default=8)
    p.add_argument("--clip-frames", type=int, default=500)
    p.add_argument("--validate-motionlib", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def finite(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def discard_category(error: Exception) -> str:
    message = str(error).lower()
    if "nan" in message or "inf" in message:
        return "nan_inf"
    if "phase" in message:
        return "invalid_phase"
    if any(word in message for word in ("align", "shape", "frame_idx", "sim_time")):
        return "alignment"
    return "other"


def motion_discard_category(error: Exception) -> str:
    category = discard_category(error)
    return category if category != "other" else "conversion_error"


def load_reference(path: Path) -> tuple[list[str], Mapping[str, Any]]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or not payload:
        raise TypeError("BFM reference must be a non-empty dict")
    record = next(iter(payload.values()))
    if not isinstance(record, Mapping):
        raise TypeError("BFM reference record must be a mapping")
    missing = sorted((set(BFM_REQUIRED) | {"fps"}) - set(record))
    if missing:
        raise ValueError(f"BFM reference is missing keys: {missing}")
    return list(record), record


def bfm_joint_contract(path: Path) -> tuple[list[str], np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    ids = [i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    names = [model.joint(i).name for i in ids]
    axes = np.asarray([model.jnt_axis[i] for i in ids], dtype=np.float32)
    if len(names) != 29 or axes.shape != (29, 3) or len(set(names)) != 29:
        raise ValueError(f"BFM XML must contain 29 unique hinge joints, got {len(names)}")
    return names, axes


def raw_root_from_arg(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if (path / "raw").is_dir():
        return path / "raw", path
    if path.name == "raw":
        return path, path.parent
    return path, path.parent


def rollout_paths(raw_root: Path, dataset_split: str) -> list[Path]:
    paths = sorted(
        path for path in raw_root.glob("round_*/rollout_*") if (path / "raw_rollout").is_dir()
    )
    if not paths:
        raise FileNotFoundError(f"no raw rollouts found below {raw_root}")
    for path in paths:
        metadata_files = sorted((path / "raw_rollout").glob("*.json"))
        if len(metadata_files) != 1:
            raise ValueError(f"{path} must contain one raw metadata JSON")
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        split = metadata.get("dataset_split")
        if split != dataset_split:
            raise ValueError(
                f"{metadata_files[0]} has dataset_split={split!r}, "
                f"expected {dataset_split!r}"
            )
    return paths


def load_raw(path: Path) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    raw_dir = path / "raw_rollout"
    files = sorted(raw_dir.glob("*.npz"))
    if len(files) != 1:
        raise ValueError(f"{raw_dir} must contain one rollout NPZ, found {len(files)}")
    raw_path = files[0]
    metadata_path = raw_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing raw metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(raw_path, allow_pickle=False) as archive:
        state = {name: archive[name] for name in archive.files}
    return raw_path, metadata, state


def validate_raw(
    raw_path: Path, metadata: Mapping[str, Any], state: Mapping[str, np.ndarray]
) -> tuple[int, dict[str, Any]]:
    frame_count = int(metadata.get("num_frames", 0))
    if frame_count <= 0:
        raise ValueError(f"{raw_path} has invalid num_frames")
    required_metadata = (
        "episode_id",
        "round_id",
        "rollout_id",
        "dataset_split",
        "fps",
        "dt",
        "joint_order",
        "body_order",
        "board_joint_order",
        "qpos_quaternion_order",
        "body_quaternion_order",
        "board_quaternion_order",
        "physics_randomization",
        "policy_checkpoint",
        "robot_xml",
        "terminal_reason",
        "fixed_bfm_joints",
    )
    missing = [name for name in required_metadata if name not in metadata]
    if missing:
        raise ValueError(f"{raw_path} missing metadata: {missing}")
    if metadata["dataset_split"] not in {"train", "validation", "test"}:
        raise ValueError(f"{raw_path} has invalid dataset_split")
    if metadata["qpos_quaternion_order"] != "wxyz":
        raise ValueError(f"{raw_path} root quaternion order must be wxyz")
    if metadata["body_quaternion_order"] != "wxyz":
        raise ValueError(f"{raw_path} body quaternion order must be wxyz")
    if metadata["board_quaternion_order"] != "wxyz":
        raise ValueError(f"{raw_path} board quaternion order must be wxyz")
    randomization = metadata["physics_randomization"]
    if not isinstance(randomization, Mapping) or "seed" not in randomization:
        raise ValueError(f"{raw_path} has invalid physics_randomization")
    missing = [name for name in RAW_REQUIRED if name not in state]
    if missing:
        raise ValueError(f"{raw_path} missing raw fields: {missing}")
    for name, value in state.items():
        if value.ndim == 0 or value.shape[0] != frame_count:
            raise ValueError(f"{raw_path}:{name} is not aligned to {frame_count} frames")
        if np.issubdtype(value.dtype, np.number):
            finite(f"{raw_path}:{name}", value)
    if not np.array_equal(state["frame_idx"], np.arange(frame_count)):
        raise ValueError(f"{raw_path}:frame_idx is not contiguous")
    if np.any(np.diff(state["sim_time"]) < 0):
        raise ValueError(f"{raw_path}:sim_time is not monotonic")
    expected = {
        "root_pos": (frame_count, 3),
        "root_quat": (frame_count, 4),
        "dof_pos": (frame_count, 23),
        "dof_vel": (frame_count, 23),
        "action": (frame_count, 23),
        "board_root_pos": (frame_count, 3),
        "board_root_quat": (frame_count, 4),
        "board_root_lin_vel": (frame_count, 3),
        "board_root_ang_vel": (frame_count, 3),
        "phase_id": (frame_count,),
        "phase_value": (frame_count,),
        "command_v": (frame_count,),
        "command_h": (frame_count,),
        "fall": (frame_count,),
        "reset": (frame_count,),
        "board_heading_delta": (frame_count,),
    }
    for name, shape in expected.items():
        if state[name].shape != shape:
            raise ValueError(f"{raw_path}:{name} has {state[name].shape}, expected {shape}")
    phase_ids = np.asarray(state["phase_id"])
    if not np.issubdtype(phase_ids.dtype, np.integer):
        raise ValueError(f"{raw_path}:phase_id must be integer")
    unknown = sorted(set(phase_ids.tolist()) - set(range(7)))
    if unknown:
        raise ValueError(f"{raw_path} contains unknown phase IDs: {unknown}")
    phase_fall = phase_ids == PHASE_IDS["fall"]
    if not np.array_equal(np.asarray(state["fall"], dtype=bool), phase_fall):
        raise ValueError(f"{raw_path}:fall and phase_id are inconsistent")
    if not np.all((0 <= state["command_v"]) & (state["command_v"] <= 1.5)):
        raise ValueError(f"{raw_path}:command_v outside [0, 1.5]")
    if not np.all(np.abs(state["command_h"]) <= math.pi / 4):
        raise ValueError(f"{raw_path}:command_h outside [-pi/4, pi/4]")
    if len(np.unique(state["command_v"])) != 1 or len(np.unique(state["command_h"])) != 1:
        raise ValueError(f"{raw_path}: command is not fixed within rollout")
    if not math.isclose(float(metadata["dt"]), 1.0 / float(metadata["fps"]), abs_tol=1e-5):
        raise ValueError(f"{raw_path}: fps/dt mismatch")
    return frame_count, {
        "command_v": float(state["command_v"][0]),
        "command_h": float(state["command_h"][0]),
        "physics_seed": int(randomization["seed"]),
    }


def phase_runs(phase_ids: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = 0
    for end in range(1, len(phase_ids) + 1):
        if end == len(phase_ids) or phase_ids[end] != phase_ids[start]:
            runs.append((int(phase_ids[start]), start, end))
            start = end
    return runs


def valid_intervals(
    state: Mapping[str, np.ndarray],
    frame_count: int,
    failure_margin_frames: int,
) -> tuple[list[tuple[int, int]], Counter[str]]:
    fall_indices = np.flatnonzero(np.asarray(state["fall"], dtype=bool))
    fall_start = int(fall_indices[0]) if fall_indices.size else frame_count
    discarded = Counter({"fall": frame_count - fall_start, "failure_margin": 0, "reset": 0})
    valid_end = fall_start
    if fall_indices.size:
        trimmed_end = max(0, valid_end - failure_margin_frames)
        discarded["failure_margin"] = valid_end - trimmed_end
        valid_end = trimmed_end

    intervals: list[tuple[int, int]] = []
    cursor = 0
    for reset_index in np.flatnonzero(np.asarray(state["reset"][:valid_end], dtype=bool)):
        reset_frame = int(reset_index)
        if cursor < reset_frame:
            intervals.append((cursor, reset_frame))
        cursor = reset_frame + 1
        discarded["reset"] += 1
    if cursor < valid_end:
        intervals.append((cursor, valid_end))
    return intervals, discarded


def fixed_windows(
    intervals: Sequence[tuple[int, int]], clip_frames: int
) -> tuple[list[tuple[int, int]], int]:
    windows: list[tuple[int, int]] = []
    discarded_tail = 0
    for start, end in intervals:
        complete_end = start + ((end - start) // clip_frames) * clip_frames
        windows.extend(
            (clip_start, clip_start + clip_frames)
            for clip_start in range(start, complete_end, clip_frames)
        )
        discarded_tail += end - complete_end
    return windows, discarded_tail


def compressed_phase_sequence(phase_ids: np.ndarray) -> list[str]:
    return [PHASE_LABELS[phase_id] for phase_id, _, _ in phase_runs(phase_ids)]


def map_dof(
    source: np.ndarray,
    source_order: Sequence[str],
    target_order: Sequence[str],
    fixed: Sequence[str],
) -> tuple[np.ndarray, dict[str, str]]:
    names = [str(name).split("/")[-1] for name in source_order]
    if len(names) != source.shape[1] or len(set(names)) != len(names):
        raise ValueError("raw joint_order does not match unique dof_pos columns")
    fixed_set = set(fixed)
    target_names = [str(name).split("/")[-1] for name in target_order]
    missing = [name for name in target_names if name not in names]
    if set(missing) != fixed_set:
        raise ValueError(f"source/target joint mismatch; missing={missing}, fixed={fixed}")
    indices = {name: index for index, name in enumerate(names)}
    result = np.zeros((source.shape[0], len(target_names)), dtype=np.float32)
    mapping: dict[str, str] = {}
    for index, name in enumerate(target_names):
        if name in indices:
            result[:, index] = source[:, indices[name]]
            mapping[name] = str(source_order[indices[name]])
        else:
            mapping[name] = "fixed_zero"
    return result, mapping


def convert_record(
    metadata: Mapping[str, Any],
    state: Mapping[str, np.ndarray],
    reference_keys: Sequence[str],
    reference_record: Mapping[str, Any],
    target_order: Sequence[str],
    target_axes: np.ndarray,
    start: int,
    end: int,
    source_path: Path,
    seq_length: int,
    clip_frames: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_count = end - start
    if frame_count < seq_length + 1:
        raise ValueError(f"segment is shorter than BFM minimum {seq_length + 1}")
    if frame_count != clip_frames:
        raise ValueError(f"continuous clip has {frame_count} frames, expected {clip_frames}")
    phase_ids = np.asarray(state["phase_id"][start:end], dtype=np.int16)
    if np.any((phase_ids < 0) | (phase_ids >= PHASE_IDS["fall"])):
        raise ValueError("continuous clip contains invalid or fall phase")
    if np.any(np.asarray(state["reset"][start:end], dtype=bool)):
        raise ValueError("continuous clip crosses reset")
    dof, mapping = map_dof(
        np.asarray(state["dof_pos"][start:end], dtype=np.float32),
        metadata["joint_order"],
        target_order,
        metadata["fixed_bfm_joints"],
    )
    root_wxyz = np.asarray(state["root_quat"][start:end], dtype=np.float64)
    norms = np.linalg.norm(root_wxyz, axis=1)
    if np.any(norms <= 1e-8):
        raise ValueError("root quaternion has zero norm")
    root_xyzw = (root_wxyz / norms[:, None])[:, [1, 2, 3, 0]]
    root_rotvec = Rotation.from_quat(root_xyzw).as_rotvec().astype(np.float32)
    pose_aa = np.zeros((frame_count, 30, 3), dtype=np.float32)
    pose_aa[:, 0] = root_rotvec
    pose_aa[:, 1:] = dof[..., None] * target_axes[None, ...]
    fps = float(metadata["fps"])
    if not math.isclose(fps, round(fps), abs_tol=1e-3):
        raise ValueError(f"fps must be integral, got {fps}")
    values: dict[str, Any] = {
        "root_trans_offset": np.asarray(state["root_pos"][start:end], dtype=np.float32),
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_xyzw.astype(np.float32),
        "smpl_joints": np.zeros((frame_count, 24, 3), dtype=np.float32),
        "fps": type(reference_record["fps"])(round(fps)),
        "motion_name": f"{metadata['round_id']}_{metadata['rollout_id']}_continuous_{start}",
    }
    missing = sorted(set(BFM_REQUIRED) - set(values))
    if missing:
        raise ValueError(f"converted record misses BFM fields: {missing}")
    record = {key: values[key] for key in reference_keys if key in values}
    for key, value in values.items():
        if key not in record:
            record[key] = value
    for name, tail_shape in BFM_REQUIRED.items():
        value = np.asarray(record[name])
        if value.shape != (frame_count, *tail_shape) or value.dtype != np.float32:
            raise ValueError(f"{name} has invalid shape/dtype: {value.shape}/{value.dtype}")
        finite(name, value)
    record.update(
        {
            "action": np.asarray(state["action"][start:end], dtype=np.float32),
            "board_root_pos": np.asarray(state["board_root_pos"][start:end], dtype=np.float32),
            "board_root_quat": np.asarray(state["board_root_quat"][start:end], dtype=np.float32),
            "board_root_lin_vel": np.asarray(
                state["board_root_lin_vel"][start:end], dtype=np.float32
            ),
            "board_root_ang_vel": np.asarray(
                state["board_root_ang_vel"][start:end], dtype=np.float32
            ),
            "board_dof_pos": np.asarray(state["board_dof_pos"][start:end], dtype=np.float32),
            "board_dof_vel": np.asarray(state["board_dof_vel"][start:end], dtype=np.float32),
            "board_heading_delta": np.asarray(
                state["board_heading_delta"][start:end], dtype=np.float32
            ),
            "phase_id": phase_ids,
            "phase_value": np.asarray(state["phase_value"][start:end], dtype=np.float32),
            "motion_type": "continuous",
            "phase_sequence": compressed_phase_sequence(phase_ids),
            "clip_frames": frame_count,
            "source_round": str(metadata["round_id"]).zfill(3),
            "source_rollout": str(metadata["rollout_id"]).zfill(3),
            "source_episode": str(metadata["episode_id"]),
            "source_start_frame": int(start),
            "source_end_frame": int(end),
            "source_raw_npz": str(source_path.resolve()),
            "command_v": float(state["command_v"][0]),
            "command_h": float(state["command_h"][0]),
            "physics_seed": int(metadata["physics_randomization"]["seed"]),
            "dataset_split": str(metadata["dataset_split"]),
        }
    )
    for name in SKATE_REQUIRED:
        value = np.asarray(record[name])
        if value.shape[0] != frame_count:
            raise ValueError(f"paired field {name} is not aligned")
        finite(name, value)
    return record, {
        "joint_mapping": mapping,
        "fixed_zero_joints": sorted(set(metadata["fixed_bfm_joints"])),
    }


def validate_official_motionlib(
    args: argparse.Namespace,
    bfm_repo: Path,
    motion_file: Path,
    robot_xml: Path,
    motion_count: int,
    seq_length: int,
    fps: float,
) -> dict[str, Any]:
    args.failed_stage = "motionlib"
    if not math.isclose(fps, 50.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Official BFM validation requires 50 Hz motions, got {fps:g} Hz")
    print("\n[Official BFM Validation]")
    print("MotionLibRobot: RUNNING")
    sys.path.insert(0, str(bfm_repo.resolve()))
    import torch
    from easydict import EasyDict
    from humanoidverse.agents.envs.humanoidverse_isaac import (
        load_expert_trajectories_from_motion_lib,
    )
    from humanoidverse.utils.motion_lib.motion_lib_base import FixHeightMode
    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

    cfg = EasyDict(
        {
            "motion_file": str(motion_file.resolve()),
            "step_dt": 1.0 / fps,
            "fix_height": FixHeightMode.no_fix,
            "asset": EasyDict(
                {"assetRoot": str(robot_xml.resolve().parent), "assetFileName": robot_xml.name}
            ),
            "extend_config": [
                EasyDict(
                    {
                        "joint_name": "head_link",
                        "parent_name": "torso_link",
                        "pos": [0.0, 0.0, 0.35],
                        "rot": [1.0, 0.0, 0.0, 0.0],
                    }
                )
            ],
            "humanoid_type": "g1_29dof",
        }
    )
    motion_lib = MotionLibRobot(cfg, num_envs=motion_count, device="cpu")
    motion_lib.load_motions_for_training()
    ids = torch.arange(motion_count, dtype=torch.long)
    states = motion_lib.get_motion_state(ids, torch.zeros(motion_count))
    required = {"rg_pos_t", "rg_rot_t", "body_vel_t", "body_ang_vel_t", "dof_pos", "dof_vel"}
    if not required <= set(states):
        raise ValueError(f"MotionLib missing fields: {sorted(required - set(states))}")
    if not all(torch.isfinite(states[name]).all() for name in required):
        raise ValueError("MotionLib returned NaN/Inf")
    print("MotionLibRobot: PASS")
    print(f"Motions loaded: {motion_lib.num_motions()}")
    print(f"FPS: {fps:g} Hz")
    print(f"DoF: {states['dof_pos'].shape[-1]}")
    print("Finite check: PASS")
    env = SimpleNamespace(
        _motion_lib=motion_lib,
        dt=1.0 / fps,
        device="cpu",
        default_dof_pos=torch.zeros((1, 29)),
        gravity_vec=torch.tensor([[0.0, 0.0, -1.0]]),
        config=SimpleNamespace(
            obs=SimpleNamespace(
                obs_auxiliary={"history_actor": {}}, obs_dims={}, root_height_obs=True
            )
        ),
    )
    print("Official expert loader: RUNNING")
    print(f"Seq length: {seq_length}")
    args.failed_stage = "expert_loader"
    buffer = load_expert_trajectories_from_motion_lib(
        env, SimpleNamespace(model=SimpleNamespace(seq_length=seq_length)), device="cpu"
    )
    sample = buffer.sample(batch_size=seq_length * 2, seq_length=seq_length)
    current_shape = list(sample["observation"]["state"].shape)
    next_shape = list(sample["next"]["observation"]["state"].shape)
    if current_shape != next_shape:
        raise ValueError("expert sequence current/next shapes differ")
    print("Official expert loader: PASS")
    print("Seq8: PASS")
    print(f"Sample batch shape: {current_shape[:2]}")
    print(f"Current observation shape: {current_shape}")
    print(f"Next observation shape: {next_shape}")
    return {
        "status": "PASS",
        "motion_count": motion_lib.num_motions(),
        "total_duration": float(motion_lib.get_total_length()),
        "state_shapes": {name: list(states[name].shape) for name in sorted(required)},
        "expert_loader": "PASS",
        "sequence_sampling": "PASS",
        "sample_batch_shape": current_shape[:2],
        "current_observation_shape": current_shape,
        "next_observation_shape": next_shape,
    }


def raw_provenance_check(
    record: Mapping[str, Any],
    raw: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    source = Path(record["source_raw_npz"])
    if not source.is_file():
        raise FileNotFoundError(f"QC source missing: {source}")
    if raw is None:
        with np.load(source, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
    start, end = int(record["source_start_frame"]), int(record["source_end_frame"])
    if record.get("motion_type") == "continuous":
        clip_frames = int(record["clip_frames"])
        if end - start != clip_frames:
            raise ValueError("continuous provenance source range does not match clip_frames")
        phase_ids = np.asarray(record["phase_id"], dtype=np.int16)
        if compressed_phase_sequence(phase_ids) != list(record["phase_sequence"]):
            raise ValueError("continuous provenance phase_sequence mismatch")
        if np.any(np.asarray(raw["fall"][start:end], dtype=bool)):
            raise ValueError("continuous provenance crosses fall")
        if np.any(np.asarray(raw["reset"][start:end], dtype=bool)):
            raise ValueError("continuous provenance crosses reset")
    if metadata is None:
        metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if not np.array_equal(raw["frame_idx"][start:end], np.arange(start, end)):
        raise ValueError("QC provenance mismatch for source frame")
    comparisons = {
        "root_trans_offset": ("root_pos", record["root_trans_offset"]),
        "action": ("action", record["action"]),
        "board_root_pos": ("board_root_pos", record["board_root_pos"]),
        "board_root_quat": ("board_root_quat", record["board_root_quat"]),
        "board_root_lin_vel": ("board_root_lin_vel", record["board_root_lin_vel"]),
        "board_root_ang_vel": ("board_root_ang_vel", record["board_root_ang_vel"]),
        "board_dof_pos": ("board_dof_pos", record["board_dof_pos"]),
        "board_dof_vel": ("board_dof_vel", record["board_dof_vel"]),
        "phase_id": ("phase_id", record["phase_id"]),
        "phase_value": ("phase_value", record["phase_value"]),
        "board_heading_delta": ("board_heading_delta", record["board_heading_delta"]),
    }
    for record_name, (raw_name, expected) in comparisons.items():
        actual = np.asarray(expected)
        raw_value = np.asarray(raw[raw_name][start:end])
        if not np.allclose(actual, raw_value, rtol=1e-5, atol=1e-6):
            raise ValueError(f"QC provenance mismatch for {record_name}")
    source_names = [str(name).split("/")[-1] for name in metadata["joint_order"]]
    source_indices = {name: index for index, name in enumerate(source_names)}
    raw_dof = np.asarray(raw["dof_pos"][start:end])
    expected_dof = np.zeros((end - start, len(BFM_JOINT_ORDER)), dtype=np.float32)
    for index, name in enumerate(BFM_JOINT_ORDER):
        if name in source_indices:
            expected_dof[:, index] = raw_dof[:, source_indices[name]]
    if not np.allclose(record["dof"], expected_dof, rtol=1e-5, atol=1e-6):
        raise ValueError("QC provenance mismatch for name-mapped BFM dof")
    raw_quat = np.asarray(raw["root_quat"][start:end], dtype=np.float64)
    raw_quat /= np.linalg.norm(raw_quat, axis=1, keepdims=True)
    expected_root_rot = raw_quat[:, [1, 2, 3, 0]]
    if not np.allclose(record["root_rot"], expected_root_rot, rtol=1e-5, atol=1e-6):
        raise ValueError("QC provenance mismatch for root_rot")
    return raw


def audit_provenance(records: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    grouped: dict[Path, list[Mapping[str, Any]]] = {}
    for record in records.values():
        grouped.setdefault(Path(record["source_raw_npz"]), []).append(record)
    print("\n[Provenance Audit]")
    for source, source_records in tqdm(
        grouped.items(), desc="Raw provenance", unit="rollout", dynamic_ncols=True
    ):
        with np.load(source, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        for record in source_records:
            raw_provenance_check(record, raw, metadata)
    result = {
        "robot_state": "PASS",
        "board_state": "PASS",
        "action": "PASS",
        "phase": "PASS",
        "source_frame": "PASS",
    }
    for name, status in result.items():
        print(f"{name.replace('_', ' ')}: {status}")
    return result


def text_frame(width: int, height: int, lines: Sequence[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), (18, 27, 38))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, line in enumerate(lines):
        draw.text((36, 40 + index * 22), line, fill=(235, 240, 245), font=font)
    return np.asarray(image)


def overlay(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    height = 8 + 14 * len(lines)
    draw.rectangle((0, 0, 560, height), fill=(0, 0, 0, 175))
    for index, line in enumerate(lines):
        draw.text((6, 4 + index * 14), line, fill="white", font=font)
    return np.asarray(image)


def generate_qc(motion_file: Path, qc_root: Path, husky_xml: Path, seed: int) -> dict[str, Any]:
    import imageio.v2 as imageio

    records = joblib.load(motion_file)
    if not isinstance(records, dict) or not records:
        raise ValueError("QC requires a non-empty continuous motion dictionary")
    fps_values = {float(record["fps"]) for record in records.values()}
    if len(fps_values) != 1:
        raise ValueError(f"QC requires one dataset FPS, got {sorted(fps_values)}")
    fps = fps_values.pop()
    model = mujoco.MjModel.from_xml_path(str(husky_xml.resolve()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 4.0, 135.0, -18.0
    qc_root.mkdir(parents=True, exist_ok=True)
    video_root = qc_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    candidates = sorted(records)
    indexes = rng.choice(len(candidates), min(10, len(candidates)), replace=False)
    selected = [candidates[int(index)] for index in indexes]
    path = video_root / "continuous_10samples.mp4"
    result: dict[str, Any] = {
        "dataset_stage": "M2.5c-C",
        "dataset_type": "continuous_fixed_window",
        "qc_type": "posthoc_random_dataset_audit",
        "qc_seed": seed,
        "sampling": "uniform_without_replacement",
        "source_pkl": str(motion_file.resolve()),
        "source_manifest": str((motion_file.parent / "manifest.json").resolve()),
        "available_motion_count": len(records),
        "requested_samples": 10,
        "rendered_samples": len(selected),
        "video_path": str(path.resolve()),
        "status": "PENDING",
        "samples": [],
    }
    print("\n[Post-hoc QC]")
    print(f"Seed: {seed}")
    print("Sampling: uniform without replacement")
    print("Population: final accepted continuous dataset")
    print(f"Available: {len(candidates)}")
    print(f"Selected: {len(selected)}")
    print("Rendering...")
    try:
        writer = imageio.get_writer(path, fps=fps, macro_block_size=1, codec="libx264")
        try:
            for sample_index, key in enumerate(
                tqdm(selected, desc="QC continuous", unit="sample", dynamic_ncols=True),
                start=1,
            ):
                record = records[key]
                raw = raw_provenance_check(record)
                start, end = int(record["source_start_frame"]), int(record["source_end_frame"])
                duration = (end - start) / float(record["fps"])
                sequence = " -> ".join(record["phase_sequence"])
                title = [
                    "Stage: M2.5c-C",
                    "Type: Continuous 10s clip",
                    f"Sample: {sample_index} / {len(selected)}",
                    f"Motion Key: {key}",
                    f"Source Round: {record['source_round']}",
                    f"Source Rollout: {record['source_rollout']}",
                    f"Source Frames: {start} : {end}",
                    f"Duration: {duration:.1f}s",
                    f"FPS: {record['fps']}",
                    f"Command: v={record['command_v']:.2f}, h={record['command_h']:.2f}",
                    f"Physics Seed: {record['physics_seed']}",
                    "Phase Sequence:",
                    *textwrap.wrap(sequence, width=100),
                ]
                for _ in range(max(1, round(0.4 * fps))):
                    writer.append_data(text_frame(1280, 720, title))
                for local, raw_index in enumerate(range(start, end)):
                    data.qpos[:] = raw["qpos"][raw_index]
                    data.qvel[:] = raw["qvel"][raw_index]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = (
                        raw["root_pos"][raw_index] + raw["board_root_pos"][raw_index]
                    ) / 2.0
                    renderer.update_scene(data, camera=camera)
                    phase_id = int(record["phase_id"][local])
                    writer.append_data(
                        overlay(
                            renderer.render(),
                            [
                                f"M2.5c-C | Sample {sample_index}/{len(selected)}",
                                f"local frame={local} | local time={local / fps:.2f}s",
                                f"source raw frame={raw_index}",
                                (
                                    f"v={record['command_v']:.2f} | "
                                    f"h={record['command_h']:.2f} | "
                                    f"physics={record['physics_seed']}"
                                ),
                                (
                                    f"phase={PHASE_LABELS[phase_id]} | id={phase_id} | "
                                    f"value={record['phase_value'][local]:.3f}"
                                ),
                                (
                                    f"board_heading_delta="
                                    f"{record['board_heading_delta'][local]:+.3f} rad"
                                ),
                            ],
                        )
                    )
                result["samples"].append(
                    {
                        "sample_index": sample_index,
                        "motion_key": key,
                        "source_round": record["source_round"],
                        "source_rollout": record["source_rollout"],
                        "source_start_frame": start,
                        "source_end_frame": end,
                        "clip_frames": end - start,
                        "duration_seconds": duration,
                        "fps": int(record["fps"]),
                        "command_v": record["command_v"],
                        "command_h": record["command_h"],
                        "physics_seed": record["physics_seed"],
                        "phase_sequence": record["phase_sequence"],
                        "phase_id_start": int(record["phase_id"][0]),
                        "phase_id_end": int(record["phase_id"][-1]),
                        "phase_value_start": float(record["phase_value"][0]),
                        "phase_value_end": float(record["phase_value"][-1]),
                        "board_heading_delta_start": float(record["board_heading_delta"][0]),
                        "board_heading_delta_end": float(record["board_heading_delta"][-1]),
                    }
                )
        finally:
            writer.close()
    finally:
        renderer.close()
    result["status"] = "PASS"
    result["qc_to_raw_validation"] = "PASS"
    write_json(qc_root / "qc_manifest.json", result)
    print("continuous: PASS")
    print(f"video: {path.resolve()}")
    return result


def plan_stats(raw_root: Path) -> dict[str, Any]:
    plan_path, summary_path = (
        raw_root / "collection_plan.json",
        raw_root / "collection_summary.json",
    )
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    failed = int(summary.get("failed_rollout_count", 0))
    return {
        "plan": plan,
        "summary": summary,
        "planned_rollouts": int(plan.get("baseline_rollouts", len(plan.get("jobs", [])))),
        "completed_rollouts": int(summary.get("completed_rollouts", 0)),
        "failed_rollouts": failed,
        "replacement_rollouts": int(summary.get("replacement_rollouts", 0)),
        "source_policy": plan.get("policy_checkpoint"),
    }


def print_build_plan(
    args: argparse.Namespace,
    raw_root: Path,
    output: Path,
    manifest_path: Path,
    rollouts: Sequence[Path],
    plan: Mapping[str, Any],
) -> None:
    summary = plan["summary"]
    print("[Continuous Dataset Build Plan]")
    print("Stage: M2.5c-C")
    print(f"Raw root: {raw_root}")
    print(f"Output pkl: {output}")
    print(f"Manifest: {manifest_path}")
    print(f"QC root: {args.qc_root.resolve() if args.qc_root else 'disabled'}")
    print(f"Clip frames: {args.clip_frames}")
    print(f"Clip duration: {args.clip_frames / 50.0:.1f}s")
    print("Overlap: 0")
    print(f"Seq length: {args.seq_length}")
    print(f"QC seed: {args.qc_seed}")
    print(f"Source raw rollouts: {len(rollouts)}")
    print(f"Source raw frames: {summary.get('raw_frames', 'unknown')}")


def range_stats(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "min": min(values) if values else 0,
        "median": float(np.median(values)) if values else 0.0,
        "max": max(values) if values else 0,
    }


def transition_stats(run_counts: Sequence[int]) -> dict[str, float | int]:
    return {
        "clips_with_phase_transition": sum(count > 1 for count in run_counts),
        "clips_without_phase_transition": sum(count == 1 for count in run_counts),
        "mean_phase_runs_per_clip": float(np.mean(run_counts)) if run_counts else 0.0,
        "min_phase_runs_per_clip": min(run_counts) if run_counts else 0,
        "median_phase_runs_per_clip": float(np.median(run_counts)) if run_counts else 0.0,
        "max_phase_runs_per_clip": max(run_counts) if run_counts else 0,
    }


def print_continuous_summary(
    motion_count: int,
    expert_frames: int,
    fps: float,
    clip_frames: int,
    valid_rollouts: int,
    clips_per_rollout: Sequence[int],
    phase_frame_distribution: Mapping[str, int],
    phase_transitions: Mapping[str, float | int],
    discard_frames: Mapping[str, int],
    discard_motions: Mapping[str, int],
    rejected_rollouts: int,
    rejected_motions: int,
) -> None:
    print("\n[Continuous Conversion Summary]")
    print(f"continuous motions: {motion_count}")
    print(f"expert frames: {expert_frames}")
    print(f"expert duration: {expert_frames / fps / 60.0:.3f} min")
    print(f"clip frames: {clip_frames}")
    print(f"clip duration: {clip_frames / fps:.1f}s")
    print(f"valid rollouts: {valid_rollouts}")
    print(f"clips per rollout: {range_stats(clips_per_rollout)}")
    print(f"discarded tail frames: {discard_frames['continuous_tail']}")
    print(f"discarded tail minutes: {discard_frames['continuous_tail'] / fps / 60.0:.3f}")
    print(f"discarded failure-margin frames: {discard_frames['failure_margin']}")
    print(f"fall frames: {discard_frames['fall']}")
    print(f"reset frames: {discard_frames['reset']}")
    print(
        "clips with phase transition: "
        f"{phase_transitions['clips_with_phase_transition']}"
    )
    print(f"mean phase runs / clip: {phase_transitions['mean_phase_runs_per_clip']:.3f}")
    print("\nPhase frame distribution:")
    for phase in PHASES:
        print(f"{phase}: {phase_frame_distribution[phase]}")
    print(f"discard_frame_counts: {dict(discard_frames)}")
    print(f"discard_motion_counts: {dict(discard_motions)}")
    print(f"rejected_rollout_count: {rejected_rollouts}")
    print(f"rejected_motion_count: {rejected_motions}")


def print_final_summary(
    manifest: Mapping[str, Any],
    output: Path,
    manifest_path: Path,
    qc: Mapping[str, Any] | None,
) -> None:
    validation = manifest.get("official_motionlib_validation", {})
    provenance = manifest.get("raw_provenance", {})
    print("\n==================================================")
    print("M2.5c-C FINAL DATASET SUMMARY")
    print("==================================================")
    print("Canonical raw: PASS")
    print(f"Raw rollouts processed: {manifest['completed_rollouts']}")
    print(f"Rejected rollouts: {manifest['rejected_rollout_count']}")
    print(f"Accepted continuous motions: {manifest['total_motion_count']}")
    print(f"Clip frames: {manifest['clip_frames']}")
    print(f"Clip seconds: {manifest['clip_seconds']:.1f}")
    print(f"Expert frames: {manifest['expert_frames']}")
    print(f"Expert duration: {manifest['expert_minutes']:.3f} min")
    print("\nPhase frame distribution:")
    for phase in PHASES:
        print(f"{phase}: {manifest['phase_frame_distribution'][phase]}")
    print(
        "\nClips with phase transitions: "
        f"{manifest['phase_transition_statistics']['clips_with_phase_transition']}"
    )
    print(f"\nDiscard frames: {manifest['discard_frame_counts']}")
    print(f"Discard motions: {manifest['discard_motion_counts']}")
    print(f"\nMotionLibRobot: {validation.get('status', 'NOT RUN')}")
    print(f"Official expert loader: {validation.get('expert_loader', 'NOT RUN')}")
    print(f"Seq8: {validation.get('sequence_sampling', 'NOT RUN')}")
    print(f"Raw provenance: {'PASS' if provenance else 'NOT RUN'}")
    print(f"Post-hoc QC generation: {'PASS' if qc is not None else 'NOT RUN'}")
    print(f"\nFinal pkl: {output}")
    print(f"Manifest: {manifest_path}")
    print(f"QC manifest: {qc.get('manifest_path') if qc else 'NOT RUN'}")
    print(f"QC video: {qc['video_path'] if qc else 'NOT RUN'}")
    print("\nHuman Motion Quality Review: PENDING")
    print("==================================================")
    print("M2.5c-C DATASET BUILD COMPLETE")
    print("Training launched: false")


def aggregate(args: argparse.Namespace) -> int:
    args.failed_stage = "aggregation"
    raw_root, _ = raw_root_from_arg(args.dataset_root)
    output = args.output.resolve()
    manifest_path = (args.manifest or output.parent / "manifest.json").resolve()
    rollouts = rollout_paths(raw_root, args.dataset_split)
    plan = plan_stats(raw_root)
    print_build_plan(args, raw_root, output, manifest_path, rollouts, plan)
    reference_keys, reference_record = load_reference(args.bfm_reference)
    target_order, target_axes = bfm_joint_contract(args.robot_xml)
    records: dict[str, dict[str, Any]] = {}
    command_stats: dict[str, dict[str, Any]] = {}
    discard_frames = Counter(
        {"fall": 0, "failure_margin": 0, "reset": 0, "continuous_tail": 0}
    )
    discard_motions = Counter(
        {
            "nan_inf": 0,
            "alignment": 0,
            "invalid_phase": 0,
            "conversion_error": 0,
        }
    )
    rejected: list[dict[str, Any]] = []
    rejected_rollouts: set[Path] = set()
    rejected_motion_count = 0
    raw_frames = raw_seconds = 0.0
    accepted_rollouts: set[Path] = set()
    physics_seeds: set[int] = set()
    fps_values: set[float] = set()
    failure_margins: set[float] = set()
    clips_per_rollout: list[int] = []
    phase_frame_counts: Counter[str] = Counter({phase: 0 for phase in PHASES})
    phase_run_counts: list[int] = []
    rollout_provenance: list[dict[str, Any]] = []

    args.failed_stage = "raw_validation"
    validated_rollouts: list[Path] = []
    validation_frames = 0
    raw_progress = tqdm(rollouts, desc="Raw validation", unit="rollout", dynamic_ncols=True)
    for rollout in raw_progress:
        try:
            raw_path, metadata, state = load_raw(rollout)
            frame_count, _ = validate_raw(raw_path, metadata, state)
        except Exception as error:
            category = discard_category(error)
            rejected_rollouts.add(rollout.resolve())
            rejected.append(
                {
                    "level": "rollout",
                    "rollout": str(rollout),
                    "reason": str(error),
                    "category": category,
                }
            )
            tqdm.write(f"REJECT {rollout.relative_to(raw_root)}: {error}")
            continue
        validated_rollouts.append(rollout)
        validation_frames += frame_count
        raw_progress.set_postfix(
            accepted=len(validated_rollouts),
            rejected=len(rejected_rollouts),
            frames=validation_frames,
            refresh=False,
        )
    raw_progress.close()

    args.failed_stage = "conversion"
    conversion_progress = tqdm(
        validated_rollouts, desc="Continuous conversion", unit="rollout", dynamic_ncols=True
    )
    for rollout in conversion_progress:
        raw_path, metadata, state = load_raw(rollout)
        frame_count, source = validate_raw(raw_path, metadata, state)
        raw_frames += frame_count
        raw_seconds += frame_count / float(metadata["fps"])
        fps_value = float(metadata["fps"])
        if not math.isclose(fps_value, round(fps_value), rel_tol=0.0, abs_tol=1e-3):
            raise RuntimeError(f"{raw_path} has non-integral FPS: {fps_value:g}")
        fps_values.add(float(round(fps_value)))
        physics_seeds.add(source["physics_seed"])
        failure_margin_s = float(metadata.get("failure_margin_s", 0.15))
        failure_margins.add(failure_margin_s)
        margin_frames = round(failure_margin_s * fps_value)
        intervals, interval_discards = valid_intervals(
            state,
            frame_count,
            margin_frames,
        )
        discard_frames.update(interval_discards)
        windows, tail_frames = fixed_windows(intervals, args.clip_frames)
        discard_frames["continuous_tail"] += tail_frames
        clips_per_rollout.append(len(windows))
        rollout_provenance.append(
            {
                "source_round": str(metadata["round_id"]).zfill(3),
                "source_rollout": str(metadata["rollout_id"]).zfill(3),
                "source_episode": str(metadata["episode_id"]),
                "source_raw_npz": str(raw_path.resolve()),
                "command_v": source["command_v"],
                "command_h": source["command_h"],
                "physics_seed": source["physics_seed"],
                "actual_physics_realization": metadata["physics_randomization"],
            }
        )
        for clip_index, (clip_start, clip_end) in enumerate(windows):
            try:
                record, _ = convert_record(
                    metadata,
                    state,
                    reference_keys,
                    reference_record,
                    target_order,
                    target_axes,
                    clip_start,
                    clip_end,
                    raw_path,
                    args.seq_length,
                    args.clip_frames,
                )
            except Exception as error:
                category = motion_discard_category(error)
                discard_motions[category] += 1
                rejected_motion_count += 1
                rejected.append(
                    {
                        "level": "motion",
                        "rollout": str(rollout),
                        "motion_type": "continuous",
                        "start_frame": clip_start,
                        "end_frame": clip_end,
                        "reason": str(error),
                        "category": category,
                    }
                )
                continue
            key = (
                "skate/continuous/"
                f"r{str(metadata['round_id']).zfill(3)}_"
                f"rollout{str(metadata['rollout_id']).zfill(3)}_"
                f"clip{clip_index:03d}"
            )
            if key in records:
                raise RuntimeError(f"duplicate motion key: {key}")
            records[key] = record
            accepted_rollouts.add(rollout)
            phase_ids = np.asarray(record["phase_id"], dtype=np.int16)
            for phase_id, count in zip(*np.unique(phase_ids, return_counts=True), strict=True):
                phase_frame_counts[PHASE_LABELS[int(phase_id)]] += int(count)
            phase_run_counts.append(len(record["phase_sequence"]))
            command_key = f"{source['command_v']:.6g},{source['command_h']:.6g}"
            command = command_stats.setdefault(
                command_key,
                {
                    "command_v": source["command_v"],
                    "command_h": source["command_h"],
                    "rollout_ids": set(),
                    "accepted_motion_count": 0,
                    "accepted_expert_seconds": 0.0,
                },
            )
            command["rollout_ids"].add(str(metadata["episode_id"]))
            command["accepted_motion_count"] += 1
            command["accepted_expert_seconds"] += args.clip_frames / fps_value
        conversion_progress.set_postfix(
            clips=len(records),
            expert_frames=len(records) * args.clip_frames,
            tail_discard=discard_frames["continuous_tail"],
            rejected_motion=rejected_motion_count,
            refresh=False,
        )
    conversion_progress.close()

    if not records:
        raise RuntimeError("no accepted continuous motions were produced")
    if len(fps_values) != 1:
        raise RuntimeError(f"accepted raw rollouts have mixed FPS values: {sorted(fps_values)}")
    if len(failure_margins) != 1:
        raise RuntimeError(f"accepted raw rollouts have mixed failure margins: {failure_margins}")
    dataset_fps = fps_values.pop()
    failure_margin_s = failure_margins.pop()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    expert_frames = len(records) * args.clip_frames
    expert_seconds = expert_frames / dataset_fps
    phase_distribution = {phase: phase_frame_counts[phase] for phase in PHASES}
    phase_transitions = transition_stats(phase_run_counts)
    print_continuous_summary(
        len(records),
        expert_frames,
        dataset_fps,
        args.clip_frames,
        len(validated_rollouts),
        clips_per_rollout,
        phase_distribution,
        phase_transitions,
        discard_frames,
        discard_motions,
        len(rejected_rollouts),
        rejected_motion_count,
    )
    print("\n[Writing Dataset]")
    print(f"motions: {len(records)}")
    print(f"expert frames: {expert_frames}")
    print(f"expert duration: {expert_seconds / 60:.3f} min")
    print(f"output: {output}")
    args.failed_stage = "aggregation"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    joblib.dump(records, temporary)
    temporary.replace(output)
    print("Dataset pickle: PASS")
    print(f"size: {output.stat().st_size / (1024 * 1024):.2f} MiB")
    for command in command_stats.values():
        command["rollout_count"] = len(command.pop("rollout_ids"))
    manifest = {
        "dataset_stage": "M2.5c-C",
        "dataset_type": "continuous_fixed_window",
        "dataset_split": args.dataset_split,
        "source_raw_root": str(args.dataset_root),
        "source_raw_rollouts": len(rollouts),
        "source_rollout_count": len(accepted_rollouts),
        "source_dataset_stage": "M2.5c-P canonical raw collection",
        "source_policy": plan["source_policy"],
        "fps": dataset_fps,
        "clip_frames": args.clip_frames,
        "clip_seconds": args.clip_frames / dataset_fps,
        "stride_frames": args.clip_frames,
        "overlap_frames": 0,
        "allow_phase_crossing": True,
        "preserve_phase_annotation": True,
        "cross_fall": False,
        "cross_reset": False,
        "failure_margin_s": failure_margin_s,
        "phase_id_to_label": PHASE_LABELS,
        "planned_rollouts": plan["planned_rollouts"],
        "completed_rollouts": plan["completed_rollouts"],
        "accepted_rollouts": len(accepted_rollouts),
        "failed_rollouts": plan["failed_rollouts"],
        "rejected_rollout_count": len(rejected_rollouts),
        "rejected_motion_count": rejected_motion_count,
        "replacement_rollouts": plan["replacement_rollouts"],
        "raw_frames": int(raw_frames),
        "raw_seconds": raw_seconds,
        "raw_minutes": raw_seconds / 60,
        "frame_count": expert_frames,
        "duration_seconds": expert_seconds,
        "duration_minutes": expert_seconds / 60,
        "motion_count": len(records),
        "expert_frames": expert_frames,
        "expert_seconds": expert_seconds,
        "expert_minutes": expert_seconds / 60,
        "total_motion_count": len(records),
        "unique_physics_seed_count": len(physics_seeds),
        "source_rounds": sorted({str(record["source_round"]) for record in records.values()}),
        "source_identity_count": len(
            {
                (
                    str(record["source_round"]),
                    str(record["source_rollout"]),
                    str(record["source_episode"]),
                )
                for record in records.values()
            }
        ),
        "physics_seed_sha256": hashlib.sha256(
            ",".join(map(str, sorted(physics_seeds))).encode()
        ).hexdigest(),
        "continuous_statistics": {
            "continuous_motion_count": len(records),
            "valid_rollouts": len(validated_rollouts),
            "clips_per_rollout": range_stats(clips_per_rollout),
            "discarded_tail_frames": discard_frames["continuous_tail"],
            "discarded_tail_minutes": discard_frames["continuous_tail"]
            / dataset_fps
            / 60,
        },
        "phase_frame_distribution": phase_distribution,
        "phase_transition_statistics": phase_transitions,
        "command_statistics": sorted(
            command_stats.values(), key=lambda item: (item["command_v"], item["command_h"])
        ),
        "rollout_provenance": rollout_provenance,
        "discard_frame_counts": dict(discard_frames),
        "discard_motion_counts": dict(discard_motions),
        "rejection_details": rejected,
        "motion_library": str(output),
        "motion_library_sha256": hashlib.file_digest(
            output.open("rb"), "sha256"
        ).hexdigest(),
        "structural_validation": "PENDING",
        "bfm_min_motion_frames": args.seq_length + 1,
    }
    write_json(manifest_path, manifest)
    print("Manifest: PASS")
    print(f"path: {manifest_path}")

    if args.validate_motionlib:
        args.failed_stage = "motionlib"
        manifest["official_motionlib_validation"] = validate_official_motionlib(
            args,
            args.bfm_repo,
            output,
            args.robot_xml,
            len(records),
            args.seq_length,
            dataset_fps,
        )
        manifest["structural_validation"] = "PASS"
        write_json(manifest_path, manifest)
        print("Structural Validation: PASS")
    args.failed_stage = "provenance"
    manifest["raw_provenance"] = audit_provenance(records)
    write_json(manifest_path, manifest)
    qc = None
    if args.qc_root:
        args.failed_stage = "qc"
        qc = generate_qc(
            output,
            args.qc_root.resolve(),
            args.husky_xml.resolve(),
            args.qc_seed,
        )
        manifest["qc"] = {
            "status": "PASS",
            "seed": args.qc_seed,
            "manifest": str(args.qc_root.resolve() / "qc_manifest.json"),
            "rendered_samples": qc["rendered_samples"],
        }
        write_json(manifest_path, manifest)
        print(f"Visual QC Generation: PASS ({qc['rendered_samples']} samples)")
    if qc is not None:
        qc["manifest_path"] = str(args.qc_root.resolve() / "qc_manifest.json")
    print_final_summary(manifest, output, manifest_path, qc)
    args.failed_stage = "complete"
    return 0


def main() -> int:
    args = parser().parse_args()
    if not args.aggregate_continuous:
        raise SystemExit("only --aggregate-continuous dataset mode is supported")
    for name in ("dataset_root", "bfm_repo", "bfm_reference", "robot_xml"):
        if not getattr(args, name).exists():
            raise SystemExit(f"--{name.replace('_', '-')} does not exist")
    if args.seq_length <= 0:
        raise SystemExit("--seq-length must be positive")
    if args.clip_frames < args.seq_length + 1:
        raise SystemExit("--clip-frames must be at least --seq-length + 1")
    if args.qc_root is not None:
        if args.husky_xml is None:
            raise SystemExit("--qc-root requires an explicit --husky-xml")
        if not args.husky_xml.is_file():
            raise SystemExit(f"--husky-xml does not exist: {args.husky_xml}")
    if args.manifest is None:
        args.manifest = args.output.parent / "manifest.json"
    try:
        return aggregate(args)
    except Exception as error:
        stage = getattr(args, "failed_stage", "aggregation")
        labels = {
            "motionlib": "MotionLibRobot",
            "expert_loader": "Official expert loader",
            "provenance": "Raw provenance",
            "qc": "Post-hoc QC",
        }
        if stage in labels:
            print(f"\n{labels[stage]}: FAIL", file=sys.stderr)
        print(f"\nFAILED STAGE: {stage}", file=sys.stderr)
        print(f"Error type: {type(error).__name__}", file=sys.stderr)
        print(f"Error message: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
