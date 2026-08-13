#!/usr/bin/env python3
"""Build the phase-structured BFM MotionLib from canonical HUSKY raw data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

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
BFM_MIN_FRAMES = 9
DISCARD_KEYS = (
    "fall",
    "too_short_for_bfm",
    "reset",
    "nan_inf",
    "alignment",
    "invalid_phase",
    "other",
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
    p.add_argument("--aggregate-phase", action="store_true")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--bfm-repo", type=Path, required=True)
    p.add_argument("--bfm-reference", type=Path, required=True)
    p.add_argument("--robot-xml", type=Path, required=True)
    p.add_argument("--husky-xml", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--qc-root", type=Path)
    p.add_argument("--qc-seed", type=int, default=20260813)
    p.add_argument("--seq-length", type=int, default=8)
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


def rollout_paths(raw_root: Path) -> list[Path]:
    paths = sorted(
        path for path in raw_root.glob("round_*/rollout_*") if (path / "raw_rollout").is_dir()
    )
    if not paths:
        raise FileNotFoundError(f"no raw rollouts found below {raw_root}")
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
    phase_name: str,
    start: int,
    end: int,
    source_path: Path,
    seq_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_count = end - start
    if frame_count < seq_length + 1:
        raise ValueError(f"segment is shorter than BFM minimum {seq_length + 1}")
    if not np.all(state["phase_id"][start:end] == PHASE_IDS[phase_name]):
        raise ValueError("phase segment is not phase-pure")
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
        "motion_name": f"{metadata['round_id']}_{metadata['rollout_id']}_{phase_name}_{start}",
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
            "phase_id": np.asarray(state["phase_id"][start:end], dtype=np.int16),
            "phase_value": np.asarray(state["phase_value"][start:end], dtype=np.float32),
            "phase_label": phase_name,
            "source_round": str(metadata["round_id"]).zfill(3),
            "source_rollout": str(metadata["rollout_id"]).zfill(3),
            "source_episode": str(metadata["episode_id"]),
            "source_start_frame": int(start),
            "source_end_frame": int(end),
            "source_raw_npz": str(source_path.resolve()),
            "command_v": float(state["command_v"][0]),
            "command_h": float(state["command_h"][0]),
            "physics_seed": int(metadata["physics_randomization"]["seed"]),
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
    bfm_repo: Path, motion_file: Path, robot_xml: Path, motion_count: int, seq_length: int
) -> dict[str, Any]:
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
            "step_dt": 1.0 / 50.0,
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
    env = SimpleNamespace(
        _motion_lib=motion_lib,
        dt=1.0 / 50.0,
        device="cpu",
        default_dof_pos=torch.zeros((1, 29)),
        gravity_vec=torch.tensor([[0.0, 0.0, -1.0]]),
        config=SimpleNamespace(
            obs=SimpleNamespace(
                obs_auxiliary={"history_actor": {}}, obs_dims={}, root_height_obs=True
            )
        ),
    )
    buffer = load_expert_trajectories_from_motion_lib(
        env, SimpleNamespace(model=SimpleNamespace(seq_length=seq_length)), device="cpu"
    )
    sample = buffer.sample(batch_size=seq_length * 2, seq_length=seq_length)
    if sample["observation"]["state"].shape != sample["next"]["observation"]["state"].shape:
        raise ValueError("expert sequence current/next shapes differ")
    return {
        "status": "PASS",
        "motion_count": motion_lib.num_motions(),
        "total_duration": float(motion_lib.get_total_length()),
        "state_shapes": {name: list(states[name].shape) for name in sorted(required)},
        "expert_loader": "PASS",
        "sequence_sampling": "PASS",
    }


def raw_provenance_check(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    source = Path(record["source_raw_npz"])
    if not source.is_file():
        raise FileNotFoundError(f"QC source missing: {source}")
    with np.load(source, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    start, end = int(record["source_start_frame"]), int(record["source_end_frame"])
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
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
    result: dict[str, Any] = {
        "dataset_stage": "M2.5c-P",
        "dataset_type": "phase_structured",
        "qc_type": "posthoc_random_dataset_audit",
        "qc_seed": seed,
        "source_pkl": str(motion_file.resolve()),
        "source_manifest": str((motion_file.parent / "manifest.json").resolve()),
        "total_dataset_motion_count": len(records),
        "phases": [],
    }
    try:
        for phase in PHASES:
            candidates = sorted(
                key for key, value in records.items() if value["phase_label"] == phase
            )
            indexes = (
                rng.choice(len(candidates), min(10, len(candidates)), replace=False)
                if candidates
                else []
            )
            selected = [candidates[int(index)] for index in indexes]
            path = video_root / f"{phase}_{len(selected)}samples.mp4"
            writer = imageio.get_writer(path, fps=50, macro_block_size=1, codec="libx264")
            samples = []
            try:
                if not selected:
                    writer.append_data(
                        text_frame(
                            1280,
                            720,
                            [
                                "M2.5c-P",
                                f"Phase Label: {phase}",
                                "No accepted motions available",
                            ],
                        )
                    )
                for sample_index, key in enumerate(selected, start=1):
                    record = records[key]
                    raw = raw_provenance_check(record)
                    start, end = int(record["source_start_frame"]), int(record["source_end_frame"])
                    duration = (end - start - 1) / float(record["fps"])
                    title = [
                        "Stage: M2.5c-P",
                        f"Phase Label: {phase}",
                        f"Phase ID: {PHASE_IDS[phase]}",
                        f"Sample: {sample_index} / {len(selected)}",
                        f"Motion Key: {key}",
                        f"Source: r{record['source_round']}/rollout{record['source_rollout']}",
                        f"Frames: {start} : {end}",
                        f"Duration: {duration:.2f}s | FPS: {record['fps']}",
                        f"Command: v={record['command_v']:.2f}, h={record['command_h']:.2f}",
                        f"Physics Seed: {record['physics_seed']}",
                    ]
                    for _ in range(20):
                        writer.append_data(text_frame(1280, 720, title))
                    for local, raw_index in enumerate(range(start, end)):
                        data.qpos[:] = raw["qpos"][raw_index]
                        data.qvel[:] = raw["qvel"][raw_index]
                        mujoco.mj_forward(model, data)
                        camera.lookat[:] = (
                            raw["root_pos"][raw_index] + raw["board_root_pos"][raw_index]
                        ) / 2.0
                        renderer.update_scene(data, camera=camera)
                        writer.append_data(
                            overlay(
                                renderer.render(),
                                [
                                    f"M2.5c-P | {phase} | Sample {sample_index}/{len(selected)}",
                                    (
                                        f"v={record['command_v']:.2f} | "
                                        f"h={record['command_h']:.2f} | "
                                        f"physics={record['physics_seed']}"
                                    ),
                                    (
                                        f"source=r{record['source_round']}/"
                                        f"rollout{record['source_rollout']} | "
                                        f"frame={raw_index}"
                                    ),
                                    (
                                        f"t={local / record['fps']:.2f}/"
                                        f"{duration:.2f}s | "
                                        f"phase_value={record['phase_value'][local]:.3f}"
                                    ),
                                    (
                                        f"board_heading_delta="
                                        f"{record['board_heading_delta'][local]:+.3f} rad"
                                    ),
                                ],
                            )
                        )
                    samples.append(
                        {
                            "sample_index": sample_index,
                            "motion_key": key,
                            "source_round": record["source_round"],
                            "source_rollout": record["source_rollout"],
                            "source_start_frame": start,
                            "source_end_frame": end,
                            "segment_frames": end - start,
                            "duration_seconds": duration,
                            "fps": int(record["fps"]),
                            "command_v": record["command_v"],
                            "command_h": record["command_h"],
                            "physics_seed": record["physics_seed"],
                            "phase_value_start": float(record["phase_value"][0]),
                            "phase_value_end": float(record["phase_value"][-1]),
                            "board_heading_delta_start": float(record["board_heading_delta"][0]),
                            "board_heading_delta_end": float(record["board_heading_delta"][-1]),
                        }
                    )
            finally:
                writer.close()
            result["phases"].append(
                {
                    "phase_label": phase,
                    "phase_id": PHASE_IDS[phase],
                    "available_motion_count": len(candidates),
                    "requested_samples": 10,
                    "rendered_samples": len(selected),
                    "video_path": str(path.resolve()),
                    "status": "PASS" if len(selected) >= 10 else "insufficient_phase_coverage",
                    "samples": samples,
                }
            )
    finally:
        renderer.close()
    result["total_rendered_samples"] = sum(item["rendered_samples"] for item in result["phases"])
    result["qc_to_raw_validation"] = "PASS"
    write_json(qc_root / "qc_manifest.json", result)
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


def aggregate(args: argparse.Namespace) -> int:
    raw_root, dataset_root = raw_root_from_arg(args.dataset_root)
    output = args.output.resolve()
    manifest_path = (args.manifest or output.parent / "manifest.json").resolve()
    reference_keys, reference_record = load_reference(args.bfm_reference)
    target_order, target_axes = bfm_joint_contract(args.robot_xml)
    records: dict[str, dict[str, Any]] = {}
    phase_stats = {
        phase: {"motion_count": 0, "frame_count": 0, "duration_seconds": 0.0, "lengths": []}
        for phase in PHASES
    }
    command_stats: dict[str, dict[str, Any]] = {}
    discard = Counter({key: 0 for key in DISCARD_KEYS})
    rejected: list[dict[str, Any]] = []
    raw_frames = raw_seconds = 0.0
    accepted_rollouts: set[Path] = set()
    physics_seeds: set[int] = set()

    for rollout in rollout_paths(raw_root):
        raw_path = rollout
        try:
            raw_path, metadata, state = load_raw(rollout)
            frame_count, source = validate_raw(raw_path, metadata, state)
        except Exception as error:
            category = discard_category(error)
            discard[category] += 1
            rejected.append(
                {
                    "rollout": str(rollout),
                    "reason": str(error),
                    "category": category,
                }
            )
            continue
        raw_frames += frame_count
        raw_seconds += frame_count / float(metadata["fps"])
        physics_seeds.add(source["physics_seed"])
        phase_ids = np.asarray(state["phase_id"], dtype=np.int16)
        fall_start = next(
            (
                start
                for phase_id, start, _ in phase_runs(phase_ids)
                if phase_id == PHASE_IDS["fall"]
            ),
            None,
        )
        margin = round(float(metadata.get("failure_margin_s", 0.15)) * float(metadata["fps"]))
        segment_index = 0
        for phase_id, start, end in phase_runs(phase_ids):
            if fall_start is not None and start >= fall_start:
                if phase_id != PHASE_IDS["fall"]:
                    discard["fall"] += end - start
                continue
            if phase_id == PHASE_IDS["fall"]:
                discard["fall"] += end - start
                continue
            if phase_id not in range(6):
                discard["invalid_phase"] += end - start
                continue
            name = next(label for label, value in PHASE_IDS.items() if value == phase_id)
            if fall_start is not None and end == fall_start:
                end = max(start, end - margin)
            reset_indices = np.flatnonzero(state["reset"][start:end])
            cuts: list[tuple[int, int]] = []
            cursor = start
            for reset_index in (start + reset_indices).tolist():
                if cursor < reset_index:
                    cuts.append((cursor, reset_index))
                cursor = reset_index + 1
                discard["reset"] += 1
            if cursor < end:
                cuts.append((cursor, end))
            if not reset_indices.size:
                cuts = [(start, end)]
            for segment_start, segment_end in cuts:
                occurrence = segment_index
                segment_index += 1
                if segment_end - segment_start < args.seq_length + 1:
                    discard["too_short_for_bfm"] += 1
                    continue
                try:
                    record, conversion = convert_record(
                        metadata,
                        state,
                        reference_keys,
                        reference_record,
                        target_order,
                        target_axes,
                        name,
                        segment_start,
                        segment_end,
                        raw_path,
                        args.seq_length,
                    )
                except Exception as error:
                    category = discard_category(error)
                    discard[category] += 1
                    rejected.append(
                        {
                            "rollout": str(rollout),
                            "phase": name,
                            "start_frame": segment_start,
                            "end_frame": segment_end,
                            "reason": str(error),
                            "category": category,
                        }
                    )
                    continue
                key = (
                    f"skate/{name}/"
                    f"r{str(metadata['round_id']).zfill(3)}_"
                    f"rollout{str(metadata['rollout_id']).zfill(3)}_"
                    f"seg{occurrence:03d}"
                )
                if key in records:
                    raise RuntimeError(f"duplicate motion key: {key}")
                records[key] = record
                accepted_rollouts.add(rollout)
                stats = phase_stats[name]
                length = segment_end - segment_start
                stats["motion_count"] += 1
                stats["frame_count"] += length
                stats["duration_seconds"] += (length - 1) / float(metadata["fps"])
                stats["lengths"].append(length)
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
                command["accepted_expert_seconds"] += (length - 1) / float(metadata["fps"])

    if not records:
        raise RuntimeError("no accepted phase motions were produced")
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    joblib.dump(records, temporary)
    temporary.replace(output)
    stats_out = {}
    for phase in PHASES:
        stats = phase_stats[phase]
        lengths = stats.pop("lengths")
        stats_out[phase] = {
            **stats,
            "min_segment_frames": min(lengths) if lengths else 0,
            "median_segment_frames": float(np.median(lengths)) if lengths else 0,
            "max_segment_frames": max(lengths) if lengths else 0,
        }
    for command in command_stats.values():
        command["rollout_count"] = len(command.pop("rollout_ids"))
    plan = plan_stats(raw_root)
    manifest = {
        "dataset_stage": "M2.5c-P",
        "dataset_type": "phase_structured",
        "source_policy": plan["source_policy"],
        "fps": 50,
        "planned_rollouts": plan["planned_rollouts"],
        "completed_rollouts": plan["completed_rollouts"],
        "accepted_rollouts": len(accepted_rollouts),
        "failed_rollouts": plan["failed_rollouts"],
        "rejected_rollouts": len(rejected),
        "replacement_rollouts": plan["replacement_rollouts"],
        "raw_frames": int(raw_frames),
        "raw_seconds": raw_seconds,
        "raw_minutes": raw_seconds / 60,
        "expert_frames": sum(item["frame_count"] for item in stats_out.values()),
        "expert_seconds": sum(item["duration_seconds"] for item in stats_out.values()),
        "expert_minutes": sum(item["duration_seconds"] for item in stats_out.values()) / 60,
        "total_motion_count": len(records),
        "unique_physics_seed_count": len(physics_seeds),
        "phase_statistics": stats_out,
        "command_statistics": sorted(
            command_stats.values(), key=lambda item: (item["command_v"], item["command_h"])
        ),
        "discard_statistics": dict(discard),
        "rejection_details": rejected,
        "motion_library": str(output),
        "structural_validation": "PENDING",
        "bfm_min_motion_frames": args.seq_length + 1,
    }
    write_json(manifest_path, manifest)
    if args.validate_motionlib:
        manifest["official_motionlib_validation"] = validate_official_motionlib(
            args.bfm_repo, output, args.robot_xml, len(records), args.seq_length
        )
        manifest["structural_validation"] = "PASS"
        write_json(manifest_path, manifest)
        print("Structural Validation: PASS")
    if args.qc_root:
        qc = generate_qc(
            output,
            args.qc_root.resolve(),
            (args.husky_xml or args.robot_xml).resolve(),
            args.qc_seed,
        )
        manifest["qc"] = {
            "status": "PASS",
            "seed": args.qc_seed,
            "manifest": str(args.qc_root.resolve() / "qc_manifest.json"),
            "rendered_samples": qc["total_rendered_samples"],
        }
        write_json(manifest_path, manifest)
        print(f"Visual QC Generation: PASS ({qc['total_rendered_samples']} samples)")
    print(
        f"Aggregated {len(records)} phase motions from {int(raw_frames)} raw frames into {output}"
    )
    return 0


def main() -> int:
    args = parser().parse_args()
    if not args.aggregate_phase:
        raise SystemExit("only --aggregate-phase dataset mode is supported")
    for name in ("dataset_root", "bfm_repo", "bfm_reference", "robot_xml"):
        if not getattr(args, name).exists():
            raise SystemExit(f"--{name.replace('_', '-')} does not exist")
    if args.seq_length <= 0:
        raise SystemExit("--seq-length must be positive")
    if args.manifest is None:
        args.manifest = args.output.parent / "manifest.json"
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
