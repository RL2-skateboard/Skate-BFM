#!/usr/bin/env python3
"""Convert recorded HUSKY segments to the official BFM-Zero motion schema."""

from __future__ import annotations

import argparse
import json
import math
import re
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

BFM_FIXED_WRIST_JOINTS = {
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}
REQUIRED_RECORD_ARRAYS = {
    "root_trans_offset": (3,),
    "pose_aa": (30, 3),
    "dof": (29,),
    "root_rot": (4,),
    "smpl_joints": (24, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--bfm-repo", type=Path, required=True)
    parser.add_argument("--bfm-reference", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-motion-pkl", type=Path)
    parser.add_argument("--combined-output", type=Path)
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--validate-motionlib", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for option in ("input_root", "bfm_repo", "bfm_reference", "robot_xml"):
        path = getattr(args, option)
        if not path.exists():
            parser.error(f"--{option.replace('_', '-')} does not exist: {path}")
    if args.seq_length <= 0:
        parser.error("--seq-length must be positive")
    if (args.base_motion_pkl is None) != (args.combined_output is None):
        parser.error("--base-motion-pkl and --combined-output must be used together")
    if args.base_motion_pkl is not None and not args.base_motion_pkl.is_file():
        parser.error(f"--base-motion-pkl does not exist: {args.base_motion_pkl}")
    if args.manifest is None:
        args.manifest = args.output.parent / "manifest.json"
    return args


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def unqualified_name(name: str) -> str:
    return str(name).split("/")[-1]


def finite(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def load_reference(path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or not payload:
        raise TypeError("BFM reference must be a non-empty dict")
    record = next(iter(payload.values()))
    if not isinstance(record, dict):
        raise TypeError("BFM reference motion record must be a dict")
    keys = list(record.keys())
    required = set(REQUIRED_RECORD_ARRAYS) | {"fps"}
    missing = sorted(required - set(keys))
    if missing:
        raise ValueError(f"BFM reference record is missing keys: {missing}")
    return keys, record


def bfm_joint_contract(robot_xml: Path) -> tuple[list[str], np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(robot_xml.resolve()))
    joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    names = [model.joint(joint_id).name for joint_id in joint_ids]
    axes = np.asarray([model.jnt_axis[joint_id] for joint_id in joint_ids])
    if len(names) != 29 or axes.shape != (29, 3):
        raise ValueError(
            f"BFM robot must expose 29 named hinge joints, got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("BFM robot joint names are not unique")
    return names, axes.astype(np.float32)


def segment_paths(input_root: Path) -> list[tuple[Path, Path]]:
    result = []
    for metadata_path in sorted(input_root.glob("*/*/metadata.json")):
        state_path = metadata_path.with_name("state.npz")
        if not state_path.is_file():
            raise FileNotFoundError(f"missing state.npz beside {metadata_path}")
        result.append((metadata_path, state_path))
    if not result:
        raise FileNotFoundError(
            f"no segment metadata found below {input_root}; expected */*/metadata.json"
        )
    return result


def load_segment(
    metadata_path: Path,
    state_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(state_path, allow_pickle=False) as archive:
        state = {name: archive[name] for name in archive.files}
    for required in ("root_pos", "root_quat", "dof_pos", "sim_time"):
        if required not in state:
            raise ValueError(f"{state_path} is missing {required}")

    frame_count = int(metadata["num_frames"])
    for name, value in state.items():
        if value.ndim > 0 and value.shape[0] != frame_count:
            raise ValueError(
                f"{state_path}:{name} has {value.shape[0]} frames, expected {frame_count}"
            )
        if np.issubdtype(value.dtype, np.number):
            finite(f"{state_path}:{name}", value)
        elif not np.issubdtype(value.dtype, np.bool_):
            raise TypeError(f"{state_path}:{name} has unsupported dtype {value.dtype}")
    if state["root_pos"].shape != (frame_count, 3):
        raise ValueError(f"invalid root_pos shape: {state['root_pos'].shape}")
    if state["root_quat"].shape != (frame_count, 4):
        raise ValueError(f"invalid root_quat shape: {state['root_quat'].shape}")
    if state["dof_pos"].ndim != 2:
        raise ValueError(f"invalid dof_pos shape: {state['dof_pos'].shape}")
    return metadata, state


def load_raw_rollout(
    rollout_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    raw_root = rollout_root / "raw_rollout"
    raw_files = sorted(raw_root.glob("*.npz"))
    if len(raw_files) != 1:
        raise ValueError(
            f"{raw_root} must contain exactly one full rollout NPZ, found {len(raw_files)}"
        )
    raw_path = raw_files[0]
    metadata_path = raw_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing full rollout metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(raw_path, allow_pickle=False) as archive:
        state = {name: archive[name] for name in archive.files}
    for required in ("root_pos", "root_quat", "dof_pos", "sim_time"):
        if required not in state:
            raise ValueError(f"{raw_path} is missing {required}")
    frame_count = int(metadata["num_frames"])
    for name, value in state.items():
        if value.ndim > 0 and value.shape[0] != frame_count:
            raise ValueError(
                f"{raw_path}:{name} has {value.shape[0]} frames, expected {frame_count}"
            )
        if np.issubdtype(value.dtype, np.number):
            finite(f"{raw_path}:{name}", value)
    return raw_path, metadata, state


def atomic_joblib_dump(payload: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        joblib.dump(payload, temporary)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def map_dof(
    source_dof: np.ndarray,
    source_order: Sequence[str],
    target_order: Sequence[str],
    declared_fixed: Sequence[str],
) -> tuple[np.ndarray, dict[str, str], list[str]]:
    names = [unqualified_name(name) for name in source_order]
    if len(names) != source_dof.shape[1]:
        raise ValueError(
            f"joint_order has {len(names)} names but dof_pos has width {source_dof.shape[1]}"
        )
    if len(set(names)) != len(names):
        raise ValueError("source joint_order contains duplicate names")
    source_index = {name: index for index, name in enumerate(names)}
    fixed = set(declared_fixed)
    if fixed != BFM_FIXED_WRIST_JOINTS:
        raise ValueError(
            "fixed_bfm_joints must explicitly declare the six omitted wrist joints"
        )

    missing = [name for name in target_order if name not in source_index]
    if set(missing) != fixed:
        raise ValueError(
            "source/target joint definitions are incompatible; "
            f"unmapped target joints: {missing}"
        )
    unexpected = sorted(set(names) - set(target_order))
    if unexpected:
        raise ValueError(f"source contains joints absent from BFM: {unexpected}")

    target = np.zeros((source_dof.shape[0], len(target_order)), dtype=np.float32)
    mapping: dict[str, str] = {}
    for target_index, target_name in enumerate(target_order):
        if target_name in source_index:
            target[:, target_index] = source_dof[:, source_index[target_name]]
            mapping[target_name] = source_order[source_index[target_name]]
        else:
            mapping[target_name] = "fixed_zero"
    return target, mapping, missing


def convert_record(
    metadata: Mapping[str, Any],
    state: Mapping[str, np.ndarray],
    reference_keys: Sequence[str],
    reference_record: Mapping[str, Any],
    target_order: Sequence[str],
    target_axes: np.ndarray,
    seq_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_count = int(metadata["num_frames"])
    if frame_count < seq_length:
        raise ValueError(
            f"{metadata['segment_id']} has {frame_count} frames, below seq_length={seq_length}"
        )
    quaternion_order = metadata.get("quaternion_order")
    if quaternion_order != "wxyz":
        raise ValueError(
            f"{metadata['segment_id']} quaternion_order must be wxyz, got {quaternion_order}"
        )

    dof, mapping, fixed = map_dof(
        np.asarray(state["dof_pos"], dtype=np.float32),
        metadata["joint_order"],
        target_order,
        metadata.get("fixed_bfm_joints", []),
    )
    root_wxyz = np.asarray(state["root_quat"], dtype=np.float64)
    norms = np.linalg.norm(root_wxyz, axis=1)
    if np.any(norms <= 1e-8):
        raise ValueError(f"{metadata['segment_id']} has invalid root quaternion")
    root_wxyz = root_wxyz / norms[:, None]
    root_xyzw = root_wxyz[:, [1, 2, 3, 0]]
    root_rotvec = Rotation.from_quat(root_xyzw).as_rotvec().astype(np.float32)

    pose_aa = np.zeros((frame_count, 30, 3), dtype=np.float32)
    pose_aa[:, 0] = root_rotvec
    pose_aa[:, 1:] = dof[..., None] * target_axes[None, ...]
    fps_value = float(metadata["fps"])
    rounded_fps = round(fps_value)
    if not math.isclose(fps_value, rounded_fps, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(
            f"{metadata['segment_id']} fps must be integral for the BFM schema"
        )
    values: dict[str, Any] = {
        "root_trans_offset": np.asarray(state["root_pos"], dtype=np.float32),
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_xyzw.astype(np.float32),
        "smpl_joints": np.zeros((frame_count, 24, 3), dtype=np.float32),
        "fps": type(reference_record["fps"])(rounded_fps),
        "motion_name": str(metadata["segment_id"]),
    }
    unsupported = sorted(set(reference_keys) - set(values))
    if unsupported:
        raise ValueError(f"unsupported BFM reference keys: {unsupported}")
    record = {name: values[name] for name in reference_keys}
    if set(record) != set(reference_keys):
        raise AssertionError("converted record schema differs from BFM reference")

    for name, tail_shape in REQUIRED_RECORD_ARRAYS.items():
        value = np.asarray(record[name])
        expected = (frame_count, *tail_shape)
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}, expected {expected}")
        if value.dtype != np.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        finite(name, value)
    if not math.isclose(
        float(metadata["dt"]),
        1.0 / float(metadata["fps"]),
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError(f"{metadata['segment_id']} has inconsistent fps/dt")

    detail = {
        "joint_mapping": mapping,
        "fixed_zero_joints": fixed,
        "source_velocity": "simulator qvel/dof_vel retained in state.npz",
        "bfm_velocity": "derived by official MotionLib from pose_aa and fps",
        "quaternion_conversion": "MuJoCo wxyz -> BFM/SciPy xyzw",
    }
    return record, detail


def validate_official_motionlib(
    bfm_repo: Path,
    motion_file: Path,
    robot_xml: Path,
    motion_count: int,
    seq_length: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(bfm_repo.resolve()))
    import torch
    from easydict import EasyDict
    from humanoidverse.utils.motion_lib.motion_lib_base import FixHeightMode
    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

    motion_cfg = EasyDict(
        {
            "motion_file": str(motion_file.resolve()),
            "step_dt": 1.0 / 50.0,
            "fix_height": FixHeightMode.no_fix,
            "asset": EasyDict(
                {
                    "assetRoot": str(robot_xml.resolve().parent),
                    "assetFileName": robot_xml.name,
                }
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
    motion_lib = MotionLibRobot(motion_cfg, num_envs=motion_count, device="cpu")
    motion_lib.load_motions_for_training()
    ids = torch.arange(motion_count, dtype=torch.long)
    times = torch.zeros(motion_count, dtype=torch.float32)
    states = motion_lib.get_motion_state(ids, times)
    required = {
        "rg_pos_t",
        "rg_rot_t",
        "body_vel_t",
        "body_ang_vel_t",
        "dof_pos",
        "dof_vel",
    }
    missing = sorted(required - set(states))
    if missing:
        raise ValueError(f"official MotionLib state is missing fields: {missing}")
    for name in required:
        if not torch.isfinite(states[name]).all():
            raise ValueError(f"official MotionLib returned non-finite {name}")

    expert_result: dict[str, Any]
    try:
        from humanoidverse.agents.envs.humanoidverse_isaac import (
            load_expert_trajectories_from_motion_lib,
        )

        env = SimpleNamespace(
            _motion_lib=motion_lib,
            dt=1.0 / 50.0,
            device="cpu",
            default_dof_pos=torch.zeros((1, 29), dtype=torch.float32),
            gravity_vec=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
            config=SimpleNamespace(
                obs=SimpleNamespace(
                    obs_auxiliary={"history_actor": {}},
                    obs_dims={},
                    root_height_obs=True,
                )
            ),
        )
        agent_cfg = SimpleNamespace(
            model=SimpleNamespace(seq_length=seq_length)
        )
        expert_buffer = load_expert_trajectories_from_motion_lib(
            env,
            agent_cfg,
            device="cpu",
        )
        storage = expert_buffer.storage
        frame_count = int(storage["truncated"].shape[0])
        aligned = {
            int(storage["observation"][name].shape[0])
            for name in ("state", "last_action", "privileged_state")
        }
        aligned.update(
            {
                int(storage[name].shape[0])
                for name in ("terminated", "truncated", "motion_id")
            }
        )
        if aligned != {frame_count}:
            raise ValueError(f"expert buffer fields are not frame-aligned: {aligned}")
        sample_batch = expert_buffer.sample(
            batch_size=seq_length * 2,
            seq_length=seq_length,
        )
        sample_state = sample_batch["observation"]["state"]
        sample_next_state = sample_batch["next"]["observation"]["state"]
        if sample_state.shape != sample_next_state.shape:
            raise ValueError(
                "expert buffer current/next sample shapes do not match: "
                f"{sample_state.shape} vs {sample_next_state.shape}"
            )
        expert_result = {
            "status": "passed",
            "frames": frame_count,
            "state_shape": list(storage["observation"]["state"].shape),
            "privileged_state_shape": list(
                storage["observation"]["privileged_state"].shape
            ),
            "sample_seq_length": seq_length,
            "sample_state_shape": list(sample_state.shape),
            "sample_next_state_shape": list(sample_next_state.shape),
        }
    except Exception as error:
        raise RuntimeError(
            "official load_expert_trajectories_from_motion_lib failed"
        ) from error

    return {
        "motionlib": "passed",
        "motion_count": motion_lib.num_motions(),
        "total_duration": float(motion_lib.get_total_length()),
        "state_shapes": {name: list(states[name].shape) for name in sorted(required)},
        "expert_loader": expert_result,
    }


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    rollout_root = input_root.parent
    rollout_name = rollout_root.name
    if input_root.name != "dynamic_motion" or not re.fullmatch(
        r"rollout_\d+", rollout_name
    ):
        raise ValueError(
            "--input-root must be <rollout_NNN>/dynamic_motion so the full "
            "and subtask rollouts remain grouped"
        )
    full_output = args.output.parent / "full_rollout" / f"{rollout_name}.pkl"
    subtask_output_root = args.output.parent / "subtask_rollouts"
    failure_output_root = args.output.parent / "failure_rollouts"
    for path in (args.output, args.manifest, args.combined_output, full_output):
        if path is not None and path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    if (
        subtask_output_root.exists()
        and any(subtask_output_root.rglob("*.pkl"))
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{subtask_output_root} contains rollout files; pass --overwrite to replace them"
        )
    if (
        failure_output_root.exists()
        and any(failure_output_root.rglob("*.pkl"))
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{failure_output_root} contains rollout files; pass --overwrite to replace them"
        )

    reference_keys, reference_record = load_reference(args.bfm_reference)
    target_order, target_axes = bfm_joint_contract(args.robot_xml)
    expert_records: dict[str, dict[str, Any]] = {}
    failure_records: dict[str, dict[str, Any]] = {}
    manifest_records = []
    subtask_outputs: dict[Path, dict[str, dict[str, Any]]] = {}
    failure_outputs: dict[Path, dict[str, dict[str, Any]]] = {}
    subtask_counts: Counter[str] = Counter()
    expert_duration = 0.0
    failure_duration = 0.0

    raw_path, raw_metadata, raw_state = load_raw_rollout(rollout_root)
    dataset_split = str(raw_metadata.get("dataset_split", ""))
    if dataset_split not in {"train", "validation", "test"}:
        raise ValueError(
            f"{raw_path} has invalid or missing dataset_split: {dataset_split!r}"
        )
    physics_randomization = raw_metadata.get("physics_randomization")
    if not isinstance(physics_randomization, dict):
        raise ValueError(f"{raw_path} has invalid or missing physics_randomization")
    round_id = raw_metadata.get("round_id")

    for metadata_path, state_path in segment_paths(input_root):
        metadata, state = load_segment(metadata_path, state_path)
        if metadata.get("dataset_split") != dataset_split:
            raise ValueError(
                f"{metadata_path} split {metadata.get('dataset_split')!r} does "
                f"not match rollout split {dataset_split!r}"
            )
        if metadata.get("round_id") != round_id:
            raise ValueError(
                f"{metadata_path} round_id does not match its rollout"
            )
        segment_randomization = metadata.get("physics_randomization")
        if segment_randomization != physics_randomization:
            raise ValueError(
                f"{metadata_path} physics_randomization does not match its rollout"
            )
        record, conversion = convert_record(
            metadata,
            state,
            reference_keys,
            reference_record,
            target_order,
            target_axes,
            args.seq_length,
        )
        motion_type = str(metadata["motion_type"])
        motion_key = f"skate/{motion_type}/{metadata['segment_id']}"
        if motion_key in expert_records or motion_key in failure_records:
            raise ValueError(f"duplicate BFM motion key: {motion_key}")
        subtask_index = subtask_counts[motion_type]
        subtask_counts[motion_type] += 1
        is_failure = motion_type == "fall"
        output_root = failure_output_root if is_failure else subtask_output_root
        subtask_output = output_root / motion_type / f"rollout_{subtask_index:03d}.pkl"
        if is_failure:
            failure_records[motion_key] = record
            failure_outputs[subtask_output] = {motion_key: record}
            library_motion_id = None
        else:
            library_motion_id = len(expert_records)
            expert_records[motion_key] = record
            subtask_outputs[subtask_output] = {motion_key: record}
        duration = (int(metadata["num_frames"]) - 1) / float(metadata["fps"])
        if is_failure:
            failure_duration += duration
        else:
            expert_duration += duration
        manifest_records.append(
            {
                "motion_key": motion_key,
                "source_episode": metadata["source_episode"],
                "source_segment": metadata["segment_id"],
                "motion_type": motion_type,
                "duration": duration,
                "fps": float(metadata["fps"]),
                "num_frames": int(metadata["num_frames"]),
                "conversion_status": "converted",
                "dataset_split": dataset_split,
                "training_role": "failure_only" if is_failure else "expert",
                "original_npz": str(state_path.resolve()),
                "output_rollout": str(subtask_output.resolve()),
                "bfm_motion_id": library_motion_id,
                **conversion,
            }
        )

    if not expert_records:
        raise ValueError("rollout contains no non-fall expert motions")

    full_metadata = {
        "segment_id": rollout_name,
        "source_episode": raw_metadata["episode_id"],
        "motion_type": "full_rollout",
        "num_frames": raw_metadata["num_frames"],
        "fps": raw_metadata["fps"],
        "dt": raw_metadata["dt"],
        "joint_order": raw_metadata["joint_order"],
        "quaternion_order": raw_metadata["qpos_quaternion_order"],
        "fixed_bfm_joints": raw_metadata["fixed_bfm_joints"],
    }
    full_record, full_conversion = convert_record(
        full_metadata,
        raw_state,
        reference_keys,
        reference_record,
        target_order,
        target_axes,
        args.seq_length,
    )
    full_motion_key = f"skate/full_rollout/{rollout_name}"
    full_payload = {full_motion_key: full_record}
    full_duration = (
        (int(full_metadata["num_frames"]) - 1) / float(full_metadata["fps"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    full_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_full_output = full_output.with_name(f".{full_output.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    temporary_full_output.unlink(missing_ok=True)
    joblib.dump(expert_records, temporary_output)
    joblib.dump(full_payload, temporary_full_output)
    manifest: dict[str, Any] = {
        "rollout": rollout_name,
        "round_id": round_id,
        "dataset_split": dataset_split,
        "physics_randomization": physics_randomization,
        "bfm_reference": str(args.bfm_reference.resolve()),
        "reference_schema": reference_keys,
        "robot_xml": str(args.robot_xml.resolve()),
        "bfm_joint_order": target_order,
        "motion_count": len(expert_records),
        "total_duration": expert_duration,
        "failure_motion_count": len(failure_records),
        "failure_total_duration": failure_duration,
        "full_rollout": {
            "motion_key": full_motion_key,
            "source_npz": str(raw_path.resolve()),
            "output": str(full_output.resolve()),
            "num_frames": int(full_metadata["num_frames"]),
            "duration": full_duration,
            "fps": float(full_metadata["fps"]),
            "conversion_status": "converted",
            "training_role": "archive_only_not_expert",
            **full_conversion,
        },
        "subtask_rollout_root": str(subtask_output_root.resolve()),
        "failure_rollout_root": str(failure_output_root.resolve()),
        "motions": manifest_records,
        "validation": {"status": "not_requested"},
    }

    try:
        if args.validate_motionlib:
            manifest["validation"] = validate_official_motionlib(
                args.bfm_repo,
                temporary_output,
                args.robot_xml,
                len(expert_records),
                args.seq_length,
            )
            manifest["full_rollout"]["validation"] = validate_official_motionlib(
                args.bfm_repo,
                temporary_full_output,
                args.robot_xml,
                1,
                args.seq_length,
            )
        temporary_output.replace(args.output)
        temporary_full_output.replace(full_output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_full_output.unlink(missing_ok=True)
        raise

    expected_subtask_outputs = set(subtask_outputs)
    if args.overwrite and subtask_output_root.exists():
        for stale_output in subtask_output_root.rglob("*.pkl"):
            if stale_output not in expected_subtask_outputs:
                stale_output.unlink()
    for subtask_output, payload in subtask_outputs.items():
        atomic_joblib_dump(payload, subtask_output)
    expected_failure_outputs = set(failure_outputs)
    if args.overwrite and failure_output_root.exists():
        for stale_output in failure_output_root.rglob("*.pkl"):
            if stale_output not in expected_failure_outputs:
                stale_output.unlink()
    for failure_output, payload in failure_outputs.items():
        atomic_joblib_dump(payload, failure_output)

    if args.base_motion_pkl is not None:
        base = joblib.load(args.base_motion_pkl)
        if not isinstance(base, dict):
            raise TypeError("--base-motion-pkl must contain a dict")
        collisions = sorted(set(base) & set(expert_records))
        if collisions:
            raise ValueError(f"combined motion key collisions: {collisions[:5]}")
        combined = dict(base)
        combined.update(expert_records)
        assert args.combined_output is not None
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(combined, args.combined_output)
        base_duration = sum(
            (len(record["pose_aa"]) - 1) / float(record["fps"])
            for record in base.values()
        )
        manifest["combined"] = {
            "base_motion_pkl": str(args.base_motion_pkl.resolve()),
            "output": str(args.combined_output.resolve()),
            "base_motion_count": len(base),
            "skate_motion_count": len(expert_records),
            "combined_motion_count": len(combined),
            "base_total_duration": base_duration,
            "combined_total_duration": base_duration + expert_duration,
        }

    write_json(args.manifest, manifest)
    print(f"Dataset split: {dataset_split}")
    print(f"BFM expert motions: {len(expert_records)}")
    print(f"Expert duration: {expert_duration:.3f}s")
    print(f"Failure motions excluded from expert buffer: {len(failure_records)}")
    print(f"Subtask library: {args.output}")
    print(f"Full rollout: {full_output}")
    print(f"Individual subtask rollouts: {len(subtask_outputs)}")
    print(f"Individual failure rollouts: {len(failure_outputs)}")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
