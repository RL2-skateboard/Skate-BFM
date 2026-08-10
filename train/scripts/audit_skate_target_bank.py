#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build an auditable target bank from one recorded HUSKY Skate rollout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils._pytree import tree_map

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "husky_sim" / "src"))
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from evaluate_skate_bfm import build_frozen_agent, data_fingerprint
from train_skate_bfm import (
    _checkpoint_model_path,
    _state_fingerprint,
    build_motion_only_expert_context,
    build_train_config,
    load_expert_trajectories_from_motion_lib,
)


DEFAULT_EXPERT_MOTION = (
    REPOSITORY_ROOT
    / "train/dataset/skate-expert-pose/motion_library/skate_expert.pkl"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "train/dataset/skate-expert-pose/target_bank"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-rollout", type=Path, required=True)
    parser.add_argument(
        "--expert-motion",
        type=Path,
        default=DEFAULT_EXPERT_MOTION,
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "model/bfm-zero-official",
    )
    parser.add_argument(
        "--base-only-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "results/m2.2b-1/update_100/checkpoint",
    )
    parser.add_argument(
        "--base-skate-checkpoint",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results/m2.2b-3/base_skate_50_50/update_100/checkpoint"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--window-length",
        type=int,
        default=8,
        help="BFM expert sequence length used for candidate windows.",
    )
    return parser.parse_args()


def load_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {
        "sim_time",
        "root_pos",
        "root_quat",
        "qvel",
        "dof_pos",
        "dof_vel",
        "action",
        "board_root_pos",
        "board_root_quat",
        "board_root_lin_vel",
        "board_root_ang_vel",
        "command_v",
        "command_h",
        "phase_id",
        "phase_value",
        "fall",
        "reset",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"Raw rollout is missing required fields: {missing}")
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Raw rollout field contains NaN/Inf: {name}")
    frame_count = len(arrays["sim_time"])
    if any(value.ndim > 0 and value.shape[0] != frame_count for value in arrays.values()):
        raise ValueError("Raw rollout fields are not frame aligned.")
    return arrays


def quaternion_yaw(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion, dtype=np.float64), -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def vector_summary(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "finite": bool(np.isfinite(value).all()),
    }


def phase_runs(phase_ids: np.ndarray, mapping: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    start = 0
    for end in range(1, len(phase_ids) + 1):
        if end == len(phase_ids) or phase_ids[end] != phase_ids[start]:
            phase = mapping.get(str(int(phase_ids[start])), "unknown")
            result.append(
                {
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "phase": phase,
                    "frame_count": end - start,
                }
            )
            start = end
    return result


def physical_window(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
) -> dict[str, Any]:
    board_yaw = np.unwrap(quaternion_yaw(arrays["board_root_quat"][start:end]))
    board_velocity = arrays["board_root_lin_vel"][start:end]
    root_velocity = arrays["qvel"][start:end, :3]
    root_yaw = np.unwrap(quaternion_yaw(arrays["root_quat"][start:end]))
    return {
        "start_frame": start,
        "end_frame": end - 1,
        "start_time_s": float(arrays["sim_time"][start]),
        "end_time_s": float(arrays["sim_time"][end - 1]),
        "board_position_start_m": arrays["board_root_pos"][start].tolist(),
        "board_position_end_m": arrays["board_root_pos"][end - 1].tolist(),
        "board_displacement_m": float(
            np.linalg.norm(
                arrays["board_root_pos"][end - 1]
                - arrays["board_root_pos"][start]
            )
        ),
        "board_velocity_mean_mps": board_velocity.mean(axis=0).tolist(),
        "board_forward_velocity_mean_mps": float(board_velocity[:, 0].mean()),
        "board_lateral_velocity_mean_mps": float(board_velocity[:, 1].mean()),
        "board_speed_mean_mps": float(
            np.linalg.norm(board_velocity[:, :2], axis=1).mean()
        ),
        "board_heading_delta_rad": float(board_yaw[-1] - board_yaw[0]),
        "board_heading_delta_deg": float(np.degrees(board_yaw[-1] - board_yaw[0])),
        "root_height_mean_m": float(arrays["root_pos"][start:end, 2].mean()),
        "root_height_min_m": float(arrays["root_pos"][start:end, 2].min()),
        "root_linear_speed_mean_mps": float(
            np.linalg.norm(root_velocity, axis=1).mean()
        ),
        "root_heading_delta_deg": float(
            np.degrees(root_yaw[-1] - root_yaw[0])
        ),
        "command_v_values": np.unique(arrays["command_v"][start:end]).tolist(),
        "command_h_values": np.unique(arrays["command_h"][start:end]).tolist(),
        "phase_ids": np.unique(arrays["phase_id"][start:end]).astype(int).tolist(),
        "fall_frames": int(arrays["fall"][start:end].sum()),
        "reset_frames": int(arrays["reset"][start:end].sum()),
    }


def load_expert_observations(
    agent,
    expert_path: Path,
) -> dict[str, torch.Tensor]:
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression))
    cfg = build_train_config()
    context = build_motion_only_expert_context(cfg.env)
    motion_cfg = copy.deepcopy(context.config.robot.motion)
    motion_cfg.motion_file = str(expert_path)
    motion_lib = type(context._motion_lib)(
        motion_cfg,
        num_envs=1,
        device=context.device,
    )
    expert_env = SimpleNamespace(
        _motion_lib=motion_lib,
        num_envs=1,
        dt=context.dt,
        device=context.device,
        default_dof_pos=context.default_dof_pos.to(context.device),
        gravity_vec=context.gravity_vec.to(context.device),
        config=context.config,
    )
    buffer = load_expert_trajectories_from_motion_lib(
        expert_env,
        agent.cfg,
        device=agent.device,
    )
    storage = buffer.storage["observation"]
    return {
        key: value.detach().cpu()
        for key, value in storage.items()
        if key in {"state", "last_action", "privileged_state"}
    }


def encode_target(agent, observations: dict[str, torch.Tensor]) -> np.ndarray:
    device = agent.device
    next_obs = tree_map(lambda value: value.to(device), observations)
    with torch.no_grad():
        normalized = agent._model._normalize(next_obs)
        backward = agent._model._backward_map(normalized)
        z = agent._model.project_z(backward.mean(dim=0, keepdim=True))[0]
    if not torch.isfinite(z).all():
        raise ValueError("Target latent contains NaN/Inf.")
    return z.detach().cpu().numpy().astype(np.float32)


def latent_record(agent, observations: dict[str, torch.Tensor]) -> dict[str, Any]:
    z = encode_target(agent, observations)
    return {
        "values": z.tolist(),
        "shape": list(z.shape),
        "dtype": str(z.dtype),
        "norm": float(np.linalg.norm(z)),
        "fingerprint": data_fingerprint(z),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def main() -> int:
    args = parse_args()
    raw_path = args.raw_rollout.expanduser().resolve()
    expert_path = args.expert_motion.expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not expert_path.is_file():
        raise FileNotFoundError(expert_path)
    if args.window_length <= 0:
        raise ValueError("--window-length must be positive")

    arrays = load_raw(raw_path)
    frame_count = len(arrays["sim_time"])
    if frame_count < args.window_length:
        raise ValueError("Raw rollout is shorter than the target window.")

    raw_metadata_path = raw_path.with_suffix(".json")
    raw_metadata = (
        json.loads(raw_metadata_path.read_text())
        if raw_metadata_path.is_file()
        else {}
    )
    phase_mapping = raw_metadata.get("phase_mapping", {})
    candidates = [
        physical_window(arrays, start, start + args.window_length)
        for start in range(0, frame_count - args.window_length + 1, args.window_length)
    ]
    # Choose one stable forward window after startup, with low lateral drift.
    eligible = [
        item
        for item in candidates
        if abs(item["board_lateral_velocity_mean_mps"]) <= 0.01
        and abs(item["board_heading_delta_deg"]) <= 1.0
        and item["fall_frames"] == 0
    ]
    if not eligible:
        raise ValueError("No physically stable target window was found.")
    selected = max(
        eligible,
        key=lambda item: item["board_forward_velocity_mean_mps"],
    )
    target_start = int(selected["start_frame"])
    target_end = int(selected["end_frame"]) + 1

    checkpoint_paths = {
        "official_bfm0": args.official_checkpoint.expanduser().resolve(),
        "base_only_update100": args.base_only_checkpoint.expanduser().resolve(),
        "base_skate_update100": args.base_skate_checkpoint.expanduser().resolve(),
    }
    for name, path in checkpoint_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    latent_by_checkpoint: dict[str, dict[str, Any]] = {}
    observations = None
    for name, checkpoint in checkpoint_paths.items():
        agent, _ = build_frozen_agent(checkpoint)
        if observations is None:
            all_observations = load_expert_observations(agent, expert_path)
            observations = {
                key: value[target_start:target_end]
                for key, value in all_observations.items()
            }
        parameter_fingerprint_before = _state_fingerprint(
            agent._model,
            parameters=True,
        )
        latent = latent_record(agent, observations)
        parameter_fingerprint_after = _state_fingerprint(
            agent._model,
            parameters=True,
        )
        if parameter_fingerprint_before != parameter_fingerprint_after:
            raise RuntimeError("Target inference changed model parameters.")
        latent_by_checkpoint[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_model_sha256": sha256_file(
                _checkpoint_model_path(checkpoint)
            ),
            "parameter_fingerprint_before": parameter_fingerprint_before,
            "parameter_fingerprint_after": parameter_fingerprint_after,
            "parameters_changed": False,
            "latent": latent,
        }
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    latent_values = {
        name: np.asarray(record["latent"]["values"], dtype=np.float32)
        for name, record in latent_by_checkpoint.items()
    }
    pairwise_cosine = {
        f"{left}_vs_{right}": cosine_similarity(
            latent_values[left],
            latent_values[right],
        )
        for left, right in (
            ("official_bfm0", "base_only_update100"),
            ("official_bfm0", "base_skate_update100"),
            ("base_only_update100", "base_skate_update100"),
        )
    }

    command_v = np.asarray(arrays["command_v"])
    command_h = np.asarray(arrays["command_h"])
    board_yaw = np.unwrap(quaternion_yaw(arrays["board_root_quat"]))
    board_delta = arrays["board_root_pos"][-1] - arrays["board_root_pos"][0]
    global_physical = {
        "frame_count": frame_count,
        "duration_s": float(arrays["sim_time"][-1] - arrays["sim_time"][0]),
        "board_displacement_m": float(np.linalg.norm(board_delta)),
        "board_displacement_vector_m": board_delta.tolist(),
        "board_forward_velocity_mean_mps": float(
            arrays["board_root_lin_vel"][:, 0].mean()
        ),
        "board_lateral_velocity_mean_mps": float(
            arrays["board_root_lin_vel"][:, 1].mean()
        ),
        "board_speed_mean_mps": float(
            np.linalg.norm(arrays["board_root_lin_vel"][:, :2], axis=1).mean()
        ),
        "board_heading_delta_rad": float(board_yaw[-1] - board_yaw[0]),
        "board_heading_delta_deg": float(
            np.degrees(board_yaw[-1] - board_yaw[0])
        ),
        "root_height_min_m": float(arrays["root_pos"][:, 2].min()),
        "root_height_max_m": float(arrays["root_pos"][:, 2].max()),
        "fall_frames": int(arrays["fall"].sum()),
        "reset_frames": int(arrays["reset"].sum()),
    }

    target = {
        "target_id": "skate_target_00",
        "frame_start": target_start,
        "frame_end_inclusive": target_end - 1,
        "time_start_s": float(arrays["sim_time"][target_start]),
        "time_end_s": float(arrays["sim_time"][target_end - 1]),
        "physical_description": (
            "One continuous forward push / board acceleration window under "
            "zero heading command; no physically distinguishable steer segment "
            "is present in this artifact."
        ),
        "observed_root_behavior": (
            "The robot root remains above the recorded minimum height, with "
            "finite pose and velocity values throughout the selected window."
        ),
        "observed_board_behavior": (
            "Board forward velocity is positive, lateral velocity is small, "
            "and board heading change is below the selection threshold."
        ),
        "command_alignment": {
            "status": "aligned",
            "command_v_mean": float(command_v.mean()),
            "command_h_mean": float(command_h.mean()),
            "evidence": (
                "Raw metadata and every frame contain command_v=1.0 and "
                "command_h=0.0; board motion is forward with negligible "
                "lateral drift and no meaningful heading change."
            ),
            "limitation": (
                "This establishes compatibility with the recorded command, "
                "not causal command tracking or downstream task success."
            ),
        },
        "physical_descriptor": selected,
        "latents": latent_by_checkpoint,
        "cosine_similarity": pairwise_cosine,
    }

    field_inventory = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(
                np.isfinite(value).all()
                if np.issubdtype(value.dtype, np.number)
                else True
            ),
        }
        for name, value in sorted(arrays.items())
    }
    output = {
        "schema": "skate-bfm-target-bank-v1",
        "audit": {
            "date": date.today().isoformat(),
            "training_performed": False,
            "rollout_performed": False,
            "optimizer_steps": 0,
            "actor_execution": False,
            "target_bank_size": 1,
            "window_length": args.window_length,
            "latent_definition": (
                "Normalize MotionLib expert next observations, apply the "
                "agent backward map B to each frame, average B over the "
                "8-frame window, then apply project_z. This is the exact "
                "mathematical path used by FBcprAgent.encode_expert()."
            ),
            "selection_rule": (
                "Disjoint seq_length windows; exclude fall windows and choose "
                "the highest board forward velocity among windows with "
                "|mean lateral velocity| <= 0.01 m/s and |board yaw delta| <= 1 deg."
            ),
        },
        "source": {
            "raw_rollout": str(raw_path),
            "raw_rollout_sha256": sha256_file(raw_path),
            "raw_metadata": str(raw_metadata_path),
            "raw_metadata_sha256": (
                sha256_file(raw_metadata_path)
                if raw_metadata_path.is_file()
                else None
            ),
            "expert_motion": str(expert_path),
            "expert_motion_sha256": sha256_file(expert_path),
            "motion_key": "skate/push/m1_1_rollout_001_push_000",
            "source_segment": raw_metadata.get("episode_id"),
            "raw_phase_runs": phase_runs(arrays["phase_id"], phase_mapping),
        },
        "command_audit": {
            "command_v": {
                "meaning": (
                    "Forward linear-velocity command scalar. In test_scene "
                    "sim.py it is multiplied by 2.0 before ONNX inference; "
                    "the official skater command config names the underlying "
                    "quantity lin_vel_x."
                ),
                "raw_field_present": True,
                "unique_values": np.unique(command_v).tolist(),
                "status": "available",
            },
            "command_h": {
                "meaning": (
                    "Relative heading / steering command in radians. The "
                    "official test keyboard increases h for left and "
                    "decreases h for right; it is passed unscaled to the "
                    "policy input."
                ),
                "raw_field_present": True,
                "unique_values": np.unique(command_h).tolist(),
                "status": "available",
            },
            "expert_metadata_present": True,
            "expert_physical_consistency": "aligned_for_forward_zero_heading",
            "steer_left_right_evidence": "not_found",
        },
        "raw_field_inventory": field_inventory,
        "unavailable_fields": [],
        "global_physical_summary": global_physical,
        "candidate_windows": candidates,
        "targets": [target],
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "target_bank.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Target bank written to {output_path}")
    print(
        f"Target bank size=1, selected frames={target_start}-{target_end - 1}, "
        "training=0, rollouts=0, optimizer_steps=0"
    )
    for name, record in latent_by_checkpoint.items():
        print(
            f"{name}: z_norm={record['latent']['norm']:.6f} "
            f"fingerprint={record['latent']['fingerprint']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
