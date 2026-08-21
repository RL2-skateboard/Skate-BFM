#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Evaluate frozen Skate-BFM checkpoints without training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
import torch
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "husky_sim" / "src"))
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from skate_bfm.integration import HuskyBfmOnlineEnv
from skate_husky import randomize_husky_play_physics
from train_skate_bfm import (
    AlignedSkateTrackingContext,
    EXPERT_DATASETS,
    encode_target,
    load_expert,
    load_frozen_agent,
)
from train_runner import (
    OFFICIAL_BFM0_SHA256,
    checkpoint_model_path,
    hash_buffers,
    hash_components,
    hash_data,
    hash_file,
    hash_params,
    load_source_rollout,
    resolve_source_rollout_path,
)


DEFAULT_PROTOCOL = REPOSITORY_ROOT / "train/evaluation_protocol.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/m2.5b-target-conditioned"
DEFAULT_RANDOM_SEEDS = (2026081101, 2026081102, 2026081103, 2026081104)
METRIC_NAMES = (
    "board_forward_displacement_m",
    "board_forward_velocity_mean_mps",
    "board_forward_velocity_peak_mps",
    "board_lateral_displacement_abs_m",
    "board_lateral_velocity_abs_mean_mps",
    "board_heading_delta_abs_deg",
    "root_height_min_m",
    "root_tilt_max_deg",
    "episode_duration_s",
    "robot_board_separation_initial_m",
    "robot_board_separation_final_m",
    "robot_board_separation_mean_m",
    "robot_board_separation_max_m",
    "robot_board_separation_delta_m",
    "board_tilt_initial_deg",
    "board_tilt_final_deg",
    "board_tilt_mean_deg",
    "board_tilt_max_deg",
    "feet_on_board_ratio",
    "illegal_contact_ratio",
    "action_soft_saturation_fraction",
    "action_hard_saturation_fraction",
    "torque_utilization_mean",
    "torque_utilization_p95",
    "torque_utilization_p99",
    "torque_utilization_max",
    "torque_utilization_ge_0_95_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("rollout", "fixed-target", "motion-reference"),
        default="rollout",
        help=(
            "rollout=legacy random robustness, fixed-target=historical evaluator, "
            "motion-reference=primary MotionLibrary-conditioned evaluation."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Frozen checkpoint directory for mode=rollout.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(EXPERT_DATASETS),
        default="phase",
        help="Formal expert dataset used to sample rollout reset states.",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument(
        "--latent-mode",
        choices=("random", "aligned-expert"),
        default="random",
        help="Use random rollout latents or the reset motion's per-step tracking latents.",
    )
    parser.add_argument(
        "--latent-refresh",
        type=int,
        default=100,
        help="Resample z at the same transition interval used during formal training.",
    )
    parser.add_argument(
        "--stochastic-actions",
        action="store_true",
        help="Sample Actor actions as in training; evaluation is deterministic by default.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the MuJoCo viewer and run at the 50 Hz control rate.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Write one selected rollout as an offscreen MP4.",
    )
    parser.add_argument(
        "--video-episode",
        type=int,
        default=0,
        help="Zero-based rollout index to record when --video is set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Frozen rollout JSON path; defaults beside the checkpoint result directory.",
    )
    parser.add_argument("--target-bank", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expert-motion",
        type=Path,
        help="Override the formal reset dataset or historical target MotionLib.",
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "model/bfm-zero-official",
    )
    parser.add_argument(
        "--checkpoint-10k",
        type=Path,
        help="Legacy fixed-target evaluator checkpoint argument.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        help="Arbitrary frozen checkpoint for motion-reference comparison against Fresh BFM0.",
    )
    parser.add_argument(
        "--checkpoint-20k",
        type=Path,
        help="M2.5b checkpoint saved after 20,000 transitions.",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        help="Optional M2.5b training summary to update after inference-only evaluation.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Explicit held-out MotionLibrary split for mode=motion-reference.",
    )
    parser.add_argument(
        "--reference-dataset",
        choices=("phase", "continuous", "both"),
        default="both",
        help="Motion-reference dataset(s) to evaluate.",
    )
    parser.add_argument(
        "--motion-key",
        help=(
            "Exact motion to run; with --viewer it is the first motion in a "
            "same-window multi-motion sequence."
        ),
    )
    parser.add_argument(
        "--checkpoint-view",
        choices=("fresh", "trained"),
        help="Checkpoint selected by --motion-key: Fresh BFM0 or --reference-checkpoint.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the two-source Phase motion-reference contract smoke.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip selected motion-reference diagnostic videos.",
    )
    parser.add_argument(
        "--reference-output-dir",
        type=Path,
        default=Path("/tmp/m26_t1e_v"),
        help="Output directory for motion-reference artifacts.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/tmp/m26_t1e_v_videos"),
        help="Directory for motion-reference diagnostic MP4 files.",
    )
    return parser.parse_args()


class ExpertResetSampler:
    """Sample the same raw expert reset distribution used by formal training."""

    def __init__(
        self,
        motion_file: Path,
        env: HuskyBfmOnlineEnv,
        seed: int,
        eligible_frame_counts: Mapping[str, int] | None = None,
    ) -> None:
        loaded = joblib.load(motion_file)
        if not isinstance(loaded, dict) or not loaded:
            raise RuntimeError("Expert reset dataset must be a non-empty motion dictionary.")
        self.records: dict[str, dict[str, Any]] = {}
        for motion_key, record in loaded.items():
            required = (
                "source_raw_npz",
                "source_start_frame",
                "source_end_frame",
                "physics_seed",
                "dof",
            )
            if any(field not in record for field in required):
                raise RuntimeError(f"Expert motion {motion_key} lacks reset provenance.")
            motion_frames = int(np.asarray(record["dof"]).shape[0])
            source_start = int(record["source_start_frame"])
            source_end = int(record["source_end_frame"])
            if motion_frames <= 0 or source_end - source_start != motion_frames:
                raise RuntimeError(f"Expert motion {motion_key} has an invalid frame range.")
            self.records[str(motion_key)] = {
                "source_raw_npz": record["source_raw_npz"],
                "source_start_frame": source_start,
                "motion_frames": motion_frames,
                "source_round": record.get("source_round"),
                "source_rollout": record.get("source_rollout"),
                "command_v": record.get("command_v"),
                "command_h": record.get("command_h"),
                "phase": record.get("phase"),
                "physics_seed": int(record["physics_seed"]),
            }
        del loaded, record
        if eligible_frame_counts is None:
            self.eligible_frame_counts = None
            self.motion_keys = tuple(self.records)
        else:
            if set(eligible_frame_counts) != set(self.records):
                raise RuntimeError("Eligible-frame and reset motion-key sets do not match.")
            self.eligible_frame_counts = dict(eligible_frame_counts)
            self.motion_keys = tuple(
                name for name in self.records if self.eligible_frame_counts[name] > 0
            )
            if not self.motion_keys:
                raise RuntimeError("No reset motion has enough future expert states.")
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.raw_cache: dict[
            Path,
            tuple[np.ndarray, np.ndarray, dict[str, object]],
        ] = {}

    def sample(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, Any]]:
        motion_key = self.motion_keys[int(self.rng.integers(len(self.motion_keys)))]
        record = self.records[motion_key]
        frame_count = (
            record["motion_frames"]
            if self.eligible_frame_counts is None
            else self.eligible_frame_counts[motion_key]
        )
        local_frame = int(self.rng.integers(frame_count))
        source_frame = int(record["source_start_frame"]) + local_frame
        source_path = resolve_source_rollout_path(record)
        if source_path not in self.raw_cache:
            self.raw_cache[source_path] = load_source_rollout(
                source_path,
                self.env,
                int(record["physics_seed"]),
            )
        raw_qpos, raw_qvel, source_physics = self.raw_cache[source_path]
        if source_frame >= raw_qpos.shape[0] or source_frame >= raw_qvel.shape[0]:
            raise RuntimeError(f"Expert reset frame is outside raw rollout: {source_path}")
        return (
            np.asarray(raw_qpos[source_frame], dtype=np.float64).copy(),
            np.asarray(raw_qvel[source_frame], dtype=np.float64).copy(),
            source_physics,
            {
                "motion_key": motion_key,
                "local_frame": local_frame,
                "source_raw_npz": str(source_path),
                "source_frame": source_frame,
                "source_round": record["source_round"],
                "source_rollout": record["source_rollout"],
                "command_v": record["command_v"],
                "command_h": record["command_h"],
                "phase": record["phase"],
                "physics_seed": record["physics_seed"],
                "source_physics_aligned": True,
            },
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def update_training_summary(
    path: Path,
    *,
    evaluation_path: Path,
    evaluation: dict[str, Any],
) -> None:
    """Attach inference-only fixed-evaluation provenance to one M2.5b summary."""

    summary = json.loads(path.read_text())
    if summary.get("milestone") != "M2.5b Original BFM-Zero Skate Baseline Training":
        raise RuntimeError("Training summary is not an M2.5b baseline result.")
    checkpoint_reports = evaluation["checkpoint_reports"]
    if set(checkpoint_reports) != {
        "official_bfm0",
        "m2.5b_10k",
        "m2.5b_20k",
    }:
        raise RuntimeError("Fixed evaluation checkpoint set is incomplete.")
    all_terminated = all(
        rollout["episode"]["terminated"]
        for rollout in evaluation["rollouts"]
    )
    if any(
        report["mutation"]["parameters_changed"]
        or report["mutation"]["buffers_changed"]
        or report["mutation"]["optimizer_steps"] != 0
        or report["mutation"]["backward_calls"] != 0
        or report["mutation"]["agent_update_calls"] != 0
        for report in checkpoint_reports.values()
    ):
        raise RuntimeError("Evaluation mutated a checkpoint state.")
    summary["evaluation"] = {
        "result_file": str(evaluation_path),
        "result_sha256": hash_file(evaluation_path),
        "protocol_version": evaluation["evaluation"]["protocol_version"],
        "rollout_count": len(evaluation["rollouts"]),
        "training_replay_mutated": False,
        "checkpoint_reports": {
            name: {
                "model_sha256": report["checkpoint_model_sha256"],
                "target_z_fingerprint": report["target_z_fingerprint"],
                "target_z_norm": report["target_z_norm"],
                "parameters_changed": report["mutation"]["parameters_changed"],
                "buffers_changed": report["mutation"]["buffers_changed"],
                "optimizer_steps": report["mutation"]["optimizer_steps"],
                "backward_calls": report["mutation"]["backward_calls"],
                "agent_update_calls": report["mutation"]["agent_update_calls"],
            }
            for name, report in checkpoint_reports.items()
        },
        "all_rollouts_terminated_before_horizon": all_terminated,
        "fixed_protocol_performance_trend": (
            "INCONCLUSIVE"
            if all_terminated
            else "REQUIRES_DOCUMENTED_INTERPRETATION"
        ),
        "interpretation": (
            "All fixed evaluation rollouts reached the native fall terminal "
            "state before the 128-step horizon; board displacement alone is "
            "not a success metric."
            if all_terminated
            else "Read the fixed protocol metrics together; no task-success "
            "claim is generated automatically."
        ),
    }
    summary["performance_evaluated"] = True
    summary["performance_trend"] = summary["evaluation"][
        "fixed_protocol_performance_trend"
    ]
    write_json(path, summary)


def load_and_validate_target_bank(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "skate-bfm-target-bank-v1":
        raise RuntimeError("Target bank schema mismatch.")
    if payload.get("audit", {}).get("target_bank_size") != 1:
        raise RuntimeError("Target bank must contain exactly one target.")
    targets = payload.get("targets", [])
    if len(targets) != 1:
        raise RuntimeError("Target bank target count mismatch.")
    target = targets[0]
    expected = {
        "target_id": "skate_target_00",
        "frame_start": 24,
        "frame_end_inclusive": 31,
    }
    for key, value in expected.items():
        if target.get(key) != value:
            raise RuntimeError(f"Target bank {key} mismatch.")
    if target.get("command_alignment", {}).get("status") != "aligned":
        raise RuntimeError("Target alignment is not marked aligned.")
    command = payload.get("command_audit", {})
    if command.get("command_v", {}).get("unique_values") != [1.0]:
        raise RuntimeError("Target command_v mismatch.")
    if command.get("command_h", {}).get("unique_values") != [0.0]:
        raise RuntimeError("Target command_h mismatch.")
    if any(
        "values" in record.get("latent", {})
        for record in target.get("latents", {}).values()
    ):
        raise RuntimeError("Tracked target bank must not contain latent values.")
    return payload, hash_file(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("evaluator_version") != "skate-bfm-fixed-eval-v1":
        raise RuntimeError("Only skate-bfm-fixed-eval-v1 is supported.")
    if protocol.get("rollout_horizon") != 128:
        raise RuntimeError("M2.3b-0 requires the fixed 128-step horizon.")
    if protocol.get("control_dt_s") != 0.02:
        raise RuntimeError("M2.3b-0 requires control_dt_s=0.02.")
    return protocol


def checkpoint_paths(
    *,
    official_checkpoint: Path,
    checkpoint_10k: Path,
    checkpoint_20k: Path,
) -> dict[str, Path]:
    return {
        "official_bfm0": official_checkpoint,
        "m2.5b_10k": checkpoint_10k,
        "m2.5b_20k": checkpoint_20k,
    }


def initial_state_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": {
            key: np.asarray(raw[key]).copy()
            for key in (
                "root_position",
                "root_quaternion",
                "root_linear_velocity",
                "root_angular_velocity",
            )
        },
        "board": {
            key: np.asarray(raw[key]).copy()
            for key in (
                "board_position",
                "board_quaternion",
                "board_linear_velocity",
                "board_angular_velocity",
            )
        },
    }


def board_yaw(quaternion: np.ndarray) -> float:
    """Return the scalar yaw angle of a MuJoCo wxyz quaternion."""

    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def board_tilt_deg(quaternion: np.ndarray) -> float:
    """Return world-up tilt from a normalized MuJoCo wxyz quaternion."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise RuntimeError("Board quaternion must be finite wxyz.")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise RuntimeError("Board quaternion has zero norm.")
    w, x, y, z = quaternion / norm
    up_dot_world_up = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_dot_world_up, -1.0, 1.0))))


class ControlDiagnostics:
    """Aggregate actual HUSKY command and generalized-torque usage per rollout."""

    _GROUPS = ("waist", "hip", "ankle")

    def __init__(self, env: HuskyBfmOnlineEnv) -> None:
        report = env.env.physical_actuator_report
        if len(report) != env.env.robot_action_dim:
            raise RuntimeError("Physical actuator report must contain exactly 23 entries.")
        self.names: tuple[str, ...] = tuple(
            str(item["joint_name"]).removeprefix("robot/") for item in report
        )
        if len(set(self.names)) != len(self.names):
            raise RuntimeError("Physical actuator joint names must be unique.")
        limits = np.asarray(
            [item["derived_joint_torque_limit"] for item in report],
            dtype=np.float64,
        )
        if limits.shape != (env.env.robot_action_dim,) or not np.isfinite(limits).all():
            raise RuntimeError("Physical actuator torque limits must be finite 23D.")
        if (limits <= 0.0).any():
            raise RuntimeError("Physical actuator torque limits must be positive.")

        dof_addresses = []
        for index, item in enumerate(report):
            actuator_name = str(item["actuator_name"])
            joint_name = str(item["joint_name"])
            actuator = env.env.model.actuator(index)
            joint = env.env.model.joint(int(env.env.model.actuator_trnid[index, 0]))
            if actuator.name != actuator_name or joint.name != joint_name:
                raise RuntimeError("Physical actuator report order does not match MuJoCo.")
            dof_addresses.append(int(joint.dofadr[0]))
        if len(set(dof_addresses)) != env.env.robot_action_dim:
            raise RuntimeError("Physical actuator report lacks one-to-one joint DoFs.")

        self.limits = limits
        self.dof_addresses = np.asarray(dof_addresses, dtype=np.int32)
        self.group_indices = {
            group: np.asarray(
                [index for index, name in enumerate(self.names) if group in name],
                dtype=np.int32,
            )
            for group in self._GROUPS
        }
        if any(indices.size == 0 for indices in self.group_indices.values()):
            raise RuntimeError("HUSKY actuator groups waist, hip, and ankle are required.")
        self.action_elements = 0
        self.action_soft = 0
        self.action_hard = 0
        self.group_action = {
            group: {"elements": 0, "soft": 0, "hard": 0}
            for group in self._GROUPS
        }
        self.torque_utilization: list[np.ndarray] = []
        self.group_torque_utilization = {
            group: [] for group in self._GROUPS
        }

    def update(self, action_husky: torch.Tensor, qfrc_actuator: np.ndarray) -> None:
        action = np.asarray(action_husky.detach().cpu(), dtype=np.float64)
        torque = np.asarray(qfrc_actuator, dtype=np.float64)[self.dof_addresses]
        if action.shape != self.limits.shape or torque.shape != self.limits.shape:
            raise RuntimeError("Action or actuator torque shape does not match 23D HUSKY.")
        if not np.isfinite(action).all() or not np.isfinite(torque).all():
            raise RuntimeError("Action and actuator torque diagnostics must be finite.")
        action_abs = np.abs(action)
        utilization = np.abs(torque) / self.limits
        self.action_elements += action.size
        self.action_soft += int(np.count_nonzero(action_abs >= 0.95))
        self.action_hard += int(np.count_nonzero(action_abs >= 0.99))
        self.torque_utilization.append(utilization)
        for group, indices in self.group_indices.items():
            grouped_action = action_abs[indices]
            group_counts = self.group_action[group]
            group_counts["elements"] += grouped_action.size
            group_counts["soft"] += int(np.count_nonzero(grouped_action >= 0.95))
            group_counts["hard"] += int(np.count_nonzero(grouped_action >= 0.99))
            self.group_torque_utilization[group].append(utilization[indices])

    @staticmethod
    def _fraction(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            raise RuntimeError("Control diagnostics require at least one action.")
        return float(numerator / denominator)

    def summary(self) -> dict[str, Any]:
        if not self.torque_utilization:
            raise RuntimeError("Control diagnostics require at least one transition.")
        utilization = np.concatenate(self.torque_utilization)
        result: dict[str, Any] = {
            "actuator_joint_names": list(self.names),
            "action_soft_saturation_fraction": self._fraction(
                self.action_soft, self.action_elements
            ),
            "action_hard_saturation_fraction": self._fraction(
                self.action_hard, self.action_elements
            ),
            "torque_utilization_mean": float(utilization.mean()),
            "torque_utilization_p95": float(np.quantile(utilization, 0.95)),
            "torque_utilization_p99": float(np.quantile(utilization, 0.99)),
            "torque_utilization_max": float(utilization.max()),
            "torque_utilization_ge_0_95_fraction": float(
                np.mean(utilization >= 0.95)
            ),
            "groups": {},
        }
        for group, values in self.group_torque_utilization.items():
            group_utilization = np.concatenate(values)
            action = self.group_action[group]
            result["groups"][group] = {
                "joint_names": [self.names[index] for index in self.group_indices[group]],
                "action_soft_saturation_fraction": self._fraction(
                    action["soft"], action["elements"]
                ),
                "action_hard_saturation_fraction": self._fraction(
                    action["hard"], action["elements"]
                ),
                "torque_utilization_p95": float(np.quantile(group_utilization, 0.95)),
                "torque_utilization_p99": float(np.quantile(group_utilization, 0.99)),
                "torque_utilization_ge_0_95_fraction": float(
                    np.mean(group_utilization >= 0.95)
                ),
            }
        return result


def physical_metrics(
    records: list[dict[str, Any]],
    initial_raw: dict[str, Any],
    control_dt: float,
) -> dict[str, float]:
    initial_board_position = np.asarray(initial_raw["board_position"], dtype=float)
    initial_root_position = np.asarray(initial_raw["root_position"], dtype=float)
    initial_board_yaw = board_yaw(initial_raw["board_quaternion"])
    forward_axis = np.asarray(
        [math.cos(initial_board_yaw), math.sin(initial_board_yaw)]
    )
    lateral_axis = np.asarray(
        [-math.sin(initial_board_yaw), math.cos(initial_board_yaw)]
    )
    board_positions = np.asarray(
        [item["board_position"] for item in records],
        dtype=float,
    )
    board_velocities = np.asarray(
        [item["board_linear_velocity"] for item in records],
        dtype=float,
    )
    board_yaws = np.unwrap(
        np.asarray(
            [initial_board_yaw]
            + [
                board_yaw(item["board_quaternion"])
                for item in records
            ],
            dtype=float,
        )
    )
    displacement = board_positions[-1] - initial_board_position
    forward_velocity = board_velocities[:, :2] @ forward_axis
    lateral_velocity = board_velocities[:, :2] @ lateral_axis
    root_heights = np.asarray([item["root_height"] for item in records])
    gravity = np.asarray([item["projected_gravity"] for item in records])
    root_tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
    root_positions = np.asarray([item["root_position"] for item in records], dtype=float)
    separations = np.linalg.norm(root_positions[:, :2] - board_positions[:, :2], axis=1)
    initial_separation = float(
        np.linalg.norm(initial_root_position[:2] - initial_board_position[:2])
    )
    board_tilts = np.asarray(
        [board_tilt_deg(item["board_quaternion"]) for item in records],
        dtype=float,
    )
    initial_board_tilt = board_tilt_deg(initial_raw["board_quaternion"])
    feet_on_board = np.asarray(
        [bool(item["feet_on_board"]) for item in records], dtype=bool
    )
    illegal_contact = np.asarray(
        [bool(item["illegal_contact"]) for item in records], dtype=bool
    )
    return {
        "board_forward_displacement_m": float(displacement[:2] @ forward_axis),
        "board_forward_velocity_mean_mps": float(forward_velocity.mean()),
        "board_forward_velocity_peak_mps": float(forward_velocity.max()),
        "board_lateral_displacement_abs_m": float(
            abs(displacement[:2] @ lateral_axis)
        ),
        "board_lateral_velocity_abs_mean_mps": float(
            np.abs(lateral_velocity).mean()
        ),
        "board_heading_delta_abs_deg": float(
            abs(np.degrees(board_yaws[-1] - board_yaws[0]))
        ),
        "root_height_min_m": float(root_heights.min()),
        "root_tilt_max_deg": float(root_tilt.max()),
        "episode_duration_s": float(len(records) * control_dt),
        "robot_board_separation_initial_m": initial_separation,
        "robot_board_separation_final_m": float(separations[-1]),
        "robot_board_separation_mean_m": float(separations.mean()),
        "robot_board_separation_max_m": float(separations.max()),
        "robot_board_separation_delta_m": float(separations[-1] - initial_separation),
        "board_tilt_initial_deg": initial_board_tilt,
        "board_tilt_final_deg": float(board_tilts[-1]),
        "board_tilt_mean_deg": float(board_tilts.mean()),
        "board_tilt_max_deg": float(board_tilts.max()),
        "feet_on_board_ratio": float(feet_on_board.mean()),
        "illegal_contact_ratio": float(illegal_contact.mean()),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in rows], dtype=float)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "count": int(values.size),
        }
    return result


def run_frozen_evaluation(args: argparse.Namespace) -> int:
    """Run inference-only rollouts for one formal Skate-BFM checkpoint."""

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for mode=rollout.")
    if args.episodes <= 0 or args.horizon <= 0 or args.latent_refresh <= 0:
        raise ValueError("--episodes, --horizon, and --latent-refresh must be positive.")
    if args.video_episode < 0 or args.video_episode >= args.episodes:
        raise ValueError("--video-episode must be within [0, --episodes).")
    if args.video is not None and args.video.suffix.lower() != ".mp4":
        raise ValueError("--video must have the .mp4 extension.")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    expert_motion = (
        args.expert_motion.expanduser().resolve()
        if args.expert_motion is not None
        else EXPERT_DATASETS[args.dataset].resolve()
    )
    if not expert_motion.is_file():
        raise FileNotFoundError(f"Expert reset dataset not found: {expert_motion}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent, checkpoint_report = load_frozen_agent(checkpoint)
    gradients_enabled = any(
        parameter.requires_grad for parameter in agent._model.parameters()
    )
    if agent._model.training or gradients_enabled:
        raise RuntimeError("Frozen evaluator requires an eval-mode model with gradients disabled.")
    before = {
        "parameters": hash_params(agent._model),
        "buffers": hash_buffers(agent._model),
        "normalizer": hash_buffers(agent._model._obs_normalizer),
        "components": hash_components(agent),
    }

    env = HuskyBfmOnlineEnv(viewer=args.viewer, realtime=args.viewer)
    tracking_context = (
        AlignedSkateTrackingContext.load(agent, expert_motion)
        if args.latent_mode == "aligned-expert"
        else None
    )
    eligible_frame_counts = (
        {
            name: tracking_context.eligible_frame_count(name, args.horizon)
            for name in tracking_context.trajectories
        }
        if tracking_context is not None
        else None
    )
    sampler = ExpertResetSampler(
        expert_motion,
        env,
        args.seed,
        eligible_frame_counts=eligible_frame_counts,
    )
    rollouts: list[dict[str, Any]] = []
    viewer_closed = False
    video_path = args.video.expanduser().resolve() if args.video is not None else None
    video_writer = None
    renderer = None
    try:
        if video_path is not None:
            import imageio.v2 as imageio

            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                video_path,
                fps=round(1.0 / env.env.control_dt),
                macro_block_size=1,
                codec="libx264",
            )
            renderer = mujoco.Renderer(env.env.model, height=720, width=1280)
        progress = tqdm(
            total=args.episodes,
            desc="Frozen evaluation",
            unit="episode",
            dynamic_ncols=True,
        )
        for episode_index in range(args.episodes):
            torch.manual_seed(args.seed + episode_index)
            qpos, qvel, source_physics, reset = sampler.sample()
            observation = env.reset(
                qpos=qpos,
                qvel=qvel,
                source_physics=source_physics,
            )
            initial_raw = env.env._observation()
            records: list[dict[str, Any]] = []
            control_diagnostics = ControlDiagnostics(env)
            latent_fingerprints: list[str] = []
            terminated = False
            truncated = False
            z: torch.Tensor | None = None
            tracking_z = None
            tracking_ranges = None
            if tracking_context is not None:
                tracking_z, tracking_ranges = tracking_context.encode(
                    agent._model,
                    reset["motion_key"],
                    reset["local_frame"],
                    args.horizon,
                )

            record_video = video_writer is not None and episode_index == args.video_episode
            if record_video:
                renderer.update_scene(env.env.data, camera="robot/tracking")
                video_writer.append_data(renderer.render())
            for step in range(args.horizon):
                if args.viewer and not env.env.is_running:
                    viewer_closed = True
                    break
                if tracking_z is not None:
                    z = tracking_z[step]
                    latent_fingerprints.append(hash_data(z))
                elif z is None or step % args.latent_refresh == 0:
                    z = agent._model.sample_z(1, device=agent.device)[0]
                    latent_fingerprints.append(hash_data(z))
                if z is None:
                    raise RuntimeError("Rollout latent was not initialized.")
                model_observation = {
                    key: value.unsqueeze(0).to(agent.device)
                    for key, value in observation.items()
                }
                with torch.no_grad():
                    action = agent.act(
                        obs=model_observation,
                        z=z.unsqueeze(0),
                        mean=not args.stochastic_actions,
                    )[0]
                transition = env.step(
                    action,
                    z,
                    truncated=step == args.horizon - 1,
                )
                control_diagnostics.update(
                    transition.action_husky,
                    env.env.data.qfrc_actuator,
                )
                records.append(dict(transition.raw_metadata))
                observation = transition.next_observation
                terminated = transition.terminated
                truncated = transition.truncated
                if record_video:
                    renderer.update_scene(env.env.data, camera="robot/tracking")
                    video_writer.append_data(renderer.render())
                if terminated or truncated:
                    break

            if not records:
                if viewer_closed:
                    break
                raise RuntimeError("Frozen rollout produced no transitions.")
            metrics = physical_metrics(records, initial_raw, env.env.control_dt)
            diagnostics = control_diagnostics.summary()
            metrics.update(
                {
                    name: float(diagnostics[name])
                    for name in (
                        "action_soft_saturation_fraction",
                        "action_hard_saturation_fraction",
                        "torque_utilization_mean",
                        "torque_utilization_p95",
                        "torque_utilization_p99",
                        "torque_utilization_max",
                        "torque_utilization_ge_0_95_fraction",
                    )
                }
            )
            row = {
                "episode_index": episode_index,
                "episode_seed": args.seed + episode_index,
                "reset": reset,
                "latent_mode": args.latent_mode,
                "latent_refresh_steps": (
                    args.latent_refresh if tracking_z is None else 1
                ),
                "latent_fingerprints": latent_fingerprints,
                "tracking": (
                    {
                        "trajectory": tracking_context.trajectories[
                            reset["motion_key"]
                        ],
                        "ranges": tracking_ranges,
                    }
                    if tracking_context is not None
                    else None
                ),
                "action_mode": (
                    "stochastic" if args.stochastic_actions else "deterministic_mean"
                ),
                "episode": {
                    "transition_count": len(records),
                    "terminated": terminated,
                    "truncated": truncated,
                    "viewer_closed": viewer_closed,
                    "time_to_fall_s": (
                        len(records) * env.env.control_dt if terminated else None
                    ),
                    "fall_reason": (
                        records[-1].get("fall_reason", "") if terminated else ""
                    ),
                },
                "metrics": metrics,
                "control_diagnostics": {
                    "actuator_joint_names": diagnostics["actuator_joint_names"],
                    "groups": diagnostics["groups"],
                },
            }
            rollouts.append(row)
            progress.update()
            progress.set_postfix(
                falls=sum(item["episode"]["terminated"] for item in rollouts),
                last_steps=len(records),
                refresh=False,
            )
            if terminated:
                progress.write(
                    "[Termination] "
                    f"episode={episode_index + 1}, step={len(records)}, "
                    f"time={metrics['episode_duration_s']:.2f}s, "
                    f"reason={row['episode']['fall_reason'] or 'unknown'}"
                )
            if viewer_closed:
                break
    finally:
        if "progress" in locals():
            progress.close()
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()
        env.close()

    after = {
        "parameters": hash_params(agent._model),
        "buffers": hash_buffers(agent._model),
        "normalizer": hash_buffers(agent._model._obs_normalizer),
        "components": hash_components(agent),
    }
    if before != after:
        raise RuntimeError("Frozen evaluation mutated model parameters or buffers.")
    if not rollouts:
        raise RuntimeError("Frozen evaluation ended before completing any rollout.")

    durations = np.asarray(
        [row["metrics"]["episode_duration_s"] for row in rollouts],
        dtype=np.float64,
    )
    terminated_count = sum(row["episode"]["terminated"] for row in rollouts)
    truncated_count = sum(row["episode"]["truncated"] for row in rollouts)
    output = {
        "schema": "skate-bfm-frozen-rollout-eval-v1",
        "evaluation": {
            "date": date.today().isoformat(),
            "evaluation_only": True,
            "training_performed": False,
            "checkpoint": str(checkpoint),
            "checkpoint_report": checkpoint_report,
            "dataset": args.dataset,
            "expert_motion": str(expert_motion),
            "expert_motion_sha256": hash_file(expert_motion),
            "reset_mode": (
                "uniform_motion_uniform_eligible_local_frame_raw_qpos_qvel"
                if tracking_context is not None
                else "uniform_motion_uniform_local_frame_raw_qpos_qvel"
            ),
            "source_physics": "canonical_raw_metadata_exact",
            "domain_randomization": False,
            "episodes_requested": args.episodes,
            "horizon": args.horizon,
            "control_dt_s": env.env.control_dt,
            "latent_refresh_steps": (
                1 if tracking_context is not None else args.latent_refresh
            ),
            "latent_mode": args.latent_mode,
            "action_mode": (
                "stochastic" if args.stochastic_actions else "deterministic_mean"
            ),
            "viewer": args.viewer,
            "video": (
                {
                    "path": str(video_path),
                    "episode_index": args.video_episode,
                    "fps": round(1.0 / env.env.control_dt),
                    "camera": "robot/tracking",
                }
                if video_path is not None
                else None
            ),
            "seed": args.seed,
            "mutation": {
                "parameters_changed": False,
                "buffers_changed": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "agent_update_calls": 0,
                "before": before,
                "after": after,
            },
        },
        "summary": {
            "episodes_completed": len(rollouts),
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "fall_rate": terminated_count / len(rollouts),
            "duration_s": {
                "mean": float(durations.mean()),
                "median": float(np.median(durations)),
                "min": float(durations.min()),
                "max": float(durations.max()),
            },
            "viewer_closed": viewer_closed,
            "performance_claim": "NONE",
        },
        "rollouts": rollouts,
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            checkpoint.parent
            / "evaluation"
            / f"{checkpoint.name}_frozen_rollout_{args.dataset}.json"
        )
    )
    write_json(output_path, output)
    print(
        "Frozen evaluation complete: "
        f"episodes={len(rollouts)}, falls={terminated_count}, "
        f"horizon_completions={truncated_count}, mutation=False"
    )
    print(f"Metrics: {output_path}")
    if video_path is not None:
        print(f"Video: {video_path}")
    return 0


REFERENCE_METRICS = (
    "joint_position_mae_rad",
    "joint_velocity_mae_rad_s",
    "root_xy_displacement_error_m",
    "root_z_error_m",
    "root_orientation_geodesic_error_deg",
    "board_xy_displacement_error_m",
    "board_orientation_geodesic_error_deg",
    "board_linear_velocity_error_mps",
    "board_angular_velocity_error_rad_s",
    "board_heading_error_deg",
    "board_tilt_error_deg",
    "coupling_xy_error_m",
    "coupling_z_error_m",
)


def quaternion_geodesic_deg(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    actual /= np.linalg.norm(actual, axis=-1, keepdims=True)
    reference /= np.linalg.norm(reference, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(actual * reference, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def wrapped_angle_deg(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.degrees(np.abs(np.angle(np.exp(1j * (actual - reference)))))


def summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Reference metrics must be finite and non-empty.")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


class ReferenceSourceResolver:
    """Resolve one held-out split without searching across dataset roots."""

    def __init__(self, split: str) -> None:
        metadata_split = {"val": "validation", "test": "test"}[split]
        self.split = split
        self.metadata_split = metadata_split
        self.raw_root = (
            REPOSITORY_ROOT / "train/dataset/sim_collected" / split / "raw"
        ).resolve()
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"Reference raw root not found: {self.raw_root}")
        self.cache: dict[Path, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}

    def load(self, record: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
        required = (
            "source_round",
            "source_rollout",
            "source_episode",
            "source_start_frame",
            "source_end_frame",
            "physics_seed",
            "dataset_split",
        )
        if any(name not in record for name in required):
            raise RuntimeError("Reference MotionLibrary record lacks provenance.")
        if record["dataset_split"] != self.metadata_split:
            raise RuntimeError(
                f"MotionLibrary record split={record['dataset_split']!r}, "
                f"expected {self.metadata_split!r}."
            )
        round_id = str(record["source_round"]).zfill(3)
        rollout_id = str(record["source_rollout"]).zfill(3)
        rollout_root = self.raw_root / f"round_{round_id}" / f"rollout_{rollout_id}" / "raw_rollout"
        matches = sorted(rollout_root.glob("*.npz"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one canonical raw NPZ under {rollout_root}.")
        source_path = matches[0].resolve()
        recorded_name = Path(str(record["source_raw_npz"])).name
        if source_path.name != recorded_name:
            raise RuntimeError("MotionLibrary raw filename does not match split resolver.")
        if source_path not in self.cache:
            metadata_path = source_path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            expected = {
                "dataset_split": self.metadata_split,
                "round_id": round_id,
                "rollout_id": rollout_id,
                "episode_id": str(record["source_episode"]),
            }
            actual = {name: str(metadata.get(name, "")) for name in expected}
            if actual != expected:
                raise RuntimeError(f"Raw metadata provenance mismatch: {actual} != {expected}")
            source_physics = metadata.get("physics_randomization")
            if not isinstance(source_physics, dict):
                raise RuntimeError("Reference raw metadata lacks source physics.")
            if int(source_physics.get("seed", -1)) != int(record["physics_seed"]):
                raise RuntimeError("Reference physics seed mismatch.")
            with np.load(source_path, allow_pickle=False) as archive:
                state = {
                    name: np.asarray(archive[name]).copy()
                    for name in (
                        "qpos",
                        "qvel",
                        "frame_idx",
                        "root_pos",
                        "root_quat",
                        "dof_pos",
                        "dof_vel",
                        "board_root_pos",
                        "board_root_quat",
                        "board_root_lin_vel",
                        "board_root_ang_vel",
                    )
                }
            count = state["qpos"].shape[0]
            if (
                state["qpos"].ndim != 2
                or state["qvel"].ndim != 2
                or any(value.shape[0] != count for value in state.values())
                or not all(np.isfinite(value).all() for value in state.values())
            ):
                raise RuntimeError("Reference raw rollout is malformed.")
            self.cache[source_path] = (state, copy.deepcopy(metadata))
        state, metadata = self.cache[source_path]
        return state, metadata, source_path


def reference_dataset_paths(split: str, dataset: str) -> tuple[Path, Path]:
    root = REPOSITORY_ROOT / "train/dataset/sim_collected" / split / dataset
    return (
        root / "motion_library" / f"skate_expert_{dataset}.pkl",
        root / "motion_library" / "manifest.json",
    )


def load_reference_records(
    split: str,
    dataset: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Path, Path]:
    motion_path, manifest_path = reference_dataset_paths(split, dataset)
    if not motion_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Reference {split}/{dataset} MotionLibrary is missing.")
    manifest = json.loads(manifest_path.read_text())
    expected_split = {"val": "validation", "test": "test"}[split]
    if manifest.get("dataset_split") != expected_split:
        raise RuntimeError("Reference manifest split mismatch.")
    # Evaluation keeps the record dictionary in memory so thousands of
    # per-motion arrays do not remain as open memmap file descriptors.
    records = joblib.load(motion_path)
    if not isinstance(records, dict) or not records:
        raise RuntimeError("Reference MotionLibrary must be a non-empty mapping.")
    if dataset == "phase":
        labels = {record.get("phase_label") for record in records.values()}
        if not labels <= {"push", "push2steer", "steer_left", "steer_forward", "steer_right", "steer2push"}:
            raise RuntimeError("Phase MotionLibrary has invalid phase labels.")
    return {str(key): value for key, value in records.items()}, manifest, motion_path, manifest_path


def reference_metric_series(
    actual: list[dict[str, Any]],
    raw: Mapping[str, np.ndarray],
    source_start: int,
) -> dict[str, np.ndarray]:
    steps = len(actual)
    reference = slice(source_start + 1, source_start + steps + 1)
    if source_start < 0 or source_start + steps >= raw["qpos"].shape[0]:
        raise RuntimeError("Reference transition slice is outside canonical raw trajectory.")
    initial = source_start
    actual_joint_pos = np.asarray([row["joint_position"] for row in actual])
    actual_joint_vel = np.asarray([row["joint_velocity"] for row in actual])
    actual_root_pos = np.asarray([row["root_position"] for row in actual])
    actual_root_quat = np.asarray([row["root_quaternion"] for row in actual])
    actual_board_pos = np.asarray([row["board_position"] for row in actual])
    actual_board_quat = np.asarray([row["board_quaternion"] for row in actual])
    actual_board_lin_vel = np.asarray([row["board_linear_velocity"] for row in actual])
    actual_board_ang_vel = np.asarray([row["board_angular_velocity"] for row in actual])
    ref_root_pos = raw["root_pos"][reference]
    ref_board_pos = raw["board_root_pos"][reference]
    values = {
        "joint_position_mae_rad": np.mean(np.abs(actual_joint_pos - raw["dof_pos"][reference]), axis=1),
        "joint_velocity_mae_rad_s": np.mean(np.abs(actual_joint_vel - raw["dof_vel"][reference]), axis=1),
        "root_xy_displacement_error_m": np.linalg.norm(
            (actual_root_pos[:, :2] - raw["root_pos"][initial, :2])
            - (ref_root_pos[:, :2] - raw["root_pos"][initial, :2]),
            axis=1,
        ),
        "root_z_error_m": np.abs(actual_root_pos[:, 2] - ref_root_pos[:, 2]),
        "root_orientation_geodesic_error_deg": quaternion_geodesic_deg(
            actual_root_quat, raw["root_quat"][reference]
        ),
        "board_xy_displacement_error_m": np.linalg.norm(
            (actual_board_pos[:, :2] - raw["board_root_pos"][initial, :2])
            - (ref_board_pos[:, :2] - raw["board_root_pos"][initial, :2]),
            axis=1,
        ),
        "board_orientation_geodesic_error_deg": quaternion_geodesic_deg(
            actual_board_quat, raw["board_root_quat"][reference]
        ),
        "board_linear_velocity_error_mps": np.linalg.norm(
            actual_board_lin_vel - raw["board_root_lin_vel"][reference], axis=1
        ),
        "board_angular_velocity_error_rad_s": np.linalg.norm(
            actual_board_ang_vel - raw["board_root_ang_vel"][reference], axis=1
        ),
        "board_heading_error_deg": wrapped_angle_deg(
            np.asarray([board_yaw(value) for value in actual_board_quat]),
            np.asarray([board_yaw(value) for value in raw["board_root_quat"][reference]]),
        ),
        "board_tilt_error_deg": np.abs(
            np.asarray([board_tilt_deg(value) for value in actual_board_quat])
            - np.asarray([board_tilt_deg(value) for value in raw["board_root_quat"][reference]])
        ),
        "coupling_xy_error_m": np.linalg.norm(
            (actual_root_pos[:, :2] - actual_board_pos[:, :2])
            - (ref_root_pos[:, :2] - ref_board_pos[:, :2]),
            axis=1,
        ),
        "coupling_z_error_m": np.abs(
            (actual_root_pos[:, 2] - actual_board_pos[:, 2])
            - (ref_root_pos[:, 2] - ref_board_pos[:, 2])
        ),
    }
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def reference_metric_summary(
    series: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    return {name: summarize_values(value) for name, value in series.items()}


def run_reference_motion(
    agent: Any,
    tracking: AlignedSkateTrackingContext,
    resolver: ReferenceSourceResolver,
    motion_key: str,
    record: Mapping[str, Any],
    *,
    env: HuskyBfmOnlineEnv | None = None,
    video_path: Path | None = None,
    local_frame: int = 0,
) -> dict[str, Any]:
    raw, metadata, source_path = resolver.load(record)
    source_start = int(record["source_start_frame"])
    source_end = int(record["source_end_frame"])
    trajectory = tracking.trajectories.get(motion_key)
    if trajectory is None:
        raise RuntimeError(f"Missing tracking trajectory for {motion_key}.")
    raw_frames = source_end - source_start
    if raw_frames != int(np.asarray(record["dof"]).shape[0]):
        raise RuntimeError("MotionLibrary source frame range is inconsistent.")
    if trajectory["raw_frames"] != raw_frames:
        raise RuntimeError("Tracking and MotionLibrary raw frame counts differ.")
    if not 0 <= local_frame < raw_frames:
        raise ValueError(f"Reset local frame is outside motion {motion_key}: {local_frame}")
    t_eval = min(raw_frames - local_frame - 1, trajectory["length"] - local_frame - 1)
    if t_eval <= 0:
        raise RuntimeError("Reset local frame has no executable transition.")
    z, ranges = tracking.encode(agent._model, motion_key, local_frame, t_eval)
    if z.shape != (t_eval, 256) or not torch.isfinite(z).all():
        raise RuntimeError("Tracking latent is not finite [T_eval, 256].")
    if any(
        item["future_start"] != local_frame + step + 1
        for step, item in enumerate(ranges)
    ):
        raise RuntimeError("Tracking future indices are not reset-aligned.")

    owns_env = env is None
    if env is None:
        env = HuskyBfmOnlineEnv()
    renderer = writer = None
    actual: list[dict[str, Any]] = []
    diagnostics: ControlDiagnostics | None = None
    terminated = truncated = False
    first_action_fingerprint = None
    try:
        reset_frame = source_start + local_frame
        qpos = np.asarray(raw["qpos"][reset_frame], dtype=np.float64)
        qvel = np.asarray(raw["qvel"][reset_frame], dtype=np.float64)
        observation = env.reset(
            qpos=qpos,
            qvel=qvel,
            source_physics=metadata["physics_randomization"],
        )
        if not (
            np.allclose(env.env.data.qpos, qpos, atol=1e-8, rtol=0.0)
            and np.allclose(env.env.data.qvel, qvel, atol=1e-8, rtol=0.0)
        ):
            raise RuntimeError("Physical reset does not reproduce canonical raw qpos/qvel.")
        diagnostics = ControlDiagnostics(env)
        if video_path is not None:
            import imageio.v2 as imageio

            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                video_path, fps=round(1.0 / env.env.control_dt), macro_block_size=1, codec="libx264"
            )
            renderer = mujoco.Renderer(env.env.model, height=720, width=1280)
            renderer.update_scene(env.env.data, camera="robot/tracking")
            writer.append_data(renderer.render())
        for step in range(t_eval):
            model_observation = {
                key: value.unsqueeze(0).to(agent.device) for key, value in observation.items()
            }
            with torch.no_grad():
                action = agent.act(obs=model_observation, z=z[step].unsqueeze(0), mean=True)[0]
            if first_action_fingerprint is None:
                first_action_fingerprint = hash_data(action)
            transition = env.step(action, z[step], truncated=step == t_eval - 1)
            diagnostics.update(transition.action_husky, env.env.data.qfrc_actuator)
            actual.append(dict(transition.raw_metadata))
            observation = transition.next_observation
            terminated, truncated = transition.terminated, transition.truncated
            if writer is not None:
                renderer.update_scene(env.env.data, camera="robot/tracking")
                writer.append_data(renderer.render())
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if owns_env:
            env.close()
    if not actual or diagnostics is None:
        raise RuntimeError("Reference rollout produced no physical transitions.")
    executed = len(actual)
    reset_frame = source_start + local_frame
    metric_series = reference_metric_series(actual, raw, reset_frame)
    metric_summary = reference_metric_summary(metric_series)
    control = diagnostics.summary()
    return {
        "motion_key": motion_key,
        "source": {
            "raw_npz": str(source_path),
            "round": str(record["source_round"]),
            "rollout": str(record["source_rollout"]),
            "episode": str(record["source_episode"]),
            "physics_seed": int(record["physics_seed"]),
            "start_frame": source_start,
            "end_frame": source_end,
            "reset_frame": reset_frame,
            "metadata_split": metadata["dataset_split"],
        },
        "phase_label": record.get("phase_label"),
        "command_v": float(record["command_v"]),
        "command_h": float(record["command_h"]),
        "t_eval": t_eval,
        "t_exec": executed,
        "completion_ratio": float(executed / t_eval),
        "full_completion": bool(executed == t_eval and not terminated),
        "terminated": terminated,
        "fall_reason": str(actual[-1].get("fall_reason", "")) if terminated else "",
        "time_to_fall_s": float(executed * 0.02) if terminated else None,
        "tracking": {
            "z_shape": list(z.shape),
            "z_norm": summarize_values(torch.linalg.vector_norm(z, dim=1).cpu().numpy()),
            "ranges": ranges,
            "frame_difference": trajectory["length"] - trajectory["raw_frames"],
        },
        "alignment": {
            "reset_local_frame": local_frame,
            "reset_raw_frame": reset_frame,
            "actual_next_reference_start": reset_frame + 1,
            "actual_next_reference_end": reset_frame + executed,
            "raw_frame_idx": [
                int(value)
                for value in raw["frame_idx"][
                    reset_frame + 1 : reset_frame + min(executed, 5) + 1
                ]
            ],
        },
        "metrics": metric_summary,
        "_metric_series": metric_series,
        "first_action_fingerprint": first_action_fingerprint,
        "control": control,
        "contact": {
            "feet_on_board_ratio": float(np.mean([row["feet_on_board"] for row in actual])),
            "illegal_contact_ratio": float(np.mean([row["illegal_contact"] for row in actual])),
            "root_tilt_max_deg": float(max(
                np.degrees(np.arccos(np.clip(-np.asarray(row["projected_gravity"])[2], -1.0, 1.0)))
                for row in actual
            )),
            "board_tilt_max_deg": float(max(board_tilt_deg(row["board_quaternion"]) for row in actual)),
        },
        "video_path": str(video_path) if video_path is not None else None,
    }


def run_reference_viewer_sequence(
    agent: Any,
    tracking: AlignedSkateTrackingContext,
    resolver: ReferenceSourceResolver,
    records: Mapping[str, Mapping[str, Any]],
    *,
    motion_key: str | None,
    episodes: int,
    seed: int,
    env: HuskyBfmOnlineEnv,
    output_dir: Path,
) -> int:
    """Run random push/steer reference resets in one live MuJoCo viewer."""

    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    motion_keys = sorted(
        key
        for key, record in records.items()
        if record.get("phase_label") == "push"
        or str(record.get("phase_label", "")).startswith("steer_")
    )
    if not motion_keys:
        raise RuntimeError("Viewer sequence has no push/steer motions.")
    eligible = [
        key for key in motion_keys if tracking.eligible_frame_count(key, 1) > 0
    ]
    if not eligible:
        raise RuntimeError("Viewer sequence has no push/steer motion with a valid tracking frame.")
    rng = np.random.default_rng(seed)
    if motion_key is not None:
        if motion_key not in eligible:
            raise KeyError(
                f"Viewer motion must be an eligible push/steer motion: {motion_key}"
            )
    rows = []
    for _ in range(episodes):
        if not env.env.is_running:
            break
        key = motion_key if motion_key is not None else eligible[
            int(rng.integers(len(eligible)))
        ]
        frame_count = tracking.eligible_frame_count(key, 1)
        local_frame = int(rng.integers(frame_count))
        row = run_reference_motion(
            agent,
            tracking,
            resolver,
            key,
            records[key],
            env=env,
            local_frame=local_frame,
        )
        motion_key = None
        rows.append(serializable_reference_rows([row])[0])
        print(
            f"[Viewer reset] sequence={len(rows)}/{episodes}, "
            f"phase={row['phase_label']}, motion={key}, "
            f"local_frame={local_frame}, steps={row['t_exec']}, "
            f"termination={row['fall_reason'] or 'natural_end'}"
        )
    if not rows:
        raise RuntimeError("Viewer closed before the first reference motion completed.")
    write_json(output_dir / "viewer_sequence.json", {
        "schema": "skate-bfm-motion-reference-viewer-sequence-v1",
        "episodes_requested": episodes,
        "episodes_completed": len(rows),
        "same_viewer_window": True,
        "seed": seed,
        "phase_filter": ["push", "steer_left", "steer_forward", "steer_right"],
        "motion_keys": [row["motion_key"] for row in rows],
        "reset_local_frames": [
            row["alignment"]["reset_local_frame"] for row in rows
        ],
        "rollouts": rows,
    })
    return len(rows)


def checkpoint_mutation(agent: Any) -> dict[str, Any]:
    return {
        "parameters": hash_params(agent._model),
        "buffers": hash_buffers(agent._model),
        "normalizer": hash_buffers(agent._model._obs_normalizer),
        "components": hash_components(agent),
    }


def evaluate_reference_dataset(
    checkpoint_name: str,
    checkpoint: Path,
    motion_path: Path,
    records: Mapping[str, Mapping[str, Any]],
    resolver: ReferenceSourceResolver,
) -> dict[str, Any]:
    agent, load_report = load_frozen_agent(checkpoint)
    if agent._model.training or any(parameter.requires_grad for parameter in agent._model.parameters()):
        raise RuntimeError("Reference evaluator requires a frozen eval-mode checkpoint.")
    before = checkpoint_mutation(agent)
    tracking = AlignedSkateTrackingContext.load(agent, motion_path)
    if set(tracking.trajectories) != set(records):
        raise RuntimeError("Reference tracking and MotionLibrary motion keys differ.")
    rows = []
    env = HuskyBfmOnlineEnv()
    progress = tqdm(
        sorted(records),
        desc=f"{checkpoint_name} reference",
        unit="motion",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    try:
        for motion_key in progress:
            row = run_reference_motion(
                agent, tracking, resolver, motion_key, records[motion_key], env=env
            )
            rows.append(row)
            progress.set_postfix(
                completed=sum(item["full_completion"] for item in rows),
                falls=sum(item["terminated"] for item in rows),
                refresh=False,
            )
    finally:
        progress.close()
        env.close()
    after = checkpoint_mutation(agent)
    if before != after:
        raise RuntimeError(f"Frozen model mutated during {checkpoint_name} reference evaluation.")
    return {
        "checkpoint_name": checkpoint_name,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": hash_file(checkpoint_model_path(checkpoint)),
        "load_report": load_report,
        "frame_difference_counts": tracking.frame_difference_counts(),
        "mutation": {
            "parameters_changed": False,
            "buffers_changed": False,
            "normalizer_changed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "agent_update_calls": 0,
            "before": before,
            "after": after,
        },
        "rows": rows,
    }


def serializable_reference_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Drop in-memory per-step series before emitting a formal artifact."""

    payload = copy.deepcopy(result)
    for row in payload["rows"]:
        row.pop("_metric_series", None)
    return payload


def serializable_reference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serializable_reference_result({"rows": [row]})["rows"][0] for row in rows]


def cluster_bootstrap_deltas(
    rows: list[dict[str, Any]],
    delta_name: str,
    *,
    seed: int = 4728,
    repetitions: int = 10_000,
) -> dict[str, float]:
    clusters: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        clusters[(row["source_round"], row["source_rollout"])].append(
            float(row[delta_name])
        )
    cluster_rows = list(clusters.values())
    if not cluster_rows:
        raise RuntimeError("Cluster bootstrap requires at least one paired motion.")
    rng = np.random.default_rng(seed)
    sample_means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        chosen = rng.integers(len(cluster_rows), size=len(cluster_rows))
        values = [value for cluster_index in chosen for value in cluster_rows[cluster_index]]
        sample_means[index] = np.mean(values)
    observed = float(np.mean([value for values in cluster_rows for value in values]))
    return {
        "mean_delta_trained_minus_fresh": observed,
        "ci95_low": float(np.quantile(sample_means, 0.025)),
        "ci95_high": float(np.quantile(sample_means, 0.975)),
        "clusters": len(cluster_rows),
        "bootstrap_repetitions": repetitions,
    }


def paired_reference_summary(
    fresh: Mapping[str, Any],
    trained: Mapping[str, Any],
    *,
    dataset: str,
) -> dict[str, Any]:
    fresh_rows = fresh["rows"]
    trained_rows = trained["rows"]
    trained_by_key = {row["motion_key"]: row for row in trained_rows}
    if set(trained_by_key) != {row["motion_key"] for row in fresh_rows}:
        raise RuntimeError("Paired reference motion keys do not match.")
    categories = Counter()
    phase_rows: dict[str, list[str]] = defaultdict(list)
    per_motion = []
    for fresh_row in fresh_rows:
        trained_row = trained_by_key[fresh_row["motion_key"]]
        for field in (
            "round",
            "rollout",
            "episode",
            "start_frame",
            "end_frame",
            "physics_seed",
        ):
            if fresh_row["source"][field] != trained_row["source"][field]:
                raise RuntimeError(
                    f"Paired source provenance mismatch for {fresh_row['motion_key']}: {field}"
                )
        t_common = min(int(fresh_row["t_exec"]), int(trained_row["t_exec"]))
        if t_common <= 0:
            raise RuntimeError("Paired reference motion has no common survival prefix.")
        fresh_complete = fresh_row["full_completion"]
        trained_complete = trained_row["full_completion"]
        category = (
            "both_complete" if fresh_complete and trained_complete
            else "fresh_only_complete" if fresh_complete
            else "trained_only_complete" if trained_complete
            else "both_incomplete"
        )
        categories[category] += 1
        if dataset == "phase":
            phase_rows[str(fresh_row["phase_label"])].append(fresh_row["motion_key"])
        item = {
            "motion_key": fresh_row["motion_key"],
            "source_round": fresh_row["source"]["round"],
            "source_rollout": fresh_row["source"]["rollout"],
            "category": category,
            "fresh_t_exec": int(fresh_row["t_exec"]),
            "trained_t_exec": int(trained_row["t_exec"]),
            "t_common": t_common,
            "completion_delta_trained_minus_fresh": (
                trained_row["completion_ratio"] - fresh_row["completion_ratio"]
            ),
        }
        for metric in REFERENCE_METRICS:
            fresh_value = float(np.mean(fresh_row["_metric_series"][metric][:t_common]))
            trained_value = float(np.mean(trained_row["_metric_series"][metric][:t_common]))
            item[f"{metric}_delta_trained_minus_fresh"] = trained_value - fresh_value
        per_motion.append(item)
    completion = cluster_bootstrap_deltas(
        per_motion, "completion_delta_trained_minus_fresh"
    )
    common_tracking = {
        metric: cluster_bootstrap_deltas(
            per_motion, f"{metric}_delta_trained_minus_fresh"
        )
        for metric in REFERENCE_METRICS
    }
    common_lengths = np.asarray([row["t_common"] for row in per_motion], dtype=np.int64)
    unequal = [
        {
            key: row[key]
            for key in ("motion_key", "fresh_t_exec", "trained_t_exec", "t_common")
        }
        for row in per_motion
        if row["fresh_t_exec"] != row["trained_t_exec"]
    ]
    phase_summary = {}
    if dataset == "phase":
        for phase, keys in sorted(phase_rows.items()):
            phase_pairs = [row for row in per_motion if row["motion_key"] in set(keys)]
            fresh_by_key = {row["motion_key"]: row for row in fresh_rows}
            trained_by_key = {row["motion_key"]: row for row in trained_rows}
            phase_fresh = [fresh_by_key[key] for key in keys]
            phase_trained = [trained_by_key[key] for key in keys]
            phase_summary[phase] = {
                "motion_count": len(keys),
                "source_rollout_count": len({
                    (row["source"]["round"], row["source"]["rollout"]) for row in phase_fresh
                }),
                "fresh_completion_mean": float(np.mean([row["completion_ratio"] for row in phase_fresh])),
                "trained_completion_mean": float(np.mean([row["completion_ratio"] for row in phase_trained])),
                "common_survival_joint_delta_trained_minus_fresh": float(np.mean([
                    row["joint_position_mae_rad_delta_trained_minus_fresh"] for row in phase_pairs
                ])),
                "common_survival_board_delta_trained_minus_fresh": float(np.mean([
                    row["board_xy_displacement_error_m_delta_trained_minus_fresh"] for row in phase_pairs
                ])),
                "common_survival_coupling_delta_trained_minus_fresh": float(np.mean([
                    row["coupling_xy_error_m_delta_trained_minus_fresh"] for row in phase_pairs
                ])),
            }
    return {
        "pair_count": len(per_motion),
        "pair_categories": dict(categories),
        "completion": {"clustered_paired_bootstrap": completion},
        "common_survival_tracking": common_tracking,
        "pair_survival_support": {
            "pair_count": len(per_motion),
            "equal_length_pair_count": len(per_motion) - len(unequal),
            "unequal_length_pair_count": len(unequal),
            "mean_t_common": float(common_lengths.mean()),
            "min_t_common": int(common_lengths.min()),
            "max_t_common": int(common_lengths.max()),
            "unequal_length_examples": unequal[:10],
        },
        "phase_breakdown": phase_summary,
        "per_motion": per_motion,
    }


def select_reference_videos(compare: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows = compare["per_motion"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    selection = []
    if by_category["trained_only_complete"]:
        selection.append(("trained_improves", min(
            by_category["trained_only_complete"],
            key=lambda row: row["joint_position_mae_rad_delta_trained_minus_fresh"],
        )["motion_key"]))
    elif rows:
        selection.append(("trained_improves", min(
            rows, key=lambda row: (
                row["joint_position_mae_rad_delta_trained_minus_fresh"]
                + row["board_xy_displacement_error_m_delta_trained_minus_fresh"]
                + row["coupling_xy_error_m_delta_trained_minus_fresh"]
            ),
        )["motion_key"]))
    if by_category["fresh_only_complete"]:
        selection.append(("fresh_improves", min(
            by_category["fresh_only_complete"],
            key=lambda row: row["completion_delta_trained_minus_fresh"],
        )["motion_key"]))
    elif rows:
        selection.append(("fresh_improves", max(
            rows, key=lambda row: (
                row["joint_position_mae_rad_delta_trained_minus_fresh"]
                + row["board_xy_displacement_error_m_delta_trained_minus_fresh"]
                + row["coupling_xy_error_m_delta_trained_minus_fresh"]
            ),
        )["motion_key"]))
    for category, label in (("both_complete", "both_complete"), ("both_incomplete", "both_fail")):
        if by_category[category]:
            selection.append((label, by_category[category][0]["motion_key"]))
    return selection


def reference_decision(phase: Mapping[str, Any], continuous: Mapping[str, Any]) -> str:
    core = (
        "joint_position_mae_rad",
        "board_xy_displacement_error_m",
        "coupling_xy_error_m",
    )
    blocks = [
        phase["common_survival_tracking"],
        continuous["common_survival_tracking"],
    ]
    improved = sum(
        block[name]["ci95_high"] < 0.0
        for block in blocks
        for name in core
    )
    regressed = sum(
        block[name]["ci95_low"] > 0.0
        for block in blocks
        for name in core
    )
    completion = [
        phase["completion"]["clustered_paired_bootstrap"],
        continuous["completion"]["clustered_paired_bootstrap"],
    ]
    if improved >= 2 and not any(item["ci95_high"] < 0.0 for item in completion) and regressed == 0:
        return "MOTION_REFERENCE_CLEAR_IMPROVEMENT"
    if regressed >= 2 and any(item["ci95_high"] < 0.0 for item in completion):
        return "MOTION_REFERENCE_REGRESSION"
    if improved == 0 and regressed == 0:
        return "MOTION_REFERENCE_NO_GAIN"
    return "MOTION_REFERENCE_MIXED"


def write_reference_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Validation Motion Reference Evaluation",
        "",
        "## Decision",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "## Validation Dataset",
        "",
        "| Dataset | Motions | Frames | Source rollouts | FPS | Motion SHA256 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, payload in report["datasets"].items():
        item = payload["dataset"]
        lines.append(
            f"| {name.title()} | {item['motion_count']:,} | {item['frame_count']:,} | "
            f"{item['source_rollout_count']} | {item['fps']:.0f} | `{item['motion_sha256']}` |"
        )
    lines += [
        "",
        "## Alignment Contract",
        "",
        "Each motion resets at local frame 0 using canonical raw `qpos/qvel` and source-realized physics. "
        "At control step `t`, `AlignedSkateTrackingContext.encode()` uses the future reference beginning at "
        "`t+1`; the post-step simulator state is compared to raw frame `source_start+t+1`. No state, board "
        "state, history, or expert action is injected after reset.",
        "",
        "## Paired Effects",
        "",
        "Values are trained checkpoint minus Fresh. Completion is better above zero; tracking errors are better below zero. "
        "95% CIs use 10,000 source-rollout-clustered paired bootstrap repetitions (seed 4728).",
        "",
        "| Metric | Phase delta [95% CI] | Continuous delta [95% CI] |",
        "| --- | --- | --- |",
    ]
    for metric in (
        "completion_ratio",
        "joint_position_mae_rad",
        "board_xy_displacement_error_m",
        "coupling_xy_error_m",
    ):
        values = []
        for name in ("phase", "continuous"):
            if name not in report["datasets"]:
                values.append("not run")
                continue
            comparison = report["datasets"][name]["comparison"]
            result = (
                comparison["completion"]["clustered_paired_bootstrap"]
                if metric == "completion_ratio"
                else comparison["common_survival_tracking"][metric]
            )
            values.append(
                f"{result['mean_delta_trained_minus_fresh']:.5g} "
                f"[{result['ci95_low']:.5g}, {result['ci95_high']:.5g}]"
            )
        lines.append(
            f"| {metric} | {values[0]} | {values[1]} |"
        )
    lines += [
        "",
        "## Execution",
        "",
        "Training performed: NO",
        "Test executed: NO",
        "Checkpoint path is explicit in the report; no training was performed.",
        "Paired tracking uses per-motion common survival prefixes.",
        "",
    ]
    path.write_text("\n".join(lines))


def run_motion_reference_evaluation(args: argparse.Namespace) -> int:
    if args.split != "val":
        raise ValueError("Test remains held out until evaluator protocol is frozen.")
    if args.reference_checkpoint is None:
        raise ValueError("--reference-checkpoint is required for mode=motion-reference.")
    output_dir = args.reference_output_dir.expanduser().resolve()
    official = args.official_checkpoint.expanduser().resolve()
    trained = args.reference_checkpoint.expanduser().resolve()
    if hash_file(checkpoint_model_path(official)) != OFFICIAL_BFM0_SHA256:
        raise RuntimeError("Official BFM0 checkpoint SHA256 mismatch.")
    trained_sha256 = hash_file(checkpoint_model_path(trained))

    resolver = ReferenceSourceResolver("val")
    if args.viewer:
        if args.reference_dataset == "both":
            raise ValueError("--viewer requires one --reference-dataset.")
        if args.checkpoint_view is None:
            raise ValueError("--viewer requires --checkpoint-view fresh|trained.")
        if args.video is not None:
            raise ValueError(
                "--video is not supported with multi-motion --viewer; run video separately."
            )
        records, _, motion_path, _ = load_reference_records(
            "val", args.reference_dataset
        )
        checkpoint_name = args.checkpoint_view
        checkpoint = {"fresh": official, "trained": trained}[checkpoint_name]
        agent, load_report = load_frozen_agent(checkpoint)
        before = checkpoint_mutation(agent)
        tracking = AlignedSkateTrackingContext.load(agent, motion_path)
        env = HuskyBfmOnlineEnv(viewer=True, realtime=True)
        try:
            completed = run_reference_viewer_sequence(
                agent,
                tracking,
                resolver,
                records,
                motion_key=args.motion_key,
                episodes=args.episodes,
                seed=args.seed,
                env=env,
                output_dir=output_dir,
            )
        finally:
            env.close()
        after = checkpoint_mutation(agent)
        if before != after:
            raise RuntimeError("Viewer sequence mutated a frozen checkpoint.")
        print(json.dumps({
            "viewer_sequence": "PASS",
            "episodes_completed": completed,
            "checkpoint_view": checkpoint_name,
            "checkpoint_model_sha256": hash_file(checkpoint_model_path(checkpoint)),
            "load_report": load_report,
            "output": str(output_dir / "viewer_sequence.json"),
        }, indent=2, sort_keys=True))
        return 0
    if args.motion_key is not None:
        if args.reference_dataset == "both":
            raise ValueError("--motion-key requires one --reference-dataset.")
        if args.checkpoint_view is None:
            raise ValueError("--motion-key requires --checkpoint-view.")
        records, _, motion_path, _ = load_reference_records(
            "val", args.reference_dataset
        )
        if args.motion_key not in records:
            raise KeyError(
                f"Motion key is not in Val {args.reference_dataset}: {args.motion_key}"
            )
        checkpoint_name = args.checkpoint_view
        checkpoint = {"fresh": official, "trained": trained}[checkpoint_name]
        agent, load_report = load_frozen_agent(checkpoint)
        before = checkpoint_mutation(agent)
        tracking = AlignedSkateTrackingContext.load(agent, motion_path)
        env = HuskyBfmOnlineEnv(viewer=args.viewer, realtime=args.viewer)
        try:
            row = run_reference_motion(
                agent,
                tracking,
                resolver,
                args.motion_key,
                records[args.motion_key],
                env=env,
                video_path=args.video,
            )
        finally:
            env.close()
        after = checkpoint_mutation(agent)
        if before != after:
            raise RuntimeError("Single-motion visualization mutated a frozen checkpoint.")
        payload = serializable_reference_rows([row])[0]
        payload["checkpoint_view"] = checkpoint_name
        payload["checkpoint_model_sha256"] = hash_file(checkpoint_model_path(checkpoint))
        payload["load_report"] = load_report
        payload["viewer"] = bool(args.viewer)
        payload["video_path"] = str(args.video) if args.video is not None else None
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.checkpoint_view is not None:
        raise ValueError("--checkpoint-view requires --motion-key.")
    if args.video is not None:
        raise ValueError("--video in motion-reference mode requires --motion-key.")

    selected = ("phase", "continuous") if args.reference_dataset == "both" else (args.reference_dataset,)
    datasets = {}
    for dataset in set((*selected, "phase")):
        records, manifest, motion_path, manifest_path = load_reference_records("val", dataset)
        datasets[dataset] = (records, manifest, motion_path, manifest_path)
    smoke_keys = []
    seen_sources = set()
    for key, record in sorted(datasets["phase"][0].items()):
        source = (str(record["source_round"]), str(record["source_rollout"]))
        if source not in seen_sources:
            smoke_keys.append(key)
            seen_sources.add(source)
        if len(smoke_keys) == 2:
            break
    if len(smoke_keys) != 2:
        raise RuntimeError("Validation smoke requires two phase motions from different sources.")

    estimates = {}
    for dataset in selected:
        records, _, motion_path, _ = datasets[dataset]
        agent, _ = load_frozen_agent(official)
        tracking = AlignedSkateTrackingContext.load(agent, motion_path)
        total = sum(
            min(
                int(np.asarray(record["dof"]).shape[0]) - 1,
                tracking.trajectories[key]["length"] - 1,
            )
            for key, record in records.items()
        )
        estimates[dataset] = total
        del agent
    total_transitions = 2 * sum(estimates.values())
    print(
        "Motion-reference estimate: "
        f"datasets={estimates}, total={total_transitions}"
    )
    if total_transitions > 1_000_000:
        raise RuntimeError("Reference evaluation exceeds 1,000,000 MuJoCo transitions.")

    smoke = {}
    for name, checkpoint in (("fresh", official), ("trained", trained)):
        agent, _ = load_frozen_agent(checkpoint)
        tracking = AlignedSkateTrackingContext.load(agent, datasets["phase"][2])
        before = checkpoint_mutation(agent)
        fresh_rows = [
            run_reference_motion(agent, tracking, resolver, key, datasets["phase"][0][key])
            for key in smoke_keys
        ]
        env = HuskyBfmOnlineEnv()
        try:
            rows = [
                run_reference_motion(
                    agent, tracking, resolver, key, datasets["phase"][0][key], env=env
                )
                for key in smoke_keys
            ]
        finally:
            env.close()
        after = checkpoint_mutation(agent)
        if before != after:
            raise RuntimeError("Smoke mutated a frozen checkpoint.")
        for fresh_row, reused_row in zip(fresh_rows, rows, strict=True):
            if (
                fresh_row["first_action_fingerprint"] != reused_row["first_action_fingerprint"]
                or fresh_row["terminated"] != reused_row["terminated"]
                or fresh_row["t_exec"] != reused_row["t_exec"]
                or not np.isclose(
                    fresh_row["metrics"]["joint_position_mae_rad"]["mean"],
                    reused_row["metrics"]["joint_position_mae_rad"]["mean"],
                    atol=1e-10,
                )
            ):
                raise RuntimeError("Reused reference environment diverges from fresh environment.")
        smoke[name] = {
            "rows": rows,
            "frame_difference_counts": tracking.frame_difference_counts(),
            "environment_reuse_equivalence": "PASS",
        }
    smoke_output = {
        "schema": "skate-bfm-motion-reference-smoke-v2",
        "motion_keys": smoke_keys,
        "source_rollouts": [
            [datasets["phase"][0][key]["source_round"], datasets["phase"][0][key]["source_rollout"]]
            for key in smoke_keys
        ],
        "fresh": {
            **smoke["fresh"],
            "rows": serializable_reference_rows(smoke["fresh"]["rows"]),
        },
        "trained": {
            **smoke["trained"],
            "rows": serializable_reference_rows(smoke["trained"]["rows"]),
        },
        "result": "PASS",
    }
    write_json(output_dir / "smoke.json", smoke_output)
    if args.smoke_only:
        print(f"Motion-reference smoke complete: {output_dir / 'smoke.json'}")
        return 0

    evaluations = {}
    runtime_results = {}
    for dataset in selected:
        records, manifest, motion_path, manifest_path = datasets[dataset]
        fresh = evaluate_reference_dataset("fresh", official, motion_path, records, resolver)
        trained_result = evaluate_reference_dataset("trained", trained, motion_path, records, resolver)
        comparison = paired_reference_summary(fresh, trained_result, dataset=dataset)
        runtime_results[dataset] = (fresh, trained_result)
        write_json(output_dir / f"{dataset}_fresh.json", serializable_reference_result(fresh))
        write_json(output_dir / f"{dataset}_trained.json", serializable_reference_result(trained_result))
        evaluations[dataset] = {
            "dataset": {
                "path": str(motion_path),
                "manifest_path": str(manifest_path),
                "motion_sha256": hash_file(motion_path),
                "manifest_sha256": hash_file(manifest_path),
                "motion_count": int(manifest["motion_count"]),
                "frame_count": int(manifest["frame_count"]),
                "source_rollout_count": int(manifest["source_rollout_count"]),
                "fps": float(manifest["fps"]),
            },
            "fresh": serializable_reference_result(fresh),
            "trained": serializable_reference_result(trained_result),
            "comparison": comparison,
        }
        write_json(output_dir / f"{dataset}_compare.json", comparison)

    audit_dataset = next(
        name
        for name in selected
        if evaluations[name]["comparison"]["pair_survival_support"]["unequal_length_pair_count"]
    )
    audit_pair = evaluations[audit_dataset]["comparison"]["pair_survival_support"]["unequal_length_examples"][0]
    fresh_rows = {row["motion_key"]: row for row in runtime_results[audit_dataset][0]["rows"]}
    trained_rows = {row["motion_key"]: row for row in runtime_results[audit_dataset][1]["rows"]}
    fresh_row = fresh_rows[audit_pair["motion_key"]]
    trained_row = trained_rows[audit_pair["motion_key"]]
    common = audit_pair["t_common"]
    manual_delta = float(
        np.mean(trained_row["_metric_series"]["joint_position_mae_rad"][:common])
        - np.mean(fresh_row["_metric_series"]["joint_position_mae_rad"][:common])
    )
    paired_delta = next(
        row["joint_position_mae_rad_delta_trained_minus_fresh"]
        for row in evaluations[audit_dataset]["comparison"]["per_motion"]
        if row["motion_key"] == audit_pair["motion_key"]
    )
    write_json(output_dir / "common_survival_audit.json", {
        **audit_pair,
        "metric": "joint_position_mae_rad",
        "manual_delta_trained_minus_fresh": manual_delta,
        "evaluator_delta_trained_minus_fresh": paired_delta,
        "match": bool(np.isclose(manual_delta, paired_delta, atol=1e-12)),
    })
    if not np.isclose(manual_delta, paired_delta, atol=1e-12):
        raise RuntimeError("Common-survival audit does not match paired summary.")

    videos = []
    video_dataset = "continuous" if "continuous" in evaluations else "phase"
    if not args.skip_videos:
        records, _, motion_path, _ = datasets[video_dataset]
        selected_videos = select_reference_videos(evaluations[video_dataset]["comparison"])
        for checkpoint_name, checkpoint in (("fresh", official), ("trained", trained)):
            agent, _ = load_frozen_agent(checkpoint)
            tracking = AlignedSkateTrackingContext.load(agent, motion_path)
            for label, key in selected_videos:
                path = output_dir / "videos" / f"{label}_{checkpoint_name}.mp4"
                run_reference_motion(agent, tracking, resolver, key, records[key], video_path=path)
                videos.append({"selection": label, "motion_key": key, "checkpoint": checkpoint_name, "path": str(path)})

    report = {
        "schema": "skate-bfm-motion-reference-v2",
        "repository": {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "origin_train": subprocess.check_output(["git", "rev-parse", "origin/train"], text=True).strip(),
            "evaluator_source_sha256": hash_file(Path(__file__)),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--short"], text=True).strip()),
        },
        "evaluation_mode": "motion-reference",
        "split": "val",
        "alignment_contract": {
            "reset": "local_frame=0 canonical raw qpos/qvel with source-realized physics",
            "reference_latent": "checkpoint-specific AlignedSkateTrackingContext.encode future_start=local_frame+1",
            "actual_state": "free HUSKY MuJoCo post-step state",
            "metric_reference_frame": "source_start_frame+t+1",
            "teacher_forcing": False,
            "source_action_used": False,
        },
        "checkpoints": {
            "fresh": {"path": str(official), "model_sha256": OFFICIAL_BFM0_SHA256},
            "trained": {"path": str(trained), "model_sha256": trained_sha256},
        },
        "estimate_transitions": estimates,
        "smoke": smoke_output,
        "datasets": evaluations,
        "videos": videos,
        "classification": (
            reference_decision(evaluations["phase"]["comparison"], evaluations["continuous"]["comparison"])
            if set(evaluations) == {"phase", "continuous"} else "NOT_CLASSIFIED_PARTIAL_DATASET"
        ),
        "test_executed": False,
        "training_performed": False,
        "production_training_control_modified": False,
    }
    write_json(output_dir / "report.json", report)
    write_reference_report(report, output_dir / "report.md")
    print(f"Validation motion-reference complete: {report['classification']}")
    return 0


def run_rollout(
    agent,
    *,
    condition: dict[str, Any],
    z: torch.Tensor,
    latent_kind: str,
    random_seed: int | None,
    horizon: int,
    control_dt: float,
    initial_reference: dict[str, tuple[str, str, str]],
    checkpoint_fingerprints: dict[str, Any],
) -> dict[str, Any]:
    random.seed(condition["rollout_seed"])
    np.random.seed(condition["rollout_seed"])
    torch.manual_seed(condition["rollout_seed"])
    env = HuskyBfmOnlineEnv(control_dt=control_dt)
    dynamics_report, joint_offsets = randomize_husky_play_physics(
        env.env.model,
        condition["rollout_id"],
        condition["dynamics_seed"],
    )
    env.env.set_reset_joint_offsets(joint_offsets)
    mujoco.mj_setConst(env.env.model, env.env.data)
    observation = env.reset()
    initial_raw = env.env._observation()
    initial_fingerprints = (
        hash_data(observation),
        hash_data(initial_state_payload(initial_raw)["root"]),
        hash_data(initial_state_payload(initial_raw)["board"]),
    )
    reference_key = condition["rollout_id"]
    if reference_key in initial_reference:
        if initial_reference[reference_key] != initial_fingerprints:
            env.close()
            raise RuntimeError(
                f"Canonical reset mismatch for {reference_key} and {latent_kind}."
            )
    else:
        initial_reference[reference_key] = initial_fingerprints

    if agent._model.training:
        env.close()
        raise RuntimeError("Frozen Actor model is not in eval mode.")
    records = []
    first_action_fingerprint = None
    terminated = False
    truncated = False
    try:
        for step in range(horizon):
            model_observation = {
                key: value.unsqueeze(0).to(agent.device)
                for key, value in observation.items()
            }
            with torch.no_grad():
                action = agent.act(
                    obs=model_observation,
                    z=z.unsqueeze(0),
                    mean=True,
                )[0]
            if first_action_fingerprint is None:
                first_action_fingerprint = hash_data(action)
            transition = env.step(
                action,
                z,
                truncated=step == horizon - 1,
            )
            records.append(dict(transition.raw_metadata))
            observation = transition.next_observation
            terminated = transition.terminated
            truncated = transition.truncated
            if terminated:
                break
    finally:
        env.close()

    payload = {
        **condition,
        "latent_kind": latent_kind,
        "random_seed": random_seed,
        "z_fingerprint": hash_data(z),
        "z_norm": float(torch.linalg.vector_norm(z)),
        "dynamics_realization": dynamics_report,
        "dynamics_fingerprint": hash_data(dynamics_report),
        "initial_fingerprints": {
            "observation": initial_fingerprints[0],
            "root_state": initial_fingerprints[1],
            "board_state": initial_fingerprints[2],
        },
        "first_action_fingerprint": first_action_fingerprint,
        "transition_fingerprint": hash_data(records),
        "episode": {
            "transition_count": len(records),
            "terminated": terminated,
            "truncated": truncated,
            "time_to_fall_s": (
                len(records) * control_dt if terminated else None
            ),
        },
        "command_injected_into_actor": False,
        "metrics": physical_metrics(records, initial_raw, control_dt),
        "mutation": {
            "parameters_changed": False,
            "buffers_changed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "agent_update_calls": 0,
            "update_fb_calls": 0,
            "before_parameter_fingerprint": checkpoint_fingerprints[
                "parameters"
            ],
            "before_buffer_fingerprint": checkpoint_fingerprints["buffers"],
            "before_component_fingerprints": checkpoint_fingerprints[
                "components"
            ],
        },
    }
    return payload


def eval_checkpoint(
    agent,
    *,
    checkpoint_name: str,
    checkpoint_path: Path,
    checkpoint_model_sha: str,
    target: dict[str, Any],
    target_bank_sha: str,
    target_observations: dict[str, torch.Tensor],
    protocol: dict[str, Any],
    initial_reference: dict[str, tuple[str, str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate random and target latents for one frozen checkpoint."""

    target_start = target["frame_start"]
    target_z = encode_target(agent, target_observations)
    target_z_tensor = torch.from_numpy(target_z).to(agent.device)
    target_fingerprint = hash_data(target_z)
    expected_record = target.get("latents", {}).get(checkpoint_name)
    if expected_record is not None:
        expected_fingerprint = expected_record["latent"]["fingerprint"]
        if target_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"Runtime target latent mismatch for {checkpoint_name}: "
                f"expected={expected_fingerprint}, actual={target_fingerprint}."
            )

    before = {
        "parameters": hash_params(agent._model),
        "buffers": hash_buffers(agent._model),
        "components": hash_components(agent),
    }
    provenance = {
        "checkpoint_name": checkpoint_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_model_sha256": checkpoint_model_sha,
        "target_bank_sha256": target_bank_sha,
        "target_id": target["target_id"],
        "target_frame_start": target_start,
        "target_frame_end_inclusive": target["frame_end_inclusive"],
    }
    rollouts = []
    for condition in protocol["rollouts"]:
        for random_seed in DEFAULT_RANDOM_SEEDS:
            torch.manual_seed(random_seed)
            random_z = agent._model.sample_z(1, device=agent.device)[0]
            row = run_rollout(
                agent,
                condition=condition,
                z=random_z,
                latent_kind="random",
                random_seed=random_seed,
                horizon=protocol["rollout_horizon"],
                control_dt=protocol["control_dt_s"],
                initial_reference=initial_reference,
                checkpoint_fingerprints=before,
            )
            row.update(provenance)
            rollouts.append(row)
        row = run_rollout(
            agent,
            condition=condition,
            z=target_z_tensor,
            latent_kind="target",
            random_seed=None,
            horizon=protocol["rollout_horizon"],
            control_dt=protocol["control_dt_s"],
            initial_reference=initial_reference,
            checkpoint_fingerprints=before,
        )
        row.update(provenance)
        rollouts.append(row)

    after = {
        "parameters": hash_params(agent._model),
        "buffers": hash_buffers(agent._model),
        "components": hash_components(agent),
    }
    parameters_changed = before["parameters"] != after["parameters"]
    buffers_changed = before["buffers"] != after["buffers"]
    components_changed = before["components"] != after["components"]
    if parameters_changed or components_changed:
        raise RuntimeError(f"Frozen model parameters changed for {checkpoint_name}.")
    if buffers_changed:
        raise RuntimeError(
            f"Normalizer/model buffers changed for {checkpoint_name}."
        )
    for row in rollouts:
        row["mutation"].update(
            {
                "parameters_changed": parameters_changed,
                "buffers_changed": buffers_changed,
                "components_changed": components_changed,
                "after_parameter_fingerprint": after["parameters"],
                "after_buffer_fingerprint": after["buffers"],
                "after_component_fingerprints": after["components"],
            }
        )

    aggregates = {}
    for split in ("seen", "unseen"):
        groups = {}
        for latent_kind in ("random", "target"):
            rows = [
                row["metrics"]
                for row in rollouts
                if row["dynamics_split"] == split
                and row["latent_kind"] == latent_kind
            ]
            groups[latent_kind] = aggregate(rows)
        groups["target_advantage"] = {
            name: {
                "delta_target_minus_random_mean": (
                    groups["target"][name]["mean"]
                    - groups["random"][name]["mean"]
                )
            }
            for name in METRIC_NAMES
        }
        aggregates[split] = groups

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_model_sha256": checkpoint_model_sha,
        "target_z_fingerprint": target_fingerprint,
        "target_z_norm": float(np.linalg.norm(target_z)),
        "target_source": {
            "target_id": target["target_id"],
            "frame_start": target_start,
            "frame_end_inclusive": target["frame_end_inclusive"],
            "latent_source": "runtime raw target window via MotionLib",
        },
        "random_seeds": list(DEFAULT_RANDOM_SEEDS),
        "aggregates_by_split": aggregates,
        "rollout_count": len(rollouts),
        "inference_only": True,
        "mutation": {
            "parameters_changed": parameters_changed,
            "buffers_changed": buffers_changed,
            "components_changed": components_changed,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "agent_update_calls": 0,
            "update_fb_calls": 0,
            "before": before,
            "after": after,
        },
    }
    return rollouts, report


def run_fixed_target_evaluation(args: argparse.Namespace) -> int:
    """Run the historical M2.5b fixed target-conditioned protocol."""

    if (
        args.checkpoint_10k is None
        or args.checkpoint_20k is None
        or args.target_bank is None
        or args.expert_motion is None
    ):
        raise ValueError(
            "--checkpoint-10k, --checkpoint-20k, --target-bank, and "
            "--expert-motion are required for mode=fixed-target."
        )
    target_bank_path = args.target_bank.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    expert_motion = args.expert_motion.expanduser().resolve()
    target_bank, target_bank_sha = load_and_validate_target_bank(
        target_bank_path
    )
    protocol = load_protocol(protocol_path)
    if not expert_motion.is_file():
        raise FileNotFoundError(expert_motion)
    source = target_bank["source"]
    if source["expert_motion_sha256"] != hash_file(expert_motion):
        raise RuntimeError("Expert MotionLib SHA256 does not match target bank.")

    paths = checkpoint_paths(
        official_checkpoint=args.official_checkpoint.expanduser().resolve(),
        checkpoint_10k=args.checkpoint_10k.expanduser().resolve(),
        checkpoint_20k=args.checkpoint_20k.expanduser().resolve(),
    )
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Required frozen checkpoints missing: {missing}")
    if hash_file(checkpoint_model_path(paths["official_bfm0"])) != OFFICIAL_BFM0_SHA256:
        raise RuntimeError("Official BFM0 checkpoint SHA256 mismatch.")

    target = target_bank["targets"][0]
    target_start = target["frame_start"]
    target_end = target["frame_end_inclusive"] + 1
    all_rollouts: list[dict[str, Any]] = []
    initial_reference: dict[str, tuple[str, str, str]] = {}
    checkpoint_reports: dict[str, Any] = {}
    first_name, first_path = next(iter(paths.items()))
    first_agent, _ = load_frozen_agent(first_path)
    observations = load_expert(first_agent, expert_motion)
    target_observations = {
        key: value[target_start:target_end].clone()
        for key, value in observations.items()
    }

    for checkpoint_name, checkpoint_path in paths.items():
        if checkpoint_name == first_name:
            agent = first_agent
            first_agent = None
        else:
            agent, _ = load_frozen_agent(checkpoint_path)
        model_sha = hash_file(checkpoint_model_path(checkpoint_path))
        rollouts, report = eval_checkpoint(
            agent,
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            checkpoint_model_sha=model_sha,
            target=target,
            target_bank_sha=target_bank_sha,
            target_observations=target_observations,
            protocol=protocol,
            initial_reference=initial_reference,
        )
        all_rollouts.extend(rollouts)
        checkpoint_reports[checkpoint_name] = report
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "schema": "skate-bfm-target-conditioned-eval-v1",
        "evaluation": {
            "date": date.today().isoformat(),
            "evaluation_only": True,
            "training_performed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "agent_update_calls": 0,
            "actor_update": False,
            "normalizer_mutation": False,
            "target_bank": str(target_bank_path),
            "target_bank_sha256": target_bank_sha,
            "target_id": target["target_id"],
            "target_source_identity": {
                "raw_rollout_sha256": source["raw_rollout_sha256"],
                "motion_key": source["motion_key"],
                "frame_start": target_start,
                "frame_end_inclusive": target["frame_end_inclusive"],
            },
            "protocol": str(protocol_path),
            "protocol_version": protocol["evaluator_version"],
            "horizon": protocol["rollout_horizon"],
            "control_dt_s": protocol["control_dt_s"],
            "command_injected_into_actor": False,
            "reset_semantics": protocol["reset_condition"],
            "random_seeds": list(DEFAULT_RANDOM_SEEDS),
            "target_latent_fixed_per_rollout": True,
            "target_window_reset_teleport": False,
        },
        "checkpoint_reports": checkpoint_reports,
        "rollouts": all_rollouts,
    }
    output_dir = args.output_dir.expanduser().resolve()
    evaluation_path = output_dir / "target_conditioned_metrics.json"
    write_json(evaluation_path, output)
    if args.training_summary is not None:
        update_training_summary(
            args.training_summary.expanduser().resolve(),
            evaluation_path=evaluation_path,
            evaluation=output,
        )
    print(
        f"Target-conditioned evaluation complete: {len(all_rollouts)} rollouts, "
        "training=0, optimizer_steps=0, normalizer_mutation=False"
    )
    print(f"Artifacts: {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "rollout":
        return run_frozen_evaluation(args)
    if args.mode == "motion-reference":
        return run_motion_reference_evaluation(args)
    return run_fixed_target_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
