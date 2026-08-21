#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Reproduce 5-second expert push/steer transitions with one frozen checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path[:0] = [
    str(REPOSITORY_ROOT / "src"),
    str(REPOSITORY_ROOT / "husky_sim" / "src"),
    str(SCRIPT_DIRECTORY),
]

from skate_bfm.integration import HuskyBfmOnlineEnv
from train_runner import (
    checkpoint_model_path,
    hash_buffers,
    hash_components,
    hash_data,
    hash_file,
    hash_params,
)
from train_skate_bfm import AlignedSkateTrackingContext, load_frozen_agent


CONTROL_DT = 0.02
FPS = 50
DURATION_S = 5.0
LEAD_S = 1.0
PHASE_NAMES = {
    0: "push",
    1: "push2steer",
    2: "steer_left",
    3: "steer_right",
    4: "steer_forward",
    5: "steer2push",
    6: "fall",
}
STEER_PHASES = frozenset(("steer_left", "steer_forward", "steer_right"))
PRIMARY_METRICS = (
    "joint_position_mae_rad",
    "joint_velocity_mae_rad_s",
    "root_orientation_geodesic_error_deg",
    "board_xy_displacement_error_m",
    "board_heading_error_deg",
    "coupling_xy_error_m",
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument(
        "--transition",
        choices=("push2steer", "steer2push", "both"),
        default="both",
    )
    parser.add_argument("--samples-per-transition", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--lead-s", type=float, default=LEAD_S)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--case-spec", type=Path)
    parser.add_argument("--motion-key")
    parser.add_argument("--local-frame", type=int)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--checkpoint-sha256")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def board_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def board_tilt_deg(quaternion: np.ndarray) -> float:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise RuntimeError("Board quaternion must be finite wxyz.")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise RuntimeError("Board quaternion has zero norm.")
    _, x, y, _ = quaternion / norm
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1, 1))))


def quaternion_geodesic_deg(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    actual_norm = np.linalg.norm(actual, axis=-1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=-1, keepdims=True)
    if (actual_norm <= 1e-12).any() or (reference_norm <= 1e-12).any():
        raise RuntimeError("Quaternion metric received a zero-norm quaternion.")
    actual = actual / actual_norm
    reference = reference / reference_norm
    dot = np.clip(np.abs(np.sum(actual * reference, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def wrapped_angle_deg(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.degrees(np.abs(np.angle(np.exp(1j * (actual - reference)))))


def summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Metric values must be finite and non-empty.")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


class ReferenceSourceResolver:
    """Load canonical validation raw sources with strict provenance checks."""

    def __init__(self) -> None:
        self.raw_root = (REPOSITORY_ROOT / "train/dataset/sim_collected/val/raw").resolve()
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"Validation raw root not found: {self.raw_root}")
        self.cache: dict[Path, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}

    def load(self, record: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
        required = (
            "source_raw_npz",
            "source_round",
            "source_rollout",
            "source_episode",
            "source_start_frame",
            "source_end_frame",
            "physics_seed",
            "dataset_split",
        )
        if any(name not in record for name in required):
            raise RuntimeError("Continuous MotionLibrary record lacks raw provenance.")
        if record["dataset_split"] != "validation":
            raise RuntimeError("Transition evaluator requires validation provenance.")
        round_id = str(record["source_round"]).zfill(3)
        rollout_id = str(record["source_rollout"]).zfill(3)
        source_root = self.raw_root / f"round_{round_id}" / f"rollout_{rollout_id}" / "raw_rollout"
        paths = sorted(source_root.glob("*.npz"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one canonical source under {source_root}.")
        path = paths[0].resolve()
        if path.name != Path(str(record["source_raw_npz"])).name:
            raise RuntimeError("MotionLibrary raw filename disagrees with validation source.")
        if path not in self.cache:
            metadata_path = path.with_suffix(".json")
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Canonical raw metadata missing: {metadata_path}")
            metadata = json.loads(metadata_path.read_text())
            expected = {
                "dataset_split": "validation",
                "round_id": round_id,
                "rollout_id": rollout_id,
                "episode_id": str(record["source_episode"]),
            }
            actual = {name: str(metadata.get(name, "")) for name in expected}
            if actual != expected:
                raise RuntimeError(f"Raw provenance mismatch: {actual} != {expected}")
            physics = metadata.get("physics_randomization")
            if not isinstance(physics, dict) or int(physics.get("seed", -1)) != int(
                record["physics_seed"]
            ):
                raise RuntimeError("Canonical source physics is missing or has the wrong seed.")
            with np.load(path, allow_pickle=False) as archive:
                names = (
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
                state = {name: np.asarray(archive[name]).copy() for name in names}
            frames = state["qpos"].shape[0]
            if (
                state["qpos"].ndim != 2
                or state["qvel"].ndim != 2
                or any(value.shape[0] != frames for value in state.values())
                or not all(np.isfinite(value).all() for value in state.values())
            ):
                raise RuntimeError(f"Malformed canonical raw source: {path}")
            self.cache[path] = (state, copy.deepcopy(metadata))
        state, metadata = self.cache[path]
        return state, metadata, path


def continuous_dataset_paths() -> tuple[Path, Path]:
    root = REPOSITORY_ROOT / "train/dataset/sim_collected/val/continuous/motion_library"
    return root / "skate_expert_continuous.pkl", root / "manifest.json"


def load_continuous_records() -> tuple[dict[str, dict[str, Any]], dict[str, Any], Path, Path]:
    motion_path, manifest_path = continuous_dataset_paths()
    if not motion_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Val Continuous MotionLibrary or manifest is missing.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_split") != "validation":
        raise RuntimeError("Val Continuous manifest split must be validation.")
    records = joblib.load(motion_path)
    if not isinstance(records, dict) or not records:
        raise RuntimeError("Val Continuous MotionLibrary must be a non-empty mapping.")
    required = ("dof", "phase_id", "source_start_frame", "source_end_frame")
    normalized = {str(key): value for key, value in records.items()}
    for key, record in normalized.items():
        if any(name not in record for name in required):
            raise RuntimeError(f"{key} lacks canonical phase or frame provenance.")
        phase_ids = np.asarray(record["phase_id"])
        frame_count = int(np.asarray(record["dof"]).shape[0])
        if (
            phase_ids.shape != (frame_count,)
            or not np.issubdtype(phase_ids.dtype, np.integer)
            or not set(phase_ids.tolist()) <= set(PHASE_NAMES)
        ):
            raise RuntimeError(f"{key} has invalid canonical phase_id annotation.")
        if int(record["source_end_frame"]) - int(record["source_start_frame"]) != frame_count:
            raise RuntimeError(f"{key} source frame range disagrees with motion length.")
    return normalized, manifest, motion_path, manifest_path


def phase_runs(phase_ids: np.ndarray) -> list[tuple[str, int, int]]:
    labels = [PHASE_NAMES[int(value)] for value in np.asarray(phase_ids)]
    runs: list[tuple[str, int, int]] = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            runs.append((labels[start], start, end))
            start = end
    return runs


def detect_transition_candidates(
    records: Mapping[str, Mapping[str, Any]], lead_steps: int, duration_steps: int
) -> dict[str, list[dict[str, Any]]]:
    """Find complete contiguous canonical phase triples, not individual frames."""

    candidates = {"push2steer": [], "steer2push": []}
    for motion_key, record in sorted(records.items()):
        runs = phase_runs(np.asarray(record["phase_id"], dtype=np.int16))
        frames = int(np.asarray(record["phase_id"]).size)
        for before, transition, after in zip(runs, runs[1:], runs[2:]):
            pre_phase, _, _ = before
            transition_phase, start, end = transition
            post_phase, _, _ = after
            kind = None
            if (
                pre_phase == "push"
                and transition_phase == "push2steer"
                and post_phase in STEER_PHASES
            ):
                kind = "push2steer"
            if (
                pre_phase in STEER_PHASES
                and transition_phase == "steer2push"
                and post_phase == "push"
            ):
                kind = "steer2push"
            if kind is None:
                continue
            reset = start - lead_steps
            if reset < 0 or reset + duration_steps >= frames:
                continue
            candidates[kind].append(
                {
                    "transition": kind,
                    "motion_key": motion_key,
                    "reset_local_frame": reset,
                    "transition_local_start": start,
                    "transition_local_end": end,
                    "pre_phase": pre_phase,
                    "post_phase": post_phase,
                    "source_round": str(record["source_round"]),
                    "source_rollout": str(record["source_rollout"]),
                    "source_episode": str(record["source_episode"]),
                    "physics_seed": int(record["physics_seed"]),
                }
            )
    if not all(candidates.values()):
        raise RuntimeError("No eligible canonical transition instances were found.")
    return candidates


def selected_transition_names(requested: str) -> tuple[str, ...]:
    return ("push2steer", "steer2push") if requested == "both" else (requested,)


def case_payload(candidate: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        **{
            key: candidate[key]
            for key in (
                "transition",
                "motion_key",
                "reset_local_frame",
                "transition_local_start",
                "transition_local_end",
                "pre_phase",
                "post_phase",
                "source_round",
                "source_rollout",
                "source_episode",
                "physics_seed",
            )
        },
    }


def validate_case(
    case: Mapping[str, Any],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    duration_steps: int,
) -> dict[str, Any]:
    required = {
        "case_id",
        "transition",
        "motion_key",
        "reset_local_frame",
        "transition_local_start",
        "transition_local_end",
        "pre_phase",
        "post_phase",
        "source_round",
        "source_rollout",
        "source_episode",
        "physics_seed",
    }
    if set(case) < required:
        raise RuntimeError("Case specification is missing required transition provenance.")
    transition = str(case["transition"])
    if transition not in candidates:
        raise RuntimeError(f"Unknown transition in case specification: {transition}")
    identity = (
        str(case["motion_key"]),
        int(case["reset_local_frame"]),
        int(case["transition_local_start"]),
        int(case["transition_local_end"]),
    )
    matches = [
        item
        for item in candidates[transition]
        if (
            item["motion_key"],
            item["reset_local_frame"],
            item["transition_local_start"],
            item["transition_local_end"],
        )
        == identity
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Case does not match one eligible canonical transition: {identity}")
    checked = case_payload(matches[0], str(case["case_id"]))
    if any(str(checked[key]) != str(case[key]) for key in checked if key != "case_id"):
        raise RuntimeError(
            f"Case provenance differs from the canonical candidate: {case['case_id']}"
        )
    return checked


def select_cases(
    args: argparse.Namespace,
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    records: Mapping[str, Mapping[str, Any]],
    duration_steps: int,
    lead_steps: int,
) -> list[dict[str, Any]]:
    if args.case_spec and (args.motion_key or args.local_frame is not None):
        raise ValueError("--case-spec cannot be combined with --motion-key/--local-frame.")
    if (args.motion_key is None) != (args.local_frame is None):
        raise ValueError("--motion-key and --local-frame must be provided together.")
    if args.case_spec:
        payload = json.loads(args.case_spec.expanduser().resolve().read_text())
        if int(payload.get("duration_steps", duration_steps)) != duration_steps:
            raise RuntimeError("Case specification duration does not match 250-step protocol.")
        if int(payload.get("lead_steps", lead_steps)) != lead_steps:
            raise RuntimeError("Case specification lead does not match 50-step protocol.")
        cases = [
            validate_case(case, candidates, duration_steps) for case in payload.get("cases", [])
        ]
        if not cases:
            raise RuntimeError("Case specification contains no cases.")
        return cases
    if args.motion_key is not None:
        if args.motion_key not in records:
            raise KeyError(f"Motion key is not in Val Continuous: {args.motion_key}")
        requested = selected_transition_names(args.transition)
        start = int(args.local_frame)
        matching = [
            candidate
            for name in requested
            for candidate in candidates[name]
            if candidate["motion_key"] == args.motion_key
            and candidate["reset_local_frame"] == start
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "Requested reset frame does not contain the requested eligible transition."
            )
        return [case_payload(matching[0], "manual_000")]
    if args.samples_per_transition <= 0:
        raise ValueError("--samples-per-transition must be positive.")
    rng = np.random.default_rng(args.seed)
    cases = []
    for name in selected_transition_names(args.transition):
        choices = candidates[name]
        if args.samples_per_transition > len(choices):
            raise RuntimeError(f"Requested too many {name} cases: {len(choices)} available.")
        for index in rng.choice(len(choices), size=args.samples_per_transition, replace=False):
            cases.append(
                case_payload(
                    choices[int(index)],
                    f"{name}_{len([c for c in cases if c['transition'] == name]):03d}",
                )
            )
    return cases


class ControlDiagnostics:
    """Actual 23-DoF action saturation and generalized torque utilization."""

    groups = ("waist", "hip", "ankle")

    def __init__(self, env: HuskyBfmOnlineEnv) -> None:
        report = env.env.physical_actuator_report
        if len(report) != 23:
            raise RuntimeError("Expected all 23 physical HUSKY actuators.")
        self.names = tuple(str(item["joint_name"]).removeprefix("robot/") for item in report)
        self.limits = np.asarray(
            [item["derived_joint_torque_limit"] for item in report], dtype=np.float64
        )
        if (
            self.limits.shape != (23,)
            or not np.isfinite(self.limits).all()
            or (self.limits <= 0).any()
        ):
            raise RuntimeError("HUSKY torque constraints are invalid.")
        self.dof_addresses = np.asarray(
            [
                int(env.env.model.joint(int(env.env.model.actuator_trnid[index, 0])).dofadr[0])
                for index in range(23)
            ],
            dtype=np.int32,
        )
        if len(set(self.dof_addresses)) != 23:
            raise RuntimeError("HUSKY actuator/joint mapping is ambiguous.")
        self.indices = {
            group: np.asarray(
                [index for index, name in enumerate(self.names) if group in name], dtype=np.int32
            )
            for group in self.groups
        }
        if any(not value.size for value in self.indices.values()):
            raise RuntimeError("Waist, hip, and ankle diagnostics require non-empty joint groups.")
        self.actions: list[np.ndarray] = []
        self.utilizations: list[np.ndarray] = []

    def update(self, action_husky: torch.Tensor, qfrc_actuator: np.ndarray) -> None:
        action = np.asarray(action_husky.detach().cpu(), dtype=np.float64)
        torque = np.asarray(qfrc_actuator, dtype=np.float64)[self.dof_addresses]
        if (
            action.shape != (23,)
            or torque.shape != (23,)
            or not np.isfinite(action).all()
            or not np.isfinite(torque).all()
        ):
            raise RuntimeError("Non-finite or malformed physical control diagnostic.")
        self.actions.append(np.abs(action))
        self.utilizations.append(np.abs(torque) / self.limits)

    def summary(self) -> dict[str, Any]:
        actions = np.concatenate(self.actions)
        utilization = np.concatenate(self.utilizations)
        result: dict[str, Any] = {
            "actuator_joint_names": list(self.names),
            "action_soft_saturation_fraction": float(np.mean(actions >= 0.95)),
            "action_hard_saturation_fraction": float(np.mean(actions >= 0.99)),
            "torque_utilization_mean": float(utilization.mean()),
            "torque_utilization_p95": float(np.quantile(utilization, 0.95)),
            "torque_utilization_p99": float(np.quantile(utilization, 0.99)),
            "torque_utilization_max": float(utilization.max()),
            "torque_utilization_ge_0_95_fraction": float(np.mean(utilization >= 0.95)),
            "groups": {},
        }
        action_rows = np.asarray(self.actions)
        torque_rows = np.asarray(self.utilizations)
        for group, indices in self.indices.items():
            group_actions = action_rows[:, indices].reshape(-1)
            group_torque = torque_rows[:, indices].reshape(-1)
            result["groups"][group] = {
                "joint_names": [self.names[index] for index in indices],
                "action_soft_saturation_fraction": float(np.mean(group_actions >= 0.95)),
                "action_hard_saturation_fraction": float(np.mean(group_actions >= 0.99)),
                "torque_utilization_p95": float(np.quantile(group_torque, 0.95)),
                "torque_utilization_p99": float(np.quantile(group_torque, 0.99)),
                "torque_utilization_ge_0_95_fraction": float(np.mean(group_torque >= 0.95)),
            }
        return result


def checkpoint_mutation(agent: Any) -> dict[str, Any]:
    return {
        "parameters": hash_params(agent._model),
        "buffers": hash_buffers(agent._model),
        "normalizer": hash_buffers(agent._model._obs_normalizer),
        "components": hash_components(agent),
    }


def reference_metric_series(
    actual: Sequence[Mapping[str, Any]], raw: Mapping[str, np.ndarray], reset_frame: int
) -> dict[str, np.ndarray]:
    steps = len(actual)
    reference = slice(reset_frame + 1, reset_frame + steps + 1)
    if reset_frame < 0 or reset_frame + steps >= raw["qpos"].shape[0]:
        raise RuntimeError("Post-step expert reference is outside canonical raw data.")
    actual_joint_pos = np.asarray([row["joint_position"] for row in actual])
    actual_joint_vel = np.asarray([row["joint_velocity"] for row in actual])
    actual_root_pos = np.asarray([row["root_position"] for row in actual])
    actual_root_quat = np.asarray([row["root_quaternion"] for row in actual])
    actual_board_pos = np.asarray([row["board_position"] for row in actual])
    actual_board_quat = np.asarray([row["board_quaternion"] for row in actual])
    actual_board_lin = np.asarray([row["board_linear_velocity"] for row in actual])
    actual_board_ang = np.asarray([row["board_angular_velocity"] for row in actual])
    ref_root = raw["root_pos"][reference]
    ref_board = raw["board_root_pos"][reference]
    return {
        "joint_position_mae_rad": np.mean(
            np.abs(actual_joint_pos - raw["dof_pos"][reference]), axis=1
        ),
        "joint_velocity_mae_rad_s": np.mean(
            np.abs(actual_joint_vel - raw["dof_vel"][reference]), axis=1
        ),
        "root_xy_displacement_error_m": np.linalg.norm(
            (actual_root_pos[:, :2] - raw["root_pos"][reset_frame, :2])
            - (ref_root[:, :2] - raw["root_pos"][reset_frame, :2]),
            axis=1,
        ),
        "root_z_error_m": np.abs(actual_root_pos[:, 2] - ref_root[:, 2]),
        "root_orientation_geodesic_error_deg": quaternion_geodesic_deg(
            actual_root_quat, raw["root_quat"][reference]
        ),
        "board_xy_displacement_error_m": np.linalg.norm(
            (actual_board_pos[:, :2] - raw["board_root_pos"][reset_frame, :2])
            - (ref_board[:, :2] - raw["board_root_pos"][reset_frame, :2]),
            axis=1,
        ),
        "board_orientation_geodesic_error_deg": quaternion_geodesic_deg(
            actual_board_quat, raw["board_root_quat"][reference]
        ),
        "board_linear_velocity_error_mps": np.linalg.norm(
            actual_board_lin - raw["board_root_lin_vel"][reference], axis=1
        ),
        "board_angular_velocity_error_rad_s": np.linalg.norm(
            actual_board_ang - raw["board_root_ang_vel"][reference], axis=1
        ),
        "board_heading_error_deg": wrapped_angle_deg(
            np.asarray([board_yaw(value) for value in actual_board_quat]),
            np.asarray([board_yaw(value) for value in raw["board_root_quat"][reference]]),
        ),
        "board_tilt_error_deg": np.abs(
            np.asarray([board_tilt_deg(value) for value in actual_board_quat])
            - np.asarray([board_tilt_deg(value) for value in raw["board_root_quat"][reference]]),
        ),
        "coupling_xy_error_m": np.linalg.norm(
            (actual_root_pos[:, :2] - actual_board_pos[:, :2])
            - (ref_root[:, :2] - ref_board[:, :2]),
            axis=1,
        ),
        "coupling_z_error_m": np.abs(
            (actual_root_pos[:, 2] - actual_board_pos[:, 2]) - (ref_root[:, 2] - ref_board[:, 2])
        ),
    }


def section_metrics(
    series: Mapping[str, np.ndarray], labels: Sequence[str], phase: str
) -> dict[str, dict[str, float]] | None:
    indices = np.asarray(
        [index for index, label in enumerate(labels) if label == phase], dtype=np.int64
    )
    if not indices.size:
        return None
    return {name: summarize_values(np.asarray(values)[indices]) for name, values in series.items()}


def retention_and_stability(
    actual: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    feet = np.asarray([bool(row["feet_on_board"]) for row in actual], dtype=bool)
    illegal = np.asarray([bool(row["illegal_contact"]) for row in actual], dtype=bool)
    heights = np.asarray([float(row["root_height"]) for row in actual], dtype=np.float64)
    tilt = np.asarray(
        [
            np.degrees(np.arccos(np.clip(-np.asarray(row["projected_gravity"])[2], -1.0, 1.0)))
            for row in actual
        ]
    )
    root = np.asarray([row["root_position"] for row in actual])
    board = np.asarray([row["board_position"] for row in actual])
    separation = np.linalg.norm(root[:, :2] - board[:, :2], axis=1)
    off_board = ~feet
    first_off = np.flatnonzero(off_board)
    max_streak = current = 0
    for value in off_board:
        current = current + 1 if value else 0
        max_streak = max(max_streak, current)
    first = int(first_off[0]) if first_off.size else None
    return {
        "feet_on_board_ratio": float(feet.mean()),
        "off_board_ratio": float(off_board.mean()),
        "final_feet_on_board": bool(feet[-1]),
        "time_to_first_off_board_s": None if first is None else float((first + 1) * CONTROL_DT),
        "longest_off_board_streak_s": float(max_streak * CONTROL_DT),
        "robot_board_separation_mean_m": float(separation.mean()),
        "robot_board_separation_final_m": float(separation[-1]),
        "robot_board_separation_max_m": float(separation.max()),
    }, {
        "root_tilt_mean_deg": float(tilt.mean()),
        "root_tilt_p95_deg": float(np.quantile(tilt, 0.95)),
        "root_tilt_max_deg": float(tilt.max()),
        "root_height_min_m": float(heights.min()),
        "illegal_contact_ratio": float(illegal.mean()),
        "off_board_upright_ratio": (
            float(np.mean((tilt[off_board] < 70.0) & (heights[off_board] >= 0.45)))
            if off_board.any()
            else None
        ),
        "post_first_off_board_survival_s": (
            None if first is None else float((len(actual) - first - 1) * CONTROL_DT)
        ),
    }


def open_video(path: Path, env: HuskyBfmOnlineEnv) -> tuple[Any, mujoco.Renderer]:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", macro_block_size=1)
    renderer = mujoco.Renderer(env.env.model, height=720, width=1280)
    return writer, renderer


def render_frame(writer: Any, renderer: mujoco.Renderer, env: HuskyBfmOnlineEnv) -> None:
    renderer.update_scene(env.env.data, camera="robot/tracking")
    writer.append_data(renderer.render())


def run_reference_motion(
    agent: Any,
    tracking: AlignedSkateTrackingContext,
    resolver: ReferenceSourceResolver,
    record: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    env: HuskyBfmOnlineEnv,
    duration_steps: int,
    model_video: Path | None = None,
) -> dict[str, Any]:
    """The only policy/controller/physics rollout path used by every mode."""

    motion_key = str(case["motion_key"])
    local = int(case["reset_local_frame"])
    source_start = int(record["source_start_frame"])
    raw, metadata, source_path = resolver.load(record)
    raw_frames = int(record["source_end_frame"]) - source_start
    trajectory = tracking.trajectories.get(motion_key)
    if trajectory is None or trajectory["raw_frames"] != raw_frames:
        raise RuntimeError(f"Tracking/raw MotionLibrary disagreement for {motion_key}.")
    if local < 0 or local + duration_steps >= raw_frames:
        raise RuntimeError(f"Fixed 250-step reference window is invalid for {motion_key}.")
    z, ranges = tracking.encode(agent._model, motion_key, local, duration_steps)
    if z.shape != (duration_steps, 256) or not torch.isfinite(z).all():
        raise RuntimeError("Tracking latent must be finite [250, 256].")
    if any(item["future_start"] != local + step + 1 for step, item in enumerate(ranges)):
        raise RuntimeError("Tracking latent future indices are not exact t+1 alignment.")

    reset_frame = source_start + local
    qpos = np.asarray(raw["qpos"][reset_frame], dtype=np.float64)
    qvel = np.asarray(raw["qvel"][reset_frame], dtype=np.float64)
    writer = renderer = None
    actual: list[dict[str, Any]] = []
    first_action = None
    terminated = truncated = False
    try:
        observation = env.reset(
            qpos=qpos, qvel=qvel, source_physics=metadata["physics_randomization"]
        )
        if not (
            np.allclose(env.env.data.qpos, qpos, atol=1e-8, rtol=0.0)
            and np.allclose(env.env.data.qvel, qvel, atol=1e-8, rtol=0.0)
        ):
            raise RuntimeError("Exact canonical qpos/qvel reset check failed.")
        diagnostics = ControlDiagnostics(env)
        if model_video is not None:
            writer, renderer = open_video(model_video, env)
        for step in range(duration_steps):
            model_obs = {
                key: value.unsqueeze(0).to(agent.device) for key, value in observation.items()
            }
            with torch.no_grad():
                action = agent.act(model_obs, z[step].unsqueeze(0), mean=True)[0]
            if action.shape != (29,) or not torch.isfinite(action).all():
                raise RuntimeError("Frozen actor produced an invalid 29D action.")
            if first_action is None:
                first_action = hash_data(action)
            transition = env.step(action, z[step], truncated=step == duration_steps - 1)
            diagnostics.update(transition.action_husky, env.env.data.qfrc_actuator)
            actual.append(dict(transition.raw_metadata))
            if writer is not None:
                render_frame(writer, renderer, env)
            observation = transition.next_observation
            terminated, truncated = transition.terminated, transition.truncated
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    if not actual:
        raise RuntimeError("Policy rollout did not execute a transition.")
    series = reference_metric_series(actual, raw, reset_frame)
    phase_ids = np.asarray(record["phase_id"], dtype=np.int16)
    labels = [PHASE_NAMES[int(value)] for value in phase_ids[local + 1 : local + len(actual) + 1]]
    transition_name = str(case["transition"])
    transition_phase = transition_name
    pre_phase = str(case["pre_phase"])
    post_phase = str(case["post_phase"])
    retention, stability = retention_and_stability(actual)
    return {
        "source_path": source_path,
        "raw": raw,
        "metadata": metadata,
        "reset_frame": reset_frame,
        "actual": actual,
        "first_action_fingerprint": first_action,
        "result": {
            "case": {
                **case,
                "transition_type": transition_name,
                "reset_raw_frame": reset_frame,
                "source_raw_npz": str(source_path),
                "source_physics_seed": int(metadata["physics_randomization"]["seed"]),
                "transition_onset_s": float(
                    (int(case["transition_local_start"]) - local) * CONTROL_DT
                ),
                "transition_end_s": float((int(case["transition_local_end"]) - local) * CONTROL_DT),
            },
            "evaluation": {
                "control_dt": CONTROL_DT,
                "T_eval": duration_steps,
                "T_exec": len(actual),
                "completion_ratio": float(len(actual) / duration_steps),
                "full_completion": bool(len(actual) == duration_steps and not terminated),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "fall_reason": str(actual[-1].get("fall_reason", "")) if terminated else "",
            },
            "metrics": {
                "full": {name: summarize_values(values) for name, values in series.items()},
                "pre": section_metrics(series, labels, pre_phase),
                "transition": section_metrics(series, labels, transition_phase),
                "post": section_metrics(series, labels, post_phase),
            },
            "retention": retention,
            "stability": stability,
            "control": diagnostics.summary(),
            "tracking": {
                "z_shape": list(z.shape),
                "finite": True,
                "future_start": [item["future_start"] for item in ranges],
            },
            "alignment": {
                "canonical_reset": True,
                "post_step_reference_start": reset_frame + 1,
                "post_step_reference_end": reset_frame + len(actual),
                "t_plus_one": True,
            },
            "first_action_fingerprint": first_action,
        },
    }


def render_expert_video(path: Path, case_run: Mapping[str, Any], duration_steps: int) -> None:
    """Render only canonical raw states; no physics steps or policy calls."""

    raw = case_run["raw"]
    metadata = case_run["metadata"]
    reset = int(case_run["reset_frame"])
    env = HuskyBfmOnlineEnv()
    writer = renderer = None
    try:
        env.reset(
            qpos=np.asarray(raw["qpos"][reset], dtype=np.float64),
            qvel=np.asarray(raw["qvel"][reset], dtype=np.float64),
            source_physics=metadata["physics_randomization"],
        )
        writer, renderer = open_video(path, env)
        for frame in range(reset + 1, reset + duration_steps + 1):
            env.env.data.qpos[:] = raw["qpos"][frame]
            env.env.data.qvel[:] = raw["qvel"][frame]
            mujoco.mj_forward(env.env.model, env.env.data)
            render_frame(writer, renderer, env)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()


def training_date(summary_path: Path) -> str:
    summary = json.loads(summary_path.expanduser().resolve().read_text())
    started = summary.get("run_provenance", {}).get("started_at")
    if not isinstance(started, str) or len(started) < 10:
        raise RuntimeError("Training summary lacks run_provenance.started_at.")
    return started[:10]


def output_dir_for(args: argparse.Namespace) -> Path:
    return (
        REPOSITORY_ROOT
        / "train/eval_res"
        / training_date(args.training_summary)
        / f"{args.label}_transition_5s"
    )


def result_payload(
    run: Mapping[str, Any], checkpoint: Path, checkpoint_sha: str, label: str, video: bool
) -> dict[str, Any]:
    result = copy.deepcopy(run["result"])
    result["checkpoint"] = {"label": label, "path": str(checkpoint), "sha256": checkpoint_sha}
    result["videos"] = {
        "model": "model.mp4" if video else None,
        "expert": "expert.mp4" if video else None,
        "compare": None,
    }
    result["mutation"] = {
        "parameters_changed": False,
        "buffers_changed": False,
        "normalizer_changed": False,
        "components_changed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "agent_update_calls": 0,
    }
    return result


def metric_mean(result: Mapping[str, Any], name: str) -> str:
    metrics = result["metrics"]["full"][name]
    return f"{metrics['mean']:.4g}"


def write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# 5s Expert Transition Reproduction",
        "",
        (
            "| Case | Completion | Joint MAE | Root Ori | Board XY | Heading | "
            "Coupling XY | Feet on board |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        case = row["case"]
        evaluation = row["evaluation"]
        retention = row["retention"]
        lines.append(
            f"| {case['case_id']} | {evaluation['completion_ratio']:.2%} | "
            f"{metric_mean(row, 'joint_position_mae_rad')} | "
            f"{metric_mean(row, 'root_orientation_geodesic_error_deg')} | "
            f"{metric_mean(row, 'board_xy_displacement_error_m')} | "
            f"{metric_mean(row, 'board_heading_error_deg')} | "
            f"{metric_mean(row, 'coupling_xy_error_m')} | "
            f"{retention['feet_on_board_ratio']:.2%} |"
        )
    for transition in ("push2steer", "steer2push"):
        selected = [row for row in rows if row["case"]["transition"] == transition]
        if not selected:
            continue
        lines += ["", f"## {transition}", ""]
        for row in selected:
            sections = []
            for name in ("pre", "transition", "post"):
                metric = row["metrics"][name]
                sections.append(
                    f"{name}: n/a"
                    if metric is None
                    else f"{name}: joint MAE {metric['joint_position_mae_rad']['mean']:.4g}, "
                    f"board XY {metric['board_xy_displacement_error_m']['mean']:.4g}"
                )
            lines.append(f"- `{row['case']['case_id']}`: " + "; ".join(sections))
    lines += ["", "## Stability / Retention", ""]
    for row in rows:
        stability = row["stability"]
        retention = row["retention"]
        evaluation = row["evaluation"]
        lines.append(
            f"- `{row['case']['case_id']}`: terminated={evaluation['terminated']}, "
            f"off-board={retention['off_board_ratio']:.2%}, "
            f"off-board upright={stability['off_board_upright_ratio']}, "
            f"fall reason={evaluation['fall_reason'] or 'none'}."
        )
    path.write_text("\n".join(lines) + "\n")


def invocation_config(
    args: argparse.Namespace,
    checkpoint: Path,
    checkpoint_sha: str,
    motion_path: Path,
    manifest_path: Path,
    duration_steps: int,
    lead_steps: int,
) -> dict[str, Any]:
    return {
        "checkpoint": {"label": args.label, "path": str(checkpoint), "sha256": checkpoint_sha},
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "evaluator_sha256": hash_file(Path(__file__)),
        "training_date": training_date(args.training_summary),
        "training_summary": str(args.training_summary.expanduser().resolve()),
        "split": "val",
        "dataset": "continuous",
        "motion_path": str(motion_path),
        "motion_sha256": hash_file(motion_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hash_file(manifest_path),
        "seed": args.seed,
        "transition": args.transition,
        "samples_per_transition": args.samples_per_transition,
        "duration_s": DURATION_S,
        "duration_steps": duration_steps,
        "lead_s": LEAD_S,
        "lead_steps": lead_steps,
        "viewer": bool(args.viewer),
        "video": bool(args.video),
        "evaluation_only": True,
        "training": False,
        "test": False,
    }


def run_once(
    agent: Any,
    tracking: AlignedSkateTrackingContext,
    resolver: ReferenceSourceResolver,
    records: Mapping[str, Mapping[str, Any]],
    case: Mapping[str, Any],
    env: HuskyBfmOnlineEnv,
    duration_steps: int,
    model_video: Path | None = None,
) -> dict[str, Any]:
    return run_reference_motion(
        agent,
        tracking,
        resolver,
        records[str(case["motion_key"])],
        case,
        env=env,
        duration_steps=duration_steps,
        model_video=model_video,
    )


def run_viewer(
    agent: Any,
    tracking: AlignedSkateTrackingContext,
    resolver: ReferenceSourceResolver,
    records: Mapping[str, Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    output: Path,
    duration_steps: int,
    checkpoint: Path,
    checkpoint_sha: str,
    seed: int,
) -> None:
    env = HuskyBfmOnlineEnv(viewer=True, realtime=True)
    before = checkpoint_mutation(agent)
    sequence: list[dict[str, Any]] = []
    index = 0
    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    previous_int = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        while env.env.is_running and not interrupted:
            case = cases[index % len(cases)]
            run = run_once(agent, tracking, resolver, records, case, env, duration_steps)
            result = run["result"]
            sequence.append(
                {
                    "sequence_index": index,
                    "case_id": case["case_id"],
                    "transition": case["transition"],
                    "motion_key": case["motion_key"],
                    "reset_local_frame": case["reset_local_frame"],
                    "T_exec": result["evaluation"]["T_exec"],
                    "termination": result["evaluation"]["terminated"],
                    "fall_reason": result["evaluation"]["fall_reason"],
                    "first_action_fingerprint": result["first_action_fingerprint"],
                    "primary_metric_means": {
                        name: result["metrics"]["full"][name]["mean"] for name in PRIMARY_METRICS
                    },
                }
            )
            print(
                f"[viewer] case={case['case_id']} steps={result['evaluation']['T_exec']} "
                f"terminated={result['evaluation']['terminated']} "
                f"reason={result['evaluation']['fall_reason'] or 'none'}"
            )
            index += 1
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        env.close()
    after = checkpoint_mutation(agent)
    if before != after:
        raise RuntimeError("Frozen checkpoint mutated during viewer session.")
    write_json(
        output / "viewer_session.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "seed": seed,
            "duration_steps": duration_steps,
            "lead_steps": int(LEAD_S / CONTROL_DT),
            "rollouts_completed": len(sequence),
            "case_sequence": sequence,
            "closed_by_user": not interrupted,
            "mutation": {
                "parameters_changed": False,
                "buffers_changed": False,
                "normalizer_changed": False,
                "components_changed": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "agent_update_calls": 0,
            },
        },
    )


def main() -> int:
    args = parse_args()
    if args.viewer and args.video:
        raise ValueError("--viewer and --video are separate inspection modes.")
    if not math.isclose(args.duration_s, DURATION_S) or not math.isclose(args.lead_s, LEAD_S):
        raise ValueError(
            "The formal transition evaluator is fixed to --duration-s 5 and --lead-s 1."
        )
    duration_steps = round(DURATION_S / CONTROL_DT)
    lead_steps = round(LEAD_S / CONTROL_DT)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    checkpoint_sha = hash_file(checkpoint_model_path(checkpoint))
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256:
        raise RuntimeError("Checkpoint SHA256 mismatch.")

    records, _, motion_path, manifest_path = load_continuous_records()
    candidates = detect_transition_candidates(records, lead_steps, duration_steps)
    cases = select_cases(args, candidates, records, duration_steps, lead_steps)
    output = output_dir_for(args)
    output.mkdir(parents=True, exist_ok=True)
    if not args.viewer or not (output / "config.json").exists():
        write_json(
            output / "config.json",
            invocation_config(
                args,
                checkpoint,
                checkpoint_sha,
                motion_path,
                manifest_path,
                duration_steps,
                lead_steps,
            ),
        )
    if not args.viewer or not (output / "cases.json").exists():
        write_json(
            output / "cases.json",
            {
                "seed": args.seed,
                "duration_s": DURATION_S,
                "duration_steps": duration_steps,
                "lead_s": LEAD_S,
                "lead_steps": lead_steps,
                "cases": cases,
            },
        )

    agent, load_report = load_frozen_agent(checkpoint)
    if agent._model.training or any(
        parameter.requires_grad for parameter in agent._model.parameters()
    ):
        raise RuntimeError("Evaluator requires a frozen eval-mode checkpoint.")
    tracking = AlignedSkateTrackingContext.load(agent, motion_path)
    if set(tracking.trajectories) != set(records):
        raise RuntimeError("Tracking MotionLibrary keys do not match Val Continuous records.")
    resolver = ReferenceSourceResolver()

    if args.viewer:
        run_viewer(
            agent,
            tracking,
            resolver,
            records,
            cases,
            output,
            duration_steps,
            checkpoint,
            checkpoint_sha,
            args.seed,
        )
        return 0

    before = checkpoint_mutation(agent)
    rows = []
    env = HuskyBfmOnlineEnv()
    try:
        for case in cases:
            case_output = output / str(case["case_id"])
            run = run_once(
                agent,
                tracking,
                resolver,
                records,
                case,
                env,
                duration_steps,
                model_video=case_output / "model.mp4" if args.video else None,
            )
            result = result_payload(run, checkpoint, checkpoint_sha, args.label, args.video)
            result["load_report"] = load_report
            write_json(case_output / "result.json", result)
            if args.video:
                render_expert_video(case_output / "expert.mp4", run, duration_steps)
            rows.append(result)
    finally:
        env.close()
    after = checkpoint_mutation(agent)
    if before != after:
        raise RuntimeError("Frozen checkpoint mutated during evaluation.")
    summary = {
        "checkpoint": {"label": args.label, "path": str(checkpoint), "sha256": checkpoint_sha},
        "candidate_counts": {name: len(items) for name, items in candidates.items()},
        "cases": [row["case"] for row in rows],
        "results": rows,
        "mutation": {
            "parameters_changed": False,
            "buffers_changed": False,
            "normalizer_changed": False,
            "components_changed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "agent_update_calls": 0,
        },
        "evaluation_only": True,
        "training": False,
        "test": False,
    }
    write_json(output / "summary.json", summary)
    write_summary(output / "summary.md", rows)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "candidate_counts": summary["candidate_counts"],
                "cases": [row["case"]["case_id"] for row in rows],
                "video": bool(args.video),
                "training": False,
                "test": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
