#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Evaluate, record, or view frozen Skate-BFM policies on Phase + Raw data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import signal
import subprocess
import sys
import tempfile
from collections import Counter
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

from data_collection.convert_continuous import (
    bfm_joint_contract,
    convert_record,
    load_reference,
)
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
EVAL_SECONDS = 5.0
EVAL_STEPS = 250
LEAD_SECONDS = 1.0
LEAD_STEPS = 50
PHASE_LABELS = {
    0: "push",
    1: "push2steer",
    2: "steer_left",
    3: "steer_right",
    4: "steer_forward",
    5: "steer2push",
    6: "fall",
}
STEER = frozenset(("steer_left", "steer_forward", "steer_right"))
BEHAVIORS = ("push", "steer", "push2steer", "steer2push")
SELECTION_SEED_OFFSETS = {
    "push": 0,
    "steer": 1000,
    "push2steer": 2000,
    "steer2push": 3000,
}
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--checkpoint", required=True, type=Path)
        subparser.add_argument("--checkpoint-sha256")
        subparser.add_argument("--label", required=True)
        subparser.add_argument("--training-summary", required=True, type=Path)
        subparser.add_argument("--seed", type=int, default=4728)

    evaluate = subparsers.add_parser("eval", help="Formal Test Phase + Raw evaluation.")
    common(evaluate)
    evaluate.add_argument(
        "--behavior",
        choices=(*BEHAVIORS, "all"),
        default="all",
    )
    evaluate.add_argument("--steady-duration-s", type=float, default=3.0)
    evaluate.add_argument("--max-cases-per-behavior", type=int)
    evaluate.add_argument("--case-spec", type=Path)

    video = subparsers.add_parser("video", help="Val Phase + Raw presentation videos.")
    common(video)
    video.add_argument(
        "--behavior",
        choices=(*BEHAVIORS, "all"),
        default="all",
    )
    video.add_argument(
        "--steer-direction", choices=("left", "forward", "right", "random"), default="random"
    )
    video.add_argument("--clip-s", type=float, default=3.0)
    video.add_argument("--context-s", type=float, default=0.5)
    video.add_argument("--with-expert", action="store_true")

    view = subparsers.add_parser("viewer", help="Persistent Val Phase + Raw viewer.")
    common(view)
    view.add_argument(
        "--behavior",
        choices=(*BEHAVIORS, "all"),
        default="all",
    )
    view.add_argument(
        "--steer-direction", choices=("left", "forward", "right", "random"), default="random"
    )
    view.add_argument("--clip-s", type=float, default=3.0)
    view.add_argument("--context-s", type=float, default=0.5)
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def board_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def board_tilt_deg(quaternion: np.ndarray) -> float:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise RuntimeError("Board quaternion must be finite wxyz.")
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise RuntimeError("Board quaternion has zero norm.")
    _, x, y, _ = q / norm
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def quaternion_geodesic_deg(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    actual_norm = np.linalg.norm(actual, axis=-1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=-1, keepdims=True)
    if (actual_norm <= 1e-12).any() or (reference_norm <= 1e-12).any():
        raise RuntimeError("Quaternion metric received zero norm.")
    dot = np.clip(
        np.abs(np.sum(actual / actual_norm * (reference / reference_norm), axis=-1)), 0, 1
    )
    return np.degrees(2 * np.arccos(dot))


def wrapped_angle_deg(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.degrees(np.abs(np.angle(np.exp(1j * (actual - reference)))))


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Metric values must be finite and non-empty.")
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


class RawResolver:
    """Resolve one Phase record to its exact split-specific raw rollout."""

    def __init__(self, split: str) -> None:
        if split not in ("val", "test"):
            raise ValueError(split)
        self.split = split
        self.metadata_split = {"val": "validation", "test": "test"}[split]
        self.root = REPOSITORY_ROOT / "train/dataset/sim_collected" / split / "raw"
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
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
            raise RuntimeError("Phase record lacks raw provenance.")
        if record["dataset_split"] != self.metadata_split:
            raise RuntimeError("Phase record split does not match evaluator split.")
        root = self.root / f"round_{str(record['source_round']).zfill(3)}"
        root /= f"rollout_{str(record['source_rollout']).zfill(3)}"
        root /= "raw_rollout"
        files = sorted(root.glob("*.npz"))
        if len(files) != 1 or files[0].name != Path(str(record["source_raw_npz"])).name:
            raise RuntimeError(f"Raw provenance is not uniquely resolvable under {root}.")
        path = files[0].resolve()
        if path not in self.cache:
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            expected = {
                "dataset_split": self.metadata_split,
                "round_id": str(record["source_round"]).zfill(3),
                "rollout_id": str(record["source_rollout"]).zfill(3),
                "episode_id": str(record["source_episode"]),
            }
            actual = {key: str(metadata.get(key, "")) for key in expected}
            if actual != expected:
                raise RuntimeError(f"Raw metadata mismatch: {actual} != {expected}")
            physics = metadata.get("physics_randomization")
            if not isinstance(physics, dict) or int(physics.get("seed", -1)) != int(
                record["physics_seed"]
            ):
                raise RuntimeError("Raw source physics provenance mismatch.")
            with np.load(path, allow_pickle=False) as archive:
                state = {name: np.asarray(archive[name]).copy() for name in archive.files}
            frames = state["qpos"].shape[0]
            if any(value.shape[0] != frames for value in state.values()):
                raise RuntimeError(f"Raw arrays are not frame aligned: {path}")
            if not all(
                np.isfinite(value).all()
                for value in state.values()
                if np.issubdtype(value.dtype, np.number)
            ):
                raise RuntimeError(f"Raw arrays contain NaN/Inf: {path}")
            self.cache[path] = (state, copy.deepcopy(metadata))
        state, metadata = self.cache[path]
        return state, metadata, path


def phase_paths(split: str) -> tuple[Path, Path]:
    root = REPOSITORY_ROOT / "train/dataset/sim_collected" / split / "phase/motion_library"
    return root / "skate_expert_phase.pkl", root / "manifest.json"


def load_phase_records(split: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Path, Path]:
    motion_path, manifest_path = phase_paths(split)
    if not motion_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{split} Phase MotionLibrary is incomplete.")
    manifest = json.loads(manifest_path.read_text())
    expected = {"val": "validation", "test": "test"}[split]
    if manifest.get("dataset_split") != expected:
        raise RuntimeError("Phase manifest split mismatch.")
    records = joblib.load(motion_path)
    if not isinstance(records, dict) or not records:
        raise RuntimeError("Phase MotionLibrary must be a non-empty mapping.")
    result = {str(key): value for key, value in records.items()}
    valid = {"push", "push2steer", "steer_left", "steer_forward", "steer_right", "steer2push"}
    for key, record in result.items():
        if record.get("phase_label") not in valid:
            raise RuntimeError(f"Invalid Phase label in {key}.")
        if int(record["source_end_frame"]) - int(record["source_start_frame"]) != len(
            record["dof"]
        ):
            raise RuntimeError(f"Phase frame range mismatch in {key}.")
    return result, manifest, motion_path, manifest_path


def source_identity(record: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(record["source_raw_npz"]),
        str(record["source_round"]).zfill(3),
        str(record["source_rollout"]).zfill(3),
        str(record["source_episode"]),
        int(record["physics_seed"]),
    )


def tracking_seq_length(agent: Any) -> int:
    value = int(agent._model.cfg.seq_length)
    if value <= 0:
        raise RuntimeError(f"Invalid model seq_length: {value}")
    return value


def phase_groups(
    records: Mapping[str, Mapping[str, Any]],
) -> list[list[tuple[str, Mapping[str, Any]]]]:
    grouped: dict[tuple[str, str, str, str, int], list[tuple[str, Mapping[str, Any]]]] = {}
    for key, record in records.items():
        grouped.setdefault(source_identity(record), []).append((key, record))
    result = []
    for items in grouped.values():
        items.sort(key=lambda item: int(item[1]["source_start_frame"]))
        for previous, current in zip(items, items[1:]):
            if int(previous[1]["source_end_frame"]) != int(current[1]["source_start_frame"]):
                raise RuntimeError("Phase boundary is not exactly source-frame contiguous.")
        result.append(items)
    return result


def transition_candidates(
    records: Mapping[str, Mapping[str, Any]],
    resolver: RawResolver,
    seq_length: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for items in phase_groups(records):
        for previous, transition, following in zip(items, items[1:], items[2:]):
            pre_key, pre = previous
            transition_key, transition_record = transition
            post_key, post = following
            label = transition_record["phase_label"]
            if label == "push2steer":
                valid = pre["phase_label"] == "push" and post["phase_label"] in STEER
                kind = "push2steer"
            elif label == "steer2push":
                valid = pre["phase_label"] in STEER and post["phase_label"] == "push"
                kind = "steer2push"
            else:
                continue
            if not valid:
                excluded[f"{kind}:phase_semantics"] += 1
                continue
            start = int(transition_record["source_start_frame"])
            end = int(transition_record["source_end_frame"])
            reset = start - LEAD_STEPS
            raw, metadata, path = resolver.load(transition_record)
            bridge_end = reset + EVAL_STEPS + seq_length + 1
            if reset < 0:
                excluded[f"{kind}:no_lead"] += 1
            elif bridge_end > len(raw["qpos"]):
                excluded[f"{kind}:raw_window"] += 1
            elif np.any(raw["reset"][reset:bridge_end]) or np.any(raw["fall"][reset:bridge_end]):
                excluded[f"{kind}:raw_reset_or_fall"] += 1
            else:
                candidate_index = len([item for item in candidates if item["transition"] == kind])
                candidates.append(
                    {
                        "case_id": f"{kind}_{candidate_index:03d}",
                        "behavior": kind,
                        "transition": kind,
                        "transition_motion_key": transition_key,
                        "pre_motion_key": pre_key,
                        "post_motion_key": post_key,
                        "motion_key": transition_key,
                        "steps": EVAL_STEPS,
                        "source_raw_npz": str(path),
                        "source_round": str(transition_record["source_round"]),
                        "source_rollout": str(transition_record["source_rollout"]),
                        "source_episode": str(transition_record["source_episode"]),
                        "physics_seed": int(transition_record["physics_seed"]),
                        "reset_raw_frame": reset,
                        "transition_start_raw": start,
                        "transition_end_raw": end,
                        "pre_phase": pre["phase_label"],
                        "transition_phase": label,
                        "post_phase": post["phase_label"],
                        "phase_ranges": [
                            {
                                "phase": pre["phase_label"],
                                "start": int(pre["source_start_frame"]),
                                "end": int(pre["source_end_frame"]),
                            },
                            {
                                "phase": label,
                                "start": start,
                                "end": end,
                            },
                            {
                                "phase": post["phase_label"],
                                "start": int(post["source_start_frame"]),
                                "end": int(post["source_end_frame"]),
                            },
                        ],
                        "source_physics": metadata["physics_randomization"],
                    }
                )
    return candidates, excluded


def steady_candidates(
    records: Mapping[str, Mapping[str, Any]],
    resolver: RawResolver,
    behavior: str,
    steps: int,
    seq_length: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    labels = {"push"} if behavior == "push" else STEER
    candidates: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    required_frames = steps + seq_length + 1
    phase_ids = {name: value for value, name in PHASE_LABELS.items()}
    for key, record in sorted(records.items()):
        label = str(record["phase_label"])
        if label not in labels:
            continue
        phase_start = int(record["source_start_frame"])
        phase_end = int(record["source_end_frame"])
        if phase_end - phase_start < required_frames:
            excluded["too_short"] += 1
            continue
        reset = (phase_start + phase_end - required_frames) // 2
        raw, metadata, path = resolver.load(record)
        raw_end = reset + required_frames
        if raw_end > len(raw["qpos"]):
            excluded["tracking_context"] += 1
            continue
        if np.any(raw["reset"][reset:raw_end]) or np.any(raw["fall"][reset:raw_end]):
            excluded["raw_reset_or_fall"] += 1
            continue
        expected_phase = phase_ids[label]
        if not np.all(np.asarray(raw["phase_id"][reset:raw_end]) == expected_phase):
            excluded["phase_contamination"] += 1
            continue
        candidates.append(
            {
                "case_id": f"{behavior}_{len(candidates):03d}",
                "behavior": behavior,
                "motion_key": key,
                "source_raw_npz": str(path),
                "source_round": str(record["source_round"]),
                "source_rollout": str(record["source_rollout"]),
                "source_episode": str(record["source_episode"]),
                "physics_seed": int(record["physics_seed"]),
                "reset_raw_frame": reset,
                "steps": steps,
                "phase_start_raw": phase_start,
                "phase_end_raw": phase_end,
                "phase_ranges": [
                    {"phase": label, "start": phase_start, "end": phase_end}
                ],
                "source_physics": metadata["physics_randomization"],
                **(
                    {"steer_direction": label.removeprefix("steer_")}
                    if behavior == "steer"
                    else {}
                ),
            }
        )
    return candidates, excluded


def formal_candidates(
    records: Mapping[str, Mapping[str, Any]],
    resolver: RawResolver,
    steady_steps: int,
    seq_length: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    push, push_excluded = steady_candidates(
        records, resolver, "push", steady_steps, seq_length
    )
    steer, steer_excluded = steady_candidates(
        records, resolver, "steer", steady_steps, seq_length
    )
    transition, transition_excluded = transition_candidates(records, resolver, seq_length)
    bank = {
        "push": push,
        "steer": steer,
        "push2steer": [item for item in transition if item["behavior"] == "push2steer"],
        "steer2push": [item for item in transition if item["behavior"] == "steer2push"],
    }
    excluded = {
        "push": dict(push_excluded),
        "steer": dict(steer_excluded),
        "push2steer": {
            key.split(":", 1)[1]: value
            for key, value in transition_excluded.items()
            if key.startswith("push2steer:")
        },
        "steer2push": {
            key.split(":", 1)[1]: value
            for key, value in transition_excluded.items()
            if key.startswith("steer2push:")
        },
    }
    return bank, excluded


def source_group(case: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(case["source_round"]).zfill(3),
        str(case["source_rollout"]).zfill(3),
        str(case["source_episode"]),
        int(case["physics_seed"]),
    )


def case_sort_key(case: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(case["source_round"]).zfill(3),
        str(case["source_rollout"]).zfill(3),
        int(case["reset_raw_frame"]),
        str(case["case_id"]),
    )


def case_identity(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "behavior": str(case["behavior"]),
        "source_raw_npz": str(case["source_raw_npz"]),
        "source_round": str(case["source_round"]).zfill(3),
        "source_rollout": str(case["source_rollout"]).zfill(3),
        "source_episode": str(case["source_episode"]),
        "physics_seed": int(case["physics_seed"]),
        "reset_raw_frame": int(case["reset_raw_frame"]),
    }


def selected_direction(case: Mapping[str, Any]) -> str | None:
    behavior = str(case["behavior"])
    if behavior == "steer":
        return str(case["steer_direction"])
    if behavior == "push2steer":
        phase = str(case.get("post_phase", ""))
    elif behavior == "steer2push":
        phase = str(case.get("pre_phase", ""))
    else:
        return None
    return phase.removeprefix("steer_") if phase in STEER else None


def selection_stats(
    bank: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    mode: str,
    seed: int | None,
    limit: int | None,
) -> dict[str, Any]:
    selected_cases = [
        case
        for behavior in BEHAVIORS
        for case in sorted(selected.get(behavior, ()), key=case_sort_key)
    ]
    identities = [case_identity(case) for case in selected_cases]
    fingerprint = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def group_count(cases: Sequence[Mapping[str, Any]]) -> int:
        paths: dict[tuple[str, str, str, int], str] = {}
        for case in cases:
            key = source_group(case)
            path = str(case["source_raw_npz"])
            if key in paths and paths[key] != path:
                raise RuntimeError(f"Source group maps to multiple raw files: {key}")
            paths[key] = path
        return len(paths)

    direction_counts = {
        behavior: dict(
            sorted(
                Counter(
                    direction
                    for case in selected.get(behavior, ())
                    if (direction := selected_direction(case)) is not None
                ).items()
            )
        )
        for behavior in ("steer", "push2steer", "steer2push")
    }
    warnings = []
    if mode == "rollout_balanced_without_replacement":
        for behavior in direction_counts:
            eligible_directions = Counter(
                direction
                for case in bank[behavior]
                if (direction := selected_direction(case)) is not None
            )
            for direction in eligible_directions:
                if direction_counts[behavior].get(direction, 0) == 0:
                    warnings.append(
                        f"{behavior} selected no {direction} case despite eligible cases."
                    )
    return {
        "mode": mode,
        "seed": seed,
        "max_cases_per_behavior": limit,
        "selection_fingerprint": fingerprint,
        "eligible_counts": {behavior: len(bank[behavior]) for behavior in BEHAVIORS},
        "selected_counts": {behavior: len(selected.get(behavior, ())) for behavior in BEHAVIORS},
        "eligible_source_groups": {
            behavior: group_count(bank[behavior]) for behavior in BEHAVIORS
        },
        "selected_unique_source_groups": {
            behavior: group_count(selected.get(behavior, ())) for behavior in BEHAVIORS
        },
        "selected_direction_counts": direction_counts,
        "warnings": warnings,
    }


def select_balanced_cases(
    cases: Sequence[Mapping[str, Any]],
    limit: int | None,
    seed: int,
    behavior: str,
) -> list[Mapping[str, Any]]:
    ordered = sorted(cases, key=case_sort_key)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit <= 0:
        raise ValueError("max-cases-per-behavior must be positive.")
    if behavior not in SELECTION_SEED_OFFSETS:
        raise ValueError(f"Unsupported selection behavior: {behavior}")

    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for case in ordered:
        key = source_group(case)
        group = grouped.setdefault(key, [])
        if group and str(group[0]["source_raw_npz"]) != str(case["source_raw_npz"]):
            raise RuntimeError(f"Source group maps to multiple raw files: {key}")
        group.append(case)
    rng = np.random.default_rng(seed + SELECTION_SEED_OFFSETS[behavior])
    group_keys = list(grouped)
    group_keys = [group_keys[index] for index in rng.permutation(len(group_keys))]
    for key in group_keys:
        group = grouped[key]
        permutation = rng.permutation(len(group))
        grouped[key] = [group[index] for index in permutation]

    selected: list[Mapping[str, Any]] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in group_keys:
            group = grouped[key]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            raise RuntimeError("Balanced selection exhausted before reaching limit.")
        depth += 1
    selected_ids = [str(case["case_id"]) for case in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Balanced selection returned duplicate case IDs.")
    return selected


def behavior_candidates(
    records: Mapping[str, Mapping[str, Any]],
    resolver: RawResolver,
    behavior: str,
    clip_steps: int,
    context_steps: int,
    direction: str,
    seq_length: int,
) -> list[dict[str, Any]]:
    if behavior in ("push", "steer"):
        labels = (
            {"push"}
            if behavior == "push"
            else (STEER if direction == "random" else {f"steer_{direction}"})
        )
        result = []
        for key, record in sorted(records.items()):
            if record["phase_label"] not in labels:
                continue
            start = int(record["source_start_frame"])
            raw, _, path = resolver.load(record)
            end = start + clip_steps + seq_length + 1
            if (
                end <= int(record["source_end_frame"])
                and not np.any(raw["reset"][start:end])
                and not np.any(raw["fall"][start:end])
            ):
                result.append(
                    {
                        "case_id": f"{behavior}_{len(result):03d}",
                        "behavior": behavior,
                        "motion_key": key,
                        "source_raw_npz": str(path),
                        "source_round": str(record["source_round"]),
                        "source_rollout": str(record["source_rollout"]),
                        "source_episode": str(record["source_episode"]),
                        "physics_seed": int(record["physics_seed"]),
                        "reset_raw_frame": start,
                        "clip_start_raw": start,
                        "clip_end_raw": start + clip_steps + 1,
                        "phase_ranges": [
                            {
                                "phase": record["phase_label"],
                                "start": start,
                                "end": start + clip_steps + 1,
                            }
                        ],
                        "source_physics": resolver.load(record)[1]["physics_randomization"],
                    }
                )
        return result
    transitions, _ = transition_candidates(records, resolver, seq_length)
    result = []
    target = "push2steer" if behavior == "push2steer" else "steer2push"
    for candidate in transitions:
        if candidate["transition"] != target:
            continue
        start = candidate["transition_start_raw"] - context_steps
        end = candidate["transition_end_raw"] + context_steps
        raw, _, path = resolver.load(records[candidate["transition_motion_key"]])
        bridge_end = start + (end - start - 1) + seq_length + 1
        if start < 0 or bridge_end > len(raw["qpos"]):
            continue
        if np.any(raw["reset"][start:bridge_end]) or np.any(raw["fall"][start:bridge_end]):
            continue
        item = dict(candidate)
        item.update(
            {
                "case_id": f"{behavior}_{len(result):03d}",
                "behavior": behavior,
                "reset_raw_frame": start,
                "clip_start_raw": start,
                "clip_end_raw": end,
                "phase_ranges": [
                    {
                        "phase": candidate["pre_phase"],
                        "start": start,
                        "end": candidate["transition_start_raw"],
                    },
                    {
                        "phase": candidate["transition_phase"],
                        "start": candidate["transition_start_raw"],
                        "end": candidate["transition_end_raw"],
                    },
                    {
                        "phase": candidate["post_phase"],
                        "start": candidate["transition_end_raw"],
                        "end": end,
                    },
                ],
                "source_raw_npz": str(path),
            }
        )
        result.append(item)
    return result


def temporary_record(
    record: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    source_path: Path,
    start: int,
    frame_count: int,
    reference_keys: Sequence[str],
    reference_record: Mapping[str, Any],
    target_order: Sequence[str],
    target_axes: np.ndarray,
    seq_length: int,
) -> dict[str, Any]:
    end = start + frame_count
    converted, _ = convert_record(
        metadata,
        raw,
        reference_keys,
        reference_record,
        target_order,
        target_axes,
        start,
        end,
        source_path,
        seq_length,
        frame_count,
    )
    converted.update(
        {
            key: record[key]
            for key in (
                "source_round",
                "source_rollout",
                "source_episode",
                "physics_seed",
                "dataset_split",
            )
            if key in record
        }
    )
    converted.update(
        {
            "source_raw_npz": str(source_path),
            "source_start_frame": start,
            "source_end_frame": end,
            "phase_label": record.get("phase_label", "mixed"),
            "phase_id": np.asarray(raw["phase_id"][start:end], dtype=np.int16),
            "phase_value": np.asarray(raw["phase_value"][start:end], dtype=np.float32),
        }
    )
    return converted


def build_temp_motionlib(
    cases: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    resolver: RawResolver,
    agent: Any,
    directory: Path,
    seq_length: int,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    reference_path = REPOSITORY_ROOT / "train/dataset/base/lafan_29dof_10s-clipped.pkl"
    robot_xml = (
        REPOSITORY_ROOT / "train/scripts/isaac_env/humanoidverse/data/robots/g1/g1_29dof.xml"
    )
    reference_keys, reference_record = load_reference(reference_path)
    target_order, target_axes = bfm_joint_contract(robot_xml)
    converted: dict[str, dict[str, Any]] = {}
    for case in cases:
        source_key = case.get("transition_motion_key", case["motion_key"])
        source_record = records[source_key]
        raw, metadata, path = resolver.load(source_record)
        start = int(case["reset_raw_frame"])
        steps = int(case.get("steps", EVAL_STEPS))
        frame_count = int(case.get("bridge_frame_count", steps + seq_length + 1))
        if case.get("clip_end_raw") is not None:
            visible_end = int(case["clip_end_raw"])
            frame_count = max(frame_count, visible_end - start + seq_length)
        if start + frame_count > len(raw["qpos"]):
            raise RuntimeError(f"Raw window cannot cover case {case['case_id']}.")
        if np.any(raw["reset"][start : start + frame_count]) or np.any(
            raw["fall"][start : start + frame_count]
        ):
            raise RuntimeError(f"Raw window contains reset/fall for case {case['case_id']}.")
        record = temporary_record(
            source_record,
            raw,
            metadata,
            path,
            start,
            frame_count,
            reference_keys,
            reference_record,
            target_order,
            target_axes,
            seq_length,
        )
        record["source_start_frame"] = start
        record["source_end_frame"] = start + frame_count
        converted[str(case["case_id"])] = record
    path = directory / "raw_window_motionlib.pkl"
    joblib.dump(converted, path)
    return (
        path,
        converted,
        {
            "converter": str(
                REPOSITORY_ROOT / "train/scripts/data_collection/convert_continuous.py"
            ),
            "converter_sha256": hash_file(
                REPOSITORY_ROOT / "train/scripts/data_collection/convert_continuous.py"
            ),
            "reference": str(reference_path),
            "reference_sha256": hash_file(reference_path),
            "robot_xml": str(robot_xml),
            "robot_xml_sha256": hash_file(robot_xml),
            "temporary_motionlib_sha256": hash_file(path),
        },
    )


def checkpoint_mutation(agent: Any) -> dict[str, Any]:
    return {
        "parameters": hash_params(agent._model),
        "buffers": hash_buffers(agent._model),
        "normalizer": hash_buffers(agent._model._obs_normalizer),
        "components": hash_components(agent),
    }


class ControlDiagnostics:
    groups = ("waist", "hip", "ankle")

    def __init__(self, env: HuskyBfmOnlineEnv) -> None:
        report = env.env.physical_actuator_report
        if len(report) != 23:
            raise RuntimeError("Expected 23 HUSKY actuators.")
        self.names = tuple(str(item["joint_name"]).removeprefix("robot/") for item in report)
        self.limits = np.asarray(
            [item["derived_joint_torque_limit"] for item in report], dtype=np.float64
        )
        self.dof = np.asarray(
            [
                int(env.env.model.joint(int(env.env.model.actuator_trnid[i, 0])).dofadr[0])
                for i in range(23)
            ],
            dtype=np.int32,
        )
        if (
            len(set(self.dof)) != 23
            or not np.isfinite(self.limits).all()
            or (self.limits <= 0).any()
        ):
            raise RuntimeError("Physical actuator mapping is invalid.")
        self.indices = {
            group: np.asarray(
                [i for i, name in enumerate(self.names) if group in name], dtype=np.int32
            )
            for group in self.groups
        }
        self.actions: list[np.ndarray] = []
        self.torques: list[np.ndarray] = []

    def update(self, action: torch.Tensor, qfrc: np.ndarray) -> None:
        action = np.asarray(action.detach().cpu(), dtype=np.float64)
        torque = np.asarray(qfrc, dtype=np.float64)[self.dof]
        if (
            action.shape != (23,)
            or torque.shape != (23,)
            or not np.isfinite(action).all()
            or not np.isfinite(torque).all()
        ):
            raise RuntimeError("Invalid physical control diagnostic.")
        self.actions.append(np.abs(action))
        self.torques.append(np.abs(torque) / self.limits)

    def summary(self) -> dict[str, Any]:
        action = np.asarray(self.actions)
        torque = np.asarray(self.torques)
        result = {
            "actuator_joint_names": list(self.names),
            "action_soft_saturation_fraction": float(np.mean(action >= 0.95)),
            "action_hard_saturation_fraction": float(np.mean(action >= 0.99)),
            "torque_utilization_mean": float(torque.mean()),
            "torque_utilization_p95": float(np.quantile(torque, 0.95)),
            "torque_utilization_p99": float(np.quantile(torque, 0.99)),
            "torque_utilization_max": float(torque.max()),
            "torque_utilization_ge_0_95_fraction": float(np.mean(torque >= 0.95)),
            "groups": {},
        }
        for group, indices in self.indices.items():
            values = torque[:, indices].reshape(-1)
            actions = action[:, indices].reshape(-1)
            result["groups"][group] = {
                "joint_names": [self.names[i] for i in indices],
                "action_soft_saturation_fraction": float(np.mean(actions >= 0.95)),
                "action_hard_saturation_fraction": float(np.mean(actions >= 0.99)),
                "torque_utilization_p95": float(np.quantile(values, 0.95)),
                "torque_utilization_p99": float(np.quantile(values, 0.99)),
                "torque_utilization_ge_0_95_fraction": float(np.mean(values >= 0.95)),
            }
        return result


def metric_series(
    actual: Sequence[Mapping[str, Any]], raw: Mapping[str, np.ndarray], reset: int
) -> dict[str, np.ndarray]:
    ref = slice(reset + 1, reset + len(actual) + 1)
    if reset < 0 or reset + len(actual) >= len(raw["qpos"]):
        raise RuntimeError("Raw metric reference is outside the source.")
    root = np.asarray([row["root_position"] for row in actual])
    board = np.asarray([row["board_position"] for row in actual])
    root_q = np.asarray([row["root_quaternion"] for row in actual])
    board_q = np.asarray([row["board_quaternion"] for row in actual])
    joint = np.asarray([row["joint_position"] for row in actual])
    velocity = np.asarray([row["joint_velocity"] for row in actual])
    board_lin = np.asarray([row["board_linear_velocity"] for row in actual])
    board_ang = np.asarray([row["board_angular_velocity"] for row in actual])
    raw_root = raw["root_pos"][ref]
    raw_board = raw["board_root_pos"][ref]
    return {
        "joint_position_mae_rad": np.mean(np.abs(joint - raw["dof_pos"][ref]), axis=1),
        "joint_velocity_mae_rad_s": np.mean(np.abs(velocity - raw["dof_vel"][ref]), axis=1),
        "root_xy_displacement_error_m": np.linalg.norm(
            (root[:, :2] - raw["root_pos"][reset, :2])
            - (raw_root[:, :2] - raw["root_pos"][reset, :2]),
            axis=1,
        ),
        "root_z_error_m": np.abs(root[:, 2] - raw_root[:, 2]),
        "root_orientation_geodesic_error_deg": quaternion_geodesic_deg(
            root_q, raw["root_quat"][ref]
        ),
        "board_xy_displacement_error_m": np.linalg.norm(
            (board[:, :2] - raw["board_root_pos"][reset, :2])
            - (raw_board[:, :2] - raw["board_root_pos"][reset, :2]),
            axis=1,
        ),
        "board_orientation_geodesic_error_deg": quaternion_geodesic_deg(
            board_q, raw["board_root_quat"][ref]
        ),
        "board_linear_velocity_error_mps": np.linalg.norm(
            board_lin - raw["board_root_lin_vel"][ref], axis=1
        ),
        "board_angular_velocity_error_rad_s": np.linalg.norm(
            board_ang - raw["board_root_ang_vel"][ref], axis=1
        ),
        "board_heading_error_deg": wrapped_angle_deg(
            np.asarray([board_yaw(q) for q in board_q]),
            np.asarray([board_yaw(q) for q in raw["board_root_quat"][ref]]),
        ),
        "board_tilt_error_deg": np.abs(
            np.asarray([board_tilt_deg(q) for q in board_q])
            - np.asarray([board_tilt_deg(q) for q in raw["board_root_quat"][ref]])
        ),
        "coupling_xy_error_m": np.linalg.norm(
            (root[:, :2] - board[:, :2]) - (raw_root[:, :2] - raw_board[:, :2]), axis=1
        ),
        "coupling_z_error_m": np.abs(
            (root[:, 2] - board[:, 2]) - (raw_root[:, 2] - raw_board[:, 2])
        ),
    }


def retention_stability(
    actual: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    feet = np.asarray([bool(row["feet_on_board"]) for row in actual])
    off = ~feet
    heights = np.asarray([row["root_height"] for row in actual], dtype=np.float64)
    tilt = np.asarray(
        [
            np.degrees(np.arccos(np.clip(-np.asarray(row["projected_gravity"])[2], -1, 1)))
            for row in actual
        ]
    )
    root = np.asarray([row["root_position"] for row in actual])
    board = np.asarray([row["board_position"] for row in actual])
    separation = np.linalg.norm(root[:, :2] - board[:, :2], axis=1)
    first = np.flatnonzero(off)
    streak = current = 0
    for value in off:
        current = current + 1 if value else 0
        streak = max(streak, current)
    first_index = int(first[0]) if first.size else None
    return {
        "feet_on_board_ratio": float(feet.mean()),
        "off_board_ratio": float(off.mean()),
        "final_feet_on_board": bool(feet[-1]),
        "time_to_first_off_board_s": None
        if first_index is None
        else (first_index + 1) * CONTROL_DT,
        "longest_off_board_streak_s": streak * CONTROL_DT,
        "robot_board_separation_mean_m": float(separation.mean()),
        "robot_board_separation_final_m": float(separation[-1]),
        "robot_board_separation_max_m": float(separation.max()),
    }, {
        "root_tilt_mean_deg": float(tilt.mean()),
        "root_tilt_p95_deg": float(np.quantile(tilt, 0.95)),
        "root_tilt_max_deg": float(tilt.max()),
        "root_height_min_m": float(heights.min()),
        "illegal_contact_ratio": float(np.mean([bool(row["illegal_contact"]) for row in actual])),
        "off_board_upright_ratio": (
            float(np.mean((tilt[off] < 70) & (heights[off] >= 0.45))) if off.any() else None
        ),
        "post_first_off_board_survival_s": (
            None if first_index is None else (len(actual) - first_index - 1) * CONTROL_DT
        ),
    }


def section_metrics(
    series: Mapping[str, np.ndarray], phases: Sequence[str], wanted: str
) -> dict[str, dict[str, float]] | None:
    indices = np.asarray([i for i, phase in enumerate(phases) if phase == wanted], dtype=np.int64)
    return (
        None
        if not indices.size
        else {name: summarize(values[indices]) for name, values in series.items()}
    )


def open_writer(path: Path, env: HuskyBfmOnlineEnv) -> tuple[Any, mujoco.Renderer]:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", macro_block_size=1)
    return writer, mujoco.Renderer(env.env.model, height=720, width=1280)


def render(writer: Any, renderer: mujoco.Renderer, env: HuskyBfmOnlineEnv) -> None:
    renderer.update_scene(env.env.data, camera="robot/tracking")
    writer.append_data(renderer.render())


def run_policy(
    agent: Any,
    tracking: AlignedSkateTrackingContext,
    tracking_key: str,
    raw: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    case: Mapping[str, Any],
    env: HuskyBfmOnlineEnv,
    steps: int,
    model_video: Path | None = None,
) -> dict[str, Any]:
    z, ranges = tracking.encode(agent._model, tracking_key, 0, steps)
    if z.shape != (steps, 256) or not torch.isfinite(z).all():
        raise RuntimeError("Temporary raw-window tracking z is invalid.")
    if any(item["future_start"] != index + 1 for index, item in enumerate(ranges)):
        raise RuntimeError("Tracking future timing is not t+1 aligned.")
    trajectory = tracking.trajectories[tracking_key]
    final_context = trajectory["length"] - steps
    if final_context < agent._model.cfg.seq_length:
        raise RuntimeError("Temporary MotionLib lacks the required final seq-length context.")
    reset = int(case["reset_raw_frame"])
    qpos = np.asarray(raw["qpos"][reset], dtype=np.float64)
    qvel = np.asarray(raw["qvel"][reset], dtype=np.float64)
    writer = renderer = None
    actual = []
    first_action = None
    terminated = truncated = False
    try:
        observation = env.reset(
            qpos=qpos, qvel=qvel, source_physics=metadata["physics_randomization"]
        )
        if not (
            np.allclose(env.env.data.qpos, qpos, atol=1e-8, rtol=0)
            and np.allclose(env.env.data.qvel, qvel, atol=1e-8, rtol=0)
        ):
            raise RuntimeError("Canonical raw reset mismatch.")
        diagnostics = ControlDiagnostics(env)
        if model_video:
            writer, renderer = open_writer(model_video, env)
        for step in range(steps):
            model_observation = {
                key: value.unsqueeze(0).to(agent.device) for key, value in observation.items()
            }
            with torch.no_grad():
                action = agent.act(model_observation, z[step].unsqueeze(0), mean=True)[0]
            if action.shape != (29,) or not torch.isfinite(action).all():
                raise RuntimeError("Frozen actor produced invalid action.")
            if first_action is None:
                first_action = hash_data(action)
            transition = env.step(action, z[step], truncated=step == steps - 1)
            diagnostics.update(transition.action_husky, env.env.data.qfrc_actuator)
            actual.append(dict(transition.raw_metadata))
            if writer:
                render(writer, renderer, env)
            observation = transition.next_observation
            terminated, truncated = transition.terminated, transition.truncated
            if terminated or truncated:
                break
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    if not actual:
        raise RuntimeError("Policy rollout executed no transitions.")
    series = metric_series(actual, raw, reset)
    phases = []
    for index in range(reset + 1, reset + len(actual) + 1):
        phases.append(str(case.get("frame_phase", {}).get(str(index), "")))
    result = {
        "case": dict(case),
        "evaluation": {
            "control_dt": CONTROL_DT,
            "T_eval": steps,
            "T_exec": len(actual),
            "completion_ratio": len(actual) / steps,
            "full_completion": len(actual) == steps and not terminated,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "fall_reason": str(actual[-1].get("fall_reason", "")) if terminated else "",
        },
        "metrics": {
            "full": {name: summarize(values) for name, values in series.items()},
            "pre": section_metrics(series, phases, case.get("pre_phase", "")),
            "transition": section_metrics(series, phases, case.get("transition_phase", "")),
            "post": section_metrics(series, phases, case.get("post_phase", "")),
        },
        "retention": retention_stability(actual)[0],
        "stability": retention_stability(actual)[1],
        "control": diagnostics.summary(),
        "tracking": {
            "z_shape": list(z.shape),
            "finite": True,
            "future_start_first": ranges[0]["future_start"],
            "future_start_last": ranges[-1]["future_start"],
        },
        "alignment": {
            "canonical_reset": True,
            "post_step_reference_start": reset + 1,
            "post_step_reference_end": reset + len(actual),
            "t_plus_one": True,
        },
        "first_action_fingerprint": first_action,
    }
    return {
        "result": result,
        "actual": actual,
        "raw": raw,
        "metadata": metadata,
        "source_path": case["source_raw_npz"],
    }


def render_expert(path: Path, run: Mapping[str, Any], frames: int) -> None:
    env = HuskyBfmOnlineEnv()
    writer = renderer = None
    raw = run["raw"]
    reset = int(run["result"]["case"]["reset_raw_frame"])
    try:
        env.reset(
            qpos=raw["qpos"][reset],
            qvel=raw["qvel"][reset],
            source_physics=run["metadata"]["physics_randomization"],
        )
        writer, renderer = open_writer(path, env)
        for frame in range(reset + 1, reset + frames + 1):
            env.env.data.qpos[:] = raw["qpos"][frame]
            env.env.data.qvel[:] = raw["qvel"][frame]
            mujoco.mj_forward(env.env.model, env.env.data)
            render(writer, renderer, env)
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
        env.close()


def build_frame_phase(
    record: Mapping[str, Any], ranges: Sequence[Mapping[str, Any]], reset: int, end: int
) -> dict[str, str]:
    result = {}
    for item in ranges:
        for frame in range(int(item["start"]), int(item["end"])):
            result[str(frame)] = str(item["phase"])
    return result


def prepare_case(
    case: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    resolver: RawResolver,
    steps: int,
    seq_length: int,
    visible_end: int | None = None,
) -> dict[str, Any]:
    result = dict(case)
    source_key = str(case.get("transition_motion_key", case["motion_key"]))
    source = records[source_key]
    raw, metadata, path = resolver.load(source)
    start = int(case["reset_raw_frame"])
    bridge_end = start + steps + seq_length + 1
    if visible_end is not None:
        bridge_end = max(bridge_end, visible_end + seq_length)
    if bridge_end > len(raw["qpos"]):
        raise RuntimeError(f"Case raw window is too short: {case['case_id']}")
    result["source_raw_npz"] = str(path)
    result["source_physics"] = metadata["physics_randomization"]
    result["steps"] = steps
    result["frame_phase"] = {}
    for item in case.get("phase_ranges", []):
        for frame in range(int(item["start"]), int(item["end"])):
            result["frame_phase"][str(frame)] = str(item["phase"])
    return result


def make_cases(
    command: str,
    records: Mapping[str, Mapping[str, Any]],
    resolver: RawResolver,
    args: argparse.Namespace,
    seq_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if command == "eval":
        steady_steps = round(args.steady_duration_s / CONTROL_DT)
        if steady_steps <= 0:
            raise ValueError("steady-duration-s must be positive.")
        bank, excluded_by_behavior = formal_candidates(
            records, resolver, steady_steps, seq_length
        )
        if args.case_spec:
            if args.max_cases_per_behavior is not None:
                raise RuntimeError("case-spec already fixes case selection.")
            payload = json.loads(args.case_spec.read_text())
            cases = payload.get("cases", [])
            if not cases:
                raise RuntimeError("Case spec has no cases.")
            normalized = []
            for case in cases:
                item = dict(case)
                if "behavior" not in item and item.get("transition") in BEHAVIORS:
                    item["behavior"] = item["transition"]
                if item.get("behavior") not in BEHAVIORS:
                    raise RuntimeError(f"Case spec has invalid behavior: {item}")
                steps = int(item.get("steps", EVAL_STEPS))
                normalized.append(
                    prepare_case(item, records, resolver, steps, seq_length)
                )
            selected_by_behavior = {
                behavior: sorted(
                    [
                        case
                        for case in normalized
                        if case["behavior"] == behavior
                    ],
                    key=case_sort_key,
                )
                for behavior in BEHAVIORS
            }
            selection = selection_stats(
                bank,
                selected_by_behavior,
                "case_spec",
                None,
                None,
            )
            selection["excluded_reasons"] = {}
            return normalized, selection
        selected_behaviors = BEHAVIORS if args.behavior == "all" else (args.behavior,)
        selected_by_behavior: dict[str, list[Mapping[str, Any]]] = {}
        for name in BEHAVIORS:
            selected_by_behavior[name] = (
                select_balanced_cases(
                    bank[name],
                    args.max_cases_per_behavior,
                    args.seed,
                    name,
                )
                if name in selected_behaviors
                else []
            )
        cases = []
        for name in selected_behaviors:
            cases.extend(
                prepare_case(item, records, resolver, int(item["steps"]), seq_length)
                for item in selected_by_behavior[name]
            )
        cases.sort(
            key=lambda case: (
                BEHAVIORS.index(str(case["behavior"])),
                case_sort_key(case),
            )
        )
        mode = (
            "rollout_balanced_without_replacement"
            if args.max_cases_per_behavior is not None
            and any(
                len(selected_by_behavior[name]) < len(bank[name])
                for name in selected_behaviors
            )
            else "all"
        )
        selection = selection_stats(
            bank,
            selected_by_behavior,
            mode,
            args.seed,
            args.max_cases_per_behavior,
        )
        selection["excluded_reasons"] = {
            f"{behavior}:{reason}": count
            for behavior, values in excluded_by_behavior.items()
            for reason, count in values.items()
        }
        return cases, selection

    cases = []
    behaviors = BEHAVIORS if args.behavior == "all" else (args.behavior,)
    for behavior in behaviors:
        if behavior in ("push2steer", "steer2push"):
            options = behavior_candidates(
                records,
                resolver,
                behavior,
                1,
                round(args.context_s / CONTROL_DT),
                args.steer_direction,
                seq_length,
            )
            if not options:
                raise RuntimeError(f"No Val {behavior} video candidate.")
            selected = options[
                int(np.random.default_rng(args.seed + len(cases)).integers(len(options)))
            ]
            start, end = selected["clip_start_raw"], selected["clip_end_raw"]
            selected = prepare_case(
                selected, records, resolver, end - start - 1, seq_length, end
            )
        else:
            steps = round(args.clip_s / CONTROL_DT)
            options = behavior_candidates(
                records, resolver, behavior, steps, 0, args.steer_direction, seq_length
            )
            if not options:
                raise RuntimeError(f"No Val {behavior} video candidate.")
            selected = prepare_case(
                options[0], records, resolver, steps, seq_length, options[0]["clip_end_raw"]
            )
        cases.append(selected)
    return cases, {}


def training_date(summary_path: Path) -> str:
    summary = json.loads(summary_path.read_text())
    started = summary.get("run_provenance", {}).get("started_at")
    if not isinstance(started, str) or len(started) < 10:
        raise RuntimeError("Training summary lacks run_provenance.started_at.")
    return started[:10]


def output_root(args: argparse.Namespace) -> Path:
    date = training_date(args.training_summary)
    suffix = (
        "test_phase_eval"
        if args.command == "eval"
        else "videos"
        if args.command == "video"
        else "viewer"
    )
    return REPOSITORY_ROOT / "train/eval_res" / date / f"{args.label}_{suffix}"


def invocation_config(
    args: argparse.Namespace,
    phase_path: Path,
    manifest_path: Path,
    raw_root: Path,
    checkpoint: Path,
    checkpoint_sha: str,
    provenance: Mapping[str, Any],
    candidates: Mapping[str, int],
    excluded: Mapping[str, int],
    full_test: bool,
    full_behavior: Mapping[str, bool],
    steady_steps: int | None = None,
    case_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "command": args.command,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "evaluator_sha256": hash_file(Path(__file__)),
        "checkpoint": {"label": args.label, "path": str(checkpoint), "sha256": checkpoint_sha},
        "training_summary": str(args.training_summary.resolve()),
        "training_date": training_date(args.training_summary),
        "phase_motionlib": str(phase_path),
        "phase_manifest": str(manifest_path),
        "phase_manifest_sha256": hash_file(manifest_path),
        "phase_motionlib_sha256": hash_file(phase_path),
        "raw_root": str(raw_root),
        "phase_boundary_convention": "exact source_end_frame == next source_start_frame",
        "candidate_counts": dict(candidates),
        "excluded_reasons": dict(excluded),
        "tracking_bridge": provenance,
        "control_dt": CONTROL_DT,
        "eval_behaviors": args.behavior if args.command == "eval" else None,
        "steady_duration_s": args.steady_duration_s if args.command == "eval" else None,
        "steady_steps": steady_steps if args.command == "eval" else None,
        "transition_duration_s": EVAL_SECONDS if args.command == "eval" else None,
        "transition_steps": EVAL_STEPS if args.command == "eval" else None,
        "transition_lead_s": LEAD_SECONDS if args.command == "eval" else None,
        "transition_lead_steps": LEAD_STEPS if args.command == "eval" else None,
        "formal_full_test": full_test,
        "formal_full_behavior": dict(full_behavior),
        "case_selection": dict(case_selection or {}),
        "protocol": (
            {
                "push": {
                    "duration_s": args.steady_duration_s,
                    "duration_steps": steady_steps,
                },
                "steer": {
                    "duration_s": args.steady_duration_s,
                    "duration_steps": steady_steps,
                },
                "push2steer": {
                    "duration_s": EVAL_SECONDS,
                    "duration_steps": EVAL_STEPS,
                    "lead_s": LEAD_SECONDS,
                    "lead_steps": LEAD_STEPS,
                },
                "steer2push": {
                    "duration_s": EVAL_SECONDS,
                    "duration_steps": EVAL_STEPS,
                    "lead_s": LEAD_SECONDS,
                    "lead_steps": LEAD_STEPS,
                },
            }
            if args.command == "eval"
            else None
        ),
        "evaluation_only": True,
        "training": False,
        "test": args.command == "eval",
        "continuous_dataset_dependency": False,
        "raw_to_bfm_converter": "data_collection.convert_continuous.convert_record",
    }


def parity_audit(
    agent: Any,
    phase_records: Mapping[str, Mapping[str, Any]],
    resolver: RawResolver,
    temp_path: Path,
    temp_records: Mapping[str, Mapping[str, Any]],
    phase_tracking: AlignedSkateTrackingContext,
    temp_tracking: AlignedSkateTrackingContext,
) -> dict[str, Any]:
    max_diff = {"state": 0.0, "privileged_state": 0.0, "last_action": 0.0}
    max_z_diff = 0.0
    max_action_diff = 0.0
    first_action_match = True
    selected = []
    env = HuskyBfmOnlineEnv()
    for label in ("push", "push2steer", "steer_left", "steer2push"):
        key = next(
            key
            for key, value in phase_records.items()
            if value["phase_label"] == label and len(value["dof"]) >= 10
        )
        temp_key = f"parity_{label}"
        # The temporary bridge is loaded once by the caller; parity uses its explicit key.
        if temp_key not in temp_records:
            raise RuntimeError(f"Missing parity bridge record: {temp_key}")
        old = phase_tracking.trajectories[key]
        new = temp_tracking.trajectories[temp_key]
        rows = min(old["length"], new["length"])
        old_start, new_start = old["start"], new["start"]
        for name in ("state", "privileged_state", "last_action"):
            diff = torch.max(
                torch.abs(
                    phase_tracking.observations[name][old_start : old_start + rows]
                    - temp_tracking.observations[name][new_start : new_start + rows]
                )
            ).item()
            max_diff[name] = max(max_diff[name], float(diff))
        steps = min(5, old["length"] - 1)
        old_z, old_ranges = phase_tracking.encode(agent._model, key, 0, steps)
        new_z, new_ranges = temp_tracking.encode(agent._model, temp_key, 0, steps)
        if (
            old_z.shape != new_z.shape
            or not torch.isfinite(old_z).all()
            or not torch.isfinite(new_z).all()
        ):
            raise RuntimeError(f"Invalid parity z for {label}.")
        max_z_diff = max(max_z_diff, float(torch.max(torch.abs(old_z - new_z)).item()))
        if any(
            old_item["future_start"] != new_item["future_start"]
            for old_item, new_item in zip(old_ranges, new_ranges, strict=True)
        ):
            raise RuntimeError(f"Parity future timing differs for {label}.")
        raw, metadata, _ = resolver.load(phase_records[key])
        reset_frame = int(phase_records[key]["source_start_frame"])
        observation = env.reset(
            qpos=np.asarray(raw["qpos"][reset_frame], dtype=np.float64),
            qvel=np.asarray(raw["qvel"][reset_frame], dtype=np.float64),
            source_physics=metadata["physics_randomization"],
        )
        model_observation = {
            name: value.unsqueeze(0).to(agent.device) for name, value in observation.items()
        }
        with torch.no_grad():
            old_action = agent.act(model_observation, old_z[0].unsqueeze(0), mean=True)[0]
            new_action = agent.act(model_observation, new_z[0].unsqueeze(0), mean=True)[0]
        action_diff = float(torch.max(torch.abs(old_action - new_action)).item())
        max_action_diff = max(max_action_diff, action_diff)
        first_action_match = first_action_match and torch.allclose(
            old_action, new_action, atol=1e-4, rtol=1e-4
        )
        selected.append({"phase": label, "motion_key": key, "rows_compared": rows})
    env.close()
    if (
        max_diff["state"] > 1e-5
        or max_diff["privileged_state"] > 1e-5
        or max_diff["last_action"] > 0
    ):
        raise RuntimeError(f"Raw-window converter parity failed: {max_diff}")
    if max_z_diff > 1e-3 or not first_action_match:
        raise RuntimeError(
            f"Tracking parity failed: z_max_abs_diff={max_z_diff}, "
            f"action_max_abs_diff={max_action_diff}, "
            f"first_action_match={first_action_match}"
        )
    return {
        "status": "PASS",
        "max_abs_diff": max_diff,
        "z_max_abs_diff": max_z_diff,
        "first_action_max_abs_diff": max_action_diff,
        "first_action_match": first_action_match,
        "selected": selected,
        "last_action_zero": True,
    }


def write_expert_video(path: Path, run: Mapping[str, Any], steps: int) -> None:
    render_expert(path, run, steps)


def write_summary_markdown(
    path: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    aggregation = summary["aggregation"]

    def value(behavior: str, metric: str) -> str:
        group = aggregation.get(behavior, {})
        metric_data = group.get(metric)
        return "-" if not metric_data else f"{metric_data['mean']:.5g}"

    def completion(behavior: str) -> str:
        data = aggregation.get(behavior, {}).get("completion", {})
        if not data:
            return "-"
        return f"{data['full_completion_rate']:.3f}"

    lines = [
        "# Formal Phase Evaluation",
        "",
        "| Behavior | Cases | Full completion | Joint MAE | Root Ori | "
        "Board XY | Coupling XY | Feet on board |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for behavior in BEHAVIORS:
        group = aggregation.get(behavior, {})
        count = group.get("completion", {}).get("case_count", 0)
        retention = group.get("retention", {}).get("feet_on_board_ratio", {}).get("mean")
        lines.append(
            f"| {behavior} | {count} | {completion(behavior)} | "
            f"{value(behavior, 'joint_position_mae_rad')} | "
            f"{value(behavior, 'root_orientation_geodesic_error_deg')} | "
            f"{value(behavior, 'board_xy_displacement_error_m')} | "
            f"{value(behavior, 'coupling_xy_error_m')} | "
            f"{'-' if retention is None else f'{retention:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "## Steer",
            "",
            "| Direction | Cases | Joint MAE | Full completion |",
            "|---|---:|---:|---:|",
        ]
    )
    for direction in ("left", "forward", "right"):
        key = "steer_by_direction"
        group = aggregation.get(key, {}).get(direction, {})
        count = group.get("completion", {}).get("case_count", 0)
        rate = group.get("completion", {}).get("full_completion_rate")
        mae = group.get("joint_position_mae_rad", {}).get("mean")
        lines.append(
            f"| {direction} | {count} | {'-' if mae is None else f'{mae:.5g}'} | "
            f"{'-' if rate is None else f'{rate:.3f}'} |"
        )
    lines.extend(["", "## Transition Sections", ""])
    for behavior in ("push2steer", "steer2push"):
        lines.append(f"### {behavior}")
        group = aggregation.get(behavior, {})
        section_lines = []
        for section in ("pre", "transition", "post"):
            section_rows = [
                row["metrics"][section]
                for row in rows
                if row["case"]["behavior"] == behavior and row["metrics"].get(section)
            ]
            metric_values = [
                item["joint_position_mae_rad"]["mean"] for item in section_rows
            ]
            section_lines.append(
                f"| {section} | {len(section_rows)} | "
                f"{'-' if not metric_values else f'{np.mean(metric_values):.5g}'} |"
            )
        lines.extend(
            [
                "",
                "| Section | Cases | Joint MAE |",
                "|---|---:|---:|",
                *section_lines,
            ]
        )
        lines.append("")
    lines.extend(
        [
            "## Protocol",
            "",
            f"- Full test: `{summary.get('formal_full_test', False)}`",
            f"- Tracking parity: `{summary.get('tracking_parity', {}).get('status', 'UNKNOWN')}`",
            f"- Training: `{summary.get('training', False)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"count": 0}
        result = {}
        for metric in REFERENCE_METRICS:
            values = np.asarray([item["metrics"]["full"][metric]["mean"] for item in items])
            result[metric] = summarize(values)
        completion = np.asarray([item["evaluation"]["completion_ratio"] for item in items])
        result["completion"] = {
            "case_count": len(items),
            "full_completion_count": int(
                sum(item["evaluation"]["full_completion"] for item in items)
            ),
            "full_completion_rate": float(
                np.mean([item["evaluation"]["full_completion"] for item in items])
            ),
            "mean_completion_ratio": float(completion.mean()),
            "termination_count": int(sum(item["evaluation"]["terminated"] for item in items)),
            "termination_rate": float(
                np.mean([item["evaluation"]["terminated"] for item in items])
            ),
        }
        for section in ("retention", "stability"):
            keys = set().union(*(item[section] for item in items))
            result[section] = {
                key: summarize(np.asarray([item[section][key] for item in items]))
                for key in sorted(keys)
                if all(
                    item[section].get(key) is not None
                    and np.isfinite(float(item[section][key]))
                    for item in items
                )
            }
        return result

    result = {"all": aggregate(rows)}
    for behavior in BEHAVIORS:
        result[behavior] = aggregate(
            [row for row in rows if row["case"]["behavior"] == behavior]
        )
    result["steer_by_direction"] = {
        direction: aggregate(
            [
                row
                for row in rows
                if row["case"]["behavior"] == "steer"
                and row["case"].get("steer_direction") == direction
            ]
        )
        for direction in ("left", "forward", "right")
    }
    return result


def select_representatives(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    selected = {}
    for behavior in BEHAVIORS:
        items = [row for row in rows if row["case"]["behavior"] == behavior]
        if not items:
            continue
        complete = [row for row in items if row["evaluation"]["full_completion"]]
        if complete:
            median = float(
                np.median(
                    [row["metrics"]["full"]["joint_position_mae_rad"]["mean"] for row in complete]
                )
            )
            selected[behavior] = min(
                complete,
                key=lambda row: (
                    abs(row["metrics"]["full"]["joint_position_mae_rad"]["mean"] - median),
                    row["case"]["case_id"],
                ),
            )
        else:
            median = float(np.median([row["evaluation"]["completion_ratio"] for row in items]))
            selected[behavior] = min(
                items,
                key=lambda row: (
                    abs(row["evaluation"]["completion_ratio"] - median),
                    row["metrics"]["full"]["joint_position_mae_rad"]["mean"],
                    row["case"]["case_id"],
                ),
            )
    return selected


def run_eval(args: argparse.Namespace) -> int:
    records, manifest, phase_path, manifest_path = load_phase_records("test")
    resolver = RawResolver("test")
    agent, load_report = load_frozen_agent(args.checkpoint.resolve())
    seq_length = tracking_seq_length(agent)
    if agent._model.training or any(
        parameter.requires_grad for parameter in agent._model.parameters()
    ):
        raise RuntimeError("Checkpoint is not frozen.")
    steady_steps = round(args.steady_duration_s / CONTROL_DT)
    if steady_steps <= 0:
        raise ValueError("steady-duration-s must be positive.")
    selected_behaviors = BEHAVIORS if args.behavior == "all" else (args.behavior,)
    cases, selection = make_cases("eval", records, resolver, args, seq_length)
    if not cases:
        counts = {
            name: selection.get("eligible_counts", {}).get(name, 0)
            for name in selected_behaviors
        }
        raise RuntimeError(f"No eligible cases for {args.behavior}: {counts}")
    candidate_counts = selection["eligible_counts"]
    excluded = selection["excluded_reasons"]
    limit = args.max_cases_per_behavior
    full_behavior = {
        name: (
            name in selected_behaviors
            and selection["selected_counts"][name] == selection["eligible_counts"][name]
            and args.case_spec is None
        )
        for name in BEHAVIORS
    }
    formal_full_test = args.behavior == "all" and limit is None and args.case_spec is None
    with tempfile.TemporaryDirectory(prefix="skate_bfm_eval_") as temp:
        temp_dir = Path(temp)
        temp_cases = []
        for case in cases:
            temp_cases.append(dict(case))
        # Include explicit parity records in the same official MotionLib load.
        for label in ("push", "push2steer", "steer_left", "steer2push"):
            key = next(
                key
                for key, value in records.items()
                if value["phase_label"] == label and len(value["dof"]) >= 10
            )
            temp_cases.append(
                {
                    "case_id": f"parity_{label}",
                    "motion_key": key,
                    "transition_motion_key": key,
                    "reset_raw_frame": int(records[key]["source_start_frame"]),
                    "steps": min(20, len(records[key]["dof"]) - 2),
                    "bridge_frame_count": len(records[key]["dof"]),
                }
            )
        temp_path, temp_records, bridge = build_temp_motionlib(
            temp_cases, records, resolver, agent, temp_dir, seq_length
        )
        temp_tracking = AlignedSkateTrackingContext.load(agent, temp_path)
        phase_tracking = AlignedSkateTrackingContext.load(agent, phase_path)
        bridge["coverage"] = {
            case["case_id"]: {
                "raw_frames": temp_tracking.trajectories[case["case_id"]]["raw_frames"],
                "observation_rows": temp_tracking.trajectories[case["case_id"]]["length"],
                "required_policy_steps": int(case["steps"]),
                "seq_length": agent._model.cfg.seq_length,
                "final_future_context": (
                    temp_tracking.trajectories[case["case_id"]]["length"] - int(case["steps"])
                ),
                "gap_free": True,
            }
            for case in cases
        }
        parity = parity_audit(
            agent, records, resolver, temp_path, temp_records, phase_tracking, temp_tracking
        )
        env = HuskyBfmOnlineEnv()
        before = checkpoint_mutation(agent)
        rows = []
        try:
            for index, case in enumerate(cases):
                source = records[str(case.get("transition_motion_key", case["motion_key"]))]
                raw, metadata, _ = resolver.load(source)
                run = run_policy(
                    agent,
                    temp_tracking,
                    case["case_id"],
                    raw,
                    metadata,
                    case,
                    env,
                    int(case["steps"]),
                )
                result = run["result"]
                result["checkpoint"] = {
                    "label": args.label,
                    "path": str(args.checkpoint.resolve()),
                    "sha256": hash_file(checkpoint_model_path(args.checkpoint.resolve())),
                }
                result["load_report"] = load_report
                rows.append(result)
                print(f"\rEval {index + 1}/{len(cases)}", end="", flush=True)
        finally:
            env.close()
        print()
        after = checkpoint_mutation(agent)
        if before != after:
            raise RuntimeError("Frozen model mutated during Test evaluation.")
        root = output_root(args)
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_sha = hash_file(checkpoint_model_path(args.checkpoint.resolve()))
        write_json(
            root / "cases.json",
            {
                "protocol": {
                    "steady_duration_s": args.steady_duration_s,
                    "steady_steps": steady_steps,
                    "transition_duration_s": EVAL_SECONDS,
                    "transition_steps": EVAL_STEPS,
                    "transition_lead_s": LEAD_SECONDS,
                    "transition_lead_steps": LEAD_STEPS,
                },
                "selection": selection,
                "cases": cases,
                "formal_full_test": formal_full_test,
            },
        )
        config = invocation_config(
            args,
            phase_path,
            manifest_path,
            resolver.root,
            args.checkpoint.resolve(),
            checkpoint_sha,
            bridge,
            candidate_counts,
            excluded,
            formal_full_test,
            full_behavior,
            steady_steps,
            selection,
        )
        config["tracking_parity"] = parity
        write_json(root / "config.json", config)
        with (root / "results.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary = {
            "aggregation": aggregate_results(rows),
            "candidate_counts": candidate_counts,
            "excluded_reasons": excluded,
            "selection": {
                key: selection[key]
                for key in ("mode", "seed", "selection_fingerprint")
            },
            "source_group_coverage": {
                behavior: {
                    "eligible": selection["eligible_source_groups"][behavior],
                    "selected": selection["selected_unique_source_groups"][behavior],
                }
                for behavior in BEHAVIORS
            },
            "selected_direction_counts": selection["selected_direction_counts"],
            "selection_warnings": selection["warnings"],
            "tracking_parity": parity,
            "evaluation_only": True,
            "training": False,
            "mutation": {
                "parameters_changed": False,
                "buffers_changed": False,
                "normalizer_changed": False,
                "components_changed": False,
            },
            "protocol": {
                "push": {"duration_s": args.steady_duration_s, "duration_steps": steady_steps},
                "steer": {"duration_s": args.steady_duration_s, "duration_steps": steady_steps},
                "push2steer": {
                    "duration_s": EVAL_SECONDS,
                    "duration_steps": EVAL_STEPS,
                    "lead_s": LEAD_SECONDS,
                    "lead_steps": LEAD_STEPS,
                },
                "steer2push": {
                    "duration_s": EVAL_SECONDS,
                    "duration_steps": EVAL_STEPS,
                    "lead_s": LEAD_SECONDS,
                    "lead_steps": LEAD_STEPS,
                },
            },
            "full_test_executed": formal_full_test,
            "formal_full_test": formal_full_test,
            "formal_full_behavior": full_behavior,
        }
        selected = select_representatives(rows)
        summary["representative_selection_rule"] = (
            "full completion first; closest joint MAE to category median; "
            "otherwise closest completion ratio, then joint MAE, then case_id."
        )
        videos = root / "videos"
        replay_checks = {}
        for behavior, row in selected.items():
            case = row["case"]
            source = records[str(case.get("transition_motion_key", case["motion_key"]))]
            raw, metadata, _ = resolver.load(source)
            replay_env = HuskyBfmOnlineEnv()
            replay_before = checkpoint_mutation(agent)
            try:
                run = run_policy(
                    agent,
                    temp_tracking,
                    case["case_id"],
                    raw,
                    metadata,
                    case,
                    env=replay_env,
                    steps=int(case["steps"]),
                    model_video=videos / behavior / "model.mp4",
                )
            finally:
                replay_env.close()
            if replay_before != checkpoint_mutation(agent):
                raise RuntimeError("Frozen model mutated during representative replay.")
            replay_result = run["result"]
            checks = {
                "first_action_match": (
                    replay_result["first_action_fingerprint"] == row["first_action_fingerprint"]
                ),
                "T_exec_match": (
                    replay_result["evaluation"]["T_exec"] == row["evaluation"]["T_exec"]
                ),
                "termination_match": (
                    replay_result["evaluation"]["terminated"] == row["evaluation"]["terminated"]
                ),
            }
            for metric in (
                "joint_position_mae_rad",
                "root_orientation_geodesic_error_deg",
                "board_xy_displacement_error_m",
                "coupling_xy_error_m",
            ):
                checks[f"{metric}_match"] = bool(
                    np.isclose(
                        replay_result["metrics"]["full"][metric]["mean"],
                        row["metrics"]["full"][metric]["mean"],
                        atol=1e-10,
                        rtol=1e-10,
                    )
                )
            if not all(checks.values()):
                raise RuntimeError(f"Representative replay mismatch for {behavior}: {checks}")
            replay_checks[behavior] = checks
            write_expert_video(
                videos / behavior / "expert.mp4",
                run,
                int(case["steps"]),
            )
            write_json(videos / behavior / "case.json", case)
        summary["representative_replay"] = replay_checks
        summary["representative_selection"] = {
            behavior: row["case"]["case_id"] for behavior, row in selected.items()
        }
        write_json(root / "summary.json", summary)
        write_summary_markdown(root / "summary.md", summary, rows)
    print(
        json.dumps(
            {
                "status": "PASS",
                "formal_full_test": formal_full_test,
                "cases": len(rows),
                "output": str(output_root(args)),
            },
            indent=2,
        )
    )
    return 0


def run_video(args: argparse.Namespace) -> int:
    records, _, phase_path, manifest_path = load_phase_records("val")
    resolver = RawResolver("val")
    agent, load_report = load_frozen_agent(args.checkpoint.resolve())
    seq_length = tracking_seq_length(agent)
    cases, _ = make_cases("video", records, resolver, args, seq_length)
    with tempfile.TemporaryDirectory(prefix="skate_bfm_video_") as temp:
        temp_path, _, bridge = build_temp_motionlib(
            cases, records, resolver, agent, Path(temp), seq_length
        )
        tracking = AlignedSkateTrackingContext.load(agent, temp_path)
        env = HuskyBfmOnlineEnv()
        before = checkpoint_mutation(agent)
        root = output_root(args)
        root.mkdir(parents=True, exist_ok=True)
        videos = []
        try:
            for case in cases:
                source = records[str(case.get("transition_motion_key", case["motion_key"]))]
                raw, metadata, _ = resolver.load(source)
                video_path = root / f"{case['behavior']}.mp4"
                run = run_policy(
                    agent,
                    tracking,
                    case["case_id"],
                    raw,
                    metadata,
                    case,
                    env,
                    case["steps"],
                    video_path,
                )
                if args.with_expert:
                    write_expert_video(root / f"{case['behavior']}_expert.mp4", run, case["steps"])
                videos.append(
                    {
                        "behavior": case["behavior"],
                        "case": case,
                        "T_exec": run["result"]["evaluation"]["T_exec"],
                        "terminated": run["result"]["evaluation"]["terminated"],
                        "fall_reason": run["result"]["evaluation"]["fall_reason"],
                    }
                )
        finally:
            env.close()
        if before != checkpoint_mutation(agent):
            raise RuntimeError("Frozen model mutated during video generation.")
        write_json(
            root / "videos.json",
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hash_file(checkpoint_model_path(args.checkpoint.resolve())),
                "phase_motionlib": str(phase_path),
                "phase_manifest": str(manifest_path),
                "videos": videos,
                "bridge": bridge,
                "training": False,
                "test": False,
            },
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_root(args)),
                "behaviors": [item["behavior"] for item in videos],
            },
            indent=2,
        )
    )
    return 0


def run_viewer(args: argparse.Namespace) -> int:
    records, _, phase_path, manifest_path = load_phase_records("val")
    resolver = RawResolver("val")
    agent, _ = load_frozen_agent(args.checkpoint.resolve())
    seq_length = tracking_seq_length(agent)
    cases, _ = make_cases("video", records, resolver, args, seq_length)
    with tempfile.TemporaryDirectory(prefix="skate_bfm_viewer_") as temp:
        temp_path, _, bridge = build_temp_motionlib(
            cases, records, resolver, agent, Path(temp), seq_length
        )
        tracking = AlignedSkateTrackingContext.load(agent, temp_path)
        env = HuskyBfmOnlineEnv(viewer=True, realtime=True)
        before = checkpoint_mutation(agent)
        sequence = []
        interrupted = False

        def stop(_signum: int, _frame: Any) -> None:
            nonlocal interrupted
            interrupted = True

        old_int, old_term = signal.signal(signal.SIGINT, stop), signal.signal(signal.SIGTERM, stop)
        index = 0
        try:
            while env.env.is_running and not interrupted:
                case = cases[index % len(cases)]
                source = records[str(case.get("transition_motion_key", case["motion_key"]))]
                raw, metadata, _ = resolver.load(source)
                run = run_policy(
                    agent, tracking, case["case_id"], raw, metadata, case, env, case["steps"]
                )
                sequence.append(
                    {
                        "behavior": case["behavior"],
                        "case_id": case["case_id"],
                        "motion_key": case["motion_key"],
                        "reset_raw_frame": case["reset_raw_frame"],
                        "T_exec": run["result"]["evaluation"]["T_exec"],
                        "terminated": run["result"]["evaluation"]["terminated"],
                    }
                )
                index += 1
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            env.close()
        if before != checkpoint_mutation(agent):
            raise RuntimeError("Frozen model mutated during viewer.")
        root = output_root(args)
        root.mkdir(parents=True, exist_ok=True)
        write_json(
            root / "viewer_session.json",
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hash_file(checkpoint_model_path(args.checkpoint.resolve())),
                "phase_motionlib": str(phase_path),
                "phase_manifest": str(manifest_path),
                "segments_completed": len(sequence),
                "segments": sequence,
                "closed_by_user": not interrupted,
                "bridge": bridge,
                "mutation": {
                    "parameters_changed": False,
                    "buffers_changed": False,
                    "normalizer_changed": False,
                    "components_changed": False,
                },
            },
        )
    return 0


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if (
        args.checkpoint_sha256
        and hash_file(checkpoint_model_path(checkpoint)) != args.checkpoint_sha256
    ):
        raise RuntimeError("Checkpoint SHA256 mismatch.")
    if args.command == "eval":
        return run_eval(args)
    if args.command == "video":
        return run_video(args)
    return run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
