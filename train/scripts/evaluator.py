#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Evaluate frozen Skate-BFM checkpoints without training."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
import torch

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("rollout", "fixed-target"),
        default="rollout",
        help="Evaluate one formal checkpoint or run the historical fixed-target protocol.",
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
        help="M2.5b checkpoint saved after 10,000 transitions.",
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


def physical_metrics(
    records: list[dict[str, Any]],
    initial_raw: dict[str, Any],
    control_dt: float,
) -> dict[str, float]:
    initial_board_position = np.asarray(initial_raw["board_position"], dtype=float)
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
    try:
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
                records.append(dict(transition.raw_metadata))
                observation = transition.next_observation
                terminated = transition.terminated
                truncated = transition.truncated
                if terminated or truncated:
                    break

            if not records:
                if viewer_closed:
                    break
                raise RuntimeError("Frozen rollout produced no transitions.")
            metrics = physical_metrics(records, initial_raw, env.env.control_dt)
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
            }
            rollouts.append(row)
            print(
                "[Frozen rollout] "
                f"episode={episode_index + 1}/{args.episodes}, "
                f"steps={len(records)}, duration={metrics['episode_duration_s']:.2f}s, "
                f"terminated={terminated}, truncated={truncated}, "
                f"fall_reason={row['episode']['fall_reason'] or 'none'}"
            )
            if viewer_closed:
                break
    finally:
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
    return run_fixed_target_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
