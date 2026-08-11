#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Evaluate frozen target-conditioned Skate responses without training."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "husky_sim" / "src"))
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from data_collection.rollout_split import randomize_husky_play_physics
from skate_bfm.integration import HuskyBfmOnlineEnv
from train_skate_bfm import (
    checkpoint_model_path,
    encode_target,
    hash_buffers,
    hash_components,
    hash_data,
    hash_file,
    hash_params,
    load_expert,
    load_frozen_agent,
)


DEFAULT_TARGET_BANK = (
    REPOSITORY_ROOT
    / "train/dataset/skate-expert-pose/target_bank/target_bank.json"
)
DEFAULT_PROTOCOL = REPOSITORY_ROOT / "train/evaluation_protocol.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/m2.3b-0-target-conditioned"
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-bank", type=Path, default=DEFAULT_TARGET_BANK)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expert-motion",
        type=Path,
        default=REPOSITORY_ROOT
        / "train/dataset/skate-expert-pose/motion_library/skate_expert.pkl",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def checkpoint_paths() -> dict[str, Path]:
    return {
        "official_bfm0": REPOSITORY_ROOT / "model/bfm-zero-official",
        "base_only_update100": (
            REPOSITORY_ROOT / "results/m2.2b-1/update_100/checkpoint"
        ),
        "base_skate_update100": (
            REPOSITORY_ROOT
            / "results/m2.2b-3/base_skate_50_50/update_100/checkpoint"
        ),
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
        "command_injected_into_actor": False,
        "metrics": physical_metrics(records, initial_raw),
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
    expected_fingerprint = target["latents"][checkpoint_name]["latent"][
        "fingerprint"
    ]
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


def main() -> int:
    args = parse_args()
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

    paths = checkpoint_paths()
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Required frozen checkpoints missing: {missing}")

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
            "date": "2026-08-11",
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
    write_json(output_dir / "target_conditioned_metrics.json", output)
    print(
        f"Target-conditioned evaluation complete: {len(all_rollouts)} rollouts, "
        "training=0, optimizer_steps=0, normalizer_mutation=False"
    )
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
