#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Run the fixed, read-only Skate-BFM evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import platform
from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
import torch
from torch.utils._pytree import tree_map

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "husky_sim" / "src"))
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from data_collection.rollout_split import randomize_husky_play_physics
from skate_bfm.integration import HuskyBfmOnlineEnv
from train_skate_bfm import (
    _checkpoint_model_path,
    _state_fingerprint,
    build_train_config,
    load_pretrained_bfm0_agent,
)
from humanoidverse.agents.buffers.transition import DictBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen BFM0/Skate-BFM checkpoint with the fixed protocol."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "train" / "evaluation_protocol.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "model" / "bfm-zero-official",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "fixed-evaluation",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def data_fingerprint(payload: Any) -> str:
    digest = hashlib.sha256()

    def update(value: Any, path: str) -> None:
        digest.update(path.encode("utf-8"))
        if isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value):
                update(value[key], f"{path}/{key}")
            return
        if isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode("ascii"))
            for index, item in enumerate(value):
                update(item, f"{path}/{index}")
            return
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
        else:
            digest.update(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            return
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())

    update(payload, "root")
    return digest.hexdigest()


def git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_provenance(protocol_path: Path) -> dict[str, Any]:
    sources = {
        "evaluation_protocol": protocol_path,
        "evaluator": Path(__file__).resolve(),
        "training_entry": REPOSITORY_ROOT / "train/scripts/train_skate_bfm.py",
        "husky_lite_env": REPOSITORY_ROOT / "husky_sim/src/skate_husky/lite_env.py",
        "husky_online": REPOSITORY_ROOT / "src/skate_bfm/integration/online.py",
        "husky_scene": (
            REPOSITORY_ROOT
            / "husky_sim/upstream/test_scene/mjlab_scene.xml"
        ),
        "randomization": (
            REPOSITORY_ROOT
            / "train/scripts/data_collection/rollout_split.py"
        ),
    }
    missing = [
        str(path)
        for path in sources.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Evaluation source files are missing: {missing}")
    return {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in sources.items()
    }


def runtime_provenance(agent) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    known_environment = (
        "SKATE_ONLINE_ENV",
        "SKATE_COLLECT_ONLY",
        "SKATE_UPDATE_MODE",
        "SKATE_ADAPTATION_UPDATES",
        "SKATE_MAX_STEPS",
        "SKATE_ADAPTATION_PROTOCOL",
        "BFM0_PRETRAINED_CHECKPOINT",
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if cuda_available
            else None
        ),
        "torch_device": str(agent.device),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {
            name: os.environ[name]
            for name in known_environment
            if name in os.environ
        },
    }


def checkpoint_provenance(
    checkpoint: Path,
    agent,
) -> dict[str, Any]:
    model_path = _checkpoint_model_path(checkpoint)
    config_path = model_path.parent / "config.json"
    init_kwargs_path = model_path.parent / "init_kwargs.json"
    return {
        "resolved_path": str(checkpoint),
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "init_kwargs": {
            "path": str(init_kwargs_path),
            "sha256": sha256_file(init_kwargs_path),
        },
        "loaded_parameter_fingerprint": _state_fingerprint(
            agent._model,
            parameters=True,
        ),
        "loaded_buffer_fingerprint": _state_fingerprint(
            agent._model,
            parameters=False,
        ),
    }


def resolved_agent_provenance(agent) -> dict[str, Any]:
    config = {
        "agent": agent.cfg.model_dump(),
        "action_dim": agent.action_dim,
        "observation_space": {
            key: {
                "shape": list(space.shape),
                "dtype": str(space.dtype),
            }
            for key, space in sorted(agent.obs_space.spaces.items())
        },
        "parameter_dtype": str(next(agent._model.parameters()).dtype),
    }
    return {
        "config": config,
        "sha256": canonical_json_sha256(config),
    }


def quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def observation_space(observation: dict[str, torch.Tensor]) -> gymnasium.spaces.Dict:
    return gymnasium.spaces.Dict(
        {
            key: gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(value.shape),
                dtype=np.float32,
            )
            for key, value in observation.items()
        }
    )


def build_frozen_agent(checkpoint: Path):
    os.environ["SKATE_ONLINE_ENV"] = "skate"
    os.environ["SKATE_COLLECT_ONLY"] = "1"
    cfg = build_train_config()
    shape_env = HuskyBfmOnlineEnv()
    try:
        obs_space = observation_space(shape_env.reset())
    finally:
        shape_env.close()
    agent = cfg.agent.build(obs_space=obs_space, action_dim=29)
    load_report = load_pretrained_bfm0_agent(agent, checkpoint)
    agent._model.eval()
    agent._model.requires_grad_(False)
    return agent, load_report


def physical_metrics(
    records: list[dict[str, Any]],
    *,
    horizon: int,
    control_dt: float,
) -> dict[str, Any]:
    root_height = np.asarray([item["root_height"] for item in records])
    gravity = np.asarray([item["projected_gravity"] for item in records])
    root_linear = np.asarray([item["root_linear_velocity"] for item in records])
    root_angular = np.asarray([item["root_angular_velocity"] for item in records])
    board_position = np.asarray([item["board_position"] for item in records])
    board_quaternion = np.asarray([item["board_quaternion"] for item in records])
    board_linear = np.asarray([item["board_linear_velocity"] for item in records])
    board_angular = np.asarray([item["board_angular_velocity"] for item in records])
    tilt = np.degrees(
        np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
    )
    board_heading = np.unwrap(
        np.asarray([quaternion_yaw(value) for value in board_quaternion])
    )
    return {
        "steps": len(records),
        "bounded_rollout_steps": len(records),
        "completed_horizon_fraction": len(records) / horizon,
        "rollout_duration_s": len(records) * control_dt,
        "survival_steps": "unavailable_without_native_termination",
        "survival_fraction": "unavailable_without_native_termination",
        "native_termination": "unavailable",
        "fall": "unavailable",
        "root_height_mean_m": float(root_height.mean()),
        "root_height_min_m": float(root_height.min()),
        "root_height_std_m": float(root_height.std()),
        "root_tilt_mean_deg": float(tilt.mean()),
        "root_tilt_max_deg": float(tilt.max()),
        "root_linear_speed_mean_mps": float(
            np.linalg.norm(root_linear, axis=1).mean()
        ),
        "root_angular_speed_mean_radps": float(
            np.linalg.norm(root_angular, axis=1).mean()
        ),
        "board_linear_speed_mean_mps": float(
            np.linalg.norm(board_linear, axis=1).mean()
        ),
        "board_linear_speed_max_mps": float(
            np.linalg.norm(board_linear, axis=1).max()
        ),
        "board_angular_speed_mean_radps": float(
            np.linalg.norm(board_angular, axis=1).mean()
        ),
        "board_displacement_m": float(
            np.linalg.norm(board_position[-1] - board_position[0])
        ),
        "board_heading_change_rad": float(board_heading[-1] - board_heading[0]),
        "command_speed_error": "unavailable",
        "command_heading_error": "unavailable",
        "joint_pose_error": "unavailable_without_aligned_reference",
        "joint_velocity_error": "unavailable_without_aligned_reference",
        "foot_contact": "unavailable",
        "slippage": "unavailable",
        "contact_force": "unavailable",
    }


def collect_fixed_rollouts(
    agent,
    protocol: dict[str, Any],
    eval_buffer: DictBuffer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    horizon = int(protocol["rollout_horizon"])
    control_dt = float(protocol["control_dt_s"])
    refresh_steps = int(protocol["latent_refresh_steps"])
    all_records: list[dict[str, Any]] = []
    rollout_results: list[dict[str, Any]] = []
    transition_index = 0

    for rollout_index, condition in enumerate(protocol["rollouts"]):
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
        initial_observation_fingerprint = data_fingerprint(observation)
        z = None
        first_z_fingerprint = None
        first_action_fingerprint = None
        records: list[dict[str, Any]] = []
        transition_ids: list[str] = []
        try:
            for step in range(horizon):
                if z is None or step % refresh_steps == 0:
                    torch.manual_seed(condition["latent_seed"] + step)
                    z = agent._model.sample_z(1, device=agent.device)[0]
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
                if first_z_fingerprint is None:
                    first_z_fingerprint = data_fingerprint(z)
                    first_action_fingerprint = data_fingerprint(action)
                transition = env.step(
                    action,
                    z,
                    truncated=step == horizon - 1,
                )
                transition_id = f"{condition['rollout_id']}:{step:04d}"
                data = transition.as_buffer_data()
                data["eval_transition_index"] = torch.tensor(
                    [[transition_index]],
                    dtype=torch.int64,
                )
                data["eval_rollout_index"] = torch.tensor(
                    [[rollout_index]],
                    dtype=torch.int64,
                )
                data["eval_step_index"] = torch.tensor(
                    [[step]],
                    dtype=torch.int64,
                )
                eval_buffer.extend(data)
                record = dict(transition.raw_metadata)
                record.update(
                    {
                        "transition_id": transition_id,
                        "rollout_id": condition["rollout_id"],
                        "dynamics_split": condition["dynamics_split"],
                        "command_v": condition["command_v"],
                        "command_h": condition["command_h"],
                    }
                )
                records.append(record)
                all_records.append(record)
                transition_ids.append(transition_id)
                transition_index += 1
                observation = transition.next_observation
        finally:
            env.close()

        rollout_results.append(
            {
                **condition,
                "transition_ids": transition_ids,
                "dynamics_realization": dynamics_report,
                "input_fingerprints": {
                    "initial_observation": initial_observation_fingerprint,
                    "first_z": first_z_fingerprint,
                    "first_action": first_action_fingerprint,
                    "dynamics_realization": canonical_json_sha256(
                        dynamics_report
                    ),
                },
                "metrics": physical_metrics(
                    records,
                    horizon=horizon,
                    control_dt=control_dt,
                ),
            }
        )
    return all_records, rollout_results


def fixed_batch(
    buffer: DictBuffer,
    batch_size: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    full = buffer.get_full_buffer()
    size = len(buffer)
    if size < batch_size:
        raise ValueError(
            f"Evaluation buffer has {size} transitions; need {batch_size}."
        )
    indices = torch.linspace(0, size - 1, batch_size).round().long()
    return tree_map(lambda value: value[indices], full), indices


def fb_diagnostics(
    agent,
    batch: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    device = agent.device
    obs = tree_map(lambda value: value.to(device), batch["observation"])
    next_obs = tree_map(
        lambda value: value.to(device),
        batch["next"]["observation"],
    )
    action = batch["action"].to(device)
    z = batch["z"].to(device)
    terminated = batch["next"]["terminated"].to(device)
    batch_size = action.shape[0]
    off_diag = 1.0 - torch.eye(batch_size, device=device)
    off_diag_sum = off_diag.sum()

    torch.manual_seed(seed)
    with torch.no_grad():
        normalized_obs = agent._model._normalize(obs)
        normalized_next_obs = agent._model._normalize(next_obs)
        next_action = agent.sample_action_from_norm_obs(normalized_next_obs, z)
        target_fs = agent._model._target_forward_map(
            normalized_next_obs,
            z,
            next_action,
        )
        target_b = agent._model._target_backward_map(normalized_next_obs)
        target_ms = torch.matmul(target_fs, target_b.T)
        target_m = agent.get_targets_uncertainty(
            target_ms,
            agent.cfg.train.fb_pessimism_penalty,
        )[2]
        fs = agent._model._forward_map(normalized_obs, z, action)
        b = agent._model._backward_map(normalized_next_obs)
        ms = torch.matmul(fs, b.T)
        discount = agent.cfg.train.discount * ~terminated
        diff = ms - discount * target_m
        fb_offdiag = 0.5 * (diff * off_diag).pow(2).sum() / off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * ms.shape[0]
        cov = torch.matmul(b, b.T)
        orth_loss_diag = -cov.diag().mean()
        orth_loss_offdiag = (
            0.5 * (cov * off_diag).pow(2).sum() / off_diag_sum
        )
        orth_loss = orth_loss_diag + orth_loss_offdiag
        fb_loss = fb_offdiag + fb_diag + agent.cfg.train.ortho_coef * orth_loss

        matching = torch.matmul(fs.mean(dim=0), b.T)
        diagonal = matching.diag()
        off_values = matching[off_diag.bool()]
        order = torch.argsort(matching, dim=1, descending=True)
        targets = torch.arange(batch_size, device=device).unsqueeze(1)
        ranks = torch.argmax((order == targets).to(torch.int64), dim=1) + 1

    fb_metrics = {
        "fb_loss": float(fb_loss),
        "fb_diag": float(fb_diag),
        "fb_offdiag": float(fb_offdiag),
        "orth_loss": float(orth_loss),
        "orth_loss_diag": float(orth_loss_diag),
        "orth_loss_offdiag": float(orth_loss_offdiag),
        "q_loss": 0.0,
        "B_norm": float(torch.linalg.vector_norm(b, dim=-1).mean()),
        "z_norm": float(torch.linalg.vector_norm(z, dim=-1).mean()),
    }
    retrieval_metrics = {
        "diagnostic": "Skate-BFM representation diagnostic",
        "diagonal_mean": float(diagonal.mean()),
        "off_diagonal_mean": float(off_values.mean()),
        "diagonal_offdiagonal_margin": float(
            diagonal.mean() - off_values.mean()
        ),
        "top_1": float((ranks <= 1).float().mean()),
        "top_5": float((ranks <= min(5, batch_size)).float().mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }
    return fb_metrics, retrieval_metrics


def behavior_coverage(
    records: list[dict[str, Any]],
    projection: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    phi = np.asarray(
        [
            [
                record["root_linear_velocity"][0],
                record["root_linear_velocity"][1],
                record["root_angular_velocity"][2],
                record["board_linear_velocity"][0],
                record["board_linear_velocity"][1],
                record["board_angular_velocity"][2],
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    ranges = np.asarray(
        [item["range"] for item in projection["dimensions"]],
        dtype=np.float64,
    )
    bins = int(projection["entropy"]["bins_per_dimension"])
    out_of_range = np.logical_or(phi < ranges[:, 0], phi > ranges[:, 1])
    clipped = np.clip(phi, ranges[:, 0], ranges[:, 1])
    scaled = (clipped - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
    bin_ids = np.floor(scaled * bins).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, bins - 1)
    _, counts = np.unique(bin_ids, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    max_entropy = math.log(bins ** phi.shape[1])

    occupancy = projection["occupancy_2d"]
    occupancy_bins = int(occupancy["bins_per_dimension"])
    occupancy_hist, x_edges, y_edges = np.histogram2d(
        phi[:, 3],
        phi[:, 4],
        bins=occupancy_bins,
        range=[ranges[3].tolist(), ranges[4].tolist()],
    )
    metrics = {
        "projection": projection["name"],
        "sample_count": len(phi),
        "occupied_bins": int(len(counts)),
        "entropy_nats": entropy,
        "normalized_entropy": entropy / max_entropy,
        "out_of_range_fraction": float(out_of_range.mean()),
        "entropy_bins_per_dimension": bins,
        "occupancy_2d_nonzero_bins": int(np.count_nonzero(occupancy_hist)),
        "density_estimator": "not_applicable",
        "inverse_density_sampling": "not_applicable",
    }
    return phi, metrics, occupancy_hist, x_edges, y_edges


def split_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in ("seen", "unseen"):
        selected = [
            item["metrics"]
            for item in rollouts
            if item["dynamics_split"] == split
        ]
        result[split] = {
            "rollout_count": len(selected),
            "completed_horizon_fraction_mean": float(
                np.mean(
                    [item["completed_horizon_fraction"] for item in selected]
                )
            ),
            "survival_fraction": "unavailable_without_native_termination",
            "board_linear_speed_mean_mps": float(
                np.mean(
                    [item["board_linear_speed_mean_mps"] for item in selected]
                )
            ),
            "root_tilt_mean_deg": float(
                np.mean([item["root_tilt_mean_deg"] for item in selected])
            ),
            "downstream_score": "unavailable_without_command_aligned_latent",
        }
    result["seen_unseen_score_gap"] = "not_applicable"
    return result


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol["evaluator_version"] != "skate-bfm-fixed-eval-v1":
        raise ValueError("Unsupported fixed evaluator version.")
    rollout_ids = [item["rollout_id"] for item in protocol["rollouts"]]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("Evaluation rollout IDs must be unique.")
    if {item["dynamics_split"] for item in protocol["rollouts"]} != {
        "seen",
        "unseen",
    }:
        raise ValueError("Protocol must define both seen and unseen dynamics.")
    if protocol["isolation"]["eval_transitions_enter_training"]:
        raise ValueError("Evaluation transitions cannot enter training.")


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_protocol(protocol)

    random.seed(protocol["protocol_seed"])
    np.random.seed(protocol["protocol_seed"])
    torch.manual_seed(protocol["protocol_seed"])
    agent, checkpoint_report = build_frozen_agent(checkpoint)
    parameter_before = _state_fingerprint(agent._model, parameters=True)
    buffer_before = _state_fingerprint(agent._model, parameters=False)
    provenance = {
        "repository": {
            "git_commit": git_commit_sha(),
        },
        "evaluation": {
            "evaluator_version": protocol["evaluator_version"],
            "formula_fidelity": {
                "source_of_truth": (
                    "train/scripts/isaac_env/humanoidverse/agents/fb/"
                    "agent.py:FBAgent.update_fb"
                ),
                "fb_diag": (
                    "-diagonal(Ms - discount * target_M).mean() "
                    "* num_parallel"
                ),
                "matches_vendored_update_fb": True,
                "q_loss_active": agent.cfg.train.q_loss_coef > 0,
            },
        },
        "sources": source_provenance(protocol_path),
        "checkpoint": checkpoint_provenance(checkpoint, agent),
        "resolved_agent": resolved_agent_provenance(agent),
        "runtime": runtime_provenance(agent),
    }

    total_transitions = (
        int(protocol["rollout_horizon"]) * len(protocol["rollouts"])
    )
    eval_buffer = DictBuffer(capacity=total_transitions, device="cpu")
    train_guard = DictBuffer(capacity=1, device="cpu")
    replay_buffers = {
        "train": train_guard,
        "train_skate": train_guard,
        "eval_skate_transition": eval_buffer,
    }
    records, rollouts = collect_fixed_rollouts(
        agent,
        protocol,
        replay_buffers["eval_skate_transition"],
    )
    if replay_buffers["train"] is replay_buffers["eval_skate_transition"]:
        raise RuntimeError("Training and evaluation buffers must be distinct.")
    if len(replay_buffers["train"]) != 0:
        raise RuntimeError("Fixed evaluation wrote into training replay.")

    batch, diagnostic_indices = fixed_batch(
        eval_buffer,
        int(protocol["fb_diagnostic_batch_size"]),
    )
    fb_metrics, retrieval_metrics = fb_diagnostics(
        agent,
        batch,
        seed=int(protocol["fb_diagnostic_seed"]),
    )
    phi, entropy_metrics, occupancy_hist, x_edges, y_edges = behavior_coverage(
        records,
        protocol["physical_behavior_projection"],
    )
    transition_ids = [
        record["transition_id"]
        for record in records
    ]
    evaluation_inputs = {
        "eval_transition_buffer": data_fingerprint(
            eval_buffer.get_full_buffer()
        ),
        "transition_ids": canonical_json_sha256(transition_ids),
        "fixed_diagnostic_batch_indices": diagnostic_indices.tolist(),
        "fixed_diagnostic_batch_indices_fingerprint": data_fingerprint(
            diagnostic_indices
        ),
        "diagnostic_batch": data_fingerprint(batch),
        "rollouts": {
            rollout["rollout_id"]: rollout["input_fingerprints"]
            for rollout in rollouts
        },
    }
    provenance["evaluation_inputs"] = evaluation_inputs

    parameter_after = _state_fingerprint(agent._model, parameters=True)
    buffer_after = _state_fingerprint(agent._model, parameters=False)
    if parameter_before != parameter_after:
        raise RuntimeError("Evaluation mutated model parameters.")
    if buffer_before != buffer_after:
        raise RuntimeError("Evaluation mutated model buffers.")

    base_impl = REPOSITORY_ROOT / protocol["base_retention"]["implementation"]
    base_motion = REPOSITORY_ROOT / protocol["base_retention"]["motion_file"]
    if not base_impl.is_file() or not base_motion.is_file():
        raise FileNotFoundError("Configured Base retention evaluator is unavailable.")

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_buffer.save(output_dir / "eval_skate_transition")
    np.savez_compressed(
        output_dir / "behavior_coverage.npz",
        phi=phi,
        occupancy=occupancy_hist,
        occupancy_x_edges=x_edges,
        occupancy_y_edges=y_edges,
    )
    resolved_manifest = {
        **protocol,
        "protocol_path": str(protocol_path),
        "checkpoint": checkpoint_report,
        "rollouts": rollouts,
        "transition_count": len(eval_buffer),
        "transition_ids": transition_ids,
        "provenance": provenance,
    }
    write_json(output_dir / "evaluation_manifest.json", resolved_manifest)

    report = {
        "evaluator_version": protocol["evaluator_version"],
        "provenance": provenance,
        "checkpoint": checkpoint_report["source"],
        "held_out_transition_count": len(eval_buffer),
        "fb_diagnostics": fb_metrics,
        "matching_retrieval": retrieval_metrics,
        "skate_rollouts": rollouts,
        "dynamics_generalization": split_summary(rollouts),
        "behavior_coverage": entropy_metrics,
        "context_protocol": protocol["context_protocol"],
        "base_retention": {
            **protocol["base_retention"],
            "entry_identified": True,
            "run_in_this_preflight": False,
        },
        "isolation": {
            "train_is_eval": False,
            "train_transition_count": len(train_guard),
            "eval_transition_count": len(eval_buffer),
            "unseen_dynamics_evaluation_only": True,
        },
        "mutation": {
            "parameters_changed": False,
            "buffers_changed": False,
            "agent_update_calls": 0,
            "optimizer_steps": 0,
        },
        "future_fields": {
            "dynamics_context_h": "unavailable",
            "rfb_kappa": "not_applicable",
            "mebe_density": "not_applicable",
            "mebe_beta": "not_applicable",
            "qaux_to_mebe_regularization": "unresolved",
        },
    }
    write_json(output_dir / "evaluation_metrics.json", report)
    print(
        "Fixed evaluation complete: "
        f"{len(eval_buffer)} held-out transitions, "
        f"FB loss {fb_metrics['fb_loss']:.6f}, "
        f"retrieval top-1 {retrieval_metrics['top_1']:.4f}, "
        f"behavior entropy {entropy_metrics['entropy_nats']:.6f}, "
        "optimizer steps 0"
    )
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
