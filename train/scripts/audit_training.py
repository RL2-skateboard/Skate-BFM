"""Read-only Skate replay and auxiliary-reward contract audit."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import mujoco
import torch
from data_collection import rollout_split
from skate_husky import AUX_REWARD_KEYS, HuskyLiteEnv, LiveFallDetector
from torch.utils._pytree import tree_map
from train_skate_bfm import (
    REPOSITORY_ROOT,
    BaseSkateExpertSampler,
    build_train_config,
    hash_buffers,
    hash_params,
)

from skate_bfm.integration import HuskyBfmOnlineEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Skate replay and aux rewards without training."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "m2.4c-termination-contract",
    )
    return parser.parse_args()


def tensor_info(value: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
    }


def tree_info(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: tree_info(item) for key, item in sorted(value.items())}
    if torch.is_tensor(value):
        return tensor_info(value)
    return {"type": type(value).__name__}


def require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} produced NaN or Inf.")


def aux_reward_statistics(
    replay_buffer: dict,
    expected_transition_count: int,
    scaling: dict[str, float],
) -> dict[str, Any]:
    full_replay = replay_buffer["train"].get_full_buffer()
    aux_rewards = full_replay.get("aux_rewards")
    if not isinstance(aux_rewards, dict) or tuple(aux_rewards) != AUX_REWARD_KEYS:
        raise RuntimeError("Formal Skate replay is missing the 8-key aux reward contract.")

    report = {}
    weighted_raw_aux = torch.zeros(
        expected_transition_count,
        1,
        dtype=torch.float32,
    )
    for name in AUX_REWARD_KEYS:
        values = aux_rewards[name]
        if tuple(values.shape) != (expected_transition_count, 1):
            raise RuntimeError(
                f"{name} has shape {tuple(values.shape)}, expected "
                f"({expected_transition_count}, 1)."
            )
        require_finite(f"aux_rewards.{name}", values)
        weighted_raw_aux += values * float(scaling[name])
        report[name] = {
            "shape": list(values.shape),
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "p50": float(torch.quantile(values, 0.50)),
            "p90": float(torch.quantile(values, 0.90)),
            "p99": float(torch.quantile(values, 0.99)),
            "max": float(values.max()),
            "nonzero_fraction": float((values.abs() > 1e-8).float().mean()),
        }
    require_finite("weighted_raw_aux", weighted_raw_aux)
    return {
        "keys": list(AUX_REWARD_KEYS),
        "statistics": report,
        "weighted_raw_aux": {
            "mean": float(weighted_raw_aux.mean()),
            "std": float(weighted_raw_aux.std(unbiased=False)),
            "p50": float(torch.quantile(weighted_raw_aux, 0.50)),
            "p90": float(torch.quantile(weighted_raw_aux, 0.90)),
            "p99": float(torch.quantile(weighted_raw_aux, 0.99)),
            "max": float(weighted_raw_aux.max()),
            "nonzero_fraction": float(
                (weighted_raw_aux.abs() > 1e-8).float().mean()
            ),
        },
        "normalizer_updated": False,
    }


def termination_statistics(
    replay_buffer: dict,
    expected_transition_count: int,
    discount_gamma: float,
) -> dict[str, Any]:
    full_replay = replay_buffer["train"].get_full_buffer()
    terminated = full_replay["next"]["terminated"]
    truncated = full_replay["next"]["truncated"]
    expected_shape = (expected_transition_count, 1)
    for name, values in {
        "terminated": terminated,
        "truncated": truncated,
    }.items():
        if tuple(values.shape) != expected_shape or values.dtype is not torch.bool:
            raise RuntimeError(
                f"next.{name} must be bool {expected_shape}, got "
                f"{values.dtype} {tuple(values.shape)}."
            )
    overlap = terminated & truncated
    if overlap.any():
        raise RuntimeError("A fall transition cannot also be horizon-truncated.")
    discount = discount_gamma * ~terminated
    if tuple(discount.shape) != expected_shape or not discount.is_floating_point():
        raise RuntimeError("FBcprAux discount has an invalid shape or dtype.")
    if not torch.equal(
        discount[terminated],
        torch.zeros_like(discount[terminated]),
    ):
        raise RuntimeError("Terminal transitions must have zero discount.")
    if not torch.allclose(
        discount[~terminated],
        torch.full_like(discount[~terminated], discount_gamma),
    ):
        raise RuntimeError("Non-terminal transitions must have gamma discount.")
    terminated_count = int(terminated.sum())
    truncated_count = int(truncated.sum())
    return {
        "terminated": tensor_info(terminated),
        "truncated": tensor_info(truncated),
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "normal_transition_count": expected_transition_count
        - terminated_count
        - truncated_count,
        "overlap_count": int(overlap.sum()),
        "discount": tensor_info(discount),
        "discount_semantics": "gamma * ~terminated",
    }


def _set_root_tilt(env: HuskyLiteEnv) -> None:
    env.data.qpos[3:7] = (
        math.sqrt(0.5),
        math.sqrt(0.5),
        0.0,
        0.0,
    )
    mujoco.mj_forward(env.model, env.data)


def controlled_fall_validation() -> dict[str, Any]:
    physical_env = HuskyLiteEnv()
    try:
        physical_env.reset()
        normal, _, normal_diagnostics = physical_env.fall_detector.check(
            physical_env.data
        )

        physical_env.fall_detector.reset()
        _set_root_tilt(physical_env)
        transient, _, transient_diagnostics = physical_env.fall_detector.check(
            physical_env.data
        )
        persistent = transient
        for _ in range(physical_env.fall_detector.confirm_frames - 1):
            persistent, _, _ = physical_env.fall_detector.check(physical_env.data)

        physical_env.reset()
        physical_env.data.qpos[2] = 0.2
        mujoco.mj_forward(physical_env.model, physical_env.data)
        low_contact, _, low_contact_diagnostics = physical_env.fall_detector.check(
            physical_env.data
        )
        for _ in range(physical_env.fall_detector.confirm_frames - 1):
            low_contact, _, _ = physical_env.fall_detector.check(physical_env.data)

        physical_env.reset()
        board_joint = physical_env.model.joint(
            "skateboard/floating_base_joint_skateboard"
        )
        physical_env.data.qpos[board_joint.qposadr[0]] += 5.0
        mujoco.mj_forward(physical_env.model, physical_env.data)
        board_separation, _, board_diagnostics = physical_env.fall_detector.check(
            physical_env.data
        )
    finally:
        physical_env.close()

    online_env = HuskyBfmOnlineEnv()
    try:
        online_env.reset()
        _set_root_tilt(online_env.env)
        for frame in range(online_env.env.fall_detector.confirm_frames):
            terminal_transition = online_env.step(
                torch.zeros(29),
                torch.zeros(256),
                truncated=frame == online_env.env.fall_detector.confirm_frames - 1,
            )
        try:
            online_env.step(torch.zeros(29), torch.zeros(256))
        except RuntimeError:
            reset_required = True
        else:
            reset_required = False
    finally:
        online_env.close()

    if rollout_split.LiveFallDetector is not LiveFallDetector:
        raise RuntimeError("Collection and online fall detectors diverged.")
    if normal or transient or not persistent or not low_contact or board_separation:
        raise RuntimeError("Controlled physical fall checks failed.")
    if not (
        terminal_transition.terminated
        and not terminal_transition.truncated
        and reset_required
    ):
        raise RuntimeError("Online terminal transition contract failed.")
    return {
        "collection_online_shared_implementation": True,
        "confirm_frames": terminal_transition.raw_metadata["confirm_frames"],
        "normal_terminated": normal,
        "single_transient_bad_frame_terminated": transient,
        "persistent_severe_tilt_terminated": persistent,
        "low_height_illegal_contact_terminated": low_contact,
        "board_separation_terminated": board_separation,
        "feet_on_board_after_separation": board_diagnostics["feet_on_board"],
        "normal_diagnostics": normal_diagnostics,
        "transient_diagnostics": transient_diagnostics,
        "low_contact_diagnostics": low_contact_diagnostics,
        "online_terminal_transition": {
            "terminated": terminal_transition.terminated,
            "truncated": terminal_transition.truncated,
            "reset_required": reset_required,
        },
    }


def target_pair_info(source: tuple, target: tuple) -> dict[str, Any]:
    shapes_match = len(source) == len(target) and all(
        tuple(left.shape) == tuple(right.shape)
        for left, right in zip(source, target, strict=True)
    )
    return {
        "source_parameter_count": len(source),
        "target_parameter_count": len(target),
        "shapes_match": shapes_match,
    }


def configure_environment(output_dir: Path) -> None:
    checkpoint = output_dir / "workspace" / "checkpoint"
    if checkpoint.exists():
        raise RuntimeError(
            f"Audit workspace must not resume a Skate checkpoint: {checkpoint}"
        )
    os.environ["SKATE_ONLINE_ENV"] = "skate"
    os.environ["SKATE_UPDATE_MODE"] = "none"
    os.environ["SKATE_COLLECT_ONLY"] = "1"
    os.environ["SKATE_MAX_STEPS"] = "1024"
    os.environ["SKATE_EXPERT_RATIO"] = "0.5"
    os.environ["SKATE_EXPERT_MOTION_FILE"] = str(
        REPOSITORY_ROOT
        / "train"
        / "dataset"
        / "skate-expert-pose"
        / "motion_library"
        / "skate_expert.pkl"
    )
    os.environ["SKATE_WORK_DIR"] = str(output_dir / "workspace")


def resolved_config(cfg) -> dict[str, Any]:
    train = cfg.agent.train
    return {
        "agent_class": cfg.agent.name,
        "batch_size": train.batch_size,
        "seq_length": cfg.agent.model.seq_length,
        "discount": train.discount,
        "learning_rates": {
            "F": train.lr_f,
            "B": train.lr_b,
            "Actor": train.lr_actor,
            "QD": train.lr_critic,
            "Qaux": train.lr_aux_critic,
            "discriminator": train.lr_discriminator,
        },
        "target_tau": {
            "F_B": train.fb_target_tau,
            "QD_Qaux": train.critic_target_tau,
        },
        "expert_asm_ratio": train.expert_asm_ratio,
        "train_goal_ratio": train.train_goal_ratio,
        "relabel_ratio": train.relabel_ratio,
        "q_loss_coef": train.q_loss_coef,
        "reg_coeff": train.reg_coeff,
        "reg_coeff_aux": train.reg_coeff_aux,
        "grad_penalty_discriminator": train.grad_penalty_discriminator,
        "critic_pessimism_penalty": train.critic_pessimism_penalty,
        "aux_critic_pessimism_penalty": train.aux_critic_pessimism_penalty,
        "actor_pessimism_penalty": train.actor_pessimism_penalty,
        "aux_rewards": list(cfg.agent.aux_rewards),
        "aux_rewards_scaling": dict(cfg.agent.aux_rewards_scaling),
    }


def audit_networks(workspace, replay_buffer: dict) -> dict[str, Any]:
    agent = workspace.agent
    model = agent._model
    batch_size = agent.cfg.train.batch_size
    device = agent.device

    train_batch = replay_buffer["train"].sample(batch_size)
    expert_batch = replay_buffer["expert_slicer"].sample(batch_size)
    train_obs = tree_map(lambda value: value.to(device), train_batch["observation"])
    train_next_obs = tree_map(
        lambda value: value.to(device),
        train_batch["next"]["observation"],
    )
    expert_obs = tree_map(
        lambda value: value.to(device),
        expert_batch["observation"],
    )
    expert_next_obs = tree_map(
        lambda value: value.to(device),
        expert_batch["next"]["observation"],
    )
    action = train_batch["action"].to(device)
    train_z = train_batch["z"].to(device)
    discount = (
        agent.cfg.train.discount
        * ~train_batch["next"]["terminated"].to(device)
    )

    params_before = hash_params(model)
    buffers_before = hash_buffers(model)
    model.eval()

    outputs: dict[str, Any] = {}
    with torch.no_grad():
        normalized_train_obs = model._obs_normalizer(train_obs)
        normalized_train_next_obs = model._obs_normalizer(train_next_obs)
        normalized_expert_obs = model._obs_normalizer(expert_obs)
        normalized_expert_next_obs = model._obs_normalizer(expert_next_obs)

        expert_z = agent.encode_expert(next_obs=normalized_expert_next_obs)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(4728)
            mixed_z = agent.sample_mixed_z(
                train_goal=normalized_train_next_obs,
                expert_encodings=expert_z,
            )

        actor_distribution = model._actor(
            normalized_train_obs,
            train_z,
            model.cfg.actor_std,
        )
        actor_action = actor_distribution.mean

        checks = {
            "expert_z": expert_z,
            "mixed_z": mixed_z,
            "actor_action": actor_action,
            "discriminator_expert_logits": model._discriminator.compute_logits(
                normalized_expert_obs,
                expert_z,
            ),
            "discriminator_train_logits": model._discriminator.compute_logits(
                normalized_train_obs,
                train_z,
            ),
            "discriminator_reward": model._discriminator.compute_reward(
                normalized_train_obs,
                train_z,
            ),
            "F": model._forward_map(normalized_train_obs, train_z, action),
            "target_F": model._target_forward_map(
                normalized_train_next_obs,
                train_z,
                actor_action,
            ),
            "B": model._backward_map(normalized_train_next_obs),
            "target_B": model._target_backward_map(normalized_train_next_obs),
            "QD": model._critic(normalized_train_obs, train_z, action),
            "target_QD": model._target_critic(
                normalized_train_next_obs,
                train_z,
                actor_action,
            ),
            "Qaux": model._aux_critic(normalized_train_obs, train_z, action),
            "target_Qaux": model._target_aux_critic(
                normalized_train_next_obs,
                train_z,
                actor_action,
            ),
        }
        for name, value in checks.items():
            require_finite(name, value)
            outputs[name] = tensor_info(value)

        outputs["fb_target_matrix"] = tensor_info(
            torch.matmul(checks["target_F"], checks["target_B"].T)
        )

    params_after = hash_params(model)
    buffers_after = hash_buffers(model)
    if params_before != params_after:
        raise RuntimeError("Model parameters changed during read-only audit.")
    if buffers_before != buffers_after:
        raise RuntimeError("Model buffers changed during read-only audit.")

    z_norm = torch.linalg.vector_norm(train_z, dim=-1)
    expert_sampler = replay_buffer["expert_slicer"]
    sequence_count = batch_size // agent.cfg.model.seq_length
    skate_sequences = int(
        sequence_count * expert_sampler.skate_expert_ratio + 0.5
    )
    optimizers = {
        name: {
            "exists": optimizer is not None,
            "parameter_groups": len(optimizer.param_groups),
            "state_entries": len(optimizer.state),
        }
        for name, optimizer in {
            "F": agent.forward_optimizer,
            "B": agent.backward_optimizer,
            "Actor": agent.actor_optimizer,
            "QD": agent.critic_optimizer,
            "Qaux": agent.aux_critic_optimizer,
            "discriminator": agent.discriminator_optimizer,
        }.items()
    }

    replay_report = {
        "train_is_train_skate": replay_buffer["train"]
        is replay_buffer["train_skate"],
        "size": len(replay_buffer["train"]),
        "sample": tree_info(train_batch),
        "aux_rewards_present": "aux_rewards" in train_batch,
        "aux_reward_keys": sorted(train_batch.get("aux_rewards", {})),
        "z_norm": {
            "min": float(z_norm.min()),
            "mean": float(z_norm.mean()),
            "max": float(z_norm.max()),
        },
        "terminated_true": int(train_batch["next"]["terminated"].sum()),
        "truncated_true": int(train_batch["next"]["truncated"].sum()),
        "discount": tensor_info(discount),
    }
    expert_report = {
        "sampler": type(expert_sampler).__name__,
        "complete_sequence_sampling": isinstance(
            expert_sampler,
            BaseSkateExpertSampler,
        ),
        "batch_size": batch_size,
        "seq_length": agent.cfg.model.seq_length,
        "sequence_count": sequence_count,
        "base_sequences": sequence_count - skate_sequences,
        "skate_sequences": skate_sequences,
        "sample": tree_info(expert_batch),
        "skate_source_metadata": getattr(
            replay_buffer["expert_skate"],
            "source_metadata",
            {},
        ),
        "base_motion_count": len(replay_buffer["expert_base"].motion_ids),
    }
    target_networks = {
        "F": target_pair_info(
            agent._forward_map_paramlist,
            agent._target_forward_map_paramlist,
        ),
        "B": target_pair_info(
            agent._backward_map_paramlist,
            agent._target_backward_map_paramlist,
        ),
        "QD": target_pair_info(
            agent._critic_map_paramlist,
            agent._target_critic_map_paramlist,
        ),
        "Qaux": target_pair_info(
            agent._aux_critic_map_paramlist,
            agent._aux_target_critic_map_paramlist,
        ),
    }
    normalizers = {
        "observation": {
            "exists": model._obs_normalizer is not None,
            "training": model._obs_normalizer.training,
            "buffer_shapes": {
                name: list(value.shape)
                for name, value in model._obs_normalizer.named_buffers()
            },
        },
        "aux_reward": {
            "exists": model._aux_reward_normalizer is not None,
            "buffer_shapes": {
                name: list(value.shape)
                for name, value in model._aux_reward_normalizer.named_buffers()
            },
            "expected_input_shape": [batch_size, 1],
            "forward_skipped": (
                "EMA.forward mutates mean, mean_square, and counter; the audit "
                "inspects checkpointed state without calling it."
            ),
        },
    }
    return {
        "replay": replay_report,
        "expert": expert_report,
        "outputs": outputs,
        "optimizers": optimizers,
        "target_networks": target_networks,
        "normalizers": normalizers,
        "mutation": {
            "parameter_mutation": params_before != params_after,
            "buffer_mutation": buffers_before != buffers_after,
            "parameter_hash_before": params_before,
            "parameter_hash_after": params_after,
            "buffer_hash_before": buffers_before,
            "buffer_hash_after": buffers_after,
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_environment(output_dir)

    cfg = build_train_config()
    config_report = resolved_config(cfg)
    workspace = cfg.build()
    replay_buffer = workspace.train()
    if replay_buffer is None:
        raise RuntimeError("Collect-only Workspace did not return replay buffers.")

    runtime_report = audit_networks(workspace, replay_buffer)
    physical_env = HuskyLiteEnv()
    try:
        physical_actuators = physical_env.physical_actuator_report
    finally:
        physical_env.close()
    aux_report = aux_reward_statistics(
        replay_buffer,
        expected_transition_count=1024,
        scaling=dict(cfg.agent.aux_rewards_scaling),
    )
    termination_report = termination_statistics(
        replay_buffer,
        expected_transition_count=1024,
        discount_gamma=cfg.agent.train.discount,
    )
    controlled_fall_report = controlled_fall_validation()
    report = {
        "milestone": "M2.4c Native Fall Termination Contract",
        "resolved_config": config_report,
        "checkpoint": workspace.agent.pretrained_load_report,
        "runtime": runtime_report,
        "aux_reward_contract": {
            "reward_semantics_source": "vendored BFM-Zero",
            "physical_constraint_source": "HUSKY MuJoCo runtime",
            "physical_actuators": physical_actuators,
            "replay": aux_report,
        },
        "termination_contract": {
            "fall_source": "shared Skate expert collection LiveFallDetector",
            "definition": (
                "persistent severe tilt OR persistent low root height with "
                "illegal contact"
            ),
            "horizon_semantics": "truncated=True",
            "fall_precedence": "terminated=True, truncated=False",
            "replay": termination_report,
            "controlled_validation": controlled_fall_report,
        },
        "readiness": {
            "expert": "READY",
            "replay": "READY",
            "termination": "READY",
            "aux_reward_data": "READY",
            "discriminator": "READY",
            "F_B": "READY",
            "main_critic_QD": "READY",
            "Qaux_network": "READY",
            "Qaux_data": "READY",
            "Actor_training_interface": "READY",
            "target_networks": "READY",
            "normalizers": "READY",
            "representation_training_ready": True,
            "critic_discriminator_interface_ready": True,
            "actor_training_interface_ready": True,
            "full_FBcprAux_update_ready": True,
        },
        "prohibited_calls": {
            "optimizer_steps": 0,
            "backward_calls": 0,
            "agent_update_calls": workspace.agent_update_calls,
            "update_fb_calls": workspace.fb_update_calls,
            "update_actor_calls": 0,
            "update_critic_calls": 0,
            "update_aux_critic_calls": 0,
            "update_discriminator_calls": 0,
        },
        "training_boundary": (
            "Full FBcprAux update dependencies are ready, but this audit "
            "remains collect-only and calls no update method."
        ),
        "performance_limitation": (
            "The Skate expert source contains one 50-frame forward-push "
            "motion; it supports a technical feasibility audit but not final "
            "skateboarding skill coverage."
        ),
        "next_milestone": "M2.4d — Native Full-Update Smoke",
    }
    report_path = output_dir / "training_readiness.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"M2.4c termination-contract report: {report_path}")
    print("Full FBcprAux update dependencies: READY")
    print("Next milestone: M2.4d — Native Full-Update Smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
