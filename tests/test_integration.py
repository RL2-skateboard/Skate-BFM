import importlib.util
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from skate_husky import (
    AUX_REWARD_KEYS,
    HuskyLiteEnv,
    randomize_husky_play_physics,
)
from skate_husky.lite_env import (
    contact_tangential_speed,
    world_horizontal_orientation_penalty,
)

from skate_bfm.integration.actions import BFM0_JOINTS, HUSKY_JOINTS, Bfm0ToHusky23
from skate_bfm.integration.online import HuskyBfmOnlineEnv


def _training_module():
    name = "train_skate_bfm_for_test"
    if name in sys.modules:
        return sys.modules[name]
    script_path = Path(__file__).parents[1] / "train/scripts/train_skate_bfm.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the training entrypoint.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _set_root_tilt(env: HuskyLiteEnv) -> None:
    env.data.qpos[3:7] = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)


def test_bfm_actions_are_mapped_by_joint_name() -> None:
    adapter = Bfm0ToHusky23(action_clip=None)
    source = torch.arange(29, dtype=torch.float32)
    source_by_name = dict(zip(BFM0_JOINTS, source, strict=True))

    assert adapter(source).tolist() == [
        source_by_name[name].item() for name in HUSKY_JOINTS
    ]
    assert adapter.mapping.dropped == (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )


def test_husky_runtime_and_auxiliary_contract() -> None:
    env = HuskyLiteEnv()
    try:
        observation = env.reset()
        env.step(np.full(23, 0.2, dtype=np.float32))
        rewards = env.last_aux_rewards
    finally:
        env.close()

    assert observation["joint_position"].shape == (23,)
    assert tuple(rewards) == AUX_REWARD_KEYS
    assert np.isclose(rewards["penalty_action_rate"], 23 * 0.2**2)
    assert all(np.isfinite(value) and value >= 0.0 for value in rewards.values())
    assert len(env.physical_actuator_report) == 23
    assert all(
        item["actuator_name"] == item["joint_name"]
        and item["transmission_type"] == "mjTRN_JOINT"
        and item["derived_joint_torque_limit"] > 0.0
        for item in env.physical_actuator_report
    )


def test_online_transition_serializes_bfm_replay_schema() -> None:
    env = HuskyBfmOnlineEnv()
    try:
        env.reset()
        action = torch.zeros(29)
        action[19:22] = 1.0
        transition = env.step(action, torch.zeros(256), truncated=True)
    finally:
        env.close()

    data = transition.as_buffer_data()
    assert transition.action_bfm.shape == (29,)
    assert transition.action_husky.shape == (23,)
    assert torch.equal(transition.action_husky, torch.zeros(23))
    assert transition.truncated and not transition.terminated
    assert tuple(data["aux_rewards"]) == AUX_REWARD_KEYS
    assert all(value.shape == (1, 1) for value in data["aux_rewards"].values())


def test_fall_contract_ignores_foot_liftoff_and_requires_confirmation() -> None:
    env = HuskyLiteEnv()
    try:
        env.reset()
        board_joint = env.model.joint("skateboard/floating_base_joint_skateboard")
        env.data.qpos[board_joint.qposadr[0]] += 5.0
        mujoco.mj_forward(env.model, env.data)
        fallen, _, diagnostics = env.fall_detector.check(env.data)
        assert not fallen
        assert not diagnostics["feet_on_board"]
        assert not diagnostics["fall_candidate"]

        env.reset()
        _set_root_tilt(env)
        for _ in range(env.fall_detector.confirm_frames - 1):
            fallen, _, _ = env.fall_detector.check(env.data)
            assert not fallen
        fallen, _, _ = env.fall_detector.check(env.data)
        assert fallen
    finally:
        env.close()


def test_surface_relative_reward_helpers() -> None:
    assert contact_tangential_speed(
        np.array((0.5, 0.0, 0.0)), np.array((0.0, 0.0, 1.0))
    ) == 0.5
    assert world_horizontal_orientation_penalty(np.array((0.0, 0.0, 1.0))) == 0.0
    assert world_horizontal_orientation_penalty(np.array((1.0, 0.0, 0.0))) == 1.0


def test_official_randomization_is_deterministic_per_rollout_id() -> None:
    first = HuskyLiteEnv()
    second = HuskyLiteEnv()
    try:
        first_report, first_offsets = randomize_husky_play_physics(
            first.model, "eval_seen_001", None
        )
        second_report, second_offsets = randomize_husky_play_physics(
            second.model, "eval_seen_001", None
        )
    finally:
        first.close()
        second.close()

    assert first_report == second_report
    assert first_offsets == second_offsets


def test_m26_configuration_and_schedule_are_parameterized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _training_module()
    cfg = module.build_train_config()
    schedule = module.training_update_steps(
        cfg.skate_max_steps,
        cfg.first_update_transition,
        cfg.update_interval,
    )

    assert cfg.expert_dataset_kind == "phase"
    assert cfg.skate_expert_motion_file.endswith("skate_expert_phase.pkl")
    assert cfg.skate_max_steps == 100_000
    assert cfg.buffer_size == 100_000
    assert cfg.skate_expert_ratio == 0.5
    assert schedule[0] == 1500
    assert schedule[-1] == 100_000
    assert len(schedule) == 198
    assert len(schedule) * cfg.updates_per_block == 9900
    assert module.training_checkpoint_steps(100_000) == (20_000, 50_000, 100_000)
    assert module.training_checkpoint_steps(300_000) == (
        20_000,
        50_000,
        100_000,
        300_000,
    )

    monkeypatch.setenv("SKATE_EXPERT_DATASET", "continuous")
    monkeypatch.setenv("SKATE_MAX_STEPS", "2000")
    monkeypatch.setenv("SKATE_UPDATES_PER_BLOCK", "1")
    cfg = module.build_train_config()
    assert cfg.expert_dataset_kind == "continuous"
    assert cfg.skate_expert_motion_file.endswith("skate_expert_continuous.pkl")
    assert module.training_update_steps(
        cfg.skate_max_steps,
        cfg.first_update_transition,
        cfg.update_interval,
    ) == (1500, 2000)
    assert module.training_checkpoint_steps(cfg.skate_max_steps) == (2000,)

    monkeypatch.setenv("SKATE_BUFFER_SIZE", "1999")
    with pytest.raises(ValueError, match="SKATE_BUFFER_SIZE"):
        module.build_train_config()
