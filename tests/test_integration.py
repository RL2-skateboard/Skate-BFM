import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from skate_husky import AUX_REWARD_KEYS, HuskyLiteEnv, LiveFallDetector
from skate_husky.lite_env import (
    contact_tangential_speed,
    world_horizontal_orientation_penalty,
)

from skate_bfm.integration.actions import BFM0_JOINTS, HUSKY_JOINTS, Bfm0ToHusky23
from skate_bfm.integration.online import HuskyBfmOnlineEnv
from skate_bfm.runner import run_smoke


def _set_root_tilt(env: HuskyLiteEnv) -> None:
    env.data.qpos[3:7] = (
        math.sqrt(0.5),
        math.sqrt(0.5),
        0.0,
        0.0,
    )
    mujoco.mj_forward(env.model, env.data)


def _collection_rollout_split():
    module_name = "rollout_split_for_test"
    module_path = (
        Path(__file__).resolve().parents[1]
        / "train"
        / "scripts"
        / "data_collection"
        / "rollout_split.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the collection fall detector.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _train_skate_bfm_module():
    module_name = "train_skate_bfm_for_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    script_dir = Path(__file__).resolve().parents[1] / "train" / "scripts"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        module_name,
        script_dir / "train_skate_bfm.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the Skate training configuration.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_joint_mapping_is_name_based() -> None:
    adapter = Bfm0ToHusky23(action_clip=None)
    source = torch.arange(29, dtype=torch.float32)
    target = adapter(source)
    source_by_name = dict(zip(BFM0_JOINTS, source, strict=True))

    assert target.shape == (23,)
    assert target.tolist() == [source_by_name[name].item() for name in HUSKY_JOINTS]
    assert adapter.mapping.dropped == (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )


def test_joint_mapping_supports_batches_and_clipping() -> None:
    adapter = Bfm0ToHusky23(action_gain=2.0, action_clip=1.0)
    target = adapter(torch.full((3, 29), 0.75))
    assert target.shape == (3, 23)
    assert torch.all(target == 1.0)


def test_husky_lite_env_runs_headless() -> None:
    env = HuskyLiteEnv()
    try:
        observation = env.reset()
        next_observation = env.step(np.zeros(23, dtype=np.float32))
    finally:
        env.close()

    assert observation["joint_position"].shape == (23,)
    assert observation["root_linear_velocity"].shape == (3,)
    assert observation["root_angular_velocity"].shape == (3,)
    assert observation["board_linear_velocity"].shape == (3,)
    assert observation["board_angular_velocity"].shape == (3,)
    assert next_observation["root_height"] > 0.0


def test_husky_reset_applies_configured_joint_offsets() -> None:
    env = HuskyLiteEnv()
    joint_name = "robot/left_hip_pitch_joint"
    joint_index = env._robot_joints.index(joint_name)
    try:
        baseline = env.reset()["joint_position"][joint_index]
        env.set_reset_joint_offsets({joint_name: 0.005})
        offset = env.reset()["joint_position"][joint_index]
    finally:
        env.close()

    assert np.isclose(offset - baseline, 0.005, atol=1e-6)


def test_infinite_smoke_requires_viewer() -> None:
    with pytest.raises(ValueError, match="viewer=True"):
        run_smoke(steps=0)


def test_husky_aux_reward_contract_uses_physical_actuators() -> None:
    env = HuskyLiteEnv()
    try:
        env.reset()
        assert env.last_aux_rewards == {name: 0.0 for name in AUX_REWARD_KEYS}

        action = np.full(23, 0.2, dtype=np.float32)
        env.step(action)
        rewards = env.last_aux_rewards
    finally:
        env.close()

    assert tuple(rewards) == AUX_REWARD_KEYS
    assert all(np.isfinite(value) and value >= 0.0 for value in rewards.values())
    assert np.isclose(rewards["penalty_action_rate"], 23 * 0.2**2)
    assert len(env.physical_actuator_report) == 23
    for item in env.physical_actuator_report:
        assert item["actuator_name"] == item["joint_name"]
        assert item["transmission_type"] == "mjTRN_JOINT"
        assert item["force_limited"] is True
        assert item["derived_joint_torque_limit"] > 0.0


def test_aux_action_rate_uses_executed_husky_action_without_wrists() -> None:
    env = HuskyBfmOnlineEnv()
    try:
        env.reset()
        action = torch.zeros(29)
        action[19:22] = 1.0
        action[26:29] = -1.0
        transition = env.step(action, torch.zeros(256), truncated=True)
    finally:
        env.close()

    assert torch.equal(transition.action_husky, torch.zeros(23))
    assert transition.aux_rewards["penalty_action_rate"] == 0.0
    buffer_data = transition.as_buffer_data()
    assert tuple(buffer_data["aux_rewards"]) == AUX_REWARD_KEYS
    assert {
        name: tuple(value.shape)
        for name, value in buffer_data["aux_rewards"].items()
    } == {name: (1, 1) for name in AUX_REWARD_KEYS}


def test_surface_relative_slippage_and_world_orientation_formulas() -> None:
    world_velocity = np.array((1.0, 0.0, 0.0))
    board_velocity = np.array((1.0, 0.0, 0.0))

    assert contact_tangential_speed(np.zeros(3), np.array((0.0, 0.0, 1.0))) == 0.0
    assert np.linalg.norm(world_velocity) > 0.0
    assert contact_tangential_speed(
        world_velocity - board_velocity,
        np.array((0.0, 0.0, 1.0)),
    ) == 0.0
    assert contact_tangential_speed(
        np.array((0.5, 0.0, 0.0)),
        np.array((0.0, 0.0, 1.0)),
    ) == 0.5
    assert world_horizontal_orientation_penalty(
        np.array((0.0, 0.0, 1.0))
    ) == 0.0
    assert world_horizontal_orientation_penalty(
        np.array((1.0, 0.0, 0.0))
    ) == 1.0


def test_ankle_roll_aux_reward_is_squared_joint_sum() -> None:
    env = HuskyLiteEnv()
    try:
        env.reset()
        left = env.model.joint("robot/left_ankle_roll_joint")
        right = env.model.joint("robot/right_ankle_roll_joint")
        env.data.qpos[left.qposadr[0]] = 0.2
        env.data.qpos[right.qposadr[0]] = -0.3
        mujoco.mj_forward(env.model, env.data)
        reward = env._compute_aux_rewards()["penalty_ankle_roll"]
    finally:
        env.close()

    assert np.isclose(reward, 0.2**2 + (-0.3) ** 2)


def test_collection_and_online_share_the_fall_detector() -> None:
    assert _collection_rollout_split().LiveFallDetector is LiveFallDetector


def test_fall_detector_requires_persistent_physical_fall() -> None:
    env = HuskyLiteEnv()
    try:
        env.reset()
        fallen, _, diagnostics = env.fall_detector.check(env.data)
        assert not fallen
        assert diagnostics["feet_on_board"] is True

        env.fall_detector.reset()
        _set_root_tilt(env)
        fallen, _, diagnostics = env.fall_detector.check(env.data)
        assert not fallen
        assert diagnostics["fall_candidate"] is True
        assert diagnostics["confirm_frames"] == 1
        for _ in range(env.fall_detector.confirm_frames - 1):
            fallen, _, _ = env.fall_detector.check(env.data)
        assert fallen

        env.reset()
        env.data.qpos[2] = 0.2
        mujoco.mj_forward(env.model, env.data)
        fallen, _, diagnostics = env.fall_detector.check(env.data)
        assert not fallen
        assert diagnostics["illegal_contact"] is True
        assert diagnostics["feet_on_board"] is False
        for _ in range(env.fall_detector.confirm_frames - 1):
            fallen, _, _ = env.fall_detector.check(env.data)
        assert fallen

        env.reset()
        board_joint = env.model.joint("skateboard/floating_base_joint_skateboard")
        env.data.qpos[board_joint.qposadr[0]] += 5.0
        mujoco.mj_forward(env.model, env.data)
        fallen, _, diagnostics = env.fall_detector.check(env.data)
        assert not fallen
        assert diagnostics["feet_on_board"] is False
        assert diagnostics["fall_candidate"] is False
    finally:
        env.close()


def test_online_fall_terminates_and_horizon_truncates() -> None:
    env = HuskyBfmOnlineEnv()
    action = torch.zeros(29)
    z = torch.zeros(256)
    try:
        env.reset()
        horizon = env.step(action, z, truncated=True)
        assert not horizon.terminated
        assert horizon.truncated

        env.reset()
        _set_root_tilt(env.env)
        for index in range(env.env.fall_detector.confirm_frames):
            transition = env.step(
                action,
                z,
                truncated=index == env.env.fall_detector.confirm_frames - 1,
            )
        assert transition.terminated
        assert not transition.truncated
        assert transition.raw_metadata["fall"] is True
        assert (
            transition.raw_metadata["confirm_frames"]
            == env.env.fall_detector.confirm_frames
        )
        with pytest.raises(RuntimeError, match="reset"):
            env.step(action, z)
    finally:
        env.close()


def test_native_full_update_mode_is_single_step_only(monkeypatch) -> None:
    module = _train_skate_bfm_module()
    root = Path(__file__).resolve().parents[1]
    expert_motion = (
        root
        / "train"
        / "dataset"
        / "skate-expert-pose"
        / "motion_library"
        / "skate_expert.pkl"
    )
    monkeypatch.setenv("SKATE_ONLINE_ENV", "skate")
    monkeypatch.setenv("SKATE_UPDATE_MODE", "full")
    monkeypatch.setenv("SKATE_COLLECT_ONLY", "0")
    monkeypatch.setenv("SKATE_ADAPTATION_UPDATES", "1")
    monkeypatch.setenv("SKATE_MAX_STEPS", "1024")
    monkeypatch.setenv("SKATE_EXPERT_RATIO", "0.5")
    monkeypatch.setenv("SKATE_EXPERT_MOTION_FILE", str(expert_motion))

    cfg = module.build_train_config()
    assert cfg.skate_update_mode == "full"
    assert not cfg.collect_only
    assert cfg.adaptation_updates == 1

    monkeypatch.setenv("SKATE_ADAPTATION_UPDATES", "0")
    monkeypatch.delenv("SKATE_MAX_STEPS")
    cfg = module.build_train_config()
    assert cfg.adaptation_updates == 0
    assert cfg.skate_max_steps == module.SKATE_CLOSED_LOOP_TRANSITIONS

    monkeypatch.setenv("SKATE_MAX_STEPS", "1024")
    monkeypatch.setenv("SKATE_ADAPTATION_UPDATES", "10")
    cfg = module.build_train_config()
    assert cfg.adaptation_updates == 10
    monkeypatch.setenv("SKATE_ADAPTATION_UPDATES", "100")
    cfg = module.build_train_config()
    assert cfg.adaptation_updates == 100

    monkeypatch.setenv("SKATE_COLLECT_ONLY", "1")
    with pytest.raises(ValueError, match="collect_only=False"):
        module.build_train_config()

    monkeypatch.setenv("SKATE_COLLECT_ONLY", "0")
    monkeypatch.setenv("SKATE_ADAPTATION_UPDATES", "2")
    with pytest.raises(
        ValueError,
        match="adaptation_updates=0, 1, 10, or 100",
    ):
        module.build_train_config()

    monkeypatch.setenv("SKATE_ADAPTATION_UPDATES", "1000")
    with pytest.raises(
        ValueError,
        match="adaptation_updates=0, 1, 10, or 100",
    ):
        module.build_train_config()


def test_closed_loop_schedule_and_checkpoint_contract() -> None:
    module = _train_skate_bfm_module()

    assert module.closed_loop_update_steps(
        module.SKATE_CLOSED_LOOP_TRANSITIONS
    ) == (1500, 2000)
    schedule = module.closed_loop_update_steps(module.SKATE_BASELINE_TRANSITIONS)
    assert schedule[0] == module.SKATE_CLOSED_LOOP_FIRST_UPDATE
    assert schedule[-1] == module.SKATE_BASELINE_TRANSITIONS
    assert len(schedule) == 38
    assert all(
        right - left == module.SKATE_CLOSED_LOOP_UPDATE_EVERY
        for left, right in zip(schedule, schedule[1:])
    )
    assert (
        len(schedule) * module.SKATE_CLOSED_LOOP_UPDATES_PER_BLOCK
        == 1900
    )
    assert module.closed_loop_checkpoint_steps(
        module.SKATE_CLOSED_LOOP_TRANSITIONS
    ) == ()
    assert module.closed_loop_checkpoint_steps(
        module.SKATE_BASELINE_TRANSITIONS
    ) == (10_000, 20_000)
    with pytest.raises(ValueError, match="2000 or 20000"):
        module.closed_loop_update_steps(10_000)


@pytest.mark.parametrize(
    ("adaptation_updates", "expected_route"),
    ((0, "closed_loop"), (1, "smoke"), (10, "smoke"), (100, "smoke")),
)
def test_full_mode_routes_closed_loop_only_for_zero_updates(
    monkeypatch,
    tmp_path: Path,
    adaptation_updates: int,
    expected_route: str,
) -> None:
    module = _train_skate_bfm_module()
    workspace = object.__new__(module.Workspace)
    workspace.cfg = SimpleNamespace(
        use_trajectory_buffer=False,
        buffer_size=8,
        buffer_device="cpu",
        skate_update_mode="full",
        adaptation_updates=adaptation_updates,
    )
    workspace.work_dir = tmp_path
    workspace.training_with_expert_data = False
    workspace.uses_base_online_env = False
    workspace.train_env = object()
    workspace.train_env_info = None
    routes = []

    class FakeBuffer:
        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(module, "DictBuffer", FakeBuffer)
    monkeypatch.setattr(
        workspace,
        "_closed_loop_skate_baseline",
        lambda replay: routes.append(("closed_loop", replay)),
    )
    monkeypatch.setattr(
        workspace,
        "_full_skate_update",
        lambda replay: routes.append(("smoke", replay)),
    )
    monkeypatch.setattr(
        workspace,
        "_collect_skate_online",
        lambda replay: routes.append(("collect", replay)),
    )

    workspace.train_online()

    assert [route for route, _ in routes] == [expected_route]
    replay = routes[0][1]
    assert replay["train"] is replay["train_skate"]
