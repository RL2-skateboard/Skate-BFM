import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import mujoco
import numpy as np
import pytest
import torch
import yaml
from skate_husky import (
    AUX_REWARD_KEYS,
    HuskyLiteEnv,
    randomize_husky_play_physics,
)
from skate_husky.lite_env import (
    contact_tangential_speed,
    world_horizontal_orientation_penalty,
)

from skate_bfm.integration.actions import (
    BFM0_ACTION_CLIP,
    BFM0_ACTION_CONSUMERS,
    BFM0_ACTION_RESCALE,
    BFM0_ACTION_SCALE,
    BFM0_ACTION_TARGET_GAINS,
    BFM0_DEFAULT_JOINT_POSITION,
    BFM0_EFFORT_LIMITS,
    BFM0_INACTIVE_ACTION_INDICES,
    BFM0_INACTIVE_JOINTS,
    BFM0_JOINTS,
    BFM0_KD,
    BFM0_KP,
    HUSKY_JOINTS,
    Bfm0ToHusky23,
    install_husky_action_projection,
    official_husky_actuator_parameters,
    official_husky_control_parameters,
    project_husky_bfm_action,
)
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


def _evaluator_module():
    name = "evaluator_for_test"
    if name in sys.modules:
        return sys.modules[name]
    script_path = Path(__file__).parents[1] / "train/scripts/evaluator.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the evaluator entrypoint.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _raw_metadata(
    env: HuskyBfmOnlineEnv,
    qpos: np.ndarray,
    qvel: np.ndarray,
    source_physics: dict[str, object],
) -> dict[str, object]:
    model = env.env.model
    return {
        "nq": model.nq,
        "nv": model.nv,
        "joint_order": [
            model.joint(index).name
            for index in range(model.njnt)
            if (model.joint(index).name or "").startswith("robot/")
            and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
        ],
        "board_joint_order": [
            model.joint(index).name
            for index in range(model.njnt)
            if (model.joint(index).name or "").startswith("skateboard/")
            and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
        ],
        "qpos_quaternion_order": "wxyz",
        "robot_xml": str(env.env.xml_path.resolve()),
        "physics_randomization": source_physics,
        "fields": {
            "qpos": {"shape": list(qpos.shape), "dtype": str(qpos.dtype)},
            "qvel": {"shape": list(qvel.shape), "dtype": str(qvel.dtype)},
        },
    }


def _set_root_tilt(env: HuskyLiteEnv) -> None:
    env.data.qpos[3:7] = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)


def _official_robot_config(name: str) -> dict[str, object]:
    path = (
        Path(__file__).parents[1]
        / "train/scripts/isaac_env/humanoidverse/config/robot/g1"
        / name
    )
    config = yaml.safe_load(path.read_text())["robot"]
    if config.get("dof_names") != list(BFM0_JOINTS):
        raise RuntimeError(f"Official BFM joint order mismatch: {path}")
    return config


def _official_joint_vector(config: dict[str, object], field: str) -> np.ndarray:
    values = config["control"][field]
    result = []
    for name in BFM0_JOINTS:
        key = name.removesuffix("_joint").removeprefix("left_").removeprefix("right_")
        if key not in values:
            raise RuntimeError(f"Official BFM {field} is missing {name}.")
        result.append(values[key])
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (29,) or not np.isfinite(result).all():
        raise RuntimeError(f"Official BFM {field} vector is invalid.")
    return result


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


def test_hard_waist_contract_has_official_config_provenance() -> None:
    hard = _official_robot_config("g1_29dof_hard_waist.yaml")
    ordinary = _official_robot_config("g1_29dof.yaml")
    q0 = np.asarray(
        [hard["init_state"]["default_joint_angles"][name] for name in BFM0_JOINTS]
    )
    effort = np.asarray(hard["dof_effort_limit_list"])
    hard_kp = _official_joint_vector(hard, "stiffness")

    assert np.max(np.abs(BFM0_DEFAULT_JOINT_POSITION - q0)) <= 1e-6
    assert np.max(np.abs(BFM0_KP - hard_kp)) <= 1e-6
    assert np.max(np.abs(BFM0_KD - _official_joint_vector(hard, "damping"))) <= 1e-6
    assert np.max(np.abs(BFM0_EFFORT_LIMITS - effort)) <= 1e-6
    expected_gain = BFM0_ACTION_RESCALE * BFM0_ACTION_SCALE * effort / hard_kp
    assert np.max(np.abs(BFM0_ACTION_TARGET_GAINS - expected_gain)) <= 1e-6
    assert hard["control"]["action_scale"] == BFM0_ACTION_SCALE
    assert hard["control"]["action_clip_value"] == BFM0_ACTION_CLIP
    assert hard["control"]["normalize_action_from"] == 1.0
    assert hard["control"]["normalize_action_to"] == BFM0_ACTION_RESCALE
    assert hard["control"]["action_rescale"] is True
    assert np.max(
        np.abs(BFM0_KP - _official_joint_vector(ordinary, "stiffness"))
    ) > 1.0


def test_online_target_uses_hard_waist_gain() -> None:
    action = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
    action = project_husky_bfm_action(action)
    indices = [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS]
    expected = (
        BFM0_DEFAULT_JOINT_POSITION + BFM0_ACTION_TARGET_GAINS * action
    )[indices]
    neutral, scale = official_husky_control_parameters()
    assert np.allclose(neutral + scale * action[indices], expected, atol=1e-6)

    env = HuskyBfmOnlineEnv()
    try:
        env.reset()
        transition = env.step(torch.from_numpy(action), torch.zeros(256))
        assert np.allclose(env.env.data.ctrl[:23], expected, atol=1e-6)
        assert torch.equal(transition.action_bfm, torch.from_numpy(action))
    finally:
        env.close()


def test_hard_waist_actuators_do_not_modify_skateboard() -> None:
    env = HuskyLiteEnv()
    try:
        board_ids = range(env.robot_action_dim, env.model.nu)
        before = {
            index: (
                env.model.actuator_gainprm[index].copy(),
                env.model.actuator_biasprm[index].copy(),
                env.model.actuator_forcerange[index].copy(),
            )
            for index in board_ids
        }
        kp, kd, effort = official_husky_actuator_parameters()
        env.set_actuator_control_parameters(HUSKY_JOINTS, kp, kd, effort)
        assert np.allclose(env.model.actuator_gainprm[:23, 0], kp)
        assert np.allclose(env.model.actuator_biasprm[:23, 1], -kp)
        assert np.allclose(env.model.actuator_biasprm[:23, 2], -kd)
        assert np.allclose(env.model.actuator_forcerange[:23, 0], -effort)
        assert np.allclose(env.model.actuator_forcerange[:23, 1], effort)
        for index, values in before.items():
            assert np.array_equal(env.model.actuator_gainprm[index], values[0])
            assert np.array_equal(env.model.actuator_biasprm[index], values[1])
            assert np.array_equal(env.model.actuator_forcerange[index], values[2])
        with pytest.raises(ValueError, match="23 unique"):
            env.set_actuator_control_parameters(HUSKY_JOINTS[:-1], kp, kd, effort)
        with pytest.raises(ValueError, match="finite 23D"):
            invalid = kp.copy()
            invalid[0] = np.nan
            env.set_actuator_control_parameters(HUSKY_JOINTS, invalid, kd, effort)
        with pytest.raises(TypeError, match="floating arrays"):
            env.set_actuator_control_parameters(
                HUSKY_JOINTS, np.ones(23, dtype=np.int64), kd, effort
            )
    finally:
        env.close()


def test_husky_action_projection_contract() -> None:
    expected_inactive = (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    active_indices = tuple(
        index for index in range(len(BFM0_JOINTS))
        if index not in BFM0_INACTIVE_ACTION_INDICES
    )
    assert BFM0_INACTIVE_JOINTS == expected_inactive
    assert BFM0_INACTIVE_ACTION_INDICES == (19, 20, 21, 26, 27, 28)
    assert {BFM0_JOINTS[index] for index in active_indices} == set(HUSKY_JOINTS)
    assert set(active_indices) | set(BFM0_INACTIVE_ACTION_INDICES) == set(range(29))

    tensor = torch.randn(2, 3, 29, dtype=torch.float64)
    projected_tensor = project_husky_bfm_action(tensor)
    assert projected_tensor.shape == tensor.shape
    assert projected_tensor.dtype == tensor.dtype
    assert projected_tensor.device == tensor.device
    assert torch.equal(projected_tensor[..., active_indices], tensor[..., active_indices])
    assert torch.count_nonzero(
        projected_tensor[..., BFM0_INACTIVE_ACTION_INDICES]
    ) == 0

    array = np.arange(58, dtype=np.float32).reshape(2, 29)
    projected_array = project_husky_bfm_action(array)
    assert projected_array.shape == array.shape
    assert projected_array.dtype == array.dtype
    assert np.array_equal(projected_array[..., active_indices], array[..., active_indices])
    assert np.count_nonzero(
        projected_array[..., BFM0_INACTIVE_ACTION_INDICES]
    ) == 0
    with pytest.raises(ValueError, match="Expected 29"):
        project_husky_bfm_action(torch.zeros(28))
    with pytest.raises(TypeError, match="floating dtype"):
        project_husky_bfm_action(np.zeros(29, dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        project_husky_bfm_action(torch.full((29,), torch.nan))


def test_husky_action_projection_preserves_physical_action_and_autograd() -> None:
    adapter = Bfm0ToHusky23(action_clip=None)
    raw = torch.randn(4, 29, requires_grad=True)
    projected = project_husky_bfm_action(raw)
    assert torch.equal(adapter(raw), adapter(projected))

    projected.sum().backward()
    active_indices = tuple(
        index for index in range(29)
        if index not in BFM0_INACTIVE_ACTION_INDICES
    )
    assert torch.equal(
        raw.grad[..., active_indices],
        torch.ones_like(raw.grad[..., active_indices]),
    )
    assert torch.count_nonzero(raw.grad[..., BFM0_INACTIVE_ACTION_INDICES]) == 0


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


def test_husky_reset_accepts_complete_expert_state() -> None:
    env = HuskyLiteEnv()
    try:
        env.reset()
        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        qpos[0] += 0.25
        env.step(np.full(23, 0.1, dtype=np.float32))
        env.data.qacc_warmstart.fill(1.0)
        env.data.qfrc_applied.fill(1.0)
        env.data.xfrc_applied.fill(1.0)
        assert env.data.time > 0.0
        env.reset(qpos=qpos, qvel=qvel)
        assert env.data.time == 0.0
        assert np.array_equal(env.data.qpos, qpos)
        assert np.array_equal(env.data.qvel, qvel)
        assert np.array_equal(env.data.qacc_warmstart, np.zeros(env.model.nv))
        assert np.array_equal(env.data.qfrc_applied, np.zeros(env.model.nv))
        assert np.array_equal(env.data.xfrc_applied, np.zeros((env.model.nbody, 6)))
        with pytest.raises(ValueError, match="provided together"):
            env.reset(qpos=qpos)
        with pytest.raises(ValueError, match="Expected qpos"):
            env.reset(qpos=qpos[:-1], qvel=qvel)
    finally:
        env.close()


def test_online_transition_serializes_bfm_replay_schema() -> None:
    env = HuskyBfmOnlineEnv()
    try:
        env.reset()
        action = torch.linspace(-0.8, 0.8, 29)
        action[list(BFM0_INACTIVE_ACTION_INDICES)] = 1.0
        z = torch.linspace(-1.0, 1.0, 256)
        transition = env.step(action, z, truncated=True)
    finally:
        env.close()

    data = transition.as_buffer_data()
    assert transition.action_bfm.shape == (29,)
    assert transition.action_husky.shape == (23,)
    assert torch.count_nonzero(
        transition.action_bfm[list(BFM0_INACTIVE_ACTION_INDICES)]
    ) == 0
    assert torch.equal(data["action"][0], transition.action_bfm)
    assert torch.equal(data["z"][0], transition.z)
    assert torch.equal(transition.z, z)
    assert torch.count_nonzero(
        transition.next_observation["last_action"][
            list(BFM0_INACTIVE_ACTION_INDICES)
        ]
    ) == 0
    assert torch.equal(
        transition.action_husky,
        env.action_adapter(project_husky_bfm_action(action)),
    )
    assert transition.truncated and not transition.terminated
    assert tuple(data["aux_rewards"]) == AUX_REWARD_KEYS
    assert all(value.shape == (1, 1) for value in data["aux_rewards"].values())


def test_online_history_contains_only_effective_husky_actions() -> None:
    env = HuskyBfmOnlineEnv()
    try:
        env.reset()
        first = torch.linspace(-0.5, 0.5, 29)
        first[list(BFM0_INACTIVE_ACTION_INDICES)] = 0.9
        env.step(first, torch.zeros(256))
        second = torch.linspace(0.5, -0.5, 29)
        second[list(BFM0_INACTIVE_ACTION_INDICES)] = -0.9
        transition = env.step(second, torch.zeros(256))

        history = env.observation_adapter._history
        widths = {
            key: int(values[0].size)
            for key, values in history.items()
        }
        action_offset = sum(
            env.observation_adapter.history_length * widths[key]
            for key in sorted(history)
            if key < "actions"
        )
        action_width = env.observation_adapter.history_length * widths["actions"]
        action_history = transition.next_observation["history_actor"][
            action_offset : action_offset + action_width
        ].reshape(env.observation_adapter.history_length, widths["actions"])
    finally:
        env.close()

    active_indices = tuple(
        index for index in range(29)
        if index not in BFM0_INACTIVE_ACTION_INDICES
    )
    assert torch.count_nonzero(
        action_history[..., BFM0_INACTIVE_ACTION_INDICES]
    ) == 0
    assert torch.equal(
        action_history[0, list(active_indices)],
        first[list(active_indices)] * 5.0,
    )


class _RecordingActionConsumer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.received: torch.Tensor | None = None

    def forward(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        self.received = action
        return action * self.scale


class _ActionConsumerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in BFM0_ACTION_CONSUMERS:
            setattr(self, name, _RecordingActionConsumer())


def test_native_action_consumers_project_positional_and_keyword_actions() -> None:
    model = _ActionConsumerModel()
    state_before = tuple(model.state_dict())
    assert install_husky_action_projection(model)
    assert not install_husky_action_projection(model)
    assert tuple(model.state_dict()) == state_before

    action = torch.ones(2, 29)
    for name in BFM0_ACTION_CONSUMERS:
        consumer = getattr(model, name)
        if name == "_target_aux_critic":
            consumer(
                observation=torch.zeros(2, 1),
                z=torch.zeros(2, 1),
                action=action,
            )
        else:
            consumer(torch.zeros(2, 1), torch.zeros(2, 1), action)
        assert consumer.received is not None
        assert torch.count_nonzero(
            consumer.received[..., BFM0_INACTIVE_ACTION_INDICES]
        ) == 0

    incomplete = _ActionConsumerModel()
    del incomplete._target_aux_critic
    with pytest.raises(RuntimeError, match="_target_aux_critic"):
        install_husky_action_projection(incomplete)
    assert not hasattr(incomplete, "_skate_husky_action_projection_handles")


class _TrackingModel:
    def __init__(self, offset: float = 0.0) -> None:
        self.cfg = SimpleNamespace(
            seq_length=3,
            archi=SimpleNamespace(z_dim=2),
        )
        self.device = torch.device("cpu")
        self.offset = offset
        self.seen: dict[str, torch.Tensor] | None = None

    def tracking_inference(
        self,
        next_observation: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        self.seen = {
            key: value.clone()
            for key, value in next_observation.items()
        }
        z = next_observation["state"][:, :2].clone() + self.offset
        for step in range(z.shape[0]):
            z[step] = z[step : step + self.cfg.seq_length].mean(dim=0)
        return z


def test_aligned_tracking_context_uses_next_states_and_stays_in_trajectory() -> None:
    module = _training_module()
    observations = {
        "state": torch.tensor([
            [0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5],
            [100.0, 100.5], [101.0, 101.5], [102.0, 102.5],
        ]),
        "privileged_state": torch.zeros(8, 1),
        "last_action": torch.zeros(8, 1),
    }
    context = module.AlignedSkateTrackingContext(
        observations,
        ["first", "second"],
        [5, 3],
        {"first": 5, "second": 4},
    )
    model = _TrackingModel()
    z, ranges = context.encode(model, "first", local_frame=1, steps=8)

    assert model.seen is not None
    assert torch.equal(model.seen["state"][:, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(
        z,
        torch.tensor([[3.0, 3.5], [3.5, 4.0], [4.0, 4.5]]),
    )
    assert ranges[0] == {
        "expert_state_index": 1,
        "future_start": 2,
        "future_end": 4,
        "future_count": 3,
    }
    assert ranges[-1]["future_count"] == 1
    assert context.eligible_frame_count("first", 1) == 4
    assert context.eligible_frame_count("first", 3) == 2
    assert context.eligible_frame_count("second", 3) == 0
    with pytest.raises(RuntimeError, match="no valid next"):
        context.encode(model, "first", local_frame=4, steps=1)


def test_aligned_tracking_context_uses_checkpoint_specific_model() -> None:
    module = _training_module()
    observations = {
        "state": torch.arange(12, dtype=torch.float32).reshape(6, 2),
        "privileged_state": torch.zeros(6, 1),
        "last_action": torch.zeros(6, 1),
    }
    context = module.AlignedSkateTrackingContext(
        observations,
        ["motion"],
        [6],
        {"motion": 6},
    )
    first, _ = context.encode(_TrackingModel(offset=0.0), "motion", 0, 2)
    second, _ = context.encode(_TrackingModel(offset=10.0), "motion", 0, 2)
    assert not torch.equal(first, second)


class _MarkerZBuffer:
    def __init__(self, marker: float | None = None) -> None:
        self.marker = marker
        self.sample_calls = 0

    def empty(self) -> bool:
        return self.marker is None

    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        self.sample_calls += 1
        return torch.full((count, 2), float(self.marker), device=device)


class _RolloutAgent:
    def __init__(self, z_buffer_marker: float | None = None) -> None:
        self.device = torch.device("cpu")
        self.sample_calls = 0
        self.z_buffer = _MarkerZBuffer(z_buffer_marker)
        self.cfg = SimpleNamespace(train=SimpleNamespace(
            use_mix_rollout=True,
            update_z_every_step=100,
            z_buffer_size=8192,
            rollout_expert_trajectories_percentage=0.5,
            rollout_expert_trajectories_length=250,
        ))
        self._model = SimpleNamespace(sample_z=self.sample_z)

    def sample_z(self, count: int, device: torch.device) -> torch.Tensor:
        self.sample_calls += 1
        return torch.full((count, 2), float(self.sample_calls), device=device)


def test_rollout_latent_roles_and_background_lifecycle() -> None:
    module = _training_module()
    first = module.select_expert_rollout_envs(4, 0.5, 4729)
    second = module.select_expert_rollout_envs(4, 0.5, 4729)
    assert first == second
    assert len(first) == len(set(first)) == 2

    agent = _RolloutAgent()
    context = module.SkateRolloutContext(agent, 1, ())
    context.reset(0, None)
    z0 = context.effective_z([0])
    z1 = context.effective_z([1])
    z100 = context.effective_z([100])
    z200 = context.effective_z([200])
    assert torch.equal(z0, z1)
    assert [z0[0, 0], z100[0, 0], z200[0, 0]] == [1, 2, 3]
    assert context.counters["background_random_samples"] == 3

    buffered_agent = _RolloutAgent(z_buffer_marker=9.0)
    buffered = module.SkateRolloutContext(buffered_agent, 1, ())
    buffered.reset(0, None)
    assert torch.equal(buffered.effective_z([0]), torch.full((1, 2), 9.0))
    assert buffered_agent.sample_calls == 0
    assert buffered_agent.z_buffer.sample_calls == 1
    assert buffered.counters["background_zbuffer_samples"] == 1


def test_rollout_tracking_override_tail_cap_and_episode_reset() -> None:
    module = _training_module()
    agent = _RolloutAgent()
    context = module.SkateRolloutContext(agent, 2, (0,))
    tracking = torch.tensor([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]])
    context.reset(0, tracking)
    context.reset(1, None)

    first = context.effective_z([0, 0])
    assert torch.equal(first[0], tracking[0])
    assert torch.equal(first[1], context.background[1])
    assert context.background[0] is not None
    context.effective_z([1, 1])
    context.effective_z([2, 2])
    tail = context.effective_z([3, 3])
    assert torch.equal(tail[0], context.background[0])
    assert context.counters["tracking_transitions"] == 3
    assert context.counters["post_tracking_free_transitions"] == 1

    capped = torch.arange(500, dtype=torch.float32).reshape(250, 2)
    context.reset(0, capped)
    assert torch.equal(context.effective_z([249, 4])[0], capped[249])
    assert torch.equal(context.effective_z([250, 5])[0], context.background[0])
    context.reset(0, torch.full((1, 2), 77.0))
    assert torch.equal(context.effective_z([0, 6])[0], torch.full((2,), 77.0))
    context.reset(0, None)
    assert context.counters["tracking_unavailable_resets"] == 1


def test_expert_tracking_uses_skate_without_changing_update_mixture() -> None:
    module = _training_module()

    class Expert:
        seq_length = 8

    base = Expert()
    skate = Expert()
    replay = module.assemble_expert_replay(base, skate)
    assert replay["expert_tracking"] is skate
    assert replay["expert_slicer"].expert_base is base
    assert replay["expert_slicer"].expert_skate is skate


def test_parallel_online_environments_reset_independently() -> None:
    envs = [HuskyBfmOnlineEnv() for _ in range(4)]
    try:
        for env in envs:
            env.reset()
        transitions = [
            env.step(torch.zeros(29), torch.zeros(256), truncated=index == 0)
            for index, env in enumerate(envs)
        ]
        assert transitions[0].truncated and not transitions[0].terminated
        assert all(
            not transition.truncated and not transition.terminated
            for transition in transitions[1:]
        )
        envs[0].reset()
        assert envs[0]._step_count == 0
        assert [env._step_count for env in envs[1:]] == [1, 1, 1]
    finally:
        for env in envs:
            env.close()


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


def test_source_physics_exact_apply_and_no_cumulative_drift() -> None:
    target = HuskyLiteEnv()
    source_one = HuskyLiteEnv()
    source_two = HuskyLiteEnv()
    try:
        realization_one, _ = randomize_husky_play_physics(
            source_one.model,
            "source_one",
            101,
        )
        realization_two, _ = randomize_husky_play_physics(
            source_two.model,
            "source_two",
            202,
        )

        target.apply_source_physics(realization_one)
        first_body = target.model.body_ipos.copy()
        first_friction = target.model.geom_friction.copy()
        assert np.array_equal(first_body, source_one.model.body_ipos)
        assert np.array_equal(first_friction, source_one.model.geom_friction)

        target.apply_source_physics(realization_one)
        assert np.array_equal(target.model.body_ipos, first_body)
        assert np.array_equal(target.model.geom_friction, first_friction)

        target.apply_source_physics(realization_two)
        assert np.array_equal(target.model.body_ipos, source_two.model.body_ipos)
        assert np.array_equal(target.model.geom_friction, source_two.model.geom_friction)
        target.apply_source_physics(realization_one)
        assert np.array_equal(target.model.body_ipos, first_body)
        assert np.array_equal(target.model.geom_friction, first_friction)
    finally:
        target.close()
        source_one.close()
        source_two.close()


def test_source_physics_is_independent_between_environments() -> None:
    first = HuskyLiteEnv()
    second = HuskyLiteEnv()
    source_one = HuskyLiteEnv()
    source_two = HuskyLiteEnv()
    try:
        realization_one, _ = randomize_husky_play_physics(
            source_one.model,
            "source_one",
            303,
        )
        realization_two, _ = randomize_husky_play_physics(
            source_two.model,
            "source_two",
            404,
        )
        first.apply_source_physics(realization_one)
        first_body = first.model.body_ipos.copy()
        first_friction = first.model.geom_friction.copy()
        second.apply_source_physics(realization_two)

        assert np.array_equal(first.model.body_ipos, first_body)
        assert np.array_equal(first.model.geom_friction, first_friction)
        assert not np.array_equal(first.model.body_ipos, second.model.body_ipos)
        assert not np.array_equal(first.model.geom_friction, second.model.geom_friction)
    finally:
        first.close()
        second.close()
        source_one.close()
        source_two.close()


def test_source_physics_reset_preserves_raw_qpos_and_historical_reset() -> None:
    env = HuskyLiteEnv()
    source = HuskyLiteEnv()
    historical = HuskyLiteEnv()
    try:
        realization, offsets = randomize_husky_play_physics(
            source.model,
            "source",
            505,
        )
        env.reset()
        raw_qpos = env.data.qpos.copy()
        raw_qvel = env.data.qvel.copy()
        for joint_name, offset in offsets.items():
            raw_qpos[env.model.joint(joint_name).qposadr[0]] += offset
        env.reset(
            qpos=raw_qpos,
            qvel=raw_qvel,
            source_physics=realization,
        )
        assert np.array_equal(env.data.qpos, raw_qpos)
        assert np.array_equal(env.data.qvel, raw_qvel)

        history_realization, history_offsets = randomize_husky_play_physics(
            historical.model,
            "historical",
            606,
        )
        historical.set_reset_joint_offsets(history_offsets)
        randomized_friction = historical.model.geom_friction.copy()
        mujoco.mj_setConst(historical.model, historical.data)
        historical.reset()
        expected_qpos = historical.model.key_qpos[0].copy()
        for joint_name, offset in history_offsets.items():
            expected_qpos[historical.model.joint(joint_name).qposadr[0]] += offset
        assert np.array_equal(historical.data.qpos, expected_qpos)
        assert np.array_equal(historical.model.geom_friction, randomized_friction)
        assert historical.validate_source_physics(history_realization)["seed"] == 606

        invalid = dict(realization)
        invalid.pop("wheel_rolling_friction_scale")
        with pytest.raises(RuntimeError, match="schema mismatch"):
            env.apply_source_physics(invalid)
        invalid = dict(realization)
        invalid["unknown"] = 1
        with pytest.raises(RuntimeError, match="schema mismatch"):
            env.apply_source_physics(invalid)
        invalid = dict(realization)
        invalid["robot_torso_com_offset_m"] = [np.nan, 0.0, 0.0]
        with pytest.raises(RuntimeError, match="finite vector"):
            env.apply_source_physics(invalid)
        with pytest.raises(ValueError, match="requires explicit raw qpos/qvel"):
            env.reset(source_physics=realization)
    finally:
        env.close()
        source.close()
        historical.close()


def test_trainer_and_evaluator_share_source_physics_provenance(
    tmp_path: Path,
) -> None:
    trainer = _training_module()
    evaluator = _evaluator_module()
    env = HuskyBfmOnlineEnv()
    source = HuskyLiteEnv()
    try:
        source_physics, _ = randomize_husky_play_physics(
            source.model,
            "shared_source",
            707,
        )
        env.reset()
        qpos = np.asarray(env.env.data.qpos[None], dtype=np.float32)
        qvel = np.asarray(env.env.data.qvel[None], dtype=np.float32)
        raw_path = tmp_path / "shared_source.npz"
        np.savez(raw_path, qpos=qpos, qvel=qvel)
        raw_path.with_suffix(".json").write_text(
            json.dumps(_raw_metadata(env, qpos, qvel, source_physics))
        )
        motion_path = tmp_path / "motion.pkl"
        joblib.dump(
            {
                "motion": {
                    "source_raw_npz": str(raw_path),
                    "source_start_frame": 0,
                    "source_end_frame": 1,
                    "physics_seed": 707,
                    "dof": [[0.0] * 23],
                }
            },
            motion_path,
        )

        direct_qpos, direct_qvel, direct_physics = trainer.load_source_rollout(
            raw_path,
            env,
            707,
        )
        sampler = evaluator.ExpertResetSampler(motion_path, env, seed=0)
        eval_qpos, eval_qvel, eval_physics, provenance = sampler.sample()
        assert np.array_equal(eval_qpos, direct_qpos[0])
        assert np.array_equal(eval_qvel, direct_qvel[0])
        assert eval_physics == direct_physics
        assert provenance["physics_seed"] == 707
        assert provenance["source_physics_aligned"] is True

        with pytest.raises(RuntimeError, match="physics seed mismatch"):
            trainer.load_source_rollout(raw_path, env, 708)
    finally:
        env.close()
        source.close()


def test_official_checkpoint_stays_strictly_loadable_with_29d_actions() -> None:
    checkpoint = Path(__file__).parents[1] / "model/bfm-zero-official"
    if not checkpoint.is_dir():
        pytest.skip("Official BFM0 checkpoint is not available.")
    module = _training_module()
    agent, report = module.load_frozen_agent(checkpoint)

    assert agent.action_dim == 29
    assert report["model_sha256"] == module.OFFICIAL_BFM0_SHA256
    assert len(agent._model.state_dict()) == 537
    assert all(
        "skate_husky_action_projection" not in name
        for name in agent._model.state_dict()
    )
    assert hasattr(agent._model, "_skate_husky_action_projection_handles")

    context = module.AlignedSkateTrackingContext.load(
        agent,
        module.EXPERT_DATASETS["phase"],
    )
    assert len(context.trajectories) == 6038
    assert context.frame_difference_counts() == {-1: 4521, 0: 1517}
    assert sum(
        context.eligible_frame_count(name, 20)
        for name in context.trajectories
    ) == 335970

    motion_key = next(
        name
        for name in context.trajectories
        if context.eligible_frame_count(name, 20) > 5
    )
    local_frame = 5
    aligned, ranges = context.encode(
        agent._model,
        motion_key,
        local_frame,
        20,
    )
    trajectory = context.trajectories[motion_key]
    start = trajectory["start"] + local_frame + 1
    end = min(
        trajectory["start"] + trajectory["length"],
        start + 20 + agent._model.cfg.seq_length - 1,
    )
    direct_observation = {
        name: value[start:end].to(agent.device)
        for name, value in context.observations.items()
    }
    with torch.no_grad():
        direct = agent._model.tracking_inference(direct_observation)[:20]
    assert torch.equal(aligned, direct)
    assert aligned.shape == (20, 256)
    assert torch.isfinite(aligned).all()
    assert torch.allclose(
        aligned.norm(dim=-1),
        torch.full((20,), 16.0, device=aligned.device),
        atol=2e-6,
    )
    assert ranges[0]["future_start"] == local_frame + 1


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
    assert cfg.online_envs == 4
    assert cfg.skate_expert_ratio == 0.5
    assert cfg.agent.train.use_mix_rollout is True
    assert cfg.agent.train.update_z_every_step == 100
    assert cfg.agent.train.z_buffer_size == 8192
    assert cfg.agent.train.rollout_expert_trajectories is True
    assert cfg.agent.train.rollout_expert_trajectories_length == 250
    assert cfg.agent.train.rollout_expert_trajectories_percentage == 0.5
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

    monkeypatch.setenv("SKATE_BUFFER_SIZE", "2000")
    monkeypatch.setenv("SKATE_ONLINE_ENVS", "3")
    with pytest.raises(ValueError, match="schedule boundary"):
        module.build_train_config()


def test_raw_layout_validation_fails_closed(
    tmp_path: Path,
) -> None:
    module = _training_module()
    env = HuskyBfmOnlineEnv()
    try:
        model = env.env.model
        qpos = np.zeros((2, model.nq), dtype=np.float32)
        qvel = np.zeros((2, model.nv), dtype=np.float32)
        metadata = {
            "nq": model.nq,
            "nv": model.nv,
            "joint_order": [
                model.joint(index).name
                for index in range(model.njnt)
                if (model.joint(index).name or "").startswith("robot/")
                and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
            ],
            "board_joint_order": [
                model.joint(index).name
                for index in range(model.njnt)
                if (model.joint(index).name or "").startswith("skateboard/")
                and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
            ],
            "qpos_quaternion_order": "wxyz",
            "robot_xml": str(env.env.xml_path.resolve()),
            "fields": {
                "qpos": {"shape": list(qpos.shape), "dtype": str(qpos.dtype)},
                "qvel": {"shape": list(qvel.shape), "dtype": str(qvel.dtype)},
            },
        }
        metadata_path = tmp_path / "raw.json"
        metadata_path.write_text(json.dumps(metadata))
        module.validate_raw_layout(metadata_path, qpos, qvel, env)

        metadata["joint_order"][0] = "robot/wrong_joint"
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(RuntimeError, match="joint order mismatch"):
            module.validate_raw_layout(metadata_path, qpos, qvel, env)
    finally:
        env.close()
