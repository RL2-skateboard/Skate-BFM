import importlib.util
import json
import math
import sys
from pathlib import Path

import joblib
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
