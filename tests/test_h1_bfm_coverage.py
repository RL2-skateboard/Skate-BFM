from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from skate_bfm.bfm0 import Bfm0Model
from skate_bfm.exp.h1_bfm_coverage import (
    _cem_config,
    _slerp_video_label,
    _video_target_label,
)
from skate_bfm.exp.h1_bfm_coverage.core import (
    BFM0_ACTION_RESCALE,
    BFM0_DEFAULT_JOINT_POSITION,
    BFM0_JOINTS,
    EXPERT_STATIC_QPOS,
    HUSKY_JOINTS,
    CemResult,
    CheckpointCompatibilityError,
    ExpertTarget,
    H1RolloutRunner,
    OfficialHuskyToBfm0Observation,
    RolloutResult,
    ScoredRollout,
    _expert_pose_observation,
    angular_distance,
    constrain_to_geodesic_cap,
    load_bfm0_checkpoint,
    load_expert_targets,
    official_husky_control_parameters,
    quaternion_rotation_error,
    run_cem,
    sample_geodesic_neighborhood,
    sample_global_latents,
    score_rollout,
    spherical_lerp,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return {
        "rollout": {
            "control_dt": 0.02,
            "horizon_seconds": 0.04,
            "action_gain": 0.0,
            "action_clip": 1.0,
            "husky_action_scale": 0.1,
            "fall_height": 0.1,
            "unsafe_root_angle": 2.0,
        },
        "scores": {
            "position_weight": 1.0,
            "rotation_weight": 0.5,
            "motion_weight": 1.0,
            "fall_penalty": 10.0,
            "unsafe_penalty": 2.0,
            "pose_error_threshold": 1.0,
            "motion_error_threshold": 1.0,
        },
    }


def _temporary_checkpoint(path: Path) -> Path:
    torch.save(
        {
            "model": Bfm0Model().state_dict(),
            "metadata": {"temporary": True, "purpose": "H1 smoke test"},
        },
        path,
    )
    return path


def test_prior_modes_select_distinct_cem_search_scales() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/h1_bfm_coverage.yaml").read_text(encoding="utf-8")
    )
    config["search"]["prior_mode"] = "with_prior"
    with_prior = _cem_config(config)
    config["search"]["prior_mode"] = "without_prior"
    without_prior = _cem_config(config)
    assert with_prior["initial_std"] == 0.25
    assert with_prior["max_angle_degrees"] == 40.0
    assert with_prior["temporal_correlation"] == 0.9
    assert without_prior["initial_std"] == 1.0
    assert without_prior["max_angle_degrees"] == 180.0
    assert without_prior["temporal_correlation"] == 0.0


def test_project_z_norm_matches_model_definition() -> None:
    model = Bfm0Model()
    projected = model.project_z(torch.randn(8, model.config.latent_dim))
    expected = math.sqrt(model.config.latent_dim)
    assert torch.allclose(
        torch.linalg.vector_norm(projected, dim=-1),
        torch.full((8,), expected),
    )


def test_global_sampling_is_reproducible() -> None:
    model = Bfm0Model()
    first = sample_global_latents(model, 4, seed=17)
    second = sample_global_latents(model, 4, seed=17)
    third = sample_global_latents(model, 4, seed=18)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_official_observation_uses_training_calibration() -> None:
    adapter = OfficialHuskyToBfm0Observation()
    observation = {
        "joint_position": np.zeros(23, dtype=np.float32),
        "joint_velocity": np.zeros(23, dtype=np.float32),
        "last_action": np.zeros(23, dtype=np.float32),
        "projected_gravity": np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        "angular_velocity": np.asarray((4.0, 8.0, 12.0), dtype=np.float32),
    }
    action = np.full(29, 0.2, dtype=np.float32)
    result = adapter(observation, action)
    assert result["state"].shape == (64,)
    assert result["history"].shape == (372,)
    assert torch.allclose(
        result["state"][:29],
        torch.from_numpy(-BFM0_DEFAULT_JOINT_POSITION),
    )
    assert torch.allclose(result["state"][-3:], torch.tensor((1.0, 2.0, 3.0)))
    assert torch.allclose(
        result["last_action"],
        torch.from_numpy(action * BFM0_ACTION_RESCALE),
    )


def test_official_control_parameters_follow_husky_joint_names() -> None:
    neutral, scale = official_husky_control_parameters(action_gain=1.0)
    assert neutral.shape == (23,)
    assert scale.shape == (23,)
    assert np.isclose(neutral[0], 0.0)
    assert np.isclose(neutral[8], -0.1)
    assert np.all(scale > 0.0)


def test_geodesic_samples_match_requested_angle() -> None:
    model = Bfm0Model()
    anchor = model.project_z(torch.randn(model.config.latent_dim))
    samples, labels = sample_geodesic_neighborhood(
        model,
        anchor,
        [5, 20, 80],
        samples_per_angle=3,
        seed=3,
    )
    actual = torch.rad2deg(
        angular_distance(samples, anchor.reshape(1, -1).repeat(len(samples), 1))
    ).numpy()
    assert np.allclose(actual, labels, atol=2e-4)


def test_geodesic_cap_limits_search_to_expert_neighborhood() -> None:
    model = Bfm0Model()
    anchor = model.project_z(torch.randn(model.config.latent_dim))
    candidates = model.project_z(torch.randn(32, model.config.latent_dim))
    capped = constrain_to_geodesic_cap(model, candidates, anchor, 20.0)
    angles = torch.rad2deg(
        angular_distance(capped, anchor.reshape(1, -1).repeat(len(capped), 1))
    )
    assert torch.all(angles <= 20.0 + 2e-4)


def test_geodesic_cap_limits_every_expert_trajectory_step() -> None:
    model = Bfm0Model()
    anchor = model.project_z(torch.randn(5, model.config.latent_dim))
    candidates = model.project_z(torch.randn(7, 5, model.config.latent_dim))
    capped = constrain_to_geodesic_cap(model, candidates, anchor, 10.0)
    angles = torch.rad2deg(angular_distance(capped, anchor.unsqueeze(0)))
    assert capped.shape == candidates.shape
    assert torch.all(angles <= 10.0 + 2e-4)


def test_cem_preserves_time_aligned_latent_trajectory() -> None:
    model = Bfm0Model()
    anchor = model.project_z(torch.randn(4, model.config.latent_dim))

    def evaluate(latent: torch.Tensor, seed: int) -> ScoredRollout:
        score = -float(angular_distance(latent, anchor).mean())
        rollout = RolloutResult(
            latent=latent.numpy(),
            states=[],
            actions=np.empty((0, 23), dtype=np.float32),
            descriptor={},
            fall=False,
            unsafe=False,
            terminated_early=False,
            seed=seed,
        )
        return ScoredRollout("trajectory", score, True, {}, rollout)

    result: CemResult = run_cem(
        model,
        evaluate,
        anchor,
        {
            "population_size": 8,
            "elite_fraction": 0.25,
            "num_iterations": 2,
            "initial_std": 0.1,
            "min_std": 0.01,
            "max_angle_degrees": 15.0,
            "temporal_correlation": 0.9,
        },
        seed=9,
    )
    assert result.best_latent.shape == anchor.shape
    assert torch.all(
        torch.rad2deg(angular_distance(result.best_latent, anchor)) <= 15.0 + 2e-4
    )


def test_static_expert_pose_builds_complete_backward_observation() -> None:
    values = np.load(
        ROOT / "husky_sim/upstream/dataset/ref_pose/push_start_pose_b.npy",
        allow_pickle=False,
    )
    observation = _expert_pose_observation(
        values,
        EXPERT_STATIC_QPOS["push_start_pose"],
    )
    assert observation["state"].shape == (1, 64)
    assert observation["privileged_state"].shape == (1, 463)
    assert all(np.isfinite(value).all() for value in observation.values())


def test_dynamic_targets_start_from_their_own_expert_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BFM_ZERO_ROOT", str(ROOT / "model/bfm-zero-source"))
    config = yaml.safe_load(
        (ROOT / "configs/h1_bfm_coverage.yaml").read_text(encoding="utf-8")
    )
    targets, _ = load_expert_targets(
        ROOT / config["expert_data"]["root"],
        ROOT / "husky_sim/upstream/test_scene/mjlab_scene.xml",
        config["expert_data"],
    )
    bfm_indices = [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS]
    for target in targets:
        if target.kind != "human_push_window":
            continue
        assert target.initial_qpos is not None
        assert target.initial_qvel is not None
        assert target.initial_pose == target.name
        assert np.allclose(
            target.initial_qpos[7:][bfm_indices],
            target.values[0],
        )
        assert target.initial_qvel.shape == (35,)
        assert np.isfinite(target.initial_qvel).all()


def test_dynamic_push_initialization_keeps_push_foot_off_board(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BFM_ZERO_ROOT", str(ROOT / "model/bfm-zero-source"))
    config = yaml.safe_load(
        (ROOT / "configs/h1_bfm_coverage.yaml").read_text(encoding="utf-8")
    )
    targets, _ = load_expert_targets(
        ROOT / config["expert_data"]["root"],
        ROOT / "husky_sim/upstream/test_scene/mjlab_scene.xml",
        config["expert_data"],
    )
    model = Bfm0Model()
    runner = H1RolloutRunner(model, _config(), device="cpu")
    latent = torch.ones(model.config.latent_dim)
    try:
        for target in targets:
            if target.kind != "human_push_window":
                continue
            rollout = runner.rollout(
                latent,
                seed=0,
                initial_qpos=target.initial_qpos,
                initial_qvel=target.initial_qvel,
            )
            feet = rollout.states[0]["body_position_board"][[6, 12]]
            left_foot, right_foot = feet
            assert abs(right_foot[0]) < 0.4
            assert abs(right_foot[1]) < 0.1
            assert right_foot[2] == pytest.approx(0.04755, abs=0.01)
            assert abs(left_foot[1]) > 0.1
            if target.name.endswith("window_00"):
                assert -0.08 < left_foot[2] < 0.0
    finally:
        runner.close()


def test_slerp_endpoints_and_norm() -> None:
    model = Bfm0Model()
    start = model.project_z(torch.randn(model.config.latent_dim))
    end = model.project_z(torch.randn(model.config.latent_dim))
    points = spherical_lerp(model, start, end, torch.tensor([0.0, 0.5, 1.0]))
    expected_end = end
    if torch.dot(start, end) < 0:
        expected_end = -end
    assert torch.allclose(points[0], start, atol=1e-5)
    assert torch.allclose(points[-1], expected_end, atol=1e-5)
    assert torch.allclose(
        torch.linalg.vector_norm(points, dim=-1),
        torch.full((3,), math.sqrt(model.config.latent_dim)),
        atol=1e-5,
    )


def test_quaternion_rotation_error_handles_sign() -> None:
    quaternion = np.array([0.5, 0.5, 0.5, 0.5])
    assert quaternion_rotation_error(quaternion, quaternion) == pytest.approx(0.0)
    assert quaternion_rotation_error(quaternion, -quaternion) == pytest.approx(0.0)


def test_video_names_use_behavior_content() -> None:
    push = ExpertTarget("push_start_pose", "static_pose", np.empty((0, 7)), "test")
    steer = ExpertTarget("steer_start_pose", "static_pose", np.empty((0, 7)), "test")
    motion = ExpertTarget(
        "human_push_1_window_02",
        "human_push_window",
        np.empty((0, 23)),
        "test",
    )
    assert _video_target_label(push) == "push_pose"
    assert _video_target_label(steer) == "steer_pose"
    assert _video_target_label(motion) == "human_push_1_window_02"
    assert _slerp_video_label(0, 11) == "push_steer_push_anchor"
    assert _slerp_video_label(5, 11) == "push_steer_midpoint_blend"
    assert _slerp_video_label(10, 11) == "push_steer_steer_anchor"


def test_static_pose_score_is_finite() -> None:
    model = Bfm0Model()
    runner = H1RolloutRunner(model, _config(), device="cpu")
    try:
        rollout = runner.rollout(torch.ones(model.config.latent_dim), seed=0)
    finally:
        runner.close()
    first_state = rollout.states[0]
    target = ExpertTarget(
        name="pose",
        kind="static_pose",
        values=np.concatenate(
            (
                first_state["body_position_board"],
                first_state["body_quaternion_board"],
            ),
            axis=1,
        ),
        source="test",
    )
    result = score_rollout(rollout, target, _config()["scores"])
    assert math.isfinite(result.score)
    assert all(math.isfinite(value) for value in result.metrics.values())
    assert result.score == pytest.approx(-result.metrics["mean_pose_error"])


def test_steer_initial_pose_places_both_feet_on_board() -> None:
    model = Bfm0Model()
    runner = H1RolloutRunner(model, _config(), device="cpu")
    target = np.load(
        ROOT / "husky_sim/upstream/dataset/ref_pose/steer_start_pose_b.npy",
        allow_pickle=False,
    )
    try:
        rollout = runner.rollout(
            torch.ones(model.config.latent_dim),
            seed=0,
            initial_qpos=EXPERT_STATIC_QPOS["steer_start_pose"],
        )
    finally:
        runner.close()
    initial = rollout.states[0]
    position_error = np.linalg.norm(
        initial["body_position_board"] - target[:, :3],
        axis=1,
    )
    assert position_error.mean() < 0.01
    assert position_error.max() < 0.02
    assert initial["body_position_board"][6, 2] > 0.04
    assert initial["body_position_board"][12, 2] > 0.04


def test_rollout_resets_env_and_observation_history() -> None:
    model = Bfm0Model()
    runner = H1RolloutRunner(model, _config(), device="cpu")
    latent = torch.ones(model.config.latent_dim)
    try:
        first = runner.rollout(latent, seed=5)
        second = runner.rollout(latent, seed=5)
    finally:
        runner.close()
    assert np.array_equal(first.actions, second.actions)
    assert np.array_equal(
        first.states[0]["root_position"],
        second.states[0]["root_position"],
    )
    assert np.array_equal(
        first.states[-1]["joint_position"],
        second.states[-1]["joint_position"],
    )


def test_rollout_executes_time_aligned_latent_trajectory() -> None:
    model = Bfm0Model()
    runner = H1RolloutRunner(model, _config(), device="cpu")
    latent_trajectory = model.project_z(torch.randn(3, model.config.latent_dim))
    try:
        rollout = runner.rollout(latent_trajectory, seed=7)
    finally:
        runner.close()
    assert rollout.latent.shape == (3, model.config.latent_dim)
    assert rollout.actions.shape[0] == 3
    assert len(rollout.states) == 4


def test_rollout_applies_expert_initial_velocity() -> None:
    model = Bfm0Model()
    runner = H1RolloutRunner(model, _config(), device="cpu")
    initial_qpos = runner.env.data.qpos[
        runner.env.model.joint("robot/floating_base_joint").qposadr[0] :
    ][:36].copy()
    initial_qvel = np.linspace(-0.2, 0.2, 35)
    try:
        rollout = runner.rollout(
            torch.ones(model.config.latent_dim),
            seed=3,
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
        )
    finally:
        runner.close()
    assert np.allclose(
        rollout.states[0]["joint_velocity"],
        initial_qvel[6:][[BFM0_JOINTS.index(name) for name in HUSKY_JOINTS]],
    )


def test_temporary_checkpoint_two_latent_two_step_smoke(tmp_path: Path) -> None:
    checkpoint = _temporary_checkpoint(tmp_path / "temporary.pt")
    model, report = load_bfm0_checkpoint(
        checkpoint,
        device="cpu",
        run_type="smoke",
    )
    runner = H1RolloutRunner(model, _config(), device="cpu")
    latents = sample_global_latents(model, 2, seed=0)
    try:
        results = [runner.rollout(latent, seed=index) for index, latent in enumerate(latents)]
    finally:
        runner.close()
    assert report["loaded_strictly"]
    assert len(results) == 2
    assert all(len(result.actions) == 2 for result in results)


def test_formal_run_rejects_compact_temporary_checkpoint(tmp_path: Path) -> None:
    checkpoint = _temporary_checkpoint(tmp_path / "temporary.pt")
    with pytest.raises(CheckpointCompatibilityError) as error:
        load_bfm0_checkpoint(checkpoint, device="cpu", run_type="formal")
    assert "not a verified official pretrained BFM0" in str(error.value)
    assert error.value.report["formal_eligible"] is False


def test_smoke_command_does_not_update_formal_results(tmp_path: Path) -> None:
    checkpoint = _temporary_checkpoint(tmp_path / "temporary.pt")
    config = yaml.safe_load((ROOT / "configs/h1_bfm_coverage.yaml").read_text(encoding="utf-8"))
    output_root = tmp_path / "results"
    config["experiment"]["output_root"] = str(output_root)
    config["checkpoint"]["device"] = "cpu"
    config["visualization"]["save_video"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    formal_results = docs_root / "exp_res.md"
    formal_results.write_text("# Experiment Results\n", encoding="utf-8")
    experiment_log = docs_root / "exp_logs.md"
    experiment_log.write_text("# Experiment Log\n", encoding="utf-8")
    before_results = formal_results.read_bytes()
    before_log = experiment_log.read_bytes()
    experiment_name = "h1_pytest_smoke"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            ("from skate_bfm.exp.h1_bfm_coverage import main; raise SystemExit(main())"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--run-type",
            "smoke",
            "--experiment-name",
            experiment_name,
            "--device",
            "cpu",
            "--no-save-video",
        ],
        cwd=ROOT,
        env=os.environ | {"SKATE_BFM_H1_DOCS_ROOT": str(docs_root)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert formal_results.read_bytes() == before_results
    assert experiment_log.read_bytes() == before_log
    assert not output_root.exists()
    assert not list(tmp_path.glob("skate_bfm_h1_smoke_*"))
