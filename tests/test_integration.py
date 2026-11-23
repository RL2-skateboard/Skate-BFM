"""Native online package and pure historical-source conversions."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

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




def test_public_import_has_no_retired_runtime_or_projection():
    import skate_bfm
    import skate_bfm.integration as integration
    assert skate_bfm.SkateSimOnlineEnv is integration.SkateSimOnlineEnv
    assert "skate_husky" not in sys.modules
    assert not hasattr(integration,"HuskyBfmOnlineEnv")
    from skate_bfm.integration import actions
    assert not hasattr(actions,"install_husky_action_projection")
    assert len(actions.BFM0_JOINTS)==29


def test_historical_source_mapping_is_offline_and_labeled():
    from skate_bfm.integration.source import (
        translate_skate_action, SOURCE_METADATA, SKATE_SOURCE_JOINTS,
        BFM0_INACTIVE_ACTION_INDICES,
    )
    result=translate_skate_action(np.zeros((2,23),dtype=np.float32))
    assert len(SKATE_SOURCE_JOINTS)==23
    assert result.bridge_29.shape==(2,29)
    assert np.all(result.bridge_29[:,BFM0_INACTIVE_ACTION_INDICES]==0)
    assert SOURCE_METADATA["source_joint_support"]==23
    assert not SOURCE_METADATA["native29_expert"]
    assert len(SOURCE_METADATA["missing_joint_names"])==6
    with pytest.raises(ValueError,match="23D"):
        translate_skate_action(np.zeros((2,29),dtype=np.float32))
