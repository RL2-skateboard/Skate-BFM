"""Native29-only training boundary regressions."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def training_module():
    name = "train_skate_bfm_native_test"
    if name not in sys.modules:
        path = Path(__file__).parents[1] / "train/scripts/train_skate_bfm.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_backend_defaults_and_aligned_tracking_rejection():
    module = training_module()
    default = module.build_train_config()
    assert default.online_env_backend == "skate-sim-mjlab"
    assert not default.agent.train.rollout_expert_trajectories
    assert default.agent.train.rollout_expert_trajectories_percentage==0
    with pytest.raises(ValueError,match="legacy23 is retired"):
        module.build_train_config(SimpleNamespace(online_env_backend="legacy"))
    native = module.build_train_config(SimpleNamespace(
        online_env_backend="skate-sim-cpu", online_envs=1, device="cpu",
    ))
    assert not native.agent.train.rollout_expert_trajectories
    assert native.agent.train.rollout_expert_trajectories_percentage == 0.0
    assert native.agent.model.device == "cpu"
    with pytest.raises(ValueError, match="cannot reproduce aligned"):
        module.build_train_config(SimpleNamespace(
            online_env_backend="skate-sim-cpu", online_envs=1,
            aligned_online_tracking=True,
        ))


def test_native_batch_transition_uses_full_action_and_terminal_observation():
    module = training_module()
    cfg = module.build_train_config(SimpleNamespace(
        online_env_backend="skate-sim-cpu", online_envs=1, device="cpu",
    ))
    env = module.make_native_online_env(cfg)
    replay = module.DictBuffer(capacity=8, device="cpu")
    try:
        initial = env.reset()
        action = torch.linspace(-0.15, 0.15, 29).reshape(1, 29)
        z = torch.ones(1, 256)
        before_qpos = torch.as_tensor(env.env.backend.data.qpos).clone()
        before_steps = env.env.steps.clone()
        for invalid_timeout in ([1], [True, False], [[True, False]]):
            with pytest.raises(ValueError, match="timeout mask"):
                env.step(action, z, truncated=invalid_timeout)
            assert torch.equal(before_qpos, torch.as_tensor(env.env.backend.data.qpos))
            assert torch.equal(before_steps, env.env.steps)
        transition = env.step(action, z, truncated=[True])
        replay.extend(transition)
        saved_terminal = {
            key: value.clone() for key, value in transition["next"]["observation"].items()
        }
        env.reset(env_ids=[0])
        for key in saved_terminal:
            assert torch.equal(saved_terminal[key], transition["next"]["observation"][key])
        batch = replay.sample(4)
        assert torch.equal(batch["action"], action.expand(4, -1))
        assert torch.equal(batch["z"], z.expand(4, -1))
        assert batch["next"]["truncated"].all()
        assert not batch["next"]["terminated"].any()
        assert tuple(batch["aux_rewards"]) == module.AUX_REWARD_KEYS
        for key, dim in (
            ("state", 64), ("privileged_state", 463),
            ("last_action", 29), ("history_actor", 372),
        ):
            assert initial[key].shape == (1, dim)
            assert batch["next"]["observation"][key].shape == (4, dim)
        assert not env.metadata["tracking_aligned"]
        assert not env.metadata["source_physics_aligned"]
        with pytest.raises(TypeError):
            env.reset(qpos=torch.zeros(36), qvel=torch.zeros(35))
    finally:
        env.close()


def test_checkpoint_device_relocation_retains_strict_architecture(tmp_path):
    import json

    import numpy as np
    import safetensors.torch

    module = training_module()
    from humanoidverse.agents.envs.utils.gym_spaces import space_to_json

    obs_space = module.gymnasium.spaces.Dict({
        "state": module.gymnasium.spaces.Box(-1, 1, (2,), np.float32),
    })
    source = torch.nn.Linear(2, 3)
    safetensors.torch.save_model(source, str(tmp_path / "model.safetensors"))
    config = {"device": "cuda", "architecture": {"input": 2, "output": 3}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "init_kwargs.json").write_text(json.dumps({
        "action_dim": 3, "obs_space": space_to_json(obs_space),
    }))
    target = torch.nn.Linear(2, 3)
    target.cfg = SimpleNamespace(model_dump=lambda: {
        **config, "device": "cpu",
    })
    agent = SimpleNamespace(_model=target, action_dim=3, obs_space=obs_space, device="cpu")
    report = module.load_bfm_checkpoint(agent, tmp_path)
    assert report["checkpoint_device"] == "cuda"
    assert report["runtime_device"] == "cpu"
    assert torch.equal(source.weight, target.weight)
    target.cfg = SimpleNamespace(model_dump=lambda: {
        **config, "device": "cpu", "architecture": {"input": 99, "output": 3},
    })
    with pytest.raises(RuntimeError, match="configuration differs"):
        module.load_bfm_checkpoint(agent, tmp_path)


def test_native_backend_closes_if_initial_reset_fails(tmp_path, monkeypatch):
    module = training_module()
    cfg = module.build_train_config(SimpleNamespace(
        online_env_backend="skate-sim-cpu", online_envs=1, device="cpu",
        work_dir=tmp_path / "failed_reset",
    ))

    class FailingReset:
        closed = False

        def reset(self):
            raise RuntimeError("injected initial reset failure")

        def close(self):
            self.closed = True

    env = FailingReset()
    monkeypatch.setattr(module, "make_native_online_env", lambda cfg: env)
    with pytest.raises(RuntimeError, match="injected initial reset failure"):
        cfg.build()
    assert env.closed


@pytest.mark.parametrize("dataset", ["phase", "continuous"])
def test_real_collection_provenance_reaches_workspace_boundary(dataset, tmp_path, monkeypatch):
    import json
    import joblib
    root=Path('/home/hm/workspace/skate-sim/runs/native29_20261104/collection_verified')
    manifest=root/dataset/'motion_library/manifest.json'
    if not manifest.is_file():
        pytest.skip('Local real collection acceptance artifacts are unavailable')
    metadata=json.loads(manifest.read_text())
    motion=Path(metadata['motion_library'])
    module=training_module()
    report=module.load_dataset_report(motion,manifest,dataset)
    records=joblib.load(motion)
    assert report['source_joint_support']==[29]
    assert report['missing_joint_names']==[]
    assert report['native29_expert'] is False
    assert report['source_quality']==metadata['source_quality']
    assert report['quality_review']==metadata['quality_review']
    assert set(report['record_provenance'])==set(records)
    for name,record in records.items():
        for key in ('source_joint_support','missing_joint_names','source_quality','native29_expert'):
            assert report['record_provenance'][name][key]==record[key]
    reached=[]
    def stop_before_backend(cfg):
        reached.append(True)
        raise RuntimeError('validated real dataset before backend')
    monkeypatch.setattr(module,'make_native_online_env',stop_before_backend)
    cfg=module.build_train_config(SimpleNamespace(
        expert_motion=motion, online_env_backend='skate-sim-cpu',online_envs=1,device='cpu',
        work_dir=tmp_path/'boundary',
    ))
    with pytest.raises(RuntimeError,match='validated real dataset'):
        cfg.build()
    assert reached


def test_dataset_provenance_scalar_legacy_mixed_and_invalid(tmp_path):
    import json
    import joblib
    from skate_bfm.integration.source import SOURCE_MISSING_JOINTS
    module=training_module()
    motion=tmp_path/'data.pkl'; manifest=tmp_path/'manifest.json'
    def load(metadata,records):
        manifest.write_text(json.dumps(metadata));joblib.dump(records,motion)
        return module.load_dataset_report(motion,manifest,'test')
    old=load({}, {'old':{}})
    assert old['source_joint_support']==[23]
    assert set(old['missing_joint_names'])==set(SOURCE_MISSING_JOINTS)
    assert not old['native29_expert']
    scalar=load({'source_joint_support':29}, {'new':{}})
    assert scalar['source_joint_support']==[29]
    assert not scalar['native29_expert']
    missing=list(SOURCE_MISSING_JOINTS)
    metadata={'source_joint_support':[23,29],'missing_joint_names':missing,
              'source_quality':['historical','policy_rollout_unvalidated'],'native29_expert':False}
    records={'old':{'source_joint_support':23,'missing_joint_names':missing,'source_quality':'historical','native29_expert':False},
             'new':{'source_joint_support':29,'missing_joint_names':[],'source_quality':'policy_rollout_unvalidated','native29_expert':False}}
    mixed=load(metadata,records)
    assert mixed['source_joint_support']==[23,29]
    assert set(mixed['missing_joint_names'])==set(missing)
    assert not mixed['native29_expert']
    assert mixed['record_provenance']['old']['missing_joint_names']
    for invalid in ([],[29,29],[True],True,'29',[23,30]):
        with pytest.raises(ValueError,match='source_joint_support'):
            load({'source_joint_support':invalid},{'x':{}})
    with pytest.raises(ValueError,match='missing_joint_names'):
        load({'source_joint_support':[23,29],'missing_joint_names':[]},records)
    with pytest.raises(ValueError,match='missing_joint_names'):
        load({'source_joint_support':29,'missing_joint_names':missing},{'x':{}})
    with pytest.raises(ValueError,match='disagrees with records'):
        load({'source_joint_support':[23,29]}, {'x':records['new']})
    # An explicit false on either side is never upgraded by a 29D dimension.
    report=load({'source_joint_support':29,'native29_expert':True}, {'x':{'native29_expert':False}})
    assert report['native29_expert'] is False
