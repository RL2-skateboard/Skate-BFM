"""Native29 evaluation schema, recording identity and replay regressions."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def evaluator_module():
    name = "native29_evaluator_test"
    if name not in sys.modules:
        path = Path(__file__).parents[1] / "train/scripts/evaluator.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_protocol_rejects_historical_schema(tmp_path):
    module = evaluator_module()
    assert module.load_protocol(module.PROTOCOL)["source_joint_support"] == 29
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"evaluator_version": "skate-bfm-fixed-eval-v1"}))
    with pytest.raises(ValueError, match="legacy raw23"):
        module.load_protocol(old)
    with pytest.raises(SystemExit):
        module.parse_args(["viewer"])


def test_context_and_checkpoint_reject_wrong_contract_before_start(tmp_path):
    module = evaluator_module()
    context = tmp_path / "old_context.npy"
    np.save(context, np.ones(23))
    with pytest.raises(ValueError, match="256"):
        module.load_context(context)
    with pytest.raises(ValueError, match="ONNX/raw23"):
        module.validate_checkpoint(tmp_path / "old_teacher.onnx")
    (tmp_path / "model.safetensors").write_bytes(b"not loaded")
    (tmp_path / "init_kwargs.json").write_text(json.dumps({"action_dim": 23}))
    with pytest.raises(ValueError, match="non-29"):
        module.validate_checkpoint(tmp_path)


def test_recorder_public_500_tick_dynamics_and_visual_gate(tmp_path):
    module = evaluator_module()
    env = module.SkateSimOnlineEnv(backend="mujoco", num_envs=1, seed=4728)
    recorder = None
    try:
        recorder = module.attach_recording(env, tmp_path, [], 4728, mode="external_control_replay_gate")
        # Exercise physics replay independently of policy quality or fall detection.
        for index in range(50):
            action = .02 * np.sin(np.arange(29) + index / 10)
            env.env.backend.control_step(action)
        assert recorder.tick_index == 500
    finally:
        if recorder:
            recorder.close()
        env.close()
    rows, meta = module.native_record(tmp_path)
    assert meta["source_joint_support"] == 29
    assert len([row for row in rows if row["event"] == "tick"]) == 500
    report = module.replay(str(tmp_path), mode="dynamics", headless=True,
                           atol=1e-7, horizon_seconds=1.0)
    assert report["status"] == "PASS"
    assert report["model_identity"] == "VERIFIED"
    assert report["compared_ticks"] == 500
    assert report["max_abs_error"] <= 1e-7
    visual = module.replay(str(tmp_path), mode="visual", headless=True)
    assert visual["status"] == "PASS"
    assert visual["verification"] == "stored_state_readback"
    assert visual["max_abs_error"] == 0


def test_interactive_replay_delegates_to_only_simulator_entry(tmp_path, monkeypatch):
    module = evaluator_module()
    (tmp_path / "events.jsonl").write_text(json.dumps({
        "event": "meta", "format_version": 2, "source_joint_support": 29,
    }) + "\n")
    monkeypatch.setattr(module, "native_record", lambda path, **kwargs: ([], {}))
    observed = []

    def run(command, check):
        observed.extend(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(module.subprocess, "run", run)
    args = module.parse_args(["replay", "--record", str(tmp_path)])
    report = module.run_replay(args)
    assert report["status"] == "PASS"
    assert observed[1] == str(module.SIM_ROOT / "play.py")
    assert "--replay" in observed and "visual" in observed


@pytest.mark.parametrize("damage", ["missing", "index", "order", "action"])
def test_dynamics_rejects_missing_or_unbound_context_rows(tmp_path, damage):
    module = evaluator_module()
    rows = [
        dict(event="meta", format_version=2, source_joint_support=29, num_envs=1),
        dict(event="initial", backend_kwargs=dict(scene_config=dict(robot=True, board=True))),
        dict(event="asset_model", files=[dict(sha256="unit-test-binding-only")]),
        dict(event="mode", controller="external_bfm29"),
    ]
    for index in range(2):
        rows.append(dict(event="tick", tick=index, action=[.25 * (index + 1)] * 29,
                         command=dict(external_action_update=True)))
        rows.append(dict(event="control", control_index=index,
                         action=[[.25 * (index + 1)] * 29], z=[[float(index)] * 256]))
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    module.native_record(tmp_path, require_control_context=True)
    if damage == "missing":
        rows.pop()
    elif damage == "index":
        rows[-1]["control_index"] = 0
    elif damage == "order":
        rows[-1], rows[-2] = rows[-2], rows[-1]
    else:
        rows[-1]["action"][0][0] = .75
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        module.native_record(tmp_path, require_control_context=True)
