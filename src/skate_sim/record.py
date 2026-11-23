from __future__ import annotations

import hashlib
import json
import platform
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def checkpoint_manifest(path: str | None, *, extra_paths: list[str] | None = None) -> list[dict]:
    """Hash actual checkpoint inputs, without copying model weights into logs."""
    root = Path(path).resolve() if path else None
    files = [] if root is None else ([p for p in (root / 'model/exported/FBcprAuxModel.onnx', root / 'config.yaml', root / 'cfg.yaml') if p.is_file()] if root.is_dir() else [root])
    files += [Path(p).resolve() for p in (extra_paths or [])]
    return [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in files]


def recording_inputs(checkpoint: str | None, policy_task: str = "reward") -> list[dict]:
    """Hash the same actor, selected context file and configuration in every viewer."""
    extra = [Path(__file__).parents[2] / 'configs/bfm' / name
             for name in ('motivo_newG1.yaml', 'g1.yaml')]
    if checkpoint:
        root = Path(checkpoint)
        if root.is_dir():
            latent = {'reward': 'reward_inference/reward_locomotion.pkl',
                      'tracking': 'tracking_inference/zs_walking.pkl',
                      'goal': 'goal_inference/goal_reaching.pkl'}[policy_task]
            extra.append(root / "model" / latent)
    return checkpoint_manifest(checkpoint, extra_paths=[str(p) for p in extra if p.is_file()])


def numerics_manifest(backend) -> dict | None:
    """Identify the actual integration implementation independently of the asset XML."""
    describe = getattr(backend, 'numerics_manifest', None)
    if callable(describe):
        return describe()
    if getattr(backend, 'name', None) == 'mujoco':
        import mujoco
        return dict(strategy='native_mujoco_v1', mujoco=mujoco.__version__, dtype='float64')
    return None


def compiled_model_xml(backend) -> str:
    """Export the full compiled model, including MjSpec-based GPU backends."""
    import mujoco
    model = backend.model if hasattr(backend, 'model') else backend.sim.mj_model
    if not hasattr(backend, 'model'):
        # MjSpec.compile() does not populate MuJoCo's legacy last-XML store.
        # Load the same spec before copying the actual compiled model values.
        mujoco.MjModel.from_xml_string(backend.spec.to_xml())
    with tempfile.TemporaryDirectory(prefix='skate-model-hash-') as directory:
        path = Path(directory) / "model.xml"
        mujoco.mj_saveLastXML(str(path), model)
        return path.read_text(encoding="utf-8")


def compiled_model_hash(backend) -> str:
    """Fingerprint full compiled XML; retain strict legacy hash semantics."""
    return hashlib.sha256(compiled_model_xml(backend).encode()).hexdigest()


class Recorder:
    """Versioned JSONL; one tick row for each actual physics integration."""

    def __init__(self, path: str | None, *, backend: str, scene: str, seed: int,
                 metadata: dict | None = None):
        self.path = Path(path) if path else None
        self.file = None
        self.tick_index = 0
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            self.file = (self.path / "events.jsonl").open("x", encoding="utf-8")
            self.write("meta", format_version=2, backend=backend, scene=scene, seed=seed,
                       wall_time=time.time(), platform=platform.platform(), **(metadata or {}))

    def write(self, event: str, **data: Any) -> None:
        if self.file:
            self.file.write(json.dumps({"event": event, "wall_time": time.time(), **_jsonable(data)}) + "\n")
            self.file.flush()

    def start(self, initial_state: dict, *, backend_kwargs: dict,
              runtime_state: dict | None = None) -> None:
        self.write("initial", state=initial_state, backend_kwargs=backend_kwargs,
                   runtime_state=runtime_state)

    def tick(self, state: dict, *, action: Any = None, action_source: str = "controller",
             command: Any = None, inputs: list | None = None,
             runtime_state: dict | None = None) -> None:
        self.write("tick", tick=self.tick_index, state=state, action=action,
                   action_source=action_source, command=command, inputs=inputs or [],
                   runtime_state=runtime_state)
        self.tick_index += 1

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None
