#!/usr/bin/env python3
"""Shared Skate-BFM runtime and checkpoint integrity helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ISAAC_ENV_ROOT = Path(
    os.environ.get("SKATE_BFM_ISAAC_ROOT", SCRIPT_DIRECTORY / "isaac_env")
).expanduser().resolve()
sys.path[:0] = [
    str(REPOSITORY_ROOT / "src"),
    str(REPOSITORY_ROOT / "husky_sim" / "src"),
    str(ISAAC_ENV_ROOT),
]

import gymnasium
import mujoco
import numpy as np
import safetensors.torch
import torch
from safetensors import safe_open

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HYDRA_CONFIG_DIR,
    HumanoidVerseIsaacConfig,
)
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from skate_bfm.integration import HuskyBfmOnlineEnv

DATASET_ROOT = REPOSITORY_ROOT / "train/dataset"
RAW_DATASET_ROOT = DATASET_ROOT / "sim_collected/train/raw"
OFFICIAL_BFM0_SHA256 = (
    "33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361"
)


def resolve_source_rollout_path(record: Mapping[str, Any]) -> Path:
    """Resolve raw provenance after moving or downloading a MotionLib."""

    def validate(path: Path) -> Path:
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Canonical raw metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        expected = {
            "dataset_split": "train",
            "round_id": str(record.get("source_round", "")).zfill(3),
            "rollout_id": str(record.get("source_rollout", "")).zfill(3),
            "episode_id": str(record.get("source_episode", "")),
        }
        actual = {name: str(metadata.get(name, "")) for name in expected}
        if actual != expected:
            raise RuntimeError(
                f"Train raw provenance mismatch for {path}: {actual} != {expected}"
            )
        return path.resolve()

    recorded = Path(str(record["source_raw_npz"])).expanduser()
    if recorded.is_file():
        return validate(recorded)

    parts = recorded.parts
    if "raw" in parts:
        relocated = RAW_DATASET_ROOT.joinpath(*parts[parts.index("raw") + 1 :])
        if relocated.is_file():
            return validate(relocated)

    round_id = str(record.get("source_round", "")).zfill(3)
    rollout_id = str(record.get("source_rollout", "")).zfill(3)
    rollout_root = (
        RAW_DATASET_ROOT
        / f"round_{round_id}"
        / f"rollout_{rollout_id}"
        / "raw_rollout"
    )
    matches = sorted(rollout_root.glob("*.npz"))
    if len(matches) == 1:
        return validate(matches[0])
    raise FileNotFoundError(
        "Cannot uniquely resolve source rollout "
        f"{recorded} under {RAW_DATASET_ROOT}: {len(matches)} candidates"
    )


def validate_raw_layout(
    metadata_path: Path,
    raw_qpos: np.ndarray,
    raw_qvel: np.ndarray,
    env: HuskyBfmOnlineEnv,
) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Canonical raw metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    required = (
        "nq",
        "nv",
        "joint_order",
        "board_joint_order",
        "qpos_quaternion_order",
        "robot_xml",
        "fields",
    )
    if any(field not in metadata for field in required):
        raise RuntimeError(f"Canonical raw layout metadata is incomplete: {metadata_path}")
    model = env.env.model
    if raw_qpos.ndim != 2 or raw_qvel.ndim != 2:
        raise RuntimeError("Canonical raw qpos/qvel must be rank-2 arrays.")
    fields = metadata["fields"]
    if (
        int(metadata["nq"]) != model.nq
        or int(metadata["nv"]) != model.nv
        or raw_qpos.shape[1] != model.nq
        or raw_qvel.shape[1] != model.nv
        or fields.get("qpos", {}).get("shape") != list(raw_qpos.shape)
        or fields.get("qvel", {}).get("shape") != list(raw_qvel.shape)
        or fields.get("qpos", {}).get("dtype") != str(raw_qpos.dtype)
        or fields.get("qvel", {}).get("dtype") != str(raw_qvel.dtype)
        or metadata["qpos_quaternion_order"] != "wxyz"
    ):
        raise RuntimeError(f"Canonical raw qpos/qvel layout mismatch: {metadata_path}")
    robot_joint_order = tuple(
        model.joint(index).name
        for index in range(model.njnt)
        if (model.joint(index).name or "").startswith("robot/")
        and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
    )
    board_joint_order = tuple(
        model.joint(index).name
        for index in range(model.njnt)
        if (model.joint(index).name or "").startswith("skateboard/")
        and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
    )
    free_joint_order = tuple(
        model.joint(index).name
        for index in range(model.njnt)
        if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
    )
    actuator_order = tuple(
        model.actuator(index).name
        for index in range(model.nu)
        if (model.actuator(index).name or "").startswith("robot/")
    )
    if (
        tuple(metadata["joint_order"]) != robot_joint_order
        or tuple(metadata["board_joint_order"]) != board_joint_order
        or len(actuator_order) != len(robot_joint_order)
        or set(actuator_order) != set(robot_joint_order)
        or free_joint_order
        != (
            "robot/floating_base_joint",
            "skateboard/floating_base_joint_skateboard",
        )
    ):
        raise RuntimeError(f"Canonical raw joint order mismatch: {metadata_path}")
    source_xml = Path(metadata["robot_xml"]).expanduser()
    current_xml = env.env.xml_path.expanduser().resolve()
    if not source_xml.is_file() or not source_xml.resolve().samefile(current_xml):
        raise RuntimeError(
            f"Canonical raw source XML differs from current HUSKY XML: {metadata_path}"
        )
    return metadata


def load_source_rollout(
    source_path: Path,
    env: HuskyBfmOnlineEnv,
    expected_physics_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load one canonical raw source and validate its recorded physics."""

    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Canonical raw rollout not found: {source_path}")
    with np.load(source_path, allow_pickle=False) as archive:
        if "qpos" not in archive or "qvel" not in archive:
            raise RuntimeError(f"Canonical raw rollout lacks qpos/qvel: {source_path}")
        raw_qpos = np.asarray(archive["qpos"]).copy()
        raw_qvel = np.asarray(archive["qvel"]).copy()
    metadata = validate_raw_layout(
        source_path.with_suffix(".json"),
        raw_qpos,
        raw_qvel,
        env,
    )
    source_physics = metadata.get("physics_randomization")
    if not isinstance(source_physics, dict):
        raise RuntimeError(f"Canonical raw source lacks physics_randomization: {source_path}")
    normalized = env.env.validate_source_physics(source_physics)
    if normalized["seed"] != int(expected_physics_seed):
        raise RuntimeError(f"Canonical raw physics seed mismatch: {source_path}")
    return raw_qpos, raw_qvel, copy.deepcopy(source_physics)


def checkpoint_model_path(checkpoint_dir: Path) -> Path:
    for path in (
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "model" / "model.safetensors",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No model.safetensors in {checkpoint_dir}.")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_tensors(tensors: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def hash_params(module: torch.nn.Module) -> str:
    return _hash_tensors(module.named_parameters())


def hash_buffers(module: torch.nn.Module) -> str:
    return _hash_tensors(module.named_buffers())


def hash_data(payload: Any) -> str:
    """Hash nested JSON data, NumPy arrays, and tensors."""

    digest = hashlib.sha256()

    def update(value: Any, path: str) -> None:
        digest.update(path.encode())
        if isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value):
                update(value[key], f"{path}/{key}")
            return
        if isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode())
            for index, item in enumerate(value):
                update(item, f"{path}/{index}")
            return
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            )
            return
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())

    update(payload, "root")
    return digest.hexdigest()


def hash_components(agent: Any) -> dict[str, str]:
    model = agent._model
    return {
        "F": hash_params(model._forward_map),
        "B": hash_params(model._backward_map),
        "Actor": hash_params(model._actor),
        "discriminator": hash_params(model._discriminator),
        "critic": hash_params(model._critic),
        "aux_critic": hash_params(model._aux_critic),
        "target_F": hash_params(model._target_forward_map),
        "target_B": hash_params(model._target_backward_map),
        "target_critic": hash_params(model._target_critic),
        "target_aux_critic": hash_params(model._target_aux_critic),
    }


def module_state_is_finite(module: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for value in (*module.parameters(), *module.buffers())
        if value.is_floating_point()
    )


def optimizer_step_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    steps: list[float] = []
    finite = True
    for state in optimizer.state.values():
        if "step" in state:
            steps.append(float(torch.as_tensor(state["step"]).item()))
        finite = finite and all(
            not torch.is_tensor(value)
            or not value.is_floating_point()
            or bool(torch.isfinite(value).all())
            for value in state.values()
        )
    return {
        "state_entries": len(optimizer.state),
        "step_values": sorted(set(steps)),
        "finite": finite,
    }


def _space_signature(space: gymnasium.spaces.Space) -> dict[str, tuple[int, ...]]:
    if not isinstance(space, gymnasium.spaces.Dict):
        raise TypeError(f"Expected Dict observation space, got {type(space).__name__}.")
    return {name: tuple(value.shape) for name, value in sorted(space.spaces.items())}


def load_bfm_checkpoint(agent: Any, checkpoint_dir: Path) -> dict[str, Any]:
    """Strictly load a complete BFM0 model without loading optimizer state."""

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    model_path = checkpoint_model_path(checkpoint_dir)
    config_path = model_path.parent / "config.json"
    init_kwargs_path = model_path.parent / "init_kwargs.json"
    if not config_path.is_file() or not init_kwargs_path.is_file():
        raise FileNotFoundError("BFM0 checkpoint metadata is incomplete.")
    checkpoint_config = json.loads(config_path.read_text())
    init_kwargs = json.loads(init_kwargs_path.read_text())
    if checkpoint_config != agent._model.cfg.model_dump():
        raise RuntimeError("Pretrained BFM0 model configuration differs from Skate-BFM.")
    if init_kwargs.get("action_dim") != agent.action_dim:
        raise RuntimeError("Pretrained BFM0 action dimension mismatch.")
    if _space_signature(json_to_space(init_kwargs["obs_space"])) != _space_signature(agent.obs_space):
        raise RuntimeError("Pretrained BFM0 observation-space mismatch.")

    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        checkpoint_shapes = {
            name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()
        }
    current_state = agent._model.state_dict()
    missing = set(current_state) - set(checkpoint_shapes)
    unexpected = set(checkpoint_shapes) - set(current_state)
    mismatched = [
        name
        for name in current_state
        if name in checkpoint_shapes
        and tuple(current_state[name].shape) != checkpoint_shapes[name]
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError("Strict pretrained BFM0 architecture validation failed.")
    safetensors.torch.load_model(
        agent._model,
        str(model_path),
        strict=True,
        device=agent.device,
    )
    return {
        "source": str(checkpoint_dir),
        "model_file": str(model_path),
        "model_sha256": hash_file(model_path),
        "optimizer_policy": "fresh optimizers; pretrained optimizer state is not loaded",
    }
