#!/usr/bin/env python3
# ruff: noqa: E402, E501, I001
"""Train the formal Skate-BFM M2.6 closed-loop baseline.

This is intentionally a small project-owned entrypoint. The vendored
``isaac_env/humanoidverse`` package retains the official BFM-Zero algorithms;
this file only connects their agent and MotionLib data to HUSKY MuJoCo.
"""

from __future__ import annotations

import copy
import hashlib
import json
import joblib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ISAAC_ENV_ROOT = Path(
    os.environ.get("SKATE_BFM_ISAAC_ROOT", SCRIPT_DIRECTORY / "isaac_env")
).expanduser().resolve()
if not (ISAAC_ENV_ROOT / "humanoidverse").is_dir():
    raise FileNotFoundError(f"Skate-BFM Isaac runtime not found: {ISAAC_ENV_ROOT}")
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
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.agents.envs.humanoidverse_isaac import (
    HYDRA_CONFIG_DIR,
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.utils import set_seed_everywhere
from skate_bfm.integration import HuskyBfmOnlineEnv
from skate_husky import AUX_REWARD_KEYS

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_float32_matmul_precision("high")

CHECKPOINT_DIR_NAME = "checkpoint"
OFFICIAL_BFM0_SHA256 = (
    "33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361"
)
SKATE_EPISODE_HORIZON = 1024
DEFAULT_MAX_STEPS = 100_000
DEFAULT_WARMUP_TRANSITIONS = 1024
DEFAULT_FIRST_UPDATE = 1500
DEFAULT_UPDATE_INTERVAL = 500
DEFAULT_UPDATES_PER_BLOCK = 50
EXPERT_DATASETS = {
    "phase": REPOSITORY_ROOT
    / "dataset/sim_collected/phase/motion_library/skate_expert_phase.pkl",
    "continuous": REPOSITORY_ROOT
    / "dataset/sim_collected/continuous/motion_library/skate_expert_continuous.pkl",
}


def training_update_steps(
    max_steps: int, first_update: int, update_interval: int
) -> tuple[int, ...]:
    return tuple(range(first_update, max_steps + 1, update_interval))


def training_checkpoint_steps(max_steps: int) -> tuple[int, ...]:
    return tuple(sorted({step for step in (20_000, 50_000, 100_000, max_steps) if step <= max_steps}))


def resolve_expert_dataset() -> tuple[str, Path, Path]:
    override = os.environ.get("SKATE_EXPERT_MOTION_FILE")
    if override:
        expert_path = Path(override).expanduser().resolve()
        dataset_kind = next(
            (
                kind
                for kind, path in EXPERT_DATASETS.items()
                if expert_path == path.resolve()
            ),
            "custom",
        )
    else:
        dataset_kind = os.environ.get("SKATE_EXPERT_DATASET", "phase").strip().lower()
        if dataset_kind not in EXPERT_DATASETS:
            raise ValueError("SKATE_EXPERT_DATASET must be 'phase' or 'continuous'.")
        expert_path = EXPERT_DATASETS[dataset_kind].resolve()
    return dataset_kind, expert_path, expert_path.with_name("manifest.json")


def validate_raw_layout(
    metadata_path: Path,
    raw_qpos: np.ndarray,
    raw_qvel: np.ndarray,
    env: HuskyBfmOnlineEnv,
) -> None:
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


class TrainConfig(BaseConfig):
    """Narrow configuration surface for formal Skate-BFM training."""

    agent: FBcprAuxAgentConfig
    env: HumanoidVerseIsaacConfig
    expert_dataset_kind: str
    skate_expert_motion_file: str
    expert_manifest_file: str
    pretrained_checkpoint: str
    work_dir: str
    seed: int = 4728
    skate_max_steps: int = DEFAULT_MAX_STEPS
    warmup_transitions: int = DEFAULT_WARMUP_TRANSITIONS
    first_update_transition: int = DEFAULT_FIRST_UPDATE
    update_interval: int = DEFAULT_UPDATE_INTERVAL
    updates_per_block: int = DEFAULT_UPDATES_PER_BLOCK
    online_envs: int = 4
    skate_expert_ratio: float = 0.5
    buffer_size: int = DEFAULT_MAX_STEPS
    buffer_device: str = "cpu"

    def model_post_init(self, context: Any) -> None:
        if self.skate_max_steps <= 0:
            raise ValueError("SKATE_MAX_STEPS must be positive.")
        if self.warmup_transitions <= 0:
            raise ValueError("SKATE_WARMUP_TRANSITIONS must be positive.")
        if self.first_update_transition < self.warmup_transitions:
            raise ValueError("SKATE_FIRST_UPDATE must be at least SKATE_WARMUP_TRANSITIONS.")
        if self.first_update_transition > self.skate_max_steps:
            raise ValueError("SKATE_FIRST_UPDATE must not exceed SKATE_MAX_STEPS.")
        if self.update_interval <= 0:
            raise ValueError("SKATE_UPDATE_INTERVAL must be positive.")
        if self.updates_per_block <= 0:
            raise ValueError("SKATE_UPDATES_PER_BLOCK must be positive.")
        if self.online_envs <= 0:
            raise ValueError("SKATE_ONLINE_ENVS must be positive.")
        if self.buffer_size < self.skate_max_steps:
            raise ValueError("SKATE_BUFFER_SIZE must be at least SKATE_MAX_STEPS.")
        if self.skate_expert_ratio != 0.5:
            raise ValueError("Formal Skate-BFM requires SKATE_EXPERT_RATIO=0.5.")
        if self.buffer_device != "cpu":
            raise ValueError("Formal Skate-BFM requires CPU DictBuffer replay.")
        boundaries = {
            self.warmup_transitions,
            self.first_update_transition,
            self.update_interval,
            *training_update_steps(
                self.skate_max_steps,
                self.first_update_transition,
                self.update_interval,
            ),
            *training_checkpoint_steps(self.skate_max_steps),
        }
        if any(boundary % self.online_envs for boundary in boundaries):
            raise ValueError(
                "SKATE_ONLINE_ENVS must divide every training schedule boundary."
            )
        for field_name, value in (
            ("Skate expert MotionLib", self.skate_expert_motion_file),
            ("Skate expert manifest", self.expert_manifest_file),
            ("official BFM0 checkpoint", self.pretrained_checkpoint),
        ):
            if not Path(value).expanduser().is_file() and not Path(value).expanduser().is_dir():
                raise FileNotFoundError(f"{field_name} not found: {value}")

    def build(self) -> Workspace:
        return Workspace(self)


class BaseSkateExpertSampler:
    """Sample complete 8-frame sequences equally from Base and Skate experts."""

    def __init__(self, expert_base: Any, expert_skate: Any) -> None:
        if expert_base.seq_length != expert_skate.seq_length:
            raise ValueError("Base and Skate expert sequence lengths must match.")
        self.expert_base = expert_base
        self.expert_skate = expert_skate
        self.seq_length = expert_base.seq_length

    def sample(self, batch_size: int = 1, seq_length: int | None = None) -> Any:
        sequence_length = seq_length or self.seq_length
        if batch_size % sequence_length:
            raise ValueError("Expert batch size must contain complete sequences.")
        sequences = batch_size // sequence_length
        if sequences % 2:
            raise ValueError("Formal Skate-BFM requires an even number of expert sequences.")
        half = sequences // 2 * sequence_length
        base = self.expert_base.sample(half, seq_length=sequence_length)
        skate = self.expert_skate.sample(half, seq_length=sequence_length)
        return tree_map(lambda left, right: torch.cat((left, right), dim=0), base, skate)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.expert_base, name)


def register_skate_replay(replay_buffer: dict[str, Any], train_skate: Any) -> None:
    """Keep the official BFM ``train`` key as an alias of HUSKY replay."""

    replay_buffer["train_skate"] = train_skate
    replay_buffer["train"] = train_skate


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
    """Hash nested JSON data, NumPy arrays, and tensors for evaluator provenance."""

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
    return {"state_entries": len(optimizer.state), "step_values": sorted(set(steps)), "finite": finite}


def _space_signature(space: gymnasium.spaces.Space) -> dict[str, tuple[int, ...]]:
    if not isinstance(space, gymnasium.spaces.Dict):
        raise TypeError(f"Expected Dict observation space, got {type(space).__name__}.")
    return {name: tuple(value.shape) for name, value in sorted(space.spaces.items())}


def load_bfm_checkpoint(agent: Any, checkpoint_dir: Path) -> dict[str, Any]:
    """Strictly load a complete BFM0 model without loading old optimizer state."""

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
        name for name in current_state
        if name in checkpoint_shapes and tuple(current_state[name].shape) != checkpoint_shapes[name]
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError("Strict pretrained BFM0 architecture validation failed.")
    safetensors.torch.load_model(agent._model, str(model_path), strict=True, device=agent.device)
    return {
        "source": str(checkpoint_dir),
        "model_file": str(model_path),
        "model_sha256": hash_file(model_path),
        "optimizer_policy": "fresh optimizers; pretrained optimizer state is not loaded",
    }


def make_expert_env(env_cfg: HumanoidVerseIsaacConfig) -> SimpleNamespace:
    """Build only the upstream MotionLib context, without IsaacLab."""

    import hydra
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression))
    with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_DIR):
        cfg = hydra.compose(
            config_name=env_cfg.relative_config_path,
            overrides=env_cfg.hydra_overrides or [],
        )
    cfg.num_envs = 1
    cfg.exp_base = "__no_exp_base__"
    for field in ("asset_root",):
        cfg.robot.asset[field] = cfg.robot.asset[field].replace(
            "humanoidverse", str(ISAAC_ENV_ROOT / "humanoidverse")
        )
    cfg.robot.motion.asset.assetRoot = cfg.robot.motion.asset.assetRoot.replace(
        "humanoidverse", str(ISAAC_ENV_ROOT / "humanoidverse")
    )
    cfg.robot.motion.motion_file = env_cfg.lafan_tail_path
    cfg.obs.root_height_obs = env_cfg.root_height_obs
    pre_process_config(cfg)
    OmegaConf.set_struct(cfg, False)
    motion_lib = MotionLibRobot(cfg.robot.motion, num_envs=1, device=env_cfg.device)
    motion_lib.load_motions_for_training()
    default_dof_pos = torch.tensor(
        [cfg.robot.init_state.default_joint_angles[name] for name in cfg.robot.dof_names],
        device=env_cfg.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    return SimpleNamespace(
        _motion_lib=motion_lib,
        num_envs=1,
        dt=float(cfg.simulator.config.sim.control_decimation) / float(cfg.simulator.config.sim.fps),
        device=env_cfg.device,
        default_dof_pos=default_dof_pos,
        gravity_vec=torch.tensor([[0.0, 0.0, -1.0]], device=env_cfg.device),
        config=cfg.env.config,
    )


def load_expert(agent: Any, motion_file: str | Path) -> dict[str, torch.Tensor]:
    """Return BFM-compatible observations for one MotionLib source."""

    env = make_expert_env(build_train_config().env)
    motion_cfg = copy.deepcopy(env.config.robot.motion)
    motion_cfg.motion_file = str(Path(motion_file).resolve())
    motion_lib = type(env._motion_lib)(motion_cfg, num_envs=1, device=env.device)
    expert_env = SimpleNamespace(
        _motion_lib=motion_lib,
        num_envs=1,
        dt=env.dt,
        device=env.device,
        default_dof_pos=env.default_dof_pos,
        gravity_vec=env.gravity_vec,
        config=env.config,
    )
    buffer = load_expert_trajectories_from_motion_lib(expert_env, agent.cfg, device=agent.device)
    return {
        name: value.detach().cpu()
        for name, value in buffer.storage["observation"].items()
        if name in {"state", "last_action", "privileged_state"}
    }


def encode_target(agent: Any, observations: dict[str, torch.Tensor]) -> np.ndarray:
    """Encode a raw target window using the checkpoint's B map and normalizer."""

    with torch.no_grad():
        normalized = agent._model._normalize(
            tree_map(lambda value: value.to(agent.device), observations)
        )
        z = agent._model.project_z(agent._model._backward_map(normalized).mean(dim=0, keepdim=True))[0]
    if not torch.isfinite(z).all():
        raise RuntimeError("Target latent contains NaN/Inf.")
    return z.detach().cpu().numpy().astype(np.float32)


def load_frozen_agent(checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    """Build the Skate-BFM architecture and freeze a validated checkpoint."""

    cfg = build_train_config()
    env = HuskyBfmOnlineEnv()
    try:
        observation = env.reset()
    finally:
        env.close()
    obs_space = gymnasium.spaces.Dict({
        key: gymnasium.spaces.Box(-np.inf, np.inf, tuple(value.shape), np.float32)
        for key, value in observation.items()
    })
    agent = cfg.agent.build(obs_space=obs_space, action_dim=29)
    report = load_bfm_checkpoint(agent, checkpoint)
    agent._model.eval()
    agent._model.requires_grad_(False)
    return agent, report


class Workspace:
    """Project-owned formal Skate-BFM training workspace."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir).expanduser().resolve()
        if (self.work_dir / "summary.json").exists() or (self.work_dir / CHECKPOINT_DIR_NAME).exists():
            raise RuntimeError("Formal Skate-BFM requires a fresh SKATE_WORK_DIR.")
        self.work_dir.mkdir(parents=True, exist_ok=False)
        set_seed_everywhere(cfg.seed)

        expert_path = Path(cfg.skate_expert_motion_file).expanduser().resolve()
        manifest_path = Path(cfg.expert_manifest_file).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        fields = {
            "dataset_stage": "dataset_stage",
            "dataset_type": "dataset_type",
            "motion_count": "total_motion_count",
            "expert_frames": "expert_frames",
            "expert_minutes": "expert_minutes",
            "fps": "fps",
        }
        self.dataset_report = {
            "kind": cfg.expert_dataset_kind,
            "motion_file": str(expert_path),
            "manifest_file": str(manifest_path),
            "motion_file_sha256": hash_file(expert_path),
            "manifest_sha256": hash_file(manifest_path),
            **{
                output_name: manifest[source_name]
                for output_name, source_name in fields.items()
                if source_name in manifest
            },
        }
        missing_fields = [
            source_name for source_name in fields.values() if source_name not in manifest
        ]
        if missing_fields:
            self.dataset_report["missing_manifest_fields"] = missing_fields
        self.reset_records = joblib.load(expert_path, mmap_mode="r")
        if not isinstance(self.reset_records, dict) or not self.reset_records:
            raise RuntimeError("Expert reset dataset must be a non-empty motion dictionary.")
        self.reset_motion_keys = tuple(self.reset_records)
        self.reset_rng = np.random.default_rng(cfg.seed)
        self.reset_raw_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

        self.train_envs = [HuskyBfmOnlineEnv() for _ in range(cfg.online_envs)]
        observation = self.train_envs[0].reset()
        self.obs_space = gymnasium.spaces.Dict({
            key: gymnasium.spaces.Box(-np.inf, np.inf, tuple(value.shape), np.float32)
            for key, value in observation.items()
        })
        self.action_dim = 29
        self.agent = cfg.agent.build(obs_space=self.obs_space, action_dim=self.action_dim)
        self.agent.pretrained_load_report = load_bfm_checkpoint(
            self.agent, Path(cfg.pretrained_checkpoint)
        )
        if self.agent.pretrained_load_report["model_sha256"] != OFFICIAL_BFM0_SHA256:
            raise RuntimeError("Formal Skate-BFM requires the verified official BFM0 checkpoint.")
        self.agent.checkpoint_source = "official_bfm0_pretrained"
        self.agent._model.eval()
        self.agent_update_calls = 0
        (self.work_dir / "config.json").write_text(cfg.model_dump_json(indent=2) + "\n")

    def _sample_expert_reset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        motion_key = self.reset_motion_keys[
            int(self.reset_rng.integers(len(self.reset_motion_keys)))
        ]
        record = self.reset_records[motion_key]
        required = ("source_raw_npz", "source_start_frame", "source_end_frame", "dof")
        if any(field not in record for field in required):
            raise RuntimeError(f"Expert motion {motion_key} lacks reset provenance.")
        motion_frames = int(np.asarray(record["dof"]).shape[0])
        source_start = int(record["source_start_frame"])
        source_end = int(record["source_end_frame"])
        if motion_frames <= 0 or source_end - source_start != motion_frames:
            raise RuntimeError(f"Expert motion {motion_key} has an invalid frame range.")
        local_frame = int(self.reset_rng.integers(motion_frames))
        source_frame = source_start + local_frame
        source_path = Path(record["source_raw_npz"]).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Canonical raw rollout not found: {source_path}")
        if source_path not in self.reset_raw_cache:
            with np.load(source_path, allow_pickle=False) as archive:
                if "qpos" not in archive or "qvel" not in archive:
                    raise RuntimeError(f"Canonical raw rollout lacks qpos/qvel: {source_path}")
                raw_qpos = np.asarray(archive["qpos"]).copy()
                raw_qvel = np.asarray(archive["qvel"]).copy()
            validate_raw_layout(
                source_path.with_suffix(".json"),
                raw_qpos,
                raw_qvel,
                self.train_envs[0],
            )
            self.reset_raw_cache[source_path] = raw_qpos, raw_qvel
        raw_qpos, raw_qvel = self.reset_raw_cache[source_path]
        if not 0 <= source_frame < raw_qpos.shape[0] or source_frame >= raw_qvel.shape[0]:
            raise RuntimeError(f"Expert reset frame is outside raw rollout: {source_path}")
        qpos = np.asarray(raw_qpos[source_frame], dtype=np.float64).copy()
        qvel = np.asarray(raw_qvel[source_frame], dtype=np.float64).copy()
        return qpos, qvel, {
            "motion_key": motion_key,
            "source_raw_npz": str(source_path),
            "source_frame": source_frame,
            "source_round": record.get("source_round"),
            "source_rollout": record.get("source_rollout"),
            "command_v": record.get("command_v"),
            "command_h": record.get("command_h"),
            "physics_seed": record.get("physics_seed"),
        }

    def _load_experts(self) -> dict[str, Any]:
        expert_env = make_expert_env(self.cfg.env)
        expert_base = load_expert_trajectories_from_motion_lib(
            expert_env, self.cfg.agent, device=self.cfg.buffer_device
        )
        skate_path = Path(self.cfg.skate_expert_motion_file).expanduser().resolve()
        motion_cfg = copy.deepcopy(expert_env.config.robot.motion)
        motion_cfg.motion_file = str(skate_path)
        skate_motion_lib = type(expert_env._motion_lib)(
            motion_cfg, num_envs=expert_env.num_envs, device=expert_env.device
        )
        skate_env = SimpleNamespace(
            _motion_lib=skate_motion_lib,
            num_envs=expert_env.num_envs,
            dt=expert_env.dt,
            device=expert_env.device,
            default_dof_pos=expert_env.default_dof_pos,
            gravity_vec=expert_env.gravity_vec,
            config=expert_env.config,
        )
        expert_skate = load_expert_trajectories_from_motion_lib(
            skate_env, self.cfg.agent, device=self.cfg.buffer_device
        )
        return {
            "expert_base": expert_base,
            "expert_tracking": expert_base,
            "expert_skate": expert_skate,
            "expert_slicer": BaseSkateExpertSampler(expert_base, expert_skate),
        }

    def _build_replay(self) -> dict[str, Any]:
        replay: dict[str, Any] = {}
        register_skate_replay(
            replay, DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        )
        replay.update(self._load_experts())
        return replay

    def _preflight(self, replay: dict[str, Any]) -> dict[str, int]:
        if replay["train"] is not replay["train_skate"]:
            raise RuntimeError("train replay must alias train_skate.")
        batch_size = self.agent.cfg.train.batch_size
        sequence_length = self.agent.cfg.model.seq_length
        if batch_size != 1024 or sequence_length != 8:
            raise RuntimeError(
                "Formal Skate-BFM expects BFM0 batch_size=1024 and sequence_length=8."
            )
        expert = replay["expert_slicer"].sample(batch_size)
        tracking = replay["expert_tracking"].sample(batch_size, seq_length=sequence_length)
        online = replay["train"].sample(min(16, len(replay["train"])))
        device = self.agent.device
        with torch.no_grad():
            expert_z = self.agent.encode_expert(
                next_obs=tree_map(lambda value: value.to(device), expert["next"]["observation"])
            )
            forward = self.agent._model.forward_map(
                tree_map(lambda value: value.to(device), online["observation"]),
                online["z"].to(device),
                online["action"].to(device),
            )
            backward = self.agent._model.backward_map(
                tree_map(lambda value: value.to(device), online["next"]["observation"])
            )
        if not all(bool(torch.isfinite(value).all()) for value in (expert_z, forward, backward)):
            raise RuntimeError("Expert/replay preflight produced NaN or Inf.")
        expert_observations = (
            *expert["observation"].values(),
            *expert["next"]["observation"].values(),
        )
        if (
            expert_z.shape != (1024, 256)
            or tracking["observation"]["state"].shape[0] != 1024
            or any(value.shape[0] != 1024 for value in expert_observations)
            or not all(bool(torch.isfinite(value).all()) for value in expert_observations)
        ):
            raise RuntimeError("Expert data contract is invalid.")
        return {"base_sequences": 64, "skate_sequences": 64, "sequence_length": 8}

    def _save_checkpoint(self, replay: dict[str, Any], env_step: int) -> dict[str, Any]:
        checkpoint_dir = self.work_dir / f"{CHECKPOINT_DIR_NAME}_{env_step:05d}"
        self.agent.save(str(checkpoint_dir))
        replay["train"].save(checkpoint_dir / "buffers" / "train")
        (checkpoint_dir / "train_status.json").write_text(json.dumps({"time": env_step}) + "\n")
        required = (
            checkpoint_dir / "config.json",
            checkpoint_dir / "init_kwargs.json",
            checkpoint_dir / "optimizers.pth",
            checkpoint_dir / "model" / "model.safetensors",
            checkpoint_dir / "model" / "config.json",
            checkpoint_dir / "model" / "init_kwargs.json",
            checkpoint_dir / "train_status.json",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"Checkpoint {env_step} is incomplete.")
        reloaded = self.cfg.agent.object_class.load(str(checkpoint_dir), device="cpu")
        try:
            if hash_params(reloaded._model) != hash_params(self.agent._model):
                raise RuntimeError(f"Checkpoint {env_step} model reload mismatch.")
            if hash_buffers(reloaded._model) != hash_buffers(self.agent._model):
                raise RuntimeError(f"Checkpoint {env_step} buffer reload mismatch.")
            expected = (
                [float(self.agent_update_calls)] if self.agent_update_calls else []
            )
            reports = [
                optimizer_step_report(optimizer)
                for optimizer in (
                    reloaded.forward_optimizer, reloaded.backward_optimizer,
                    reloaded.discriminator_optimizer, reloaded.critic_optimizer,
                    reloaded.aux_critic_optimizer, reloaded.actor_optimizer,
                )
            ]
            if any(report["step_values"] != expected or not report["finite"] for report in reports):
                raise RuntimeError(f"Checkpoint {env_step} optimizer reload mismatch.")
        finally:
            del reloaded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {
            "path": str(checkpoint_dir),
            "model_sha256": hash_file(checkpoint_dir / "model" / "model.safetensors"),
            "reload": "PASS",
            "optimizer_step": self.agent_update_calls,
        }

    def train(self) -> dict[str, Any]:
        """Collect HUSKY online data and call only native ``agent.update``."""

        update_steps = training_update_steps(
            self.cfg.skate_max_steps,
            self.cfg.first_update_transition,
            self.cfg.update_interval,
        )
        checkpoint_steps = training_checkpoint_steps(self.cfg.skate_max_steps)
        expected_updates = len(update_steps) * self.cfg.updates_per_block
        dataset = self.dataset_report
        print(
            "\n".join(
                (
                    "=" * 50,
                    "M2.6 FORMAL TRAINING PLAN",
                    "=" * 50,
                    f"Expert dataset: {dataset['kind']}",
                    f"Expert pkl: {dataset['motion_file']}",
                    f"Manifest: {dataset['manifest_file']}",
                    f"Motions: {dataset.get('motion_count', 'not available')}",
                    f"Expert frames: {dataset.get('expert_frames', 'not available')}",
                    f"Expert minutes: {dataset.get('expert_minutes', 'not available')}",
                    f"Expert SHA256: {dataset['motion_file_sha256']}",
                    f"BFM0 SHA256: {self.agent.pretrained_load_report['model_sha256']}",
                    f"Seed: {self.cfg.seed}",
                    f"Online envs: {self.cfg.online_envs}",
                    f"Transitions: {self.cfg.skate_max_steps}",
                    f"Replay capacity: {self.cfg.buffer_size}",
                    f"Warmup: {self.cfg.warmup_transitions}",
                    f"First update: {self.cfg.first_update_transition}",
                    f"Update interval: {self.cfg.update_interval}",
                    f"Updates / block: {self.cfg.updates_per_block}",
                    f"Expected update blocks: {len(update_steps)}",
                    f"Expected native updates: {expected_updates}",
                    f"Checkpoints: {', '.join(map(str, checkpoint_steps))}",
                    f"Work dir: {self.work_dir}",
                    "=" * 50,
                )
            )
        )
        replay = self._build_replay()
        optimizers = {
            "forward": self.agent.forward_optimizer,
            "backward": self.agent.backward_optimizer,
            "discriminator": self.agent.discriminator_optimizer,
            "critic": self.agent.critic_optimizer,
            "aux_critic": self.agent.aux_critic_optimizer,
            "actor": self.agent.actor_optimizer,
        }
        if any(optimizer_step_report(item)["state_entries"] for item in optimizers.values()):
            raise RuntimeError("Formal Skate-BFM requires fresh optimizers.")

        model = self.agent._model
        observations: list[dict[str, torch.Tensor]] = []
        z: list[torch.Tensor | None] = [None] * self.cfg.online_envs
        episode_steps = [0] * self.cfg.online_envs
        reset_counts = [0] * self.cfg.online_envs
        for index, env in enumerate(self.train_envs):
            qpos, qvel, _ = self._sample_expert_reset()
            observations.append(env.reset(qpos=qpos, qvel=qvel))
            reset_counts[index] += 1
        transitions: list[Any] = []
        update_blocks: list[dict[str, Any]] = []
        checkpoints: dict[str, Any] = {}
        actor_hashes = {"A0": hash_params(model._actor)}
        start = 1
        episodes = 0
        falls = 0
        progress = tqdm(
            total=self.cfg.skate_max_steps,
            desc="M2.6 training",
            unit="transition",
            dynamic_ncols=True,
        )

        def collect(end: int) -> None:
            nonlocal episodes, falls, start
            model.eval()
            while start <= end:
                refresh = [
                    index
                    for index, value in enumerate(z)
                    if value is None
                    or episode_steps[index] % self.agent.cfg.train.update_z_every_step == 0
                ]
                if refresh:
                    sampled_z = model.sample_z(len(refresh), device=self.agent.device)
                    for offset, index in enumerate(refresh):
                        z[index] = sampled_z[offset]
                obs_batch = {
                    key: torch.stack([item[key] for item in observations]).to(self.agent.device)
                    for key in observations[0]
                }
                z_batch = torch.stack([value for value in z if value is not None])
                with torch.no_grad():
                    actions = self.agent.act(
                        obs=obs_batch,
                        z=z_batch,
                        mean=False,
                    )
                for index, env in enumerate(self.train_envs):
                    episode_steps[index] += 1
                    transition_z = z_batch[index]
                    transition = env.step(
                        actions[index],
                        transition_z,
                        truncated=episode_steps[index] >= SKATE_EPISODE_HORIZON,
                    )
                    replay["train"].extend(transition.as_buffer_data())
                    transitions.append(transition)
                    if transition.terminated or transition.truncated:
                        episodes += 1
                        falls += int(transition.terminated)
                        qpos, qvel, _ = self._sample_expert_reset()
                        observations[index] = env.reset(qpos=qpos, qvel=qvel)
                        z[index] = None
                        episode_steps[index] = 0
                        reset_counts[index] += 1
                    else:
                        observations[index] = transition.next_observation
                progress.update(self.cfg.online_envs)
                start += self.cfg.online_envs
            progress.set_postfix(
                replay=len(replay["train"]),
                episodes=episodes,
                falls=falls,
                updates=self.agent_update_calls,
                block=len(update_blocks),
            )

        try:
            events = sorted(
                set(update_steps)
                | set(checkpoint_steps)
                | {self.cfg.warmup_transitions}
            )
            for env_step in events:
                collect(env_step)
                if env_step in update_steps:
                    block_index = len(update_blocks) + 1
                    expert_contract = self._preflight(replay)
                    model.train()
                    model.requires_grad_(True)
                    metrics: list[dict[str, float]] = []
                    for _ in range(self.cfg.updates_per_block):
                        self.agent_update_calls += 1
                        result = {
                            name: float(value.detach().mean().cpu())
                            for name, value in self.agent.update(replay, env_step).items()
                        }
                        if not result or not all(np.isfinite(value) for value in result.values()):
                            raise RuntimeError(f"Non-finite native update at step {env_step}.")
                        if any(f"aux_rew/{name}" not in result for name in AUX_REWARD_KEYS):
                            raise RuntimeError("Native update did not consume all auxiliary rewards.")
                        metrics.append(result)
                    if not all(module_state_is_finite(item) for item in (
                        model, model._obs_normalizer, model._aux_reward_normalizer
                    )):
                        raise RuntimeError("Native update produced non-finite model state.")
                    actor_hash = hash_params(model._actor)
                    if actor_hash == actor_hashes[f"A{block_index - 1}"]:
                        raise RuntimeError("Actor did not change after native update block.")
                    actor_hashes[f"A{block_index}"] = actor_hash
                    metric_summary = {
                        name: {
                            "first": values[0], "mean": float(np.mean(values)),
                            "min": min(values), "max": max(values), "last": values[-1],
                        }
                        for name, values in {
                            name: [row[name] for row in metrics] for name in metrics[0]
                        }.items()
                    }
                    update_blocks.append({
                        "env_step": env_step,
                        "native_updates": self.cfg.updates_per_block,
                        "metric_summary": metric_summary,
                    })
                    key_metrics = ", ".join(
                        f"{name}=({values['first']:.4g}/{values['mean']:.4g}/{values['last']:.4g})"
                        for name, values in list(sorted(metric_summary.items()))[:3]
                    )
                    progress.write(
                        f"[Update Block] env step={env_step}, block={block_index}, "
                        f"native updates total={self.agent_update_calls}, "
                        f"first/mean/last: {key_metrics}"
                    )
                if env_step in checkpoint_steps:
                    checkpoints[str(env_step)] = self._save_checkpoint(replay, env_step)
                    report = checkpoints[str(env_step)]
                    progress.write(
                        f"[Checkpoint] step={env_step}, path={report['path']}, "
                        f"model reload={report['reload']}, "
                        f"optimizer step={report['optimizer_step']}"
                    )
        finally:
            progress.close()
            for env in self.train_envs:
                env.close()

        full_replay = replay["train"].get_full_buffer()
        terminated = full_replay["next"]["terminated"]
        truncated = full_replay["next"]["truncated"]
        replay_size = self.cfg.skate_max_steps
        if len(transitions) != replay_size or len(replay["train"]) != replay_size:
            raise RuntimeError("Formal Skate-BFM replay length mismatch.")
        if self.agent_update_calls != expected_updates:
            raise RuntimeError("Formal Skate-BFM native update count mismatch.")
        if (
            tuple(full_replay["action"].shape) != (replay_size, 29)
            or tuple(full_replay["z"].shape) != (replay_size, 256)
        ):
            raise RuntimeError("Formal Skate-BFM replay action/latent schema mismatch.")
        if bool((terminated & truncated).any()) or tuple(full_replay["aux_rewards"]) != AUX_REWARD_KEYS:
            raise RuntimeError("Formal Skate-BFM terminal or auxiliary-reward contract mismatch.")
        for value in full_replay["aux_rewards"].values():
            if tuple(value.shape) != (replay_size, 1) or not bool(torch.isfinite(value).all()):
                raise RuntimeError("Formal Skate-BFM auxiliary replay contains invalid values.")
        optimizer_report = {name: optimizer_step_report(item) for name, item in optimizers.items()}
        expected_optimizer_steps = [float(expected_updates)]
        if any(
            report["step_values"] != expected_optimizer_steps or not report["finite"]
            for report in optimizer_report.values()
        ):
            raise RuntimeError("Formal Skate-BFM optimizer contract mismatch.")

        summary = {
            "milestone": "M2.6 Formal Skate-BFM Training",
            "dataset": self.dataset_report,
            "checkpoint": self.agent.pretrained_load_report,
            "training": {
                "env_transitions": self.cfg.skate_max_steps,
                "warmup_transitions": self.cfg.warmup_transitions,
                "first_update_transition": self.cfg.first_update_transition,
                "update_every_transitions": self.cfg.update_interval,
                "updates_per_block": self.cfg.updates_per_block,
                "total_update_blocks": len(update_steps),
                "total_native_updates": expected_updates,
                "seed": self.cfg.seed,
                "online_env_count": self.cfg.online_envs,
                "warmup_source": "pretrained_actor_stochastic",
                "domain_randomization": False,
            },
            "online_reset": {
                "mode": "expert_raw_qpos_qvel",
                "dataset": self.cfg.expert_dataset_kind,
                "total_resets": sum(reset_counts),
            },
            "replay": {
                "capacity": self.cfg.buffer_size,
                "final_size": len(replay["train"]),
                "train_is_train_skate": replay["train"] is replay["train_skate"],
                "terminated_count": int(terminated.sum()),
                "truncated_count": int(truncated.sum()),
                "normal_count": int((~(terminated | truncated)).sum()),
            },
            "expert": {
                "base_skate_ratio": [0.5, 0.5],
                "base_sequences_per_batch": expert_contract["base_sequences"],
                "skate_sequences_per_batch": expert_contract["skate_sequences"],
                "sequence_length": expert_contract["sequence_length"],
            },
            "update_blocks": update_blocks,
            "policy_versions": actor_hashes,
            "checkpoint_reports": checkpoints,
            "optimizer": optimizer_report,
            "normalizers_finite": module_state_is_finite(model._obs_normalizer)
            and module_state_is_finite(model._aux_reward_normalizer),
            "native_closed_loop": "PASS",
            "performance_evaluated": False,
            "next_milestone": "M2.6 Formal Phase/Continuous 100k Training",
        }
        (self.work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(
            "Formal Skate-BFM complete: "
            f"{len(transitions)} transitions, {self.agent_update_calls} updates"
        )
        return summary


def build_train_config() -> TrainConfig:
    """Build formal Skate-BFM configuration from environment variables."""

    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentTrainConfig
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from humanoidverse.agents.nn_filters import DictInputFilterConfig
    from humanoidverse.agents.nn_models import (
        ActorArchiConfig, BackwardArchiConfig, DiscriminatorArchiConfig,
        ForwardArchiConfig, RewardNormalizerConfig,
    )
    from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig

    dataset_kind, expert_path, manifest_path = resolve_expert_dataset()
    max_steps = int(os.environ.get("SKATE_MAX_STEPS", str(DEFAULT_MAX_STEPS)))
    seed = int(os.environ.get("SKATE_SEED", "4728"))
    budget = f"{max_steps // 1000}k" if max_steps % 1000 == 0 else str(max_steps)
    TrainConfig.model_rebuild(
        _types_namespace={
            "FBcprAuxAgentConfig": FBcprAuxAgentConfig,
            "HumanoidVerseIsaacConfig": HumanoidVerseIsaacConfig,
        }
    )
    return TrainConfig(
        name="TrainConfig",
        agent=FBcprAuxAgentConfig(
            name="FBcprAuxAgent",
            model=FBcprAuxModelConfig(
                name="FBcprAuxModel", device="cuda",
                archi=FBcprAuxModelArchiConfig(
                    name="FBcprAuxModelArchiConfig", z_dim=256, norm_z=True,
                    f=ForwardArchiConfig(name="ForwardArchi", hidden_dim=2048, model="residual", hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode="batch", input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"])),
                    b=BackwardArchiConfig(name="BackwardArchi", hidden_dim=256, hidden_layers=1, norm=True, input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"])),
                    actor=ActorArchiConfig(name="actor", model="residual", hidden_dim=2048, hidden_layers=6, embedding_layers=2, input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "last_action", "history_actor"])),
                    critic=ForwardArchiConfig(name="ForwardArchi", hidden_dim=2048, model="residual", hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode="batch", input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"])),
                    discriminator=DiscriminatorArchiConfig(name="DiscriminatorArchi", hidden_dim=1024, hidden_layers=3, input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"])),
                    aux_critic=ForwardArchiConfig(name="ForwardArchi", hidden_dim=2048, model="residual", hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode="batch", input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"])),
                ),
                obs_normalizer=ObsNormalizerConfig(name="ObsNormalizerConfig", normalizers={
                    name: BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01)
                    for name in ("state", "privileged_state", "last_action", "history_actor")
                }, allow_mismatching_keys=True),
                inference_batch_size=500000, seq_length=8, actor_std=0.05, amp=False,
                norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
            ),
            train=FBcprAuxAgentTrainConfig(
                name="FBcprAuxAgentTrainConfig", lr_f=0.0003, lr_b=1e-05, lr_actor=0.0003,
                weight_decay=0.0, clip_grad_norm=0.0, fb_target_tau=0.01, ortho_coef=100.0,
                train_goal_ratio=0.2, fb_pessimism_penalty=0.0, actor_pessimism_penalty=0.5,
                stddev_clip=0.3, q_loss_coef=0.0, batch_size=1024, discount=0.98,
                use_mix_rollout=True, update_z_every_step=100, z_buffer_size=8192,
                rollout_expert_trajectories=True, rollout_expert_trajectories_length=250,
                rollout_expert_trajectories_percentage=0.5, lr_discriminator=1e-05,
                lr_critic=0.0003, critic_target_tau=0.005, critic_pessimism_penalty=0.5,
                reg_coeff=0.05, scale_reg=True, expert_asm_ratio=0.6, relabel_ratio=0.8,
                grad_penalty_discriminator=10.0, weight_decay_discriminator=0.0,
                lr_aux_critic=0.0003, reg_coeff_aux=0.02, aux_critic_pessimism_penalty=0.5,
            ),
            aux_rewards=list(AUX_REWARD_KEYS),
            aux_rewards_scaling={
                "penalty_action_rate": -0.1, "penalty_feet_ori": -0.4,
                "penalty_ankle_roll": -4.0, "limits_dof_pos": -10.0,
                "penalty_slippage": -2.0, "penalty_undesired_contact": -1.0,
                "penalty_torques": 0.0, "limits_torque": 0.0,
            },
            cudagraphs=False, compile=False,
        ),
        env=HumanoidVerseIsaacConfig(
            name="humanoidverse_isaac", device="cuda:0",
            lafan_tail_path=str(REPOSITORY_ROOT / "train/dataset/BFM-Zero/train/lafan_29dof_10s-clipped.pkl"),
            enable_cameras=False, camera_render_save_dir="isaac_videos", max_episode_length_s=None,
            disable_obs_noise=False, disable_domain_randomization=False,
            relative_config_path="exp/bfm_zero/bfm_zero", include_last_action=True,
            hydra_overrides=["robot=g1/g1_29dof_hard_waist", "robot.control.action_scale=0.25", "robot.control.action_clip_value=5.0", "robot.control.normalize_action_to=5.0", "env.config.lie_down_init=True", "env.config.lie_down_init_prob=0.3"],
            context_length=None, include_dr_info=False, included_dr_obs_names=None,
            include_history_actor=True, include_history_noaction=False,
            make_config_g1env_compatible=False, root_height_obs=True,
        ),
        expert_dataset_kind=dataset_kind,
        skate_expert_motion_file=str(expert_path),
        expert_manifest_file=str(manifest_path),
        pretrained_checkpoint=os.environ.get("BFM0_PRETRAINED_CHECKPOINT", str(REPOSITORY_ROOT / "model/bfm-zero-official")),
        work_dir=os.environ.get(
            "SKATE_WORK_DIR",
            str(REPOSITORY_ROOT / f"results/m2.6-{dataset_kind}-{budget}-seed{seed}"),
        ),
        seed=seed,
        skate_max_steps=max_steps,
        warmup_transitions=int(
            os.environ.get("SKATE_WARMUP_TRANSITIONS", str(DEFAULT_WARMUP_TRANSITIONS))
        ),
        first_update_transition=int(
            os.environ.get("SKATE_FIRST_UPDATE", str(DEFAULT_FIRST_UPDATE))
        ),
        update_interval=int(
            os.environ.get("SKATE_UPDATE_INTERVAL", str(DEFAULT_UPDATE_INTERVAL))
        ),
        updates_per_block=int(
            os.environ.get("SKATE_UPDATES_PER_BLOCK", str(DEFAULT_UPDATES_PER_BLOCK))
        ),
        online_envs=int(os.environ.get("SKATE_ONLINE_ENVS", "4")),
        skate_expert_ratio=float(os.environ.get("SKATE_EXPERT_RATIO", "0.5")),
        buffer_size=int(os.environ.get("SKATE_BUFFER_SIZE", str(max_steps))),
    )


def main() -> int:
    build_train_config().build().train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
