# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ISAAC_ENV_ROOT = Path(
    os.environ.get("SKATE_BFM_ISAAC_ROOT", SCRIPT_DIRECTORY / "isaac_env")
).expanduser().resolve()
if not (ISAAC_ENV_ROOT / "humanoidverse").is_dir():
    raise FileNotFoundError(f"Skate-BFM Isaac runtime not found: {ISAAC_ENV_ROOT}")
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "husky_sim" / "src"))
sys.path.insert(0, str(ISAAC_ENV_ROOT))

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HYDRA_CONFIG_DIR,
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.evaluations.humanoidverse_isaac import (
    HumanoidVerseIsaacTrackingEvaluation,
    HumanoidVerseIsaacTrackingEvaluationConfig,
)
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from humanoidverse.agents.nn_models import _soft_update_params, eval_mode

os.environ["OMP_NUM_THREADS"] = "1"

import mujoco
import torch
import safetensors.torch
from safetensors import safe_open

torch.set_float32_matmul_precision("high")

import json
import time
import typing as tp
import warnings
from typing import Dict, List
from torch.utils._pytree import tree_map

import exca as xk
import gymnasium
import numpy as np
import pydantic
import torch  # better to use scoped import if we use processes
import tyro
import wandb
from packaging.version import Version
from torch.utils._pytree import tree_map
from tqdm import tqdm


from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.fb_cpr.agent import FBcprAgentConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere
from data_collection.rollout_split import randomize_husky_play_physics
from skate_bfm.integration import HuskyBfmOnlineEnv

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"

CHECKPOINT_DIR_NAME = "checkpoint"

_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER = {
    HumanoidVerseIsaacConfig: None,
}


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")



Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseIsaacTrackingEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

Agent = FBcprAgentConfig | FBcprAuxAgentConfig


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None
    skate_expert_motion_file: str | None = None
    skate_expert_ratio: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
    online_env: tp.Literal["base", "skate"] = "base"
    collect_only: bool = False
    skate_update_mode: tp.Literal["none", "fb_only"] = "none"
    adaptation_updates: int = pydantic.Field(default=0, ge=0)
    adaptation_protocol: str | None = None
    skate_max_steps: int = pydantic.Field(default=64, gt=0)
    pretrained_checkpoint: str | None = None

    env: HumanoidVerseIsaacConfig = pydantic.Field(discriminator="name")

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    seed: int = 0
    online_parallel_envs: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    log_every_updates: int = 100_000
    num_env_steps: int = 30_000_000
    # Note: this is in env steps (multiples of online_parallel_envs)
    update_agent_every: int = 500
    # Note: this is in env steps (multiples of online_parallel_envs)
    num_seed_steps: int = 50_000
    num_agent_updates: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    checkpoint_every_steps: int = 5_000_000
    checkpoint_buffer: bool = True
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"  # ["bin", "exp", "lin"]
    padding_beginning: int = 0
    padding_end: int = 0

    # Buffer
    use_trajectory_buffer: bool = False
    buffer_size: int = 5_000_000

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None

    # misc
    load_isaac_expert_data: bool = True
    buffer_device: str = "cpu"
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True

    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    # Note: this is in env steps (multiples of online_parallel_envs)
    eval_every_steps: int = 1_000_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    # exca
    infra: xk.TaskInfra = xk.TaskInfra(version="1")

    def model_post_init(self, context):
        # TODO prioritization needs tracking eval to work, but this is bit hacky to check for it
        if self.load_isaac_expert_data and not isinstance(self.env, HumanoidVerseIsaacConfig):
            raise ValueError("Loading expert isaac data is only supported for HumanoidVerseIsaacConfig")
        if self.skate_expert_motion_file is not None and not self.load_isaac_expert_data:
            raise ValueError("Skate expert MotionLib data requires load_isaac_expert_data=True")
        if self.online_env == "skate":
            if not self.load_isaac_expert_data:
                raise ValueError(
                    "Skate Workspace expert integration requires "
                    "load_isaac_expert_data=True."
                )
            if self.skate_update_mode == "none" and not self.collect_only:
                raise ValueError(
                    "skate_update_mode='none' requires collect_only=True."
                )
            if self.skate_update_mode == "fb_only" and self.collect_only:
                raise ValueError(
                    "skate_update_mode='fb_only' requires collect_only=False."
                )
            if self.skate_update_mode == "none" and self.adaptation_updates != 0:
                raise ValueError(
                    "skate_update_mode='none' requires adaptation_updates=0."
                )
            if (
                self.skate_update_mode == "fb_only"
                and self.adaptation_updates not in {1, 10, 100}
            ):
                raise ValueError(
                    "B/F-only adaptation_updates must be one of 1, 10, or 100."
                )
            if self.skate_update_mode == "fb_only" and self.adaptation_protocol is None:
                raise ValueError(
                    "B/F-only adaptation requires the fixed evaluation protocol "
                    "to define seen dynamics."
                )
            if (
                self.skate_update_mode == "fb_only"
                and self.skate_expert_ratio > 0.0
                and self.skate_expert_motion_file is None
            ):
                raise ValueError(
                    "B/F-only adaptation with skate_expert_ratio > 0 requires "
                    "skate_expert_motion_file. Set SKATE_EXPERT_RATIO=0 for "
                    "an explicit Base-only control."
                )
            if (
                self.skate_update_mode == "fb_only"
                and self.skate_expert_motion_file is not None
                and not Path(self.skate_expert_motion_file).expanduser().is_file()
            ):
                raise ValueError(
                    "Configured Skate expert motion file does not exist: "
                    f"{self.skate_expert_motion_file}"
                )
            if self.online_parallel_envs != 1:
                raise ValueError("Skate online mode currently supports exactly one environment.")
            if self.use_trajectory_buffer:
                raise ValueError("Skate online mode currently requires DictBuffer.")
            if self.prioritization:
                raise ValueError("Skate online mode does not support prioritization.")
            if self.evaluations:
                raise ValueError("Skate online mode does not run Isaac evaluations.")

        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, (HumanoidVerseIsaacTrackingEvaluationConfig)):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")


        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
            elif self.agent == FBcprAgentConfig:
                # TODO how to do checks like these in pydantic or more systematically?
                raise ValueError("FBcprAgent requires expert data to be provided (motions and motions_root)")

        # Ensure all evaluations have unique log names
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        """In case of cluster run, use exca and process instead of explivit build"""
        return Workspace(self)


class BaseSkateExpertSampler:
    """Sample complete expert sequences from the Base and Skate buffers."""

    def __init__(self, expert_base, expert_skate, skate_expert_ratio: float) -> None:
        if expert_base.seq_length != expert_skate.seq_length:
            raise ValueError(
                "Base and Skate expert buffers must use the same sequence length, "
                f"got {expert_base.seq_length} and {expert_skate.seq_length}."
            )
        if not 0.0 < skate_expert_ratio <= 1.0:
            raise ValueError(
                "BaseSkateExpertSampler requires 0 < skate_expert_ratio <= 1, "
                f"got {skate_expert_ratio}."
            )
        self.expert_base = expert_base
        self.expert_skate = expert_skate
        self.skate_expert_ratio = skate_expert_ratio
        self.seq_length = expert_base.seq_length

    def sample(self, batch_size: int = 1, seq_length: int | None = None):
        seq_length = seq_length or self.seq_length
        if batch_size < seq_length or batch_size % seq_length != 0:
            raise ValueError(
                "The batch size must be at least one sequence and divisible by "
                f"the sequence length, got batch_size={batch_size} and "
                f"seq_length={seq_length}."
            )

        sequence_count = batch_size // seq_length
        skate_sequence_count = min(
            sequence_count,
            int(sequence_count * self.skate_expert_ratio + 0.5),
        )
        base_sequence_count = sequence_count - skate_sequence_count

        batches = []
        if base_sequence_count:
            batches.append(
                self.expert_base.sample(
                    base_sequence_count * seq_length,
                    seq_length=seq_length,
                )
            )
        if skate_sequence_count:
            batches.append(
                self.expert_skate.sample(
                    skate_sequence_count * seq_length,
                    seq_length=seq_length,
                )
            )
        if len(batches) == 1:
            return batches[0]
        return tree_map(lambda base, skate: torch.cat((base, skate), dim=0), *batches)

    def __getattr__(self, name):
        # Tracking prioritization remains attached to the original Base buffer.
        return getattr(self.expert_base, name)


def register_skate_train_replay(replay_buffer: dict, train_skate) -> dict:
    """Register Skate online replay and retain the official train alias."""

    replay_buffer["train_skate"] = train_skate
    replay_buffer["train"] = train_skate
    return replay_buffer


def collect_skate_online_replay(
    model,
    *,
    steps: int = 64,
    capacity: int | None = None,
    device: str = "cuda",
    update_z_every: int = 150,
) -> tuple[dict, list]:
    """Collect a bounded frozen-model HUSKY rollout into official DictBuffer."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if update_z_every <= 0:
        raise ValueError("update_z_every must be positive.")

    replay_buffer = register_skate_train_replay(
        {},
        DictBuffer(capacity=capacity or steps, device="cpu"),
    )
    online_env = HuskyBfmOnlineEnv()
    observation = online_env.reset()
    z = None
    transitions = []
    try:
        for step in range(steps):
            if z is None or step % update_z_every == 0:
                z = model.sample_z(1, device=device)[0]
            model_observation = {
                key: value.unsqueeze(0).to(device)
                for key, value in observation.items()
            }
            with torch.no_grad():
                action_bfm = model.act(model_observation, z.unsqueeze(0), mean=True)[0]
            transition = online_env.step(
                action_bfm,
                z,
                truncated=step == steps - 1,
            )
            replay_buffer["train_skate"].extend(transition.as_buffer_data())
            transitions.append(transition)
            observation = transition.next_observation
    finally:
        online_env.close()
    return replay_buffer, transitions


def _checkpoint_model_path(checkpoint_dir: Path) -> Path:
    direct_path = checkpoint_dir / "model.safetensors"
    nested_path = checkpoint_dir / "model" / "model.safetensors"
    if direct_path.is_file():
        return direct_path
    if nested_path.is_file():
        return nested_path
    raise FileNotFoundError(
        f"Pretrained BFM0 checkpoint must contain {direct_path} or {nested_path}."
    )


def _state_fingerprint(module: torch.nn.Module, *, parameters: bool) -> str:
    """Hash model state incrementally without creating a second full model copy."""

    digest = hashlib.sha256()
    tensors = module.named_parameters() if parameters else module.named_buffers()
    for name, tensor in tensors:
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _data_fingerprint(payload: tp.Any) -> str:
    digest = hashlib.sha256()

    def update(value: tp.Any, path: str) -> None:
        digest.update(path.encode("utf-8"))
        if isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value):
                update(value[key], f"{path}/{key}")
            return
        if isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode("ascii"))
            for index, item in enumerate(value):
                update(item, f"{path}/{index}")
            return
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
        else:
            digest.update(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            return
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())

    update(payload, "root")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_fingerprints(agent) -> dict[str, str]:
    model = agent._model
    return {
        "F": _state_fingerprint(model._forward_map, parameters=True),
        "B": _state_fingerprint(model._backward_map, parameters=True),
        "target_F": _state_fingerprint(
            model._target_forward_map,
            parameters=True,
        ),
        "target_B": _state_fingerprint(
            model._target_backward_map,
            parameters=True,
        ),
        "Actor": _state_fingerprint(model._actor, parameters=True),
        "discriminator": _state_fingerprint(
            model._discriminator,
            parameters=True,
        ),
        "QD": _state_fingerprint(model._critic, parameters=True),
        "target_QD": _state_fingerprint(
            model._target_critic,
            parameters=True,
        ),
        "Qaux": _state_fingerprint(model._aux_critic, parameters=True),
        "target_Qaux": _state_fingerprint(
            model._target_aux_critic,
            parameters=True,
        ),
    }


def _space_signature(space: gymnasium.spaces.Space) -> dict[str, tuple[int, ...]]:
    if not isinstance(space, gymnasium.spaces.Dict):
        raise TypeError(f"Expected a Dict observation space, got {type(space).__name__}.")
    return {
        key: tuple(value.shape)
        for key, value in sorted(space.spaces.items())
    }


def load_pretrained_bfm0_agent(
    agent,
    checkpoint_dir: Path,
) -> dict[str, tp.Any]:
    """Strictly load the complete official BFM0 model into the built agent."""

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Official BFM0 checkpoint directory not found: {checkpoint_dir}")
    model_path = _checkpoint_model_path(checkpoint_dir)
    config_path = model_path.parent / "config.json"
    init_kwargs_path = model_path.parent / "init_kwargs.json"
    if not config_path.is_file() or not init_kwargs_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint metadata is incomplete beside {model_path}; "
            "expected config.json and init_kwargs.json."
        )

    with config_path.open() as handle:
        checkpoint_config = json.load(handle)
    with init_kwargs_path.open() as handle:
        checkpoint_init_kwargs = json.load(handle)
    if checkpoint_config.get("name") != agent._model.cfg.name:
        raise RuntimeError(
            "Pretrained BFM0 model config does not match the configured agent: "
            f"{checkpoint_config.get('name')!r} != {agent._model.cfg.name!r}."
        )
    current_model_config = agent._model.cfg.model_dump()
    if checkpoint_config != current_model_config:
        raise RuntimeError(
            "Pretrained BFM0 model configuration differs from the configured "
            "Skate agent."
        )
    checkpoint_action_dim = checkpoint_init_kwargs.get("action_dim")
    if checkpoint_action_dim != agent.action_dim:
        raise RuntimeError(
            f"Pretrained action dimension mismatch: checkpoint={checkpoint_action_dim}, "
            f"agent={agent.action_dim}."
        )
    checkpoint_obs_space = json_to_space(checkpoint_init_kwargs["obs_space"])
    if _space_signature(checkpoint_obs_space) != _space_signature(agent.obs_space):
        raise RuntimeError(
            "Pretrained observation-space mismatch: "
            f"checkpoint={_space_signature(checkpoint_obs_space)}, "
            f"agent={_space_signature(agent.obs_space)}."
        )

    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())
        checkpoint_shapes = {
            key: tuple(handle.get_slice(key).get_shape())
            for key in checkpoint_keys
        }

    expected_state = agent._model.state_dict()
    expected_keys = set(expected_state.keys())
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    shape_mismatches = sorted(
        key
        for key in expected_keys & checkpoint_keys
        if tuple(expected_state[key].shape) != checkpoint_shapes[key]
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Strict pretrained BFM0 architecture validation failed: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatches={shape_mismatches[:8]}."
        )

    safetensors.torch.load_model(
        agent._model,
        str(model_path),
        strict=True,
        device=agent.device,
    )

    component_prefixes = {
        "B": "_backward_map.",
        "target B": "_target_backward_map.",
        "F": "_forward_map.",
        "target F": "_target_forward_map.",
        "Actor": "_actor.",
        "discriminator": "_discriminator.",
        "critic": "_critic.",
        "target critic": "_target_critic.",
        "aux critic": "_aux_critic.",
        "target aux critic": "_target_aux_critic.",
        "observation normalizer": "_obs_normalizer.",
        "reward normalizer": "_aux_reward_normalizer.",
    }
    loaded_components = [
        name
        for name, prefix in component_prefixes.items()
        if any(key.startswith(prefix) for key in checkpoint_keys)
    ]
    missing_components = [
        name
        for name, prefix in component_prefixes.items()
        if not any(key.startswith(prefix) for key in checkpoint_keys)
    ]
    optimizer_present = (checkpoint_dir / "optimizers.pth").is_file()
    return {
        "source": str(checkpoint_dir),
        "model_file": str(model_path),
        "checkpoint_config": str(config_path),
        "checkpoint_init_kwargs": str(init_kwargs_path),
        "checkpoint_keys": len(checkpoint_keys),
        "loaded_components": loaded_components,
        "missing_components": missing_components,
        "optimizer_states": optimizer_present,
        "optimizer_policy": (
            "not restored; official pretrained bundle has no optimizer state"
            if not optimizer_present
            else "not restored for pretrained initialization; current-config optimizers are fresh"
        ),
    }


def build_motion_only_expert_context(env_cfg: HumanoidVerseIsaacConfig):
    """Build the vendored MotionLib expert context without IsaacLab."""

    import hydra
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
    from omegaconf import OmegaConf

    with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_DIR):
        cfg = hydra.compose(
            config_name=env_cfg.relative_config_path,
            overrides=env_cfg.hydra_overrides or [],
        )
    cfg.num_envs = 1
    cfg.exp_base = "__no_exp_base__"
    cfg.robot.asset.asset_root = cfg.robot.asset.asset_root.replace(
        "humanoidverse",
        str(ISAAC_ENV_ROOT / "humanoidverse"),
    )
    cfg.robot.motion.asset.assetRoot = cfg.robot.motion.asset.assetRoot.replace(
        "humanoidverse",
        str(ISAAC_ENV_ROOT / "humanoidverse"),
    )
    cfg.robot.motion.motion_file = env_cfg.lafan_tail_path
    cfg.obs.root_height_obs = env_cfg.root_height_obs
    pre_process_config(cfg)
    OmegaConf.set_struct(cfg, False)
    motion_lib = MotionLibRobot(
        cfg.robot.motion,
        num_envs=1,
        device=env_cfg.device,
    )
    motion_lib.load_motions_for_training()
    default_dof_pos = torch.tensor(
        [
            cfg.robot.init_state.default_joint_angles[name]
            for name in cfg.robot.dof_names
        ],
        dtype=torch.float32,
        device=env_cfg.device,
    ).unsqueeze(0)
    dt = (
        float(cfg.simulator.config.sim.control_decimation)
        / float(cfg.simulator.config.sim.fps)
    )
    return SimpleNamespace(
        _motion_lib=motion_lib,
        num_envs=1,
        dt=dt,
        device=env_cfg.device,
        default_dof_pos=default_dof_pos,
        gravity_vec=torch.tensor(
            [[0.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=env_cfg.device,
        ),
        config=cfg.env.config,
    )


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    if checkpoint_dir.exists():
        if cfg.online_env == "skate" and cfg.skate_update_mode == "fb_only":
            raise RuntimeError(
                "B/F-only diagnostic milestones must start from the official "
                "BFM0 checkpoint in an empty work directory."
            )
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]

        print(f"Loading the agent at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
        checkpoint_source = "skate_resume"
    else:
        if cfg.online_env == "skate":
            if cfg.pretrained_checkpoint is None:
                raise RuntimeError(
                    "Skate Workspace requires an official pretrained BFM0 checkpoint "
                    "when no Skate resume checkpoint exists."
                )
        agent = cfg.agent.build(**agent_build_kwargs)
        checkpoint_source = "random_initialization"
        if cfg.online_env == "skate":
            agent.pretrained_load_report = load_pretrained_bfm0_agent(
                agent,
                Path(cfg.pretrained_checkpoint),
            )
            checkpoint_source = "official_bfm0_pretrained"
            print(
                "Loaded official pretrained BFM0: "
                f"{agent.pretrained_load_report['source']}"
            )
    agent.checkpoint_source = checkpoint_source
    return agent, cfg, checkpoint_time


def init_wandb(cfg: TrainConfig):
    exp_name = "BFM-Zero"
    wandb_name = exp_name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=wandb_name, config=wandb_config, dir="./_wandb")


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.uses_base_online_env = cfg.online_env == "base"

        # HACK with Isaac, we can not recreate environments with current code, so we need to
        #      create the environment with desired number of envs here
        if self.uses_base_online_env and isinstance(cfg.env, HumanoidVerseIsaacConfig):
            from omegaconf import OmegaConf

            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
        elif self.uses_base_online_env:
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space
        else:
            self.train_env = HuskyBfmOnlineEnv()
            sample_observation = self.train_env.reset()
            self.train_env_info = {"online_env": "skate"}
            self.obs_space = gymnasium.spaces.Dict(
                {
                    key: gymnasium.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=tuple(value.shape),
                        dtype=np.float32,
                    )
                    for key, value in sample_observation.items()
                }
            )
            self.action_space = gymnasium.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(29,),
                dtype=np.float32,
            )

        if self.uses_base_online_env:
            assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
            # Original BFM agents do not consume the TimeAwareObservation field.
            del self.obs_space.spaces["time"]
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if self.uses_base_online_env and isinstance(cfg.env, HumanoidVerseIsaacConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        if self.cfg.online_env == "skate" and self.cfg.collect_only:
            # Collect-only is an inference preflight. Keep BatchNorm and all
            # other stateful modules in inference mode and freeze the model.
            self.agent._model.eval()
            self.agent._model.requires_grad_(False)
        elif (
            self.cfg.online_env == "skate"
            and self.cfg.skate_update_mode == "fb_only"
        ):
            self._configure_fb_only_boundary()
        else:
            self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()}

        if self.cfg.use_wandb:
            init_wandb(self.cfg)

        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

        self.priorization_eval_name = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseIsaacTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        # Online environment and expert source are independent. Skate uses
        # HUSKY online while its expert buffers still come from MotionLib.
        self.training_with_expert_data = True

        self.manager = None
        self.last_replay_buffer = None
        self.last_skate_transitions = []
        self.last_skate_sample = None
        self.agent_update_calls = 0
        self.fb_update_calls = 0
        self.preflight_report = {}

    def train(self):
        self.start_time = time.time()
        return self.train_online()

    def train_online(self) -> dict | None:
        if self.training_with_expert_data:
            expert_loader_env = None
            expert_loader_env_owner = None
            if self.uses_base_online_env:
                expert_loader_env = self.train_env._env
            elif isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                print("Building minimal HumanoidVerse context for expert MotionLib only")
                try:
                    expert_loader_env_owner, _ = self.cfg.env.build(num_envs=1)
                    expert_loader_env = expert_loader_env_owner._env
                except ModuleNotFoundError as error:
                    if error.name != "isaaclab":
                        raise
                    print(
                        "IsaacLab is unavailable; using the vendored MotionLib "
                        "data-only expert context."
                    )
                    expert_loader_env_owner = None
                    expert_loader_env = build_motion_only_expert_context(
                        self.cfg.env,
                    )
            else:
                raise RuntimeError(
                    "MotionLib expert loading requires HumanoidVerseIsaacConfig."
                )
            if self.cfg.load_isaac_expert_data:
                print("Loading Base expert trajectories")
                expert_base = load_expert_trajectories_from_motion_lib(
                    expert_loader_env,
                    self.cfg.agent,
                    device=self.cfg.buffer_device,
                )
            else:
                from humanoidverse.agents.buffers.load_data import (
                    load_expert_trajectories,
                )

                print("Loading Base expert trajectories")
                expert_base = load_expert_trajectories(
                    self.cfg.motions,
                    self.cfg.motions_root,
                    seq_length=self.agent.cfg.model.seq_length,
                    device=self.cfg.buffer_device,
                    # TODO data stored in disk does not have dictionary obs, so we need to manually
                    #      define what obs key the data on disk corresponds to
                    obs_dict_mapper=_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER[self.cfg.env.__class__],
                )
            expert_base.source = "base"

            expert_skate = None
            if self.cfg.skate_expert_motion_file is not None:
                skate_motion_path = Path(self.cfg.skate_expert_motion_file).expanduser().resolve()
                if not skate_motion_path.is_file():
                    raise FileNotFoundError(f"Skate expert motion file not found: {skate_motion_path}")
                skate_motion_cfg = copy.deepcopy(expert_loader_env.config.robot.motion)
                skate_motion_cfg.motion_file = str(skate_motion_path)
                skate_motion_lib = type(expert_loader_env._motion_lib)(
                    skate_motion_cfg,
                    num_envs=expert_loader_env.num_envs,
                    device=expert_loader_env.device,
                )
                skate_expert_env = SimpleNamespace(
                    _motion_lib=skate_motion_lib,
                    dt=expert_loader_env.dt,
                    device=expert_loader_env.device,
                    default_dof_pos=expert_loader_env.default_dof_pos,
                    gravity_vec=expert_loader_env.gravity_vec,
                    config=expert_loader_env.config,
                )
                print(f"Loading Skate expert trajectories from {skate_motion_path}")
                expert_skate = load_expert_trajectories_from_motion_lib(
                    skate_expert_env,
                    self.cfg.agent,
                    device=self.cfg.buffer_device,
                )
                expert_skate.source = "skate"
                motion_records = skate_motion_lib._motion_data_load
                expert_skate.source_metadata = {
                    "resolved_path": str(skate_motion_path),
                    "sha256": _file_sha256(skate_motion_path),
                    "motion_count": len(motion_records),
                    "frame_count": sum(
                        int(record["dof"].shape[0])
                        for record in motion_records.values()
                    ),
                    "fps": sorted(
                        {
                            float(record["fps"])
                            for record in motion_records.values()
                        }
                    ),
                }
            if expert_loader_env_owner is not None:
                expert_loader_env_owner.close()
        print("Creating the training environment")

        if self.uses_base_online_env and isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        elif self.uses_base_online_env:
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)
        else:
            train_env = self.train_env
            train_env_info = self.train_env_info

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        if (checkpoint_dir / "buffers/train").exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                train_skate = TrajectoryDictBufferMultiDim.load(
                    checkpoint_dir / "buffers/train",
                    device=self.cfg.buffer_device,
                )
            else:
                train_skate = DictBuffer.load(
                    checkpoint_dir / "buffers/train",
                    device=self.cfg.buffer_device,
                )
            print(f"Loaded buffer of size {len(train_skate)}")
        else:
            if self.cfg.use_trajectory_buffer:
                output_key_t = ["observation", "action", "z", "terminated", "truncated", "step_count", "reward"]
                # TODO this interface should be more elegant (how to inform buffer what keys are coming in / need to be sampled?)
                if isinstance(self.cfg.agent, (FBcprAuxAgentConfig)):
                    output_key_t.append("aux_rewards")

                train_skate = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="truncated",
                    output_key_t=output_key_t,  # TODO(team): fix this. in principle we could avoid to sample qpos, qvel for training but we need them for reward evaluation
                    output_key_tp1=["observation", "terminated"],
                )
            else:
                train_skate = DictBuffer(
                    capacity=self.cfg.buffer_size,
                    device=self.cfg.buffer_device,
                )
        register_skate_train_replay(replay_buffer, train_skate)
        if self.training_with_expert_data:
            replay_buffer["expert_base"] = expert_base
            replay_buffer["expert_tracking"] = expert_base
            if expert_skate is not None:
                replay_buffer["expert_skate"] = expert_skate
            if expert_skate is not None and self.cfg.skate_expert_ratio > 0.0:
                replay_buffer["expert_slicer"] = BaseSkateExpertSampler(
                    expert_base,
                    expert_skate,
                    self.cfg.skate_expert_ratio,
                )
            else:
                # Preserve the original BFM-Zero Base-only sampling object.
                replay_buffer["expert_slicer"] = replay_buffer["expert_base"]

        if not self.uses_base_online_env:
            if self.cfg.skate_update_mode == "fb_only":
                return self._adapt_skate_fb(replay_buffer)
            return self._collect_skate_online(replay_buffer)

        print("Starting training")
        progb = tqdm(total=self.cfg.num_env_steps, disable=self.cfg.disable_tqdm)
        td, info = train_env.reset()
        # see https://farama.org/Vector-Autoreset-Mode
        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        total_metrics, context = None, None
        start_time = time.time()
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)

        eval_instances = []
        for evaluation_name in self.evaluations.keys():
            evaluation = self.evaluations[evaluation_name]
            eval_instances.append(isinstance(evaluation, HumanoidVerseIsaacTrackingEvaluation))
        uses_humanoidverse_eval = True if any(eval_instances) else False

        for t in range(self._checkpoint_time, self.cfg.num_env_steps + self.cfg.online_parallel_envs, self.cfg.online_parallel_envs):
            if (t != self._checkpoint_time) and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer)

            if (self.evaluate and eval_time_checker.check(t)) or (self.evaluate and t == self._checkpoint_time):
                eval_metrics = self.eval(t, replay_buffer=replay_buffer)
                eval_time_checker.update_last_step(t)
                if uses_humanoidverse_eval:
                    # reset if there is a humanoidverse evaluation
                    td, info = train_env.reset()
                    terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)

                if self.cfg.prioritization:
                    assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids), (
                        "Mismatch in number of motions returned by the eval"
                    )
                    # priorities
                    index_in_buffer, name_in_buffer = {}, {}
                    for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                        index_in_buffer[motion_id] = i
                        if hasattr(replay_buffer["expert_slicer"], "file_names"):
                            name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]
                    motions_id, priorities, idxs = [], [], []
                    for _, metr in eval_metrics[self.priorization_eval_name].items():
                        motions_id.append(metr["motion_id"])
                        priorities.append(metr["emd"])
                        idxs.append(index_in_buffer[metr["motion_id"]])
                    priorities = (
                        torch.clamp(
                            torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                            min=self.cfg.prioritization_min_val,
                            max=self.cfg.prioritization_max_val,
                        )
                        * self.cfg.prioritization_scale
                    )

                    if self.cfg.prioritization_mode == "lin":
                        pass
                    elif self.cfg.prioritization_mode == "exp":
                        priorities = 2**priorities
                    elif self.cfg.prioritization_mode == "bin":
                        bins = torch.floor(priorities)
                        for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                            mask = bins == i
                            n = mask.sum().item()
                            if n > 0:
                                priorities[mask] = 1 / n
                    else:
                        raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")

                    train_env._env._motion_lib.update_sampling_weight_by_id(
                        priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer
                    )

                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=priorities.to(self.cfg.buffer_device), idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
                    )

            with torch.no_grad():
                obs = tree_map(lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device), td)
                # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
                step_count = obs.pop("time")

                history_context = None
                if "history" in obs:
                    # this works in inference mode
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(self.cfg.online_parallel_envs)
                    else:
                        history_context = self.agent.history_inference(obs=obs["history"]["observation"], action=obs["history"]["action"])[
                            :, -1
                        ].clone()

                context = self.agent.maybe_update_rollout_context(z=context, step_count=step_count, replay_buffer=replay_buffer)
                if t < self.cfg.num_seed_steps:
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # this works in inference mode
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    # TODO a bit hard-coded -- just to avoid moving stuff from cpu to cuda
                    if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                        action = action.cpu().detach().numpy()
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)

            # we check if at the next iteration we will evaluate
            next_t = t + self.cfg.online_parallel_envs
            if (self.evaluate and eval_time_checker.check(next_t)) or (self.evaluate and next_t == self._checkpoint_time):
                if isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and uses_humanoidverse_eval:
                    # make sure we set truncated since at the next iteration we are forced to reset the environment
                    # after the evaluation. This is because we share the environment with the evaluation
                    new_truncated = np.ones_like(new_truncated, dtype=bool)
                    truncated = np.ones_like(new_truncated, dtype=bool)

            if Version(gymnasium.__version__) >= Version("1.0"):
                if self.cfg.use_trajectory_buffer:
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": terminated[None, ..., None],
                        "truncated": truncated[None, ..., None],
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[None, ...]
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
                else:
                    # We add only transitions corresponding to environments that have not reset in the previous step.
                    # For environments that have reset in the previous step, the new observation corresponds to the state after reset.
                    indexes = ~done

                    real_next_obs = tree_map(lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes], new_td)
                    # TODO again, we need to remove "time" from the observation (to stay consistent with obs_space)
                    _ = real_next_obs.pop("time")
                    _ = real_next_obs.pop("history", None)

                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": real_next_obs,
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[indexes]
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = new_info["qpos"][indexes]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = new_info["qvel"][indexes]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1) for k, v in new_info["aux_rewards"].items() if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            replay_buffer["train"].extend(data)

            if (
                not self.cfg.collect_only
                and len(replay_buffer["train"]) > 0
                and t > self.cfg.num_seed_steps
                and update_agent_time_checker.check(t)
            ):
                update_agent_time_checker.update_last_step(t)
                for _ in range(self.cfg.num_agent_updates):
                    self.agent_update_calls += 1
                    metrics = self.agent.update(replay_buffer, t)
                    if total_metrics is None:
                        num_metrics_updates = 1
                        total_metrics = {k: metrics[k].float().clone() for k in metrics.keys()}
                    else:
                        num_metrics_updates += 1
                        total_metrics = {k: total_metrics[k] + metrics[k].float() for k in metrics.keys()}

            if log_time_checker.check(t) and total_metrics is not None:
                log_time_checker.update_last_step(t)
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / num_metrics_updates
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
                m_dict["timestep"] = t
                self.train_logger.log(m_dict)

            progb.update(self.cfg.online_parallel_envs)
            td = new_td
            terminated = new_terminated
            truncated = new_truncated
            done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info
        train_env.close()

    def _run_skate_preflight(self, replay_buffer: dict) -> None:
        if "expert_base" not in replay_buffer or "expert_slicer" not in replay_buffer:
            raise RuntimeError("Skate Workspace must build Base expert and expert_slicer buffers.")
        if "expert_tracking" not in replay_buffer:
            raise RuntimeError("Skate Workspace must build Base-only expert_tracking.")
        if replay_buffer["expert_tracking"] is not replay_buffer["expert_base"]:
            raise RuntimeError("expert_tracking must remain the Base expert buffer.")
        if (
            self.cfg.skate_expert_motion_file is not None
            and self.cfg.skate_expert_ratio > 0.0
        ):
            if "expert_skate" not in replay_buffer:
                raise RuntimeError("Configured Skate expert buffer was not created.")
            if not isinstance(
                replay_buffer["expert_slicer"],
                BaseSkateExpertSampler,
            ):
                raise RuntimeError(
                    "Configured Skate expert must participate through "
                    "BaseSkateExpertSampler."
                )

        expert_batch_size = self.agent.cfg.train.batch_size
        sequence_length = self.agent.cfg.model.seq_length
        sequence_count = expert_batch_size // sequence_length
        if isinstance(replay_buffer["expert_slicer"], BaseSkateExpertSampler):
            skate_sequence_count = min(
                sequence_count,
                int(
                    sequence_count
                    * replay_buffer["expert_slicer"].skate_expert_ratio
                    + 0.5
                ),
            )
        else:
            skate_sequence_count = 0
        base_sequence_count = sequence_count - skate_sequence_count
        expert_batch = replay_buffer["expert_slicer"].sample(expert_batch_size)
        tracking_batch = replay_buffer["expert_tracking"].sample(
            expert_batch_size,
            seq_length=self.agent.cfg.model.seq_length,
        )
        train_batch = replay_buffer["train_skate"].sample(
            min(16, len(replay_buffer["train_skate"]))
        )
        device = self.agent.device
        expert_next_obs = tree_map(
            lambda value: value.to(device),
            expert_batch["next"]["observation"],
        )
        train_obs = tree_map(
            lambda value: value.to(device),
            train_batch["observation"],
        )
        train_next_obs = tree_map(
            lambda value: value.to(device),
            train_batch["next"]["observation"],
        )
        train_action = train_batch["action"].to(device)
        train_z = train_batch["z"].to(device)

        with torch.no_grad():
            expert_z = self.agent.encode_expert(next_obs=expert_next_obs)
            forward_output = self.agent._model.forward_map(
                train_obs,
                train_z,
                train_action,
            )
            backward_output = self.agent._model.backward_map(train_next_obs)
            normalized_train_obs = self.agent._model._normalize(train_obs)
            discriminator_output = self.agent._model._discriminator.compute_logits(
                normalized_train_obs,
                train_z,
            )

        outputs = {
            "expert_z": expert_z,
            "forward": forward_output,
            "backward": backward_output,
            "discriminator": discriminator_output,
        }
        for name, value in outputs.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Pretrained {name} forward produced NaN or Inf.")
        if expert_z.shape != (expert_batch_size, self.agent.cfg.model.archi.z_dim):
            raise RuntimeError(f"Unexpected expert latent shape: {expert_z.shape}")
        if train_z.shape[-1] != self.agent.cfg.model.archi.z_dim:
            raise RuntimeError(f"Unexpected replay latent shape: {train_z.shape}")

        self.preflight_report.update(
            {
                "expert_base": True,
                "expert_skate": "expert_skate" in replay_buffer,
                "expert_slicer": True,
                "expert_tracking_base_only": replay_buffer["expert_tracking"]
                is replay_buffer["expert_base"],
                "expert_batch_shape": tuple(expert_z.shape),
                "expert_z_norm": float(torch.linalg.vector_norm(expert_z, dim=-1).mean()),
                "expert_base_sequences": base_sequence_count,
                "expert_skate_sequences": skate_sequence_count,
                "expert_sequence_count": sequence_count,
                "expert_sequence_length": sequence_length,
                "expert_complete_sequence_mixture": True,
                "expert_effective_skate_ratio": (
                    skate_sequence_count / sequence_count
                ),
                "train_batch_size": len(train_batch["action"]),
                "forward_shape": tuple(forward_output.shape),
                "backward_shape": tuple(backward_output.shape),
                "discriminator_shape": tuple(discriminator_output.shape),
                "tracking_batch_shape": tuple(tracking_batch["observation"]["state"].shape),
            }
        )
        if (
            self.cfg.skate_update_mode == "fb_only"
            and self.cfg.skate_expert_ratio > 0.0
        ):
            if "expert_skate" not in replay_buffer:
                raise RuntimeError("B/F-only mixture requires expert_skate.")
            if not isinstance(
                replay_buffer["expert_slicer"],
                BaseSkateExpertSampler,
            ):
                raise RuntimeError(
                    "B/F-only mixture requires BaseSkateExpertSampler."
                )
            if (
                replay_buffer["expert_slicer"].skate_expert_ratio
                != self.cfg.skate_expert_ratio
            ):
                raise RuntimeError("Effective Skate expert ratio is incorrect.")
            if self.cfg.skate_expert_ratio == 0.5 and (
                base_sequence_count != 64
                or skate_sequence_count != 64
                or sequence_count != 128
            ):
                raise RuntimeError(
                    "Expected 64 Base and 64 Skate complete sequences."
                )

    def _configure_fb_only_boundary(self) -> None:
        model = self.agent._model
        model.train()
        model.requires_grad_(False)
        model._forward_map.requires_grad_(True)
        model._backward_map.requires_grad_(True)
        model._obs_normalizer.train()

    def _load_seen_training_conditions(self) -> list[dict[str, tp.Any]]:
        protocol_path = Path(self.cfg.adaptation_protocol).expanduser().resolve()
        with protocol_path.open(encoding="utf-8") as handle:
            protocol = json.load(handle)
        if protocol.get("evaluator_version") != "skate-bfm-fixed-eval-v1":
            raise ValueError("Unsupported adaptation protocol version.")
        seen = [
            condition
            for condition in protocol["rollouts"]
            if condition["dynamics_split"] == "seen"
        ]
        if not seen:
            raise ValueError("Adaptation protocol does not define seen dynamics.")
        return [
            {
                **condition,
                "rollout_id": condition["rollout_id"].replace(
                    "eval_seen_",
                    "train_seen_",
                ),
                "rollout_seed": int(condition["rollout_seed"]) + 100_000,
                "latent_seed": int(condition["latent_seed"]) + 100_000,
            }
            for condition in seen
        ]

    def _collect_seen_skate_replay(
        self,
        replay_buffer: dict,
    ) -> tuple[list, list[dict[str, tp.Any]]]:
        conditions = self._load_seen_training_conditions()
        base_steps, remainder = divmod(self.cfg.skate_max_steps, len(conditions))
        transitions = []
        rollout_reports = []
        self.train_env.close()

        for index, condition in enumerate(conditions):
            rollout_steps = base_steps + int(index < remainder)
            random.seed(condition["rollout_seed"])
            np.random.seed(condition["rollout_seed"])
            torch.manual_seed(condition["rollout_seed"])
            env = HuskyBfmOnlineEnv()
            dynamics_report, joint_offsets = randomize_husky_play_physics(
                env.env.model,
                condition["rollout_id"],
                condition["dynamics_seed"],
            )
            env.env.set_reset_joint_offsets(joint_offsets)
            mujoco.mj_setConst(env.env.model, env.env.data)
            observation = env.reset()
            z = None
            transition_ids = []
            try:
                for step in range(rollout_steps):
                    if (
                        z is None
                        or step % self.cfg.agent.train.update_z_every_step == 0
                    ):
                        torch.manual_seed(condition["latent_seed"] + step)
                        z = self.agent._model.sample_z(
                            1,
                            device=self.agent.device,
                        )[0]
                    model_observation = {
                        key: value.unsqueeze(0).to(self.agent.device)
                        for key, value in observation.items()
                    }
                    with torch.no_grad():
                        action_bfm = self.agent.act(
                            obs=model_observation,
                            z=z.unsqueeze(0),
                            mean=True,
                        )[0]
                    transition = env.step(
                        action_bfm,
                        z,
                        truncated=step == rollout_steps - 1,
                    )
                    replay_buffer["train_skate"].extend(
                        transition.as_buffer_data()
                    )
                    transitions.append(transition)
                    transition_ids.append(
                        f"{condition['rollout_id']}:{step:04d}"
                    )
                    observation = transition.next_observation
            finally:
                env.close()
            rollout_reports.append(
                {
                    "rollout_id": condition["rollout_id"],
                    "source_eval_rollout_id": condition["rollout_id"].replace(
                        "train_seen_",
                        "eval_seen_",
                    ),
                    "dynamics_split": "seen",
                    "rollout_seed": condition["rollout_seed"],
                    "dynamics_seed": condition["dynamics_seed"],
                    "latent_seed": condition["latent_seed"],
                    "transition_count": rollout_steps,
                    "transition_ids": transition_ids,
                    "dynamics_realization": dynamics_report,
                }
            )
        return transitions, rollout_reports

    def _fb_only_update(self, replay_buffer: dict) -> dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(
            self.agent.cfg.train.batch_size
        )
        train_batch = replay_buffer["train_skate"].sample(
            self.agent.cfg.train.batch_size
        )
        device = self.agent.device
        train_obs = tree_map(
            lambda value: value.to(device),
            train_batch["observation"],
        )
        train_action = train_batch["action"].to(device)
        train_next_obs = tree_map(
            lambda value: value.to(device),
            train_batch["next"]["observation"],
        )
        discount = (
            self.agent.cfg.train.discount
            * ~train_batch["next"]["terminated"].to(device)
        )
        expert_obs = tree_map(
            lambda value: value.to(device),
            expert_batch["observation"],
        )
        expert_next_obs = tree_map(
            lambda value: value.to(device),
            expert_batch["next"]["observation"],
        )

        self.agent._model._obs_normalizer(train_obs)
        self.agent._model._obs_normalizer(train_next_obs)
        with torch.no_grad(), eval_mode(self.agent._model._obs_normalizer):
            train_obs = self.agent._model._obs_normalizer(train_obs)
            train_next_obs = self.agent._model._obs_normalizer(train_next_obs)
            expert_obs = self.agent._model._obs_normalizer(expert_obs)
            expert_next_obs = self.agent._model._obs_normalizer(
                expert_next_obs
            )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.agent.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(device)
        mixed_z = self.agent.sample_mixed_z(
            train_goal=train_next_obs,
            expert_encodings=expert_z,
        ).clone()
        self.agent.z_buffer.add(mixed_z)
        if self.agent.cfg.train.relabel_ratio is not None:
            mask = (
                torch.rand(
                    (self.agent.cfg.train.batch_size, 1),
                    device=device,
                )
                <= self.agent.cfg.train.relabel_ratio
            )
            train_z = torch.where(mask, mixed_z, train_z)

        q_loss_coef = (
            self.agent.cfg.train.q_loss_coef
            if self.agent.cfg.train.q_loss_coef > 0
            else None
        )
        clip_grad_norm = (
            self.agent.cfg.train.clip_grad_norm
            if self.agent.cfg.train.clip_grad_norm > 0
            else None
        )
        metrics = self.agent.update_fb(
            obs=train_obs,
            action=train_action,
            discount=discount,
            next_obs=train_next_obs,
            goal=train_next_obs,
            z=train_z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        with torch.no_grad():
            _soft_update_params(
                self.agent._forward_map_paramlist,
                self.agent._target_forward_map_paramlist,
                self.agent.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self.agent._backward_map_paramlist,
                self.agent._target_backward_map_paramlist,
                self.agent.cfg.train.fb_target_tau,
            )
        return metrics

    def _adapt_skate_fb(self, replay_buffer: dict) -> dict:
        if self.cfg.collect_only:
            raise RuntimeError("B/F-only adaptation cannot run collect-only.")
        if self.cfg.skate_update_mode != "fb_only":
            raise RuntimeError("Full Skate agent updates remain prohibited.")
        if replay_buffer["train"] is not replay_buffer["train_skate"]:
            raise RuntimeError("train must remain an alias of train_skate.")

        transitions, rollout_reports = self._collect_seen_skate_replay(
            replay_buffer
        )
        if len(transitions) != self.cfg.skate_max_steps:
            raise RuntimeError("Seen Skate replay collection was incomplete.")
        if any(
            report["dynamics_split"] != "seen"
            for report in rollout_reports
        ):
            raise RuntimeError("Unseen dynamics entered Skate training replay.")
        if any(
            transition_id.startswith("eval_")
            for report in rollout_reports
            for transition_id in report["transition_ids"]
        ):
            raise RuntimeError("Evaluation transitions entered training replay.")

        self._run_skate_preflight(replay_buffer)
        transition_ids = [
            transition_id
            for rollout in rollout_reports
            for transition_id in rollout["transition_ids"]
        ]
        replay_fingerprint = _data_fingerprint(
            replay_buffer["train_skate"].get_full_buffer()
        )
        transition_ids_fingerprint = _data_fingerprint(transition_ids)
        dynamics_fingerprint = _data_fingerprint(
            [
                rollout["dynamics_realization"]
                for rollout in rollout_reports
            ]
        )
        before = _component_fingerprints(self.agent)
        normalizer_before = _state_fingerprint(
            self.agent._model._obs_normalizer,
            parameters=False,
        )
        z_buffer_before = _tensor_fingerprint(self.agent.z_buffer._storage)
        metric_history = []
        max_gradient = {"F": 0.0, "B": 0.0}

        for update_index in range(self.cfg.adaptation_updates):
            metrics = self._fb_only_update(replay_buffer)
            scalar_metrics = {
                name: float(value.detach().mean().cpu())
                for name, value in metrics.items()
            }
            if not all(np.isfinite(value) for value in scalar_metrics.values()):
                raise RuntimeError(
                    f"Non-finite B/F metric at update {update_index + 1}."
                )
            for name, module in (
                ("F", self.agent._model._forward_map),
                ("B", self.agent._model._backward_map),
            ):
                gradients = [
                    parameter.grad
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ]
                if not gradients or not all(
                    torch.isfinite(gradient).all() for gradient in gradients
                ):
                    raise RuntimeError(
                        f"{name} gradients are missing or non-finite at "
                        f"update {update_index + 1}."
                    )
                max_gradient[name] = max(
                    max_gradient[name],
                    max(float(gradient.detach().abs().max()) for gradient in gradients),
                )
            metric_history.append(scalar_metrics)
            self.fb_update_calls += 1

        after = _component_fingerprints(self.agent)
        changed = {
            name: before[name] != after[name]
            for name in before
        }
        expected_changed = {"F", "B", "target_F", "target_B"}
        for name in expected_changed:
            if not changed[name]:
                raise RuntimeError(f"Expected {name} to change during adaptation.")
        for name in set(changed) - expected_changed:
            if changed[name]:
                raise RuntimeError(f"Forbidden component changed: {name}.")

        normalizer_after = _state_fingerprint(
            self.agent._model._obs_normalizer,
            parameters=False,
        )
        z_buffer_after = _tensor_fingerprint(self.agent.z_buffer._storage)
        report = {
            "stage": "M2.2b-1",
            "method": "Original FB + Skate",
            "adaptation": "experimental B/F-only adaptation",
            "pretrained_checkpoint": self.agent.pretrained_load_report,
            "training_replay": {
                "buffer": "train_skate",
                "train_alias_is_train_skate": replay_buffer["train"]
                is replay_buffer["train_skate"],
                "transition_count": len(replay_buffer["train_skate"]),
                "rollouts": rollout_reports,
                "transition_ids_fingerprint": transition_ids_fingerprint,
                "replay_tensor_fingerprint": replay_fingerprint,
                "dynamics_realization_fingerprint": dynamics_fingerprint,
                "skate_only": True,
                "unseen_transition_count": 0,
                "eval_transition_count": 0,
            },
            "expert": {
                "source": (
                    "Base + Skate"
                    if "expert_skate" in replay_buffer
                    else "Base only"
                ),
                "skate_buffer_present": "expert_skate" in replay_buffer,
                "skate_expert_ratio": self.cfg.skate_expert_ratio,
                "base_sequences": self.preflight_report[
                    "expert_base_sequences"
                ],
                "skate_sequences": self.preflight_report[
                    "expert_skate_sequences"
                ],
                "total_sequences": self.preflight_report[
                    "expert_sequence_count"
                ],
                "sequence_length": self.preflight_report[
                    "expert_sequence_length"
                ],
                "complete_sequence_mixture": self.preflight_report[
                    "expert_complete_sequence_mixture"
                ],
                "skate_artifact": (
                    replay_buffer["expert_skate"].source_metadata
                    if "expert_skate" in replay_buffer
                    else None
                ),
            },
            "updates": self.cfg.adaptation_updates,
            "optimizer": {
                "F_learning_rate": self.agent.cfg.train.lr_f,
                "B_learning_rate": self.agent.cfg.train.lr_b,
                "fresh_optimizer_state": True,
                "allowed": ["forward_optimizer", "backward_optimizer"],
                "forbidden": [
                    "actor_optimizer",
                    "discriminator_optimizer",
                    "critic_optimizer",
                    "aux_critic_optimizer",
                ],
            },
            "fb_config": {
                "discount": self.agent.cfg.train.discount,
                "fb_target_tau": self.agent.cfg.train.fb_target_tau,
                "ortho_coef": self.agent.cfg.train.ortho_coef,
                "q_loss_coef": self.agent.cfg.train.q_loss_coef,
                "clip_grad_norm": self.agent.cfg.train.clip_grad_norm,
                "train_goal_ratio": self.agent.cfg.train.train_goal_ratio,
                "expert_asm_ratio": self.agent.cfg.train.expert_asm_ratio,
                "relabel_ratio": self.agent.cfg.train.relabel_ratio,
            },
            "parameter_fingerprints": {
                name: {
                    "before": before[name],
                    "after": after[name],
                    "changed": changed[name],
                }
                for name in before
            },
            "normalizer": {
                "before": normalizer_before,
                "after": normalizer_after,
                "changed": normalizer_before != normalizer_after,
                "semantics": (
                    "updated from train_obs and train_next_obs exactly as in "
                    "the vendored FBcprAuxAgent.update path"
                ),
            },
            "z_buffer": {
                "before": z_buffer_before,
                "after": z_buffer_after,
                "changed": z_buffer_before != z_buffer_after,
                "size": len(self.agent.z_buffer),
            },
            "gradients": {
                "all_finite": True,
                "maximum_absolute": max_gradient,
            },
            "losses": {
                "all_finite": True,
                "first": metric_history[0],
                "last": metric_history[-1],
            },
            "agent_update_calls": self.agent_update_calls,
            "direct_update_fb_calls": self.fb_update_calls,
            "native_termination": "unresolved",
            "Qaux": "unresolved",
            "command_aligned_downstream": "unresolved",
        }
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with (self.work_dir / "adaptation_report.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self.save(self.cfg.adaptation_updates, replay_buffer)
        self.last_replay_buffer = replay_buffer
        self.last_skate_transitions = transitions
        self.preflight_report = report
        print(
            "M2.2b-1 B/F-only adaptation complete: "
            f"{len(replay_buffer['train_skate'])} Skate transitions, "
            f"{self.cfg.adaptation_updates} direct update_fb calls, "
            "forbidden optimizer calls 0"
        )
        return replay_buffer

    def _collect_skate_online(self, replay_buffer: dict) -> dict:
        if not self.cfg.collect_only:
            raise RuntimeError("Skate online updates are disabled in M2.1b.")
        if not isinstance(self.train_env, HuskyBfmOnlineEnv):
            raise RuntimeError("Skate online mode must use HuskyBfmOnlineEnv.")

        self.preflight_report.update(
            {
                "model_parameters_before": _state_fingerprint(
                    self.agent._model,
                    parameters=True,
                ),
                "model_buffers_before": _state_fingerprint(
                    self.agent._model,
                    parameters=False,
                ),
            }
        )
        print(
            f"Starting formal {type(self.train_env).__name__} collect-only path "
            f"for {self.cfg.skate_max_steps} steps"
        )
        observation = self.train_env.reset()
        transitions = []
        z = None
        try:
            for step in range(self.cfg.skate_max_steps):
                if z is None or step % self.cfg.agent.train.update_z_every_step == 0:
                    z = self.agent._model.sample_z(1, device=self.agent.device)[0]
                model_observation = {
                    key: value.unsqueeze(0).to(self.agent.device)
                    for key, value in observation.items()
                }
                with torch.no_grad():
                    action_bfm = self.agent.act(
                        obs=model_observation,
                        z=z.unsqueeze(0),
                        mean=True,
                    )[0]
                transition = self.train_env.step(
                    action_bfm,
                    z,
                    truncated=step == self.cfg.skate_max_steps - 1,
                )
                replay_buffer["train_skate"].extend(
                    transition.as_buffer_data()
                )
                transitions.append(transition)

                if transition.terminated or transition.truncated:
                    if step != self.cfg.skate_max_steps - 1:
                        observation = self.train_env.reset()
                    else:
                        observation = transition.next_observation
                else:
                    observation = transition.next_observation
        finally:
            self.train_env.close()

        self.last_replay_buffer = replay_buffer
        self.last_skate_transitions = transitions
        if replay_buffer["train"] is not replay_buffer["train_skate"]:
            raise RuntimeError("train must remain an alias of train_skate.")
        if len(transitions) != self.cfg.skate_max_steps:
            raise RuntimeError(
                f"Expected {self.cfg.skate_max_steps} transitions, got {len(transitions)}."
            )
        if not transitions[-1].truncated:
            raise RuntimeError("The bounded Skate rollout must end with truncation.")
        if any(
            transition.action_bfm.shape != (29,)
            or transition.action_husky.shape != (23,)
            for transition in transitions
        ):
            raise RuntimeError("Skate action contract must remain 29D stored / 23D executed.")
        self.last_skate_sample = replay_buffer["train_skate"].sample(
            min(16, len(replay_buffer["train_skate"]))
        )
        expected_observation_shapes = {
            "state": 64,
            "privileged_state": 463,
            "last_action": 29,
            "history_actor": 372,
        }
        for key, width in expected_observation_shapes.items():
            if self.last_skate_sample["observation"][key].shape[-1] != width:
                raise RuntimeError(f"Invalid Skate replay observation field: {key}.")
        if self.last_skate_sample["action"].shape[-1] != 29:
            raise RuntimeError("Skate replay must store the 29D BFM action.")
        if self.last_skate_sample["z"].shape[-1] != 256:
            raise RuntimeError("Skate replay must store the 256D rollout latent.")
        if self.agent_update_calls != 0:
            raise RuntimeError("Collect-only mode must not call agent.update().")
        self._run_skate_preflight(replay_buffer)
        model_parameters_after = _state_fingerprint(self.agent._model, parameters=True)
        model_buffers_after = _state_fingerprint(self.agent._model, parameters=False)
        if model_parameters_after != self.preflight_report["model_parameters_before"]:
            raise RuntimeError("Pretrained model parameters changed during collect-only preflight.")
        if model_buffers_after != self.preflight_report["model_buffers_before"]:
            raise RuntimeError("Pretrained model buffers changed during collect-only preflight.")
        self.preflight_report.update(
            {
                "model_parameters_after": model_parameters_after,
                "model_buffers_after": model_buffers_after,
                "parameter_mutation": False,
                "buffer_mutation": False,
                "agent_update_calls": self.agent_update_calls,
                "optimizer_steps": 0,
            }
        )
        print(
            "M2.2a preflight complete: "
            f"expert Base/Skate sequences "
            f"{self.preflight_report['expert_base_sequences']}/"
            f"{self.preflight_report['expert_skate_sequences']}, "
            f"expert_z {self.preflight_report['expert_batch_shape']}, "
            f"F {self.preflight_report['forward_shape']}, "
            f"B {self.preflight_report['backward_shape']}, "
            "parameter mutation false, buffer mutation false"
        )
        print(
            "HUSKY collect-only complete: "
            f"{len(replay_buffer['train_skate'])} transitions, "
            "optimizer updates 0"
        )
        return replay_buffer

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations.keys():
            logger = self.eval_loggers[evaluation_name]
            evaluation = self.evaluations[evaluation_name]

            # NOTE we have this inside the loop so that the agent is not moved to cpu if we don't evaluate
            if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                self.agent._model.to("cpu")
            self.agent._model.train(False)

            if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                # Pass train env
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t, agent_or_model=self.agent, replay_buffer=replay_buffer, logger=logger, env=self.train_env
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )
            # For wandb dict, put it on wandb
            if self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()

        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        print(f"Checkpointing at time {time}")
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))
        if self.cfg.checkpoint_buffer:
            replay_buffer["train"].save(self.work_dir / CHECKPOINT_DIR_NAME / "buffers" / "train")
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time}, f, indent=4)


def build_train_config() -> TrainConfig:
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentTrainConfig
    from humanoidverse.agents.nn_models import ForwardArchiConfig, BackwardArchiConfig, ActorArchiConfig, DiscriminatorArchiConfig, RewardNormalizerConfig
    from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig
    from humanoidverse.agents.nn_filters import DictInputFilterConfig

    online_env = os.environ.get("SKATE_ONLINE_ENV", "base").strip().lower()
    if online_env not in {"base", "skate"}:
        raise ValueError(
            f"SKATE_ONLINE_ENV must be 'base' or 'skate', got {online_env!r}."
        )
    skate_mode = online_env == "skate"
    skate_update_mode = os.environ.get(
        "SKATE_UPDATE_MODE",
        "none",
    ).strip().lower()
    if skate_update_mode not in {"none", "fb_only"}:
        raise ValueError(
            "SKATE_UPDATE_MODE must be 'none' or 'fb_only', "
            f"got {skate_update_mode!r}."
        )
    collect_only = _environment_flag(
        "SKATE_COLLECT_ONLY",
        default=skate_mode and skate_update_mode == "none",
    )
    adaptation_updates = int(os.environ.get("SKATE_ADAPTATION_UPDATES", "0"))
    default_skate_steps = "1024" if skate_update_mode == "fb_only" else "64"
    skate_max_steps = int(
        os.environ.get("SKATE_MAX_STEPS", default_skate_steps)
    )

    cfg = TrainConfig(
        name='TrainConfig',
        agent=FBcprAuxAgentConfig(
            name='FBcprAuxAgent',
            model=FBcprAuxModelConfig(
                name='FBcprAuxModel',
                device='cuda',
                archi=FBcprAuxModelArchiConfig(
                    name='FBcprAuxModelArchiConfig',
                    z_dim=256,
                    norm_z=True,
                    f=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                    b=BackwardArchiConfig(name='BackwardArchi', hidden_dim=256, hidden_layers=1, norm=True, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    actor=ActorArchiConfig(name='actor', model='residual', hidden_dim=2048, hidden_layers=6, embedding_layers=2, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'last_action', 'history_actor'])),
                    critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                    discriminator=DiscriminatorArchiConfig(name='DiscriminatorArchi', hidden_dim=1024, hidden_layers=3, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    aux_critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor']))
                ),
                obs_normalizer=ObsNormalizerConfig(
                    name='ObsNormalizerConfig',
                    normalizers={
                        'state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'privileged_state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'last_action': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'history_actor': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01)
                    },
                    allow_mismatching_keys=True
                ),
                inference_batch_size=500000,
                seq_length=8,
                actor_std=0.05,
                amp=False,
                norm_aux_reward=RewardNormalizerConfig(name='RewardNormalizer', translate=False, scale=True)
            ),
            train=FBcprAuxAgentTrainConfig(
                name='FBcprAuxAgentTrainConfig',
                lr_f=0.0003,
                lr_b=1e-05,
                lr_actor=0.0003,
                weight_decay=0.0,
                clip_grad_norm=0.0,
                fb_target_tau=0.01,
                ortho_coef=100.0,
                train_goal_ratio=0.2,
                fb_pessimism_penalty=0.0,
                actor_pessimism_penalty=0.5,
                stddev_clip=0.3,
                q_loss_coef=0.0,
                batch_size=1024,
                discount=0.98,
                use_mix_rollout=True,
                update_z_every_step=100,
                z_buffer_size=8192,
                rollout_expert_trajectories=True,
                rollout_expert_trajectories_length=250,
                rollout_expert_trajectories_percentage=0.5,
                lr_discriminator=1e-05,
                lr_critic=0.0003,
                critic_target_tau=0.005,
                critic_pessimism_penalty=0.5,
                reg_coeff=0.05,
                scale_reg=True,
                expert_asm_ratio=0.6,
                relabel_ratio=0.8,
                grad_penalty_discriminator=10.0,
                weight_decay_discriminator=0.0,
                lr_aux_critic=0.0003,
                reg_coeff_aux=0.02,
                aux_critic_pessimism_penalty=0.5
            ),
            aux_rewards=['penalty_torques', 'penalty_action_rate', 'limits_dof_pos', 'limits_torque', 'penalty_undesired_contact', 'penalty_feet_ori', 'penalty_ankle_roll', 'penalty_slippage'],
            aux_rewards_scaling={'penalty_action_rate': -0.1, 'penalty_feet_ori': -0.4, 'penalty_ankle_roll': -4.0, 'limits_dof_pos': -10.0, 'penalty_slippage': -2.0, 'penalty_undesired_contact': -1.0, 'penalty_torques': 0.0, 'limits_torque': 0.0},
            cudagraphs=False,
            compile=not skate_mode
        ),
        motions='',
        motions_root='',
        skate_expert_motion_file=os.environ.get("SKATE_EXPERT_MOTION_FILE"),
        skate_expert_ratio=float(os.environ.get("SKATE_EXPERT_RATIO", "0.5")),
        online_env=online_env,
        collect_only=collect_only,
        skate_update_mode=skate_update_mode,
        adaptation_updates=adaptation_updates,
        adaptation_protocol=os.environ.get(
            "SKATE_ADAPTATION_PROTOCOL",
            str(REPOSITORY_ROOT / "train" / "evaluation_protocol.json"),
        ),
        skate_max_steps=skate_max_steps,
        pretrained_checkpoint=os.environ.get(
            "BFM0_PRETRAINED_CHECKPOINT",
            str(REPOSITORY_ROOT / "model" / "bfm-zero-official"),
        ),
        env=HumanoidVerseIsaacConfig(
            name='humanoidverse_isaac',
            device='cuda:0',
            lafan_tail_path=str(
                REPOSITORY_ROOT
                / "train"
                / "dataset"
                / "BFM-Zero"
                / "train"
                / "lafan_29dof_10s-clipped.pkl"
            ),
            enable_cameras=False,
            camera_render_save_dir='isaac_videos',
            max_episode_length_s=None,
            disable_obs_noise=False,
            disable_domain_randomization=False,
            relative_config_path='exp/bfm_zero/bfm_zero',
            include_last_action=True,
            hydra_overrides=['robot=g1/g1_29dof_hard_waist', 'robot.control.action_scale=0.25', 'robot.control.action_clip_value=5.0', 'robot.control.normalize_action_to=5.0', 'env.config.lie_down_init=True', 'env.config.lie_down_init_prob=0.3'],
            context_length=None,
            include_dr_info=False,
            included_dr_obs_names=None,
            include_history_actor=True,
            include_history_noaction=False,
            make_config_g1env_compatible=False,
            root_height_obs=True
        ),
        work_dir=os.environ.get(
            "SKATE_WORK_DIR",
            "results/skate-online-dry-run" if skate_mode else "results/bfmzero-isaac",
        ),
        seed=4728,
        online_parallel_envs=1 if skate_mode else 1024,
        log_every_updates=384000,
        num_env_steps=skate_max_steps if skate_mode else 384000000,
        update_agent_every=1024,
        num_seed_steps=10240,
        num_agent_updates=16,
        checkpoint_every_steps=9600000,
        checkpoint_buffer=not skate_mode,
        prioritization=False if skate_mode else True,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode='exp',
        use_trajectory_buffer=False if skate_mode else True,
        buffer_size=max(skate_max_steps, 1024) if skate_mode else 5120000,
        use_wandb=False,
        wandb_ename='yitangl',  # your wandb entity (username/team), empty = default from wandb login
        wandb_gname='bfmzero-isaac',  # run group
        wandb_pname='bfmzero-isaac',  # your wandb project name
        load_isaac_expert_data=True,
        buffer_device='cpu' if skate_mode else 'cuda',
        disable_tqdm=True,
        evaluations=[] if skate_mode else [HumanoidVerseIsaacTrackingEvaluationConfig(name='HumanoidVerseIsaacTrackingEvaluationConfig', generate_videos=False, videos_dir='videos', video_name_prefix='unknown_agent', name_in_logs='humanoidverse_tracking_eval', env=None, num_envs=1024, n_episodes_per_motion=1)],
        eval_every_steps=9600000,
        tags={},
    )
    return cfg


def train_skate_bfm():
    workspace = build_train_config().build()
    return workspace.train()


if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    train_skate_bfm()

# uv run --no-cache -m humanoidverse.meta_online_entry_point
