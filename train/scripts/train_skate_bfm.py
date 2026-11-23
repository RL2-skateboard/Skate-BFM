#!/usr/bin/env python3
# ruff: noqa: E402, E501, I001
"""Train Skate-BFM on the native29 skateboard environment.

This is intentionally a small project-owned entrypoint. The vendored
``isaac_env/humanoidverse`` package retains the official BFM-Zero algorithms;
this file connects their agent and MotionLib data to native29 skate-sim.
"""

from __future__ import annotations

import argparse
import copy
import json
import joblib
import os
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime
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
    str(ISAAC_ENV_ROOT),
    str(SCRIPT_DIRECTORY),
]

import gymnasium
import numpy as np
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.agents.envs.humanoidverse_isaac import (
    HYDRA_CONFIG_DIR,
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.utils import set_seed_everywhere
from skate_bfm.integration import SkateSimOnlineEnv, AUX_REWARD_KEYS
from train_runner import (
    OFFICIAL_BFM0_SHA256,
    checkpoint_model_path,
    hash_buffers,
    hash_components,
    hash_data,
    hash_file,
    hash_params,
    load_bfm_checkpoint,
    module_state_is_finite,
    optimizer_step_report,
)

os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_float32_matmul_precision("high")

CHECKPOINT_DIR_NAME = "checkpoint"
SKATE_EPISODE_HORIZON = 1024
DEFAULT_MAX_STEPS = 100_000
DEFAULT_WARMUP_TRANSITIONS = 1024
DEFAULT_FIRST_UPDATE = 1500
DEFAULT_UPDATE_INTERVAL = 500
DEFAULT_UPDATES_PER_BLOCK = 50
DATASET_ROOT = REPOSITORY_ROOT / "train/dataset"
BASE_MOTION_FILE = DATASET_ROOT / "base/lafan_29dof_10s-clipped.pkl"
SKATE_DATASET_ROOT = DATASET_ROOT / "sim_collected"
EXPERT_DATASETS = {
    "phase": (
        SKATE_DATASET_ROOT
        / "train/phase/motion_library/skate_expert_phase.pkl"
    ),
    "continuous": (
        SKATE_DATASET_ROOT
        / "train/continuous/motion_library/skate_expert_continuous.pkl"
    ),
}


def training_update_steps(
    max_steps: int, first_update: int, update_interval: int
) -> tuple[int, ...]:
    return tuple(range(first_update, max_steps + 1, update_interval))


def training_checkpoint_steps(max_steps: int) -> tuple[int, ...]:
    candidates = (20_000, 50_000, 100_000, 120_000, 150_000, 200_000, max_steps)
    return tuple(sorted({step for step in candidates if step <= max_steps}))


def resolve_expert_dataset(
    dataset: str | None = None,
    expert_motion: str | Path | None = None,
) -> tuple[str, Path, Path]:
    override = expert_motion or os.environ.get("SKATE_EXPERT_MOTION_FILE")
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
        dataset_kind = (
            dataset or os.environ.get("SKATE_EXPERT_DATASET", "phase")
        ).strip().lower()
        if dataset_kind not in EXPERT_DATASETS:
            raise ValueError("SKATE_EXPERT_DATASET must be 'phase' or 'continuous'.")
        expert_path = EXPERT_DATASETS[dataset_kind].resolve()
    return dataset_kind, expert_path, expert_path.with_name("manifest.json")


class TrainConfig(BaseConfig):
    """Narrow configuration surface for formal Skate-BFM training."""

    agent: FBcprAuxAgentConfig
    env: HumanoidVerseIsaacConfig
    expert_dataset_kind: str
    skate_expert_motion_file: str
    expert_manifest_file: str
    pretrained_checkpoint: str
    work_dir: str
    checkpoint_dir: str
    seed: int = 4728
    skate_max_steps: int = DEFAULT_MAX_STEPS
    warmup_transitions: int = DEFAULT_WARMUP_TRANSITIONS
    first_update_transition: int = DEFAULT_FIRST_UPDATE
    update_interval: int = DEFAULT_UPDATE_INTERVAL
    updates_per_block: int = DEFAULT_UPDATES_PER_BLOCK
    online_envs: int = 4
    online_env_backend: str = "skate-sim-mjlab"
    native_reset_preset: str = "steer_start"
    aligned_online_tracking: bool = False
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
        if self.online_env_backend not in ("skate-sim-cpu", "skate-sim-mjlab"):
            raise ValueError("Only native29 skate-sim-cpu/skate-sim-mjlab backends are supported; legacy23 is retired.")
        if self.native_reset_preset not in ("steer_start", "push_start"):
            raise ValueError("Unknown native reset preset.")
        if self.aligned_online_tracking:
            raise ValueError("Native29 preset resets cannot reproduce aligned legacy expert motion/frame tracking.")
        if self.online_env_backend == "skate-sim-cpu" and self.online_envs != 1:
            raise ValueError("skate-sim CPU currently supports online_envs=1; use mjlab for batched worlds.")
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


def assemble_expert_replay(expert_base: Any, expert_skate: Any) -> dict[str, Any]:
    return {
        "expert_base": expert_base,
        "expert_tracking": expert_skate,
        "expert_skate": expert_skate,
        "expert_slicer": BaseSkateExpertSampler(expert_base, expert_skate),
    }


def register_skate_replay(replay_buffer: dict[str, Any], train_skate: Any) -> None:
    """Keep the official BFM ``train`` key as an alias of online skate replay."""

    replay_buffer["train_skate"] = train_skate
    replay_buffer["train"] = train_skate


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


def load_expert_buffer(
    agent: Any,
    motion_file: str | Path,
    *,
    device: str | torch.device | None = None,
) -> Any:
    """Load one source through the official BFM MotionLib expert path."""

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
    return load_expert_trajectories_from_motion_lib(
        expert_env,
        agent.cfg,
        device=agent.device if device is None else device,
    )


def load_expert(agent: Any, motion_file: str | Path) -> dict[str, torch.Tensor]:
    """Return BFM-compatible observations for one MotionLib source."""

    buffer = load_expert_buffer(agent, motion_file)
    return {
        name: value.detach().cpu()
        for name, value in buffer.storage["observation"].items()
        if name in {"state", "last_action", "privileged_state"}
    }


class AlignedSkateTrackingContext:
    """Strict reset-to-MotionLib lookup for per-step expert tracking latents."""

    def __init__(
        self,
        observations: Mapping[str, torch.Tensor],
        file_names: list[str],
        trajectory_lengths: list[int],
        motion_frames: Mapping[str, int],
    ) -> None:
        names = tuple(map(str, file_names))
        if len(names) != len(set(names)):
            raise RuntimeError("Skate expert MotionLib contains duplicate motion keys.")
        if set(names) != set(motion_frames):
            raise RuntimeError("Reset and MotionLib motion-key sets do not match.")
        if len(names) != len(trajectory_lengths):
            raise RuntimeError("MotionLib names and trajectory lengths do not match.")
        if any(value.shape[0] != sum(trajectory_lengths) for value in observations.values()):
            raise RuntimeError("MotionLib observation storage length is inconsistent.")

        self.observations = dict(observations)
        self.trajectories: dict[str, dict[str, int]] = {}
        offset = 0
        for trajectory_id, (name, length) in enumerate(
            zip(names, trajectory_lengths, strict=True)
        ):
            raw_frames = int(motion_frames[name])
            if length <= 0 or raw_frames <= 0 or length > raw_frames:
                raise RuntimeError(f"Invalid frame relation for Skate motion {name}.")
            self.trajectories[name] = {
                "trajectory_id": trajectory_id,
                "start": offset,
                "length": int(length),
                "raw_frames": raw_frames,
            }
            offset += int(length)

    @classmethod
    def from_buffer(
        cls,
        buffer: Any,
        motion_frames: Mapping[str, int],
    ) -> AlignedSkateTrackingContext:
        done = torch.as_tensor(buffer.storage["truncated"]).flatten().cpu()
        ends = torch.nonzero(done, as_tuple=False).flatten().tolist()
        previous = -1
        lengths = []
        for end in ends:
            lengths.append(int(end - previous))
            previous = int(end)
        observations = {
            name: value.detach().cpu()
            for name, value in buffer.storage["observation"].items()
            if name in {"state", "last_action", "privileged_state"}
        }
        return cls(observations, list(buffer.file_names), lengths, motion_frames)

    @classmethod
    def load(cls, agent: Any, motion_file: str | Path) -> AlignedSkateTrackingContext:
        motion_file = Path(motion_file).expanduser().resolve()
        records = joblib.load(motion_file)
        if not isinstance(records, dict) or not records:
            raise RuntimeError("Skate expert dataset must be a non-empty motion dictionary.")
        motion_frames = {
            str(name): int(np.asarray(record["dof"]).shape[0])
            for name, record in records.items()
        }
        del records
        return cls.from_buffer(
            load_expert_buffer(agent, motion_file, device="cpu"),
            motion_frames,
        )

    def frame_difference_counts(self) -> dict[int, int]:
        return dict(sorted(Counter(
            item["length"] - item["raw_frames"]
            for item in self.trajectories.values()
        ).items()))

    def eligible_frame_count(self, motion_key: str, future_steps: int = 1) -> int:
        if future_steps <= 0:
            raise ValueError("future_steps must be positive.")
        trajectory = self.trajectories[motion_key]
        return max(
            min(
                trajectory["raw_frames"],
                trajectory["length"] - future_steps,
            ),
            0,
        )

    def encode(
        self,
        model: torch.nn.Module,
        motion_key: str,
        local_frame: int,
        steps: int,
    ) -> tuple[torch.Tensor, list[dict[str, int]]]:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        trajectory = self.trajectories.get(motion_key)
        if trajectory is None:
            raise RuntimeError(f"Reset motion has no expert trajectory: {motion_key}")
        if not 0 <= local_frame < trajectory["raw_frames"]:
            raise ValueError(f"Reset local frame is outside motion {motion_key}.")
        future_start = local_frame + 1
        if future_start >= trajectory["length"]:
            raise RuntimeError("Reset frame has no valid next expert state.")

        future_steps = min(steps, trajectory["length"] - future_start)
        slice_end = min(
            trajectory["length"],
            future_start + future_steps + model.cfg.seq_length - 1,
        )
        start = trajectory["start"]
        next_observation = {
            name: value[start + future_start : start + slice_end].to(model.device)
            for name, value in self.observations.items()
        }
        with torch.no_grad():
            z = model.tracking_inference(next_observation)[:future_steps]
        ranges = [
            {
                "expert_state_index": local_frame + step,
                "future_start": local_frame + step + 1,
                "future_end": min(
                    local_frame + step + model.cfg.seq_length,
                    trajectory["length"] - 1,
                ),
                "future_count": min(
                    model.cfg.seq_length,
                    trajectory["length"] - local_frame - step - 1,
                ),
            }
            for step in range(future_steps)
        ]
        if z.shape != (future_steps, model.cfg.archi.z_dim) or not torch.isfinite(z).all():
            raise RuntimeError("Aligned tracking z has an invalid shape or value.")
        return z, ranges


def select_expert_rollout_envs(
    env_count: int,
    percentage: float,
    seed: int,
) -> tuple[int, ...]:
    selected = env_count * percentage
    if not 0 <= percentage <= 1 or not float(selected).is_integer():
        raise ValueError("Expert rollout percentage must select a whole number of envs.")
    rng = np.random.default_rng(seed)
    return tuple(sorted(
        map(int, rng.choice(env_count, size=int(selected), replace=False))
    ))


class SkateRolloutContext:
    """Per-environment background and reset-aligned tracking latents."""

    def __init__(self, agent: Any, env_count: int, expert_envs: tuple[int, ...]) -> None:
        if len(expert_envs) != len(set(expert_envs)):
            raise ValueError("Expert rollout env indices must be unique.")
        self.agent = agent
        self.expert_envs = frozenset(expert_envs)
        self.background: list[torch.Tensor | None] = [None] * env_count
        self.tracking: list[torch.Tensor | None] = [None] * env_count
        self.had_tracking = [False] * env_count
        self.counters = {
            "background_random_samples": 0,
            "background_zbuffer_samples": 0,
            "expert_tracking_resets": 0,
            "tracking_unavailable_resets": 0,
            "tracking_transitions": 0,
            "free_transitions": 0,
            "post_tracking_free_transitions": 0,
        }
        self.z_norm_sum = 0.0
        self.z_norm_count = 0
        self.z_norm_max = 0.0
        self.tracking_lengths: list[int] = []

    def reset(self, env_index: int, tracking: torch.Tensor | None) -> None:
        if env_index not in self.expert_envs and tracking is not None:
            raise ValueError("Free rollout env cannot receive expert tracking z.")
        self.background[env_index] = None
        self.tracking[env_index] = tracking
        self.had_tracking[env_index] = tracking is not None
        if env_index in self.expert_envs:
            if tracking is None:
                self.counters["tracking_unavailable_resets"] += 1
            else:
                self.counters["expert_tracking_resets"] += 1
                self.tracking_lengths.append(len(tracking))

    def effective_z(self, episode_steps: list[int]) -> torch.Tensor:
        refresh = [
            index
            for index, value in enumerate(self.background)
            if value is None
            or episode_steps[index] % self.agent.cfg.train.update_z_every_step == 0
        ]
        if refresh:
            use_buffer = (
                self.agent.cfg.train.use_mix_rollout
                and not self.agent.z_buffer.empty()
            )
            sampled = (
                self.agent.z_buffer.sample(len(refresh), device=self.agent.device)
                if use_buffer
                else self.agent._model.sample_z(len(refresh), device=self.agent.device)
            )
            counter = (
                "background_zbuffer_samples"
                if use_buffer
                else "background_random_samples"
            )
            self.counters[counter] += len(refresh)
            for offset, index in enumerate(refresh):
                self.background[index] = sampled[offset]

        effective = []
        for index, step in enumerate(episode_steps):
            tracking = self.tracking[index]
            if tracking is not None and step < len(tracking):
                effective.append(tracking[step])
                self.counters["tracking_transitions"] += 1
            else:
                background = self.background[index]
                if background is None:
                    raise RuntimeError("Background rollout z was not initialized.")
                effective.append(background)
                self.counters["free_transitions"] += 1
                if self.had_tracking[index] and tracking is not None:
                    self.counters["post_tracking_free_transitions"] += 1
        batch = torch.stack(effective)
        norms = torch.linalg.vector_norm(batch, dim=1)
        self.z_norm_sum += float(norms.sum().item())
        self.z_norm_count += int(norms.numel())
        self.z_norm_max = max(self.z_norm_max, float(norms.max().item()))
        return batch

    def report(self) -> dict[str, Any]:
        lengths = np.asarray(self.tracking_lengths, dtype=np.float64)
        total = (
            self.counters["tracking_transitions"]
            + self.counters["free_transitions"]
        )
        return {
            "expert_rollout_env_indices": sorted(self.expert_envs),
            "use_mix_rollout": self.agent.cfg.train.use_mix_rollout,
            "update_z_every_step": self.agent.cfg.train.update_z_every_step,
            "z_buffer_size": self.agent.cfg.train.z_buffer_size,
            "expert_percentage": (
                self.agent.cfg.train.rollout_expert_trajectories_percentage
            ),
            "tracking_max_length": (
                self.agent.cfg.train.rollout_expert_trajectories_length
            ),
            **self.counters,
            "tracking_lengths": {
                "min": int(lengths.min()) if lengths.size else None,
                "mean": float(lengths.mean()) if lengths.size else None,
                "median": float(np.median(lengths)) if lengths.size else None,
                "max": int(lengths.max()) if lengths.size else None,
            },
            "realized_tracking_transition_ratio": (
                self.counters["tracking_transitions"] / total if total else 0.0
            ),
            "z_norm": {
                "mean": (
                    self.z_norm_sum / self.z_norm_count
                    if self.z_norm_count else None
                ),
                "max": self.z_norm_max if self.z_norm_count else None,
                "count": self.z_norm_count,
            },
            "z_buffer": {
                "occupancy": len(self.agent.z_buffer),
                "capacity": self.agent.cfg.train.z_buffer_size,
            },
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


def make_native_online_env(cfg: TrainConfig) -> Any:
    """Construct one native backend owning all worlds, without HUSKY projection."""
    sim_src = REPOSITORY_ROOT.parent / "skate-sim" / "src"
    if str(sim_src) not in sys.path:
        sys.path.insert(0, str(sim_src))
    if cfg.online_env_backend not in ("skate-sim-cpu", "skate-sim-mjlab"):
        raise ValueError("Only native29 online backends are supported; legacy23 is retired")
    return SkateSimOnlineEnv(
        backend="mujoco" if cfg.online_env_backend == "skate-sim-cpu" else "mjlab",
        num_envs=cfg.online_envs, preset=cfg.native_reset_preset, seed=cfg.seed,
    )


def load_frozen_agent(checkpoint: Path, *, online_env_backend: str = "skate-sim-mjlab", device: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Load a complete native29 BFM model without initializing a simulator."""
    cfg = build_train_config(SimpleNamespace(
        online_env_backend=online_env_backend, online_envs=1,
        **({"device":device} if device is not None else {}),
    ))
    obs_space = gymnasium.spaces.Dict({
        key:gymnasium.spaces.Box(-np.inf,np.inf,(width,),np.float32)
        for key,width in (("state",64),("privileged_state",463),("last_action",29),("history_actor",372))
    })
    agent=cfg.agent.build(obs_space=obs_space,action_dim=29)
    report=load_bfm_checkpoint(agent,checkpoint)
    agent._model.eval()
    agent._model.requires_grad_(False)
    return agent,report


def load_dataset_report(motion_file: Path, manifest_file: Path, kind: str) -> dict[str, Any]:
    """Validate provenance at the trainer's actual offline-data loading boundary."""
    from skate_bfm.integration.source import SOURCE_MISSING_JOINTS

    manifest = json.loads(manifest_file.read_text())
    records = joblib.load(motion_file)
    if not isinstance(records, dict) or not records:
        raise ValueError("Offline dataset must be a non-empty motion dictionary")

    def supports(value):
        values = value if isinstance(value, list) else [value]
        if not values or any(type(item) is not int or item not in (23, 29) for item in values):
            raise ValueError("source_joint_support must be 23/29 or a nonempty list of these integers")
        if len(set(values)) != len(values):
            raise ValueError("source_joint_support contains duplicates")
        return sorted(values)

    def missing(value, expected):
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("missing_joint_names must be a list of joint names")
        if len(set(value)) != len(value) or set(value) != set(expected):
            raise ValueError("missing_joint_names disagrees with source_joint_support")
        return sorted(value)

    def quality(value):
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError("source_quality must contain nonempty strings")
        return sorted(set(values))

    declared = supports(manifest.get("source_joint_support", 23))
    expected_missing = list(SOURCE_MISSING_JOINTS) if 23 in declared else []
    missing(manifest.get("missing_joint_names", expected_missing), expected_missing)
    manifest_quality = quality(manifest.get("source_quality", "historical_expert_unvalidated"))
    manifest_expert = manifest.get("native29_expert", False)
    if type(manifest_expert) is not bool:
        raise ValueError("native29_expert must be an explicit boolean")
    provenance = {}
    for name, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid offline record {name}")
        fallback = declared[0] if len(declared) == 1 else None
        support = supports(record.get("source_joint_support", fallback))
        if len(support) != 1:
            raise ValueError("Each record must declare one source_joint_support")
        support = support[0]
        absent = list(SOURCE_MISSING_JOINTS) if support == 23 else []
        absent = missing(record.get("missing_joint_names", absent), absent)
        source_quality = record.get("source_quality", manifest_quality[0] if len(manifest_quality) == 1 else None)
        if not isinstance(source_quality, str) or not source_quality.strip():
            raise ValueError("Each record must declare one source_quality")
        expert = record.get("native29_expert", False)
        if type(expert) is not bool or (expert and support != 29):
            raise ValueError("native29_expert must be boolean and cannot certify a 23-joint source")
        provenance[str(name)] = {
            "source_joint_support": support, "missing_joint_names": absent,
            "source_quality": source_quality, "native29_expert": expert,
            "source_raw_npz": record.get("source_raw_npz"),
            "source_round": record.get("source_round"),
            "source_rollout": record.get("source_rollout"),
        }
        if "quality_review" in record:
            provenance[str(name)]["quality_review"] = copy.deepcopy(record["quality_review"])
    actual_support = sorted({item["source_joint_support"] for item in provenance.values()})
    actual_quality = sorted({item["source_quality"] for item in provenance.values()})
    if actual_support != declared:
        raise ValueError("Manifest source_joint_support disagrees with records")
    if actual_quality != manifest_quality:
        raise ValueError("Manifest source_quality disagrees with records")
    fields = {
        "dataset_stage":"dataset_stage", "dataset_type":"dataset_type",
        "motion_count":"total_motion_count", "expert_frames":"expert_frames",
        "expert_minutes":"expert_minutes", "fps":"fps",
    }
    report = {
        "kind":kind, "motion_file":str(motion_file), "manifest_file":str(manifest_file),
        "motion_file_sha256":hash_file(motion_file), "manifest_sha256":hash_file(manifest_file),
        **{out:manifest[key] for out,key in fields.items() if key in manifest},
        "source_joint_support":declared,
        "missing_joint_names":sorted({joint for item in provenance.values() for joint in item["missing_joint_names"]}),
        "source_quality":actual_quality,
        "native29_expert":manifest_expert and actual_support == [29] and all(item["native29_expert"] for item in provenance.values()),
        "record_provenance":provenance,
        "source_rollout_count":len({(str(item["source_round"]),str(item["source_rollout"]),str(item["source_raw_npz"])) for item in provenance.values()}),
    }
    if "quality_review" in manifest:
        report["quality_review"] = copy.deepcopy(manifest["quality_review"])
    missing_fields = [key for key in fields.values() if key not in manifest]
    if missing_fields:
        report["missing_manifest_fields"] = missing_fields
    return report




class Workspace:
    """Project-owned formal Skate-BFM training workspace."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        if cfg.agent.train.rollout_expert_trajectories or cfg.agent.train.rollout_expert_trajectories_percentage != 0:
            raise ValueError("Native29 preset resets do not support aligned historical expert rollouts.")
        self.run_started_at = datetime.now().isoformat(timespec="seconds")
        self.git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
        self.work_dir = Path(cfg.work_dir).expanduser().resolve()
        if (self.work_dir / "summary.json").exists() or (self.work_dir / CHECKPOINT_DIR_NAME).exists():
            raise RuntimeError("Formal Skate-BFM requires a fresh SKATE_WORK_DIR.")
        set_seed_everywhere(cfg.seed)

        expert_path = Path(cfg.skate_expert_motion_file).expanduser().resolve()
        manifest_path = Path(cfg.expert_manifest_file).expanduser().resolve()
        self.dataset_report = load_dataset_report(expert_path, manifest_path, cfg.expert_dataset_kind)

        self.work_dir.mkdir(parents=True, exist_ok=False)
        self.native_env = None
        try:
            self.native_env = make_native_online_env(cfg)
            observation = {key: value[0] for key, value in self.native_env.reset().items()}
            self.online_metadata = {"backend": cfg.online_env_backend, **self.native_env.metadata}
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
            self.expert_rollout_env_indices = ()
            self.rollout_context = SkateRolloutContext(
                self.agent,
                cfg.online_envs,
                self.expert_rollout_env_indices,
            )
            (self.work_dir / "config.json").write_text(cfg.model_dump_json(indent=2) + "\n")
        except BaseException:
            self.close()
            raise


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
        return assemble_expert_replay(expert_base, expert_skate)

    def _build_replay(self) -> dict[str, Any]:
        replay: dict[str, Any] = {}
        register_skate_replay(
            replay, DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        )
        replay.update(self._load_experts())
        # MotionLib's loader sets one thread; restore the training CPU budget.
        torch.set_num_threads(8)
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
        checkpoint_dir = (
            Path(self.cfg.checkpoint_dir).expanduser().resolve()
            / f"{CHECKPOINT_DIR_NAME}_{env_step:05d}"
        )
        if checkpoint_dir.exists():
            raise RuntimeError(f"Checkpoint output already exists: {checkpoint_dir}")
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
            "optimizer_present": (checkpoint_dir / "optimizers.pth").is_file(),
            "normalizer_present": (checkpoint_dir / "model" / "model.safetensors").is_file(),
        }

    def close(self) -> None:
        if self.native_env is not None:
            self.native_env.close()

    def train(self) -> dict[str, Any]:
        """Run training and close every backend even if replay setup fails."""
        try:
            return self._train()
        finally:
            self.close()

    def _train(self) -> dict[str, Any]:
        """Collect online skate transitions and call the existing ``agent.update``."""

        update_steps = training_update_steps(
            self.cfg.skate_max_steps,
            self.cfg.first_update_transition,
            self.cfg.update_interval,
        )
        checkpoint_steps = training_checkpoint_steps(self.cfg.skate_max_steps)
        expected_updates = len(update_steps) * self.cfg.updates_per_block
        dataset = self.dataset_report
        diagnostics_path = self.work_dir / "training_diagnostics.jsonl"
        diagnostics_path.touch()

        def diagnostic(event: str, **payload: Any) -> None:
            record = {
                "event": event,
                "time": datetime.now().isoformat(timespec="seconds"),
                "env_transitions": int(payload.pop("env_transitions", 0)),
                "update_blocks": len(update_blocks),
                "native_updates": self.agent_update_calls,
                "replay_size": int(payload.pop("replay_size", 0)),
                "episodes": episodes,
                "falls": falls,
                **payload,
            }
            with diagnostics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()

        print(
            "\n".join(
                (
                    "=" * 50,
                    "NATIVE29 TRAINING PLAN",
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
                    f"Expert rollout envs: {self.expert_rollout_env_indices}",
                    f"Transitions: {self.cfg.skate_max_steps}",
                    f"Replay capacity: {self.cfg.buffer_size}",
                    f"Warmup: {self.cfg.warmup_transitions}",
                    f"First update: {self.cfg.first_update_transition}",
                    f"Update interval: {self.cfg.update_interval}",
                    f"Updates / block: {self.cfg.updates_per_block}",
                    f"Expected update blocks: {len(update_steps)}",
                    f"Expected native updates: {expected_updates}",
                    f"Checkpoints: {', '.join(map(str, checkpoint_steps))}",
                    f"Checkpoint root: {self.cfg.checkpoint_dir}",
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
        episode_steps = [0] * self.cfg.online_envs
        reset_counts = [0] * self.cfg.online_envs
        initial = self.native_env.reset()
        for index in range(self.cfg.online_envs):
            observations.append({key: value[index] for key, value in initial.items()})
            self.rollout_context.reset(index, None)
            reset_counts[index] += 1
        transition_count = 0
        update_blocks: list[dict[str, Any]] = []
        checkpoints: dict[str, Any] = {}
        actor_hashes = {"A0": hash_params(model._actor)}
        start = 1
        episodes = 0
        falls = 0
        termination_reasons: Counter[str] = Counter()
        diagnostic(
            "run_start",
            env_transitions=0,
            replay_size=0,
            run_name=self.work_dir.name,
            git_head=self.git_head,
            seed=self.cfg.seed,
            device=str(self.agent.device),
            online_envs=self.cfg.online_envs,
            dataset=dataset,
            pretrained_checkpoint=self.agent.pretrained_load_report,
            schedule={
                "total_transitions": self.cfg.skate_max_steps,
                "warmup": self.cfg.warmup_transitions,
                "first_update": self.cfg.first_update_transition,
                "update_interval": self.cfg.update_interval,
                "updates_per_block": self.cfg.updates_per_block,
                "expected_update_blocks": len(update_steps),
                "expected_native_updates": expected_updates,
                "checkpoints": list(checkpoint_steps),
            },
            rollout_latent=self.rollout_context.report(),
            online_environment=self.online_metadata,
            optimizer={
                name: optimizer_step_report(item)
                for name, item in optimizers.items()
            },
        )
        progress = tqdm(
            total=self.cfg.skate_max_steps,
            desc="Native29 training",
            unit="transition",
            dynamic_ncols=True,
        )

        def collect(end: int) -> None:
            nonlocal episodes, falls, start, transition_count
            model.eval()
            while start <= end:
                obs_batch = {
                    key: torch.stack([item[key] for item in observations]).to(self.agent.device)
                    for key in observations[0]
                }
                z_batch = self.rollout_context.effective_z(episode_steps)
                with torch.no_grad():
                    actions = self.agent.act(
                        obs=obs_batch,
                        z=z_batch,
                        mean=False,
                    )
                episode_steps[:] = [step + 1 for step in episode_steps]
                batch = self.native_env.step(
                    actions, z_batch,
                    truncated=[step >= SKATE_EPISODE_HORIZON for step in episode_steps],
                )
                # Transfer once per batch; terminal next observations precede reset.
                replay["train"].extend(tree_map(lambda value: value.detach().cpu(), batch))
                transition_count += self.cfg.online_envs
                terminated_mask = batch["next"]["terminated"].flatten().cpu()
                truncated_mask = batch["next"]["truncated"].flatten().cpu()
                done_ids = (terminated_mask | truncated_mask).nonzero().flatten().tolist()
                next_obs = batch["next"]["observation"]
                for index in range(self.cfg.online_envs):
                    observations[index] = {key: value[index] for key, value in next_obs.items()}
                if done_ids:
                    reset_obs = self.native_env.reset(env_ids=done_ids)
                    for index in done_ids:
                        episodes += 1
                        falls += int(terminated_mask[index])
                        termination_reasons["native_terminated" if terminated_mask[index] else "truncated"] += 1
                        observations[index] = {key: value[index] for key, value in reset_obs.items()}
                        self.rollout_context.reset(index, None)
                        episode_steps[index] = 0
                        reset_counts[index] += 1
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
                diagnostic(
                    "collection_boundary",
                    env_transitions=env_step,
                    replay_size=len(replay["train"]),
                    termination_reason_counts=dict(sorted(termination_reasons.items())),
                    rollout_latent=self.rollout_context.report(),
                    finite_model=module_state_is_finite(model),
                )
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
                    diagnostic(
                        "update_block",
                        env_transitions=env_step,
                        replay_size=len(replay["train"]),
                        block_index=block_index,
                        metric_summary=metric_summary,
                        expert_contract=expert_contract,
                        rollout_latent=self.rollout_context.report(),
                        finite_model=module_state_is_finite(model),
                        optimizer={
                            name: optimizer_step_report(item)
                            for name, item in optimizers.items()
                        },
                    )
                if env_step in checkpoint_steps:
                    checkpoints[str(env_step)] = self._save_checkpoint(replay, env_step)
                    report = checkpoints[str(env_step)]
                    progress.write(
                        f"[Checkpoint] step={env_step}, path={report['path']}, "
                        f"model reload={report['reload']}, "
                        f"optimizer step={report['optimizer_step']}"
                    )
                    diagnostic(
                        "checkpoint",
                        env_transitions=env_step,
                        replay_size=len(replay["train"]),
                        checkpoint=report,
                        rollout_latent=self.rollout_context.report(),
                    )
        except BaseException as exc:
            failure = {
                "event": "failure",
                "time": datetime.now().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "env_transitions": min(
                    self.cfg.skate_max_steps,
                    max(0, start - 1),
                ),
                "update_blocks": len(update_blocks),
                "native_updates": self.agent_update_calls,
                "replay_size": len(replay["train"]),
                "episodes": episodes,
                "falls": falls,
                "termination_reason_counts": dict(sorted(termination_reasons.items())),
                "rollout_latent": self.rollout_context.report(),
                "optimizer": {
                    name: optimizer_step_report(item)
                    for name, item in optimizers.items()
                },
                "finite_model": module_state_is_finite(model),
            }
            with diagnostics_path.open("a") as handle:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")
                handle.flush()
            (self.work_dir / "training_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n"
            )
            (self.work_dir / "training_failure.md").write_text(
                "\n".join(
                    (
                        "# Skate-BFM Training Failure",
                        "",
                        f"- Error: `{type(exc).__name__}: {exc}`",
                        f"- Transitions: `{failure['env_transitions']}`",
                        f"- Update blocks / native updates: "
                        f"`{len(update_blocks)}` / `{self.agent_update_calls}`",
                        f"- Replay size: `{len(replay['train'])}`",
                        f"- Diagnostics: `{diagnostics_path}`",
                        "",
                    )
                )
            )
            raise
        finally:
            progress.close()

        full_replay = replay["train"].get_full_buffer()
        terminated = full_replay["next"]["terminated"]
        truncated = full_replay["next"]["truncated"]
        replay_size = self.cfg.skate_max_steps
        if transition_count != replay_size or len(replay["train"]) != replay_size:
            raise RuntimeError("Formal Skate-BFM replay length mismatch.")
        if self.agent_update_calls != expected_updates:
            raise RuntimeError("Formal Skate-BFM native update count mismatch.")
        if (
            tuple(full_replay["action"].shape) != (replay_size, 29)
            or tuple(full_replay["z"].shape) != (replay_size, 256)
        ):
            raise RuntimeError("Formal Skate-BFM replay action/latent schema mismatch.")
        for observations_key in (full_replay["observation"], full_replay["next"]["observation"]):
            for key, size in (("state", 64), ("privileged_state", 463), ("last_action", 29), ("history_actor", 372)):
                value = observations_key[key]
                if tuple(value.shape) != (replay_size, size) or not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"Online replay observation contract mismatch: {key}.")
        if tuple(terminated.shape) != (replay_size, 1) or tuple(truncated.shape) != (replay_size, 1):
            raise RuntimeError("Online replay terminal masks must have shape [transitions, 1].")
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
            "milestone": "Native29 Skate-BFM Training",
            "run_provenance": {
                "run_name": self.work_dir.name,
                "started_at": self.run_started_at,
                "git_head": self.git_head,
                "seed": self.cfg.seed,
                "num_envs": self.cfg.online_envs,
                "device": str(self.agent.device),
                "work_dir": str(self.work_dir),
                "checkpoint_dir": str(Path(self.cfg.checkpoint_dir).expanduser().resolve()),
                "fresh_restart": True,
                "resumed": False,
                "val_used_for_training": False,
                "test_used_for_training": False,
            },
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
                "physics_mode": "native_model_preset",
                "source_physics_aligned": False,
                "online_environment": self.online_metadata,
            },
            "online_reset": {
                "mode": "native_preset",
                "metadata": self.online_metadata,
                "distribution_changed_from_legacy": True,
                "aligned_online_expert_tracking": False,
                "dataset": self.cfg.expert_dataset_kind,
                "total_resets": sum(reset_counts),
                "uniform_motion_uniform_local_frame": False,
            },
            "episodes": {
                "episodes": episodes,
                "resets": sum(reset_counts),
                "falls": falls,
                "truncations": int(truncated.sum()),
                "termination_reason_counts": dict(sorted(termination_reasons.items())),
            },
            "rollout_latent": self.rollout_context.report(),
            "replay": {
                "capacity": self.cfg.buffer_size,
                "final_size": len(replay["train"]),
                "train_is_train_skate": replay["train"] is replay["train_skate"],
                "terminated_count": int(terminated.sum()),
                "truncated_count": int(truncated.sum()),
                "normal_count": int((~(terminated | truncated)).sum()),
                "termination_reason_counts": dict(sorted(termination_reasons.items())),
            },
            "expert": {
                "base_skate_ratio": [0.5, 0.5],
                "tracking_source": "expert_skate",
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
            "next_milestone": "Native29 Phase/Continuous 100k Training",
        }
        summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        (self.work_dir / "summary.json").write_text(summary_text)
        (self.work_dir / "training_summary.json").write_text(summary_text)
        (self.work_dir / "training_summary.md").write_text(
            "\n".join(
                (
                    "# Skate-BFM Training Summary",
                    "",
                    f"- Run: `{self.work_dir.name}`",
                    f"- Git HEAD: `{self.git_head}`",
                    f"- Dataset: `{dataset['kind']}` ({dataset['motion_count']} motions, "
                    f"{dataset['source_rollout_count']} source rollouts)",
                    f"- Transitions: `{self.cfg.skate_max_steps}`",
                    f"- Update blocks / native updates: `{len(update_steps)}` / `{expected_updates}`",
                    f"- Checkpoints: `{', '.join(map(str, checkpoint_steps))}`",
                    f"- Replay: `{len(replay['train'])}`; train aliases train_skate: "
                    f"`{replay['train'] is replay['train_skate']}`",
                    f"- Falls / truncations: `{int(terminated.sum())}` / `{int(truncated.sum())}`",
                    f"- Evaluation during training: `NO` (Val/Test not used)",
                    "",
                )
            )
        )
        print(
            "Formal Skate-BFM complete: "
            f"{transition_count} transitions, {self.agent_update_calls} updates"
        )
        return summary


def build_train_config(args: argparse.Namespace | None = None) -> TrainConfig:
    """Build the formal Skate-BFM configuration."""

    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentTrainConfig
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from humanoidverse.agents.nn_filters import DictInputFilterConfig
    from humanoidverse.agents.nn_models import (
        ActorArchiConfig, BackwardArchiConfig, DiscriminatorArchiConfig,
        ForwardArchiConfig, RewardNormalizerConfig,
    )
    from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig

    def value(name: str, environment_name: str, default: Any) -> Any:
        explicit = getattr(args, name, None) if args is not None else None
        return (
            explicit
            if explicit is not None
            else os.environ.get(environment_name, default)
        )

    dataset_kind, expert_path, manifest_path = resolve_expert_dataset(
        value("dataset", "SKATE_EXPERT_DATASET", "phase"),
        value("expert_motion", "SKATE_EXPERT_MOTION_FILE", None),
    )
    max_steps = int(value("max_steps", "SKATE_MAX_STEPS", DEFAULT_MAX_STEPS))
    seed = int(value("seed", "SKATE_SEED", 4728))
    budget = f"{max_steps // 1000}k" if max_steps % 1000 == 0 else str(max_steps)
    TrainConfig.model_rebuild(
        _types_namespace={
            "FBcprAuxAgentConfig": FBcprAuxAgentConfig,
            "HumanoidVerseIsaacConfig": HumanoidVerseIsaacConfig,
        }
    )
    online_backend = str(value("online_env_backend", "SKATE_ONLINE_ENV_BACKEND", "skate-sim-mjlab"))
    training_device = str(value("device", "SKATE_TRAIN_DEVICE", "cuda"))
    return TrainConfig(
        name="TrainConfig",
        online_env_backend=online_backend,
        native_reset_preset=str(value("native_reset_preset", "SKATE_NATIVE_RESET_PRESET", "steer_start")),
        aligned_online_tracking=bool(getattr(args, "aligned_online_tracking", False)),
        agent=FBcprAuxAgentConfig(
            name="FBcprAuxAgent",
            model=FBcprAuxModelConfig(
                name="FBcprAuxModel", device=training_device,
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
                rollout_expert_trajectories=False, rollout_expert_trajectories_length=250,
                rollout_expert_trajectories_percentage=0.0, lr_discriminator=1e-05,
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
            name="humanoidverse_isaac", device="cuda:0" if training_device == "cuda" else training_device,
            lafan_tail_path=str(BASE_MOTION_FILE),
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
        pretrained_checkpoint=str(
            value(
                "pretrained_checkpoint",
                "BFM0_PRETRAINED_CHECKPOINT",
                REPOSITORY_ROOT / "model/bfm-zero-official",
            )
        ),
        work_dir=str(
            value(
                "work_dir",
                "SKATE_WORK_DIR",
                REPOSITORY_ROOT / f"results/m2.6-{dataset_kind}-{budget}-seed{seed}",
            )
        ),
        checkpoint_dir=str(
            value(
                "checkpoint_dir",
                "SKATE_CHECKPOINT_DIR",
                REPOSITORY_ROOT
                / "model/motion_library"
                / datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            )
        ),
        seed=seed,
        skate_max_steps=max_steps,
        warmup_transitions=int(
            value(
                "warmup_transitions",
                "SKATE_WARMUP_TRANSITIONS",
                DEFAULT_WARMUP_TRANSITIONS,
            )
        ),
        first_update_transition=int(
            value(
                "first_update",
                "SKATE_FIRST_UPDATE",
                DEFAULT_FIRST_UPDATE,
            )
        ),
        update_interval=int(
            value(
                "update_interval",
                "SKATE_UPDATE_INTERVAL",
                DEFAULT_UPDATE_INTERVAL,
            )
        ),
        updates_per_block=int(
            value(
                "updates_per_block",
                "SKATE_UPDATES_PER_BLOCK",
                DEFAULT_UPDATES_PER_BLOCK,
            )
        ),
        online_envs=int(value("online_envs", "SKATE_ONLINE_ENVS", 4)),
        skate_expert_ratio=float(
            value("expert_ratio", "SKATE_EXPERT_RATIO", 0.5)
        ),
        buffer_size=int(
            value("buffer_size", "SKATE_BUFFER_SIZE", max_steps)
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPERT_DATASETS), default="phase")
    parser.add_argument("--expert-motion", type=Path)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--warmup-transitions", type=int)
    parser.add_argument("--first-update", type=int)
    parser.add_argument("--update-interval", type=int)
    parser.add_argument("--updates-per-block", type=int)
    parser.add_argument("--online-envs", type=int)
    parser.add_argument("--online-env-backend", choices=("skate-sim-cpu", "skate-sim-mjlab"))
    parser.add_argument("--native-reset-preset", choices=("steer_start", "push_start"))
    parser.add_argument("--aligned-online-tracking", action="store_true", help="Require motion/frame aligned online tracking (unsupported by native29).")
    parser.add_argument("--device", help="BFM training and offline MotionLib device (default cuda).")
    parser.add_argument("--expert-ratio", type=float)
    parser.add_argument("--buffer-size", type=int)
    return parser.parse_args()


def main() -> int:
    build_train_config(parse_args()).build().train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
