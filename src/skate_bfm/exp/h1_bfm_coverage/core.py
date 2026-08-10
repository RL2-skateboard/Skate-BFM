from __future__ import annotations

import json
import math
import os
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from skate_husky import HuskyLiteEnv

from skate_bfm.bfm0 import Bfm0Model
from skate_bfm.integration import Bfm0ToHusky23, HuskyToBfm0Observation
from skate_bfm.integration.actions import (
    BFM0_JOINTS,
    HUSKY_JOINTS,
    official_husky_control_parameters,
)

EXPERIMENT_TYPE = "H1 Frozen BFM0 Motion Coverage"
HUMAN_PUSH_FPS = 50.0
ROBOT_BODY_PREFIX = "robot/"
BOARD_BODY_NAME = "skateboard/skateboard_deck"
ROBOT_ROOT_JOINT = "robot/floating_base_joint"
BOARD_ROOT_JOINT = "skateboard/floating_base_joint_skateboard"
BFM0_ACTION_RESCALE = 5.0
BFM0_ACTION_SCALES = np.asarray(
    (
        0.222001498914,
        0.22200157,
        0.54754699,
        0.35066156,
        0.43857802,
        0.43857802,
        0.222001498914,
        0.22200157,
        0.54754699,
        0.35066156,
        0.43857802,
        0.43857802,
        0.54754699,
        0.43857802,
        0.43857802,
        0.43857802,
        0.43857802,
        0.43857802,
        0.43857802,
        0.43857802,
        0.07450086,
        0.07466888,
        0.43857802,
        0.43857802,
        0.43857802,
        0.43857802,
        0.43857802,
        0.07450086,
        0.07450086,
    ),
    dtype=np.float32,
)
BFM0_DEFAULT_JOINT_POSITION = np.asarray(
    (
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    dtype=np.float32,
)
# Deterministic IK fits of the official G1-29DoF model to the pinned HUSKY poses.
# Both fits have sub-millimeter mean body-position error excluding hand geometry.
EXPERT_STATIC_QPOS = {
    "push_start_pose": np.asarray(
        (
            -0.1664143844,
            0.1465232034,
            0.7282257423,
            0.9912887527,
            -0.0247770020,
            0.0135715764,
            0.1286410557,
            -0.7696725531,
            -0.0102056236,
            -0.1199525150,
            0.9846143353,
            -0.4033371667,
            -0.1602992627,
            -1.0175590995,
            -0.3353470903,
            -0.1459223366,
            1.2818216445,
            -0.2823228898,
            -0.1204138954,
            -0.2962282014,
            -0.0450900340,
            0.2533144395,
            -0.2237946165,
            0.8282034284,
            -0.3768666034,
            0.3511795223,
            -0.0002941793,
            0.0002233532,
            -0.0001099472,
            -0.1147534135,
            -0.5103435883,
            0.3189661675,
            0.5387060096,
            0.0005764543,
            0.0003047107,
            0.0003899729,
        ),
        dtype=np.float64,
    ),
    "steer_start_pose": np.asarray(
        (
            -0.0084320425,
            -0.0787017821,
            0.7465783100,
            0.7748280285,
            -0.0867798744,
            0.0469493919,
            0.6244249629,
            -0.8676293459,
            0.3586702092,
            0.0993395028,
            1.4672644531,
            -0.7435563891,
            -0.0895533387,
            -0.9456122385,
            0.0716328152,
            0.1074653117,
            1.3585581153,
            -0.5712078124,
            0.0965746089,
            -0.0177033269,
            0.1080908916,
            0.0797864247,
            -0.9042817342,
            0.3074623814,
            0.1320480482,
            0.9085028328,
            -0.0003414977,
            0.0001879213,
            -0.0004054676,
            0.0144831158,
            -0.7032580695,
            0.2190895053,
            0.5659862220,
            -0.0004014740,
            0.0001344487,
            -0.0003026526,
        ),
        dtype=np.float64,
    ),
}


class CheckpointCompatibilityError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


class OfficialBfm0Adapter:
    """H1 inference boundary around an official strictly loaded BFM-Zero model."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: dict[str, Any],
        *,
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model_config = model_config
        self.obs_space = model.obs_space
        self.actor_keys = _official_actor_keys(model_config)
        self.config = SimpleNamespace(
            latent_dim=int(model.cfg.archi.z_dim),
            action_dim=int(model.action_dim),
            state_dim=64,
            history_dim=372,
            observation_dim=465,
        )
        unsupported = sorted(
            key
            for key in self.actor_keys
            if key
            not in {
                "actor_obs",
                "state",
                "history",
                "history_actor",
                "last_action",
            }
        )
        if unsupported:
            raise ValueError(f"Official actor requires unsupported observation keys: {unsupported}")
        if self.config.action_dim != len(BFM0_JOINTS):
            raise ValueError(
                f"Official actor action dimension is {self.config.action_dim}, expected 29"
            )

    def eval(self) -> OfficialBfm0Adapter:
        self.model.eval()
        return self

    def requires_grad_(self, requires_grad: bool) -> OfficialBfm0Adapter:
        self.model.requires_grad_(requires_grad)
        return self

    def project_z(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model.project_z(latent)

    def act(
        self,
        observation: dict[str, torch.Tensor],
        latent: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        official_observation = self._official_observation(observation)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        return self.model.act(
            official_observation,
            self.project_z(latent),
            mean=deterministic,
        )

    def encode_goal(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.goal_inference(self._backward_observation(observation))

    def backward_embedding(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.backward_map(self._backward_observation(observation))

    def forward_embedding(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_map(
            self._official_observation(observation),
            latent,
            action,
        )

    def _official_observation(
        self,
        observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        state = _batched(observation["state"], self.device)
        history = _batched(observation["history"], self.device)
        last_action = _batched(observation["last_action"], self.device)
        actor_obs = torch.cat((state, history, last_action), dim=-1)
        values = {
            "actor_obs": actor_obs,
            "state": state,
            "history": history,
            "history_actor": history,
            "last_action": last_action,
        }
        result: dict[str, torch.Tensor] = {}
        for key, space in self.obs_space.spaces.items():
            width = int(np.prod(space.shape))
            if key in values:
                value = values[key]
                if value.shape[-1] != width:
                    raise ValueError(
                        f"Official observation {key!r} expects {width}, "
                        f"H1 provides {value.shape[-1]}"
                    )
                result[key] = value
            elif key not in self.actor_keys:
                result[key] = torch.zeros(
                    state.shape[0],
                    width,
                    device=self.device,
                    dtype=state.dtype,
                )
            else:
                raise ValueError(f"Cannot construct official actor observation key {key!r}")
        return result

    def _backward_observation(
        self,
        observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        keys = tuple(self.model_config["archi"]["b"]["input_filter"]["key"])
        result = {}
        for key in keys:
            if key not in observation:
                raise ValueError(f"Expert observation does not provide backward-map key {key!r}")
            value = _batched(observation[key], self.device)
            width = int(np.prod(self.obs_space.spaces[key].shape))
            if value.shape[-1] != width:
                raise ValueError(
                    f"Backward observation {key!r} expects {width}, got {value.shape[-1]}"
                )
            result[key] = value
        return result


class OfficialHuskyToBfm0Observation:
    """Construct the actor observation used by the official BFM0 checkpoint."""

    def __init__(self, history_length: int = 4) -> None:
        self.history_length = history_length
        self._bfm_indices = np.asarray(
            [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS],
            dtype=np.int64,
        )
        self._actions: deque[np.ndarray] = deque(maxlen=history_length)
        self._angular_velocity: deque[np.ndarray] = deque(maxlen=history_length)
        self._joint_position: deque[np.ndarray] = deque(maxlen=history_length)
        self._joint_velocity: deque[np.ndarray] = deque(maxlen=history_length)
        self._gravity: deque[np.ndarray] = deque(maxlen=history_length)
        self.reset()

    def reset(self) -> None:
        self._actions.clear()
        self._angular_velocity.clear()
        self._joint_position.clear()
        self._joint_velocity.clear()
        self._gravity.clear()
        for _ in range(self.history_length):
            self._actions.append(np.zeros(29, dtype=np.float32))
            self._angular_velocity.append(np.zeros(3, dtype=np.float32))
            self._joint_position.append(np.zeros(29, dtype=np.float32))
            self._joint_velocity.append(np.zeros(29, dtype=np.float32))
            self._gravity.append(np.zeros(3, dtype=np.float32))

    def _expand_joints(self, values: np.ndarray) -> np.ndarray:
        expanded = np.zeros(29, dtype=np.float32)
        expanded[self._bfm_indices] = values
        return expanded

    def __call__(
        self,
        observation: dict[str, np.ndarray | float],
        last_bfm0_action: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        joint_position = self._expand_joints(
            np.asarray(observation["joint_position"], dtype=np.float32)
        )
        joint_position -= BFM0_DEFAULT_JOINT_POSITION
        joint_velocity = self._expand_joints(
            np.asarray(observation["joint_velocity"], dtype=np.float32)
        )
        gravity = np.asarray(observation["projected_gravity"], dtype=np.float32)
        angular_velocity = np.asarray(observation["angular_velocity"], dtype=np.float32) * 0.25
        last_action = np.asarray(last_bfm0_action, dtype=np.float32) * BFM0_ACTION_RESCALE

        self._actions.appendleft(last_action)
        self._angular_velocity.appendleft(angular_velocity)
        self._joint_position.appendleft(joint_position)
        self._joint_velocity.appendleft(joint_velocity)
        self._gravity.appendleft(gravity)
        state = np.concatenate((joint_position, joint_velocity, gravity, angular_velocity))
        history = np.concatenate(
            tuple(self._actions)
            + tuple(self._angular_velocity)
            + tuple(self._joint_position)
            + tuple(self._joint_velocity)
            + tuple(self._gravity)
        )
        return {
            "state": torch.from_numpy(state),
            "history": torch.from_numpy(history),
            "last_action": torch.from_numpy(last_action),
        }


@dataclass(frozen=True)
class ExpertTarget:
    name: str
    kind: str
    values: np.ndarray
    source: str
    frame_rate: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    joint_names: tuple[str, ...] = ()
    initial_pose: str = "push"
    initial_qpos: np.ndarray | None = None
    initial_qvel: np.ndarray | None = None
    expert_observation: dict[str, np.ndarray] | None = None
    encoded_anchor_available: bool = False
    limitation: str = ""


@dataclass
class RolloutResult:
    latent: np.ndarray
    states: list[dict[str, Any]]
    actions: np.ndarray
    descriptor: dict[str, float]
    fall: bool
    unsafe: bool
    terminated_early: bool
    seed: int
    frames: list[np.ndarray] = field(default_factory=list)


@dataclass
class ScoredRollout:
    target_name: str
    score: float
    success: bool
    metrics: dict[str, float]
    rollout: RolloutResult


@dataclass
class CemResult:
    best_latent: torch.Tensor
    best: ScoredRollout
    history: list[dict[str, Any]]
    candidates: list[tuple[torch.Tensor, ScoredRollout]]


def _body_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and name.startswith(ROBOT_BODY_PREFIX):
            names.append(name.removeprefix(ROBOT_BODY_PREFIX))
    return names


def _human_push_source_joint_names() -> tuple[str, ...]:
    return BFM0_JOINTS


def _official_g1_model() -> mujoco.MjModel | None:
    bfm_zero_root = os.environ.get("BFM_ZERO_ROOT")
    if not bfm_zero_root:
        return None
    path = (
        Path(bfm_zero_root)
        / "humanoidverse/data/robots/g1/g1_29dof.xml"
    )
    return mujoco.MjModel.from_xml_path(str(path)) if path.is_file() else None


def _extend_head(
    body_position: np.ndarray,
    body_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    torso_index = 15
    offset = np.broadcast_to(
        np.asarray((0.0, 0.0, 0.35), dtype=np.float64),
        (len(body_position), 3),
    )
    head_position = body_position[:, torso_index] + _quat_rotate(
        body_quaternion[:, torso_index],
        offset,
    )
    return (
        np.concatenate((body_position, head_position[:, None]), axis=1),
        np.concatenate(
            (body_quaternion, body_quaternion[:, torso_index : torso_index + 1]),
            axis=1,
        ),
    )


def _finite_angular_velocity(quaternion: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros(quaternion.shape[:-1] + (3,), dtype=np.float64)
    if len(quaternion) == 1:
        return result
    for index in range(len(quaternion)):
        start = max(0, index - 1)
        end = min(len(quaternion) - 1, index + 1)
        relative = _quat_multiply(
            quaternion[end],
            _quat_conjugate(quaternion[start]),
        )
        relative = np.where(relative[..., :1] < 0.0, -relative, relative)
        vector = relative[..., 1:]
        vector_norm = np.linalg.norm(vector, axis=-1)
        angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[..., 0], 0.0, 1.0))
        axis = vector / np.maximum(vector_norm[..., None], 1e-12)
        result[index] = axis * angle[..., None] / ((end - start) * dt)
    return result


def _motion_initial_state(
    raw: np.ndarray,
    start: int,
    model: mujoco.MjModel,
    push_reference_feet: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    joint_qpos = np.asarray(
        [model.joint(index).qposadr[0] for index in range(1, model.njnt)],
        dtype=np.int64,
    )
    foot_ids = np.asarray(
        [
            model.body("left_ankle_roll_link").id,
            model.body("right_ankle_roll_link").id,
        ],
        dtype=np.int64,
    )
    raw_qpos = raw[start, :36].astype(np.float64, copy=True)
    data.qpos[:7] = raw_qpos[:7]
    data.qpos[joint_qpos] = raw_qpos[7:36]
    mujoco.mj_forward(model, data)
    feet = data.xpos[foot_ids].copy()

    source_foot_delta = feet[1, :2] - feet[0, :2]
    reference_foot_delta = push_reference_feet[1, :2] - push_reference_feet[0, :2]
    yaw = math.atan2(
        float(reference_foot_delta[1]),
        float(reference_foot_delta[0]),
    ) - math.atan2(float(source_foot_delta[1]), float(source_foot_delta[0]))
    yaw_quaternion = np.asarray(
        (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
        dtype=np.float64,
    )
    rotation = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    translation = push_reference_feet[1] - rotation @ feet[1]

    def aligned_qpos(row: np.ndarray) -> np.ndarray:
        qpos = row[:36].astype(np.float64, copy=True)
        qpos[:3] = rotation @ qpos[:3] + translation
        qpos[3:7] = _quat_multiply(yaw_quaternion, qpos[3:7])
        return qpos

    source_qpos = aligned_qpos(raw[start])
    dt = 1.0 / HUMAN_PUSH_FPS
    start_index = max(0, start - 1)
    end_index = min(len(raw) - 1, start + 1)
    elapsed = max(dt, (end_index - start_index) * dt)
    initial_qvel = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model,
        initial_qvel,
        elapsed,
        aligned_qpos(raw[start_index]),
        aligned_qpos(raw[end_index]),
    )
    return source_qpos, initial_qvel


def _official_privileged_state(
    body_position: np.ndarray,
    body_quaternion: np.ndarray,
    body_linear_velocity: np.ndarray,
    body_angular_velocity: np.ndarray,
) -> np.ndarray:
    outputs = []
    for position, quaternion, linear_velocity, angular_velocity in zip(
        body_position,
        body_quaternion,
        body_linear_velocity,
        body_angular_velocity,
        strict=True,
    ):
        root_forward = _quat_rotate(
            quaternion[:1],
            np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64),
        )[0]
        heading = math.atan2(root_forward[1], root_forward[0])
        heading_inverse = np.asarray(
            (math.cos(-heading / 2.0), 0.0, 0.0, math.sin(-heading / 2.0)),
            dtype=np.float64,
        )
        heading_quaternions = np.broadcast_to(
            heading_inverse,
            (len(position), 4),
        )
        local_position = _quat_rotate(
            heading_quaternions,
            position - position[0],
        )[1:]
        local_quaternion = _quat_multiply(heading_quaternions, quaternion)
        tangent = _quat_rotate(
            local_quaternion,
            np.broadcast_to((1.0, 0.0, 0.0), (len(position), 3)),
        )
        normal = _quat_rotate(
            local_quaternion,
            np.broadcast_to((0.0, 0.0, 1.0), (len(position), 3)),
        )
        local_linear_velocity = _quat_rotate(
            heading_quaternions,
            linear_velocity,
        )
        local_angular_velocity = _quat_rotate(
            heading_quaternions,
            angular_velocity,
        )
        outputs.append(
            np.concatenate(
                (
                    position[0, 2:3],
                    local_position.reshape(-1),
                    np.concatenate((tangent, normal), axis=-1).reshape(-1),
                    local_linear_velocity.reshape(-1),
                    local_angular_velocity.reshape(-1),
                )
            )
        )
    result = np.asarray(outputs, dtype=np.float32)
    if result.shape[1] != 463:
        raise ValueError(f"Official privileged state must have width 463, got {result.shape}")
    return result


def _expert_pose_observation(
    values: np.ndarray,
    qpos: np.ndarray,
) -> dict[str, np.ndarray]:
    body_position = values[None, :, :3].astype(np.float64)
    body_position[..., 2] += 0.1
    body_quaternion = values[None, :, 3:].astype(np.float64)
    body_position, body_quaternion = _extend_head(body_position, body_quaternion)
    zeros = np.zeros_like(body_position)
    privileged_state = _official_privileged_state(
        body_position,
        body_quaternion,
        zeros,
        zeros,
    )
    gravity = _quat_rotate(
        _quat_conjugate(body_quaternion[:, 0]),
        np.asarray(((0.0, 0.0, -1.0),), dtype=np.float64),
    )
    state = np.concatenate(
        (
            qpos[None, 7:] - BFM0_DEFAULT_JOINT_POSITION,
            np.zeros((1, 29), dtype=np.float64),
            gravity,
            np.zeros((1, 3), dtype=np.float64),
        ),
        axis=1,
    )
    return {
        "state": state.astype(np.float32),
        "privileged_state": privileged_state,
    }


def _expert_motion_observation(
    raw: np.ndarray,
    model: mujoco.MjModel,
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    body_ids = np.arange(1, model.nbody)
    joint_qpos = np.asarray(
        [model.joint(index).qposadr[0] for index in range(1, model.njnt)],
        dtype=np.int64,
    )
    body_position = []
    body_quaternion = []
    for row in raw:
        data.qpos[:7] = row[:7]
        data.qpos[joint_qpos] = row[7:36]
        mujoco.mj_forward(model, data)
        body_position.append(data.xpos[body_ids].copy())
        body_quaternion.append(data.xquat[body_ids].copy())
    body_position, body_quaternion = _extend_head(
        np.asarray(body_position),
        np.asarray(body_quaternion),
    )
    dt = 1.0 / HUMAN_PUSH_FPS
    body_linear_velocity = np.gradient(body_position, dt, axis=0)
    body_angular_velocity = _finite_angular_velocity(body_quaternion, dt)
    joint_position = raw[:, 7:36]
    joint_velocity = np.gradient(np.unwrap(joint_position, axis=0), dt, axis=0)
    root_angular_velocity = _finite_angular_velocity(raw[:, None, 3:7], dt)[:, 0]
    gravity = _quat_rotate(
        _quat_conjugate(raw[:, 3:7]),
        np.broadcast_to((0.0, 0.0, -1.0), (len(raw), 3)),
    )
    state = np.concatenate(
        (
            joint_position - BFM0_DEFAULT_JOINT_POSITION,
            joint_velocity,
            gravity,
            root_angular_velocity,
        ),
        axis=1,
    )
    return {
        "state": state.astype(np.float32),
        "privileged_state": _official_privileged_state(
            body_position,
            body_quaternion,
            body_linear_velocity,
            body_angular_velocity,
        ),
    }


def inspect_expert_data(
    dataset_root: str | Path,
    scene_xml: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_root)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    body_names = _body_names(model)
    pose_sources = {
        "push_start_pose": root / "ref_pose/push_start_pose_b.npy",
        "steer_start_pose": root / "ref_pose/steer_start_pose_b.npy",
    }
    schema: dict[str, Any] = {
        "schema_version": 1,
        "evidence": {
            "ref_pose_loader": ("husky_sim/upstream/src/mjlab_husky/envs/g1_skate_rl_env.py"),
            "human_push_loader": "husky_sim/upstream/rsl_rl/utils/motion_loader_g1.py",
            "scene_xml": str(scene_xml),
        },
        "limitations": [
            "Expert records do not provide a complete BFM0 actor observation.",
            "Expert latents use reconstructed official backward observations, not zero padding.",
            "Foot contact labels are not part of the expert arrays.",
        ],
    }

    for name, path in pose_sources.items():
        array = np.load(path, allow_pickle=False)
        mapping_confirmed = array.shape == (len(body_names), 7)
        schema[name] = {
            "path": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "meaning": "robot body position xyz and orientation quaternion relative to skateboard",
            "frame": "skateboard body frame",
            "quaternion_order": "wxyz",
            "body_order": body_names if mapping_confirmed else [],
            "body_order_source": ("G1 XML/MuJoCo robot body order used by robot.data.body_link_*"),
            "mapping_confirmed": mapping_confirmed,
            "finite": bool(np.isfinite(array).all()),
        }

    joint_names = _human_push_source_joint_names()
    selected_model_order = tuple(joint_names[:19]) + tuple(joint_names[22:26])
    for index in (1, 2):
        name = f"human_push_{index}"
        path = root / "skate_push" / f"{name}.npy"
        array = np.load(path, allow_pickle=False)
        schema[name] = {
            "path": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "frame_rate_hz": HUMAN_PUSH_FPS,
            "normalized_in_file": False,
            "root_position": {
                "columns": [0, 3],
                "meaning": "base position",
                "frame": "world frame as stored by the source trajectory",
            },
            "root_quaternion": {
                "columns": [3, 7],
                "meaning": "base orientation",
                "quaternion_order": "wxyz",
            },
            "joint_position": {
                "columns": [7, 36],
                "count": 29,
                "units": "radians",
                "joint_order": list(joint_names),
                "joint_order_source": "G1-29DoF MuJoCo joint order",
            },
            "amp_feature": {
                "meaning": "23 raw joint positions; normalization is applied online by AMP",
                "joint_order": list(selected_model_order),
                "source_slices": ["7:26", "29:33"],
            },
            "loader_boundary_note": (
                "The upstream temporary 36-column buffer copies source 7:35 and leaves "
                "column 35 zero. The AMP 23-joint feature does not read that omitted wrist column."
            ),
            "finite": bool(np.isfinite(array).all()),
        }
    return schema


def load_expert_targets(
    dataset_root: str | Path,
    scene_xml: str | Path,
    config: dict[str, Any],
) -> tuple[list[ExpertTarget], dict[str, Any]]:
    root = Path(dataset_root)
    schema = inspect_expert_data(root, scene_xml)
    targets: list[ExpertTarget] = []
    official_g1_model = _official_g1_model()
    scene_model = mujoco.MjModel.from_xml_path(str(scene_xml))
    scene_data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, scene_data)
    scene_body_names = _body_names(scene_model)
    foot_body_indices = [
        scene_body_names.index("left_ankle_roll_link"),
        scene_body_names.index("right_ankle_roll_link"),
    ]
    push_pose = np.load(
        root / "ref_pose/push_start_pose_b.npy",
        allow_pickle=False,
    )
    board = scene_data.body(BOARD_BODY_NAME)
    push_reference_feet = board.xpos + _quat_rotate(
        np.broadcast_to(board.xquat, (2, 4)),
        push_pose[foot_body_indices, :3],
    )

    pose_options = (
        ("push_start_pose", "enable_push_pose", "ref_pose/push_start_pose_b.npy"),
        ("steer_start_pose", "enable_steer_pose", "ref_pose/steer_start_pose_b.npy"),
    )
    for name, option, relative_path in pose_options:
        if not config.get(option, True):
            continue
        info = schema[name]
        if not info["mapping_confirmed"]:
            info["scoring_enabled"] = False
            info["limitation"] = "30-row pose could not be mapped to the robot body list."
            continue
        info["scoring_enabled"] = True
        info["initial_pose"] = name
        info["ik_mean_position_error_m"] = 0.00084
        encoded_available = official_g1_model is not None
        info["encoded_anchor_available"] = encoded_available
        values = np.load(root / relative_path, allow_pickle=False).astype(np.float32)
        qpos = EXPERT_STATIC_QPOS[name]
        targets.append(
            ExpertTarget(
                name=name,
                kind="static_pose",
                values=values,
                source=relative_path,
                initial_pose=name,
                initial_qpos=qpos,
                expert_observation=(
                    _expert_pose_observation(values, qpos) if encoded_available else None
                ),
                encoded_anchor_available=encoded_available,
                limitation="Official backward observation reconstructed from the 30-body pose.",
            )
        )

    if config.get("enable_human_push", True):
        window_seconds = float(config.get("human_push_window_seconds", 0.5))
        windows_per_file = int(config.get("human_push_windows_per_file", 3))
        window_frames = max(2, int(round(window_seconds * HUMAN_PUSH_FPS)))
        bfm_indices = [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS]
        for file_index in (1, 2):
            name = f"human_push_{file_index}"
            raw = np.load(root / "skate_push" / f"{name}.npy", allow_pickle=False)
            official_observation = (
                _expert_motion_observation(raw, official_g1_model)
                if official_g1_model is not None
                else None
            )
            joint_positions = raw[:, 7:36][:, bfm_indices].astype(np.float32)
            max_start = max(0, len(raw) - window_frames)
            starts = np.linspace(0, max_start, num=windows_per_file, dtype=int)
            starts = np.unique(starts)
            schema[name]["scoring_enabled"] = True
            schema[name]["encoded_anchor_available"] = official_observation is not None
            schema[name]["selected_windows"] = []
            for window_index, start in enumerate(starts):
                end = min(len(raw), start + window_frames)
                target_name = f"{name}_window_{window_index:02d}"
                schema[name]["selected_windows"].append(
                    {
                        "target_name": target_name,
                        "start_frame": int(start),
                        "end_frame_exclusive": int(end),
                        "encoded_anchor_available": official_observation is not None,
                        "rollout_initial_pose": target_name,
                        "rollout_initialization": (
                            "Window-first expert qpos/qvel; rigidly aligned to the "
                            "push-start reference with the right support foot on the "
                            "deck and the left push foot preserving its source offset"
                        ),
                    }
                )
                window_observation = (
                    {
                        key: value[start:end]
                        for key, value in official_observation.items()
                    }
                    if official_observation is not None
                    else None
                )
                initial_qpos, initial_qvel = _motion_initial_state(
                    raw,
                    int(start),
                    official_g1_model,
                    push_reference_feet,
                )
                targets.append(
                    ExpertTarget(
                        name=target_name,
                        kind="human_push_window",
                        values=joint_positions[start:end],
                        source=f"skate_push/{name}.npy",
                        frame_rate=HUMAN_PUSH_FPS,
                        start_frame=int(start),
                        end_frame=int(end),
                        joint_names=HUSKY_JOINTS,
                        initial_pose=target_name,
                        initial_qpos=initial_qpos,
                        initial_qvel=initial_qvel,
                        expert_observation=window_observation,
                        encoded_anchor_available=window_observation is not None,
                        limitation=(
                            "Scores common joint positions; the encoded latent trajectory also "
                            "uses the confirmed root and full 29DoF trajectory. The source "
                            "does not contain synchronized skateboard state."
                        ),
                    )
                )
    else:
        for file_index in (1, 2):
            schema[f"human_push_{file_index}"]["scoring_enabled"] = False

    schema["enabled_targets"] = [target.name for target in targets]
    schema["encoded_anchor_available"] = bool(targets) and all(
        target.encoded_anchor_available for target in targets
    )
    return targets, schema


def _checkpoint_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return None, metadata
    if isinstance(payload.get("metadata"), dict):
        metadata = dict(payload["metadata"])
    candidate = payload.get("model", payload)
    if not isinstance(candidate, dict):
        return None, metadata
    if not all(isinstance(key, str) for key in candidate):
        return None, metadata
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        return None, metadata
    return candidate, metadata


def _batched(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = value.to(device=device, dtype=torch.float32)
    return value.unsqueeze(0) if value.ndim == 1 else value


def _official_actor_keys(model_config: dict[str, Any]) -> tuple[str, ...]:
    actor_config = model_config.get("archi", {}).get("actor", {})
    input_filter = actor_config.get("input_filter", {})
    keys = input_filter.get("key", "actor_obs")
    if isinstance(keys, str):
        return (keys,)
    return tuple(keys)


def _official_model_directory(path: Path) -> Path | None:
    candidates = (
        path,
        path / "model",
        path / "checkpoint/model",
    )
    for candidate in candidates:
        if (
            (candidate / "config.json").is_file()
            and (candidate / "model.safetensors").is_file()
            and (
                (candidate / "init_kwargs.json").is_file()
                or (candidate / "init_kwargs.pkl").is_file()
            )
        ):
            return candidate
    return None


def _load_official_bfm0(
    path: Path,
    *,
    device: str | torch.device,
    report: dict[str, Any],
) -> OfficialBfm0Adapter:
    model_directory = _official_model_directory(path)
    if model_directory is None:
        report["error"] = (
            "No official model bundle found. Expected config.json, model.safetensors, "
            "and init_kwargs.* in the supplied directory, model/, or checkpoint/model/."
        )
        raise CheckpointCompatibilityError(report["error"], report)
    report["official_model_directory"] = str(model_directory)
    model_config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    report["official_model_name"] = model_config.get("name")
    report["official_actor_keys"] = list(_official_actor_keys(model_config))

    bfm_zero_root = os.environ.get("BFM_ZERO_ROOT")
    if bfm_zero_root:
        source_root = Path(bfm_zero_root).expanduser().resolve()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        report["bfm_zero_root"] = str(source_root)
    try:
        from humanoidverse.agents.load_utils import MODEL_NAME_TO_CLASS
    except Exception as exc:
        report["error"] = (
            "The BFM-Zero inference runtime is unavailable. Set BFM_ZERO_ROOT "
            "to train/scripts/isaac_env or another compatible runtime root."
        )
        report["import_error"] = f"{type(exc).__name__}: {exc}"
        raise CheckpointCompatibilityError(report["error"], report) from exc

    model_name = model_config.get("name")
    if model_name not in MODEL_NAME_TO_CLASS:
        report["error"] = f"Unsupported official BFM-Zero model class: {model_name!r}"
        raise CheckpointCompatibilityError(report["error"], report)
    try:
        load_device = torch.device(device).type
        model = MODEL_NAME_TO_CLASS[model_name].load(
            str(model_directory),
            device=load_device,
            strict=True,
        )
        model.to(device)
        adapter = OfficialBfm0Adapter(
            model,
            model_config,
            device=device,
        )
    except Exception as exc:
        report["error"] = f"Official strict load failed: {type(exc).__name__}: {exc}"
        raise CheckpointCompatibilityError(report["error"], report) from exc
    adapter.eval()
    adapter.requires_grad_(False)
    report["expected_dimensions"] = {
        "observation_space": {
            key: list(space.shape) for key, space in model.obs_space.spaces.items()
        },
        "action": adapter.config.action_dim,
        "latent": adapter.config.latent_dim,
    }
    report["format"] = "official_bfm_zero_bundle"
    report["compatible"] = True
    report["formal_eligible"] = True
    report["loaded_strictly"] = True
    report["frozen"] = True
    report["adapter"] = (
        "Official actor uses only provable H1 keys; non-actor observation-space "
        "keys receive recorded zero placeholders required by the official normalizer."
    )
    report["interfaces"] = {
        "actor": hasattr(model, "_actor") and callable(getattr(model, "act", None)),
        "backward_map": hasattr(model, "_backward_map")
        and callable(getattr(model, "backward_map", None)),
        "forward_map": hasattr(model, "_forward_map")
        and callable(getattr(model, "forward_map", None)),
    }
    if not all(report["interfaces"].values()):
        report["compatible"] = False
        report["formal_eligible"] = False
        report["error"] = "Official model is missing a required BFM0 inference interface"
        raise CheckpointCompatibilityError(report["error"], report)
    return adapter


def load_bfm0_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device,
    run_type: str,
) -> tuple[Bfm0Model | OfficialBfm0Adapter, dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    model = Bfm0Model().to(device)
    expected = model.state_dict()
    report: dict[str, Any] = {
        "checkpoint": str(path),
        "run_type": run_type,
        "loader": "skate_bfm.bfm0.Bfm0Model strict state_dict loader",
        "expected_dimensions": {
            "observation": model.config.observation_dim,
            "action": model.config.action_dim,
            "latent": model.config.latent_dim,
        },
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "compatible": False,
        "formal_eligible": False,
    }
    if not path.exists():
        report["error"] = "checkpoint path does not exist"
        raise CheckpointCompatibilityError(report["error"], report)

    if path.is_dir():
        return _load_official_bfm0(path, device=device, report=report), report

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        report["format"] = "unreadable_torch_checkpoint"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise CheckpointCompatibilityError(report["error"], report) from exc

    state_dict, metadata = _checkpoint_state_dict(payload)
    report["format"] = "torch_state_dict"
    report["metadata"] = metadata
    if state_dict is None:
        report["error"] = "checkpoint does not contain a tensor state_dict"
        raise CheckpointCompatibilityError(report["error"], report)

    expected_keys = set(expected)
    actual_keys = set(state_dict)
    report["missing_keys"] = sorted(expected_keys - actual_keys)
    report["unexpected_keys"] = sorted(actual_keys - expected_keys)
    for key in sorted(expected_keys & actual_keys):
        if tuple(expected[key].shape) != tuple(state_dict[key].shape):
            report["shape_mismatches"].append(
                {
                    "key": key,
                    "expected": list(expected[key].shape),
                    "actual": list(state_dict[key].shape),
                }
            )
    compatible = not (
        report["missing_keys"] or report["unexpected_keys"] or report["shape_mismatches"]
    )
    report["compatible"] = compatible
    if not compatible:
        report["error"] = "state_dict is not strictly compatible with Bfm0Model"
        raise CheckpointCompatibilityError(report["error"], report)

    is_temporary = bool(metadata.get("temporary", False))
    report["temporary"] = is_temporary
    if run_type == "formal":
        report["error"] = (
            "The compatible .pt graph is the repository's compact compatibility scaffold, "
            "not a verified official pretrained BFM0 model. Formal execution is refused."
        )
        raise CheckpointCompatibilityError(report["error"], report)
    if run_type != "smoke":
        report["error"] = f"unsupported run type: {run_type}"
        raise CheckpointCompatibilityError(report["error"], report)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)
    report["loaded_strictly"] = True
    report["frozen"] = True
    return model, report


def sample_global_latents(
    model: Bfm0Model | OfficialBfm0Adapter,
    count: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = torch.randn(count, model.config.latent_dim, generator=generator)
    return model.project_z(samples)


def encode_expert_latents(
    model: Bfm0Model | OfficialBfm0Adapter,
    target: ExpertTarget,
) -> torch.Tensor | None:
    if not isinstance(model, OfficialBfm0Adapter) or target.expert_observation is None:
        return None
    observation = {
        key: torch.from_numpy(value)
        for key, value in target.expert_observation.items()
    }
    with torch.no_grad():
        embeddings = model.backward_embedding(observation)
        if target.kind == "human_push_window":
            if len(embeddings) < 2:
                raise ValueError(
                    f"Dynamic expert target {target.name} needs at least two observations"
                )
            embeddings = embeddings[1:]
        latents = model.project_z(embeddings).detach().cpu()
        return latents[0] if target.kind == "static_pose" else latents


def angular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    cosine = torch.sum(a_norm * b_norm, dim=-1)
    return torch.acos(torch.clamp(cosine, -1.0, 1.0))


def constrain_to_geodesic_cap(
    model: Bfm0Model | OfficialBfm0Adapter,
    latents: torch.Tensor,
    anchor: torch.Tensor,
    max_angle_degrees: float,
) -> torch.Tensor:
    projected = model.project_z(latents)
    anchor = model.project_z(anchor.to(projected))
    while anchor.ndim < projected.ndim:
        anchor = anchor.unsqueeze(0)
    anchor_hat = torch.nn.functional.normalize(anchor, dim=-1)
    directions = torch.nn.functional.normalize(projected, dim=-1)
    cosine = torch.clamp(torch.sum(directions * anchor_hat, dim=-1), -1.0, 1.0)
    angles = torch.acos(cosine)
    maximum = math.radians(float(max_angle_degrees))
    outside = angles > maximum
    if not torch.any(outside):
        return projected
    tangent = directions - cosine.unsqueeze(-1) * anchor_hat
    tangent = torch.nn.functional.normalize(tangent, dim=-1)
    radius = torch.linalg.vector_norm(anchor, dim=-1, keepdim=True)
    capped = radius * (
        math.cos(maximum) * anchor_hat
        + math.sin(maximum) * tangent
    )
    result = torch.where(outside.unsqueeze(-1), capped, projected)
    return model.project_z(result)


def sample_geodesic_neighborhood(
    model: Bfm0Model | OfficialBfm0Adapter,
    anchor: torch.Tensor,
    angles_degrees: list[float],
    samples_per_angle: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    anchor = model.project_z(anchor).cpu()
    anchor_hat = torch.nn.functional.normalize(anchor, dim=-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    outputs = []
    labels = []
    radius = torch.linalg.vector_norm(anchor, dim=-1, keepdim=True)
    for angle_degrees in angles_degrees:
        epsilon = torch.randn(
            samples_per_angle,
            *anchor.shape,
            generator=generator,
        )
        tangent = epsilon - (
            torch.sum(epsilon * anchor_hat, dim=-1, keepdim=True) * anchor_hat
        )
        tangent = torch.nn.functional.normalize(tangent, dim=-1)
        alpha = math.radians(float(angle_degrees))
        points = radius * (math.cos(alpha) * anchor_hat + math.sin(alpha) * tangent)
        outputs.append(model.project_z(points))
        labels.extend([float(angle_degrees)] * samples_per_angle)
    return torch.cat(outputs), np.asarray(labels, dtype=np.float32)


def spherical_lerp(
    model: Bfm0Model | OfficialBfm0Adapter,
    start: torch.Tensor,
    end: torch.Tensor,
    fractions: torch.Tensor,
) -> torch.Tensor:
    start = model.project_z(start.reshape(1, -1))[0]
    end = model.project_z(end.reshape(1, -1))[0]
    start_hat = torch.nn.functional.normalize(start, dim=-1)
    end_hat = torch.nn.functional.normalize(end, dim=-1)
    dot = torch.clamp(torch.dot(start_hat, end_hat), -1.0, 1.0)
    if dot < 0:
        end_hat = -end_hat
        dot = -dot
    theta = torch.acos(dot)
    fractions = fractions.to(start).reshape(-1, 1)
    if float(theta) < 1e-6:
        directions = (1.0 - fractions) * start_hat + fractions * end_hat
    else:
        denominator = torch.sin(theta)
        directions = (
            torch.sin((1.0 - fractions) * theta) / denominator * start_hat
            + torch.sin(fractions * theta) / denominator * end_hat
        )
    return model.project_z(directions)


def quaternion_rotation_error(
    first: np.ndarray | torch.Tensor,
    second: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        first_t = torch.as_tensor(first)
        second_t = torch.as_tensor(second, device=first_t.device, dtype=first_t.dtype)
        first_t = torch.nn.functional.normalize(first_t, dim=-1)
        second_t = torch.nn.functional.normalize(second_t, dim=-1)
        dot = torch.sum(first_t * second_t, dim=-1).abs()
        return 2.0 * torch.acos(torch.clamp(dot, 0.0, 1.0))
    first_a = np.asarray(first)
    second_a = np.asarray(second)
    first_a = first_a / np.linalg.norm(first_a, axis=-1, keepdims=True)
    second_a = second_a / np.linalg.norm(second_a, axis=-1, keepdims=True)
    dot = np.abs(np.sum(first_a * second_a, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(np.asarray(first), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(second), -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quat_rotate(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    qvec = quaternion[..., 1:]
    uv = np.cross(qvec, vectors)
    uuv = np.cross(qvec, uv)
    return vectors + 2.0 * (quaternion[..., :1] * uv + uuv)


def _quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def _object_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        0,
    )
    return velocity[3:].copy(), velocity[:3].copy()


def extract_sim_state(env: HuskyLiteEnv) -> dict[str, Any]:
    model = env.model
    data = env.data
    robot_body_ids = [
        body_id
        for body_id in range(model.nbody)
        if (
            (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            and name.startswith(ROBOT_BODY_PREFIX)
        )
    ]
    robot_body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id).removeprefix(ROBOT_BODY_PREFIX)
        for body_id in robot_body_ids
    ]
    pelvis_id = model.body(f"{ROBOT_BODY_PREFIX}pelvis").id
    board_id = model.body(BOARD_BODY_NAME).id
    root_linear_velocity, root_angular_velocity = _object_velocity(model, data, pelvis_id)
    board_linear_velocity, board_angular_velocity = _object_velocity(model, data, board_id)

    joint_position = np.empty(len(HUSKY_JOINTS), dtype=np.float32)
    joint_velocity = np.empty(len(HUSKY_JOINTS), dtype=np.float32)
    for index, joint_name in enumerate(HUSKY_JOINTS):
        joint = model.joint(f"{ROBOT_BODY_PREFIX}{joint_name}")
        joint_position[index] = data.qpos[joint.qposadr[0]]
        joint_velocity[index] = data.qvel[joint.dofadr[0]]

    board_position = data.xpos[board_id].copy()
    board_quaternion = data.xquat[board_id].copy()
    body_position_world = data.xpos[robot_body_ids].copy()
    body_quaternion_world = data.xquat[robot_body_ids].copy()
    board_inverse = _quat_conjugate(board_quaternion)
    relative_position = _quat_rotate(
        np.broadcast_to(board_inverse, (len(robot_body_ids), 4)),
        body_position_world - board_position,
    )
    relative_quaternion = _quat_multiply(
        np.broadcast_to(board_inverse, (len(robot_body_ids), 4)),
        body_quaternion_world,
    )

    root_quaternion = data.xquat[pelvis_id].copy()
    return {
        "time": float(data.time),
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "root_position": data.xpos[pelvis_id].copy(),
        "root_quaternion": root_quaternion,
        "root_euler": _quaternion_to_euler(root_quaternion),
        "root_linear_velocity": root_linear_velocity,
        "root_angular_velocity": root_angular_velocity,
        "body_names": robot_body_names,
        "body_position_world": body_position_world,
        "body_quaternion_world": body_quaternion_world,
        "body_position_board": relative_position.astype(np.float32),
        "body_quaternion_board": relative_quaternion.astype(np.float32),
        "board_position": board_position,
        "board_quaternion": board_quaternion,
        "board_euler": _quaternion_to_euler(board_quaternion),
        "board_linear_velocity": board_linear_velocity,
        "board_angular_velocity": board_angular_velocity,
        "finite": bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.xpos).all()
        ),
    }


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def behavior_descriptor(
    states: list[dict[str, Any]],
    actions: np.ndarray,
    fall: bool,
) -> dict[str, float]:
    initial = states[0]
    final = states[-1]
    root_euler = np.stack([state["root_euler"] for state in states])
    root_angular_velocity = np.stack([state["root_angular_velocity"] for state in states])
    root_linear_velocity = np.stack([state["root_linear_velocity"] for state in states])
    joints = np.stack([state["joint_position"] for state in states])
    board_linear_velocity = np.stack([state["board_linear_velocity"] for state in states])
    left_indices = [
        index
        for index, name in enumerate(HUSKY_JOINTS)
        if name.startswith("left_") and any(part in name for part in ("hip", "knee", "ankle"))
    ]
    right_indices = [
        index
        for index, name in enumerate(HUSKY_JOINTS)
        if name.startswith("right_") and any(part in name for part in ("hip", "knee", "ankle"))
    ]
    joint_delta = joints - joints[0]
    left_rms = float(np.sqrt(np.mean(np.square(joint_delta[:, left_indices]))))
    right_rms = float(np.sqrt(np.mean(np.square(joint_delta[:, right_indices]))))
    return {
        "delta_root_x": float(final["root_position"][0] - initial["root_position"][0]),
        "delta_root_y": float(final["root_position"][1] - initial["root_position"][1]),
        "delta_root_z": float(final["root_position"][2] - initial["root_position"][2]),
        "mean_root_velocity_x": float(np.mean(root_linear_velocity[:, 0])),
        "mean_root_velocity_y": float(np.mean(root_linear_velocity[:, 1])),
        "delta_root_roll": _wrap_angle(root_euler[-1, 0] - root_euler[0, 0]),
        "delta_root_pitch": _wrap_angle(root_euler[-1, 1] - root_euler[0, 1]),
        "delta_root_yaw": _wrap_angle(root_euler[-1, 2] - root_euler[0, 2]),
        "max_abs_root_roll": float(np.max(np.abs(root_euler[:, 0]))),
        "max_abs_root_pitch": float(np.max(np.abs(root_euler[:, 1]))),
        "max_root_angular_velocity": float(np.max(np.linalg.norm(root_angular_velocity, axis=1))),
        "joint_motion_rms": float(np.sqrt(np.mean(np.square(joint_delta)))),
        "left_leg_motion_rms": left_rms,
        "right_leg_motion_rms": right_rms,
        "leg_motion_asymmetry": float(abs(left_rms - right_rms)),
        "action_rms": float(np.sqrt(np.mean(np.square(actions)))) if actions.size else 0.0,
        "delta_board_x": float(final["board_position"][0] - initial["board_position"][0]),
        "delta_board_y": float(final["board_position"][1] - initial["board_position"][1]),
        "delta_board_roll": _wrap_angle(final["board_euler"][0] - initial["board_euler"][0]),
        "delta_board_yaw": _wrap_angle(final["board_euler"][2] - initial["board_euler"][2]),
        "final_board_speed": float(np.linalg.norm(board_linear_velocity[-1])),
        "max_board_speed": float(np.max(np.linalg.norm(board_linear_velocity, axis=1))),
        "fall": float(fall),
    }


class H1RolloutRunner:
    def __init__(
        self,
        model: Bfm0Model | OfficialBfm0Adapter,
        config: dict[str, Any],
        *,
        device: str | torch.device,
    ) -> None:
        rollout_config = config["rollout"]
        self.model = model
        self.device = torch.device(device)
        self.control_dt = float(rollout_config["control_dt"])
        self.horizon_seconds = float(rollout_config["horizon_seconds"])
        self.steps = max(1, int(round(self.horizon_seconds / self.control_dt)))
        self.fall_height = float(rollout_config["fall_height"])
        self.unsafe_angle = float(rollout_config.get("unsafe_root_angle", 1.2))
        self.action_adapter = Bfm0ToHusky23(
            action_gain=1.0
            if isinstance(model, OfficialBfm0Adapter)
            else float(rollout_config["action_gain"]),
            action_clip=1.0
            if isinstance(model, OfficialBfm0Adapter)
            else float(rollout_config.get("action_clip", 1.0)),
        )
        self.observation_adapter = (
            OfficialHuskyToBfm0Observation()
            if isinstance(model, OfficialBfm0Adapter)
            else HuskyToBfm0Observation()
        )
        self.env = HuskyLiteEnv(
            control_dt=self.control_dt,
            action_scale=float(rollout_config.get("husky_action_scale", 0.1)),
        )
        if isinstance(model, OfficialBfm0Adapter):
            neutral, action_scale = official_husky_control_parameters(
                float(rollout_config["action_gain"])
            )
            self.env.set_control_mapping(neutral, action_scale)
        self.total_rollouts = 0
        self.successful_rollouts = 0
        self.failed_rollouts = 0
        self.fall_count = 0

    def close(self) -> None:
        self.env.close()

    def _apply_perturbation(self, seed: int, config: dict[str, float]) -> None:
        rng = np.random.default_rng(seed)
        model = self.env.model
        data = self.env.data
        root_joint = model.joint(ROBOT_ROOT_JOINT)
        board_joint = model.joint(BOARD_ROOT_JOINT)
        root_qpos = root_joint.qposadr[0]
        board_qpos = board_joint.qposadr[0]
        board_dof = board_joint.dofadr[0]

        root_angle = float(config.get("root_angle_noise", 0.0))
        roll = rng.normal(0.0, root_angle)
        pitch = rng.normal(0.0, root_angle)
        root_delta = _euler_delta_quaternion(roll, pitch, 0.0)
        data.qpos[root_qpos + 3 : root_qpos + 7] = _quat_multiply(
            data.qpos[root_qpos + 3 : root_qpos + 7],
            root_delta,
        )

        joint_noise = float(config.get("joint_position_noise", 0.0))
        for joint_name in HUSKY_JOINTS:
            joint = model.joint(f"{ROBOT_BODY_PREFIX}{joint_name}")
            data.qpos[joint.qposadr[0]] += rng.normal(0.0, joint_noise)

        board_roll = rng.normal(0.0, float(config.get("board_roll_noise", 0.0)))
        board_delta = _euler_delta_quaternion(board_roll, 0.0, 0.0)
        data.qpos[board_qpos + 3 : board_qpos + 7] = _quat_multiply(
            data.qpos[board_qpos + 3 : board_qpos + 7],
            board_delta,
        )
        velocity_noise = float(config.get("board_velocity_noise", 0.0))
        data.qvel[board_dof : board_dof + 3] += rng.normal(0.0, velocity_noise, size=3)
        mujoco.mj_normalizeQuat(model, data.qpos)
        mujoco.mj_forward(model, data)

    def _apply_initial_pose(
        self,
        initial_qpos: np.ndarray | None,
        initial_qvel: np.ndarray | None,
    ) -> None:
        if initial_qpos is None:
            return
        initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
        if initial_qpos.shape != (36,):
            raise ValueError(f"Expert initial qpos must have shape (36,), got {initial_qpos.shape}")
        model = self.env.model
        data = self.env.data
        root_joint = model.joint(ROBOT_ROOT_JOINT)
        root_qpos = root_joint.qposadr[0]
        data.qpos[root_qpos : root_qpos + 7] = initial_qpos[:7]
        for joint_name in HUSKY_JOINTS:
            joint = model.joint(f"{ROBOT_BODY_PREFIX}{joint_name}")
            data.qpos[joint.qposadr[0]] = initial_qpos[7 + BFM0_JOINTS.index(joint_name)]
        data.qvel[:] = 0.0
        if initial_qvel is not None:
            initial_qvel = np.asarray(initial_qvel, dtype=np.float64)
            if initial_qvel.shape != (35,):
                raise ValueError(
                    f"Expert initial qvel must have shape (35,), got {initial_qvel.shape}"
                )
            root_dof = root_joint.dofadr[0]
            data.qvel[root_dof : root_dof + 6] = initial_qvel[:6]
            for joint_name in HUSKY_JOINTS:
                joint = model.joint(f"{ROBOT_BODY_PREFIX}{joint_name}")
                data.qvel[joint.dofadr[0]] = initial_qvel[
                    6 + BFM0_JOINTS.index(joint_name)
                ]
        mujoco.mj_normalizeQuat(model, data.qpos)
        mujoco.mj_forward(model, data)

    def rollout(
        self,
        latent: torch.Tensor,
        *,
        seed: int,
        initial_qpos: np.ndarray | None = None,
        initial_qvel: np.ndarray | None = None,
        perturbation: dict[str, float] | None = None,
        capture_frames: bool = False,
        render_size: tuple[int, int] = (640, 480),
    ) -> RolloutResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        observation = self.env.reset()
        self._apply_initial_pose(initial_qpos, initial_qvel)
        observation = self.env._observation()
        self.observation_adapter.reset()
        last_bfm0_action = np.zeros(29, dtype=np.float32)
        if perturbation:
            self._apply_perturbation(seed, perturbation)
            observation = self.env._observation()

        latent = torch.as_tensor(latent, device=self.device)
        if latent.ndim == 1:
            latent_schedule = self.model.project_z(latent)
            rollout_steps = self.steps
        elif latent.ndim == 2:
            if len(latent) == 0:
                raise ValueError("Latent trajectory cannot be empty")
            latent_schedule = self.model.project_z(latent)
            rollout_steps = len(latent_schedule)
        else:
            raise ValueError(
                f"Latent must have shape [D] or [T, D], got {tuple(latent.shape)}"
            )
        states = [extract_sim_state(self.env)]
        actions = []
        frames: list[np.ndarray] = []
        renderer = None
        if capture_frames:
            width, height = render_size
            renderer = mujoco.Renderer(self.env.model, height=height, width=width)
            renderer.update_scene(self.env.data)
            frames.append(renderer.render().copy())

        terminated_early = False
        try:
            for step in range(rollout_steps):
                if isinstance(self.observation_adapter, OfficialHuskyToBfm0Observation):
                    bfm_observation = self.observation_adapter(
                        observation,
                        last_bfm0_action,
                    )
                else:
                    bfm_observation = self.observation_adapter(observation)
                bfm_observation = {
                    key: value.to(self.device) for key, value in bfm_observation.items()
                }
                with torch.no_grad():
                    bfm_action = self.model.act(
                        bfm_observation,
                        (
                            latent_schedule
                            if latent_schedule.ndim == 1
                            else latent_schedule[step]
                        ),
                        deterministic=True,
                    )[0]
                husky_action = self.action_adapter(bfm_action).detach().cpu().numpy()
                last_bfm0_action = bfm_action.detach().cpu().numpy()
                actions.append(husky_action.copy())
                observation = self.env.step(husky_action)
                state = extract_sim_state(self.env)
                states.append(state)
                if renderer is not None:
                    renderer.update_scene(self.env.data)
                    frames.append(renderer.render().copy())
                if not state["finite"]:
                    terminated_early = True
                    break
        finally:
            if renderer is not None:
                renderer.close()

        action_array = np.asarray(actions, dtype=np.float32)
        root_heights = np.asarray([state["root_position"][2] for state in states])
        root_angles = np.asarray([state["root_euler"][:2] for state in states])
        fall = bool(np.any(root_heights < self.fall_height) or terminated_early)
        unsafe = bool(np.any(np.abs(root_angles) > self.unsafe_angle))
        descriptor = behavior_descriptor(states, action_array, fall)
        self.total_rollouts += 1
        if terminated_early:
            self.failed_rollouts += 1
        else:
            self.successful_rollouts += 1
        if fall:
            self.fall_count += 1
        return RolloutResult(
            latent=latent_schedule.detach().cpu().numpy(),
            states=states,
            actions=action_array,
            descriptor=descriptor,
            fall=fall,
            unsafe=unsafe,
            terminated_early=terminated_early,
            seed=seed,
            frames=frames,
        )


def _euler_delta_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float64,
    )


def _resample_sequence(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == count:
        return values
    if len(values) == 1:
        return np.repeat(values, count, axis=0)
    source_time = np.linspace(0.0, 1.0, len(values))
    target_time = np.linspace(0.0, 1.0, count)
    output = np.empty((count, values.shape[1]), dtype=np.float64)
    for index in range(values.shape[1]):
        output[:, index] = np.interp(target_time, source_time, values[:, index])
    return output


def score_rollout(
    rollout: RolloutResult,
    target: ExpertTarget,
    scores: dict[str, float],
) -> ScoredRollout:
    fall_penalty = float(scores["fall_penalty"]) if rollout.fall else 0.0
    unsafe_penalty = float(scores["unsafe_penalty"]) if rollout.unsafe else 0.0
    if target.kind == "static_pose":
        position_errors = []
        rotation_errors = []
        total_errors = []
        for state in rollout.states:
            position_error = float(
                np.mean(
                    np.linalg.norm(
                        state["body_position_board"] - target.values[:, :3],
                        axis=1,
                    )
                )
            )
            rotation_error = float(
                np.mean(
                    quaternion_rotation_error(
                        state["body_quaternion_board"],
                        target.values[:, 3:],
                    )
                )
            )
            total_error = (
                float(scores["position_weight"]) * position_error
                + float(scores["rotation_weight"]) * rotation_error
            )
            position_errors.append(position_error)
            rotation_errors.append(rotation_error)
            total_errors.append(total_error)
        minimum_index = int(np.argmin(total_errors))
        initial_error = total_errors[0]
        minimum_error = total_errors[minimum_index]
        final_error = total_errors[-1]
        mean_error = float(np.mean(total_errors))
        minimum_progress = initial_error - minimum_error
        final_progress = initial_error - final_error
        score = -mean_error - fall_penalty - unsafe_penalty
        metrics = {
            "initial_pose_error": initial_error,
            "final_pose_error": final_error,
            "mean_pose_error": mean_error,
            "minimum_pose_error": minimum_error,
            "pose_error_improvement": final_progress,
            "minimum_pose_error_improvement": minimum_progress,
            "minimum_position_error": position_errors[minimum_index],
            "minimum_rotation_error": rotation_errors[minimum_index],
        }
        success = (
            final_error <= float(scores["pose_error_threshold"])
            and mean_error <= float(scores["pose_error_threshold"])
            and not rollout.fall
            and not rollout.unsafe
        )
    elif target.kind == "human_push_window":
        rollout_features = np.stack([state["joint_position"] for state in rollout.states])
        rollout_features = _resample_sequence(rollout_features, len(target.values))
        difference = rollout_features - target.values
        motion_error = float(np.sqrt(np.mean(np.square(difference))))
        score = -float(scores["motion_weight"]) * motion_error - fall_penalty - unsafe_penalty
        metrics = {
            "motion_error": motion_error,
            "initial_motion_error": float(
                np.sqrt(np.mean(np.square(rollout_features[0] - target.values[0])))
            ),
            "final_motion_error": float(
                np.sqrt(np.mean(np.square(rollout_features[-1] - target.values[-1])))
            ),
        }
        success = (
            motion_error <= float(scores["motion_error_threshold"])
            and not rollout.fall
            and not rollout.unsafe
        )
    else:
        raise ValueError(f"Unsupported target kind: {target.kind}")
    metrics["fall"] = float(rollout.fall)
    metrics["unsafe"] = float(rollout.unsafe)
    return ScoredRollout(
        target_name=target.name,
        score=float(score),
        success=bool(success),
        metrics=metrics,
        rollout=rollout,
    )


def run_cem(
    model: Bfm0Model | OfficialBfm0Adapter,
    evaluate: Callable[[torch.Tensor, int], ScoredRollout],
    initial_latent: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
) -> CemResult:
    population_size = int(config["population_size"])
    elite_count = max(
        1,
        int(math.ceil(population_size * float(config["elite_fraction"]))),
    )
    iterations = int(config["num_iterations"])
    anchor = model.project_z(initial_latent.detach().cpu())
    mean = anchor.clone()
    initial_std = float(config["initial_std"])
    std = torch.full_like(mean, initial_std)
    min_std = float(config["min_std"])
    max_angle_degrees = float(config["max_angle_degrees"])
    temporal_correlation = float(config.get("temporal_correlation", 0.0))
    if not 0.0 <= temporal_correlation < 1.0:
        raise ValueError("CEM temporal_correlation must be in [0, 1)")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_pair: tuple[torch.Tensor, ScoredRollout] | None = None
    history = []
    candidates: list[tuple[torch.Tensor, ScoredRollout]] = []
    for iteration in range(iterations):
        noise = torch.randn(
            population_size,
            *anchor.shape,
            generator=generator,
        )
        if anchor.ndim == 2 and temporal_correlation > 0.0:
            innovation_scale = math.sqrt(1.0 - temporal_correlation**2)
            for step in range(1, anchor.shape[0]):
                noise[:, step] = (
                    temporal_correlation * noise[:, step - 1]
                    + innovation_scale * noise[:, step]
                )
        unprojected = mean + std * noise
        unprojected[0] = mean
        latents = constrain_to_geodesic_cap(
            model,
            unprojected,
            anchor,
            max_angle_degrees,
        )
        scored = [
            evaluate(latent, seed + iteration * population_size + candidate_index)
            for candidate_index, latent in enumerate(latents)
        ]
        candidate_offset = len(candidates)
        candidates.extend(
            (latent.detach().cpu(), result) for latent, result in zip(latents, scored, strict=True)
        )
        score_values = torch.tensor([result.score for result in scored])
        elite_indices = torch.topk(score_values, elite_count).indices
        elite_latents = latents[elite_indices]
        mean = elite_latents.mean(dim=0)
        std = torch.clamp(
            elite_latents.std(dim=0, unbiased=False),
            min=min_std,
            max=initial_std,
        )
        iteration_best_index = int(torch.argmax(score_values))
        iteration_pair = (
            latents[iteration_best_index].detach().cpu(),
            scored[iteration_best_index],
        )
        if best_pair is None or iteration_pair[1].score > best_pair[1].score:
            best_pair = iteration_pair
        history.append(
            {
                "iteration": iteration,
                "best_candidate_index": candidate_offset + iteration_best_index,
                "best_latent": (
                    iteration_pair[0]
                    if iteration_pair[0].ndim == 1
                    else iteration_pair[0][len(iteration_pair[0]) // 2]
                ).numpy(),
                "best_score": iteration_pair[1].score,
                "elite_mean_score": float(score_values[elite_indices].mean()),
                "trajectory_steps": 1 if mean.ndim == 1 else len(mean),
                "mean": (
                    mean if mean.ndim == 1 else mean[len(mean) // 2]
                ).numpy().copy(),
                "std": (
                    std if std.ndim == 1 else std[len(std) // 2]
                ).numpy().copy(),
                "std_overall_mean": float(std.mean()),
                "std_overall_max": float(std.max()),
            }
        )
    if best_pair is None:
        raise RuntimeError("CEM produced no candidates")
    return CemResult(
        best_latent=best_pair[0],
        best=best_pair[1],
        history=history,
        candidates=candidates,
    )


def evaluate_geodesic_support(
    model: Bfm0Model | OfficialBfm0Adapter,
    anchor: torch.Tensor,
    evaluate: Callable[[torch.Tensor, int], ScoredRollout],
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[list[dict[str, float]], list[tuple[torch.Tensor, float, ScoredRollout]]]:
    latents, angles = sample_geodesic_neighborhood(
        model,
        anchor,
        [float(value) for value in config["angles_degrees"]],
        int(config["samples_per_angle"]),
        seed,
    )
    scored = [evaluate(latent, seed + index) for index, latent in enumerate(latents)]
    records = [
        (latent, float(angle), result)
        for latent, angle, result in zip(latents, angles, scored, strict=True)
    ]
    summary = []
    for angle in np.unique(angles):
        selected = [result for _, value, result in records if value == angle]
        descriptors = np.asarray(
            [list(result.rollout.descriptor.values()) for result in selected],
            dtype=np.float64,
        )
        summary.append(
            {
                "angle_degrees": float(angle),
                "mean_score": float(np.mean([result.score for result in selected])),
                "best_score": float(np.max([result.score for result in selected])),
                "success_rate": float(np.mean([result.success for result in selected])),
                "fall_rate": float(np.mean([result.rollout.fall for result in selected])),
                "descriptor_variance": float(np.mean(np.var(descriptors, axis=0))),
            }
        )
    return summary, records


def evaluate_robustness(
    runner: H1RolloutRunner,
    latent: torch.Tensor,
    target: ExpertTarget,
    scores: dict[str, float],
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[float, list[ScoredRollout]]:
    trials = int(config["trials"])
    perturbation = {key: float(value) for key, value in config.items() if key != "trials"}
    results = []
    for trial in range(trials):
        rollout = runner.rollout(
            latent,
            seed=seed + trial,
            initial_qpos=target.initial_qpos,
            initial_qvel=target.initial_qvel,
            perturbation=perturbation,
        )
        results.append(score_rollout(rollout, target, scores))
    return float(np.mean([result.success for result in results])), results


def classify_coverage(
    *,
    encoded_success: bool,
    encoded_robust_success_rate: float,
    global_success_rate: float,
    searched_success: bool,
    robust_success_rate: float,
    small_angle_support: float,
    search_angle_radians: float,
    config: dict[str, float],
) -> str:
    if (
        encoded_success
        and encoded_robust_success_rate >= config["robust_success_threshold"]
    ):
        return "zero_shot_covered"
    if (
        global_success_rate >= config["natural_success_rate"]
        and robust_success_rate >= config["robust_success_threshold"]
    ):
        return "naturally_covered"
    if searched_success and robust_success_rate < config["robust_success_threshold"]:
        return "fragile"
    if searched_success and small_angle_support >= config["local_support_threshold"]:
        return "locally_covered"
    if searched_success and search_angle_radians >= math.radians(config["distant_angle_degrees"]):
        return "searchable_but_distant"
    if searched_success:
        return "locally_covered"
    return "not_covered"
