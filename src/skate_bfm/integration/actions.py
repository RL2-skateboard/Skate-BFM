from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

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

BFM0_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

HUSKY_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_yaw_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

BFM0_INACTIVE_JOINTS = tuple(
    name for name in BFM0_JOINTS if name not in set(HUSKY_JOINTS)
)
BFM0_INACTIVE_ACTION_INDICES = tuple(
    BFM0_JOINTS.index(name) for name in BFM0_INACTIVE_JOINTS
)
BFM0_ACTION_CONSUMERS = (
    "_forward_map",
    "_target_forward_map",
    "_critic",
    "_target_critic",
    "_aux_critic",
    "_target_aux_critic",
)


def project_husky_bfm_action(
    action: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Project a 29D BFM action onto HUSKY's effective 23D subspace."""

    if action.ndim == 0 or action.shape[-1] != len(BFM0_JOINTS):
        actual = None if action.ndim == 0 else action.shape[-1]
        raise ValueError(f"Expected 29 BFM0 actions, got {actual}")
    projected = action.clone() if torch.is_tensor(action) else np.array(action, copy=True)
    projected[..., BFM0_INACTIVE_ACTION_INDICES] = 0
    return projected


def _project_action_input(
    _module: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[tuple[object, ...], dict[str, object]]:
    if len(args) > 2:
        if "action" in kwargs:
            raise TypeError("Action was provided as both a positional and keyword argument.")
        updated = list(args)
        updated[2] = project_husky_bfm_action(updated[2])
        return tuple(updated), kwargs
    if "action" in kwargs:
        return args, {
            **kwargs,
            "action": project_husky_bfm_action(kwargs["action"]),
        }
    raise TypeError("Action consumer requires a third positional or 'action' argument.")


def install_husky_action_projection(model: torch.nn.Module) -> bool:
    """Install the HUSKY action projection on every native action consumer."""

    marker = "_skate_husky_action_projection_handles"
    if hasattr(model, marker):
        return False
    consumers = []
    for name in BFM0_ACTION_CONSUMERS:
        module = getattr(model, name, None)
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError(f"Required BFM action consumer is unavailable: {name}")
        consumers.append(module)
    handles = tuple(
        module.register_forward_pre_hook(_project_action_input, with_kwargs=True)
        for module in consumers
    )
    setattr(model, marker, handles)
    return True


@dataclass(frozen=True)
class JointMapping:
    source: tuple[str, ...]
    target: tuple[str, ...]
    shared: tuple[str, ...]
    dropped: tuple[str, ...]


class Bfm0ToHusky23:
    """Name-based 29 DoF BFM0 to 23 DoF HUSKY action adapter."""

    def __init__(self, action_gain: float = 1.0, action_clip: float | None = 1.0) -> None:
        source_index = {name: index for index, name in enumerate(BFM0_JOINTS)}
        self._indices = torch.tensor([source_index[name] for name in HUSKY_JOINTS])
        self.action_gain = float(action_gain)
        self.action_clip = action_clip
        shared = tuple(name for name in BFM0_JOINTS if name in HUSKY_JOINTS)
        dropped = tuple(name for name in BFM0_JOINTS if name not in HUSKY_JOINTS)
        self.mapping = JointMapping(BFM0_JOINTS, HUSKY_JOINTS, shared, dropped)

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != len(BFM0_JOINTS):
            raise ValueError(f"Expected 29 BFM0 actions, got {action.shape[-1]}")
        indices = self._indices.to(action.device)
        mapped = action.index_select(-1, indices) * self.action_gain
        if self.action_clip is not None:
            mapped = torch.clamp(mapped, -self.action_clip, self.action_clip)
        return mapped


def official_husky_control_parameters(
    action_gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized BFM actions to the HUSKY joint-control convention."""

    indices = np.asarray(
        [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS],
        dtype=np.int64,
    )
    neutral = BFM0_DEFAULT_JOINT_POSITION[indices].copy()
    scale = BFM0_ACTION_SCALES[indices] * BFM0_ACTION_RESCALE * float(action_gain)
    return neutral, scale
