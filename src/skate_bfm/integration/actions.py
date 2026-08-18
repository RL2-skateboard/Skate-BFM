from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

BFM0_ACTION_RESCALE = 5.0
BFM0_ACTION_SCALE = 0.25
BFM0_ACTION_CLIP = 5.0
BFM0_KP = np.asarray(
    (
        99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
        99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
        300.0, 300.0, 300.0,
        14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
        14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
    ),
    dtype=np.float64,
)
BFM0_KD = np.asarray(
    (
        6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
        6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
        5.0, 5.0, 5.0,
        0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
        0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
    ),
    dtype=np.float64,
)
BFM0_EFFORT_LIMITS = np.asarray(
    (
        139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 50.0, 50.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    ),
    dtype=np.float64,
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
BFM0_ACTION_TARGET_GAINS = (
    BFM0_ACTION_RESCALE * BFM0_ACTION_SCALE * BFM0_EFFORT_LIMITS / BFM0_KP
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

    if not isinstance(action, (np.ndarray, torch.Tensor)):
        raise TypeError("BFM action must be a NumPy array or Torch tensor.")
    if action.ndim == 0 or action.shape[-1] != len(BFM0_JOINTS):
        actual = None if action.ndim == 0 else action.shape[-1]
        raise ValueError(f"Expected 29 BFM0 actions, got {actual}")
    floating = (
        torch.is_floating_point(action)
        if torch.is_tensor(action)
        else np.issubdtype(action.dtype, np.floating)
    )
    finite = (
        torch.isfinite(action).all().item()
        if torch.is_tensor(action)
        else np.isfinite(action).all()
    )
    if not floating:
        raise TypeError("BFM actions must use a floating dtype.")
    if not finite:
        raise ValueError("BFM actions must be finite.")
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


@dataclass(frozen=True)
class SkateActionTranslation:
    """Derived BFM actions for canonical Skate source actions."""

    equivalent_23: np.ndarray
    valid_23: np.ndarray
    bridge_23: np.ndarray
    bridge_29: np.ndarray
    mode: str | np.ndarray


SKATE_SOURCE_JOINTS = tuple(
    name for name in BFM0_JOINTS if name in set(HUSKY_JOINTS)
)
SKATE_SOURCE_DEFAULT_BY_JOINT = dict(
    zip(
        SKATE_SOURCE_JOINTS,
        (
            0.0,
            0.0,
            0.0,
            0.23,
            -0.2,
            0.0,
            -0.7,
            0.0,
            0.0,
            1.17,
            -0.45,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.03,
            0.45,
            -0.21,
            1.32,
            -0.7,
            -0.845,
            0.83,
            1.19,
        ),
        strict=True,
    )
)
SKATE_SOURCE_SCALE_BY_JOINT = dict(
    zip(
        SKATE_SOURCE_JOINTS,
        (
            0.5475,
            0.3507,
            0.5475,
            0.3507,
            0.4386,
            0.4386,
            0.5475,
            0.3507,
            0.5475,
            0.3507,
            0.4386,
            0.4386,
            0.5475,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
        ),
        strict=True,
    )
)
SKATE_SOURCE_TO_HUSKY_INDICES = np.asarray(
    [SKATE_SOURCE_JOINTS.index(name) for name in HUSKY_JOINTS],
    dtype=np.int64,
)
SKATE_BFM_INDICES = np.asarray(
    [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS],
    dtype=np.int64,
)


def skate_source_control_parameters() -> tuple[np.ndarray, np.ndarray]:
    """Return source neutral targets and scales in HUSKY actuator order."""

    neutral = np.asarray(
        [SKATE_SOURCE_DEFAULT_BY_JOINT[name] for name in HUSKY_JOINTS],
        dtype=np.float64,
    )
    scale = np.asarray(
        [SKATE_SOURCE_SCALE_BY_JOINT[name] for name in HUSKY_JOINTS],
        dtype=np.float64,
    )
    return neutral, scale


def translate_skate_action(
    source_action: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> SkateActionTranslation:
    """Translate source-policy actions into the BFM normalized action contract."""

    source_action = np.asarray(source_action)
    if source_action.ndim == 0 or source_action.shape[-1] != len(SKATE_SOURCE_JOINTS):
        actual = None if source_action.ndim == 0 else source_action.shape[-1]
        raise ValueError(f"Expected 23D Skate source actions, got {actual}")
    if not np.issubdtype(source_action.dtype, np.floating):
        raise TypeError("Skate source actions must use a floating dtype.")
    if not np.isfinite(source_action).all():
        raise ValueError("Skate source actions must be finite.")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("Translation tolerance must be finite and non-negative.")

    dtype = np.result_type(source_action.dtype, np.float32)
    source_neutral, source_scale = skate_source_control_parameters()
    source_husky = source_action.astype(dtype, copy=False)[
        ..., SKATE_SOURCE_TO_HUSKY_INDICES
    ]
    bfm_default = BFM0_DEFAULT_JOINT_POSITION[SKATE_BFM_INDICES].astype(dtype)
    bfm_scale = BFM0_ACTION_TARGET_GAINS[SKATE_BFM_INDICES].astype(dtype)
    equivalent = (
        source_neutral.astype(dtype)
        + source_scale.astype(dtype) * source_husky
        - bfm_default
    ) / bfm_scale
    valid = np.abs(equivalent) <= 1.0 + tolerance
    bridge = np.clip(equivalent, -1.0, 1.0)
    bridge_29 = np.zeros(
        source_action.shape[:-1] + (len(BFM0_JOINTS),),
        dtype=bridge.dtype,
    )
    bridge_29[..., SKATE_BFM_INDICES] = bridge
    mode = np.where(np.all(valid, axis=-1), "EXACT", "PROJECTED")
    if source_action.ndim == 1:
        mode = str(mode.item())
    return SkateActionTranslation(
        equivalent_23=equivalent,
        valid_23=valid,
        bridge_23=bridge,
        bridge_29=bridge_29,
        mode=mode,
    )


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
    """Return the official hard-waist BFM target contract in HUSKY order."""

    action_gain = float(action_gain)
    if not np.isfinite(action_gain):
        raise ValueError("Action gain must be finite.")

    indices = np.asarray(
        [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS],
        dtype=np.int64,
    )
    neutral = BFM0_DEFAULT_JOINT_POSITION[indices].copy()
    scale = BFM0_ACTION_TARGET_GAINS[indices] * action_gain
    return neutral, scale


def official_husky_actuator_parameters(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hard-waist Kp, Kd, and effort limits in HUSKY order."""

    indices = np.asarray(
        [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS],
        dtype=np.int64,
    )
    return (
        BFM0_KP[indices].copy(),
        BFM0_KD[indices].copy(),
        BFM0_EFFORT_LIMITS[indices].copy(),
    )
