"""Pure offline conversion of historical 23-joint expert data. No simulator API."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .actions import BFM0_JOINTS, BFM0_DEFAULT_JOINT_POSITION, BFM0_ACTION_TARGET_GAINS

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



SOURCE_JOINT_SUPPORT = 23
SOURCE_MISSING_JOINTS = BFM0_INACTIVE_JOINTS
SOURCE_METADATA = {"source_joint_support": 23, "missing_joint_names": list(SOURCE_MISSING_JOINTS), "native29_expert": False}
