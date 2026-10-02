from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# This is the environment-side 29-DoF contract. The exact BFM checkpoint
# ordering, normalization and privileged layout remain checkpoint-dependent;
# they must be filled from the selected official config before policy use.
G1_29_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
    "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
    "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch",
    "right_wrist_yaw",
)


@dataclass(frozen=True)
class RobotContract:
    joint_names: tuple[str, ...] = G1_29_JOINTS
    physics_dt: float = .002
    control_dt: float = .02
    action_scale: float | np.ndarray = .25
    action_clip: float = 1.0
    kp: float = 40.0
    kd: float = 1.0
    effort_limit: float = 100.0
    velocity_limit: float = 30.0
    history_length: int = 4
    actor_dim: int | None = None
    privileged_dim: int | None = None
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def decimation(self) -> int:
        return round(self.control_dt / self.physics_dt)

    def validate(self) -> None:
        if self.dof != 29:
            raise ValueError(f"robot contract requires 29 joints, got {self.dof}")
        if self.decimation < 1 or self.history_length < 1:
            raise ValueError("invalid control/history timing")
        if not self.joint_names or len(set(self.joint_names)) != self.dof:
            raise ValueError("joint names must be unique")
        scale = np.asarray(self.action_scale)
        if scale.ndim > 0 and scale.shape != (self.dof,):
            raise ValueError("action_scale must be scalar or one value per joint")


@dataclass
class RobotState:
    q: np.ndarray
    dq: np.ndarray
    base_quat_wxyz: np.ndarray
    base_angvel: np.ndarray
    gravity_body: np.ndarray


@dataclass
class HistoryState:
    frames: list[np.ndarray] = field(default_factory=list)
    action_frames: list[np.ndarray] = field(default_factory=list)
    angvel_frames: list[np.ndarray] = field(default_factory=list)
    pos_frames: list[np.ndarray] = field(default_factory=list)
    vel_frames: list[np.ndarray] = field(default_factory=list)
    gravity_frames: list[np.ndarray] = field(default_factory=list)
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))

    @property
    def stacked(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.frames, axis=0).astype(np.float32)

    def reset(self) -> None:
        self.frames.clear()
        self.action_frames.clear(); self.angvel_frames.clear(); self.pos_frames.clear()
        self.vel_frames.clear(); self.gravity_frames.clear()
        self.last_action[:] = 0

    def push(self, frame: np.ndarray, length: int) -> np.ndarray:
        self.frames.append(np.asarray(frame, dtype=np.float32).copy())
        del self.frames[:-length]
        while len(self.frames) < length:
            self.frames.insert(0, np.zeros_like(self.frames[0]))
        return np.concatenate(self.frames, axis=0)

    def push_field(self, field_name: str, value: np.ndarray, length: int = 4) -> np.ndarray:
        values = getattr(self, f"{field_name}_frames")
        values.insert(0, np.asarray(value, dtype=np.float32).copy())
        del values[length:]
        while len(values) < length:
            values.append(np.zeros_like(values[0]))
        return np.concatenate(values, axis=0)


def make_contract(**overrides: Any) -> RobotContract:
    contract = RobotContract(**overrides)
    contract.validate()
    return contract


def pd_torque(contract: RobotContract, q: np.ndarray, dq: np.ndarray,
              target: np.ndarray) -> np.ndarray:
    """Compute one current-state PD torque; caller runs this every physics substep."""
    contract.validate()
    q, dq, target = map(lambda x: np.asarray(x, dtype=np.float64), (q, dq, target))
    if q.shape != (29,) or dq.shape != (29,) or target.shape != (29,):
        raise ValueError("PD vectors must have shape (29,)")
    tau = contract.kp * (target - q) - contract.kd * dq
    return np.clip(tau, -contract.effort_limit, contract.effort_limit)


def build_history_frame(state: RobotState, action: np.ndarray) -> np.ndarray:
    """Checkpoint-neutral real-state frame; official BFM field order is pending."""
    return np.concatenate((state.q, state.dq, state.base_angvel, state.gravity_body,
                           np.asarray(action, dtype=np.float32)), axis=0).astype(np.float32)
