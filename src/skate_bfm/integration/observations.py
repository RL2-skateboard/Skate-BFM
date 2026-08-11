from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch

from skate_bfm.integration.actions import (
    BFM0_ACTION_RESCALE,
    BFM0_DEFAULT_JOINT_POSITION,
    BFM0_JOINTS,
    HUSKY_JOINTS,
)


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion)
    vector = np.asarray(vector)
    q_vector = quaternion[..., 1:]
    return (
        vector
        + 2.0 * quaternion[..., :1] * np.cross(q_vector, vector)
        + 2.0 * np.cross(q_vector, np.cross(q_vector, vector))
    )


def _extend_head_state(
    body_position: np.ndarray,
    body_quaternion: np.ndarray,
    body_linear_velocity: np.ndarray,
    body_angular_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torso_index = 15
    head_offset = _quat_rotate(
        body_quaternion[torso_index],
        np.asarray((0.0, 0.0, 0.35), dtype=np.float64),
    )
    head_position = body_position[torso_index] + head_offset
    head_angular_velocity = body_angular_velocity[torso_index]
    head_linear_velocity = (
        body_linear_velocity[torso_index]
        + np.cross(head_angular_velocity, head_offset)
    )
    return (
        np.concatenate((body_position, head_position[None]), axis=0),
        np.concatenate((body_quaternion, body_quaternion[torso_index : torso_index + 1]), axis=0),
        np.concatenate((body_linear_velocity, head_linear_velocity[None]), axis=0),
        np.concatenate((body_angular_velocity, head_angular_velocity[None]), axis=0),
    )


def bfm0_privileged_state(
    body_position: np.ndarray,
    body_quaternion: np.ndarray,
    body_linear_velocity: np.ndarray,
    body_angular_velocity: np.ndarray,
) -> np.ndarray:
    """Build the official 463D max-local-self observation from HUSKY bodies."""

    position, quaternion, linear_velocity, angular_velocity = _extend_head_state(
        np.asarray(body_position, dtype=np.float64),
        np.asarray(body_quaternion, dtype=np.float64),
        np.asarray(body_linear_velocity, dtype=np.float64),
        np.asarray(body_angular_velocity, dtype=np.float64),
    )
    root_forward = _quat_rotate(
        quaternion[:1],
        np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64),
    )[0]
    heading = math.atan2(root_forward[1], root_forward[0])
    heading_inverse = np.asarray(
        (math.cos(-heading / 2.0), 0.0, 0.0, math.sin(-heading / 2.0)),
        dtype=np.float64,
    )
    heading_quaternions = np.broadcast_to(heading_inverse, quaternion.shape)
    local_position = _quat_rotate(
        heading_quaternions,
        position - position[0],
    )[1:]
    local_quaternion = _quat_multiply(heading_quaternions, quaternion)
    tangent = _quat_rotate(
        local_quaternion,
        np.broadcast_to((1.0, 0.0, 0.0), position.shape),
    )
    normal = _quat_rotate(
        local_quaternion,
        np.broadcast_to((0.0, 0.0, 1.0), position.shape),
    )
    local_linear_velocity = _quat_rotate(heading_quaternions, linear_velocity)
    local_angular_velocity = _quat_rotate(heading_quaternions, angular_velocity)
    result = np.concatenate(
        (
            position[0, 2:3],
            local_position.reshape(-1),
            np.concatenate((tangent, normal), axis=-1).reshape(-1),
            local_linear_velocity.reshape(-1),
            local_angular_velocity.reshape(-1),
        )
    ).astype(np.float32)
    if result.shape != (463,):
        raise ValueError(
            f"Official BFM privileged state must have shape (463,), got {result.shape}."
        )
    return result


class HuskyToBfm0Observation:
    """Convert the HUSKY 23 DoF state into BFM0's 29 DoF history schema."""

    def __init__(self, history_length: int = 4) -> None:
        self.history_length = history_length
        self._target_indices = np.array([BFM0_JOINTS.index(name) for name in HUSKY_JOINTS])
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
            self._gravity.append(np.array((0.0, 0.0, -1.0), dtype=np.float32))

    def _expand_joints(self, value: np.ndarray) -> np.ndarray:
        expanded = np.zeros(29, dtype=np.float32)
        expanded[self._target_indices] = value
        return expanded

    def __call__(self, observation: dict[str, np.ndarray | float]) -> dict[str, torch.Tensor]:
        joint_position = self._expand_joints(np.asarray(observation["joint_position"]))
        joint_velocity = self._expand_joints(np.asarray(observation["joint_velocity"]))
        last_action = self._expand_joints(np.asarray(observation["last_action"]))
        gravity = np.asarray(observation["projected_gravity"], dtype=np.float32)
        angular_velocity = np.asarray(observation["angular_velocity"], dtype=np.float32)

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


class HuskyToBfm0OnlineObservation:
    """Convert HUSKY state to the official BFM-Zero online observation contract."""

    def __init__(self, history_length: int = 4) -> None:
        self.history_length = history_length
        self._target_indices = np.asarray(
            [BFM0_JOINTS.index(name) for name in HUSKY_JOINTS],
            dtype=np.int64,
        )
        self._history: dict[str, deque[np.ndarray]] = {
            "actions": deque(maxlen=history_length),
            "base_ang_vel": deque(maxlen=history_length),
            "dof_pos": deque(maxlen=history_length),
            "dof_vel": deque(maxlen=history_length),
            "projected_gravity": deque(maxlen=history_length),
        }
        self.reset()

    def reset(self) -> None:
        widths = {
            "actions": 29,
            "base_ang_vel": 3,
            "dof_pos": 29,
            "dof_vel": 29,
            "projected_gravity": 3,
        }
        for key, history in self._history.items():
            history.clear()
            for _ in range(self.history_length):
                history.append(np.zeros(widths[key], dtype=np.float32))

    def _expand_joints(self, value: np.ndarray) -> np.ndarray:
        expanded = np.zeros(29, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (23,):
            raise ValueError(f"Expected 23 HUSKY joint values, got {value.shape}.")
        expanded[self._target_indices] = value
        return expanded

    def __call__(
        self,
        observation: dict[str, np.ndarray | float],
        last_bfm0_action: np.ndarray | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        joint_position = (
            self._expand_joints(np.asarray(observation["joint_position"]))
            - BFM0_DEFAULT_JOINT_POSITION
        )
        joint_velocity = self._expand_joints(np.asarray(observation["joint_velocity"]))
        gravity = np.asarray(observation["projected_gravity"], dtype=np.float32)
        angular_velocity = (
            np.asarray(observation["angular_velocity"], dtype=np.float32) * 0.25
        )
        last_action = np.asarray(last_bfm0_action, dtype=np.float32)
        if last_action.shape != (29,):
            raise ValueError(f"Expected 29D previous BFM action, got {last_action.shape}.")
        last_action = last_action * BFM0_ACTION_RESCALE

        history_actor = np.concatenate(
            [
                value
                for key in sorted(self._history)
                for value in self._history[key]
            ]
        ).astype(np.float32)
        privileged_state = bfm0_privileged_state(
            np.asarray(observation["body_position"]),
            np.asarray(observation["body_quaternion"]),
            np.asarray(observation["body_linear_velocity"]),
            np.asarray(observation["body_angular_velocity"]),
        )
        state = np.concatenate(
            (
                joint_position,
                joint_velocity,
                gravity,
                angular_velocity,
            )
        ).astype(np.float32)

        current = {
            "actions": last_action,
            "base_ang_vel": angular_velocity,
            "dof_pos": joint_position,
            "dof_vel": joint_velocity,
            "projected_gravity": gravity,
        }
        for key, value in current.items():
            self._history[key].appendleft(value.copy())

        result = {
            "state": torch.from_numpy(state),
            "privileged_state": torch.from_numpy(privileged_state),
            "last_action": torch.from_numpy(last_action),
            "history_actor": torch.from_numpy(history_actor),
        }
        for key, value in result.items():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid BFM online observation field: {key}.")
        return result
