from __future__ import annotations

from collections import deque

import numpy as np
import torch

from skate_bfm.integration.actions import BFM0_JOINTS, HUSKY_JOINTS


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

