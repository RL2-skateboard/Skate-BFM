from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from skate_husky import AUX_REWARD_KEYS, HuskyLiteEnv

from skate_bfm.integration.actions import (
    Bfm0ToHusky23,
    official_husky_control_parameters,
)
from skate_bfm.integration.observations import HuskyToBfm0OnlineObservation


@dataclass(frozen=True)
class SkateOnlineTransition:
    observation: dict[str, torch.Tensor]
    z: torch.Tensor
    action_bfm: torch.Tensor
    action_husky: torch.Tensor
    next_observation: dict[str, torch.Tensor]
    aux_rewards: dict[str, float]
    terminated: bool
    truncated: bool
    step_count: int
    raw_metadata: dict[str, object]

    def as_buffer_data(self) -> dict:
        return {
            "observation": {
                key: value.unsqueeze(0)
                for key, value in self.observation.items()
            },
            "action": self.action_bfm.unsqueeze(0),
            "z": self.z.unsqueeze(0),
            "step_count": torch.tensor([[self.step_count]], dtype=torch.int64),
            "aux_rewards": {
                name: torch.tensor([[self.aux_rewards[name]]], dtype=torch.float32)
                for name in AUX_REWARD_KEYS
            },
            "next": {
                "observation": {
                    key: value.unsqueeze(0)
                    for key, value in self.next_observation.items()
                },
                "terminated": torch.tensor([[self.terminated]], dtype=torch.bool),
                "truncated": torch.tensor([[self.truncated]], dtype=torch.bool),
            },
        }


class HuskyBfmOnlineEnv:
    """One independent BFM-to-HUSKY online transition boundary."""

    def __init__(
        self,
        *,
        control_dt: float = 0.02,
        action_gain: float = 1.0,
        viewer: bool = False,
        realtime: bool = False,
    ) -> None:
        self.env = HuskyLiteEnv(
            control_dt=control_dt,
            viewer=viewer,
            realtime=realtime,
        )
        neutral, scale = official_husky_control_parameters(action_gain)
        self.env.set_control_mapping(neutral, scale)
        self.action_adapter = Bfm0ToHusky23(action_gain=1.0, action_clip=1.0)
        self.observation_adapter = HuskyToBfm0OnlineObservation()
        self._observation: dict[str, torch.Tensor] | None = None
        self._episode_done = True
        self._step_count = 0

    def reset(
        self,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        source_physics: Mapping[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        raw_observation = self.env.reset(
            qpos=qpos,
            qvel=qvel,
            source_physics=source_physics,
        )
        self.observation_adapter.reset()
        self._observation = self.observation_adapter(
            raw_observation,
            np.zeros(29, dtype=np.float32),
        )
        self._episode_done = False
        self._step_count = 0
        return self._observation

    def step(
        self,
        action_bfm: torch.Tensor,
        z: torch.Tensor,
        *,
        truncated: bool = False,
    ) -> SkateOnlineTransition:
        if self._episode_done or self._observation is None:
            raise RuntimeError("reset() is required before the next online transition.")
        action_bfm = torch.as_tensor(action_bfm, dtype=torch.float32).detach().cpu()
        z = torch.as_tensor(z, dtype=torch.float32).detach().cpu()
        if action_bfm.shape != (29,) or z.shape != (256,):
            raise ValueError(
                f"Expected action [29] and z [256], got {action_bfm.shape} and {z.shape}."
            )
        if not torch.isfinite(action_bfm).all() or not torch.isfinite(z).all():
            raise ValueError("BFM action and z must be finite.")

        action_husky = self.action_adapter(action_bfm)
        raw_next = self.env.step(action_husky.numpy())
        next_observation = self.observation_adapter(raw_next, action_bfm)
        aux_rewards = self.env.last_aux_rewards
        terminated = self.env.fallen
        fall_diagnostics = self.env.last_fall_diagnostics
        if tuple(aux_rewards) != AUX_REWARD_KEYS:
            raise RuntimeError("HUSKY auxiliary reward contract is incomplete.")
        transition = SkateOnlineTransition(
            observation=self._observation,
            z=z,
            action_bfm=action_bfm,
            action_husky=action_husky,
            next_observation=next_observation,
            aux_rewards=aux_rewards,
            terminated=terminated,
            truncated=bool(truncated) and not terminated,
            step_count=self._step_count,
            raw_metadata={
                "root_height": float(raw_next["root_height"]),
                "root_position": np.asarray(raw_next["root_position"]).copy(),
                "root_quaternion": np.asarray(raw_next["root_quaternion"]).copy(),
                "root_linear_velocity": np.asarray(
                    raw_next["root_linear_velocity"]
                ).copy(),
                "root_angular_velocity": np.asarray(
                    raw_next["root_angular_velocity"]
                ).copy(),
                "projected_gravity": np.asarray(
                    raw_next["projected_gravity"]
                ).copy(),
                "joint_position": np.asarray(raw_next["joint_position"]).copy(),
                "joint_velocity": np.asarray(raw_next["joint_velocity"]).copy(),
                "board_speed": float(raw_next["board_speed"]),
                "board_position": np.asarray(raw_next["board_position"]).copy(),
                "board_quaternion": np.asarray(raw_next["board_quaternion"]).copy(),
                "board_linear_velocity": np.asarray(
                    raw_next["board_linear_velocity"]
                ).copy(),
                "board_angular_velocity": np.asarray(
                    raw_next["board_angular_velocity"]
                ).copy(),
                "fall": terminated,
                **fall_diagnostics,
            },
        )
        self._observation = next_observation
        self._step_count += 1
        self._episode_done = transition.terminated or transition.truncated
        return transition

    def close(self) -> None:
        self.env.close()
