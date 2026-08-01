from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from skate_husky import HuskyLiteEnv

from skate_bfm.bfm0 import Bfm0Model
from skate_bfm.integration import Bfm0ToHusky23, HuskyToBfm0Observation


@dataclass(frozen=True)
class RolloutSummary:
    steps: int
    initial_root_height: float
    final_root_height: float
    final_board_speed: float


def run_smoke(
    steps: int = 20,
    seed: int = 42,
    *,
    viewer: bool = False,
    realtime: bool | None = None,
    action_gain: float = 0.0,
) -> RolloutSummary:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0 and not viewer:
        raise ValueError("steps=0 is only supported with viewer=True")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Bfm0Model().eval()
    adapter = Bfm0ToHusky23(action_gain=action_gain)
    observation_adapter = HuskyToBfm0Observation()
    env = HuskyLiteEnv(
        viewer=viewer,
        realtime=viewer if realtime is None else realtime,
    )
    observation = env.reset()
    initial_height = float(observation["root_height"])
    latent = torch.zeros(model.config.latent_dim)
    latent[0] = 1.0
    completed_steps = 0

    try:
        while env.is_running and (steps == 0 or completed_steps < steps):
            bfm_observation = observation_adapter(observation)
            with torch.no_grad():
                bfm_action = model.act(bfm_observation, latent)[0]
            husky_action = adapter(bfm_action).numpy()
            observation = env.step(husky_action)
            completed_steps += 1
    finally:
        env.close()

    return RolloutSummary(
        steps=completed_steps,
        initial_root_height=initial_height,
        final_root_height=float(observation["root_height"]),
        final_board_speed=float(observation["board_speed"]),
    )
