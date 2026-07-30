from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class Bfm0Config:
    """Minimal BFM0-compatible network dimensions for the integration baseline."""

    state_dim: int = 64
    history_dim: int = 372
    action_dim: int = 29
    latent_dim: int = 256
    hidden_dim: int = 256
    hidden_layers: int = 2
    actor_std: float = 0.2

    @property
    def observation_dim(self) -> int:
        return self.state_dim + self.history_dim + self.action_dim


class Bfm0Model(nn.Module):
    """Compact forward-backward behavior model with the BFM0 policy interface.

    This module establishes the algorithm boundary used by Skate-BFM. It is not
    a replacement for the official BFM-Zero checkpoint or training pipeline.
    """

    def __init__(self, config: Bfm0Config | None = None) -> None:
        super().__init__()
        self.config = config or Bfm0Config()
        cfg = self.config
        self.backward_map = _mlp(
            cfg.observation_dim, cfg.latent_dim, cfg.hidden_dim, cfg.hidden_layers
        )
        self.forward_map = _mlp(
            cfg.observation_dim + cfg.action_dim + cfg.latent_dim,
            cfg.latent_dim,
            cfg.hidden_dim,
            cfg.hidden_layers,
        )
        self.actor = _mlp(
            cfg.observation_dim + cfg.latent_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.hidden_layers,
        )

    def flatten_observation(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        expected = {
            "state": self.config.state_dim,
            "history": self.config.history_dim,
            "last_action": self.config.action_dim,
        }
        parts = []
        for key, width in expected.items():
            if key not in observation:
                raise KeyError(f"Missing BFM0 observation field: {key}")
            value = observation[key]
            if value.ndim == 1:
                value = value.unsqueeze(0)
            if value.shape[-1] != width:
                raise ValueError(f"{key} must end in {width}, got {tuple(value.shape)}")
            parts.append(value)
        return torch.cat(parts, dim=-1)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        radius = math.sqrt(self.config.latent_dim)
        return radius * torch.nn.functional.normalize(z, dim=-1)

    def encode_goal(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.project_z(self.backward_map(self.flatten_observation(observation)))

    def forward_embedding(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        obs = self.flatten_observation(observation)
        return self.forward_map(torch.cat((obs, action, z), dim=-1))

    def act(
        self,
        observation: dict[str, torch.Tensor],
        z: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        obs = self.flatten_observation(observation)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        mean = torch.tanh(self.actor(torch.cat((obs, self.project_z(z)), dim=-1)))
        if deterministic:
            return mean
        return torch.clamp(Normal(mean, self.config.actor_std).rsample(), -1.0, 1.0)

    def infer_reward_latent(
        self,
        observations: dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.backward_map(self.flatten_observation(observations))
        weights = torch.softmax(rewards.reshape(-1), dim=0).unsqueeze(-1)
        return self.project_z((weights * embeddings).sum(dim=0, keepdim=True))[0]

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        config: Bfm0Config | None = None,
        device: str | torch.device = "cpu",
    ) -> Bfm0Model:
        model = cls(config).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict)
        model.eval()
        return model

