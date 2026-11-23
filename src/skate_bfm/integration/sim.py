"""Explicit native 29-DoF skate-sim adapter; no legacy HUSKY projection."""
from __future__ import annotations

import torch
from skate_sim.env import AUX_REWARD_KEYS


class SkateSimOnlineEnv:
    """One native backend with batched worlds and replay-ready transitions.

    Native preset resets do not claim alignment with offline HUSKY experts.
    The caller owns reset scheduling, timeouts and 256D rollout latents.
    """
    def __init__(self, *, backend="mjlab", num_envs=4, preset="steer_start", seed=0, scene_config=None):
        from skate_sim.env import SkateEnv
        self.env = SkateEnv(backend=backend, num_envs=num_envs, preset=preset, seed=seed, scene_config=scene_config)
        self.num_envs = num_envs
        self.device = self.env.device
        self.metadata = {"online_reset_source":"native_preset", "preset":self.env.preset, "seed":seed,
                         "model_sha256":self.env.model_sha256,"source_physics_aligned":False,
                         "tracking_aligned":False}

    def reset(self, *, env_ids=None, seed=None):
        return self.env.reset(env_ids=env_ids, seed=seed)

    def step(self, action_bfm, z, *, truncated=None):
        z=torch.as_tensor(z,device=self.device,dtype=torch.float32).detach()
        if z.shape != (self.num_envs,256) or not bool(torch.isfinite(z).all()):
            raise ValueError(f"z must be finite [{self.num_envs},256]")
        timeout=None
        if truncated is not None:
            timeout=torch.as_tensor(truncated,device=self.device)
            if timeout.dtype != torch.bool:
                raise ValueError("timeout mask must have boolean dtype")
            if timeout.shape not in ((self.num_envs,), (self.num_envs,1)):
                raise ValueError("timeout mask must have one entry per world")
        data=self.env.step(action_bfm)
        if timeout is not None:
            data["next"]["truncated"] |= timeout.reshape(self.num_envs,1) & ~data["next"]["terminated"]
            self.env._done |= data["next"]["truncated"].flatten()
        data["z"]=z.clone()
        return data

    def close(self):
        self.env.close()
