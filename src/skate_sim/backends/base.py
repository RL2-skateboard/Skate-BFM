from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def validate_env_ids(env_ids, count):
    if env_ids is None:
        return np.arange(count, dtype=np.int64)
    if hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu().numpy()
    raw = np.asarray(env_ids)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size == 0:
        raise ValueError("env_ids must be a nonempty integer vector")
    if np.any(raw >= count) or np.any(raw < 0) or len(np.unique(raw)) != len(raw):
        raise ValueError("env_ids must be unique and in range")
    return raw.astype(np.int64)


class Backend(ABC):
    name: str

    @abstractmethod
    def reset(self, seed: int | None = None, env_ids: Any = None) -> np.ndarray | None: ...

    @abstractmethod
    def step(self, action: Any = None) -> None: ...

    @abstractmethod
    def control_step(self, action: Any) -> None: ...

    @abstractmethod
    def get_state(self, env_ids: Any = None) -> dict[str, Any]: ...

    @abstractmethod
    def set_state(self, state: dict[str, Any], env_ids: Any = None) -> None: ...

    @abstractmethod
    def apply_wrench(self, body: str, force: tuple[float, float, float], point: tuple[float, float, float] | None = None, *, env_ids: Any = None,
                     frame: str = "world", duration_steps: int | None = None) -> None: ...

    @abstractmethod
    def diagnostics(self) -> dict[str, Any]: ...

    @abstractmethod
    def render(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...
