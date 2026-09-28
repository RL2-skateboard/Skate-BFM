from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    name: str

    @abstractmethod
    def reset(self, seed: int = 0, env_ids: Any = None) -> None: ...

    @abstractmethod
    def step(self, action: Any = None) -> None: ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def apply_wrench(self, body: str, force: tuple[float, float, float], point: tuple[float, float, float] | None = None) -> None: ...

    @abstractmethod
    def diagnostics(self) -> dict[str, Any]: ...

    @abstractmethod
    def render(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...
