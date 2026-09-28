from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    physics_dt: float = 0.002
    control_dt: float = 0.02
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    cube_mass: float = 1.0
    cube_size: float = 0.20

    @property
    def substeps(self) -> int:
        return round(self.control_dt / self.physics_dt)
