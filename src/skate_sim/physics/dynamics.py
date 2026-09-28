from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsConfig:
    model: str = "truck"
    wheel_spin_damping: float = .00002
    bearing_torque: float = .0004
    truck_k1: float = 2.0
    truck_k3: float = .25
    truck_c: float = .08
    lean_target: float = .08
    fixture_k: float = 8.0
    fixture_c: float = .6
    initial_speed: float = 1.0

    def validate(self) -> None:
        if self.model not in ("reduced", "truck"):
            raise ValueError("model must be reduced or truck")
        if min(self.wheel_spin_damping, self.bearing_torque, self.truck_k1, self.truck_k3, self.truck_c) < 0:
            raise ValueError("dissipation and restoring parameters must be non-negative")


def experiment_config(model: str, experiment: str) -> DynamicsConfig:
    cfg = DynamicsConfig(model=model)
    cfg.validate()
    if experiment == "manual":
        return DynamicsConfig(model=model, initial_speed=0.0)
    if experiment == "coast":
        return cfg
    if experiment in ("lean", "release", "turn", "drop"):
        if experiment == "turn":
            return DynamicsConfig(model=model, lean_target=.15, fixture_k=12., fixture_c=.8)
        return cfg
    raise ValueError(f"unknown dynamics experiment {experiment!r}")
