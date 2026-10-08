"""Auditable episode parameters shared by the native and CUDA backends.

The values in this module are engineering ranges, not measurements.  A
parameter file is JSON-in-YAML (JSON is valid YAML), so the project does not
need a second mandatory parser dependency just to reproduce a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "deck_mass_scale": (0.85, 1.15),
    "truck_mass_scale": (0.90, 1.10),
    "wheel_mass_scale": (0.90, 1.10),
    "truck_stiffness_scale": (0.70, 1.30),
    "truck_damping_scale": (0.70, 1.30),
    "wheel_spin_damping_scale": (0.75, 1.25),
    "bearing_torque_scale": (0.75, 1.25),
    "friction_scale": (0.70, 1.30),
    "pd_kp_scale": (0.80, 1.20),
    "pd_kd_scale": (0.80, 1.20),
    "deck_com_offset_x": (-0.02, 0.02),
    "deck_com_offset_y": (-0.01, 0.01),
    "observation_noise_std": (0.0, 0.01),
    "control_delay_steps": (0.0, 2.0),
}


def nominal_parameters() -> dict[str, float]:
    values = {name: 1.0 for name in PARAMETER_RANGES}
    values["deck_com_offset_x"] = 0.0
    values["deck_com_offset_y"] = 0.0
    values["observation_noise_std"] = 0.0
    values["control_delay_steps"] = 0.0
    return values


def _read_mapping(source: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if source is None or source == "nominal":
        return {}
    if isinstance(source, dict):
        return dict(source.get("parameters", source))
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"parameter file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("parameter files must be JSON or require PyYAML for YAML syntax") from exc
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"parameter file must contain a mapping: {path}")
    return dict(data.get("parameters", data))


def validate_parameters(values: dict[str, Any], *, clip: bool = False) -> dict[str, float]:
    result = nominal_parameters()
    unknown = sorted(set(values) - set(PARAMETER_RANGES))
    if unknown:
        raise ValueError(f"unknown episode parameter(s): {', '.join(unknown)}")
    for name, raw in values.items():
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"parameter {name} must be finite")
        lower, upper = PARAMETER_RANGES[name]
        if clip:
            value = float(np.clip(value, lower, upper))
        elif not lower <= value <= upper:
            raise ValueError(f"parameter {name}={value} outside [{lower}, {upper}]")
        if name == "control_delay_steps":
            value = float(round(value))
        result[name] = value
    return result


def resolve_parameters(source: str | Path | dict[str, Any] | None = None,
                       *, randomize: bool = False, seed: int = 0) -> dict[str, float]:
    values = validate_parameters(_read_mapping(source))
    if randomize:
        rng = np.random.default_rng(seed)
        for name, (lower, upper) in PARAMETER_RANGES.items():
            if name == "control_delay_steps":
                values[name] = float(rng.integers(int(lower), int(upper) + 1))
            else:
                values[name] = float(rng.uniform(lower, upper))
    return values


def sample_parameter_batch(num_envs: int, *, seed: int, randomize: bool,
                           source: str | Path | dict[str, Any] | None = None) -> list[dict[str, float]]:
    if not randomize:
        base = resolve_parameters(source)
        return [dict(base) for _ in range(num_envs)]
    return [resolve_parameters(source, randomize=True, seed=seed + env_id * 1009)
            for env_id in range(num_envs)]


def effective_dynamics(base: dict[str, Any] | None,
                       parameters: dict[str, float]) -> dict[str, Any]:
    """Translate dimensionless episode scales into backend dynamics values."""
    result = dict(base or {})
    for key in ("truck_k1", "truck_c", "wheel_spin_damping", "bearing_torque",
                "fixture_k", "fixture_c"):
        scale_name = {
            "truck_k1": "truck_stiffness_scale",
            "truck_c": "truck_damping_scale",
            "wheel_spin_damping": "wheel_spin_damping_scale",
            "bearing_torque": "bearing_torque_scale",
            "fixture_k": "truck_stiffness_scale",
            "fixture_c": "truck_damping_scale",
        }[key]
        if key in result:
            result[key] = float(result[key]) * parameters[scale_name]
    result.update({
        "deck_mass_scale": parameters["deck_mass_scale"],
        "truck_mass_scale": parameters["truck_mass_scale"],
        "wheel_mass_scale": parameters["wheel_mass_scale"],
        "friction_scale": parameters["friction_scale"],
        "friction": parameters["friction_scale"],
        "deck_com_offset_x": parameters["deck_com_offset_x"],
        "deck_com_offset_y": parameters["deck_com_offset_y"],
    })
    return result


def parameter_metadata(parameters: dict[str, float], *, source: str,
                       seed: int, randomize: bool) -> dict[str, Any]:
    return {
        "schema": "skate-sim.stage09.parameters.v1",
        "source": source,
        "seed": int(seed),
        "randomized": bool(randomize),
        "ranges_are_engineering_assumptions": True,
        "parameters": {key: float(value) for key, value in parameters.items()},
    }


def save_parameters(path: str | Path, parameters: dict[str, float], *, source: str,
                    seed: int, randomize: bool, extra: dict[str, Any] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = parameter_metadata(parameters, source=source, seed=seed, randomize=randomize)
    if extra:
        payload.update(extra)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def parameter_summary(batch: list[dict[str, float]]) -> dict[str, Any]:
    if not batch:
        return {}
    keys = list(batch[0])
    return {key: {"min": float(min(row[key] for row in batch)),
                  "max": float(max(row[key] for row in batch)),
                  "mean": float(np.mean([row[key] for row in batch]))}
            for key in keys}
