"""Small, reproducible CEM calibration pipeline for stage 09.

The synthetic generator uses the same native board release/coast wrapper that
the user runs interactively.  It is deliberately a method test, not evidence
of real-world sim-to-real improvement.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import experiment_config
from .randomize import PARAMETER_RANGES, nominal_parameters, resolve_parameters, save_parameters


FIT_NAMES_BY_EXPERIMENT = {
    "coast": ("wheel_spin_damping_scale", "bearing_torque_scale", "truck_damping_scale"),
    "release": ("truck_stiffness_scale", "truck_damping_scale", "bearing_torque_scale"),
}


def fit_names(experiment: str) -> tuple[str, ...]:
    try:
        return FIT_NAMES_BY_EXPERIMENT[experiment]
    except KeyError as exc:
        raise ValueError(f"unsupported calibration experiment {experiment!r}") from exc


def _rollout(parameters: dict[str, float], speed: float, *, experiment: str = "coast",
             initial_roll: float = 0.0, steps: int = 160) -> dict[str, float]:
    from ..backends.mujoco import MujocoBackend

    dynamics = {
        **vars(experiment_config("truck", experiment)),
        "initial_speed": float(speed),
    }
    backend = MujocoBackend(scene="board", style="standard", experiment=experiment,
                            initial_roll=float(initial_roll), dynamics=dynamics, params=parameters)
    try:
        for _ in range(steps):
            backend.step()
        state = backend.get_state()
        diag = backend.diagnostics()
        return {
            "time": float(state["time"]),
            "x": float(state["qpos"][0]),
            "vx": float(state["qvel"][0]),
            "wheel_speed": float(np.mean(list(diag["wheel_speed_rad_s"].values()))),
            "deck_roll": float(diag["deck_roll_rad"]),
            "steer_front": float(diag["steer_rad"]["front"]),
            "steer_rear": float(diag["steer_rad"]["rear"]),
        }
    finally:
        backend.close()


def generate_synthetic(*, seed: int = 0, train_count: int = 4,
                       heldout_count: int = 2, experiment: str = "release") -> dict[str, Any]:
    fit_names_for_experiment = fit_names(experiment)
    rng = np.random.default_rng(seed)
    truth = nominal_parameters()
    if experiment == "coast":
        truth.update({"wheel_spin_damping_scale": 1.18,
                      "bearing_torque_scale": 1.24,
                      "truck_damping_scale": 0.86})
    else:
        truth.update({"truck_stiffness_scale": 1.18,
                      "truck_damping_scale": 0.86,
                      "bearing_torque_scale": 1.16})
    if experiment == "release":
        train_excitation = np.linspace(-.14, .14, max(train_count, 2), dtype=float)
        heldout_excitation = np.linspace(-.07, .07, max(heldout_count, 1), dtype=float)
    else:
        train_excitation = np.linspace(.70, 1.40, max(train_count, 2), dtype=float)
        heldout_excitation = np.linspace(.82, 1.28, max(heldout_count, 1), dtype=float)
    trials: list[dict[str, Any]] = []
    for index in range(train_count + heldout_count):
        split_index = index if index < train_count else index - train_count
        speed = (float(train_excitation[split_index]) if index < train_count
                 else float(heldout_excitation[split_index])) if experiment == "coast" else 0.0
        initial_roll = (float(train_excitation[split_index]) if index < train_count
                        else float(heldout_excitation[split_index])) if experiment == "release" else 0.0
        excitation = "train" if index < train_count else "held_out"
        trials.append({"trial_id": f"synthetic-{index:03d}", "split": excitation,
                       "initial_speed": speed, "initial_roll": initial_roll,
                       "target": _rollout(truth, speed, experiment=experiment,
                                          initial_roll=initial_roll)})
    return {"schema": "skate-sim.stage09.synthetic.v1", "origin": "synthetic",
            "seed": seed, "experiment": experiment, "fit_names": list(fit_names_for_experiment),
            "truth": truth, "trials": trials,
            "split_rule": "complete trials; held-out excitations are fixed outside the training set"}


def _load_data(source: str | Path, *, seed: int, experiment: str) -> dict[str, Any]:
    if source == "synthetic":
        return generate_synthetic(seed=seed, experiment=experiment)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"calibration data not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "trials" not in data:
            raise ValueError("real calibration JSON must contain a complete-trial 'trials' list")
        if not isinstance(data.get("units"), dict) or not data["units"]:
            raise ValueError("real calibration JSON must declare a units mapping")
        trial_ids = [trial.get("trial_id") for trial in data["trials"]]
        if (not data["trials"] or any(not isinstance(trial, dict) for trial in data["trials"])
                or any(trial.get("split") not in ("train", "held_out", "validation", "test")
                       or "trial_id" not in trial or "initial_speed" not in trial
                       or not isinstance(trial.get("target"), dict)
                       for trial in data["trials"])
                or len(set(trial_ids)) != len(trial_ids)):
            raise ValueError("real calibration trials require split, initial_speed and target fields")
        data["origin"] = "target_real"
        data.setdefault("experiment", experiment)
        return data
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        required = {"trial_id", "split", "initial_speed", "x", "vx", "wheel_speed"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"CSV requires columns: {', '.join(sorted(required))}")
        units_candidates = (path.with_suffix(path.suffix + ".units.json"),
                           path.with_suffix(".units.json"))
        units_path = next((candidate for candidate in units_candidates if candidate.exists()), None)
        if units_path is None:
            raise ValueError("CSV calibration requires a sibling .units.json file with declared SI units")
        units = json.loads(units_path.read_text(encoding="utf-8"))
        if not isinstance(units, dict) or not units:
            raise ValueError("CSV units sidecar must contain a non-empty mapping")
        trial_ids = [row["trial_id"] for row in rows]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("CSV calibration requires one complete trial per row with unique trial_id")
        if any(row["split"] not in ("train", "held_out", "validation", "test") for row in rows):
            raise ValueError("CSV split must be train, held_out, validation, or test")
        trials = []
        for row in rows:
            trials.append({"trial_id": row["trial_id"], "split": row["split"],
                           "initial_speed": float(row["initial_speed"]),
                           "target": {key: float(row[key]) for key in ("x", "vx", "wheel_speed")}})
        return {"schema": "skate-sim.stage09.measurements.v1", "origin": "target_real",
                "experiment": experiment, "trials": trials, "units": units}
    raise ValueError("calibration data must be 'synthetic', JSON, or CSV")


def _loss(candidate: dict[str, float], trials: list[dict[str, Any]], *, experiment: str) -> float:
    errors = []
    for trial in trials:
        pred = _rollout(candidate, float(trial["initial_speed"]), experiment=experiment,
                        initial_roll=float(trial.get("initial_roll", 0.0)), steps=160)
        target = trial["target"]
        if experiment == "coast":
            errors.extend(((pred["x"] - target["x"]) / .02,
                           (pred["vx"] - target["vx"]) / .05,
                           (pred["wheel_speed"] - target["wheel_speed"]) / 1.0))
        else:
            errors.extend(((pred["deck_roll"] - target["deck_roll"]) / .005,
                           (pred["steer_front"] - target["steer_front"]) / .005,
                           (pred["steer_rear"] - target["steer_rear"]) / .005))
    return float(np.mean(np.square(errors)))


def fit_cem(data: dict[str, Any], *, seed: int = 0, iterations: int = 4,
            population: int = 8) -> dict[str, Any]:
    train = [trial for trial in data["trials"] if trial.get("split") == "train"]
    heldout = [trial for trial in data["trials"] if trial.get("split") in ("held_out", "validation", "test")]
    if not train or not heldout:
        raise ValueError("calibration requires non-empty train and held-out complete-trial splits")
    rng = np.random.default_rng(seed)
    experiment = data.get("experiment", "coast")
    names = fit_names(experiment)
    lower = np.asarray([PARAMETER_RANGES[name][0] for name in names])
    upper = np.asarray([PARAMETER_RANGES[name][1] for name in names])
    mean = np.ones(len(names))
    std = np.full(len(names), .20)
    history = []
    for iteration in range(iterations):
        samples = np.clip(rng.normal(mean, std, size=(population, len(names))), lower, upper)
        losses = []
        for row in samples:
            candidate = nominal_parameters()
            candidate.update(dict(zip(names, row.tolist())))
            losses.append(_loss(candidate, train, experiment=experiment))
        order = np.argsort(losses)[:max(2, population // 2)]
        elite = samples[order]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), .025)
        history.append({"iteration": iteration, "best_train_loss": float(min(losses)),
                        "mean": mean.tolist(), "std": std.tolist()})
    fitted = nominal_parameters()
    fitted.update(dict(zip(names, mean.tolist())))
    baseline = nominal_parameters()
    truth = data.get("truth") if data.get("origin") == "synthetic" else None
    result = {"parameters": fitted, "fit_names": list(names),
            "train_loss": _loss(fitted, train, experiment=experiment),
            "heldout_loss": _loss(fitted, heldout, experiment=experiment),
            "baseline_train_loss": _loss(baseline, train, experiment=experiment),
            "baseline_heldout_loss": _loss(baseline, heldout, experiment=experiment),
            "history": history, "method": "cem", "origin": data.get("origin", "unknown"),
            "experiment": experiment,
            "heldout_trials": [trial["trial_id"] for trial in heldout]}
    if truth:
        result["truth_parameter_rmse"] = float(np.sqrt(np.mean(
            [(fitted[name] - truth[name]) ** 2 for name in names])))
    return result


def calibrate(*, data_source: str | Path, output: str | Path, seed: int = 0,
              method: str = "cem", experiment: str = "release") -> dict[str, Any]:
    if method != "cem":
        raise ValueError("only the bounded derivative-free method 'cem' is implemented")
    data = _load_data(data_source, seed=seed, experiment=experiment)
    fit = fit_cem(data, seed=seed)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    data_name = "synthetic_data.json" if data.get("origin") == "synthetic" else "input_data.json"
    (out / data_name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    save_parameters(out / "params.yaml", fit["parameters"], source=f"{data.get('origin')}_cem",
                    seed=seed, randomize=False,
                    extra={"fit": fit, "data_schema": data.get("schema")})
    (out / "fit.json").write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "origin": data.get("origin"), "method": method,
            "output": str(out), "experiment": experiment, "train_loss": fit["train_loss"],
            "heldout_loss": fit["heldout_loss"], "heldout_trials": fit["heldout_trials"],
            "parameters": fit["parameters"]}
