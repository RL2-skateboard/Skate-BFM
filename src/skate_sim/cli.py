from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .models.board import board_hash, board_styles, build_board_xml


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar/array values emitted by checks to JSON values."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_result(output: str | None, result: dict[str, Any]) -> None:
    if not output:
        return
    path = Path(output)
    if path.suffix.lower() != ".json":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "result.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _status(checks: dict[str, Any], *, incomplete: bool = False,
            blocked: bool = False) -> str:
    if blocked:
        return "BLOCKED"
    if incomplete:
        return "INCOMPLETE"
    return "PASS" if all(bool(value) for value in checks.values()) else "FAIL"


def _mujoco_check(stage: str, checkpoint: str | None) -> dict[str, Any]:
    import numpy as np
    from .backends.mujoco import MujocoBackend

    checks: dict[str, Any] = {}
    details: dict[str, Any] = {}

    if stage == "01":
        backend = MujocoBackend(scene="bench")
        try:
            start = backend.get_state()
            backend.step()
            checks["step_advances_time"] = backend.get_state()["time"] > start["time"]
            backend.apply_wrench("cube", (20.0, 0.0, 0.0))
            for _ in range(100):
                backend.step()
            checks["wrench_changes_position"] = bool(
                backend.get_state()["qpos"][0] > start["qpos"][0]
            )
            backend.reset()
            reset = backend.get_state()
            checks["reset_restores_time"] = abs(reset["time"]) < 1e-12
            checks["finite_state"] = bool(
                np.isfinite(reset["qpos"]).all() and np.isfinite(reset["qvel"]).all()
            )
        finally:
            backend.close()

    elif stage == "02":
        for name, style in board_styles().items():
            model = MujocoBackend(scene="board", style=name)
            try:
                model.step()
                state = model.get_state()
                checks[f"style_{name}"] = bool(np.isfinite(state["qpos"]).all() and np.isfinite(state["qvel"]).all())
                details[name] = {"nq": model.model.nq, "nv": model.model.nv, "ncon": model.data.ncon,
                                 "hash": board_hash(style)}
            finally:
                model.close()

    elif stage == "03":
        from .physics.dynamics import experiment_config
        for model_name in ("truck", "reduced"):
            config = asdict(experiment_config(model_name, "lean"))
            backend = MujocoBackend(scene="board", experiment="lean", dynamics=config)
            try:
                for _ in range(500):
                    backend.step()
                diagnostics = backend.diagnostics()
                checks[f"{model_name}_finite"] = bool(np.isfinite(backend.data.qpos).all())
                checks[f"{model_name}_lean_response"] = abs(diagnostics["deck_roll_rad"]) > 1e-5
                details[model_name] = {"deck_roll_rad": diagnostics["deck_roll_rad"],
                                       "steer_rad": diagnostics["steer_rad"]}
            finally:
                backend.close()

    elif stage == "04":
        backend = MujocoBackend(scene="contact", experiment="pull")
        try:
            backend.pull_force = 10.0
            for _ in range(250):
                backend.step()
            diagnostics = backend.diagnostics()
            checks["contact_exists"] = diagnostics["contacts"] > 0
            checks["normal_force_positive"] = diagnostics["normal_force_N"] > 0.0
            checks["pull_reaches_contact"] = diagnostics["tangential_force_N"] > 1.0
            checks["finite_state"] = bool(np.isfinite(backend.data.qpos).all())
            details["diagnostics"] = {key: diagnostics[key] for key in (
                "contacts", "normal_force_N", "tangential_force_N", "relative_slip_speed_m_s")}
        finally:
            backend.close()

    elif stage == "05":
        from .validation.observation05 import numeric_oracle, native_contract_probe, test_noise_preserves_actions_and_context, tracking_reference_probe
        details["independent_721_oracle"] = numeric_oracle()
        details["native_128_oracle"] = numeric_oracle(backend_entry=True)
        details["tracking_reference"] = tracking_reference_probe()
        details["native_contract"] = native_contract_probe()
        test_noise_preserves_actions_and_context()
        checks["independent_721_oracle"] = True
        checks["native_contract"] = True
        checks["sensor_noise_preserves_latent_and_actions"] = True
        backend = MujocoBackend(scene="robot", controller="pose")
        try:
            for _ in range(20):
                backend.step()
            diagnostics = backend.diagnostics()
            checks["29_joint_mapping"] = len(backend.robot_joint_names) == 29
            checks["finite_state"] = bool(np.isfinite(backend.data.qpos).all() and np.isfinite(backend.data.qvel).all())
            checks["pose_torque_path"] = diagnostics["torque_injection_error"] <= 1e-12
            details["pose"] = {"qfrc_applied_norm": diagnostics["qfrc_applied_norm"],
                               "physics_steps": diagnostics["physics_step_counter"]}
        finally:
            backend.close()
        if checkpoint:
            from .validation.observation05 import policy_rollout
            rollout = policy_rollout(checkpoint)
            checks["bfm_30s_1500_inferences"] = rollout["inferences"] == 1500
            checks["bfm_finite"] = rollout["finite"]
            details["policy"] = rollout
        else:
            return {"status": "BLOCKED", "stage": stage, "backend": "mujoco",
                    "checks": checks, "details": details,
                    "reason": "stage 05 policy check requires --checkpoint"}

    elif stage == "06":
        if not checkpoint:
            return {"status": "BLOCKED", "stage": stage, "backend": "mujoco",
                    "reason": "stage 06 actual policy replay requires --checkpoint"}
        from .validation.noncontact06 import run
        details = run(checkpoint)
        checks = details["checks"]

    else:
        return {"status": "INCOMPLETE", "stage": stage, "backend": "mujoco",
                "reason": "no active MuJoCo check is defined for this stage"}

    result = {"status": _status(checks), "stage": stage, "backend": "mujoco",
              "checks": checks, "details": details}
    return result


def _mjlab_check(stage: str, num_envs: int, checkpoint: str | None) -> dict[str, Any]:
    import numpy as np
    from .backends.mjlab import MjlabBackend

    if stage in ("05", "06", "07"):
        if not checkpoint:
            return {"status": "BLOCKED", "stage": stage, "backend": "mjlab",
                    "reason": "actual policy checkpoint required"}
        from .validation.gpu_contract import run as gpu_contract
        details = gpu_contract(checkpoint, num_envs=max(64, num_envs) if stage == "07" else max(1, num_envs),
                               duration=30., scene="robot" if stage == "05" else "rider")
        checks = {key: bool(details[key]) for key in ("finite", "subset_atomic_validation",
                  "partial_reset_isolation", "substep_pd", "wrench_isolation_duration")}
        checks.update(cuda_physics=str(details["physics_device"]).startswith("cuda"),
                      nonzero_policy=details["peak_raw_action"] > 0,
                      nonzero_torque=details["peak_tau"] > 0)
        if stage == "05":
            from .validation.observation05 import numeric_oracle
            details["shared_observation_oracle"] = numeric_oracle()
            checks["shared_observation_oracle"] = True
        return {"status": _status(checks), "stage": stage, "backend": "mjlab",
                "checks": checks, "details": details, "submatrix": "noncontact_batch_control_contract",
                "disclosure": "Noncontact automatic contract only; contact model acceptance and manual UI confirmation remain outstanding."}
    if stage == "01":
        backend = MjlabBackend(scene="bench", num_envs=max(1, num_envs))
        try:
            start = backend.get_state()
            backend.step()
            after_step = backend.get_state()
            backend.apply_wrench("cube", (20.0, 0.0, 0.0))
            for _ in range(100):
                backend.step()
            after_force = backend.get_state()
            backend.reset()
            reset = backend.get_state()
            checks = {
                "cuda_physics": backend.diagnostics()["physics_owner"] == "mjlab.mujoco_warp",
                "step_advances_time": bool(np.all(after_step["time"] > start["time"])),
                "wrench_changes_position": bool(after_force["qpos"][backend.view_env, 0] > start["qpos"][backend.view_env, 0]),
                "reset_restores_time": bool(np.all(abs(reset["time"]) < 1e-12)),
                "finite_state": bool(np.isfinite(reset["qpos"]).all() and np.isfinite(reset["qvel"]).all()),
            }
            details = {"num_envs": num_envs, "device": backend.diagnostics()["device"]}
        finally:
            backend.close()
        return {"status": _status(checks), "stage": stage, "backend": "mjlab",
                "checks": checks, "details": details}
    return {"status": "INCOMPLETE", "stage": stage, "backend": "mjlab",
            "reason": "dedicated acceptance matrix not implemented for this backend/stage"}


def _stage09_native_parameter_effects() -> dict[str, bool]:
    """Verify each board randomization field reaches a live MuJoCo target."""
    import numpy as np
    from .backends.mujoco import MujocoBackend
    from .physics.dynamics import experiment_config
    from .physics.randomize import nominal_parameters

    base = asdict(experiment_config("truck", "coast"))
    probes = {
        "deck_mass_scale": 1.10,
        "truck_mass_scale": 1.08,
        "wheel_mass_scale": 1.08,
        "truck_stiffness_scale": 1.10,
        "truck_damping_scale": 1.10,
        "wheel_spin_damping_scale": 1.10,
        "bearing_torque_scale": 1.10,
        "friction_scale": 1.10,
        "deck_com_offset_x": 0.01,
        "deck_com_offset_y": 0.006,
    }
    effects: dict[str, bool] = {}
    for name, value in probes.items():
        params = nominal_parameters()
        params[name] = value
        backend = MujocoBackend(scene="board", style="standard", experiment="coast",
                                dynamics=base, params=params)
        try:
            model = backend.model
            deck_id = model.body("deck").id
            if name == "deck_mass_scale":
                effects[name] = bool(abs(model.body_mass[deck_id] - backend._nominal_body_mass[deck_id]) > 1e-8)
            elif name == "truck_mass_scale":
                effects[name] = bool(any(
                    abs(model.body_mass[model.body(f"{prefix}_truck").id] -
                        backend._nominal_body_mass[model.body(f"{prefix}_truck").id]) > 1e-8
                    for prefix in ("front", "rear")))
            elif name == "wheel_mass_scale":
                effects[name] = bool(any(
                    abs(model.body_mass[model.body(f"{prefix}_{side}").id] -
                        backend._nominal_body_mass[model.body(f"{prefix}_{side}").id]) > 1e-8
                    for prefix in ("front", "rear") for side in ("left", "right")))
            elif name == "truck_stiffness_scale":
                effects[name] = bool(any(
                    abs(model.jnt_stiffness[model.joint(f"{prefix}_truck_tilt").id] -
                        backend._nominal_jnt_stiffness[model.joint(f"{prefix}_truck_tilt").id]) > 1e-8
                    for prefix in ("front", "rear")))
            elif name in ("truck_damping_scale", "wheel_spin_damping_scale"):
                if name == "truck_damping_scale":
                    addresses = [model.joint(f"{prefix}_truck_tilt").dofadr[0] for prefix in ("front", "rear")]
                else:
                    addresses = [model.joint(f"{prefix}_{side}_spin").dofadr[0]
                                 for prefix in ("front", "rear") for side in ("left", "right")]
                effects[name] = bool(any(
                    abs(model.dof_damping[address] - backend._nominal_dof_damping[address]) > 1e-12
                    for address in addresses))
            elif name == "friction_scale":
                effects[name] = bool(np.max(np.abs(model.geom_friction - backend._nominal_geom_friction)) > 1e-8)
            elif name in ("deck_com_offset_x", "deck_com_offset_y"):
                axis = 0 if name.endswith("x") else 1
                effects[name] = bool(abs(model.body_ipos[deck_id, axis] -
                                         backend._nominal_body_ipos[deck_id, axis]) > 1e-8)
            else:
                backend.step()
                wheel_dof = model.joint("front_left_spin").dofadr[0]
                effects[name] = bool(abs(backend.data.qfrc_applied[wheel_dof]) > 1e-12)
        finally:
            backend.close()
    return effects


def _stage09_policy_parameter_effects(checkpoint: str) -> dict[str, bool]:
    """Verify observation noise, control delay and PD scales on a live policy path."""
    import numpy as np
    from .backends.mujoco import MujocoBackend
    from .physics.randomize import nominal_parameters

    def make(params):
        return MujocoBackend(scene="rider", controller="policy", checkpoint=checkpoint,
                             preset="steer_start", params=params)

    nominal = nominal_parameters()
    noisy = nominal_parameters(); noisy["observation_noise_std"] = .01
    delayed = nominal_parameters(); delayed["control_delay_steps"] = 2.0
    scaled = nominal_parameters(); scaled["pd_kp_scale"] = 1.20; scaled["pd_kd_scale"] = .80
    backends = [make(nominal), make(noisy), make(delayed), make(scaled)]
    try:
        for _ in range(12):
            for backend in backends:
                backend.step()
        clean_obs, noisy_obs = backends[0].last_obs, backends[1].last_obs
        clean_action, delayed_action = backends[0].policy_action, backends[2].policy_action
        clean_tau, scaled_tau = backends[0].last_tau, backends[3].last_tau
        return {
            "observation_noise_std": bool(np.max(np.abs(clean_obs - noisy_obs)) > 1e-7),
            "control_delay_steps": bool(np.linalg.norm(clean_action - delayed_action) > 1e-7),
            "pd_kp_kd_scale": bool(np.linalg.norm(clean_tau - scaled_tau) > 1e-7),
        }
    finally:
        for backend in backends:
            backend.close()


def _stage09_mujoco_check(checkpoint: str | None = None) -> dict[str, Any]:
    import numpy as np
    from .backends.mujoco import MujocoBackend
    from .physics.calibrate import fit_cem, generate_synthetic
    from .physics.dynamics import experiment_config
    from .physics.randomize import PARAMETER_RANGES

    base = asdict(experiment_config("truck", "release"))
    nominal = MujocoBackend(scene="board", style="standard", experiment="release",
                            dynamics=base, seed=17)
    randomized = MujocoBackend(scene="board", style="standard", experiment="release",
                               dynamics=base, randomize=True, seed=17)
    try:
        nominal_roll = []
        random_roll = []
        for _ in range(80):
            nominal.step()
            randomized.step()
            nominal_roll.append(nominal.diagnostics()["deck_roll_rad"])
            random_roll.append(randomized.diagnostics()["deck_roll_rad"])
        nominal_diag = nominal.diagnostics()
        random_diag = randomized.diagnostics()
        random_params = random_diag["parameters"]
        checks = {
            "parameter_ranges": all(PARAMETER_RANGES[name][0] <= value <= PARAMETER_RANGES[name][1]
                                     for name, value in random_params.items()),
            "parameter_response": bool(abs(random_roll[-1] - nominal_roll[-1]) > 1e-6),
            "finite_state": bool(np.isfinite(randomized.get_state()["qpos"]).all()),
        }
        parameter_effects = _stage09_native_parameter_effects()
        checks["each_board_parameter_applies"] = all(parameter_effects.values())
        randomized.reset(seed=17)
        first_reset = randomized.get_state()["qpos"].copy()
        first_parameters = dict(randomized.diagnostics()["parameters"])
        for _ in range(100):
            randomized.reset(seed=17)
        checks["same_seed_reproducible"] = bool(
            np.array_equal(first_reset, randomized.get_state()["qpos"]) and
            first_parameters == randomized.diagnostics()["parameters"])
        details = {"nominal_roll": nominal_roll[-1], "random_roll": random_roll[-1],
                   "parameters": random_params, "reset_count": 100,
                   "parameter_effects": parameter_effects,
                   "nominal_sim_time": nominal_diag["sim_time"],
                   "random_sim_time": random_diag["sim_time"]}
    finally:
        nominal.close()
        randomized.close()
    data = generate_synthetic(seed=23, train_count=4, heldout_count=2, experiment="release")
    fit = fit_cem(data, seed=23, iterations=4, population=8)
    checks["synthetic_calibration"] = bool(
        np.isfinite(fit["train_loss"]) and np.isfinite(fit["heldout_loss"]) and
        np.isfinite(fit["truth_parameter_rmse"]) and fit["heldout_loss"] < 0.1 and
        fit["heldout_loss"] < fit["baseline_heldout_loss"] and
        fit["truth_parameter_rmse"] < 0.25 and fit["heldout_trials"])
    details["calibration"] = {"method": "cem", "train_loss": fit["train_loss"],
                               "heldout_loss": fit["heldout_loss"],
                               "baseline_heldout_loss": fit["baseline_heldout_loss"],
                               "truth_parameter_rmse": fit["truth_parameter_rmse"],
                               "experiment": fit["experiment"],
                               "heldout_trials": fit["heldout_trials"]}
    if checkpoint:
        policy_effects = _stage09_policy_parameter_effects(checkpoint)
        checks["each_policy_parameter_applies"] = all(policy_effects.values())
        details["policy_parameter_effects"] = policy_effects
    else:
        details["policy_parameter_effects"] = "not checked: supply --checkpoint"
    return {"status": _status(checks, blocked=not checkpoint), "stage": "09", "backend": "mujoco",
            "checks": checks, "details": details,
            "disclosure": "synthetic calibration only; no real-data sim2real claim"}


def _stage09_mjlab_check(num_envs: int) -> dict[str, Any]:
    import numpy as np
    from .backends.mjlab import MjlabBackend

    count = max(4, int(num_envs))
    backend = MjlabBackend(scene="board", num_envs=count, view_env=0,
                           randomize=True, seed=17)
    try:
        before_params = [dict(row) for row in backend.parameter_batch]
        qpos = backend.sim.data.qpos._tensor.clone()
        qvel = backend.sim.data.qvel._tensor.clone()
        for qpos_index, _ in backend._board_truck_joints:
            qpos[:, qpos_index] = .1
        for _, qvel_index in backend._board_truck_joints:
            qvel[:, qvel_index] = .2
        qvel[:, backend._board_wheel_dof[0]] = 2.0
        backend._write_field("qpos", qpos)
        backend._write_field("qvel", qvel)
        backend.sim.forward()
        backend._sync_viewer()
        backend.step()
        qfrc = backend.sim.data.qfrc_applied._tensor.detach().cpu().numpy()
        truck_dof = backend._board_truck_joints[0][1]
        wheel_dof = backend._board_wheel_dof[0]
        root_torque = backend._board_root_dof + 3
        runtime_effects = {
            "truck_stiffness_damping": bool(np.std(qfrc[:, truck_dof]) > 1e-8),
            "wheel_loss": bool(np.std(qfrc[:, wheel_dof]) > 1e-8),
            "com_gravity_torque": bool(np.std(qfrc[:, root_torque]) > 1e-8),
        }
        for _ in range(4):
            backend.step()
        before = backend.get_state()
        ids = np.asarray([1, count - 1], dtype=np.int64)
        untouched = np.ones(count, dtype=bool)
        untouched[ids] = False
        backend.reset(env_ids=ids)
        after = backend.get_state()
        after_params = [dict(row) for row in backend.parameter_batch]
        checks = {
            "batch_parameter_diversity": len({row["friction_scale"] for row in before_params}) > 1,
            "partial_reset_state_isolated": bool(np.array_equal(
                before["qpos"][untouched], after["qpos"][untouched])),
            "partial_reset_parameters_isolated": all(
                before_params[index] == after_params[index]
                for index in range(count) if index not in set(ids.tolist())),
            "runtime_parameter_response": all(runtime_effects.values()),
            "finite_state": bool(np.isfinite(after["qpos"]).all() and np.isfinite(after["qvel"]).all()),
        }
        details = {"num_envs": count, "view_env": backend.view_env,
                   "parameter_summary": backend.diagnostics()["parameter_summary"],
                   "runtime_parameter_effects": runtime_effects,
                   "reset_ids": ids.tolist()}
    finally:
        backend.close()
    return {"status": _status(checks), "stage": "09", "backend": "mjlab",
            "checks": checks, "details": details,
            "disclosure": "mjlab randomized drag/loss terms are explicit runtime forces; no real-data claim"}


def _run_check(stage: str, backend: str, num_envs: int, checkpoint: str | None, scope: str = "full") -> dict[str, Any]:
    if backend == "all":
        children = [_backend_process("check", name, stage=stage, num_envs=num_envs,
                                     checkpoint=checkpoint, scope=scope)
                    for name in ("mujoco", "mjlab")]
        results = [item for child in children for item in child.get("result", {}).get("results", [])]
        statuses = {child.get("result", {}).get("status", "FAIL") for child in children}
        overall = "PASS" if statuses == {"PASS"} and all(c["exit_code"] == 0 for c in children) else "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else "INCOMPLETE"
        return {"status": overall, "stage": stage, "backend": backend, "scope": scope,
                "results": results, "processes": children, "scope_status": overall,
                "stage_acceptance_status": "INCOMPLETE", "automatic_checks_status": overall,
                "overall_project_status": "INCOMPLETE"}
    stages = [stage] if stage != "all" else ["01", "02", "03", "04", "05", "06", "07", "09"]
    results = []
    for current in stages:
        if current == "04":
            results.append({"status": "DEFERRED", "stage": current, "backend": backend,
                            "reason": "contact model and contact learning frozen pending user discussion"})
            continue
        if backend == "mjlab" and current not in ("01", "05", "06", "07", "09"):
            results.append({"status": "INCOMPLETE", "stage": current, "backend": backend,
                            "reason": "dedicated acceptance matrix not implemented for this backend/stage"})
            continue
        if current == "09" and backend in ("mujoco", "all"):
            results.append(_stage09_mujoco_check(checkpoint))
        elif backend in ("mujoco", "all") and current != "07":
            results.append(_mujoco_check(current, checkpoint))
        if current == "09" and backend in ("mjlab", "all"):
            results.append(_stage09_mjlab_check(num_envs))
        elif backend in ("mjlab", "all") and current in ("01", "05", "06", "07"):
            results.append(_mjlab_check(current, num_envs, checkpoint))
            if current in ("06", "07"):
                results.append({"status": "NOT_RUN", "stage": current, "backend": "mjlab",
                                "submatrix": "strict_dynamics_replay", "acceptance_gate": True,
                                "reason": "batch control contracts do not certify initial-state/action/event replay; a dedicated strict replay check is required"})
                results.append({"status": "NOT_RUN", "stage": current, "backend": "mjlab",
                                "submatrix": "manual_ui_confirmation", "acceptance_gate": True,
                                "reason": "manual window acceptance is not provided by automatic batch checks"})
        if current == "09":
            results.append({"status": "DEFERRED", "stage": "09", "backend": backend,
                            "scope": "contact_learning", "reason": "contact learning frozen pending user discussion; noncontact parameter checks executed separately"})
        if backend == "mujoco" and current == "07":
            results.append({"status": "NOT_APPLICABLE", "stage": current, "backend": backend,
                            "reason": "stage 07 is the mjlab GPU batch stage"})
    evaluated = [item for item in results if item["status"] != "NOT_APPLICABLE"
                 and not (scope == "noncontact" and (item["status"] == "DEFERRED" or item.get("acceptance_gate")))]
    statuses = {item["status"] for item in evaluated}
    overall = "PASS" if statuses == {"PASS"} else "BLOCKED" if "BLOCKED" in statuses else "FAIL" if "FAIL" in statuses else "INCOMPLETE"
    # This command certifies automatic checks, never user stage sign-off.
    stage_acceptance = "INCOMPLETE"
    return {"status": overall, "scope_status": overall, "automatic_checks_status": overall, "stage_acceptance_status": stage_acceptance,
            "overall_project_status": "INCOMPLETE",
            "stage": stage, "backend": backend, "scope": scope, "results": results}


def _backend_process(command: str, backend: str, *, stage=None, num_envs=1,
                     checkpoint=None, scope="full", display=False) -> dict[str, Any]:
    """Each backend owns a fresh interpreter; children never dispatch all again."""
    with tempfile.TemporaryDirectory(prefix="skate-check-") as directory:
        target = Path(directory) / "result.json"
        argv = [sys.executable, "-m", "skate_sim", command, "--backend", backend]
        if command == "check":
            argv += ["--stage", stage, "--num-envs", str(num_envs), "--scope", scope,
                     "--output", str(target)]
        if checkpoint:
            argv += ["--checkpoint", checkpoint]
        if display:
            argv += ["--display"]
        try:
            child = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
            if target.exists():
                result = json.loads(target.read_text())
            else:
                try:
                    # Engines may print initialization diagnostics before final JSON.
                    result = None
                    for index, character in enumerate(child.stdout):
                        if character == "{":
                            try:
                                result = json.loads(child.stdout[index:])
                                break
                            except ValueError:
                                pass
                    if result is None:
                        raise ValueError("no terminal JSON result")
                except ValueError:
                    result = {"status": "FAIL", "error": "child produced no machine-readable result"}
            return {"backend": backend, "command": argv, "exit_code": child.returncode,
                    "result": result, "stderr": child.stderr[-16000:],
                    "stdout_tail": child.stdout[-4000:]}
        except Exception as exc:
            return {"backend": backend, "command": argv, "exit_code": -1,
                    "result": {"status": "FAIL"}, "exception": repr(exc)}


def _doctor(backend: str, checkpoint: str | None = None, display: bool = False) -> tuple[dict[str, Any], int]:
    if backend == "all":
        children = [_backend_process("doctor", name, checkpoint=checkpoint, display=display) for name in ("mujoco", "mjlab")]
        return {"processes": children}, 0 if all(c["exit_code"] == 0 for c in children) else 2
    names = (backend,)
    result: dict[str, Any] = {}
    exit_code = 0
    for name in names:
        try:
            if name == "mujoco":
                from .backends.mujoco import MujocoBackend
                instance = MujocoBackend()
                try:
                    instance.step()
                    result[name] = {"status": "AVAILABLE", **instance.diagnostics()}
                finally:
                    instance.close()
            else:
                from .backends.mjlab import MjlabBackend
                instance = MjlabBackend(scene="bench", num_envs=1)
                try:
                    instance.step()
                    result[name] = {"status": "AVAILABLE", "mjlab": version("mjlab"),
                                    "mujoco_warp": version("mujoco-warp"), **instance.diagnostics()}
                finally:
                    instance.close()
        except Exception as exc:
            result[name] = {"status": "ERROR", "error": str(exc)}
            exit_code = 2
    if checkpoint:
        from .record import checkpoint_manifest
        try:
            files = checkpoint_manifest(checkpoint)
            if not files:
                raise ValueError("no actual ONNX input found")
            result["checkpoint"] = {"status": "AVAILABLE", "files": files}
        except Exception as exc:
            result["checkpoint"] = {"status": "ERROR", "error": str(exc)}
            exit_code = 2
    if display:
        address = os.environ.get("DISPLAY")
        available = bool(address and (not address.startswith(":") or Path("/tmp/.X11-unix/X" + address[1:].split(".")[0]).exists()))
        result["display"] = {"status": "AVAILABLE" if available else "ERROR", "DISPLAY": address,
                             "scope": "display endpoint presence only; visual confirmation uses play.py"}
        if not available:
            exit_code = 2
    return result, exit_code


def _write_report(run: str, output: str) -> Path:
    root = Path(run)
    result_path = root if root.is_file() else root / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"check result not found: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    target = Path(output)
    if target.suffix.lower() != ".md":
        target.mkdir(parents=True, exist_ok=True)
        target = target / "acceptance.md"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# skate-sim acceptance report", "",
             f"- Overall status: **{result.get('status', 'UNKNOWN')}**",
             f"- Requested stage: `{result.get('stage', 'unknown')}`",
             f"- Requested backend: `{result.get('backend', 'unknown')}`",
             f"- Requested scope: `{result.get('scope', 'unknown')}`",
             f"- Stage acceptance: **{result.get('stage_acceptance_status', 'UNKNOWN')}**",
             f"- Project acceptance: **{result.get('overall_project_status', 'INCOMPLETE')}**", ""]
    rows = result.get("results", [result])
    for item in rows:
        lines.extend([f"## Stage {item.get('stage')} / {item.get('backend')}", "",
                      f"Status: **{item.get('status', 'UNKNOWN')}**", ""])
        checks = item.get("checks", {})
        if checks:
            lines.append("| Check | Result |")
            lines.append("| --- | --- |")
            lines.extend(f"| `{name}` | {'PASS' if bool(value) else 'FAIL'} |"
                         for name, value in checks.items())
            lines.append("")
        if item.get("reason"):
            lines.extend([f"Reason: {item['reason']}", ""])
        disclosure = item.get("disclosure")
        if disclosure:
            lines.extend([f"Disclosure: {disclosure}", ""])
        details = item.get("details")
        if details:
            lines.extend(["```json", json.dumps(details, indent=2, default=_json_default), "```", ""])
    if result.get("processes"):
        lines.extend(["## Backend process evidence", "", "```json",
                      json.dumps(result["processes"], indent=2, default=_json_default), "```", ""])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m skate_sim")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--backend", choices=["mujoco", "mjlab", "all"], default="mujoco")
    doctor.add_argument("--checkpoint")
    doctor.add_argument("--display", action="store_true")

    build = sub.add_parser("build")
    build.add_argument("--asset", choices=["board"], required=True)
    build.add_argument("--style", choices=["all", *board_styles().keys()], default="all")

    check = sub.add_parser("check")
    check.add_argument("--stage", choices=["01", "02", "03", "04", "05", "06", "07", "09", "all"], default="07")
    check.add_argument("--backend", choices=["mujoco", "mjlab", "all"], default="mjlab")
    check.add_argument("--num-envs", type=int, default=1)
    check.add_argument("--checkpoint")
    check.add_argument("--output")
    check.add_argument("--scope", choices=["full", "noncontact"], default="full")

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--experiment", choices=["release", "coast"], default="release")
    calibrate.add_argument("--data", required=True, help="synthetic, JSON, or audited complete-trial CSV")
    calibrate.add_argument("--split", choices=["trial"], default="trial")
    calibrate.add_argument("--method", choices=["cem"], default="cem")
    calibrate.add_argument("--seed", type=int, default=0)
    calibrate.add_argument("--output", required=True)

    report = sub.add_parser("report")
    report.add_argument("--run", required=True)
    report.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "build":
        out = Path("assets/board")
        out.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, style in board_styles(args.style).items():
            path = out / f"{name}.xml"
            path.write_text(build_board_xml(style), encoding="utf-8")
            manifest[name] = {"path": str(path), "hash": board_hash(style), **style.as_dict()}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, default=_json_default))
        return 0
    if args.command == "doctor":
        result, code = _doctor(args.backend, args.checkpoint, args.display)
        print(json.dumps(result, indent=2, default=_json_default))
        return code
    if args.command == "calibrate":
        from .physics.calibrate import calibrate as run_calibration
        result = run_calibration(data_source=args.data, output=args.output,
                                 seed=args.seed, method=args.method,
                                 experiment=args.experiment)
        print(json.dumps(result, indent=2, default=_json_default))
        return 0 if result.get("status") == "PASS" else 1
    if args.command == "report":
        try:
            path = _write_report(args.run, args.output)
            source = Path(args.run)
            acceptance = json.loads((source if source.is_file() else source / "result.json").read_text()).get("status", "UNKNOWN")
            print(json.dumps({"generation_status": "PASS", "acceptance_status": acceptance,
                              "report": str(path)}, indent=2))
            return 0 if acceptance == "PASS" else 3
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
            return 1
    try:
        result = _run_check(args.stage, args.backend, args.num_envs, args.checkpoint, args.scope)
    except Exception as exc:
        result = {"status": "FAIL", "stage": args.stage, "backend": args.backend,
                  "scope": args.scope, "exception": repr(exc)}
    _write_result(args.output, result)
    print(json.dumps(result, indent=2, default=_json_default))
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2, "INCOMPLETE": 3, "NOT_APPLICABLE": 3}.get(result["status"], 3)
