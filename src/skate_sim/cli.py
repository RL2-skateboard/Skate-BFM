from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

from .models.board import board_hash, board_styles, build_board_xml
from importlib.metadata import PackageNotFoundError, version


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m skate_sim", description="skate-sim non-visual tools")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--backend", choices=["mujoco", "mjlab", "isaacsim", "all"], default="mujoco")
    doctor.add_argument("--display", action="store_true")
    check = sub.add_parser("check")
    check.add_argument("--stage", default="01")
    check.add_argument("--backend", choices=["mujoco", "mjlab", "isaacsim", "all"], default="mujoco")
    check.add_argument("--output", default="runs/check")
    check.add_argument("--styles", default="all")
    build = sub.add_parser("build")
    build.add_argument("--asset", choices=["board"], required=True)
    build.add_argument("--style", choices=["all", "standard", "cruiser", "longboard", "street", "downhill"], default="all")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        out = Path("assets/board")
        out.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, style in board_styles(args.style).items():
            xml = build_board_xml(style)
            path = out / f"{name}.xml"
            path.write_text(xml, encoding="utf-8")
            manifest[name] = {"path": str(path), "hash": board_hash(style), **style.as_dict()}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "check" and args.stage == "02":
        from .models.checks import check_assets
        if args.backend != "mujoco":
            raise SystemExit("Stage 02 checks require --backend mujoco")
        results = check_assets()
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return 0 if all(v['status'] == 'PASS' for v in results.values()) else 1
    if args.command == "check" and args.stage == "03":
        from .physics.checks import check_dynamics
        if args.backend != 'mujoco':
            raise SystemExit('Stage 03 checks require MuJoCo')
        results=check_dynamics(args.styles)
        out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
        (out/'result.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps({k:{'status':v['status'],'error':v.get('error')} for k,v in results.items()},indent=2))
        return 0 if all(v['status']=='PASS' for v in results.values()) else 1
    names = (["mujoco", "mjlab", "isaacsim"] if args.backend == "all" else [args.backend])
    if args.backend == "all":
        env_root = os.environ.get("CONDA_ENVS_PATH")
        if env_root:
            env_base = env_root.split(os.pathsep)[0]
        else:
            env_base = str(__import__("pathlib").Path(sys.prefix).parent)
        env_names = {"mujoco": "skate-sim-mjlab", "mjlab": "skate-sim-mjlab",
                     "isaacsim": "skate-sim-isaac"}
        checks = {}
        for name in names:
            python = __import__("pathlib").Path(env_base) / env_names[name] / "bin" / "python"
            result = subprocess.run(
                [str(python), "-m", "skate_sim", "check", "--stage", args.stage,
                 "--backend", name, "--output", str(__import__("pathlib").Path(args.output) / name)],
                text=True, capture_output=True, check=False,
                env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y"} if name == "isaacsim" else os.environ,
            )
            try:
                checks[name] = json.loads(result.stdout)
            except json.JSONDecodeError:
                error = (result.stderr + "\n" + result.stdout)[-4000:].strip()
                checks[name] = {name: {"status": "BLOCKED", "error": error or f"backend process exited {result.returncode}"}}
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        combined = {key: val.get(key, val) for val in checks.values() for key in val}
        (out / "result.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(json.dumps(combined, indent=2))
        return 0 if all(v.get("status") == "PASS" for v in combined.values()) else 2
    results = {}
    backend = None
    for name in names:
        try:
            module = importlib.import_module(f"skate_sim.backends.{name}")
            if name == "mujoco":
                backend = module.MujocoBackend()
                state0 = backend.get_state()
                backend.step()
                state1 = backend.get_state()
                if not state1["time"] > state0["time"]:
                    raise RuntimeError("MuJoCo bench failed to advance simulation time")
                backend.apply_wrench("cube", (1.0, 0.0, 0.0))
                backend.step()
                if backend.diagnostics()["force_world_N"] != [1.0, 0.0, 0.0]:
                    raise RuntimeError("MuJoCo wrench was not applied")
                backend.reset()
                if backend.get_state()["time"] != 0.0:
                    raise RuntimeError("MuJoCo reset did not restore time")
                results[name] = {"status": "PASS", **backend.diagnostics()}
                backend.close()
            elif name == "mjlab":
                backend = module.MjlabBackend()
                state0 = backend.get_state()
                backend.step()
                state1 = backend.get_state()
                if not state1["time"] > state0["time"]:
                    raise RuntimeError("mjlab bench failed to advance simulation time")
                backend.apply_wrench("cube", (1.0, 0.0, 0.0))
                backend.step()
                if backend.diagnostics()["cube_vel"][0] <= 0:
                    raise RuntimeError("mjlab wrench caused no horizontal response")
                backend.reset()
                if backend.get_state()["time"] != 0.0:
                    raise RuntimeError("mjlab reset did not restore time")
                results[name] = {"status": "PASS", **backend.diagnostics(),
                                 "mjlab": version("mjlab"), "mujoco_warp": version("mujoco-warp")}
                backend.close()
            else:
                backend = module.IsaacSimBackend(headless=True)
                state0 = backend.get_state()
                backend.step()
                state1 = backend.get_state()
                if not state1["time"] > state0["time"]:
                    raise RuntimeError("Isaac Sim bench failed to advance simulation time")
                backend.apply_wrench("cube", (1.0, 0.0, 0.0))
                backend.step()
                if backend.get_state()["linear_velocity"][0] <= 0:
                    raise RuntimeError("Isaac Sim wrench caused no horizontal response")
                backend.reset()
                reset_state = backend.get_state()
                if abs(float(reset_state["position"][2]) - 1.0) > 1e-4 or any(abs(reset_state["linear_velocity"])):
                    raise RuntimeError("Isaac Sim reset did not restore nominal cube state")
                results[name] = {"status": "PASS", **backend.diagnostics(),
                                 "package": version("isaacsim")}
                backend.close()
        except PackageNotFoundError as exc:
            results[name] = {"status": "BLOCKED", "error": str(exc)}
        except Exception as exc:  # report each backend, never silently pass
            results[name] = {"status": "BLOCKED", "error": str(exc)}
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
                backend = None
    print(json.dumps(results, indent=2))
    if args.command == "check":
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(v["status"] == "PASS" for v in results.values()) else 2
