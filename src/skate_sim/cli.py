from __future__ import annotations

import argparse
import importlib
import json
from importlib.metadata import version
from pathlib import Path

from .models.board import board_hash, board_styles, build_board_xml


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m skate_sim")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--backend", choices=["mujoco", "mjlab"], default="mujoco")

    build = sub.add_parser("build")
    build.add_argument("--asset", choices=["board"], required=True)
    build.add_argument("--style", choices=["all", *board_styles().keys()], default="all")

    check = sub.add_parser("check")
    check.add_argument("--stage", default="07")
    check.add_argument("--backend", choices=["mujoco", "mjlab"], default="mjlab")
    check.add_argument("--num-envs", type=int, default=1)
    check.add_argument("--output")


    args = parser.parse_args(argv)
    if args.command == "build":
        out = Path("assets/board")
        out.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, style in board_styles(args.style).items():
            path = out / f"{name}.xml"
            path.write_text(build_board_xml(style), encoding="utf-8")
            manifest[name] = {"path": str(path), "hash": board_hash(style), **style.as_dict()}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "check":
        if args.stage != "07":
            print(json.dumps({"status": "SKIP", "reason": "only stage 07 is implemented by this check"}))
            return 0
        try:
            from .backends.mjlab import MjlabBackend
            backend = MjlabBackend(scene="board", num_envs=args.num_envs)
            before = backend.get_state()["qpos"].copy()
            backend.step()
            after_step = backend.get_state()
            ids = list(range(0, args.num_envs, 2))
            backend.reset(ids)
            after_reset = backend.get_state()["qpos"]
            untouched = [i for i in range(args.num_envs) if i not in ids]
            result = {"status": "PASS", "stage": "07", "backend": "mjlab",
                      "num_envs": args.num_envs, "device": backend.diagnostics()["device"],
                      "physics_owner": backend.diagnostics()["physics_owner"],
                      "step_time": after_step["time"],
                      "partial_reset_untouched": bool(__import__('numpy').allclose(after_reset[untouched], after_step["qpos"][untouched], atol=1e-4, rtol=1e-5)) if untouched else True}
            backend.close()
        except Exception as exc:
            result = {"status": "ERROR", "stage": "07", "error": str(exc)}
            print(json.dumps(result, indent=2))
            return 2
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    try:
        if args.backend == "mujoco":
            from .backends.mujoco import MujocoBackend
            backend = MujocoBackend()
            result = {"status": "AVAILABLE", **backend.diagnostics()}
            backend.close()
        else:
            import mjlab
            result = {"status": "AVAILABLE", "mjlab": version("mjlab"),
                      "mujoco_warp": version("mujoco-warp")}
    except Exception as exc:
        result = {"status": "ERROR", "error": str(exc)}
    print(json.dumps({args.backend: result}, indent=2))
    return 0 if result["status"] == "AVAILABLE" else 2
