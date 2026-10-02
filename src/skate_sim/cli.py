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
