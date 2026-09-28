from __future__ import annotations

import argparse
import time
import webbrowser
import hashlib
import json
from skate_sim.backends import create_backend
from skate_sim.record import Recorder


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="skate-sim interactive workbench")
    p.add_argument("--backend", choices=["mujoco", "mjlab"], default="mujoco")
    p.add_argument("--scene", choices=["bench", "board", "contact", "robot", "rider", "gallery"], default="bench")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record")
    p.add_argument("--duration", type=float, help="Close after this many simulation seconds")
    p.add_argument("--replay")
    p.add_argument("--mode", choices=['visual'], default='visual')
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--style", choices=["standard", "cruiser", "longboard", "street", "downhill"], default="standard")
    p.add_argument("--model", choices=["reduced", "truck"], default="truck")
    p.add_argument("--experiment", choices=["manual", "coast", "lean", "release", "turn", "drop"])
    return p


def main() -> int:
    args = parser().parse_args()
    if args.replay:
        from skate_sim.replay import replay
        replay(args.replay)
        return 0
    if args.scene not in ("bench", "board", "gallery"):
        raise SystemExit("stage 02 currently implements bench, board, and gallery scenes")
    if args.num_envs != 1:
        raise SystemExit("Current workbench supports --num-envs 1")
    extra = {}
    if args.experiment:
        if args.backend != "mujoco" or args.scene != "board":
            raise SystemExit("Stage 03 experiments currently require --backend mujoco --scene board")
        from skate_sim.physics.dynamics import experiment_config
        from dataclasses import asdict
        extra = dict(dynamics=asdict(experiment_config(args.model, args.experiment)),
                     experiment=args.experiment)
    backend = create_backend(args.backend, seed=args.seed, scene=args.scene, style=args.style,
                             inspect=args.inspect, **extra)
    backend.paused = args.inspect
    recorder = Recorder(args.record, backend=args.backend, scene=args.scene, seed=args.seed)
    if args.experiment:
        recorder.write("experiment", model=args.model, experiment=args.experiment,
                       dynamics=extra["dynamics"], initial_state=backend.get_state())
    if args.scene in ("board", "gallery"):
        from skate_sim.models.board import board_hash
        print(backend.board_style.as_dict(), "hash=", board_hash(backend.board_style), flush=True)
        print("Inspection: Space dynamics/pause; 1/2/3/4 rotate individual wheels while paused; V transparent; J joint axes; R reset", flush=True)
        recorder.write("asset", config=backend.board_style.as_dict(), config_hash=board_hash(backend.board_style))
    print(f"skate-sim | backend={args.backend} | scene={args.scene} | Space pause, N physics step, M control step, R reset, F: +X 20N for 0.2s (MuJoCo/mjlab), E release")
    try:
        next_frame = time.perf_counter()
        if args.backend == "mjlab":
            from skate_sim.models.board import board_hash
            backend.render_web(board_hash(backend.board_style) if getattr(backend,"board_style",None) else "bench")
            while True:
                requested_style = backend.take_requested_style()
                if requested_style:
                    backend.switch_style(requested_style)
                backend.render_web(board_hash(backend.board_style) if getattr(backend,"board_style",None) else "bench")
                recorder.write("step",state=backend.get_state())
                time.sleep(.01)
        while getattr(backend, "is_running", lambda: backend.viewer is None or backend.viewer.is_running())():
            if getattr(backend, "viewer", None) is None and args.backend != "isaacsim":
                backend.render()
            if args.backend != "isaacsim" and backend.viewer is None:
                break
            # The passive viewer gives us a real window; keyboard callbacks are
            # intentionally kept in the small loop so the backend contract stays pure.
            if not backend.paused:
                backend.step()
                recorder.write("step", state=backend.get_state(), diagnostics=backend.diagnostics())
                if args.duration is not None and backend.get_state()['time'] >= args.duration:
                    break
            if time.perf_counter() >= next_frame:
                backend.render()
                next_frame = time.perf_counter() + 1/60
            time.sleep(backend.config.physics_dt)
    finally:
        recorder.close()
        backend.close()
        time.sleep(.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
