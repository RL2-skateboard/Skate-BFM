from __future__ import annotations

import argparse
import time
from skate_sim.backends import create_backend
from skate_sim.record import Recorder


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="skate-sim interactive workbench")
    p.add_argument("--backend", choices=["mujoco", "mjlab"], default=None)
    p.add_argument("--scene", choices=["bench", "board", "contact", "robot", "rider", "gallery"], default="bench")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record")
    p.add_argument("--duration", type=float, help="Close after this many simulation seconds")
    p.add_argument("--replay")
    p.add_argument("--mode", choices=['visual', 'dynamics'], default='visual')
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--view-env", type=int, default=0)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--style", choices=["standard", "cruiser", "longboard", "street", "downhill"], default="standard")
    p.add_argument("--model", choices=["reduced", "truck"], default="truck")
    p.add_argument("--experiment", choices=["manual", "coast", "lean", "release", "turn", "drop", "pull", "twist", "unload", "freeboard"])
    p.add_argument("--contact-model", choices=["native"], default="native")
    p.add_argument("--controller", choices=["none", "pose", "policy"], default="none")
    p.add_argument("--checkpoint")
    p.add_argument("--params", help="nominal or a JSON/YAML parameter file")
    p.add_argument("--randomize", action="store_true", help="sample bounded episode parameters from the declared ranges")
    p.add_argument("--policy-task", choices=["reward", "tracking", "goal"], default="reward")
    p.add_argument("--latent-name")
    p.add_argument("--preset", choices=["separate", "push_start", "steer_start"], default="separate")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.replay:
        from skate_sim.replay import replay
        replay(args.replay, mode=args.mode, backend=args.backend)
        return 0
    args.backend = args.backend or "mujoco"
    if args.scene not in ("bench", "board", "gallery", "contact", "robot", "rider"):
        raise SystemExit("active workbench implements bench, board, gallery, contact, robot and rider scenes")
    extra = {}
    if args.experiment:
        if args.backend != "mujoco":
            raise SystemExit("active experiments require MuJoCo")
        from skate_sim.physics.dynamics import experiment_config
        from dataclasses import asdict
        if args.scene == "contact":
            extra = dict(experiment=args.experiment, contact_model=args.contact_model)
        else:
            extra = dict(dynamics=asdict(experiment_config(args.model, args.experiment)), experiment=args.experiment)
    backend_kwargs = dict(seed=args.seed, scene=args.scene, style=args.style,
                          inspect=args.inspect, controller=args.controller,
                          checkpoint=args.checkpoint, policy_task=args.policy_task,
                          latent_name=args.latent_name, preset=args.preset,
                          params=args.params, randomize=args.randomize, **extra)
    if args.backend == "mjlab":
        backend_kwargs.update(num_envs=args.num_envs, view_env=args.view_env)
    backend = create_backend(args.backend, **backend_kwargs)
    backend.paused = args.inspect
    recorder = Recorder(args.record, backend=args.backend, scene=args.scene, seed=args.seed)
    recorder.write("mode", controller=args.controller, preset=args.preset, policy_task=args.policy_task)
    recorder.write("parameters", **backend.diagnostics().get("parameter_metadata", {}),
                   parameter_summary=backend.diagnostics().get("parameter_summary"))
    if args.experiment:
        recorder.write("experiment", model=args.model, experiment=args.experiment,
                       dynamics=extra.get("dynamics", {}), contact_model=args.contact_model,
                       initial_state=backend.get_state())
    if args.scene == "rider":
        recorder.write("mode", controller=args.controller, preset=args.preset,
                       policy_task=args.policy_task, latent_name=args.latent_name)
        recorder.write("rider_init", **backend.rider_initial_state())
    if args.scene in ("board", "gallery"):
        from skate_sim.models.board import board_hash
        print(backend.board_style.as_dict(), "hash=", board_hash(backend.board_style), flush=True)
        print("Inspection: Space dynamics/pause; 1/2/3/4 rotate individual wheels while paused; V transparent; J joint axes; R reset", flush=True)
        recorder.write("asset", config=backend.board_style.as_dict(), config_hash=board_hash(backend.board_style))
    print(f"skate-sim | backend={args.backend} | scene={args.scene} | controller={args.controller} task={args.policy_task} | Space pause, N physics step, M control step, R reset, F push, E release | contact: A/D pull, T/Y twist")
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
                if not backend.paused:
                    backend.step()
                if args.duration is not None and backend.get_state()['time'] >= args.duration:
                    break
                time.sleep(.01)
        while getattr(backend, "is_running", lambda: backend.viewer is None or backend.viewer.is_running())():
            if getattr(backend, "viewer", None) is None:
                backend.render()
            if backend.viewer is None:
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
