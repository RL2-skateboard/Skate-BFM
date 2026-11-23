from __future__ import annotations

import argparse
import time
import numpy as np
from pathlib import Path
from skate_sim.backends import create_backend
from skate_sim.record import Recorder, checkpoint_manifest, recording_inputs, compiled_model_xml, numerics_manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="skate-sim interactive workbench")
    p.add_argument("--backend", choices=["mujoco", "mjlab"], default=None)
    p.add_argument("--scene", help="scene JSON preset; default: robot + project skateboard (paused)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record")
    p.add_argument("--contact-run", help="offline train_parameters.py run: show target/baseline/trained curves")
    p.add_argument("--contact-trial", help="complete trial ID (default: first untouched test trial)")
    p.add_argument("--duration", type=float, help="Close after this many simulation seconds")
    p.add_argument("--replay")
    p.add_argument("--export-video", help="export stored replay to MP4 without physics integration")
    p.add_argument("--caption", default="Recorded native simulation")
    p.add_argument("--focus-body", help="display-only close view for replay video export")
    p.add_argument("--mode", choices=['visual', 'dynamics'], default='visual')
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--view-env", type=int, default=0)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--style", choices=["standard", "cruiser", "longboard", "street", "downhill"], default="standard")
    p.add_argument("--experiment", choices=["manual", "coast", "lean", "release", "turn", "drop", "pull", "twist", "unload", "freeboard"])
    p.add_argument("--contact-model", choices=["native"], default="native")
    p.add_argument("--controller", choices=["none", "joint", "pose", "policy"], default="none")
    p.add_argument("--checkpoint")
    p.add_argument("--gripper", choices=["none", "left", "right", "both"], default=None, help="opt-in official Robotiq2F-85; native composed scene only")
    p.add_argument("--params", help="nominal or a JSON/YAML parameter file")
    p.add_argument("--randomize", action="store_true", help="sample bounded episode parameters from the declared ranges")
    p.add_argument("--policy-task", choices=["reward", "tracking", "goal"], default="reward")
    p.add_argument("--latent-name")
    p.add_argument("--preset", choices=["separate", "push_start", "steer_start"], default="separate")
    return p


def _recording_inputs(args):
    return recording_inputs(args.checkpoint, args.policy_task)


def _duration_reached(sim_time, duration):
    """A batched duration completes only when every environment reaches it."""
    values = np.asarray(sim_time)
    return bool(np.all(values >= duration))


def main() -> int:
    args = parser().parse_args()
    if args.contact_run:
        if args.replay or args.params or args.randomize or args.duration is not None:
            raise SystemExit("--contact-run is an offline comparison; do not combine it with replay/params/randomize/duration")
        from skate_sim.physics.contact import contact_figure
        import matplotlib.pyplot as plt
        contact_figure(args.contact_run, args.contact_trial)
        plt.show()
        return 0
    if args.contact_trial:
        raise SystemExit("--contact-trial requires --contact-run")
    if args.export_video and not args.replay:
        raise SystemExit("--export-video requires --replay")
    if args.replay:
        if args.export_video:
            from skate_sim.replay import export_video
            print(export_video(args.replay, args.export_video, caption=args.caption, focus_body=args.focus_body))
            return 0
        from skate_sim.replay import replay
        result = replay(args.replay, mode=args.mode, backend=args.backend)
        return 0 if result["status"] == "PASS" else 1
    args.backend = args.backend or "mujoco"
    legacy_scenes = ("bench", "board", "gallery", "contact", "robot", "rider")
    if args.scene not in legacy_scenes:
        from skate_sim.models.scene import load_scene_config, normalize_scene_config
        from skate_sim.viewer import run_scene
        if args.experiment:
            raise SystemExit("--experiment is a legacy validation protocol; use an explicit legacy scene")
        scene_config = load_scene_config(args.scene) if args.scene else normalize_scene_config(
            dict(style=args.style, preset=args.preset))
        if args.gripper is not None:
            scene_config = normalize_scene_config(dict(scene_config, gripper=args.gripper))
        kwargs = dict(seed=args.seed, scene_config=scene_config, inspect=True,
                      controller=args.controller, checkpoint=args.checkpoint,
                      policy_task=args.policy_task, latent_name=args.latent_name,
                      params=args.params, randomize=args.randomize)
        if args.backend == "mjlab":
            kwargs.update(num_envs=args.num_envs, view_env=args.view_env)
        return run_scene(args.backend, kwargs, record_path=args.record, duration=args.duration)
    if args.scene not in ("bench", "board", "gallery", "contact", "robot", "rider"):
        raise SystemExit("active workbench implements bench, board, gallery, contact, robot and rider scenes")
    if args.gripper not in (None, "none"):
        raise SystemExit("--gripper requires the composed scene (omit legacy --scene name)")
    extra = {}
    if args.experiment:
        if args.backend != "mujoco":
            raise SystemExit("active experiments require MuJoCo")
        from skate_sim.physics.dynamics import experiment_config
        from dataclasses import asdict
        if args.scene == "contact":
            extra = dict(experiment=args.experiment, contact_model=args.contact_model)
        else:
            extra = dict(dynamics=asdict(experiment_config("truck", args.experiment)), experiment=args.experiment)
    backend_kwargs = dict(seed=args.seed, scene=args.scene, style=args.style,
                          inspect=args.inspect, controller=args.controller,
                          checkpoint=args.checkpoint, policy_task=args.policy_task,
                          latent_name=args.latent_name, preset=args.preset,
                          params=args.params, randomize=args.randomize, **extra)
    if args.backend == "mjlab":
        backend_kwargs.update(num_envs=args.num_envs, view_env=args.view_env)
    backend = create_backend(args.backend, **backend_kwargs)
    backend.paused = args.inspect
    recorder = Recorder(None, backend=args.backend, scene=args.scene, seed=args.seed)
    if args.record:
        checkpoint_files = _recording_inputs(args)
        recorder = Recorder(args.record, backend=args.backend, scene=args.scene, seed=args.seed,
                            metadata={"physics_dt": backend.config.physics_dt,
                                      "control_dt": backend.config.control_dt,
                                      "num_envs": args.num_envs, "view_env": args.view_env,
                                      "checkpoint": checkpoint_files,
                                      "numerics": numerics_manifest(backend),
                                      "policy_task": args.policy_task, "latent_name": args.latent_name,
                                      "guard_thresholds": backend.diagnostics().get("guard_thresholds")})
        from dataclasses import asdict
        recorded_kwargs = dict(backend_kwargs, config=asdict(backend.config))
        recorder.start(backend.get_state(), backend_kwargs=recorded_kwargs,
                       runtime_state=backend.export_replay_state() if hasattr(backend, "export_replay_state") else None)
        backend.recorder = recorder
        model_path = Path(args.record).resolve() / "model.xml"
        model_path.write_text(compiled_model_xml(backend), encoding="utf-8")
        recorder.write("asset_model", files=checkpoint_manifest(str(model_path)),
                       source_policy="compiled XML retains original mesh paths; meshes are not copied into the log")
        recorder.write("checkpoint", files=checkpoint_files)
        recorder.write("mode", controller=args.controller, preset=args.preset, policy_task=args.policy_task)
        recorder.write("parameters", **backend.diagnostics().get("parameter_metadata", {}),
                       parameter_summary=backend.diagnostics().get("parameter_summary"))
    if args.experiment:
        recorder.write("experiment", model=extra.get("dynamics", {}).get("model"), experiment=args.experiment,
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
                if not backend.paused:
                    backend.step()
                if args.duration is not None and _duration_reached(backend.get_state()['time'], args.duration):
                    break
                time.sleep(backend.config.physics_dt)
            return 0
        while getattr(backend, "is_running", lambda: backend.viewer is None or backend.viewer.is_running())():
            if getattr(backend, "viewer", None) is None:
                backend.render()
            if backend.viewer is None:
                break
            # The passive viewer gives us a real window; keyboard callbacks are
            # intentionally kept in the small loop so the backend contract stays pure.
            if not backend.paused:
                backend.step()
                if args.duration is not None and _duration_reached(backend.get_state()['time'], args.duration):
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
