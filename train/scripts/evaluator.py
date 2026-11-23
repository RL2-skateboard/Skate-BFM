#!/usr/bin/env python3
"""Evaluate native29 BFM policies and use skate-sim's versioned public replay.

Historical 23-DoF raw resets, teacher parity and paired trajectory scores are
retired. An offline latent is an independent condition, never an aligned reset.
Interactive playback belongs to skate-sim/play.py --replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import subprocess
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = REPOSITORY_ROOT.parent / "skate-sim"
sys.path[:0] = [str(REPOSITORY_ROOT / "src"), str(SIM_ROOT / "src"),
                str(REPOSITORY_ROOT / "train/scripts")]

import numpy as np
import torch

from skate_bfm.integration import AUX_REWARD_KEYS, SkateSimOnlineEnv
from skate_bfm.integration.actions import BFM0_JOINTS
from skate_sim.record import (
    Recorder, checkpoint_manifest, compiled_model_xml, numerics_manifest,
)
from skate_sim.replay import read_record, replay

SCHEMA = "skate-bfm-native29-evaluation-v1"
PROTOCOL = REPOSITORY_ROOT / "train/evaluation_protocol.json"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("schema") != SCHEMA or protocol.get("source_joint_support") != 29:
        raise ValueError("Retired/unsupported protocol: native29 cannot restore or score legacy raw23 dynamics.")
    if protocol.get("control_dt_s") != .02 or protocol.get("physics_dt_s") != .002:
        raise ValueError("Native29 protocol requires control_dt=.02 and physics_dt=.002.")
    return protocol


def load_context(path):
    if path is None:
        return None
    value = np.load(path, allow_pickle=False)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2 or value.shape[1] != 256 or not len(value) or not np.isfinite(value).all():
        raise ValueError("Independent context must be a finite [256] or [T,256] NumPy array.")
    return value.astype(np.float32)


def recording_kwargs(env, seed):
    backend = env.env.backend
    kwargs = dict(seed=seed, scene=backend.scene, controller="pose",
                  scene_config=backend.scene_config, config=asdict(backend.config))
    if env.device.type == "cuda":
        kwargs["num_envs"] = env.num_envs
    return kwargs


def attach_recording(env, output, checkpoint_files, seed, *, mode="external_bfm29"):
    backend = env.env.backend
    recorder = Recorder(str(output), backend=backend.name, scene=backend.scene, seed=seed,
                        metadata={"schema": "native29_control_v1", "source_joint_support": 29,
                                  "physics_dt": backend.config.physics_dt,
                                  "control_dt": backend.config.control_dt,
                                  "num_envs": env.num_envs, "checkpoint": checkpoint_files,
                                  "numerics": numerics_manifest(backend),
                                  "reset_alignment": False})
    recorder.start(backend.get_state(), backend_kwargs=recording_kwargs(env, seed),
                   runtime_state=backend.export_replay_state())
    model_path = output / "model.xml"
    model_path.write_text(compiled_model_xml(backend))
    recorder.write("asset_model", files=checkpoint_manifest(str(model_path)))
    recorder.write("mode", controller=mode, preset=env.env.preset,
                   initial_state_aligned=False, source_physics_aligned=False,
                   scene_config=backend.scene_config, joint_order=BFM0_JOINTS)
    backend.recorder = recorder
    return recorder


def validate_checkpoint(path):
    """Reject retired controllers/layouts before allocating an agent or simulator."""
    from train_runner import checkpoint_model_path

    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in (".onnx", ".npz", ".pkl"):
        raise ValueError("Native29 evaluation requires a complete BFM checkpoint; legacy ONNX/raw23 resets are unsupported.")
    model = checkpoint_model_path(path)
    metadata = json.loads((model.parent / "init_kwargs.json").read_text())
    if metadata.get("action_dim") != 29:
        raise ValueError("Native29 evaluation rejects checkpoints with a non-29 action dimension.")
    return path


def run_eval(args):
    from train_skate_bfm import load_frozen_agent

    protocol = load_protocol(args.protocol)
    if not np.isfinite(args.duration_s) or args.duration_s <= 0:
        raise ValueError("duration-s must be finite and positive.")
    if args.num_envs < 1 or (args.backend == "mujoco" and args.num_envs != 1):
        raise ValueError("MuJoCo evaluation supports exactly one world; mjlab supports positive batches.")
    context = load_context(args.context_npy)
    checkpoint = validate_checkpoint(args.checkpoint)
    if args.output.expanduser().resolve().exists():
        raise FileExistsError("Evaluation output must be a fresh directory.")
    if args.checkpoint_sha256:
        from train_runner import checkpoint_model_path, hash_file
        if hash_file(checkpoint_model_path(checkpoint)) != args.checkpoint_sha256:
            raise ValueError("Checkpoint SHA256 differs from the explicitly requested identity.")
    scene = None
    if args.scene_config:
        from skate_sim.models.scene import normalize_scene_config
        scene = normalize_scene_config(json.loads(args.scene_config.read_text()))
        if not scene["robot"] or not scene["board"]:
            raise ValueError("Native29 evaluation requires both robot and board.")
    device = args.device or ("cpu" if args.backend == "mujoco" else "cuda")
    agent, checkpoint_report = load_frozen_agent(
        checkpoint, online_env_backend="skate-sim-cpu" if args.backend == "mujoco" else "skate-sim-mjlab",
        device=device,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = SkateSimOnlineEnv(backend=args.backend, num_envs=args.num_envs,
                           preset=args.preset, seed=args.seed, scene_config=scene)
    recorder = None
    samples = []
    resets = terminations = truncations = 0
    torch.manual_seed(args.seed)
    try:
        observation = env.reset()
        files = checkpoint_manifest(checkpoint_report["model_file"])
        recorder = attach_recording(env, output, files, args.seed)
        if context is not None:
            context_path = output / "context.npy"
            np.save(context_path, context)
            recorder.write("context_source", files=checkpoint_manifest(str(context_path)),
                           alignment="independent_offline_condition", initial_state_aligned=False)
        z = agent._model.sample_z(args.num_envs, device=agent.device)
        horizon = math.ceil(args.duration_s / env.env.backend.config.control_dt)
        for index in range(horizon):
            if context is not None:
                z = torch.as_tensor(context[min(index, len(context) - 1)], device=agent.device).expand(args.num_envs, -1)
            with torch.no_grad():
                action = agent.act(obs={key: value.to(agent.device) for key, value in observation.items()}, z=z, mean=args.mean_action)
                transition = env.step(action, z)
            if not all(bool(torch.isfinite(value).all()) for value in transition["next"]["observation"].values()):
                raise RuntimeError("Native29 rollout produced non-finite observations.")
            backend = env.env.backend
            state = backend.get_state()
            q, dq, pos, quat, vel, omega = [numpy(value) for value in env.env._fields()]
            sample = dict(qpos=numpy(state["qpos"]), qvel=numpy(state["qvel"]), time=numpy(state["time"]),
                          action=numpy(transition["action"]), actor_action=numpy(action), z=numpy(z),
                          joint_position=q, joint_velocity=dq,
                          root_position=pos[:, env.env.root], root_quaternion=quat[:, env.env.root],
                          root_linear_velocity=vel[:, env.env.root], root_angular_velocity=omega[:, env.env.root],
                          board_position=pos[:, env.env.deck], board_quaternion=quat[:, env.env.deck],
                          board_linear_velocity=vel[:, env.env.deck], board_angular_velocity=omega[:, env.env.deck],
                          terminated=numpy(transition["next"]["terminated"]),
                          truncated=numpy(transition["next"]["truncated"]))
            sample.update({name: numpy(transition["aux_rewards"][name]) for name in AUX_REWARD_KEYS})
            if not all(np.isfinite(value).all() for value in sample.values()):
                raise RuntimeError("Native29 record contains non-finite physical/control state.")
            samples.append(sample)
            recorder.write("control", control_index=index, **sample)
            done = (transition["next"]["terminated"] | transition["next"]["truncated"]).flatten()
            ids = done.nonzero().flatten().cpu().tolist()
            terminations += int(transition["next"]["terminated"].sum())
            truncations += int(transition["next"]["truncated"].sum())
            observation = transition["next"]["observation"]
            if ids and index + 1 < horizon:
                observation = env.reset(env_ids=ids)
                if context is None:
                    z[ids] = agent._model.sample_z(len(ids), device=agent.device)
                resets += len(ids)
        np.savez_compressed(output / "raw_rollout.npz", **{
            key: np.stack([sample[key] for sample in samples]) for key in samples[0]
        })
        result = dict(schema=SCHEMA, source_joint_support=29, backend=args.backend,
                      num_envs=args.num_envs, seed=args.seed, control_steps=horizon,
                      transitions=horizon * args.num_envs, duration_s=horizon * .02,
                      physics_ticks=recorder.tick_index, model=env.metadata,
                      scene_config=env.env.backend.scene_config, protocol=protocol,
                      checkpoint=checkpoint_report, context_mode="independent_offline_condition" if context is not None else "sampled_bfm_prior",
                      initial_state_aligned=False, source_physics_aligned=False,
                      terminated_transitions=terminations, truncated_transitions=truncations,
                      reset_worlds=resets, all_recorded_values_finite=True,
                      dynamics_skill_or_generalization_claim=False,
                      replay_command=f"{SIM_ROOT}/play.py --replay {output} --mode visual")
        write_json(output / "metadata.json", result)
        print(json.dumps(result, indent=2))
        return result
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def native_record(path, *, require_control_context=False):
    rows = read_record(path)
    meta = next((row for row in rows if row.get("event") == "meta"), {})
    if meta.get("format_version") != 2 or meta.get("source_joint_support") != 29:
        raise ValueError("This evaluator accepts native29 Recorder v2 only; legacy raw23 replay is retired.")
    initial = next((row for row in rows if row.get("event") == "initial"), None)
    asset = next((row for row in rows if row.get("event") == "asset_model"), None)
    mode = next((row for row in rows if row.get("event") == "mode"), None)
    if not initial or not asset or not mode or not initial.get("backend_kwargs", {}).get("scene_config"):
        raise ValueError("Native29 recording requires initial runtime/scene, compiled model identity and mode.")
    if require_control_context and mode.get("controller") == "external_bfm29":
        controls = [(offset, row) for offset, row in enumerate(rows) if row.get("event") == "control"]
        updates = [(offset, row) for offset, row in enumerate(rows)
                   if row.get("event") == "tick" and row.get("command", {}).get("external_action_update")]
        if not controls or len(controls) != len(updates):
            raise ValueError("BFM dynamics replay requires one context/control row per external action update.")
        previous_control = -1
        for index, ((offset, control), (tick_offset, tick)) in enumerate(zip(controls, updates, strict=True)):
            if type(control.get("control_index")) is not int or control["control_index"] != index:
                raise ValueError("BFM control indices must be contiguous and start at zero.")
            if not previous_control < tick_offset < offset:
                raise ValueError("BFM control/context rows are out of order relative to physics action updates.")
            previous_control = offset
            action, z = np.asarray(control.get("action")), np.asarray(control.get("z"))
            if action.shape != (meta["num_envs"], 29) or z.shape != (meta["num_envs"], 256):
                raise ValueError("Native29 BFM control/context record shape mismatch.")
            if not np.isfinite(action).all() or not np.isfinite(z).all():
                raise ValueError("Native29 BFM control/context record is non-finite.")
            raw = np.asarray(tick.get("action"))
            if raw.shape == (29,) and meta["num_envs"] == 1:
                raw = raw[None]
            if raw.shape != action.shape or not np.isfinite(raw).all() or not np.array_equal(np.clip(raw, -1, 1), action):
                raise ValueError("BFM control action differs from the recorded physics action update.")
            if "actor_action" in control and not np.array_equal(np.asarray(control["actor_action"]), raw):
                raise ValueError("BFM raw actor action differs from the physics action update.")
    return rows, meta


def run_replay(args):
    native_record(args.record, require_control_context=args.mode == "dynamics")
    if not args.headless:
        # The sole interactive viewer entry point remains the simulator root CLI.
        if args.report:
            raise ValueError("Use --headless for a machine-readable replay report; interactive replay delegates to skate-sim/play.py.")
        completed = subprocess.run([sys.executable, str(SIM_ROOT / "play.py"),
                                    "--replay", str(args.record), "--mode", args.mode], check=False)
        return {"status": "PASS" if completed.returncode == 0 else "FAIL",
                "delegated_to": str(SIM_ROOT / "play.py")}
    result = replay(str(args.record), mode=args.mode, headless=args.headless,
                    atol=1e-7, horizon_seconds=1.0)
    if args.report:
        write_json(args.report, result)
    return result


def run_video(args):
    """Render recorded states with their own model; never integrate dynamics."""
    import imageio.v2 as imageio
    import mujoco

    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("Video dimensions and fps must be positive.")
    rows, meta = native_record(args.record)
    root = args.record if args.record.is_dir() else args.record.parent
    model_path = root / "model.xml"
    asset = next(row for row in rows if row.get("event") == "asset_model")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != asset["files"][0]["sha256"]:
        raise ValueError("Recorded model.xml identity changed.")
    if args.world < 0 or args.world >= meta["num_envs"]:
        raise ValueError("Requested video world is outside the recorded batch.")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    frame_digest = hashlib.sha256()
    try:
        with imageio.get_writer(str(args.output), fps=args.fps, codec="libx264",
                                macro_block_size=1, ffmpeg_params=["-threads", "1"]) as writer:
            for row in rows:
                if row.get("event") != "tick":
                    continue
                # Tick index remains monotonic across episode resets.
                if row["tick"] * meta["physics_dt"] + 1e-12 < frames / args.fps:
                    continue
                state = row["state"]
                for key, destination in (("qpos", data.qpos), ("qvel", data.qvel)):
                    value = np.asarray(state[key])
                    if value.ndim == 2:
                        value = value[args.world]
                    if value.shape != destination.shape or not np.isfinite(value).all():
                        raise ValueError(f"Recorded {key} does not match its own native model.")
                    destination[:] = value
                data.time = float(np.asarray(state["time"]).reshape(-1)[args.world if np.asarray(state["time"]).size > 1 else 0])
                mujoco.mj_forward(model, data)
                camera = int(args.camera) if str(args.camera).lstrip("-").isdigit() else args.camera
                renderer.update_scene(data, camera=camera)
                image = renderer.render()
                frame_digest.update(image.tobytes())
                writer.append_data(image)
                frames += 1
    finally:
        renderer.close()
    result = dict(schema=SCHEMA, mode="stored_state_video_no_integration", frames=frames,
                  fps=args.fps, world=args.world, model_sha256=asset["files"][0]["sha256"],
                  rendered_frames_sha256=frame_digest.hexdigest(),
                  output=str(args.output), video_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    write_json(args.output.with_suffix(args.output.suffix + ".json"), result)
    print(json.dumps(result, indent=2))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("eval", help="Native29 frozen BFM rollout and Recorder v2 capture.")
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--checkpoint-sha256")
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--protocol", default=PROTOCOL, type=Path)
    evaluate.add_argument("--backend", choices=("mujoco", "mjlab"), default="mujoco")
    evaluate.add_argument("--num-envs", type=int, default=1)
    evaluate.add_argument("--duration-s", type=float, default=1.0)
    evaluate.add_argument("--preset", choices=("steer_start", "push_start"), default="steer_start")
    evaluate.add_argument("--scene-config", type=Path)
    evaluate.add_argument("--seed", type=int, default=4728)
    evaluate.add_argument("--device", choices=("cpu", "cuda"))
    evaluate.add_argument("--context-npy", type=Path, help="Independent offline z condition; never an aligned raw-state reset.")
    evaluate.add_argument("--mean-action", action="store_true")
    playback = commands.add_parser("replay", help="Use skate-sim public replay with its strict dynamics gate.")
    playback.add_argument("--record", required=True, type=Path)
    playback.add_argument("--mode", choices=("visual", "dynamics"), default="visual")
    playback.add_argument("--headless", action="store_true")
    playback.add_argument("--report", type=Path)
    video = commands.add_parser("video", help="Offscreen video of stored native states; no integration.")
    video.add_argument("--record", required=True, type=Path)
    video.add_argument("--output", required=True, type=Path)
    video.add_argument("--world", type=int, default=0)
    video.add_argument("--fps", type=int, default=50)
    video.add_argument("--width", type=int, default=640)
    video.add_argument("--height", type=int, default=480)
    video.add_argument("--camera", default="-1", help="Recorded model camera name or -1 for free camera.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    result = {"eval": run_eval, "replay": run_replay, "video": run_video}[args.command](args)
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
