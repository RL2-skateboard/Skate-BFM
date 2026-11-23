#!/usr/bin/env python3
"""Collect native 29-DoF policy rollouts; these are not certified expert data.

Physical replay uses skate-sim Recorder v2. NPZ rows are pre-action control
states, paired with the action actually applied over the next ten physics ticks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train/scripts'))
sys.path.insert(0, str(ROOT / 'src'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parallel-config', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--backend', choices=('mujoco', 'mjlab'))
    p.add_argument('--scene-config', type=Path)
    p.add_argument('--context-npy', type=Path)
    p.add_argument('--max-policy-frames', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--dataset-split', choices=('train', 'validation', 'test'))
    p.add_argument('--round-count', type=int)
    p.add_argument('--rollouts-per-round', type=int)
    return p


def collect(args):
    import torch
    import mujoco
    torch.set_num_threads(1)
    from evaluator import attach_recording, validate_checkpoint
    from train_runner import checkpoint_model_path
    from skate_bfm.integration.sim import SkateSimOnlineEnv
    from train_skate_bfm import load_frozen_agent
    from skate_sim.env import BFM_BODIES
    from skate_sim.robot.contract import G1_29_JOINTS
    from skate_sim.models.scene import normalize_scene_config
    from skate_sim.record import checkpoint_manifest

    cfg = dict(checkpoint=str(ROOT / 'model/bfm-zero-official'), backend='mujoco',
               max_policy_frames=500, seed=0, dataset_split='train', round_count=1,
               rollouts_per_round=1, round_start=1, rollout_start=1,
               output_dir=str(ROOT / 'train/dataset/native29/train/raw'), scene_config=None, context_npy=None)
    if args.parallel_config:
        supplied = json.loads(args.parallel_config.read_text())
        unknown = supplied.keys() - cfg.keys()
        if unknown:
            raise ValueError(f'unsupported collector config fields: {sorted(unknown)}; legacy XML/23DoF policies are retired')
        cfg.update(supplied)
    for key in cfg:
        value = getattr(args, key, None)
        if value is not None: cfg[key] = value
    for key in ('max_policy_frames', 'round_count', 'rollouts_per_round'):
        if type(cfg[key]) is not int or cfg[key] <= 0: raise ValueError(f'{key} must be a positive integer')
    if cfg['backend'] not in ('mujoco', 'mjlab'):
        raise ValueError('backend must be native29 mujoco or mjlab')
    if cfg['dataset_split'] not in ('train', 'validation', 'test'):
        raise ValueError('dataset_split must be train, validation or test')
    checkpoint = Path(cfg['checkpoint']).resolve()
    if checkpoint.suffix.lower() == '.onnx':
        raise ValueError('collector requires a validated native29 BFM checkpoint directory; legacy 23DoF ONNX is unsupported')
    checkpoint = validate_checkpoint(checkpoint)
    checkpoint_files = checkpoint_manifest(str(checkpoint_model_path(checkpoint)))
    scene = cfg['scene_config']
    if scene is not None:
        scene = json.loads(Path(scene).read_text()) if isinstance(scene, (str, Path)) else scene
    scene = normalize_scene_config(scene or {'preset': 'steer_start'})
    device = 'cpu' if cfg['backend'] == 'mujoco' else 'cuda'
    agent, report = load_frozen_agent(checkpoint, online_env_backend='skate-sim-cpu' if cfg['backend'] == 'mujoco' else 'skate-sim-mjlab', device=device)
    output = Path(cfg['output_dir']).resolve()
    for round_offset in range(cfg['round_count']):
        for offset in range(cfg['rollouts_per_round']):
            round_id = cfg['round_start'] + round_offset
            rollout_id = cfg['rollout_start'] + offset
            seed = int(cfg['seed']) + round_offset * cfg['rollouts_per_round'] + offset
            path = output / f'round_{round_id:03d}' / f'rollout_{rollout_id:03d}'
            if path.exists(): raise FileExistsError(f'refusing to overwrite {path}')
            raw_dir = path / 'raw_rollout'; raw_dir.mkdir(parents=True)
            env = SkateSimOnlineEnv(backend=cfg['backend'], num_envs=1, seed=seed, scene_config=scene)
            recorder = None
            try:
                b = env.env.backend; model = env.env.model
                obs = env.reset(seed=seed)
                torch.manual_seed(seed)
                z = torch.randn((1, 256), device=env.device)
                z = 16 * z / z.norm(dim=-1, keepdim=True)
                if cfg['context_npy']:
                    value = np.load(cfg['context_npy'], allow_pickle=False)
                    if value.shape != (256,) or not np.isfinite(value).all():
                        raise ValueError('context-npy must be finite [256]')
                    z = torch.as_tensor(value, device=env.device, dtype=torch.float32)[None]
                replay_path = path / 'replay'
                recorder = attach_recording(env, replay_path, checkpoint_files, seed)
                model_path = replay_path / 'model.xml'
                robot_ids = [model.joint(name + '_joint').id for name in G1_29_JOINTS]
                board_ids = [i for i in range(model.njnt) if i not in robot_ids and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
                board_q = [int(model.jnt_qposadr[i]) for i in board_ids]
                board_dq = [int(model.jnt_dofadr[i]) for i in board_ids]
                rows = []
                terminal = 'horizon'
                reset_pending = False
                termination_count = 0
                previous_board_yaw = None
                board_heading_delta = 0.
                for frame in range(cfg['max_policy_frames']):
                    was_reset = reset_pending
                    if reset_pending:
                        obs = env.reset()
                        previous_board_yaw = None
                        board_heading_delta = 0.
                        reset_pending = False
                    with torch.no_grad(): action = agent.act(obs, z, mean=True)
                    if action.shape != (1, 29) or not bool(torch.isfinite(action).all()):
                        raise ValueError('controller must produce finite native29 [1,29] actions')
                    q,dq,pos,quat,vel,omega = [v.detach().cpu().numpy()[0] for v in env.env._fields()]
                    physical = b.get_state()
                    qpos, qvel = np.asarray(physical['qpos']), np.asarray(physical['qvel'])
                    if qpos.ndim == 2: qpos, qvel = qpos[0], qvel[0]
                    root, deck, bodies = env.env.root, env.env.deck, env.env.body_ids
                    w,x,y,zq = quat[deck]
                    board_yaw = float(np.arctan2(2*(w*zq+x*y), 1-2*(y*y+zq*zq)))
                    if previous_board_yaw is not None:
                        board_heading_delta += (board_yaw-previous_board_yaw+np.pi) % (2*np.pi)-np.pi
                    previous_board_yaw = board_yaw
                    row = dict(qpos=qpos.copy(), qvel=qvel.copy(), action=action[0].detach().cpu().numpy(),
                        root_pos=pos[root], root_quat=quat[root], dof_pos=q, dof_vel=dq,
                        body_pos=pos[bodies], body_quat=quat[bodies], body_lin_vel=vel[bodies], body_ang_vel=omega[bodies],
                        board_root_pos=pos[deck], board_root_quat=quat[deck], board_root_lin_vel=vel[deck], board_root_ang_vel=omega[deck],
                        board_dof_pos=qpos[board_q], board_dof_vel=qvel[board_dq], board_heading_delta=board_heading_delta,
                        frame_idx=frame, sim_time=float(np.asarray(physical['time']).reshape(-1)[0]), phase_id=7, phase_value=0.,
                        fall=False, reset=was_reset, context=z[0].cpu().numpy())
                    for key, value in obs.items(): row['observation_' + key] = value[0].detach().cpu().numpy()
                    transition = env.step(action, z)
                    row['actor_action'] = row['action'].copy()
                    row['action'] = transition['action'][0].detach().cpu().numpy()
                    for key,value in transition['next']['observation'].items():
                        row['next_observation_' + key] = value[0].detach().cpu().numpy()
                    for key in ('terminated', 'truncated'):
                        row[key] = bool(transition['next'][key].item())
                    for key, value in transition['aux_rewards'].items():
                        row['aux_' + key] = value[0].detach().cpu().numpy()
                    rows.append(row)
                    recorder.write('control', control_index=frame, action=row['action'][None], actor_action=row['actor_action'][None], z=row['context'][None],
                                   terminated=transition['next']['terminated'].cpu().numpy(),
                                   truncated=transition['next']['truncated'].cpu().numpy())
                    obs = transition['next']['observation']
                    if bool(transition['next']['terminated'].item()) or bool(transition['next']['truncated'].item()):
                        terminal = 'terminated' if bool(transition['next']['terminated'].item()) else 'truncated'
                        termination_count += 1
                        reset_pending = True
                arrays = {key:np.stack([row[key] for row in rows]) for key in rows[0]}
                raw_path = raw_dir / 'raw_rollout.npz'; np.savez_compressed(raw_path, **arrays)
                metadata = dict(schema='native29_control_v1', source_joint_support=29, missing_joint_names=[], native29_expert=False,
                    quality='policy_rollout_unvalidated', phase_annotation='unavailable', command_annotation='unavailable',
                    state_action_alignment='pre_action', context_mode='provided_npy' if cfg['context_npy'] else 'seeded_uniform_sphere_256',
                    context_file=str(cfg['context_npy']) if cfg['context_npy'] else None, termination_count=termination_count, episode_id=f'{round_id:03d}_{rollout_id:03d}', round_id=round_id, rollout_id=rollout_id,
                    dataset_split=cfg['dataset_split'], num_frames=len(rows), fps=50, dt=.02, control_hz=50, physics_hz=500,
                    joint_order=[n+'_joint' for n in G1_29_JOINTS], body_order=list(BFM_BODIES), board_joint_order=[model.joint(i).name for i in board_ids],
                    qpos_quaternion_order='wxyz', body_quaternion_order='wxyz', board_quaternion_order='wxyz',
                    physics_randomization=dict(seed=seed, randomized=False), policy_checkpoint=str(checkpoint), policy_checkpoint_files=checkpoint_files, checkpoint_load=report,
                    robot_xml=str(model_path), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), scene_config=scene,
                    state_units='SI: metre, radian, second', velocity_frame='world',
                    action_contract=dict(order='joint_order', semantics='normalized_bfm29', clip=[-1.,1.],
                                         default=env.env.values['default'].cpu().tolist(), action_rescale=env.env.values['action_rescale'].cpu().tolist(),
                                         action_scale=env.env.values['action_scale'].cpu().tolist()),
                    terminal_reason=terminal if reset_pending else 'horizon', fixed_bfm_joints=[], seed=seed, replay_path=str(replay_path), physics_ticks=recorder.tick_index)
                raw_path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
                print(json.dumps(dict(raw=str(raw_path), frames=len(rows), physics_ticks=recorder.tick_index, terminal_reason=metadata['terminal_reason'])))
            finally:
                if recorder: recorder.close()
                env.close()
    return 0


def main():
    args = parser().parse_args()
    try: return collect(args)
    except (ValueError, FileExistsError) as exc: raise SystemExit(str(exc)) from exc

if __name__ == '__main__': raise SystemExit(main())
