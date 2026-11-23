"""Recorded-state display and independently integrated dynamics replay."""
from __future__ import annotations

import json
import hashlib
import math
import time
from pathlib import Path
from fractions import Fraction

import numpy as np

from .backends import create_backend
from .config import SimConfig
from .record import compiled_model_hash, numerics_manifest

# Registered before comparison: short deterministic same-backend replay only.
DEFAULT_ATOL = 1e-7
DEFAULT_HORIZON_SECONDS = 1.0


def read_record(path):
    source = Path(path)
    if source.is_dir():
        source /= 'events.jsonl'
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def _backend_kwargs(rows, meta):
    initial = next((r for r in rows if r['event'] == 'initial'), None)
    if initial:
        kwargs = dict(initial['backend_kwargs'])
        if isinstance(kwargs.get('config'), dict):
            kwargs['config'] = SimConfig(**kwargs['config'])
        return kwargs, initial
    # Historical logs remain usable for visual playback, preserving scene identity.
    mode = next((r for r in rows if r['event'] == 'mode'), {})
    asset = next((r for r in rows if r['event'] == 'asset'), {})
    parameters = next((r for r in rows if r['event'] == 'parameters'), {})
    experiment = next((r for r in rows if r['event'] == 'experiment'), {})
    kwargs = dict(scene=meta['scene'], seed=meta.get('seed', 0), inspect=True,
                  controller='none', preset=mode.get('preset', 'separate'))
    if asset:
        kwargs['style'] = asset['config']['name']
    if parameters.get('parameters'):
        kwargs['params'] = parameters['parameters']
    if experiment:
        kwargs.update(dynamics=experiment.get('dynamics'), experiment=experiment.get('experiment'))
    return kwargs, None


def _draw(b):
    if getattr(b, 'viewer', None) is None:
        b.render()
    viewer = getattr(b, 'viewer', None)
    if viewer is not None:
        if not viewer.is_running():
            return False
        viewer.sync()
    return True


def _state_errors(expected, actual, *, visual=False):
    """Compare fields separately, preserving units and empty-scene semantics."""
    errors, nonfinite = {}, []
    largest = None
    fields = ['qpos', 'qvel', 'time']
    if visual:
        fields.extend(key for key in ('xfrc_applied', 'qfrc_applied') if key in expected)
    for key in fields:
        wanted, observed = np.asarray(expected[key]), np.asarray(actual[key])
        if wanted.shape != observed.shape:
            raise ValueError(f'replay state shape mismatch: {key}')
        if not np.isfinite(wanted).all() or not np.isfinite(observed).all():
            nonfinite.append(key)
            continue
        delta = np.abs(wanted - observed)
        error = float(delta.max()) if delta.size else 0.0
        errors[key] = error
        if largest is None or error > largest['error']:
            index = list(np.unravel_index(int(delta.argmax()), delta.shape)) if delta.size else []
            largest = dict(field=key, index=[int(i) for i in index], error=error)
    return errors, nonfinite, largest


def replay(path, *, mode='visual', backend=None, headless=False,
           atol=DEFAULT_ATOL, horizon_seconds=DEFAULT_HORIZON_SECONDS):
    if mode not in ('visual', 'dynamics'):
        raise ValueError('replay mode must be visual or dynamics')
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError('atol must be finite and positive')
    if not np.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError('horizon_seconds must be finite and positive')
    rows = read_record(path)
    meta = next(r for r in rows if r['event'] == 'meta')
    selected = backend or meta['backend']
    if backend is not None and backend != meta['backend']:
        raise ValueError('replay requires the recorded backend')
    kwargs, initial = _backend_kwargs(rows, meta)
    ticks = [r for r in rows if r['event'] in ('tick', 'step')]
    sequence_verified = meta.get('format_version', 1) >= 2
    if sequence_verified and any(type(row.get('tick')) is not int or row['tick'] != index
                                 for index, row in enumerate(ticks)):
        raise ValueError('record tick sequence must be contiguous and start at zero')
    if mode == 'dynamics':
        if meta.get('format_version', 1) < 2 or initial is None or any(r['event'] != 'tick' for r in ticks):
            raise ValueError('dynamics replay requires v2 initial state and per-physics-tick inputs; legacy logs support visual replay only')
        if selected != meta['backend']:
            raise ValueError('dynamics comparison requires the recorded backend')
        if not ticks:
            raise ValueError('record contains no physics ticks')
        for item in meta.get('checkpoint', []):
            model_path = Path(item['path'])
            if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f"recorded checkpoint input missing or hash changed: {model_path}")
    else:
        # A visual replay must not initialize policy inference or accept live force input.
        kwargs.update(controller='joint' if kwargs.get('controller') == 'joint' else 'none',
                      checkpoint=None, latent_name=None, policy_task='reward')
    b = create_backend(selected, **kwargs)
    if selected == 'mujoco' and initial is not None and np.asarray(initial['state']['qpos']).shape != np.asarray(b.get_state()['qpos']).shape:
        b.close()
        raise ValueError("Recorded joint layout differs from current asset; obsolete synthetic-gripper records require their archived source/model, not Robotiq replay")
    comparison_atol = 0.0 if mode == 'visual' else atol
    report = dict(mode=mode, backend=selected, ticks=0, atol=comparison_atol,
                  horizon_seconds=horizon_seconds, first_deviation=None, max_abs_error=0.0,
                  verification='stored_state_readback' if mode == 'visual' else 'independent_integration',
                  tick_sequence='VERIFIED' if sequence_verified else 'UNVERIFIED_LEGACY',
                  max_abs_error_by_field=dict(qpos=0.0, qvel=0.0, time=0.0))
    try:
        recorded_numerics = meta.get('numerics')
        report['numerics_identity'] = 'LEGACY_UNVERSIONED'
        if mode == 'dynamics' and recorded_numerics is not None:
            if numerics_manifest(b) != recorded_numerics:
                raise ValueError('numerical implementation differs from recording')
            report['numerics_identity'] = 'VERIFIED'
        elif mode == 'visual':
            report['numerics_identity'] = 'NOT_REQUIRED_FOR_STORED_STATES'
        asset = next((row for row in rows if row['event'] == 'asset_model'), None)
        report['model_identity'] = 'UNVERIFIED_NO_MODEL_HASH'
        if asset and asset.get('files'):
            expected_hash = asset['files'][0]['sha256']
            if compiled_model_hash(b) != expected_hash:
                raise ValueError('compiled model hash differs from recording')
            report['model_identity'] = 'VERIFIED'
        if mode == 'dynamics':
            restore = getattr(b, 'import_replay_state', None)
            advance = getattr(b, 'step_recorded', None)
            if restore is None or advance is None or initial.get('runtime_state') is None:
                raise ValueError('backend recording lacks complete dynamics replay state/input support')
            b.set_state(initial['state'])
            restore(initial['runtime_state'])
        else:
            b.paused = True
            # Backend may use this to reject keys/mouse perturbations during display.
            b.visual_replay = True
        previous = None
        dt = float(b.config.physics_dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('backend physics_dt must be finite and positive')
        comparison_ticks = math.ceil(Fraction(str(horizon_seconds)) / Fraction(str(dt)))
        report['required_ticks_per_segment'] = comparison_ticks
        report['compared_ticks'] = 0
        report['visual_force_verified_frames'] = 0
        continuous_ticks = longest_continuous_ticks = reset_count = 0
        segments, segment_compared, segment_start = [], 0, 0
        playback_interrupted = False
        for row in ticks:
            state = row['state']
            if mode == 'visual':
                if not headless and previous is not None:
                    time.sleep(max(0, min(.1, float(np.asarray(state['time']).max()) - previous)))
                previous = float(np.asarray(state['time']).max())
                if hasattr(b, 'set_visual_state'):
                    actual = b.set_visual_state(state)
                else:
                    b.set_state(state)
                    actual = b.get_state()
                if 'xfrc_applied' in state:
                    report['visual_force_verified_frames'] += 1
                in_horizon = True
                report['compared_ticks'] += 1
            else:
                for key in ('action', 'command', 'inputs', 'action_source'):
                    if key not in row:
                        raise ValueError(f'tick {row.get("tick")} lacks required dynamics input {key}')
                if any(event.get('event') == 'reset' or
                       (event.get('event') == 'key' and event.get('key') == 82)
                       for event in row['inputs']):
                    if continuous_ticks:
                        segments.append(dict(start_tick=segment_start, ticks=continuous_ticks,
                                             compared_ticks=segment_compared))
                    segment_start, segment_compared = report['ticks'], 0
                    continuous_ticks = 0
                    reset_count += 1
                continuous_ticks += 1
                longest_continuous_ticks = max(longest_continuous_ticks, continuous_ticks)
                advance(row['action'], row['command'], row['inputs'])
                actual = b.get_state()
                in_horizon = continuous_ticks <= comparison_ticks
                if in_horizon:
                    report['compared_ticks'] += 1
                    segment_compared += 1
            errors, nonfinite_fields, largest = _state_errors(state, actual, visual=mode == 'visual')
            if nonfinite_fields:
                report['max_abs_error'] = None
                if report['first_deviation'] is None:
                    report['first_deviation'] = dict(tick=row.get('tick'), error=None,
                                                     nonfinite_fields=nonfinite_fields)
            elif in_horizon:
                for key, error in errors.items():
                    report['max_abs_error_by_field'][key] = max(report['max_abs_error_by_field'].get(key, 0.), error)
                error = largest['error']
                if report['max_abs_error'] is not None:
                    report['max_abs_error'] = max(report['max_abs_error'], error)
                if error > comparison_atol and report['first_deviation'] is None:
                    report['first_deviation'] = dict(tick=row.get('tick'), **largest)
            report['ticks'] += 1
            if not headless and not _draw(b):
                playback_interrupted = True
                break
        report.update(compared_seconds=report['compared_ticks'] * dt,
                      longest_continuous_seconds=longest_continuous_ticks * dt, reset_count=reset_count)
        if mode == 'dynamics' and continuous_ticks:
            segments.append(dict(start_tick=segment_start, ticks=continuous_ticks,
                                 compared_ticks=segment_compared))
        report['segments'] = segments
        report['complete_horizon'] = (mode == 'visual' or
                                      (bool(segments) and all(s['compared_ticks'] >= comparison_ticks for s in segments)))
        report['interrupted'] = playback_interrupted
        report['status'] = ('FAIL' if report['first_deviation'] is not None else
                            'PASS' if report['ticks'] == len(ticks) and report['ticks'] > 0
                            and report['complete_horizon'] and not playback_interrupted else 'INCOMPLETE')
        print(json.dumps(report, indent=2))
        return report
    finally:
        b.close()


def export_video(path, output, *, caption='Recorded native simulation', focus_body=None):
    """Offline stored-state video, launched only by play.py; never integrates."""
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    rows = read_record(path)
    meta = next(r for r in rows if r['event'] == 'meta')
    kwargs, initial = _backend_kwargs(rows, meta)
    kwargs.update(controller='none', checkpoint=None)
    b = create_backend('mujoco', **kwargs)
    if np.asarray(initial['state']['qpos']).shape != (b.model.nq,):
        b.close()
        raise ValueError("Recorded joint layout differs: use matching archived model/source for obsolete gripper records")
    states = [initial['state']] + [r['state'] for r in rows if r['event'] == 'tick']
    times = np.array([float(s['time']) for s in states])
    targets = np.arange(times[0], times[-1]+1e-9, 1/25)
    indices = np.searchsorted(times, targets, side='left').clip(0,len(states)-1).tolist()
    if indices[-1] != len(states)-1:
        indices.append(len(states)-1)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    b.model.vis.global_.offwidth=960;b.model.vis.global_.offheight=640
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
    cam.distance=3.0;cam.azimuth=135;cam.elevation=-18
    try:
        with mujoco.Renderer(b.model,640,960) as renderer,imageio.get_writer(output,fps=25,codec='libx264') as writer:
            for index in indices:
                b.set_visual_state(states[index])
                points=[b.data.xpos[b.model.body(n).id] for n in ['pelvis','deck'] if n in [b.model.body(i).name for i in range(b.model.nbody)]]
                cam.lookat[:]=np.mean(points,axis=0)
                cam.lookat[2] += .18
                if focus_body is not None:
                    bid=b.model.body(focus_body).id
                    cam.lookat[:]=b.data.xpos[bid]
                    cam.lookat[2]+=.025
                    cam.distance=.55
                renderer.update_scene(b.data,camera=cam)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,960,68),fill=(18,20,25))
                draw.text((12,8),caption,fill='white')
                draw.text((12,30),f't={times[index]:.3f}s | stored states | 1x | no replay physics | no board weld/board force',fill='white')
                writer.append_data(np.asarray(frame))
        return dict(output=str(output),frames=len(indices),fps=25,source_seconds=float(times[-1]-times[0]),physics_steps=0)
    finally:b.close()
