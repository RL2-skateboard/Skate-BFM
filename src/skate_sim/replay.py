"""Read recorded MuJoCo states; invoked only from play.py."""
import json
import time
from pathlib import Path
from .backends.mujoco import MujocoBackend


def replay(path, *, mode="visual", backend=None):
    with (Path(path)/'events.jsonl').open() as file:
        rows=[json.loads(line) for line in file]
    meta=next(r for r in rows if r['event']=='meta')
    asset=next((r for r in rows if r['event']=='asset'), None)
    experiment=next((r for r in rows if r['event']=='experiment'), None)
    mode_event=next((r for r in rows if r['event']=='mode'), {})
    rider_init=next((r for r in rows if r['event']=='rider_init'), None)
    recorded_backend = meta['backend']
    if mode not in ('visual', 'dynamics'):
        raise ValueError("replay mode must be visual or dynamics")
    selected_backend = backend or recorded_backend
    if mode == 'visual' and recorded_backend != 'mujoco':
        raise ValueError('visual replay currently requires a MuJoCo recording')
    if selected_backend != 'mujoco':
        raise ValueError('dynamics replay currently supports the native MuJoCo backend only')
    parameter_event = next((r for r in rows if r['event'] == 'parameters'), None)
    params = parameter_event.get('parameters') if parameter_event else None
    if mode_event.get('controller') in ('pose','policy') or meta.get('scene') == 'rider':
        b=MujocoBackend(scene='rider', controller=mode_event.get('controller','none'),
                        preset=mode_event.get('preset','separate'), inspect=True,
                        checkpoint='checkpoints/bfm-zero' if mode_event.get('controller')=='policy' else None,
                        policy_task=mode_event.get('policy_task','reward'), params=params)
        if rider_init:
            print('RIDER REPLAY INIT:', {k: rider_init.get(k) for k in ('preset','left_foot_deck_contact_count','right_foot_deck_contact_count','left_foot_ground_contact_count','right_foot_ground_contact_count','board_linear_velocity','penetration_min_m')})
    else:
        if meta.get('scene') == 'bench':
            b = MujocoBackend(scene='bench', inspect=True, params=params)
        elif asset:
            b = MujocoBackend(scene=meta.get('scene', 'board'), style=asset['config']['name'],
                              dynamics=experiment['dynamics'] if experiment else None,
                              experiment=experiment['experiment'] if experiment else None,
                              inspect=True, params=params)
        else:
            raise ValueError('record lacks asset metadata for non-rider replay')
    if mode == 'visual':
        print('VISUAL REPLAY: recorded states only; no live forces')
    else:
        print('DYNAMICS REPLAY: rerunning the recorded scene configuration with live MuJoCo physics')
    try:
        previous=None
        next_frame=time.perf_counter()
        if mode == 'dynamics':
            steps = sum(1 for row in rows if row['event'] == 'step')
            for _ in range(steps):
                b.step()
                if b.viewer is None:
                    b.render()
                if b.viewer is None or not b.viewer.is_running():
                    break
                b.viewer.sync()
            return
        for row in rows:
            if row['event']!='step': continue
            state=row['state']
            if previous is not None:
                time.sleep(max(0,min(.1,state['time']-previous)))
            previous=state['time']
            b.set_state(state)
            if b.viewer is None:
                b.render()
            if not b.viewer.is_running(): break
            if time.perf_counter() >= next_frame:
                b.viewer.sync()
                next_frame=time.perf_counter()+1/60
    finally:
        b.close()
        time.sleep(.2)
