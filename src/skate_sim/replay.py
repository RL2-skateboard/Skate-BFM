"""Read recorded MuJoCo states; invoked only from play.py."""
import json
import time
from pathlib import Path
from .backends.mujoco import MujocoBackend


def replay(path):
    with (Path(path)/'events.jsonl').open() as file:
        rows=[json.loads(line) for line in file]
    meta=next(r for r in rows if r['event']=='meta')
    asset=next((r for r in rows if r['event']=='asset'), None)
    experiment=next((r for r in rows if r['event']=='experiment'), None)
    mode=next((r for r in rows if r['event']=='mode'), {})
    rider_init=next((r for r in rows if r['event']=='rider_init'), None)
    if meta['backend'] != 'mujoco':
        raise ValueError('Stage 03 visual replay requires a MuJoCo recording')
    if mode.get('controller') in ('pose','policy') or meta.get('scene') == 'rider':
        b=MujocoBackend(scene='rider', controller=mode.get('controller','none'),
                        preset=mode.get('preset','separate'), inspect=True,
                        checkpoint='checkpoints/bfm-zero' if mode.get('controller')=='policy' else None,
                        policy_task=mode.get('policy_task','reward'))
        if rider_init:
            print('RIDER REPLAY INIT:', {k: rider_init.get(k) for k in ('preset','left_foot_deck_contact_count','right_foot_deck_contact_count','left_foot_ground_contact_count','right_foot_ground_contact_count','board_linear_velocity','penetration_min_m')})
    else:
        if not asset or not experiment:
            raise ValueError('record lacks asset/experiment metadata')
        b=MujocoBackend(scene='board', style=asset['config']['name'],
                        dynamics=experiment['dynamics'],experiment=experiment['experiment'],inspect=True)
    print('VISUAL REPLAY: recorded states only; no live forces')
    try:
        previous=None
        next_frame=time.perf_counter()
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
