"""Read recorded MuJoCo states; invoked only from play.py."""
import json
import time
from pathlib import Path
from .backends.mujoco import MujocoBackend


def replay(path):
    with (Path(path)/'events.jsonl').open() as file:
        rows=[json.loads(line) for line in file]
    meta=next(r for r in rows if r['event']=='meta')
    asset=next(r for r in rows if r['event']=='asset')
    experiment=next(r for r in rows if r['event']=='experiment')
    if meta['backend'] != 'mujoco':
        raise ValueError('Stage 03 visual replay requires a MuJoCo recording')
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
