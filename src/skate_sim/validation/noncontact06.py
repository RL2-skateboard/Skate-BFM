"""Native automatic noncontact slice; no foot/deck contact certification."""
from __future__ import annotations
import tempfile
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import mujoco
from skate_sim.backends.mujoco import MujocoBackend
from skate_sim.record import Recorder, numerics_manifest
from skate_sim.replay import replay


def run(checkpoint):
    checks = {}
    for preset in ('separate', 'push_start', 'steer_start'):
        b = MujocoBackend(scene='rider', controller='pose', preset=preset)
        try:
            initial = b.get_state()
            for _ in range(10):
                b.step(); b.reset()
                for field in ('qpos', 'qvel', 'time'):
                    np.testing.assert_array_equal(b.get_state()[field], initial[field])
            checks[preset + '_ten_resets'] = True
            state = b.get_state()
            b.step(); b.set_state(state)
            checks[preset + '_state_roundtrip'] = all(np.array_equal(b.get_state()[k], v) for k,v in state.items())
        finally:
            b.close()
    b = MujocoBackend(scene='bench')
    try:
        b.apply_wrench('cube', (2.,0.,0.), point=(0.,1.,0.), frame='body', duration_steps=2)
        b.step()
        checks['point_wrench_torque'] = bool(np.linalg.norm(b.data.xfrc_applied[b._body_id,3:]) > 0)
        b.step(); b.step()
        checks['wrench_expiry'] = not b._wrenches and not np.any(b.data.xfrc_applied)
        b.reset()
        checks['wrench_reset'] = not b._wrenches and not np.any(b.data.xfrc_applied)
    finally:
        b.close()
    kwargs = dict(scene='rider', controller='policy', checkpoint=checkpoint, preset='separate', seed=606)
    record_root = Path('runs/noncontact_completion/native06_records')
    record_root.mkdir(parents=True, exist_ok=True)
    with nullcontext(tempfile.mkdtemp(prefix='policy_', dir=record_root)) as directory:
        b = MujocoBackend(**kwargs)
        r = Recorder(directory, backend='mujoco', scene='rider', seed=606,
                     metadata={'numerics': numerics_manifest(b)})
        try:
            r.start(b.get_state(), backend_kwargs=kwargs, runtime_state=b.export_replay_state())
            b.recorder = r
            # Each side of reset covers a complete registered 1-second horizon.
            for index in range(1000):
                if index == 20: b._handle_key(b.KEY_F)
                if index == 500: b._handle_key(b.KEY_R)
                b.step()
            checks['actual_policy_nonzero'] = bool(np.linalg.norm(b.raw_action) > 0)
        finally:
            r.close(); b.close()
        replay_result = replay(directory, mode='dynamics', headless=True)
        checks['policy_force_reset_replay'] = replay_result['status'] == 'PASS'
    b = MujocoBackend(scene='rider', controller='none', preset='separate', episode_time_limit=.1)
    def info():
        pelvis = b.data.xpos[b.model.body('pelvis').id]
        deck = b.data.xpos[b.model.body('deck').id]
        return dict(pelvis_position=pelvis.copy(), penetration_min_m=0.,
                    pelvis_deck_horizontal_distance=float(np.linalg.norm(pelvis[:2]-deck[:2])))
    try:
        b.data.time = .1
        done = b._rider_termination(info())
        checks['time_limit_separate'] = done['truncated'] and not done['terminated']
        for name, axis, value in [('fall',2,.2), ('out_of_bounds',0,6.)]:
            b.reset(); b.data.qpos[axis] = value; mujoco.mj_forward(b.model,b.data)
            checks[name] = b._rider_termination(info())['termination_reason'] == name
        b.reset(); b.data.qpos[0] += 1.5; b._rider_seen_near_board = True
        mujoco.mj_forward(b.model,b.data)
        checks['offboard'] = b._rider_termination(info())['termination_reason'] == 'offboard'
        b.reset()
        checks['guard_reset'] = not b._rider_seen_near_board and not b._rider_termination(info())['truncated']
    finally:
        b.close()
    return {'checks': checks, 'replay': replay_result, 'record_path': directory,
            'scope': 'noncontact automatic contracts; manual UI and foot/deck contact are not certified'}
