"""Every reset entry point must emit replayable inputs exactly once."""
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from skate_sim.backends.mujoco import MujocoBackend
from skate_sim.record import Recorder


class NativeResetTests(unittest.TestCase):
    def test_direct_and_key_resets_round_trip(self):
        for use_key in (False, True):
            with self.subTest(use_key=use_key), tempfile.TemporaryDirectory() as path:
                kwargs = dict(scene_config={'robot': False}, controller='none')
                b = MujocoBackend(**kwargs)
                r = Recorder(path, backend='mujoco', scene=b.scene, seed=b.seed)
                initial, runtime = b.get_state(), b.export_replay_state()
                b.recorder = r
                b.step()
                if use_key:
                    b._handle_key(b.KEY_R)
                else:
                    b.reset(seed=7, env_ids=[0])
                b.step()
                r.close()
                b.recorder = None
                rows = [json.loads(line) for line in (r.path/'events.jsonl').read_text().splitlines()]
                ticks = [row for row in rows if row['event'] == 'tick']
                self.assertEqual([e['event'] for e in ticks[1]['inputs']], ['reset'])
                c = MujocoBackend(**kwargs)
                try:
                    c.set_state(initial)
                    c.import_replay_state(runtime)
                    for row in ticks:
                        c.step_recorded(row['action'], row['command'], row['inputs'])
                        for field in ('qpos', 'qvel', 'time'):
                            np.testing.assert_array_equal(c.get_state()[field], row['state'][field])
                    for inputs in ([{'event': 'key', 'key': 82}],
                                   [{'event': 'key', 'key': 82}, {'event': 'reset', 'seed': None, 'env_ids': None}]):
                        with patch.object(c, 'reset', wraps=c.reset) as reset:
                            c.step_recorded(ticks[1]['action'], ticks[1]['command'], inputs)
                            self.assertEqual(reset.call_count, 1)
                finally:
                    c.close()
                    b.close()
