"""Visual state restores the recorded post-step wrench without integrating."""
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from skate_sim.backends.mujoco import MujocoBackend
from skate_sim.record import Recorder


class NativeVisualTests(unittest.TestCase):
    def test_force_frames_restore_actual_post_step_state(self):
        kwargs = dict(scene_config={'robot': False}, controller='none')
        b, c = MujocoBackend(**kwargs), MujocoBackend(**kwargs)
        try:
            with tempfile.TemporaryDirectory() as path:
                r = Recorder(path, backend='mujoco', scene=b.scene, seed=b.seed)
                b.recorder = r
                snapshots = []
                b.apply_wrench('deck', (4., 2., 1.), point=(.2, .1, .3), duration_steps=2)
                for _ in range(3):
                    b.step()
                    snapshots.append(b.data.xfrc_applied.copy())
                b.apply_wrench('deck', (3., 0., 0.))
                b.step()
                snapshots.append(b.data.xfrc_applied.copy())
                b.clear_wrenches()
                b.step()
                snapshots.append(b.data.xfrc_applied.copy())
                r.close()
                b.recorder = None
                rows = [json.loads(line) for line in (r.path/'events.jsonl').read_text().splitlines()]
                frames = [row['state'] for row in rows if row['event'] == 'tick']
                self.assertGreater(np.linalg.norm(snapshots[0][:, 3:]), 0.)
                self.assertGreater(np.linalg.norm(snapshots[1]), 0.)
                self.assertEqual(np.linalg.norm(snapshots[2]), 0.)
                self.assertGreater(np.linalg.norm(snapshots[3]), 0.)
                self.assertEqual(np.linalg.norm(snapshots[4]), 0.)
                with patch('skate_sim.backends.mujoco.mujoco.mj_step', side_effect=AssertionError('visual must not integrate')):
                    for frame, force in zip(frames, snapshots):
                        np.testing.assert_array_equal(frame['xfrc_applied'], force)
                        actual = c.set_visual_state(frame)
                        for field in ('qpos', 'qvel', 'time', 'xfrc_applied'):
                            np.testing.assert_array_equal(actual[field], frame[field])
                self.assertEqual(c.physics_step_counter, 0)
                c.set_visual_state(frames[0])
                legacy = {k: frames[1][k] for k in ('qpos', 'qvel', 'time')}
                c.set_visual_state(legacy)
                self.assertFalse(c.data.xfrc_applied.any())
        finally:
            b.recorder = None
            b.close()
            c.close()

    def test_invalid_force_does_not_partially_restore_core_state(self):
        b = MujocoBackend(scene_config={'robot': False}, controller='none')
        try:
            before = b.get_state()
            for force in (np.zeros((1, 6)), np.full((b.model.nbody, 6), np.nan)):
                frame = dict(before, qpos=before['qpos'] + 1., time=42., xfrc_applied=force)
                with self.assertRaisesRegex(ValueError, 'xfrc_applied'):
                    b.set_visual_state(frame)
                for field in before:
                    np.testing.assert_array_equal(b.get_state()[field], before[field])
                self.assertFalse(b.data.xfrc_applied.any())
        finally:
            b.close()
