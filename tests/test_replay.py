"""Recording must drive physics inputs, never repair errors with snapshots."""
import json

import numpy as np
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skate_sim.record import Recorder
from skate_sim.replay import replay


class FakeBackend:
    class Config:
        physics_dt = .01
    config = Config()
    def __init__(self):
        self.x = 0.
        self.restores = 0
    def set_state(self, state):
        self.restores += 1
        self.x = float(state['qpos'][0])
    def import_replay_state(self, state):
        self.speed = state['speed']
    def step_recorded(self, action, command, inputs):
        self.x += action
    def get_state(self):
        return dict(qpos=[self.x], qvel=[0.], time=self.x * .01)
    def close(self):
        pass


class ReplayTests(unittest.TestCase):
    def test_dynamics_never_overwrites_tick_states(self):
        with tempfile.TemporaryDirectory() as directory:
            b = FakeBackend()
            r = Recorder(directory, backend='mujoco', scene='robot', seed=4)
            r.start(dict(qpos=[0.], qvel=[0.], time=0.), backend_kwargs={'scene': 'robot'},
                    runtime_state={'speed': 1.})
            r.tick(dict(qpos=[1.], qvel=[0.], time=.01), action=1.)
            r.tick(dict(qpos=[99.], qvel=[0.], time=.02), action=1.)
            r.close()
            with patch('skate_sim.replay.create_backend', return_value=b):
                result = replay(directory, mode='dynamics', headless=True)
            self.assertEqual(b.restores, 1)
            self.assertEqual(b.x, 2.)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['first_deviation']['tick'], 1)
            with self.assertRaises(FileExistsError):
                Recorder(directory, backend='mujoco', scene='robot', seed=4)

    def test_legacy_rejects_dynamics_before_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'events.jsonl').write_text(json.dumps(dict(event='meta', backend='mujoco', scene='robot')) + '\n')
            with self.assertRaisesRegex(ValueError, 'legacy logs'):
                replay(directory, mode='dynamics', headless=True)

    def _record_ticks(self, directory, count, last_qvel=0., last_qpos=None):
        r = Recorder(directory, backend='mujoco', scene='robot', seed=4)
        r.start(dict(qpos=[0.], qvel=[0.], time=0.), backend_kwargs={'scene': 'robot'},
                runtime_state={'speed': 1.})
        for index in range(count):
            x = index + 1.
            r.tick(dict(qpos=[last_qpos if index == count-1 and last_qpos is not None else x],
                        qvel=[last_qvel if index == count-1 else 0.], time=x*.01), action=1.)
        r.close()

    def test_exact_horizon_boundary_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 500, last_qpos=501.)
            b = FakeBackend()
            b.config = type('Config', (), {'physics_dt': .002})()
            with patch('skate_sim.replay.create_backend', return_value=b):
                result = replay(directory, mode='dynamics', headless=True)
            self.assertEqual(result['compared_ticks'], 500)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['first_deviation']['tick'], 499)

    def test_nan_qvel_cannot_be_hidden_by_other_field_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 2, last_qvel=float('nan'))
            with patch('skate_sim.replay.create_backend', return_value=FakeBackend()):
                result = replay(directory, mode='dynamics', headless=True)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['first_deviation']['nonfinite_fields'], ['qvel'])
            self.assertIsNone(result['max_abs_error'])

    def test_invalid_thresholds_rejected(self):
        for parameter in ('atol', 'horizon_seconds'):
            for value in (float('nan'), float('inf'), 0., -1.):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(ValueError, 'finite and positive'):
                        replay('/does/not/exist', **{parameter: value})

    def test_visual_never_loads_recorded_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            r = Recorder(directory, backend='mujoco', scene='robot', seed=4)
            r.start(dict(qpos=[0.], qvel=[0.], time=0.),
                    backend_kwargs={'scene': 'robot', 'controller': 'policy',
                                    'checkpoint': '/missing/model.onnx', 'latent_name': 'missing'})
            r.tick(dict(qpos=[1.], qvel=[0.], time=.01))
            r.close()
            with patch('skate_sim.replay.create_backend', return_value=FakeBackend()) as factory:
                result = replay(directory, mode='visual', headless=True)
            kwargs = factory.call_args.kwargs
            self.assertEqual(kwargs['scene'], 'robot')
            self.assertEqual(kwargs['controller'], 'none')
            self.assertIsNone(kwargs['checkpoint'])
            self.assertIsNone(kwargs['latent_name'])
            self.assertEqual(result['status'], 'PASS')
            with self.assertRaisesRegex(ValueError, 'recorded backend'):
                replay(directory, mode='visual', backend='mjlab', headless=True)

    def test_short_dynamics_is_not_full_horizon_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 2)
            with patch('skate_sim.replay.create_backend', return_value=FakeBackend()):
                result = replay(directory, mode='dynamics', headless=True)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertFalse(result['complete_horizon'])

    def test_reset_segments_cannot_be_joined_into_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 100)
            path = Path(directory)/'events.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            next(r for r in rows if r.get('tick') == 50)['inputs'] = [dict(event='reset')]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with patch('skate_sim.replay.create_backend', return_value=FakeBackend()):
                result = replay(directory, mode='dynamics', headless=True)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertEqual([s['ticks'] for s in result['segments']], [50,50])

    def test_horizon_is_not_rounded_down(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 1)
            with patch('skate_sim.replay.create_backend', return_value=FakeBackend()):
                result = replay(directory, mode='dynamics', headless=True, horizon_seconds=.015)
            self.assertEqual(result['required_ticks_per_segment'], 2)
            self.assertEqual(result['status'], 'INCOMPLETE')

    def test_missing_tick_rejected_before_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 3)
            path = Path(directory)/'events.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r.get('tick') != 1))
            with patch('skate_sim.replay.create_backend') as create:
                with self.assertRaisesRegex(ValueError, 'tick sequence'):
                    replay(directory, mode='visual', headless=True)
                create.assert_not_called()

    def test_visual_checks_actual_state_after_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 1)
            b = FakeBackend()
            b.set_state = lambda state: None
            with patch('skate_sim.replay.create_backend', return_value=b):
                result = replay(directory, mode='visual', headless=True)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['first_deviation']['field'], 'qpos')
            self.assertEqual(result['first_deviation']['index'], [0])

    def test_empty_visual_record_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 0)
            with patch('skate_sim.replay.create_backend', return_value=FakeBackend()):
                result = replay(directory, mode='visual', headless=True)
            self.assertEqual(result['status'], 'INCOMPLETE')

    def test_visual_also_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 1)
            path = Path(directory)/'events.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps(dict(event='asset_model', files=[dict(sha256='original')]))+'\n')
            with patch('skate_sim.replay.create_backend', return_value=FakeBackend()), \
                 patch('skate_sim.replay.compiled_model_hash', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'model hash differs'):
                    replay(directory, mode='visual', headless=True)

    def test_ground_only_scene_record_and_replay(self):
        from skate_sim.backends.mujoco import MujocoBackend
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(scene_config=dict(robot=False, board=False))
            b = MujocoBackend(**kwargs)
            try:
                r = Recorder(directory, backend='mujoco', scene='bench', seed=0)
                r.start(b.get_state(), backend_kwargs=kwargs, runtime_state=b.export_replay_state())
                b.recorder = r
                for _ in range(2):
                    b.step()
                r.close()
            finally:
                b.close()
            result = replay(directory, mode='dynamics', headless=True, horizon_seconds=.004)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['max_abs_error_by_field']['qpos'], 0.)
            self.assertEqual(result['max_abs_error_by_field']['qvel'], 0.)

    def test_changed_numerical_implementation_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            self._record_ticks(directory, 1)
            path = Path(directory)/'events.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['numerics'] = dict(strategy='recorded-order')
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            b = FakeBackend()
            b.numerics_manifest = lambda: dict(strategy='different-order')
            with patch('skate_sim.replay.create_backend', return_value=b):
                with self.assertRaisesRegex(ValueError, 'numerical implementation differs'):
                    replay(directory, mode='dynamics', headless=True)
            self.assertEqual(b.restores, 0)
