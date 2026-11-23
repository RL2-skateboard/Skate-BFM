import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skate_sim import cli


class DeliveryTests(unittest.TestCase):
    def test_all_uses_two_independent_backend_children(self):
        def child(command, backend, **kwargs):
            self.assertIn(backend, ('mujoco', 'mjlab'))
            return dict(exit_code=0, result=dict(status='PASS', results=[]))
        with patch.object(cli, '_backend_process', side_effect=child) as dispatch:
            result = cli._run_check('05', 'all', 1, 'checkpoint', 'noncontact')
        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual(result['scope_status'], 'PASS')
        self.assertEqual(result['overall_project_status'], 'INCOMPLETE')

    def test_report_generation_does_not_turn_failed_acceptance_into_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'result.json'
            source.write_text(json.dumps(dict(status='FAIL', stage='05', backend='mujoco')))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = cli.main(['report', '--run', str(source), '--output', str(Path(directory) / 'report.md')])
            self.assertNotEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result['generation_status'], 'PASS')
            self.assertEqual(result['acceptance_status'], 'FAIL')

    def test_frozen_contact_is_not_pass(self):
        result = cli._run_check('04', 'mujoco', 1, None)
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertEqual(result['results'][0]['status'], 'DEFERRED')

    def test_gpu_duration_uses_all_environments(self):
        import numpy as np
        import play
        self.assertFalse(play._duration_reached(np.array([1., .5]), 1.))
        self.assertTrue(play._duration_reached(np.array([1., 1.]), 1.))

    def test_no_record_attaches_no_recorder_and_copies_no_initial_state(self):
        import play
        import numpy as np
        class Backend:
            config = type('Config', (), {'physics_dt': .002})()
            paused = False
            def __init__(self): self.steps = 0
            def render_web(self, *args): pass
            def take_requested_style(self): return None
            def step(self): self.steps += 1
            def get_state(self): return {'time': np.array([self.steps*.002, self.steps*.002])}
            def close(self): pass
        b = Backend()
        with patch('sys.argv', ['play.py','--scene','bench','--backend','mjlab','--duration','.004']), \
             patch.object(play, 'create_backend', return_value=b), \
             patch.object(play, 'checkpoint_manifest', side_effect=AssertionError('unrequested hash')), \
             patch.object(play.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(play.main(), 0)
        self.assertEqual(b.steps, 2)
        self.assertFalse(hasattr(b, 'recorder'))

    def test_full_gpu07_batch_pass_does_not_certify_stage_or_project(self):
        batch = dict(status='PASS', stage='07', backend='mjlab', checks={'batch': True})
        with patch.object(cli, '_mjlab_check', return_value=batch):
            result = cli._run_check('07', 'mjlab', 64, 'checkpoint', 'full')
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertEqual(result['stage_acceptance_status'], 'INCOMPLETE')
        self.assertEqual(result['overall_project_status'], 'INCOMPLETE')
        self.assertTrue(any(r.get('submatrix') == 'strict_dynamics_replay' and r['status'] == 'NOT_RUN'
                            for r in result['results']))
        with patch.object(cli, '_mjlab_check', return_value=batch):
            scoped = cli._run_check('07', 'mjlab', 64, 'checkpoint', 'noncontact')
        self.assertEqual(scoped['scope_status'], 'PASS')
        self.assertEqual(scoped['stage_acceptance_status'], 'INCOMPLETE')
        self.assertEqual(scoped['overall_project_status'], 'INCOMPLETE')

    def test_automatic_stage05_pass_is_not_user_stage_acceptance(self):
        with patch.object(cli, '_mujoco_check', return_value=dict(status='PASS', stage='05', backend='mujoco')):
            result = cli._run_check('05', 'mujoco', 1, 'checkpoint', 'noncontact')
        self.assertEqual(result['scope_status'], 'PASS')
        self.assertEqual(result['automatic_checks_status'], 'PASS')
        self.assertEqual(result['stage_acceptance_status'], 'INCOMPLETE')
        self.assertEqual(result['overall_project_status'], 'INCOMPLETE')

    def test_default_dispatches_unified_robot_board(self):
        import play
        with patch('sys.argv', ['play.py']), patch('skate_sim.viewer.run_scene', return_value=0) as run:
            self.assertEqual(play.main(), 0)
        config = run.call_args.args[1]['scene_config']
        self.assertTrue(config['robot'])
        self.assertTrue(config['board'])
        self.assertNotIn('model', run.call_args.args[1])
