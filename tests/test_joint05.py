"""Actual MuJoCo joint inspection and shared keyboard semantics."""
import numpy as np
import unittest
from skate_sim.backends.mujoco import MujocoBackend


def backend():
    return MujocoBackend(scene="robot", controller="joint", inspect=True)


class NativeJointTests(unittest.TestCase):
    def test_all_29_joint_directions_and_limits(self):
        b = backend()
        base = b.data.qpos[:7].copy()
        responses = []
        for i in range(29):
            ends = []
            for sign in (-1, 1):
                b.reset()
                b.selected_joint = i
                b._handle_key(61 if sign > 0 else 45)
                assert np.isclose(b.q_target[i], b.policy_values['default'][i] + sign*.05)
                for _ in range(250):
                    b.step()
                ends.append(b._robot_q()[i])
                assert np.max(np.abs(b.data.qpos[:7]-base)) < 2e-4
                assert np.all(np.abs(b.last_tau) <= b.policy_values['joint_effort_limit'] + 1e-9)
            responses.append(ends[1]-ends[0])
            assert responses[-1] > .015, (b.robot_joint_names[i], ends)
            b.adjust_joint_target(i, 1e6)
            assert b.q_target[i] == b.policy_values['joint_pos_upper_limit'][i]
            b.adjust_joint_target(i, -1e6)
            assert b.q_target[i] == b.policy_values['joint_pos_lower_limit'][i]
        assert len(responses) == 29
        b.close()


    def test_keyboard_reset_and_free_mode(self):
        b = backend()
        b._handle_key(91)
        assert b.selected_joint == 28
        b._handle_key(93)
        assert b.selected_joint == 0
        b._handle_key(61)
        b._handle_key(b.KEY_N)
        assert b.physics_step_counter == 1 and b.policy_counter == 0
        b._handle_key(b.KEY_M)
        assert b.physics_step_counter == b.config.substeps+1
        b._handle_key(b.KEY_R)
        assert b.physics_step_counter == 0
        np.testing.assert_equal(b.q_target,b.policy_values['default'])
        assert b.diagnostics()['inspection_fixture_active']
        b._handle_key(80)
        assert b.controller == 'pose'
        assert not b.diagnostics()['inspection_fixture_active']
        b._handle_key(b.KEY_R)
        assert not b.diagnostics()['inspection_fixture_active']
        b._handle_key(79)  # missing checkpoint preserves current mode
        assert b.controller == 'pose'
        b._handle_key(b.KEY_F)
        assert b._force_body_id == b.model.body('pelvis').id
        b.close()


    def test_policy_boundary_single_step_without_inference(self):
        b = backend()
        calls = []
        b.policy = lambda obs: calls.append(obs.copy()) or np.zeros(29)
        b.policy_z = np.zeros(256,dtype=np.float32)
        b._save_policy_debug = lambda obs: None
        assert b.switch_controller('policy')
        b._handle_key(b.KEY_N)
        assert not calls and b.policy_counter == 0
        # Paused idle/readback has no policy side effects.
        for _ in range(4): b.diagnostics()
        assert not calls
        b._handle_key(b.KEY_M)
        assert len(calls) == 1
        assert b.switch_controller('pose')
        assert b.policy_counter == 0 and not np.any(b.robot_history.last_action)
        assert not b._action_queue
        assert b.switch_controller('policy')
        assert not np.any(b.last_obs)
        assert not b.diagnostics()['inspection_fixture_active']
        b.close()


    def test_rider_noncontact_guards_and_reset(self):
        import mujoco
        b = MujocoBackend(scene='rider',controller='pose',preset='separate',episode_time_limit=.1)
        def info():
            return dict(pelvis_position=b.data.xpos[b.model.body('pelvis').id].copy(),
                        penetration_min_m=0.,
                        pelvis_deck_horizontal_distance=float(np.linalg.norm(
                            b.data.xpos[b.model.body('pelvis').id,:2]-b.data.xpos[b.model.body('deck').id,:2])))
        assert not b._rider_termination(info())['terminated']
        b.data.time = .1
        result = b._rider_termination(info())
        assert result['truncated'] and not result['terminated']
        b.reset()
        assert not b._rider_termination(info())['truncated']
        b.data.qpos[2] = .2
        mujoco.mj_forward(b.model,b.data)
        assert b._rider_termination(info())['termination_reason'] == 'fall'
        b.reset()
        b.data.qpos[0] += 6.
        mujoco.mj_forward(b.model,b.data)
        assert b._rider_termination(info())['termination_reason'] == 'out_of_bounds'
        b.reset()
        b.data.qpos[0] += 1.5
        mujoco.mj_forward(b.model,b.data)
        assert not b._rider_termination(info())['terminated']  # separate has not approached board
        b._rider_seen_near_board = True
        assert b._rider_termination(info())['termination_reason'] == 'offboard'
        b.reset()
        assert not b._rider_seen_near_board
        b.episode_time_limit = b.out_of_bounds_distance = b.offboard_distance = None
        b.data.time = 100.
        b.data.qpos[0] += 6.
        mujoco.mj_forward(b.model,b.data)
        assert b._rider_termination(info()) == dict(terminated=False,truncated=False,
                                                   termination_reason=None,truncation_reason=None)
        b.close()


    def test_native_wrench_duration_frames_and_state_selection(self):
        import mujoco
        b = MujocoBackend(scene='bench')
        b.data.qpos[3:7] = (np.cos(np.pi/4),0,0,np.sin(np.pi/4))
        mujoco.mj_forward(b.model,b.data)
        state = b.get_state(env_ids=[0])
        b.set_state(state,env_ids=[0])
        b.apply_wrench('cube',(2,0,0),point=(0,1,0),frame='body',env_ids=[0],duration_steps=2)
        value = b._wrenches[b._body_id]['value'].copy()
        np.testing.assert_allclose(value[:3],(0,2,0),atol=1e-12)
        assert np.isclose(value[5], -2.)
        b.step()
        np.testing.assert_allclose(b.data.xfrc_applied[b._body_id],value)
        b.step()
        np.testing.assert_allclose(b.data.xfrc_applied[b._body_id],value)
        b.step()
        assert not np.any(b.data.xfrc_applied)
        b.apply_wrench('cube',(1,0,0))
        b.clear_wrenches(env_ids=[0])
        assert not b._wrenches
        with self.assertRaises(ValueError): b.get_state(env_ids=[1])
        with self.assertRaises(ValueError): b.apply_wrench('cube',(1,0,0),duration_steps=0)
        b.reset(env_ids=[0])
        b.close()
