import unittest
import numpy as np
import torch

from skate_sim.backends.mujoco import MujocoBackend
from skate_sim.env import SkateEnv, BFM_BODIES, privileged_state


class TrainingTests(unittest.TestCase):
    def test_external_action_and_control_clock(self):
        b=MujocoBackend(scene="robot",controller="pose")
        self.addCleanup(b.close)
        q=b.get_state()["qpos"].copy()
        b.control_step(np.full(29,.1))
        self.assertAlmostEqual(b.data.time,.02)
        self.assertEqual(b.policy_counter,1)
        self.assertGreater(np.linalg.norm(b.q_target-b.policy_values["default"]),.1)
        self.assertGreater(np.linalg.norm(b.last_tau),.1)
        self.assertGreater(np.linalg.norm(b.get_state()["qpos"]-q),1e-5)
        before=b.get_state()
        for invalid in (np.zeros(28),np.full(29,np.nan),np.zeros((1,29))):
            with self.assertRaises(ValueError): b.step(invalid)
            np.testing.assert_array_equal(b.get_state()["qpos"],before["qpos"])

    def test_delay_advances_per_control(self):
        b=MujocoBackend(scene="robot",controller="pose")
        self.addCleanup(b.close)
        b.parameters["control_delay_steps"]=1
        b.control_step(np.full(29,.1))
        np.testing.assert_array_equal(b.q_target,b.policy_values["default"])
        b.control_step(np.full(29,-.1))
        np.testing.assert_array_equal(b.policy_action,np.full(29,.1))
        self.assertEqual(b.policy_counter,2)

    def test_external_replay_applies_once(self):
        b=MujocoBackend(scene="robot",controller="pose")
        self.addCleanup(b.close)
        state=b.get_state(); runtime=b.export_replay_state(); rows=[]
        class Recorder:
            def tick(self,state,**kw): rows.append((state,kw))
        b.recorder=Recorder()
        b.control_step(np.full(29,.1)); b.control_step(np.full(29,-.2))
        del b.recorder
        b.set_state(state); b.import_replay_state(runtime)
        for expected,kw in rows:
            b.step_recorded(kw["action"],kw["command"],kw["inputs"])
            np.testing.assert_array_equal(b.get_state()["qpos"],expected["qpos"])

    def test_training_observations_and_terminal_storage(self):
        e=SkateEnv(); self.addCleanup(e.close)
        obs=e.reset()
        expected={"state":64,"privileged_state":463,"last_action":29,"history_actor":372}
        self.assertEqual({k:tuple(v.shape) for k,v in obs.items()},{k:(1,n) for k,n in expected.items()})
        e.backend.episode_time_limit=.02
        transition=e.step(torch.full((1,29),.1))
        self.assertFalse(transition["next"]["terminated"].item())
        self.assertTrue(transition["next"]["truncated"].item())
        torch.testing.assert_close(transition["next"]["observation"]["last_action"],torch.full((1,29),.5))
        torch.testing.assert_close(transition["aux_rewards"]["penalty_torques"],torch.tensor([[float(np.square(e.backend.last_tau).sum())]]))
        terminal={k:v.clone() for k,v in transition["next"]["observation"].items()}
        e.reset()
        for k in terminal: torch.testing.assert_close(terminal[k],transition["next"]["observation"][k])
        self.assertTrue(all(torch.isfinite(v).all() and (v>=0).all() for v in transition["aux_rewards"].values()))
        self.assertEqual(len(transition["aux_rewards"]),8)

    def test_invalid_reset_is_atomic_and_scene_is_shared(self):
        e=SkateEnv(); self.addCleanup(e.close)
        self.assertIsNotNone(e.backend.scene_config)
        self.assertEqual(e.model.body(e.model.body_rootid[e.deck]).name,"chassis")
        e.step(torch.zeros(1,29))
        before=e.backend.get_state(); history={k:v.clone() for k,v in e.history.items()}
        for ids in ([0.0],[True],[],[0,0],[-1],[1],[[0]]):
            with self.assertRaises(ValueError): e.reset(env_ids=ids)
            np.testing.assert_array_equal(before["qpos"],e.backend.get_state()["qpos"])
            self.assertEqual(before["time"],e.backend.data.time)
            for k in history: torch.testing.assert_close(history[k],e.history[k])
        with self.assertRaisesRegex(ValueError,"both robot and board"):
            SkateEnv(scene_config={"robot":False})

    def test_body_origin_velocity_with_rotation(self):
        import mujoco
        e=SkateEnv(); self.addCleanup(e.close)
        e.backend.data.qvel[:6]=[.2,-.1,.3,.4,-.6,1.1]
        mujoco.mj_forward(e.model,e.backend.data)
        origin=np.empty(6); com=np.empty(6)
        mujoco.mj_objectVelocity(e.model,e.backend.data,mujoco.mjtObj.mjOBJ_XBODY,e.root,origin,0)
        mujoco.mj_objectVelocity(e.model,e.backend.data,mujoco.mjtObj.mjOBJ_BODY,e.root,com,0)
        self.assertGreater(np.linalg.norm(origin[3:]-com[3:]),1e-4)
        np.testing.assert_allclose(e._fields()[4][0,e.root].numpy(),origin[3:],atol=1e-7)

    def test_real_platform_contact_cost(self):
        import mujoco
        config={"preset":"steer_start","board_pose":{"pos":[3.,0.,.1125],"quat":[1.,0.,0.,0.]},"obstacles":[{"id":"platform","kind":"platform","pos":[0.,0.,-.07],"quat":[1.,0.,0.,0.]}]}
        e=SkateEnv(scene_config=config); self.addCleanup(e.close)
        m=e.model; d=e.backend.data
        d.qvel[:3]=[.5,.1,0.]; mujoco.mj_forward(m,d)
        rewards=e._rewards(e._fields(),e.last_action)
        expected=0.
        for foot in e.foot_ids:
            hits=[]
            for i in range(d.ncon):
                c=d.contact[i]; bodies=[int(m.geom_bodyid[g]) for g in c.geom]
                if any(body in e.foot_bodies[foot].tolist() for body in bodies):
                    other=bodies[1] if bodies[0] in e.foot_bodies[foot].tolist() else bodies[0]
                    if m.body(other).name == "obstacle_platform":
                        force=np.empty(6);mujoco.mj_contactForce(m,d,i,force)
                        hits.append((np.linalg.norm(force[:3]),i))
            self.assertTrue(hits)
            strength,index=max(hits,key=lambda x:x[0]); self.assertGreater(strength,1.)
            c=d.contact[index]; v=np.empty(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,foot,v,0)
            velocity=v[3:]+np.cross(v[:3],c.pos-d.xpos[foot]);normal=c.frame.reshape(3,3)[0]
            expected+=np.linalg.norm(velocity-normal*np.dot(normal,velocity))
        self.assertGreater(expected,.5)
        self.assertAlmostEqual(expected,float(rewards["penalty_slippage"]),places=5)

    def test_real_bodies_and_virtual_head_velocity(self):
        self.assertEqual(len(BFM_BODIES),30)
        p=torch.zeros((1,30,3)); q=torch.zeros((1,30,4)); q[...,0]=1
        v=torch.zeros_like(p); w=torch.zeros_like(p); w[:,15,0]=2.
        obs=privileged_state(p,q,v,w)
        # Linear velocities begin after root z + 30 relative positions + 31*6 rotations.
        begin=1+90+186
        torch.testing.assert_close(obs[:,begin+30*3:begin+31*3],torch.tensor([[0.,-.7,0.]]))

if __name__ == "__main__": unittest.main()
