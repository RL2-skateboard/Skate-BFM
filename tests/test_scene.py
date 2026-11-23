import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
import mujoco
from skate_sim.models.board import board_styles
from skate_sim.models.scene import build_scene_xml, initialize_scene, normalize_scene_config, load_scene_config, save_scene_config


class SceneTests(unittest.TestCase):
    def test_assets_combinations(self):
        for robot in (False, True):
            for board in (False, True):
                model = mujoco.MjModel.from_xml_string(build_scene_xml(dict(robot=robot, board=board)))
                self.assertEqual(model.neq, 2 if board else 0)
                self.assertEqual(model.nq, 36*robot + 14*board)
                self.assertEqual(sum(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE), 1)

    def test_original_assets_and_motion_coupling(self):
        m = mujoco.MjModel.from_xml_string(build_scene_xml(dict(robot=False)))
        np.testing.assert_allclose(m.joint('deck_roll').axis, [-1,0,0])
        np.testing.assert_allclose(m.joint('front_truck_tilt').axis, [0,0,-1])
        np.testing.assert_allclose(m.joint('rear_truck_tilt').axis, [0,0,1])
        np.testing.assert_allclose(m.eq_data[:,1], [-.577,-.577])
        self.assertAlmostEqual(float(m.body('chassis').mass[0]), .264)
        self.assertEqual(m.nu, 0)
        for end in ('front', 'rear'):
            joint = m.joint(f'{end}_truck_tilt')
            self.assertAlmostEqual(m.jnt_stiffness[joint.id], 2.)
            self.assertAlmostEqual(m.dof_damping[joint.dofadr[0]], .08)
        self.assertEqual(m.jnt_stiffness[m.joint('deck_roll').id], 0)

    def test_styles_and_obstacles(self):
        obstacles = [dict(id=k,kind=k,pos=[i,0,0],quat=[1,0,0,0]) for i,k in enumerate(['ramp','rail','platform'])]
        for style in board_styles():
            m=mujoco.MjModel.from_xml_string(build_scene_xml(dict(style=style,obstacles=obstacles)))
            self.assertGreater(m.body('obstacle_ramp').id,0)
            self.assertEqual(m.neq,2)
            self.assertTrue(np.isfinite(m.body_inertia).all())

    def test_roundtrip_and_validation(self):
        c=normalize_scene_config({'obstacles':[dict(id='a',kind='rail',pos=[1,2,3],quat=[2,0,0,0])]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.json';save_scene_config(c,path);self.assertEqual(c,load_scene_config(path))
        for value in ({'robot':1},{'robot_pose':dict(pos=[float('nan'),0,0],quat=[1,0,0,0])},{'board_pose':dict(pos=[0,0,0],quat=[0,0,0,0])}):
            with self.assertRaises(ValueError):normalize_scene_config(value)

    def test_derived_wheels_start_on_ground(self):
        for style in board_styles():
            model = mujoco.MjModel.from_xml_string(build_scene_xml(dict(robot=False, style=style)))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for end in ('front', 'rear'):
                for side in ('left', 'right'):
                    geom = model.geom(f'{end}_{side}_collision')
                    gap = data.geom_xpos[geom.id, 2] - geom.size[0]
                    self.assertLessEqual(abs(gap), 1e-4, (style, end, side, gap))

    def test_name_based_board_pose_and_wheel_speeds(self):
        c=dict(preset='steer_start',board_pose=dict(pos=[2,3,1],quat=[1,0,0,0]))
        m=mujoco.MjModel.from_xml_string(build_scene_xml(c));d=mujoco.MjData(m);initialize_scene(m,d,c)
        np.testing.assert_allclose(d.joint('deck_free').qpos[:3],[2,3,1])
        self.assertNotEqual(int(m.joint('deck_free').qposadr[0]),0)
        for end in ['front','rear']:
            for side in ['left','right']:
                np.testing.assert_allclose(d.joint(f'{end}_{side}_spin').qvel,1/.035)

    def test_rotated_start_moves_along_board(self):
        c = dict(robot=False, preset='steer_start', board_pose=dict(
            pos=[0,0,.1], quat=[np.sqrt(.5),0,0,np.sqrt(.5)]))
        model = mujoco.MjModel.from_xml_string(build_scene_xml(c))
        data = mujoco.MjData(model)
        initialize_scene(model, data, c)
        np.testing.assert_allclose(data.joint('deck_free').qvel[:3], [0,1,0], atol=1e-12)

if __name__ == '__main__':unittest.main()
