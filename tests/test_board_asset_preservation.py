"""Neutral asset identity and physical collision geometry for the motion board."""
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from skate_sim.models.board import board_styles, build_board_xml, build_motion_board_xml


def aggregate(model, data):
    mass = model.body_mass[1:]
    pos = data.xipos[1:]
    center = (mass[:, None]*pos).sum(axis=0)/mass.sum()
    inertia = np.zeros((3, 3))
    for i, (weight, offset) in enumerate(zip(mass, pos-center), start=1):
        rotation = data.ximat[i].reshape(3, 3)
        inertia += rotation @ np.diag(model.body_inertia[i]) @ rotation.T
        inertia += weight*(np.dot(offset, offset)*np.eye(3)-np.outer(offset, offset))
    return mass.sum(), center, inertia


class BoardAssetPreservationTests(unittest.TestCase):
    def test_neutral_geometry_and_mass_properties(self):
        for name, style in board_styles().items():
            with self.subTest(style=name):
                old_xml = build_board_xml(style)
                new_xml = build_motion_board_xml(name)
                old = mujoco.MjModel.from_xml_string(old_xml)
                new = mujoco.MjModel.from_xml_string(new_xml)
                od, nd = mujoco.MjData(old), mujoco.MjData(new)
                mujoco.mj_forward(old, od); mujoco.mj_forward(new, nd)
                geoms = ET.fromstring(old_xml).findall(".//body[@name='deck']//geom")
                new_geoms = {g.get('name'): g for g in ET.fromstring(new_xml).findall('.//geom')}
                for geom in geoms:
                    key = geom.get('name')
                    a, b = old.geom(key).id, new.geom(key).id
                    np.testing.assert_allclose(od.geom_xpos[a], nd.geom_xpos[b], atol=1e-14)
                    np.testing.assert_allclose(od.geom_xmat[a], nd.geom_xmat[b], atol=1e-14)
                    np.testing.assert_allclose(old.geom_size[a], new.geom_size[b], atol=1e-14)
                    self.assertEqual(geom.get('mass'), new_geoms[key].get('mass'))
                for a, b in zip(aggregate(old, od), aggregate(new, nd)):
                    np.testing.assert_allclose(a, b, atol=1e-13, rtol=1e-13)
                self.assertAlmostEqual(aggregate(new, nd)[0], style.total_mass, places=10)
                self.assertEqual(new.nexclude, 0)
                for end in ('front', 'rear'):
                    bridge = new.geom(f'{end}_truck_collision')
                    self.assertEqual(bridge.bodyid[0], new.body('chassis').id)
                    self.assertNotEqual(bridge.contype[0], 0)
                    top = nd.geom_xpos[bridge.id, 2]+bridge.size[2]
                    bottom = nd.geom_xpos[bridge.id, 2]-bridge.size[2]
                    deck_bottom = nd.xpos[new.body('deck').id, 2]-style.deck_thickness/2
                    axle_z = nd.geom_xpos[new.geom(f'{end}_axle_collision').id, 2]
                    self.assertAlmostEqual(top, deck_bottom)
                    self.assertAlmostEqual(bottom, axle_z)

    def test_kicked_tips_are_real_collision_surfaces(self):
        for name, style in board_styles().items():
            model = mujoco.MjModel.from_xml_string(build_motion_board_xml(name))
            data = mujoco.MjData(model)
            model.geom_group[:] = 4
            for i in range(model.ngeom):
                if model.geom(i).name.startswith('deck_strip_'):
                    model.geom_group[i] = 0
            groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
            for roll in (0., -.1, .1):
                data.joint('deck_roll').qpos[:] = roll
                for end in ('front', 'rear'):
                    data.joint(f'{end}_truck_tilt').qpos[:] = -roll/.577
                mujoco.mj_forward(model, data)
                heights = []
                for x in np.linspace(-style.deck_length*.49, style.deck_length*.49, 101):
                    point = np.array([x+1e-7, 0., 1.])
                    hit = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(model, data, point, np.array([0.,0.,-1.]), groups, True, -1, hit)
                    self.assertGreater(distance, 0, (name, roll, x))
                    self.assertTrue(model.geom(hit[0]).name.startswith('deck_strip_'))
                    heights.append(1-distance)
                self.assertGreater(heights[0]-heights[50], style.kick_height*.7)
                self.assertGreater(heights[-1]-heights[50], style.kick_height*.7)
                self.assertTrue(np.isfinite(data.qacc).all())

if __name__ == '__main__':
    unittest.main()
