"""Compiled asset checks; tolerances frozen before stage-02 verification."""
import numpy as np
import mujoco
from .board import board_styles, build_board_xml, board_hash


def check_assets():
    results = {}
    for name, s in board_styles().items():
        try:
            xml = build_board_xml(s)
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            core = m.geom('deck_collision').size * 2
            np.testing.assert_allclose(core,
                                       [s.deck_length-2*s.kick_length, s.deck_width, s.deck_thickness], atol=1e-6)
            assert m.nbody == 8 and m.njnt == 7 and m.nu == 0
            assert np.all(m.body_mass[1:] > 0) and np.all(m.body_inertia[1:] > 0)
            assert np.all(2*m.body_inertia[1:].max(axis=1) <= m.body_inertia[1:].sum(axis=1)+1e-12)
            np.testing.assert_allclose(m.body_mass.sum(), s.total_mass, atol=1e-6)
            errors = []
            for prefix, sign in [('front', 1), ('rear', -1)]:
                truck = m.body(prefix+'_truck').id
                assert m.body_parentid[truck] == m.body('deck').id
                for side, lateral in [('left', 1), ('right', -1)]:
                    wheel = prefix+'_'+side
                    bid, gid, jid = m.body(wheel).id, m.geom(wheel+'_collision').id, m.joint(wheel+'_spin').id
                    assert m.body_parentid[bid] == truck
                    np.testing.assert_allclose(d.xpos[bid], [sign*s.wheelbase/2, lateral*s.track/2, s.wheel_radius], atol=1e-6)
                    np.testing.assert_allclose(m.geom_size[gid,:2], [s.wheel_radius,s.wheel_width/2], atol=1e-6)
                    axis = d.geom_xmat[gid].reshape(3,3)[:,2]
                    angle = np.degrees(np.arccos(np.clip(abs(axis @ d.xaxis[jid]),0,1)))
                    assert angle < .1
                    errors.append(float(angle))
                    before = d.xpos[bid].copy()
                    d.qpos[m.jnt_qposadr[jid]] = .3
                    mujoco.mj_forward(m,d)
                    np.testing.assert_allclose(d.xpos[bid], before, atol=1e-9)
                    d.qpos[m.jnt_qposadr[jid]] = 0
            mujoco.mj_forward(m,d)
            assert all(d.contact[i].dist >= -1e-6 for i in range(d.ncon))
            assert xml == build_board_xml(s)
            for _ in range(500):
                mujoco.mj_step(m,d)
            assert np.isfinite(d.qpos).all() and .02 < d.qpos[2] < .3
            results[name] = dict(status='PASS', hash=board_hash(s), mass_kg=float(m.body_mass.sum()),
                                 max_axis_error_deg=max(errors), bodies=m.nbody, joints=m.njnt)
        except Exception as exc:
            results[name] = dict(status='FAIL', error=repr(exc))
    try:
        s = board_styles('standard')['standard']
        m = mujoco.MjModel.from_xml_string(build_board_xml(s,gallery=True))
        for name in ('gallery_ramp','gallery_low_platform','gallery_rail'):
            assert m.geom(name).contype[0] != 0
        assert m.geom('gallery_ramp').type[0] == mujoco.mjtGeom.mjGEOM_MESH
        results['gallery'] = dict(status='PASS', obstacles=3)
    except Exception as exc:
        results['gallery'] = dict(status='FAIL', error=repr(exc))
    return results
