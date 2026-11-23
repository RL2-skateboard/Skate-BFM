"""Small risk regressions; CUDA execution remains an explicit host check."""
import numpy as np
import mujoco
import pytest

from skate_sim.backends.mjlab import MjlabBackend
from skate_sim.models.scene import build_scene_xml
from skate_sim.validation.gpu_robotiq import native_forces, closure, contact_groups, recorded_window


def test_public_backend_keeps_unvalidated_gripper_guard():
    pytest.importorskip('mjlab')
    with pytest.raises(ValueError,match='native MuJoCo only'):
        MjlabBackend(scene_config={'gripper':'right'})


def test_paired_force_probe_preserves_original_body_and_board_semantics():
    m=mujoco.MjModel.from_xml_string(build_scene_xml({'gripper':'right'}));d=mujoco.MjData(m)
    tau=np.linspace(-1,1,29)
    for end in ('front','rear'):
        d.joint(end+'_truck_tilt').qpos[0]=.05
        for side in ('left','right'):d.joint(f'{end}_{side}_spin').qvel[0]=2.
    native_forces(m,d,tau,128.)
    np.testing.assert_array_equal(d.qfrc_applied[:6],0)
    np.testing.assert_array_equal(d.ctrl[:6],0)
    assert d.ctrl[-1]==128
    assert d.qfrc_applied[m.joint('front_truck_tilt').dofadr[0]]==0
    native_forces(m,d,tau,255.,True)
    assert d.qfrc_applied[m.joint('front_truck_tilt').dofadr[0]]==pytest.approx(-.25*.05**3)
    assert d.qfrc_applied[m.joint('rear_left_spin').dofadr[0]]==pytest.approx(-.0004)
    mujoco.mj_forward(m,d)
    assert len(closure(m,d))==2
    assert set(contact_groups(m,d))=={'wheel_ground','foot_ground','toe_board','pad_board','left_pad_board','right_pad_board'}


def test_selected_window_requires_actual_58_56_source(tmp_path):
    p=tmp_path/'events.jsonl';p.write_text('{"event":"initial","state":{}}\n')
    with pytest.raises(ValueError,match='10 recorded ticks'):recorded_window(p,0.)
