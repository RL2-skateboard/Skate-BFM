import contextlib
import io
import numpy as np
import pytest
import mujoco
from skate_sim.models.scene import normalize_scene_config,build_scene_xml
from skate_sim.models.gripper import ASSET,compare_mechanics,measured_opening
from skate_sim.backends.mujoco import MujocoBackend
from skate_sim.record import Recorder,numerics_manifest
from skate_sim.replay import replay


def make(selection='right'):
    with contextlib.redirect_stdout(io.StringIO()):
        return MujocoBackend(scene_config=normalize_scene_config(dict(gripper=selection,board=False)),controller='pose')


def test_official_mechanics_and_named29():
    assert build_scene_xml({})==build_scene_xml({'gripper':'none'})
    plain=make('none');grip=make();source=mujoco.MjModel.from_xml_path(str(ASSET))
    try:
        assert grip.model.nq==plain.model.nq+8
        assert grip.model.nu==plain.model.nu+1
        assert grip.model.neq==plain.model.neq+3
        assert grip.model.ntendon==plain.model.ntendon+1
        assert grip.model.nexclude==plain.model.nexclude+6
        assert grip.robot_joint_names==plain.robot_joint_names and len(grip.robot_joint_names)==29
        assert compare_mechanics(grip.model,'right')['property_differences']==[]
        np.testing.assert_allclose(grip.model.eq_data[-3:],source.eq_data,atol=1e-12,rtol=0)
        for name in plain.robot_joint_names:
            np.testing.assert_array_equal(grip.model.joint(name).range,plain.model.joint(name).range)
        with pytest.raises(ValueError):grip.set_gripper_command(float('nan'))
        with pytest.raises(ValueError):grip.set_gripper_command(256)
    finally:plain.close();grip.close()


def test_raw_command_direction_and_reset():
    b=make()
    try:
        # No-load actuator test: high initial free root, no board/fixture/ground contact.
        openings=[]
        for raw in [0.,128.,255.]:
            b.reset();b.data.joint('floating_base_joint').qpos[:3]=[0,0,4]
            mujoco.mj_forward(b.model,b.data);b.set_gripper_command(raw)
            for _ in range(150):b.step(update_policy=False)
            openings.append(measured_opening(b.model,b.data,'right'))
            assert np.max(abs(b.data.actuator_force[-1:]))<=5+1e-8
            np.testing.assert_array_equal(b.data.xfrc_applied,0)
            np.testing.assert_array_equal(b.data.qfrc_applied[:6],0)
        assert openings[0]>.075 and .025<openings[1]<.065 and openings[2]<.005
        assert openings[0]>openings[1]>openings[2]
        b.reset();assert b.gripper_command=={'right':0.}
    finally:b.close()


def test_recorded_raw_control(tmp_path):
    b=make();r=Recorder(str(tmp_path/'record'),backend='mujoco',scene='robot',seed=0,metadata={'numerics':numerics_manifest(b)})
    try:
        r.start(b.get_state(),backend_kwargs=dict(scene_config=b.scene_config,controller='pose'),runtime_state=b.export_replay_state());b.recorder=r
        for raw in [0.,255.,128.]:
            b.set_gripper_command(raw)
            for _ in range(170):b.step(update_policy=False)
    finally:r.close();b.close()
    result=replay(tmp_path/'record',mode='dynamics',headless=True)
    assert result['status']=='PASS'
    assert result['max_abs_error']==0
