"""Raw native29/offline23 conversion contracts; no retired simulator imports."""
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pytest


def module(name):
    path = Path(__file__).parents[1] / 'train/scripts/data_collection' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('name', ['convert_phase', 'convert_continuous'])
def test_native_wrist_and_action_provenance(name, tmp_path):
    m = module(name)
    n = 12
    # A reversed source joint order catches accidental positional copy and zero wrists.
    order = list(reversed(m.BFM_JOINT_ORDER))
    state = {key:np.zeros((n,)) for key in m.RAW_REQUIRED}
    state.update(dof_pos=np.arange(n*29, dtype=np.float32).reshape(n,29)/100,
                 action=np.arange(n*29, dtype=np.float32).reshape(n,29)/500,
                 root_pos=np.zeros((n,3)),root_quat=np.tile([1.,0,0,0],(n,1)),
                 board_root_pos=np.zeros((n,3)),board_root_quat=np.tile([1.,0,0,0],(n,1)),
                 board_root_lin_vel=np.zeros((n,3)),board_root_ang_vel=np.zeros((n,3)),
                 board_dof_pos=np.zeros((n,6)),board_dof_vel=np.zeros((n,6)),
                 phase_id=np.full(n,7,dtype=np.int16),frame_idx=np.arange(n))
    meta = dict(joint_order=order,fixed_bfm_joints=[],fps=50,round_id=1,rollout_id=1,
                episode_id='1',physics_randomization={'seed':0},dataset_split='train',
                source_joint_support=29,quality='policy_rollout_unvalidated',phase_annotation='unavailable',command_annotation='unavailable')
    state.pop('command_v');state.pop('command_h')
    args = [meta,state,[],{'fps':50},m.BFM_JOINT_ORDER,np.tile([1,0,0],(29,1))]
    if name == 'convert_phase': args += ['unlabelled_rollout',0,n,tmp_path/'raw.npz',8]
    else: args += [0,n,tmp_path/'raw.npz',8,n]
    record, summary = m.convert_record(*args)
    np.testing.assert_array_equal(record['dof'], state['dof_pos'][:,::-1])
    np.testing.assert_array_equal(record['action'], state['action'])
    assert record['source_joint_support']==29
    assert record['missing_joint_names']==[]
    assert not record['native29_expert']
    assert summary['fixed_zero_joints']==[]
    assert record['phase_annotation']=='unavailable'
    assert record['command_v'] is None and record['command_h'] is None


@pytest.mark.parametrize('name', ['convert_phase', 'convert_continuous'])
def test_historical23_is_offline_and_explicitly_incomplete(name, tmp_path):
    m = module(name)
    wrists = [joint for joint in m.BFM_JOINT_ORDER if 'wrist' in joint]
    source = [joint for joint in m.BFM_JOINT_ORDER if joint not in wrists]
    converted, mapping = m.map_dof(np.ones((12,23)),source,m.BFM_JOINT_ORDER,wrists)
    assert sum(value=='fixed_zero' for value in mapping.values())==6
    assert converted.shape==(12,29)
    raw = tmp_path/'raw.npz'
    raw.with_suffix('.json').write_text(json.dumps({'joint_order':source}))
    motion = tmp_path/'motion.pkl'
    joblib.dump({'legacy':{'fps':50,'source_raw_npz':str(raw)}},motion)
    model = tmp_path/'model.xml';model.write_text('not a MuJoCo model')
    with pytest.raises(ValueError,match='historical23 raw is offline conversion only'):
        m.generate_qc(motion,tmp_path/'qc',model,0)


@pytest.mark.parametrize('name', ['convert_phase', 'convert_continuous'])
def test_reset_clock_and_seq8_minimum(name, tmp_path):
    m = module(name)
    n=10
    state={key:np.zeros((n,)) for key in m.RAW_REQUIRED}
    for key,shape in {'root_pos':3,'root_quat':4,'dof_pos':29,'dof_vel':29,'action':29,
                      'board_root_pos':3,'board_root_quat':4,'board_root_lin_vel':3,'board_root_ang_vel':3}.items():
        state[key]=np.zeros((n,shape))
    state.update(frame_idx=np.arange(n),phase_id=np.full(n,7,dtype=np.int16),
                 sim_time=np.r_[np.arange(5)*.02,np.arange(5)*.02])
    state['reset'][5]=True
    meta=dict(num_frames=n,episode_id='1',round_id=1,rollout_id=1,dataset_split='train',fps=50,dt=.02,
              joint_order=m.BFM_JOINT_ORDER,body_order=[],board_joint_order=[],
              qpos_quaternion_order='wxyz',body_quaternion_order='wxyz',board_quaternion_order='wxyz',
              physics_randomization={'seed':0},policy_checkpoint='native29',robot_xml='model.xml',terminal_reason='horizon',fixed_bfm_joints=[])
    assert m.validate_raw(tmp_path/'raw.npz',meta,state)[0]==n
    state['reset'][5]=False
    with pytest.raises(ValueError,match='without a recorded reset'):
        m.validate_raw(tmp_path/'raw.npz',meta,state)
    # Nine raw frames provide only eight MotionLib states and cannot supply Seq8 + next.
    args=[meta,state,[],{'fps':50},m.BFM_JOINT_ORDER,np.tile([1,0,0],(29,1))]
    if name=='convert_phase':args+=['unlabelled_rollout',0,9,tmp_path/'raw.npz',8]
    else:args+=[0,9,tmp_path/'raw.npz',8,9]
    with pytest.raises(ValueError,match='BFM minimum 10'):
        m.convert_record(*args)
