"""Independent geometry/field reference for stage 05's 721-vector contract."""
import numpy as np
import mujoco
from skate_sim.robot.bfm import build_official_actor_input, deploy_joint_values
from skate_sim.robot.contract import HistoryState
from pathlib import Path

def official_reference():
    """Independent fixed official YAML bytes; never production config helpers."""
    import yaml, hashlib, re
    root=Path(__file__).with_name('reference05')
    hashes={'policy':'87ef74c51972f418aedc0838802965ef8238c08b958d58cebe4c41159977cfc9',
            'robot':'5a255ece4a10f6b0f412846fb593107cee969ffa2b0fcd4451acb52df762a75a',
            'walking':'4f148fa63c6fcdb0de203bef87cf3ce5872cee55d5fc7153a1e88452dbb08d0b'}
    cfg={}
    for name,sha in hashes.items():
        data=(root/(name+'.yaml')).read_bytes()
        if hashlib.sha256(data).hexdigest()!=sha:
            raise AssertionError('Official reference bytes changed: '+name)
        cfg[name]=yaml.safe_load(data)
    policy=cfg['policy']; names=policy['policy_joint_names']; values={}
    for key in ('default_joint_pos','action_scale','joint_kp','joint_kd'):
        values[key]=np.array([next((float(v) for pattern,v in policy[key].items() if re.fullmatch(pattern,name)),0.) for name in names])
    for key in ('joint_pos_lower_limit','joint_pos_upper_limit','joint_effort_limit'):
        values[key]=np.array([cfg['robot'][key][name] for name in names],dtype=float)
    values['names']=names
    values['obs_default']=np.array(policy['observation']['policy']['dof_pos_minus_default']['default_pos'])
    values['walking']=cfg['walking']; values['hashes']=hashes
    return values

FIELDS = dict(pos=(0,29), vel=(29,58), gravity=(58,61), angvel=(61,64),
              action=(64,93), history_action=(93,209), history_angvel=(209,221),
              history_pos=(221,337), history_vel=(337,453), history_gravity=(453,465), latent=(465,721))

def numeric_oracle(samples=128, *, backend_entry=False):
    rng=np.random.default_rng(5105)
    config=Path(__file__).resolve().parents[3]/'configs/bfm/motivo_newG1.yaml'
    ref=official_reference()
    v={'default':ref['obs_default'],'joint_pos_lower_limit':ref['joint_pos_lower_limit'],'joint_pos_upper_limit':ref['joint_pos_upper_limit']}
    history=HistoryState(); expected_hist={k:[] for k in ('action','angvel','pos','vel','gravity')}
    maxima={key:0. for key in FIELDS}; sign_error=0.
    backend=None
    if backend_entry:
        from skate_sim.backends.mujoco import MujocoBackend
        backend=MujocoBackend(scene='robot',controller='pose')
        backend.controller='policy';backend.policy_task='reward';backend.policy=lambda obs:np.zeros(29)
        backend._first_policy_debug=True;backend.robot_history=history
    for i in range(samples):
        if i % 17 == 0:
            history.reset(); expected_hist={k:[] for k in expected_hist}
        quat=rng.normal(size=4); quat/=np.linalg.norm(quat)
        if i < 6:
            quat=np.zeros(4); quat[0]=np.sqrt(.5); quat[1+i//2]=(-1)**i*np.sqrt(.5)
        rotation=np.empty(9); mujoco.mju_quat2Mat(rotation,quat); rotation=rotation.reshape(3,3)
        q=rng.uniform(v['joint_pos_lower_limit'],v['joint_pos_upper_limit'])
        dq=rng.uniform(-2,2,29); world_w=np.eye(3)[i%3]*(.25+i/128)
        w=rotation.T@world_w
        action=rng.uniform(-5,5,29); z=rng.normal(size=256)
        if backend is not None:
            backend.data.qpos[:3]=[0,0,3]
            backend.data.qpos[3:7]=quat
            backend.data.qpos[backend.robot_qpos_adr]=q
            backend.data.qvel[:6]=np.r_[np.zeros(3),world_w]
            backend.data.qvel[backend.robot_dof_adr]=dq
            mujoco.mj_forward(backend.model,backend.data)
            base=backend.model.body('pelvis').id
            rotation=backend.data.xmat[base].reshape(3,3).copy()
            velocity=np.empty(6)
            mujoco.mj_objectVelocity(backend.model,backend.data,mujoco.mjtObj.mjOBJ_BODY,base,velocity,0)
            w=rotation.T@velocity[:3]
        vals={'pos':q-v['default'],'vel':dq,'gravity':rotation.T@np.array([0,0,-1]),'angvel':.25*w,'action':action}
        expected=[]
        for key in ('pos','vel','gravity','angvel','action'): expected.extend(vals[key])
        for key in ('action','angvel','pos','vel','gravity'):
            expected_hist[key].insert(0,vals[key].copy()); expected_hist[key]=expected_hist[key][:4]
            expected.extend(np.concatenate(expected_hist[key]+[np.zeros_like(vals[key])]*(4-len(expected_hist[key]))))
        expected=np.r_[expected,z].astype(np.float32)
        if backend is None:
            obs=build_official_actor_input(q,dq,quat,w,action,history,z)
        else:
            backend.policy_z=z;backend.robot_history.last_action[:]=action
            backend.physics_step_counter=0
            backend.step();obs=backend.last_obs.copy()
        np.testing.assert_allclose(obs,expected,atol=1e-5,rtol=1e-4)
        other=build_official_actor_input(q,dq,-quat,w,action,HistoryState(),z)
        sign_error=max(sign_error,float(abs(other[:93]-obs[:93]).max()))
        for key,(a,b) in FIELDS.items(): maxima[key]=max(maxima[key],float(abs(obs[a:b]-expected[a:b]).max()))
    if backend is not None:
        backend.close()
    assert sign_error <= 1e-6, f'quaternion sign symmetry error: {sign_error}'
    return {'samples':samples,'backend_entry':backend_entry,'atol':1e-5,'rtol':1e-4,'field_max_abs':maxima,'quaternion_sign_max_abs':sign_error}

def test_noise_preserves_actions_and_context():
    args=(np.arange(29)*.01,np.ones(29),np.array([1.,0,0,0]),np.ones(3),np.arange(29)*.1)
    z=np.arange(256,dtype=np.float32)
    obs=build_official_actor_input(*args,HistoryState(),z,sensor_noise_std=.2,rng=np.random.default_rng(5))
    np.testing.assert_array_equal(obs[465:],z)
    np.testing.assert_allclose(obs[64:93],args[-1])
    np.testing.assert_array_equal(obs[64:93],obs[93:122])
    np.testing.assert_array_equal(obs[:29],obs[221:250])
    np.testing.assert_array_equal(obs[29:58],obs[337:366])
    np.testing.assert_array_equal(obs[58:61],obs[453:456])
    np.testing.assert_array_equal(obs[61:64],obs[209:212])

def native_contract_probe():
    """Exercise native backend mapping, held target, dynamic PD and reset."""
    from skate_sim.backends.mujoco import MujocoBackend
    b=MujocoBackend(scene='robot',controller='pose')
    try:
        b.controller='policy'; b.policy_task='reward'; b.policy_z=np.linspace(-1,1,256,dtype=np.float32)
        b.policy_context_name='contract-probe'; b._first_policy_debug=True
        actions=np.linspace(-2,2,29); calls=[]
        b.policy=lambda obs: (calls.append(obs.copy()) or actions.copy())
        b.data.qpos[3:7]=np.array([.5,.5,.5,.5]); b.data.qvel[3:6]=[.3,-.2,.7]
        mujoco.mj_forward(b.model,b.data)
        base=b.model.body('pelvis').id
        expected_w=.25*b.data.xmat[base].reshape(3,3).T@b.data.cvel[base,:3]
        expected_g=b.data.xmat[base].reshape(3,3).T@np.array([0,0,-1])
        b.step()
        np.testing.assert_allclose(calls[0][58:61],expected_g,atol=1e-6)
        np.testing.assert_allclose(calls[0][61:64],expected_w,atol=1e-6)
        ref=official_reference()
        assert b.robot_joint_names==ref['names']
        target=np.clip(ref['default_joint_pos']+5*ref['action_scale']*np.clip(actions,-1,1),ref['joint_pos_lower_limit'],ref['joint_pos_upper_limit'])
        np.testing.assert_allclose(b.q_target,target,atol=1e-12)
        target=b.q_target.copy()
        tau=[b.last_tau.copy()]; errors=[]
        for _ in range(9):
            q=b._robot_q().copy(); dq=b._robot_dq().copy()
            expected_tau=np.clip(ref['joint_kp']*(target-q)-ref['joint_kd']*dq,-ref['joint_effort_limit'],ref['joint_effort_limit'])
            b.step(); np.testing.assert_allclose(b.q_target,target,atol=0,rtol=0)
            errors.append(float(abs(b.last_tau-expected_tau).max()))
            np.testing.assert_allclose(b.last_qfrc_applied,expected_tau,atol=1e-12)
            tau.append(b.last_tau.copy())
        assert len(calls)==1
        b.step(); assert len(calls)==2
        np.testing.assert_allclose(calls[1][64:93],5*np.clip(actions,-1,1),atol=1e-6)
        np.testing.assert_allclose(calls[1][93:122],calls[1][64:93],atol=0)
        assert np.max(np.ptp(tau,axis=0))>1e-3
        # Each name maps to the actual named hinge, all 29 including wrists.
        for i,name in enumerate(b.robot_joint_names):
            assert b.model.joint(name).qposadr[0]==b.robot_qpos_adr[i]
            assert b.model.joint(name).dofadr[0]==b.robot_dof_adr[i]
        b.parameters['control_delay_steps']=2
        b._action_queue.append(np.ones(29))
        reset_obs=b.reset()
        np.testing.assert_array_equal(reset_obs,b.last_obs)
        np.testing.assert_array_equal(reset_obs[465:],b.policy_z)
        assert not b._action_queue and not b.robot_history.action_frames
        assert not np.any(b.robot_history.last_action) and not np.any(b.policy_action)
        assert b.physics_step_counter==0 and b.policy_counter==0
        b.step(); np.testing.assert_array_equal(calls[-1][64:93],0)
        assert np.count_nonzero(calls[-1][122:209])==0
        b.reset();b.parameters['control_delay_steps']=2
        for cycle in range(3):
            for _ in range(10): b.step()
            if cycle<2:
                np.testing.assert_array_equal(b.policy_action,0)
                np.testing.assert_array_equal(b.q_target,ref['default_joint_pos'])
            else:
                np.testing.assert_allclose(b.policy_action,np.clip(actions,-1,1))
            np.testing.assert_allclose(b.robot_history.last_action,5*np.clip(actions,-1,1),atol=1e-6)
            if cycle>0:
                np.testing.assert_allclose(calls[-1][64:93],5*np.clip(actions,-1,1),atol=1e-6)
        unconstrained=ref['default_joint_pos']+5*ref['action_scale']*np.clip(actions,-1,1)
        lower=unconstrained<ref['joint_pos_lower_limit'];upper=unconstrained>ref['joint_pos_upper_limit']
        assert lower.any() and upper.any()
        np.testing.assert_array_equal(b.q_target[lower],ref['joint_pos_lower_limit'][lower])
        np.testing.assert_array_equal(b.q_target[upper],ref['joint_pos_upper_limit'][upper])
        return {'all_29_named_mapping':True,'all_29_action_scale_clip':True,'world_to_body_backend':True,'target_held_physics_substeps':10,'inferences_in_first_10_substeps':1,'pd_written_qfrc_max_abs_error':max(errors),'tau_changed':True,'history_rescaled_previous_action':True,'reset_history_action_delay':True,'delay_2_command_feedback_distinct_from_applied':True,'actual_lower_upper_clip':True,'reset_returns_fresh_obs':True}
    finally:
        b.close()

def policy_rollout(checkpoint, *, task='reward', name=None, output=None, steps=15000):
    """Real CPU ONNX rollout; durable complete inputs/actions, no auto reset."""
    import time
    import json
    import hashlib
    from skate_sim.backends.mujoco import MujocoBackend
    out=Path(output) if output else Path('runs/05_observation_fix')/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=True)
    record=out/(task+'.npz')
    if record.exists():
        raise FileExistsError(f'Refusing to overwrite existing evidence: {record}')
    b=MujocoBackend(scene='robot',controller='policy',checkpoint=checkpoint,policy_task=task,latent_name=name)
    b._first_policy_debug=True
    obs=[]; raw=[]; applied=[]; latency=[]; pelvis=[]; saturation=[]
    first_fall=None; start=time.perf_counter()
    try:
        for _ in range(steps):
            boundary=b.physics_step_counter % b.config.substeps==0
            b.step()
            height=float(b.data.xpos[b.model.body('pelvis').id,2]); pelvis.append(height)
            if height<.35 and first_fall is None:
                first_fall=float(b.data.time)
            if boundary:
                obs.append(b.last_obs.copy()); raw.append(b.raw_action.copy())
                applied.append(b.policy_action.copy()); latency.append(b.inference_time_ms)
            saturation.append(np.count_nonzero(np.isclose(abs(b.last_tau),b.policy_values['joint_effort_limit'],atol=1e-6,rtol=0)))
        np.savez_compressed(record,obs=obs,raw_action=raw,applied_action=applied,inference_ms=latency,pelvis_height=pelvis,saturated_joint_count=saturation)
        result={'task':task,'context_name':b.policy_context_name,'physics_steps':b.physics_step_counter,
                'inferences':b.policy_counter,'sim_seconds':float(b.data.time),
                'wall_seconds':time.perf_counter()-start,'min_pelvis_height':min(pelvis),
                'first_fall_seconds':first_fall,'fall_height_threshold_m':.35,'resets_during_run':0,
                'inference_ms_mean':float(np.mean(latency)),'inference_ms_max':max(latency),
                'saturation_fraction':float(np.sum(saturation)/(steps*29)),
                'finite':bool(np.isfinite(obs).all() and np.isfinite(raw).all() and np.isfinite(b.data.qpos).all()),
                'record':str(record.resolve()),'environment':'independent skate-sim native MuJoCo CPU'}
        repo=Path(__file__).resolve().parents[3]
        def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        checkpoint_root=Path(checkpoint).resolve()
        latent_file={'reward':'reward_inference/reward_locomotion.pkl','tracking':'tracking_inference/zs_walking.pkl','goal':'goal_inference/goal_reaching.pkl'}[task]
        result['provenance']={
            'official_deploy_commit':'de9e248ce8b315508aafaecc8f15def3e057fccc',
            'seed':0,'physics_dt':b.config.physics_dt,'control_dt':b.config.control_dt,
            'onnx_sha256':digest(checkpoint_root/'model/exported/FBcprAuxModel.onnx'),
            'latent_file':latent_file,'latent_sha256':digest(checkpoint_root/'model'/latent_file),
            'source_sha256':{str(p.relative_to(repo)):digest(p) for p in [
                repo/'src/skate_sim/robot/bfm.py',repo/'src/skate_sim/robot/contract.py',
                repo/'src/skate_sim/backends/mujoco.py',repo/'src/skate_sim/backends/mjlab.py',
                repo/'src/skate_sim/validation/observation05.py',repo/'src/skate_sim/cli.py']},
            'config_sha256':{str(p.relative_to(repo)):digest(p) for p in [
                repo/'configs/bfm/motivo_newG1.yaml',repo/'configs/bfm/g1.yaml']}}
        (out/(task+'_summary.json')).write_text(json.dumps(result,indent=2))
        return result
    finally:
        b.close()


def mjlab_cpu_mock_probe():
    """Call GPU adapter methods against CPU tensors; does NOT validate CUDA."""
    import torch
    from types import SimpleNamespace as NS
    from skate_sim.backends.mjlab import MjlabBackend
    from skate_sim.config import SimConfig
    b=object.__new__(MjlabBackend); b.num_envs=2; b._torch=torch; b.config=SimConfig()
    def field(a): return NS(_tensor=torch.as_tensor(a,dtype=torch.float64).clone())
    quat=np.array([[.5,.5,.5,.5],[np.sqrt(.5),0,np.sqrt(.5),0]])
    rot=[]
    for q in quat:
        r=np.empty(9);mujoco.mju_quat2Mat(r,q);rot.append(r)
    data=NS(qpos=field(np.ones((2,29))*.02),qvel=field(np.ones((2,29))*.01),
            xquat=field(quat[:,None,:]),xmat=field(np.array(rot)[:,None,:]),
            cvel=field(np.array([[[.2,.5,.7,0,0,0]],[[.6,.1,.3,0,0,0]]])),
            qfrc_applied=field(np.zeros((2,29))),xfrc_applied=field(np.zeros((2,1,6))),time=field([1.,1.]))
    b.sim=NS(data=data,forward=lambda:None); b.pelvis_body_id=0; b.robot_qpos_index=np.arange(29);b.robot_dof_index=np.arange(29)
    b._synchronize=lambda:None; b.board_style=None;b.scene='robot';b.controller='policy';b.seed=0
    b.policy_task='tracking';b.policy_z=np.arange(20*256,dtype=np.float32).reshape(20,256)+1
    b.policy_context_index=np.array([3,7]);b.policy_tracking_active=np.array([True,True]);b.robot_history=[HistoryState(),HistoryState()]
    b.parameter_batch=[{'pd_kp_scale':1.,'pd_kd_scale':1.,'observation_noise_std':0.,'control_delay_steps':0} for _ in range(2)]
    b.policy_values=deploy_joint_values(Path(__file__).resolve().parents[3]/'configs/bfm/motivo_newG1.yaml')
    b.physics_step_counter=0;b.policy_counter=0;b.policy_action=np.zeros((2,29));b.raw_action=b.policy_action.copy()
    b.last_obs=np.zeros((2,721));b.q_target=np.zeros((2,29));b.pose_target=np.zeros((2,29));b._action_queues=[[],[]]
    seen=[];b.policy=lambda obs:(seen.append(obs.copy()) or np.linspace(-2,2,29))
    b._write_field=lambda name,value:getattr(data,name)._tensor.copy_(torch.as_tensor(value,dtype=torch.float64))
    b._update_controller()
    for i in range(2):
        rotation=np.array(rot[i]).reshape(3,3)
        np.testing.assert_allclose(seen[i][61:64],.25*rotation.T@data.cvel._tensor[i,0,:3].numpy(),atol=1e-6)
    np.testing.assert_array_equal(b.policy_context_index,[4,8])
    # Reset one real adapter slot and ensure untouched env's cursor/history/action survives.
    b._resample_parameters=lambda ids,seed:None
    b._read_state_snapshot=lambda:{name:getattr(data,name)._tensor.numpy().copy() for name in ('qpos','qvel','time','xfrc_applied','qfrc_applied')}
    b._nominal_qpos=torch.zeros((2,29),dtype=torch.float64);b._nominal_qvel=b._nominal_qpos.clone()
    b._force=np.zeros(3);b._mouse_force=np.zeros(3);b.view_env=0;b._board_root_qpos=None
    b._sync_viewer=lambda:None;b._capture_state_snapshot=lambda:None
    untouched=b.last_obs[1].copy();action=b.policy_action[1].copy()
    reset_obs=b.reset(env_ids=[0])
    np.testing.assert_array_equal(reset_obs,b.last_obs[[0]])
    np.testing.assert_array_equal(b.policy_tracking_active,[True,True])
    np.testing.assert_array_equal(b.policy_context_index,[0,8])
    np.testing.assert_array_equal(b.last_obs[1],untouched);np.testing.assert_array_equal(b.policy_action[1],action)
    assert not b.robot_history[0].action_frames and b.robot_history[1].action_frames
    assert not np.any(b.robot_history[0].last_action)
    return {'execution':'CPU tensor mock only; no CUDA simulation','body_angvel_adapter':True,'per_env_tracking_cursor':True,'partial_reset_isolation':True,'reset_returns_fresh_obs':True}


def tracking_reference_probe():
    from skate_sim.robot.bfm import tracking_context, next_tracking_cursor
    import math
    ref=official_reference(); cfg=ref['walking']
    assert (cfg['start'],cfg['end'],cfg['stop'])==(0,2000,0)
    rng=np.random.default_rng(5003); errors=[]
    for n in (2,7,2000,2002):
        contexts=rng.normal(size=(n,256)).astype(np.float32)
        for index in (0,1,n-1):
            rows=contexts[index:min(index+cfg['window_size'],n)]
            weights=[cfg['gamma']**j for j in range(len(rows))]
            mean=np.array([math.fsum(float(row[k])*w for row,w in zip(rows,weights))/math.fsum(weights) for k in range(256)])
            expected=(mean/math.sqrt(math.fsum(float(x*x) for x in mean))*math.sqrt(math.fsum(float(x*x) for x in contexts[0].astype(float)))).astype(np.float32)
            actual=tracking_context(contexts,index)
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
            errors.append(float(abs(actual-expected).max()))
    index=1999;active=True;seen=[]
    for _ in range(4):
        seen.append(index);index,active=next_tracking_cursor(index,active,2500)
    assert seen==[1999,2000,0,0] and not active
    index=0;active=True;seen=[]
    for _ in range(5):
        seen.append(index);index,active=next_tracking_cursor(index,active,2)
    assert seen==[0,1,0,1,0] and active
    return {'context_window_max_abs':max(errors),'gamma':cfg['gamma'],'window_size':cfg['window_size'],
            'end':cfg['end'],'large_sequence_boundary':[1999,2000,0,0],
            'short_sequence_cycles':True,'tail_window_does_not_wrap':True,
            'official_commit':'de9e248ce8b315508aafaecc8f15def3e057fccc','reference_sha256':ref['hashes']}

