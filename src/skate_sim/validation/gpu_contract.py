"""Real CUDA non-contact checks; no CPU mock may satisfy this entry point."""
from __future__ import annotations
import time
import numpy as np
from skate_sim.backends.mjlab import MjlabBackend


def run(checkpoint, num_envs=64, duration=30., scene='robot'):
    backend = MjlabBackend(scene=scene, controller='policy', checkpoint=checkpoint,
                           num_envs=num_envs)
    assert str(backend.sim.device).startswith('cuda')
    started = time.monotonic()
    oracle = observation_oracle(backend)
    interfaces = interface_probe(backend)
    ids = np.asarray(sorted(set([0, min(7,num_envs-1), num_envs-1])))
    untouched = np.ones(num_envs,bool); untouched[ids] = False
    before = backend.get_state()
    subset = backend.get_state(ids)
    subset['qvel'][:,0] = .1
    backend.set_state(subset,ids)
    after = backend.get_state()
    for key in before:
        assert np.array_equal(before[key][untouched],after[key][untouched]), key
    invalid = backend.get_state(ids); invalid['qvel'][0,0] = np.nan
    before_invalid = backend.get_state()
    try: backend.set_state(invalid,ids)
    except ValueError: pass
    else: raise AssertionError('invalid state accepted')
    for key,value in before_invalid.items(): assert np.array_equal(value,backend.get_state()[key])
    backend.apply_wrench('pelvis',(2,0,0),env_ids=ids,duration_steps=2)
    assert np.count_nonzero(backend.get_state()['xfrc_applied'][untouched]) == 0
    backend.step(); backend.step()
    assert np.count_nonzero(backend.get_state()['xfrc_applied']) == 0
    before = backend.get_state()
    backend.reset(env_ids=ids)
    after = backend.get_state()
    for key in before: assert np.array_equal(before[key][untouched],after[key][untouched]), key
    # Perturb one joint, test live substep PD rather than a shape-only contract.
    subset = backend.get_state(ids)
    subset['qpos'][:,backend.robot_qpos_adr[0]] += .03
    backend.set_state(subset,ids)
    backend.step(); tau0=backend.last_tau.copy(); backend.step()
    assert not np.array_equal(tau0,backend.last_tau)
    start_counter=backend.policy_counter; start_step=backend.physics_step_counter
    start_time=backend.get_state()["time"].tolist()
    peak_action=0.; peak_tau=0.; finite=True
    steps=round(duration/backend.config.physics_dt)
    for step in range(steps):
        backend.step()
        if step % backend.config.substeps == 0:
            finite &= bool(np.isfinite(backend.last_obs).all() and np.isfinite(backend.last_tau).all())
            peak_action=max(peak_action,float(np.max(np.abs(backend.raw_action))))
            peak_tau=max(peak_tau,float(np.max(np.abs(backend.last_tau))))
        if step % 2500 == 0: print(f'GPU_PROGRESS scene={scene} N={num_envs} t={step*.002:.1f}',flush=True)
    assert finite and peak_action > 0 and peak_tau > 0
    report={'scene':scene,'num_envs':num_envs,'physics_device':str(backend.sim.device),
            'policy_device':'CPUExecutionProvider','duration_s':duration,'wall_s':time.monotonic()-started,
            'steps':steps,'start_step':start_step,'start_time':start_time,'end_time':backend.get_state()['time'].tolist(),
            'start_policy_counter':start_counter,'policy_calls_per_env':backend.policy_counter-start_counter,'policy_counter_total':backend.policy_counter,'policy_task':'reward','finite':finite,
            'interfaces':interfaces,'observation_oracle':oracle,'peak_raw_action':peak_action,'peak_tau':peak_tau,'subset_atomic_validation':True,
            'partial_reset_isolation':True,'substep_pd':True,'wrench_isolation_duration':True,
            'termination_batch':backend.termination(),
            'limits':'Contact experiments and long-term stability/generalization are not certified.'}
    backend.close()
    return report


def observation_oracle(backend, samples=128):
    """Compare CUDA state adapter/history to independent native kinematics and YAML fields."""
    import mujoco
    from skate_sim.validation.observation05 import official_reference
    from skate_sim.robot.contract import HistoryState
    ref=official_reference(); rng=np.random.default_rng(705)
    native=mujoco.MjData(backend.sim.mj_model)
    joint=int(backend.sim.mj_model.body('pelvis').jntadr[0])
    root_q=int(backend.sim.mj_model.jnt_qposadr[joint]);root_v=int(backend.sim.mj_model.jnt_dofadr[joint])
    saved=(backend.policy,backend.policy_z,backend.policy_task,backend.controller)
    backend.controller='policy';backend.policy_task='reward';backend.policy_z=np.linspace(-1,1,256,dtype=np.float32)
    backend.policy=lambda obs: np.linspace(-2,2,29)
    backend.robot_history=[HistoryState() for _ in range(backend.num_envs)]
    histories=[{k:[] for k in ('action','angvel','pos','vel','gravity')} for _ in range(backend.num_envs)]
    maximum=0.; count=0
    try:
        while count < samples:
            state=backend.get_state(); expected=[]
            for env in range(backend.num_envs):
                quat=rng.normal(size=4);quat/=np.linalg.norm(quat)
                q=rng.uniform(ref['joint_pos_lower_limit'],ref['joint_pos_upper_limit'])
                dq=rng.uniform(-2,2,29); world_w=rng.normal(size=3)
                state['qpos'][env,root_q:root_q+3]=[0,0,3]
                state['qpos'][env,root_q+3:root_q+7]=quat
                state['qpos'][env,backend.robot_qpos_adr]=q
                state['qvel'][env,root_v:root_v+6]=np.r_[np.zeros(3),world_w]
                state['qvel'][env,backend.robot_dof_adr]=dq
                native.qpos[:]=state['qpos'][env];native.qvel[:]=state['qvel'][env]
                mujoco.mj_forward(backend.sim.mj_model,native)
                rot=native.xmat[backend.pelvis_body_id].reshape(3,3)
                velocity=np.empty(6);mujoco.mj_objectVelocity(backend.sim.mj_model,native,mujoco.mjtObj.mjOBJ_BODY,backend.pelvis_body_id,velocity,0)
                action=rng.uniform(-5,5,29).astype(np.float32);backend.robot_history[env].last_action[:]=action
                fields={'pos':q-ref['obs_default'],'vel':dq,'gravity':rot.T@np.array([0,0,-1]),'angvel':.25*rot.T@velocity[:3],'action':action}
                values=[]
                for name in ('pos','vel','gravity','angvel','action'): values.extend(fields[name])
                for name in ('action','angvel','pos','vel','gravity'):
                    history=histories[env][name];history.insert(0,fields[name].copy());del history[4:]
                    values.extend(np.concatenate(history+[np.zeros_like(fields[name])]*(4-len(history))))
                expected.append(np.r_[values,backend.policy_z].astype(np.float32))
            backend.set_state(state)
            backend.physics_step_counter=0
            backend._update_controller()
            np.testing.assert_allclose(backend.last_obs,np.asarray(expected),atol=1e-5,rtol=1e-4)
            maximum=max(maximum,float(np.max(np.abs(backend.last_obs-expected))))
            count+=backend.num_envs
        for i,name in enumerate(ref['names']):
            assert int(backend.sim.mj_model.joint(name).qposadr[0])==backend.robot_qpos_adr[i]
            assert int(backend.sim.mj_model.joint(name).dofadr[0])==backend.robot_dof_adr[i]
        return {'samples':count,'field_max_abs_error':maximum,'all_29_named_mapping':True,'history_field_major_newest_first':True,'execution':'real CUDA forward and adapter; independent native kinematics'}
    finally:
        backend.policy,backend.policy_z,backend.policy_task,backend.controller=saved
        backend.reset()

def interface_probe(backend):
    """Point/frame/guard isolation and paused controls against the actual CUDA state."""
    import mujoco
    ids=np.array([0]); saved=backend.get_state(); model=backend.sim.mj_model
    body=backend.pelvis_body_id
    rotation=backend.sim.data.xmat._tensor[0,body].cpu().numpy().reshape(3,3)
    origin=backend.sim.data.xpos._tensor[0,body].cpu().numpy()
    com=backend.sim.data.xipos._tensor[0,body].cpu().numpy()
    force=np.array([1.,2.,3.]);point=np.array([.1,-.2,.3])
    backend.apply_wrench('pelvis',force,point,env_ids=ids,frame='body')
    actual=backend.get_state()['xfrc_applied']
    expected_force=rotation@force;expected_torque=np.cross(origin+rotation@point-com,expected_force)
    np.testing.assert_allclose(actual[0,body,:3],expected_force,atol=1e-6)
    np.testing.assert_allclose(actual[0,body,3:],expected_torque,atol=1e-6)
    assert not np.count_nonzero(actual[1:])
    backend.apply_wrench('pelvis',(0,0,0),env_ids=ids)
    state=backend.get_state(ids);joint=int(model.body('pelvis').jntadr[0]);adr=int(model.jnt_qposadr[joint])
    state['qpos'][0,adr+2]=.2
    state['time'][0]=31
    backend.set_state(state,ids)
    done=backend.termination();assert done[0]['terminated'] and done[0]['truncated']
    assert not any(row['terminated'] or row['truncated'] for row in done[1:])
    backend.set_state(saved);backend.reset()
    # N and M are valid only while paused; N cannot advance policy/history.
    backend.paused=True
    counter=backend.policy_counter;history=[h.last_action.copy() for h in backend.robot_history]
    backend._key_callback(backend.KEY_N)
    assert backend.policy_counter==counter
    for h,before in zip(backend.robot_history,history):np.testing.assert_array_equal(h.last_action,before)
    step=backend.physics_step_counter;backend.paused=False;backend._key_callback(backend.KEY_N);backend._key_callback(backend.KEY_M)
    assert backend.physics_step_counter==step
    backend.visual_replay=True;backend.paused=True
    for key in (backend.KEY_N,backend.KEY_M,backend.KEY_R,backend.KEY_F,backend.KEY_E):backend._key_callback(key)
    assert backend.physics_step_counter==step
    backend.visual_replay=False;backend.reset()
    if backend.scene == 'rider':
        # No diagnostics between entering and leaving proximity: physics owns the latch.
        state=backend.get_state(ids); deck_joint=int(model.joint('deck_free').qposadr[0])
        state['qpos'][0,adr:adr+2]=state['qpos'][0,deck_joint:deck_joint+2]+np.array([0.,.3])
        backend.set_state(state,ids);backend.step()
        state=backend.get_state(ids);state['qpos'][0,adr:adr+2]=state['qpos'][0,deck_joint:deck_joint+2]+np.array([0.,2.])
        backend.set_state(state,ids);assert backend.termination()[0]['termination_reason']=='offboard'
        backend.reset();state=backend.get_state(ids);state['qpos'][0,deck_joint]+=6.
        backend.set_state(state,ids);assert backend.termination()[0]['termination_reason']=='out_of_bounds'
        backend.reset()
    return {'body_frame_point_torque':True,'unselected_force_isolation':True,'single_env_fall_timeout':True,'paused_N_no_policy_history':True,'unpaused_N_M_ignored':True,'visual_replay_keys_readonly':True,'offboard_physics_latch_and_deck_bounds':backend.scene=='rider'}
