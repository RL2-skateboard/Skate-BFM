"""Explicit opt-in GPU physics fidelity probes for the original Robotiq asset.

This is not an RL environment benchmark: no QP, policy, reward or reset speed
is represented by the reported physics throughput.
"""
from __future__ import annotations

import argparse
import json
import time
import subprocess
import threading
from pathlib import Path

import mujoco
import numpy as np

from skate_sim.backends.mjlab import MjlabBackend
from skate_sim.models.gripper import compare_mechanics, measured_opening, prefix
from skate_sim.models.scene import build_scene_xml
from skate_sim.robot.bfm import DEPLOY_JOINTS


def native_forces(model, data, tau, raw, hooks=False):
    data.qfrc_applied[:] = 0
    ids = [int(model.joint(n).dofadr[0]) for n in DEPLOY_JOINTS]
    data.qfrc_applied[ids] = tau
    data.ctrl[:] = 0
    data.ctrl[model.actuator(prefix('right')+'fingers_actuator').id] = raw
    if hooks:
        from skate_sim.physics.dynamics import DynamicsConfig
        cfg = DynamicsConfig()
        for end in ('front', 'rear'):
            j = model.joint(end+'_truck_tilt')
            data.qfrc_applied[j.dofadr[0]] = -cfg.truck_k3*data.qpos[j.qposadr[0]]**3
            for side in ('left','right'):
                a = int(model.joint(f'{end}_{side}_spin').dofadr[0])
                data.qfrc_applied[a] = -cfg.bearing_torque*np.tanh(data.qvel[a]/.1)


def closure(model, data):
    residual=[]
    for i in range(model.neq):
        if int(model.eq_type[i]) != int(mujoco.mjtEq.mjEQ_CONNECT): continue
        a,b=int(model.eq_obj1id[i]),int(model.eq_obj2id[i])
        p=data.xpos[a]+data.xmat[a].reshape(3,3)@model.eq_data[i,:3]
        q=data.xpos[b]+data.xmat[b].reshape(3,3)@model.eq_data[i,3:6]
        residual.append(float(np.linalg.norm(p-q)))
    return residual


def contact_arrays(model,pairs,normal):
    def label(g):
        n=model.geom(int(g)).name or ''
        if n=='ground':return 'ground'
        if '_pad' in n and '_2f85_' in n:return 'left_pad' if 'left_pad' in n else 'right_pad'
        if n in ('front_left_collision','front_right_collision','rear_left_collision','rear_right_collision'):return 'wheel'
        body=int(model.geom_bodyid[g]);names=[]
        while body:
            names.append(model.body(body).name);body=int(model.body_parentid[body])
        if any('ankle_roll_link' in n for n in names):return 'foot'
        if 'chassis' in names:return 'board'
        return 'other'
    result={k:0. for k in ('wheel_ground','foot_ground','toe_board','pad_board','left_pad_board','right_pad_board')}
    for geoms,force in zip(pairs,normal):
        pair={label(int(geoms[0])),label(int(geoms[1]))}
        key=next((k for p,k in (({'wheel','ground'},'wheel_ground'),({'foot','ground'},'foot_ground'),
                 ({'foot','board'},'toe_board'),({'left_pad','board'},'left_pad_board'),
                 ({'right_pad','board'},'right_pad_board')) if pair==p),None)
        if key:
            result[key]+=max(0.,float(force))
    result['pad_board']=result['left_pad_board']+result['right_pad_board']
    return result


def contact_groups(model,data):
    """Integrable normal forces from the engine's actual contact solution."""
    force=np.zeros(6);normal=[]
    for i in range(data.ncon):
        mujoco.mj_contactForce(model,data,i,force);normal.append(float(force[0]))
    return contact_arrays(model,data.contact.geom,normal)


def gates(report):
    """Frozen Astra engineering thresholds; 50 ticks is trend evidence only."""
    no=report['cases']['no_contact']['1']
    smooth=bool(no['root_position_m']<=.0002 and no['root_angle_rad']<=.002 and
        no['robot_q_max_rad']<=.002 and no['robot_dq_max_rad_s']<=.1 and no['torque_Nm']<=1e-3)
    contact={}
    coverage={k:False for k in ('wheel_ground','foot_ground','toe_board','left_pad_board','right_pad_board')}
    for name,rows in report['cases'].items():
        r=rows['10'];imp=r['normal_impulse_Ns']
        for k in coverage:coverage[k]|=imp['cpu'][k]>1e-7 and imp['gpu'][k]>1e-7
        checks=dict(state=r['root_position_m']<=.005 and r['root_angle_rad']<=np.deg2rad(2) and
            r['robot_q_rmse_rad']<=.03 and r['board_position_m']<=.005 and
            r['board_angle_rad']<=np.deg2rad(2) and r['opening_m']<=.003,
            closure=max(r['cpu_closure_m']+r['gpu_closure_m'])<=.002,
            driver_sync=max(r['cpu_driver_sync_rad'],r['gpu_driver_sync_rad'])<=.02,
            impulse=all(abs(imp['cpu'][k]-imp['gpu'][k])<=max(.05,.25*imp['cpu'][k]) for k in imp['cpu']),
            finite_root_overflow=all(x['finite'] and x['root_forces_zero'] and not x['overflow'] for x in rows.values()))
        contact[name]=dict(**checks,passed=all(checks.values()))
    return dict(smooth_single_step=smooth,contact=contact,semantic_coverage=coverage,
                contact_pass=all(c['passed'] for c in contact.values()) and all(coverage.values()))


def recorded_window(path, start_time):
    """Read only a selected 100 ms window, retaining its preceding true state."""
    previous=None;commands=[];state=None
    with Path(path).open() as f:
        for line in f:
            event=json.loads(line)
            if event['event']=='initial':previous=event['state']
            if event['event']!='tick':continue
            if event['command']['time_before']>=start_time-1e-8:
                if state is None:state=previous
                commands.append(event['command'])
                if len(commands)==50:break
            previous=event['state']
    if state is None or len(commands)<10:raise ValueError(f'{path}: selected window needs at least 10 recorded ticks')
    if len(state['qpos'])!=58 or len(state['qvel'])!=56:raise ValueError('window is not real native29+8hinge source')
    return state,commands


def errors(model, cpu, gpu, robot_q, robot_v):
    delta=np.zeros(model.nv)
    mujoco.mj_differentiatePos(model,delta,1.,cpu.qpos,gpu.qpos)
    deck=int(model.joint('deck_free').dofadr[0])
    return dict(root_position_m=float(np.linalg.norm(delta[:3])),root_angle_rad=float(np.linalg.norm(delta[3:6])),
        robot_q_max_rad=float(np.max(abs(cpu.qpos[robot_q]-gpu.qpos[robot_q]))),
        robot_q_rmse_rad=float(np.sqrt(np.mean((cpu.qpos[robot_q]-gpu.qpos[robot_q])**2))),
        robot_dq_max_rad_s=float(np.max(abs(cpu.qvel[robot_v]-gpu.qvel[robot_v]))),
        board_position_m=float(np.linalg.norm(delta[deck:deck+3])),board_angle_rad=float(np.linalg.norm(delta[deck+3:deck+6])),
        opening_m=abs(measured_opening(model,cpu,'right')-measured_opening(model,gpu,'right')),
        cpu_closure_m=closure(model,cpu),gpu_closure_m=closure(model,gpu),
        torque_Nm=float(np.max(abs(cpu.qfrc_applied[robot_v]-gpu.qfrc_applied[robot_v]))),
        cpu_driver_sync_rad=abs(float(cpu.joint(prefix('right')+'left_driver_joint').qpos[0]-cpu.joint(prefix('right')+'right_driver_joint').qpos[0])),
        gpu_driver_sync_rad=abs(float(gpu.joint(prefix('right')+'left_driver_joint').qpos[0]-gpu.joint(prefix('right')+'right_driver_joint').qpos[0])))


def gpu_collector_smoke(b, control_steps):
    """Untrained tiny GPU NN/PD collector; explicitly not BFM or task training."""
    import torch
    torch.manual_seed(0)
    m=b.sim.mj_model;device=b.sim.data.qpos.device;batch=b.num_envs
    d=b.sim.data;qpos=d.qpos._tensor;qvel=d.qvel._tensor
    rq=b.robot_qpos_index;rv=b.robot_dof_index
    boardq=int(m.joint('deck_free').qposadr[0]);boardv=int(m.joint('deck_free').dofadr[0])
    hinge=torch.as_tensor([m.joint(prefix('right')+name).qposadr[0] for name in
        ('right_driver_joint','right_coupler_joint','right_spring_link_joint','right_follower_joint',
         'left_driver_joint','left_coupler_joint','left_spring_link_joint','left_follower_joint')],device=device,dtype=torch.long)
    values={k:torch.as_tensor(v,device=device,dtype=qpos.dtype) for k,v in b.policy_values.items()}
    previous=torch.zeros((batch,30),device=device);episode=torch.zeros(batch,device=device,dtype=torch.long)
    horizons=20+torch.arange(batch,device=device)%5
    nominal_q=b._nominal_qpos.to(qpos);nominal_v=b._nominal_qvel.to(qvel)
    qref=nominal_q[:,rq].clone()
    def observe():return torch.cat((qpos[:,rq]-qref,qvel[:,rv],qpos[:,:7],qvel[:,:6],
        qpos[:,boardq:boardq+7],qvel[:,boardv:boardv+6],qpos[:,hinge],previous),-1)
    obs=observe();policy=torch.nn.Sequential(torch.nn.Linear(obs.shape[1],64),torch.nn.Tanh(),torch.nn.Linear(64,30)).to(device).eval()
    observations=torch.empty((control_steps,batch,obs.shape[1]),device=device)
    actions=torch.empty((control_steps,batch,30),device=device)
    rewards=torch.empty((control_steps,batch),device=device);done_buffer=torch.empty((control_steps,batch),device=device,dtype=torch.bool)
    reset_counts=torch.zeros((),device=device,dtype=torch.long)
    start=time.perf_counter()
    with torch.inference_mode(),b._wp.ScopedStream(b._wp.stream_from_torch(torch.cuda.current_stream())):
        for k in range(control_steps):
            obs=observe();output=policy(obs)
            action=output.tanh();target=(qref+.005*action[:,:29]).clamp(values['joint_pos_lower_limit'],values['joint_pos_upper_limit'])
            raw=(127.5*(action[:,29:]+1)).clamp(0,255)
            for _ in range(10):
                tau=(values['kp']*(target-qpos[:,rq])-values['kd']*qvel[:,rv]).clamp(-values['joint_effort_limit'],values['joint_effort_limit'])
                b._validation_torque_tick(tau,raw)
            episode+=1
            reward=-(qpos[:,rq]-qref).square().sum(-1)-1e-4*tau.square().sum(-1)
            done=(qpos[:,2]<.35)|~torch.isfinite(qpos).all(-1)|~torch.isfinite(qvel).all(-1)|(episode>=horizons)
            observations[k]=obs;actions[k]=action;rewards[k]=reward;done_buffer[k]=done
            previous.copy_(action)
            # A boolean GPU mask is accepted directly; no nonzero()/CPU lane list.
            b.sim.reset(env_ids=done)
            qpos.copy_(torch.where(done[:,None],nominal_q,qpos))
            qvel.copy_(torch.where(done[:,None],nominal_v,qvel))
            for field in ('ctrl','qfrc_applied','qacc_warmstart'):
                tensor=getattr(d,field)._tensor;tensor.masked_fill_(done[:,None],0)
            d.xfrc_applied._tensor.masked_fill_(done[:,None,None],0)
            d.time._tensor.masked_fill_(done,0)
            previous.masked_fill_(done[:,None],0);episode.masked_fill_(done,0);reset_counts+=done.sum()
            b.sim.forward()
    b._synchronize();elapsed=time.perf_counter()-start
    finite=bool(torch.isfinite(observations).all() and torch.isfinite(actions).all() and torch.isfinite(rewards).all())
    if not finite:raise RuntimeError('GPU collector smoke produced nonfinite data')
    return dict(kind='untrained_tiny_NN_GPU_collector_smoke_not_BFM_not_trained_policy',
        includes='GPU 122 features, 122x64x30 NN, limited29PD every2ms, raw tendon, 10physics ticks, simple tracking reward, done, masked reset and transition storage',
        excluded='teacher/QP, official BFM observations/rewards, policy optimization and native closed-loop reevaluation',
        transitions=batch*control_steps,seconds=elapsed,active_transitions_s=batch*control_steps/elapsed,
        finite=finite,resets=int(reset_counts.item()),observation_shape=list(observations.shape),all_transition_tensors_device=str(device),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def run(out, batch=2, profile_ticks=0, states=None, windows=(),smoke_steps=0):
    import torch
    import mujoco_warp as mjw
    baseline_card=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
    if profile_ticks:
        # Every capacity is a fresh process. Use measured whole-card increments,
        # not Torch-only allocations (which exclude most simulator memory).
        previous=[]
        for path in Path(out).parent.glob('b*.json'):
            try:
                r=json.loads(path.read_text())
                if r.get('profile') and r['batch']<batch:previous.append(r)
            except (OSError,ValueError,KeyError):pass
        if previous:
            r=max(previous,key=lambda r:r['batch'])
            base=r.get('baseline_card_used_bytes',1239*1024**2)
            increment=max(0,r['whole_card_peak_used_bytes']-base)
            estimated=baseline_card+increment*batch/r['batch']*1.15
            total=torch.cuda.mem_get_info()[1]
            if estimated>total-2*1024**3:
                raise RuntimeError(f'capacity preflight stops B{batch}: measured-memory estimate {estimated/1024**3:.2f} GiB would breach 2 GiB reserve')
    stop=threading.Event();card_samples=[]
    def sample_card():
        while not stop.is_set():
            try:
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                card_samples.append(int(value.splitlines()[0])*1024**2)
            except (OSError,ValueError,subprocess.SubprocessError):pass
            stop.wait(.1)
    monitor=threading.Thread(target=sample_card,daemon=True);monitor.start()
    cfg=dict(gripper='right')
    start=time.perf_counter()
    b=MjlabBackend(scene_config=cfg,num_envs=batch,_validate_gpu_gripper=True,
                   episode_time_limit=None,out_of_bounds_distance=None,offboard_distance=None)
    cold=time.perf_counter()-start
    model=b.sim.mj_model
    original=mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    option_fields=['timestep','integrator','cone','impratio','solver','iterations','tolerance','ls_iterations','ls_tolerance','ccd_iterations','enableflags','disableflags']
    report=dict(kind='physics_only_not_end_to_end',batch=batch,cold_init_s=cold,
        baseline_card_used_bytes=baseline_card,
        shape=dict(nq=model.nq,nv=model.nv,nu=model.nu,neq=model.neq,ntendon=model.ntendon),
        mechanics=compare_mechanics(model,'right'),
        option_differences={n:[float(getattr(original.opt,n)),float(getattr(model.opt,n))] for n in option_fields if getattr(original.opt,n)!=getattr(model.opt,n)},
        board_hooks='current composed native default off; separate explicitly paired hook probe',cases={})
    def download(lane=0):
        b._synchronize()
        d=mujoco.MjData(model)
        for field in ('qpos','qvel','qfrc_applied','xfrc_applied','ctrl'):
            getattr(d,field)[:]=getattr(b.sim.data,field)._tensor[lane].cpu().numpy()
        # Position-level diagnostics must refer to the integrated state.
        mujoco.mj_kinematics(model,d)
        return d
    def gpu_contacts(lane=0):
        wp=b._wp;data=b.sim.wp_data
        count=int(data.nacon.numpy()[0])
        if count>data.naconmax:raise RuntimeError('GPU contacts overflowed')
        if not count:return contact_arrays(model,[],[])
        ids=wp.array(np.arange(count,dtype=np.int32),dtype=int,device=b.sim.wp_device)
        forces=wp.empty(count,dtype=wp.spatial_vector,device=b.sim.wp_device)
        mjw.contact_force(b.sim.wp_model,data,ids,False,forces)
        force=forces.numpy()[:,0];pairs=data.contact.geom.numpy()[:count]
        mask=data.contact.worldid.numpy()[:count]==lane
        return contact_arrays(model,pairs[mask],force[mask])
    def load(q,v):
        b.reset()
        b.set_state(dict(qpos=np.repeat(q[None],batch,axis=0),qvel=np.repeat(v[None],batch,axis=0)))
        # Both engines start with zero qacc warmstart. Integration qpos/qvel
        # is identical to float32 precision; collision recomputes independently.
        b.sim.data.qacc_warmstart._tensor.zero_()
        b._synchronize()
    zero=np.zeros((batch,29),np.float32)
    try:
        assert report['shape']==dict(nq=58,nv=56,nu=36,neq=5,ntendon=1)
        assert not report['mechanics']['property_differences']
        nominal=b._nominal_qpos[0].cpu().numpy().copy()
        velocity=np.zeros(model.nv)
        initial=[]
        air=nominal.copy();air[2]=4.;air[int(model.joint('deck_free').qposadr[0])+2]=3.
        initial.append(('no_contact',air,velocity.copy(),False,None))
        initial.append(('support',nominal.copy(),velocity.copy(),False,None))
        roll=nominal.copy();spin=velocity.copy()
        for end in ('front','rear'):
            roll[model.joint(end+'_truck_tilt').qposadr[0]]=.05
            for side in ('left','right'):spin[model.joint(f'{end}_{side}_spin').dofadr[0]]=5.
        initial.append(('rolling_tilt_hooks',roll,spin,True,None))
        if states:
            with np.load(states) as z:
                for k in sorted(set([0,len(z['qpos'])//2,len(z['qpos'])-1])):
                    initial.append((f'recorded_{k}',z['qpos'][k].copy(),z['qvel'][k].copy(),False,None))
            report['recorded_source']=str(states)
        report['windows']=[]
        for path,seconds in windows:
            state,commands=recorded_window(path,seconds)
            name=f'{Path(path).parent.name}_{seconds:.3f}'
            initial.append((name,np.array(state['qpos']),np.array(state['qvel']),False,commands))
            report['windows'].append(dict(name=name,source=str(path),start_time=seconds,
                recorded_ticks=len(commands),held_last_command_ticks=50-len(commands),
                control='same recorded body torque and raw grip per tick, hold last if source ends; zero warmstart both engines; not closed-loop replay'))
        for name,q,v,hooks,commands in initial:
            load(q,v)
            cpu=mujoco.MjData(model);cpu.qpos[:]=q;cpu.qvel[:]=v;mujoco.mj_forward(model,cpu)
            rows={};cpu_imp={k:0. for k in contact_groups(model,cpu)};gpu_imp=cpu_imp.copy()
            tau=np.linspace(-.5,.5,29)*.1*b.policy_values['joint_effort_limit']
            torque_batch=np.repeat(tau[None],batch,axis=0)
            for tick in range(1,51):
                raw=128.
                if commands:
                    cmd=commands[min(tick-1,len(commands)-1)];tau=np.asarray(cmd['actual_qfrc_body_Nm']);raw=cmd['gripper_command']['right']
                    torque_batch=np.repeat(tau[None],batch,axis=0)
                native_forces(model,cpu,tau,raw,hooks);mujoco.mj_step(model,cpu)
                b.validation_torque_step(torque_batch,np.full((batch,1),raw,np.float32),board_force_hooks=hooks)
                gpu=download()
                for k,f in contact_groups(model,cpu).items():cpu_imp[k]+=f*.002
                for k,f in gpu_contacts().items():gpu_imp[k]+=f*.002
                if tick in (1,10,50):
                    mujoco.mj_kinematics(model,cpu)
                    rows[str(tick)]=errors(model,cpu,gpu,b.robot_qpos_adr,b.robot_dof_adr)
                    rows[str(tick)]['finite']=bool(np.isfinite(gpu.qpos).all() and np.isfinite(gpu.qvel).all())
                    rows[str(tick)]['normal_impulse_Ns']=dict(cpu=cpu_imp.copy(),gpu=gpu_imp.copy())
                    rows[str(tick)]['overflow']=bool(b.sim.wp_data.nacon.numpy()[0]>b.sim.wp_data.naconmax or np.any(b.sim.wp_data.nefc.numpy()>b.sim.wp_data.njmax))
                    board_ids=[int(model.joint(end+'_truck_tilt').dofadr[0]) for end in ('front','rear')]
                    board_ids += [int(model.joint(f'{end}_{side}_spin').dofadr[0]) for end in ('front','rear') for side in ('left','right')]
                    rows[str(tick)]['board_hook_error_Nm']=float(np.max(abs(cpu.qfrc_applied[board_ids]-gpu.qfrc_applied[board_ids])))
                    rows[str(tick)]['root_forces_zero']=bool(not np.any(gpu.qfrc_applied[:6]) and not np.any(gpu.xfrc_applied) and not np.any(gpu.ctrl[:6]))
            report['cases'][name]=rows
        openings=[]
        open_lane_state=None
        for raw in (0.,128.,255.):
            load(air,velocity)
            for _ in range(150):b.validation_torque_step(zero,np.full((batch,1),raw,np.float32))
            gpu=download();openings.append(measured_opening(model,gpu,'right'))
            if raw==0.:open_lane_state=b.get_state()
        report['raw_commands']=[0,128,255];report['openings_m']=openings
        if batch>=2:
            load(air,velocity)
            raw=np.zeros((batch,1),np.float32);raw[1]=255.
            mixed=zero.copy();mixed[1,-1]=.1*float(b.policy_values['joint_effort_limit'][-1])
            for _ in range(150):b.validation_torque_step(mixed,raw)
            before=b.get_state();ctrl=b.sim.data.ctrl._tensor.cpu().numpy().copy()
            untouched=all(np.array_equal(open_lane_state[k][0],before[k][0]) for k in ('qpos','qvel','time'))
            b.reset(env_ids=[0]);after=b.get_state()
            report['lane_isolation']=dict(open_m=measured_opening(model,download(0),'right'),
                closed_m=measured_opening(model,download(1),'right'),
                unselected_exact=all(np.array_equal(before[k][1:],after[k][1:]) for k in ('qpos','qvel','time')),
                unselected_ctrl_exact=bool(np.array_equal(ctrl[1:],b.sim.data.ctrl._tensor.cpu().numpy()[1:])),
                lane0_unchanged_by_lane1_body_and_grip=untouched,
                selected_ctrl_zero=bool((b.sim.data.ctrl._tensor[0]==0).all()))
        if profile_ticks:
            if not gates(report)['smooth_single_step'] or not gates(report)['contact_pass']:
                raise RuntimeError('fidelity gate not passed; refusing batch capacity/throughput claim')
            load(nominal,velocity)
            raw=torch.zeros((batch,1),device='cuda');tau=torch.zeros((batch,29),device='cuda')
            b.validation_torque_step(tau,raw)
            for _ in range(10):b._validation_torque_tick(tau,raw)
            b._synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(3):
                start=time.perf_counter()
                for _ in range(profile_ticks):b._validation_torque_tick(tau,raw)
                b._synchronize();times.append(time.perf_counter()-start)
            report['profile']=dict(ticks=profile_ticks,seconds=times,physics_steps_s=[batch*profile_ticks/x for x in times],
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                warp_peak_used_bytes=b._wp.get_mempool_used_mem_high(b.sim.wp_device),
                warp_current_used_bytes=b._wp.get_mempool_used_mem_current(b.sim.wp_device),
                free_total_bytes=list(torch.cuda.mem_get_info()))
        if smoke_steps:
            if not gates(report)['contact_pass']:raise RuntimeError('contact gate failed; collector smoke refused')
            load(nominal,velocity)
            report['collector_smoke']=gpu_collector_smoke(b,smoke_steps)
        report['gates']=gates(report)
        report['status']='MEASURED_GATES_PENDING_REVIEW'
    finally:
        b.close()
        stop.set();monitor.join(timeout=2.)
        report['whole_card_peak_used_bytes']=max(card_samples,default=None)
        report['whole_card_samples']=len(card_samples)
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--profile-ticks',type=int,default=0);p.add_argument('--states',type=Path)
    p.add_argument('--window',nargs=2,action='append',metavar=('EVENTS_JSONL','START_SECONDS'),default=[])
    p.add_argument('--smoke-steps',type=int,default=0)
    a=p.parse_args()
    try:r=run(a.out,a.batch,a.profile_ticks,a.states,[(Path(p),float(t)) for p,t in a.window],a.smoke_steps)
    except Exception as e:
        if not a.out.exists():
            a.out.parent.mkdir(parents=True,exist_ok=True)
            a.out.write_text(json.dumps(dict(status='FAILED_OR_RESERVE_STOP',batch=a.batch,error=str(e)),indent=2)+'\n')
        raise
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
