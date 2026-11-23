"""Native29 static-start/end-grasp validation with official Robotiq2F-85."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from dataclasses import asdict
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from skate_sim.backends.mujoco import MujocoBackend
from skate_sim.models.scene import normalize_scene_config
from skate_sim.record import Recorder, numerics_manifest


def backend(fixture=None, gripper="none"):
    with contextlib.redirect_stdout(io.StringIO()):
        return MujocoBackend(scene_config=normalize_scene_config({'gripper':gripper}), controller='pose',
            episode_time_limit=None, out_of_bounds_distance=None, offboard_distance=None,
            component_root_fixture=fixture)


def body_geoms(b, root):
    bid = b.model.body(root).id
    result = []
    for gid in range(b.model.ngeom):
        i = int(b.model.geom_bodyid[gid])
        while i and i != bid:
            i = int(b.model.body_parentid[i])
        if i == bid and (b.model.geom_contype[gid] or b.model.geom_conaffinity[gid]):
            result.append(gid)
    return result


def point(b, name, offset):
    bid = b.model.body(name).id
    return b.data.xpos[bid] + b.data.xmat[bid].reshape(3, 3) @ np.asarray(offset)


def ik(b, qpos, names, targets, regularization=.003):
    """Offline kinematic pose/conditional reach query only; never integrates."""
    data = mujoco.MjData(b.model)
    indices = np.array([b.model.joint(name).qposadr[0] for name in names], int)
    joints = [b.model.joint(name).id for name in names]
    bounds = b.model.jnt_range[joints].T
    initial = np.clip(qpos[indices], bounds[0]+1e-6, bounds[1]-1e-6)
    def residual(q):
        data.qpos[:] = qpos
        data.qpos[indices] = q
        mujoco.mj_forward(b.model, data)
        values = []
        for entry in targets:
            body, offset, goal = entry[:3]
            i = b.model.body(body).id
            p = data.xpos[i] + data.xmat[i].reshape(3, 3) @ offset
            values.extend(p - goal)
            if len(entry) == 4:
                from scipy.spatial.transform import Rotation
                values.extend(.15*Rotation.from_matrix(entry[3].T@data.xmat[i].reshape(3,3)).as_rotvec())
        return np.r_[values, regularization*(q-initial)]
    opt = least_squares(residual, initial, bounds=bounds, max_nfev=120,
                        ftol=1e-8, xtol=1e-8, gtol=1e-8)
    result = qpos.copy()
    result[indices] = opt.x
    return result, float(np.linalg.norm(residual(opt.x)[:3]))


def static_pd_balance(b, support_sides=("left","right"), place_feet=True, payload=None):
    """Solve a constrained foot-force equilibrium and realize joint tau via PD offset."""
    from scipy.optimize import minimize, LinearConstraint, Bounds
    from scipy.spatial import ConvexHull
    feet=[g for side in support_sides for g in body_geoms(b,side+'_ankle_roll_link')]
    gaps=np.array([b.data.geom_xpos[g,2]-b.model.geom_size[g,0] for g in feet])
    initial_gaps=gaps.copy()
    # Initial placement only: both horizontal sole pads just touch the plane.
    if place_feet:
        b.data.joint('floating_base_joint').qpos[2]-=float(gaps.min())
    b.data.qvel[:]=0;mujoco.mj_forward(b.model,b.data)
    root=b.model.body('pelvis').id;bodyids=[]
    for i in range(b.model.nbody):
        parent=i
        while parent and parent!=root:parent=int(b.model.body_parentid[parent])
        if parent==root:bodyids.append(i)
    mass=float(b.model.body_mass[bodyids].sum())
    com=np.average(b.data.xipos[bodyids],axis=0,weights=b.model.body_mass[bodyids])
    contacts=np.array([b.data.geom_xpos[g]-np.array([0,0,b.model.geom_size[g,0]]) for g in feet])
    hull=ConvexHull(contacts[:,:2]);margin=float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
    rows=np.r_[np.arange(6),b.robot_dof_adr]
    matrix=np.zeros((len(rows),3*len(feet)))
    for i,(g,pos) in enumerate(zip(feet,contacts)):
        jac=np.zeros((3,b.model.nv));rotjac=np.zeros_like(jac)
        mujoco.mj_jac(b.model,b.data,jac,rotjac,pos,int(b.model.geom_bodyid[g]))
        matrix[:,3*i:3*i+3]=jac[:,rows].T
    bias=b.data.qfrc_bias[rows].copy()
    support_mass=mass
    loaded_com=com.copy()
    if payload is not None:
        payload_mass=float(payload['mass_kg']);payload_com=np.asarray(payload['com_world_m'])
        payload_J=np.zeros((3,b.model.nv));payload_rot=np.zeros_like(payload_J)
        mujoco.mj_jac(b.model,b.data,payload_J,payload_rot,payload_com,int(payload['attached_body']))
        bias+=(payload_J.T@np.array([0.,0.,payload_mass*9.81]))[rows]
        support_mass+=payload_mass;loaded_com=(mass*com+payload_mass*payload_com)/support_mass
        margin=float(np.min(-(hull.equations[:,:2]@loaded_com[:2]+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
    limit=b.policy_values['joint_effort_limit']
    friction=[]
    for i,g in enumerate(feet):
        mu=float(min(b.model.geom_friction[g,0],b.model.geom_friction[b.model.geom('ground').id,0]))
        for axis in [0,1]:
            for sign in [-1,1]:
                row=np.zeros(matrix.shape[1]);row[3*i+axis]=sign;row[3*i+2]=-mu;friction.append(row)
    start=np.tile([0.,0.,support_mass*9.81/len(feet)],len(feet))
    kp=b.policy_values['kp'];current=b.data.qpos[b.robot_qpos_adr]
    tau_lower=np.maximum(-limit,kp*(b.policy_values['joint_pos_lower_limit']-current))
    tau_upper=np.minimum(limit,kp*(b.policy_values['joint_pos_upper_limit']-current))
    constraints=[LinearConstraint(matrix[:6],bias[:6],bias[:6]),
                 LinearConstraint(np.array(friction),-np.inf,0),
                 LinearConstraint(matrix[6:],bias[6:]-tau_upper,bias[6:]-tau_lower)]
    lower=np.tile([-np.inf,-np.inf,0.],len(feet))
    opt=minimize(lambda f:float(np.sum((f-start)**2)*1e-3+np.sum((bias[6:]-matrix[6:]@f)**2)*1e-4),
                 start,method='SLSQP',bounds=Bounds(lower,np.inf),constraints=constraints,
                 options=dict(maxiter=500,ftol=1e-10))
    tau=bias[6:]-matrix[6:]@opt.x
    target=b.data.qpos[b.robot_qpos_adr]+tau/b.policy_values['kp']
    valid=bool(opt.success and np.linalg.norm(matrix[:6]@opt.x-bias[:6])<1e-6 and
               np.all(abs(tau)<=limit+1e-7) and np.all(target>=b.policy_values['joint_pos_lower_limit']) and
               np.all(target<=b.policy_values['joint_pos_upper_limit']))
    report=dict(feasible=valid,solver_message=str(opt.message),initial_foot_sphere_gaps_m=initial_gaps.tolist(),
        root_initial_lowering_m=float(initial_gaps.min()) if place_feet else 0.,robot_mass_kg=mass,com_world_m=com.tolist(),
        support_polygon_xy_m=contacts[hull.vertices,:2].tolist(),signed_support_margin_m=margin,
        loaded_support_mass_kg=support_mass,loaded_com_world_m=loaded_com.tolist(),payload=payload,
        base_equilibrium_residual=float(np.linalg.norm(matrix[:6]@opt.x-bias[:6])),
        foot_forces_world_N=opt.x.reshape(-1,3).tolist(),joint_tau_static_Nm=tau.tolist(),
        pd_target_offset_rad=(tau/b.policy_values['kp']).tolist(),
        method='Static foot friction-pyramid constrained equilibrium; joint torque only via existing limited PD target offset. No base/board force; no production gains changed.')
    return target,report


def run_robotiq_opening(out):
    from skate_sim.models.gripper import measured_opening,manifest,compare_mechanics,tcp_pose
    scene=normalize_scene_config(dict(gripper='right',robot_pose=dict(pos=[-.12,-.35,.793],quat=[1,0,0,0])))
    with contextlib.redirect_stdout(io.StringIO()):
        b=MujocoBackend(scene_config=scene,controller='pose',episode_time_limit=None,
                       out_of_bounds_distance=None,offboard_distance=None)
    target,balance=static_pd_balance(b)
    if not balance['feasible']:raise RuntimeError('Robotiq loaded static equilibrium infeasible')
    b.pose_target=target.copy();out.mkdir(parents=True,exist_ok=True)
    recorder=Recorder(str(out),backend='mujoco',scene='rider',seed=0,
        metadata=dict(numerics=numerics_manifest(b),gripper_model=manifest(),free_root=True,
                      task='Robotiq raw-control opening only, no board grasp'))
    kwargs=dict(scene_config=scene,controller='pose',episode_time_limit=None,
                out_of_bounds_distance=None,offboard_distance=None,config=asdict(b.config))
    recorder.start(b.get_state(),backend_kwargs=kwargs,runtime_state=b.export_replay_state());b.recorder=recorder
    rows=[];positions=[];v=np.zeros(6);force=np.zeros(6)
    feet=[set(body_geoms(b,s+'_ankle_roll_link')) for s in ['left','right']]
    ground=b.model.geom('ground').id
    root=b.model.body('pelvis').id
    try:
        for k in range(1200):
            t=k*.002;cmd=0. if t<.4 or t>=1.5 else 128. if t<.9 else 255.
            b.set_gripper_command(cmd);b.step(update_policy=False)
            assert np.isfinite(b.data.qpos).all() and np.isfinite(b.data.qvel).all()
            assert not np.any(b.data.xfrc_applied) and not np.any(b.data.qfrc_applied[:6])
            mujoco.mj_objectVelocity(b.model,b.data,mujoco.mjtObj.mjOBJ_BODY,root,v,0)
            counts=[0,0]
            for i in range(b.data.ncon):
                c=b.data.contact[i];pair={int(c.geom1),int(c.geom2)}
                if ground not in pair:continue
                mujoco.mj_contactForce(b.model,b.data,i,force)
                if force[0]<=1e-5:continue
                for j,g in enumerate(feet):counts[j]+=bool(pair&g)
            rows.append(dict(time=float(b.data.time),raw_command=cmd,opening_m=measured_opening(b.model,b.data,'right'),
                driver_angles_rad=[float(b.data.joint('right_2f85_'+s+'_driver_joint').qpos[0]) for s in ['left','right']],
                tendon_actuator_output=float(b.data.actuator_force[b.model.actuator('right_2f85_fingers_actuator').id]),
                root_z_m=float(b.data.xpos[root,2]),root_tilt_rad=float(np.arccos(np.clip(b.data.xmat[root].reshape(3,3)[2,2],-1,1))),
                root_linear_m_s=float(np.linalg.norm(v[3:])),root_angular_rad_s=float(np.linalg.norm(v[:3])),
                foot_ground_contacts=counts))
            positions.append(np.array([b.data.xpos[b.model.body(s+'_ankle_roll_link').id,:2] for s in ['left','right']]))
        tail=rows[-125:];feetpos=np.array(positions[-125:]);slip=float(np.linalg.norm(feetpos-feetpos[0],axis=-1).max())
        phases={label:min(rows,key=lambda x:abs(x['time']-t)) for label,t in [('open',.38),('middle',.88),('closed',1.48),('reopened',2.38)]}
        report=dict(gripper=manifest(),mechanics=compare_mechanics(b.model,'right'),static_balance=balance,
            phase_samples=phases,finite=True,free_root=True,simulated_seconds=2.4,
            late_window=dict(max_root_linear_m_s=max(x['root_linear_m_s'] for x in tail),
                max_root_angular_rad_s=max(x['root_angular_rad_s'] for x in tail),
                max_root_tilt_rad=max(x['root_tilt_rad'] for x in tail),foot_slip_m=slip,
                min_foot_contact_counts=[min(x['foot_ground_contacts'][i] for x in tail) for i in range(2)]),
            task='No board grasp/foot action; actual loaded free robot opening test; not certified hardware validation')
        (out/'metrics.json').write_text(json.dumps(rows)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(samples={k:(x['raw_command'],x['opening_m']) for k,x in phases.items()},late=report['late_window'])),flush=True)
        return report
    finally:recorder.close();b.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',choices=['opening','sequence'],default='opening')
    args=parser.parse_args()
    if args.suite=='opening':
        run_robotiq_opening(args.output)
    else:
        for name,press,close in [('active_end_grasp',True,True),('press_open_control',True,False),('raised_open_control',False,False)]:
            run_static_sequence(args.output,name,press,close,analytic_shift=True)




def run_static_sequence(out, name, press=True, close=True, analytic_shift=False):
    """One prescribed continuous static-start end-grasp attempt; no state teleport."""
    import hashlib
    from skate_sim.models.gripper import pad_tcp, manifest
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    scene=normalize_scene_config(dict(gripper='right',robot_pose=dict(pos=[-.12,-.35,.793],quat=[1,0,0,0])))
    with contextlib.redirect_stdout(io.StringIO()):
        b=MujocoBackend(scene_config=scene,controller='pose',episode_time_limit=None,
                       out_of_bounds_distance=None,offboard_distance=None)
    initial_target,balance=static_pd_balance(b)
    if not balance['feasible']:raise RuntimeError('Initial equilibrium not feasible')
    b.pose_target=initial_target.copy()
    reference=b.data.qpos.copy();base_z=float(reference[2]);path=out/name
    plan=dict(prefix=1.2,shift_end=1.8,lift_end=2.4,press_end=2.7,approach_end=3.3,
              insert_end=3.9,close_end=4.2,retract_end=4.8,hold_end=5.8,
              root_reference_y_end=-.44,root_reference_z_end=.70,
              nose_material_local_x=.365,closing_position_tolerance_m=.03,closing_normal_degrees=15.)
    r=Recorder(str(path),backend='mujoco',scene='rider',seed=0,
        metadata=dict(numerics=numerics_manifest(b),free_root=True,board_initially_grounded=True,
                      task='static-start tail press to opposite end grasp',gripper_model=manifest(),plan=plan,press=press,close=close))
    kwargs=dict(scene_config=scene,controller='pose',episode_time_limit=None,
                out_of_bounds_distance=None,offboard_distance=None,config=asdict(b.config))
    r.start(b.get_state(),backend_kwargs=kwargs,runtime_state=b.export_replay_state());b.recorder=r
    ankle_goals={side:point(b,side+'_ankle_roll_link',[0,0,0]) for side in ['left','right']}
    shift_y=-.44;shift_audit=None
    if analytic_shift:
        try:
            shift_y,shift_audit=analytic_support_shift(b,reference,ankle_goals)
        except Exception:
            r.close();b.close();raise
        plan['root_reference_y_end']=shift_y
        plan['unload_timeout_seconds']=.6
        plan['crouch_during_shift']=True
    unloaded_at=None;unload_good=0;actual_unload=None
    leg_seed=reference.copy();arm_seed=reference.copy();board=set(body_geoms(b,'chassis'))
    deck_surface=set(body_geoms(b,'deck'));foot=set(body_geoms(b,'left_ankle_roll_link'))
    rightfoot=set(body_geoms(b,'right_ankle_roll_link'));ground=b.model.geom('ground').id
    pads=[{b.model.geom('right_2f85_'+side+'_pad'+str(i)).id for i in [1,2]} for side in ['left','right']]
    force=np.empty(6);rows=[];qs=[];vs=[];taus=[];prefix=hashlib.sha256()
    stopped='completed prescribed attempt';last_phase='static_prefix';last_static=None;captured=None
    hand_goal=None;hand_rot=None;ik_residual=None
    def ramp(x):
        u=float(np.clip(x,0,1));return u*u*(3-2*u)
    try:
        for k in range(3200 if analytic_shift else 2900):
            physical_time=k*.002
            t=physical_time
            if analytic_shift and physical_time>=1.8:
                if unloaded_at is None:
                    last=rows[-1]
                    xy=np.array(shift_audit['right_foot_polygon_xy'])
                    from scipy.spatial import ConvexHull
                    hull=ConvexHull(xy);com=robot_com(b.model,b.data)
                    actual_margin=float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
                    mg=balance['robot_mass_kg']*9.81
                    ready=last['left_foot_support_N']<.05*mg and last['right_foot_support_N']>.8*mg and actual_margin>=.01 and last['root_tilt_rad']<.1
                    unload_good=unload_good+1 if ready else 0
                    if unload_good>=25:
                        unloaded_at=physical_time
                        actual_unload=dict(time=physical_time,left_normal_N=last['left_foot_support_N'],right_normal_N=last['right_foot_support_N'],actual_com_margin_m=actual_margin)
                    elif physical_time>=2.4:
                        stopped='actual left foot did not unload within prescribed support-transfer window';break
                    t=1.8
                if unloaded_at is not None:t=physical_time-(unloaded_at-1.8)
            if t>=5.8:break
            if t<1.2:
                last_phase='static_prefix';b.pose_target=initial_target.copy()
                b.set_gripper_command(0.)
            elif k%10==0:
                ref=reference.copy()
                ref[1]=-.35+(shift_y+.35)*ramp((t-1.2)/.6)
                ref[2]=base_z+(.70-base_z)*ramp((t-(1.2 if analytic_shift else 1.8))/.6)
                ref[b.robot_qpos_adr]=leg_seed[b.robot_qpos_adr]
                left=ankle_goals['left'].copy()
                if t>=1.8:
                    u=ramp((t-1.8)/.6)
                    toe=(1-u)*(ankle_goals['left']+np.array([.12,0,-.03]))+u*np.array([-.375,.01,.24])
                    if t>=2.4 and press:toe[2]=.24-.15*ramp((t-2.4)/.3)
                    left=toe-np.array([.12,0,-.03])
                targets=[('left_ankle_roll_link',np.zeros(3),left,np.eye(3)),
                         ('right_ankle_roll_link',np.zeros(3),ankle_goals['right'],np.eye(3))]
                ref,_=ik(b,ref,b.robot_joint_names[:12],targets,regularization=.0001)
                leg_seed=ref.copy()
                last_phase='shift' if t<1.8 else 'lift_to_tail' if t<2.4 else 'press_tail' if t<2.7 else 'approach_end' if t<3.3 else 'insert_end' if t<3.9 else 'close_end' if t<4.2 else 'retract' if t<4.8 else 'hold'
                if analytic_shift and unloaded_at is None and physical_time>=1.8:
                    last_phase='support_transfer_wait'
                if t>=2.7:
                    deck_id=b.model.body('deck').id;dr=b.data.xmat[deck_id].reshape(3,3)
                    slope=.39
                    inward=np.array([-1.,0,-slope]);inward/=np.linalg.norm(inward)
                    normal=np.array([-slope,0,1.]);normal/=np.linalg.norm(normal)
                    yaxis=np.cross(normal,inward)
                    hand_rot=dr@np.column_stack([np.cross(normal,inward),normal,inward])
                    material=np.array([.365,0,.03*.65**2])
                    outside=.08*(1-ramp((t-3.3)/.6))
                    hand_goal=b.data.xpos[deck_id]+dr@material-outside*(dr@inward)
                    if t>=4.2:
                        if captured is None:captured=(hand_goal.copy(),hand_rot.copy())
                        hand_goal=captured[0]+ramp((t-4.2)/.6)*np.array([-.10,-.05,.20]);hand_rot=captured[1]
                    ref[b.robot_qpos_adr[22:29]]=arm_seed[b.robot_qpos_adr[22:29]]
                    base=b.model.body('right_2f85_base').id
                    actual_tcp,_=pad_tcp(b.model,b.data,'right')
                    local_tcp=b.data.xmat[base].reshape(3,3).T@(actual_tcp-b.data.xpos[base])
                    ref,ik_residual=ik(b,ref,b.robot_joint_names[22:29],
                        [('right_2f85_base',local_tcp,hand_goal,hand_rot)],regularization=.0001)
                    arm_seed=ref.copy()
                data=mujoco.MjData(b.model);data.qpos[:]=ref;mujoco.mj_forward(b.model,data)
                sandbox=SimpleNamespace(model=b.model,data=data,robot_qpos_adr=b.robot_qpos_adr,
                    robot_dof_adr=b.robot_dof_adr,policy_values=b.policy_values)
                target,last_static=static_pd_balance(sandbox,support_sides=('right',) if t>=1.86 else ('left','right'),place_feet=False)
                if not last_static['feasible']:
                    stopped='reference single-support static equilibrium infeasible';break
                b.pose_target=target
                tcp,_=pad_tcp(b.model,b.data,'right')
                _,gr=pad_tcp(b.model,b.data,'right')
                distance=np.linalg.norm(tcp-hand_goal) if hand_goal is not None else np.inf
                angle=np.degrees(np.arccos(np.clip(abs(gr[:,1]@hand_rot[:,1]),-1,1))) if hand_rot is not None else np.inf
                eligible=t>=3.9 and distance<=.03 and angle<=15
                b.set_gripper_command(255. if close and eligible else 0.)
            b.step(update_policy=False)
            assert not np.any(b.data.xfrc_applied) and not np.any(b.data.qfrc_applied[:6])
            assert np.isfinite(b.data.qpos).all() and np.isfinite(b.data.qvel).all()
            pad_counts=[0,0];end_positions=[];tail_impulse=0.;other_support=[];support_force=0.;left_support_force=0.
            for i in range(b.data.ncon):
                c=b.data.contact[i];pair={int(c.geom1),int(c.geom2)}
                mujoco.mj_contactForce(b.model,b.data,i,force)
                if force[0]<=1e-5:continue
                if ground in pair and pair&rightfoot:support_force+=force[0]
                if ground in pair and pair&foot:left_support_force+=force[0]
                if pair&board and not pair<=board:
                    other=pair-board
                    for j,g in enumerate(pads):
                        if g & other:
                            pad_counts[j]+=1
                            deck_id=b.model.body('deck').id
                            end_positions.append((b.data.xmat[deck_id].reshape(3,3).T@(c.pos-b.data.xpos[deck_id])).tolist())
                    if not any(other & group for group in pads):other_support.extend(b.model.geom(g).name or str(g) for g in other)
                    if pair&deck_surface and other&foot:
                        local=b.data.xmat[b.model.body('deck').id].reshape(3,3).T@(c.pos-b.data.xpos[b.model.body('deck').id])
                        if local[0]<=-.30:tail_impulse+=force[0]*.002
            root=b.model.body('pelvis').id;tilt=float(np.arccos(np.clip(b.data.xmat[root].reshape(3,3)[2,2],-1,1)))
            tcp,_=pad_tcp(b.model,b.data,'right');actual_distance=float(np.linalg.norm(tcp-hand_goal)) if hand_goal is not None else None
            rows.append(dict(time=float(b.data.time),phase=last_phase,root_z=float(b.data.xpos[root,2]),
                root_tilt_rad=tilt,right_foot_support_N=support_force,left_foot_support_N=left_support_force,tail_impulse_Ns=tail_impulse,
                nose_z=float(point(b,'deck',[.4,0,.03])[2]),pad_counts=pad_counts,
                pad_positions_deck_m=end_positions,other_board_support=other_support,
                actual_tcp_error_m=actual_distance,kinematic_ik_error_m=ik_residual,
                raw_gripper_command=b.gripper_command['right'],tau_saturation_count=int(np.sum(abs(b.last_tau)>=b.policy_values['joint_effort_limit']-1e-8))))
            qs.append(b.data.qpos.copy());vs.append(b.data.qvel.copy());taus.append(b.last_tau.copy())
            if k<600:prefix.update(b.data.qpos.tobytes()+b.data.qvel.tobytes())
            if t>=1.2 and (tilt>.25 or b.data.xpos[root,2]<.62):
                stopped='actual free robot lost upright phase gate';break
        report=dict(name=name,free_root=True,plan=plan,static_initial_balance=balance,
                    analytic_shift=shift_audit,actual_unload=actual_unload,
                    final_phase=last_phase,stop_reason=stopped,simulated_seconds=float(b.data.time),
                    prefix_600_ticks_sha256=prefix.hexdigest(),prefix_completed=len(qs)>=600,
                    tail_normal_impulse_Ns=sum(x['tail_impulse_Ns'] for x in rows),
                    nose_max_z_m=max(x['nose_z'] for x in rows),
                    both_pad_contact_steps=sum(all(x['pad_counts']) for x in rows),
                    actual_endpoint_grasp=False,whole_board_held_1s=False,
                    last_reference_static=last_static,
                    note='Only initialization writes actual qpos; leg/root references and arm IK use separate MjData, execution solely finite29PD plus aperture. No board/base forces or welds; official Robotiq tendon ctrl is independent.')
        np.savez_compressed(path/'states.npz',qpos=np.array(qs),qvel=np.array(vs),tau=np.array(taus),dt=.002)
        (path/'metrics.json').write_text(json.dumps(rows)+'\n');(path/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['static_initial_balance','last_reference_static']}),flush=True)
        return report
    finally:r.close();b.close()



def robot_com(model,data):
    root=model.body('pelvis').id;ids=[]
    for i in range(model.nbody):
        parent=i
        while parent and parent!=root:parent=int(model.body_parentid[parent])
        if parent==root:ids.append(i)
    return np.average(data.xipos[ids],axis=0,weights=model.body_mass[ids])


def analytic_support_shift(b,reference,ankle_goals):
    """At most two Newton corrections of a1D geometric COM equation, no trials."""
    from scipy.spatial import ConvexHull
    feet=body_geoms(b,'right_ankle_roll_link')
    xy=np.array([b.data.geom_xpos[g,:2] for g in feet]);hull=ConvexHull(xy)
    target=float(xy[hull.vertices,1].mean())
    def evaluate(y):
        q=reference.copy();q[1]=y;q[2]=.70
        q,_=ik(b,q,b.robot_joint_names[:12],[(s+'_ankle_roll_link',np.zeros(3),ankle_goals[s],np.eye(3)) for s in ['left','right']],regularization=.0001)
        data=mujoco.MjData(b.model);data.qpos[:]=q;mujoco.mj_forward(b.model,data)
        com=robot_com(b.model,data)
        margin=float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
        errors=[np.linalg.norm(data.xpos[b.model.body(s+'_ankle_roll_link').id]-ankle_goals[s]) for s in ['left','right']]
        return float(com[1]),margin,max(errors)
    y=-.44;steps=[]
    for _ in range(2):
        value,margin,error=evaluate(y)
        shifted,_,_=evaluate(y+.002);derivative=(shifted-value)/.002
        if abs(derivative)<.1:raise RuntimeError('COM/root lateral derivative too small')
        proposed=y+(target-value)/derivative
        steps.append(dict(root_y=y,com_y=value,target_y=target,derivative=derivative,proposed_root_y=proposed))
        y=proposed
        if abs(value-target)<1e-5:break
    value,margin,error=evaluate(y)
    if margin<.01 or error>.005:raise RuntimeError(f'Analytic shift geometry fails: root_y={y}, margin={margin}, foot_ik_error={error}, steps={steps}')
    return y,dict(method='1D finite-difference Newton, maximum2 corrections; fixed world feet; no simulation search',
        iterations=steps,root_y_target=y,target_support_y=target,reference_com_y=value,
        reference_root_z_m=.70,reference_support_margin_m=margin,feet_ik_max_error_m=error,right_foot_polygon_xy=xy[hull.vertices].tolist())


if __name__=='__main__':
    main()
