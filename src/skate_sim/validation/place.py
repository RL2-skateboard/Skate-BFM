"""Explicit aligned local Robotiq holding/placement component, not ground pickup."""
from pathlib import Path
import contextlib,io,json,time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from skate_sim.backends.mujoco import MujocoBackend
from skate_sim.models.scene import normalize_scene_config
from skate_sim.models.gripper import ASSET,pad_tcp,measured_opening,manifest
from skate_sim.record import numerics_manifest,compiled_model_hash
from .tail_catch import ik,static_pd_balance,body_geoms
from .balance import ContactBalance,TorqueRecorder
from .toe_stages import ToeStages


def geom_bottom(model,data,g):
    typ=int(model.geom_type[g]);R=data.geom_xmat[g].reshape(3,3);size=model.geom_size[g];pos=data.geom_xpos[g]
    if typ==mujoco.mjtGeom.mjGEOM_MESH:
        mid=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid]);return float(np.min(model.mesh_vert[start:start+count]@R[2]+pos[2]))
    if typ==mujoco.mjtGeom.mjGEOM_SPHERE:return float(pos[2]-size[0])
    if typ==mujoco.mjtGeom.mjGEOM_BOX:return float(pos[2]-abs(R[2])@size)
    if typ==mujoco.mjtGeom.mjGEOM_CYLINDER:return float(pos[2]-size[0]*np.linalg.norm(R[2,:2])-size[1]*abs(R[2,2]))
    if typ==mujoco.mjtGeom.mjGEOM_CAPSULE:return float(pos[2]-size[0]-size[1]*abs(R[2,2]))
    raise RuntimeError(f'Unsupported collision support geometry{typ}')


def prepare(branch=1,ground_entry=False,seed_record=None):
    with contextlib.redirect_stdout(io.StringIO()):
        b=MujocoBackend(scene_config=normalize_scene_config(dict(gripper='right',robot_pose=dict(pos=[-.12,-.35,.793],quat=[1,0,0,0]))),controller='pose',episode_time_limit=None,out_of_bounds_distance=None,offboard_distance=None)
    static_pd_balance(b)
    p,R=pad_tcp(b.model,b.data,'right');base=b.model.body('right_2f85_base').id
    local=b.data.xmat[base].reshape(3,3).T@(p-b.data.xpos[base])
    slope=.39;boardR=np.column_stack(([0,0,1],[0,-1,0],[1,0,0]))
    normal=boardR@np.array([-slope,0,1.]);normal/=np.linalg.norm(normal)
    inward=boardR@np.array([-1.,0,-slope]);inward/=np.linalg.norm(inward)
    desiredR=np.column_stack([np.cross(branch*normal,inward),branch*normal,inward])
    seed_tcp=np.array([.10,-.75,.78 if ground_entry else .86])
    if seed_record is not None:
        from skate_sim.replay import read_record
        source=read_record(seed_record);initial=next(r for r in source if r['event']=='initial')
        b.set_state(initial['state']);b.import_replay_state(initial['runtime_state'])
        p,R=pad_tcp(b.model,b.data,'right');local=b.data.xmat[base].reshape(3,3).T@(p-b.data.xpos[base])
        minimum=min(geom_bottom(b.model,b.data,g) for g in body_geoms(b,'chassis'))
        seed_tcp=p-np.array([0,0,minimum]) if ground_entry else p.copy();desiredR=R.copy()
    q,error=ik(b,b.data.qpos.copy(),b.robot_joint_names[22:29],[(b.model.body(base).name,local,seed_tcp,desiredR)],regularization=.0001)
    b.data.qpos[:]=q;mujoco.mj_forward(b.model,b.data)
    # No-load official mechanism equilibrium, used only to initialize its8 joints.
    sm=mujoco.MjModel.from_xml_path(str(ASSET));sd=mujoco.MjData(sm);sd.ctrl[:]=140.
    for _ in range(1250):mujoco.mj_step(sm,sd)
    for j in range(sm.njnt):
        name=sm.joint(j).name;b.data.joint('right_2f85_'+name).qpos[:]=sd.joint(name).qpos
    b.data.qvel[:]=0;b.set_gripper_command(140.);mujoco.mj_forward(b.model,b.data)
    board=set(body_geoms(b,'chassis'));chassis=b.model.body('chassis').id;deck=b.model.body('deck').id
    bm=float(b.model.body_subtreemass[chassis]);wrist_audit=[]
    # One analytic margin correction; no posture/force/friction search.
    p,G=pad_tcp(b.model,b.data,'right')
    Bx=(-G[:,2]-slope*branch*G[:,1])/np.sqrt(1+slope*slope)
    Bz=(branch*G[:,1]-slope*G[:,2])/np.sqrt(1+slope*slope)
    Br=np.column_stack([Bx,np.cross(Bz,Bx),Bz])
    nose=np.array([.365,0,.03*.65**2])
    oldR=b.data.xmat[chassis].reshape(3,3);oldD=b.data.xmat[deck].reshape(3,3)
    localD=oldR.T@(b.data.xpos[deck]-b.data.xpos[chassis]);relative=oldR.T@oldD
    def place_board():
        nonlocal p,G,Br
        p,G=pad_tcp(b.model,b.data,'right')
        bx=(-G[:,2]-slope*branch*G[:,1])/np.sqrt(1+slope*slope)
        bz=(branch*G[:,1]-slope*G[:,2])/np.sqrt(1+slope*slope)
        Br=np.column_stack([bx,np.cross(bz,bx),bz]);cr=Br@relative.T
        pose=b.data.joint('deck_free').qpos;pose[:3]=p-Br@nose-cr@localD
        xyzw=Rotation.from_matrix(cr).as_quat();pose[3:]=xyzw[[3,0,1,2]]
        mujoco.mj_forward(b.model,b.data)
    place_board()
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull
    root=b.model.body('pelvis').id
    feet=[g for side in ['left','right'] for g in body_geoms(b,side+'_ankle_roll_link')]
    points=np.array([b.data.geom_xpos[g]-np.array([0,0,b.model.geom_size[g,0]]) for g in feet])
    hull=ConvexHull(points[:,:2]);rows=np.r_[np.arange(6),b.robot_dof_adr]
    footmap=np.zeros((35,3*len(feet)));JP=np.zeros((3,b.model.nv));JR=np.zeros_like(JP)
    for i,(g,point) in enumerate(zip(feet,points)):
        mujoco.mj_jac(b.model,b.data,JP,JR,point,int(b.model.geom_bodyid[g]));footmap[:,3*i:3*i+3]=JP[:,rows].T
    selected=np.r_[b.robot_qpos_adr[22:29],b.robot_qpos_adr[14]]
    jointids=np.r_[[b.model.joint(n).id for n in b.robot_joint_names[22:29]],b.model.joint('waist_pitch_joint').id]
    qseed=b.data.qpos[selected].copy()
    if seed_record is None:qseed[-1]=.15
    pad_ids=set(b.model.geom('right_2f85_'+side+'_pad'+str(j)).id for side in ['left','right'] for j in [1,2])
    nonpads=set(body_geoms(b,'pelvis'))-pad_ids
    # Conservative fixed candidate list covers the complete local insertion range.
    pairs=np.array([(ga,gb) for ga in sorted(nonpads) for gb in sorted(board)],dtype=int)
    pair_radii=b.model.geom_rbound[pairs[:,0]]+b.model.geom_rbound[pairs[:,1]]
    fromto=np.zeros(6);lastx=None;cache=None;evaluations=0
    mg=(float(b.model.body_subtreemass[root])+(0. if ground_entry else bm))*9.81
    def bottom(g):
        typ=int(b.model.geom_type[g]);R=b.data.geom_xmat[g].reshape(3,3);size=b.model.geom_size[g];pos=b.data.geom_xpos[g]
        if typ==mujoco.mjtGeom.mjGEOM_MESH:
            mid=int(b.model.geom_dataid[g]);start=int(b.model.mesh_vertadr[mid]);count=int(b.model.mesh_vertnum[mid]);return float(np.min(b.model.mesh_vert[start:start+count]@R[2]+pos[2]))
        if typ==mujoco.mjtGeom.mjGEOM_SPHERE:return float(pos[2]-size[0])
        if typ==mujoco.mjtGeom.mjGEOM_BOX:return float(pos[2]-abs(R[2])@size)
        if typ==mujoco.mjtGeom.mjGEOM_CYLINDER:return float(pos[2]-size[0]*np.linalg.norm(R[2,:2])-size[1]*abs(R[2,2]))
        if typ==mujoco.mjtGeom.mjGEOM_CAPSULE:return float(pos[2]-size[0]-size[1]*abs(R[2,2]))
        raise RuntimeError(f'Unsupported collision support geometry{typ}')
    def evaluate(x):
        nonlocal lastx,cache,evaluations
        if lastx is not None and np.array_equal(lastx,x):return cache
        evaluations+=1;b.data.qpos[selected]=x[:8];nose[0]=x[8];mujoco.mj_forward(b.model,b.data);place_board()
        mujoco.mj_jac(b.model,b.data,JP,JR,b.data.subtree_com[chassis],base)
        ground_wrench_residual=np.zeros(3);ground_force=np.zeros(3);hand_force=np.array([0.,0.,bm*9.81])
        if ground_entry:
            # Ground-supported pregrasp is a TWO-body balance, not an airborne
            # payload. The true lowest collision point is the board support.
            candidates=[]
            for gg in sorted(board):
                Rg=b.data.geom_xmat[gg].reshape(3,3);sz=b.model.geom_size[gg];pg=b.data.geom_xpos[gg];typ=int(b.model.geom_type[gg])
                if typ==mujoco.mjtGeom.mjGEOM_MESH:
                    mid=int(b.model.geom_dataid[gg]);adr=int(b.model.mesh_vertadr[mid]);num=int(b.model.mesh_vertnum[mid]);verts=b.model.mesh_vert[adr:adr+num]@Rg.T+pg;candidates.extend(verts[verts[:,2]<=np.min(verts[:,2])+1e-7])
                elif typ==mujoco.mjtGeom.mjGEOM_BOX:candidates.append(pg-Rg@(np.sign(Rg[2])*sz))
                elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:candidates.append(pg-np.array([0.,0.,sz[0]]))
                elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:
                    v=Rg[2,:2];v=v/max(np.linalg.norm(v),1e-12);candidates.append(pg-Rg@np.r_[sz[0]*v,np.sign(Rg[2,2])*sz[1]])
            candidates=np.asarray(candidates);bottom_point=candidates[candidates[:,2]<=np.min(candidates[:,2])+1e-7].mean(axis=0);hand_point=b.data.xpos[deck]+Br@nose;arm=hand_point-bottom_point;cg=b.data.subtree_com[chassis]-bottom_point
            hand_force=np.r_[-bm*9.81*cg[:2]/max(arm[2],.01),0.]
            ground_force=np.array([0.,0.,bm*9.81])-hand_force
            ground_wrench_residual=np.cross(arm,hand_force)-np.cross(cg,np.array([0.,0.,bm*9.81]))
            # Force on the robot is -hand_force, hence inverse-dynamics bias
            # includes +J.T hand_force (zero artificial vertical payload).
        loaded=b.data.qfrc_bias[rows]+(JP.T@hand_force)[rows]
        f=x[9:]*mg;tau=loaded[6:]-footmap[6:]@f
        kp=b.policy_values['kp'];angles=b.data.qpos[b.robot_qpos_adr]
        low=np.maximum(-b.policy_values['joint_effort_limit'],kp*(b.policy_values['joint_pos_lower_limit']-angles))
        high=np.minimum(b.policy_values['joint_effort_limit'],kp*(b.policy_values['joint_pos_upper_limit']-angles))
        combined=b.data.subtree_com[root].copy() if ground_entry else (b.model.body_subtreemass[root]*b.data.subtree_com[root]+bm*b.data.subtree_com[chassis])/(b.model.body_subtreemass[root]+bm)
        support=-(hull.equations[:,:2]@combined[:2]+hull.equations[:,2])-.01
        friction=[]
        for i,g in enumerate(feet):
            fx,fy,fz=x[9+3*i:12+3*i];mu=min(b.model.geom_friction[g,0],b.model.geom_friction[b.model.geom('ground').id,0])
            friction.extend([mu*fz-a*fx-c*fy for a in [-1,1] for c in [-1,1]])
        distances=np.full(len(pairs),.05)
        broad=np.linalg.norm(b.data.geom_xpos[pairs[:,0]]-b.data.geom_xpos[pairs[:,1]],axis=1)<pair_radii+.05
        for k in np.flatnonzero(broad):
            ga,gb=pairs[k];distances[k]=mujoco.mj_geomDistance(b.model,b.data,int(ga),int(gb),.05,fromto)
        bottoms=np.array([bottom(g) for g in sorted(board)])
        ground=bottoms+(.0002 if ground_entry else -.003)
        # Pad centres straddle the same nose surface by construction; require
        # the material centre lies within both pad tangential box footprints.
        coverage=[]
        material=b.data.xpos[deck]+b.data.xmat[deck].reshape(3,3)@nose
        for g in [b.model.geom('right_2f85_'+side+'_pad1').id for side in ['left','right']]:
            localpad=b.data.geom_xmat[g].reshape(3,3).T@(material-b.data.geom_xpos[g])
            # box's thinnest axis is its closing normal; both tangent axes matter.
            tangent=np.argsort(b.model.geom_size[g])[1:]
            coverage.extend((b.model.geom_size[g,tangent]-abs(localpad[tangent])+.002).tolist())
        eq=(footmap[:6]@f-loaded[:6])/mg
        if ground_entry:eq=np.r_[eq,float(np.min(bottoms))/.1]
        ground_friction=[.5*ground_force[2]-abs(ground_force[0])-abs(ground_force[1]),*(1e-5-abs(ground_wrench_residual))] if ground_entry else []
        inequalities=np.r_[(tau-low)/50,(high-tau)/50,friction,support/.1,(distances-.001)/.01,ground/.1,np.array(coverage)/.02,np.asarray(ground_friction)/mg]
        point,rotation=pad_tcp(b.model,b.data,'right')
        objective=float(np.sum(((x[:8]-qseed)/.5)**2)*.05+np.sum(((point-(seed_tcp if ground_entry else np.array([.03,-.57,.86])))/.1)**2)+2*(1-abs(Br[2,0]))+((x[8]-.365)/.02)**2*.05+.01*np.sum(x[9:]**2))
        cache=dict(eq=eq,ineq=inequalities,tau=tau,combined=combined,distances=distances,ground=ground,coverage=coverage,objective=objective,loaded=loaded,point=point,hand_force=hand_force,ground_force=ground_force,ground_wrench_residual=ground_wrench_residual)
        lastx=x.copy();return cache
    start=np.r_[qseed,.365,np.tile([0,0,1/len(feet)],len(feet))]
    bounds=list(map(tuple,b.model.jnt_range[jointids]))+[(.34,.385)]+[item for _ in feet for item in [(-2,2),(-2,2),(0,2)]]
    feasibility=None
    if ground_entry:
        from scipy.optimize import least_squares
        lower=np.array([item[0] for item in bounds]);upper=np.array([item[1] for item in bounds])
        seed=start.copy()
        def phase_one(x):
            value=evaluate(x)
            return np.r_[10*value['eq'],10*np.minimum(value['ineq'],0),.001*(x-seed)]
        feasibility=least_squares(phase_one,np.clip(start,lower+1e-8,upper-1e-8),bounds=(lower,upper),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        start=feasibility.x
    opt=minimize(lambda x:evaluate(x)['objective'],start,method='SLSQP',bounds=bounds,
        constraints=[dict(type='eq',fun=lambda x:evaluate(x)['eq']),dict(type='ineq',fun=lambda x:evaluate(x)['ineq'])],
        options=dict(maxiter=100,ftol=1e-9))
    result=evaluate(opt.x);tau=result['tau'];target=b.data.qpos[b.robot_qpos_adr]+tau/b.policy_values['kp']
    force_ok=bool(np.all(abs(tau)<=b.policy_values['joint_effort_limit']+1e-6));target_ok=bool(np.all(target>=b.policy_values['joint_pos_lower_limit']-1e-6) and np.all(target<=b.policy_values['joint_pos_upper_limit']+1e-6))
    initial_collision_pairs=[]
    for i in range(b.data.ncon):
        contact=b.data.contact[i];pair={int(contact.geom1),int(contact.geom2)}
        if pair&board and pair&nonpads and contact.dist< -1e-6:
            initial_collision_pairs.append(dict(geom_ids=[int(contact.geom1),int(contact.geom2)],body_pair=[b.model.body(int(b.model.geom_bodyid[g])).name for g in [contact.geom1,contact.geom2]],distance_m=float(contact.dist)))
    collision_ok=bool(not initial_collision_pairs and np.min(result['distances'],initial=1)>=-1e-6 and np.min(result['ground'])>=-1e-6)
    valid=bool(opt.success and np.max(abs(result['eq']))<1e-6 and np.min(result['ineq'])>=-1e-5)
    report=dict(kinematic_seed_source=str(seed_record) if seed_record else None,ground_supported_entry=ground_entry,initialization='ONLYt0 aligned39mm preclosed local component; NOT ground pickup or successful grasp',
        orientation_branch=branch,phase_one=dict(success=bool(feasibility.success),cost=float(feasibility.cost),evaluations=int(feasibility.nfev)) if feasibility is not None else None,solver=dict(success=bool(opt.success),message=str(opt.message),evaluations=evaluations,equality_inf=float(np.max(abs(result['eq']))),inequality_min=float(np.min(result['ineq']))),
        ground_entry_static_forces=dict(hand_on_board_world_N=result['hand_force'].tolist(),ground_on_board_world_N=result['ground_force'].tolist(),board_moment_residual_Nm=result['ground_wrench_residual'].tolist(),method='actual lowest geometry support + zero vertical hand force; finite tilt moment balance; ground friction L1 mu .5') if ground_entry else None,
        initial_collision_free=collision_ok,initial_invalid_contacts=initial_collision_pairs,torque_limit_ok=force_ok,pd_target_limit_ok=target_ok,static_robot_feasible=valid,
        initial_aperture_m=measured_opening(b.model,b.data,'right'),raw_preclose=140.,board_mass_kg=bm,estimated_loaded_tau_Nm=tau.tolist(),
        board_minimum_surface_z_m=float(np.min(result['ground'])-(.0002 if ground_entry else -.003)),board_long_axis_from_vertical_deg=float(np.rad2deg(np.arccos(np.clip(abs(Br[2,0]),0,1)))),
        board_nose_local_m=nose.tolist(),tcp_world_m=result['point'].tolist(),combined_com_world_m=result['combined'].tolist(),
        selected_joints=[b.model.joint(int(j)).name for j in jointids],selected_qpos=opt.x[:8].tolist(),foot_forces_world_N=(opt.x[9:]*mg).reshape(-1,3).tolist(),
        nonpad_clearance_min_m=float(np.min(result['distances'],initial=1)),pad_tangent_margins_m=result['coverage'],gripper=manifest())
    b.pose_target=target.copy();b.data.qvel[:]=0;mujoco.mj_forward(b.model,b.data)
    return b,target,report


def run(out,close=True,branch=1,ground_entry=False,seed_record=None):
    from dataclasses import asdict
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    b,target,pre=prepare(branch=branch,ground_entry=ground_entry,seed_record=seed_record);(out/'preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    if not(pre['torque_limit_ok'] and pre['pd_target_limit_ok'] and pre['static_robot_feasible'] and pre['initial_collision_free']):
        b.close();print(json.dumps(dict(stage='preflight rejected',solver=pre.get('solver'),torque_ok=pre['torque_limit_ok'],target_ok=pre['pd_target_limit_ok'],collision_ok=pre['initial_collision_free'])),flush=True);return
    rec=TorqueRecorder(b,out,dict(numerics=numerics_manifest(b),controller='pose',gripper_attachment=manifest(),task='ALIGNED LOCAL PRE-CLOSED holding/place component; not ground pickup',preflight=pre))
    rec.start(b.get_state(),backend_kwargs=dict(scene_config=b.scene_config,controller='pose',episode_time_limit=None,out_of_bounds_distance=None,offboard_distance=None,config=asdict(b.config)),runtime_state=b.export_replay_state())
    rec.write('asset_model',files=[dict(sha256=compiled_model_hash(b),kind='compiled_native_model')]);b.recorder=rec
    board=set(body_geoms(b,'chassis'));deck=b.model.body('deck').id;root=b.model.body('pelvis').id
    pads=[set(b.model.geom('right_2f85_'+side+'_pad'+str(j)).id for j in [1,2]) for side in ['left','right']]
    force=np.zeros(6);rows=[];controller=None;desired=np.zeros(29);stable=0.;maxstable=0.;hold_at=None;stop='holding deadline';error=None
    p0,R0=pad_tcp(b.model,b.data,'right');relative0=R0.T@(b.data.xpos[deck]-p0);reference=None
    nose_local=np.array([.365,0,.03*.65**2]);nose_relative0=R0.T@(b.data.xpos[deck]+b.data.xmat[deck].reshape(3,3)@nose_local-p0);hold_reference=None;gripper_geoms=set(body_geoms(b,'right_2f85_base_mount'))
    phase='ground_build_grasp' if ground_entry else 'local_hold';phase_start=0.;pad_stable=0.;wheel_stable=0.;lift_at=None;lower_at=None;release_at=None;placement_pass=False;wheel_geoms={b.model.geom(n).id for n in ['front_left_collision','front_right_collision','rear_left_collision','rear_right_collision']}
    try:
        for k in range(4000 if ground_entry else 2000):
            t=k*.002;rec.time_before=float(b.data.time);raw=255. if close and release_at is None else 0.;b.set_gripper_command(raw)
            if k<25:
                b.pose_target=target.copy();desired=b.policy_values['kp']*(target-b._robot_q())-b.policy_values['kd']*b._robot_dq()
            else:
                if controller is None:controller=ContactBalance(b);controller.feedback_enabled=True;reference=controller.com0.copy()
                if not ground_entry:phase='local_hold' if hold_at is None else 'held_precondition_achieved'
                hand_goal=p0.copy();hand_velocity=np.zeros(3);hand_acceleration=np.zeros(3);hand_R=R0.copy();hand_omega=np.zeros(3);hand_alpha=np.zeros(3);controller.grasp_wrench_tracking=lift_at is not None
                if ground_entry and lift_at is not None:
                    sm,sv,sa=ToeStages.ramp(t-lift_at,.8);hand_goal=lift_hand_p+np.array([0.,0.,.10*sm]);hand_velocity=np.array([0.,0.,.10*sv]);hand_acceleration=np.array([0.,0.,.10*sa]);hand_R=lift_hand_R
                if ground_entry and lower_at is not None:
                    sm,sv,sa=ToeStages.ramp(t-lower_at,.8);hand_goal=lift_hand_p+np.array([0.,0.,.10*(1-sm)]);hand_velocity=np.array([0.,0.,-.10*sv]);hand_acceleration=np.array([0.,0.,-.10*sa]);hand_R=lift_hand_R
                # Stop at the validated precondition; downstream placement is
                # constructed only if this real grasp passes, never by timeout.
                limits=b.policy_values['joint_effort_limit'];kp=b.policy_values['kp'];kd=b.policy_values['kd'];q=b._robot_q();v=b._robot_dq()
                lower=np.maximum(-limits,kp*(b.policy_values['joint_pos_lower_limit']-q)-kd*v)
                upper=np.minimum(limits,kp*(b.policy_values['joint_pos_upper_limit']-q)-kd*v)
                if k%5==0 or np.maximum(lower-desired,desired-upper).max()>1e-3:
                    try:desired,diag=controller.solve(reference,np.zeros(3),np.zeros(3),hand=dict(pos=hand_goal,vel=hand_velocity,acc=hand_acceleration,rotation=hand_R,omega=hand_omega,alpha=hand_alpha))
                    except RuntimeError as e:error=str(e);stop='QP rejected';break
                    rec.write('control_update',time=t,phase=phase,torque_desired_Nm=desired,qp=diag)
                controller.apply(desired)
            rec.phase=phase if ground_entry else ('aligned_close_hold' if close else 'aligned_open_control');rec.desired=desired.copy();b.step(update_policy=False)
            padforce=np.zeros(2);innerforce=np.zeros(2);other=[];robot_other=[];back_contacts=[];allcontacts=[];ground_force=0.;wheels=set();ground_id=b.model.geom('ground').id
            for i in range(b.data.ncon):
                c=b.data.contact[i];pair={int(c.geom1),int(c.geom2)}
                if not pair&board:continue
                mujoco.mj_contactForce(b.model,b.data,i,force)
                if force[0]<=1e-5:continue
                matched=False
                for j,geoms in enumerate(pads):
                    if pair&geoms:padforce[j]+=force[0];matched=True
                inner_alignment=None
                for side_index,geoms in enumerate(pads):
                    pg=next(iter(pair&geoms),None)
                    if pg is None:continue
                    partner=next(g for g in pads[1-side_index] if b.model.geom(g).name.endswith(b.model.geom(pg).name[-4:]))
                    PR=b.data.geom_xmat[pg].reshape(3,3);axis=int(np.argmin(b.model.geom_size[pg]));normal=PR[:,axis].copy();normal*=np.sign(normal@(b.data.geom_xpos[partner]-b.data.geom_xpos[pg]));on_board=(1. if c.geom2 in board else -1.)*c.frame.reshape(3,3)[0];inner_alignment=float(on_board@normal)
                    if inner_alignment>.5:innerforce[side_index]+=float(force[0])
                contact=dict(geom_ids=[int(c.geom1),int(c.geom2)],body_pair=[b.model.body(int(b.model.geom_bodyid[g])).name for g in [int(c.geom1),int(c.geom2)]],inner_normal_alignment=inner_alignment,pair=[b.model.geom(c.geom1).name,b.model.geom(c.geom2).name],normal_N=float(force[0]),deck_local=(b.data.xmat[deck].reshape(3,3).T@(c.pos-b.data.xpos[deck])).tolist())
                allcontacts.append(contact)
                if ground_id in pair:
                    ground_force+=max(float(force[0]),0.);wheels.update(pair&wheel_geoms)
                if not matched:
                    other.append(contact)
                    if pair&gripper_geoms:back_contacts.append(contact)
                    elif ground_id not in pair and not pair.issubset(board):robot_other.append(contact)
            p,R=pad_tcp(b.model,b.data,'right');relative=R.T@(b.data.xpos[deck]-p);drift=float(np.linalg.norm(relative-relative0))
            nose_relative=R.T@(b.data.xpos[deck]+b.data.xmat[deck].reshape(3,3)@nose_local-p);nose_slip=float(np.linalg.norm(nose_relative-nose_relative0))
            unsupported=bool(ground_force<=1e-5 and not robot_other and np.all(innerforce>1e-5))
            if unsupported and hold_reference is None:hold_reference=relative.copy()
            hold_drift=float(np.linalg.norm(relative-hold_reference)) if hold_reference is not None else 0.
            if not unsupported or hold_drift>.03:hold_reference=None
            tilt=float(np.arccos(np.clip(b.data.xmat[root].reshape(3,3)[2,2],-1,1)))
            purepad_valid=bool(np.all(innerforce>1e-5) and not other and drift<=.03 and tilt<.1)
            valid=bool(unsupported and hold_reference is not None and hold_drift<=.03 and tilt<.1)
            stable=stable+.002 if valid else 0.;maxstable=max(maxstable,stable)
            if stable>=1. and hold_at is None:hold_at=t;stop='true holding1s precondition achieved';
            if ground_entry:
                pad_stable=pad_stable+.002 if np.all(innerforce>1e-5) and nose_slip<=.03 and not robot_other else 0.
                if lift_at is None and pad_stable>=.15:lift_at=t;lift_hand_p, lift_hand_R=pad_tcp(b.model,b.data,'right');phase='actual_lift_after_ground_grasp';phase_start=t;rec.write('actual_grasp_to_lift',time=float(b.data.time),actual_tcp_position=lift_hand_p,actual_tcp_rotation=lift_hand_R,previous_reference_position=p0,transition='actual measured capture pose; native dynamics remains continuous')
                if hold_at is not None and lower_at is None:lower_at=t;phase='actual_lower_after_unsupported_hold';phase_start=t
                if lower_at is not None and release_at is None and t-lower_at>=.8 and ground_force>=.8*pre['board_mass_kg']*9.81:release_at=t;phase='actual_release_on_ground';phase_start=t
                wheel_stable=wheel_stable+.002 if release_at is not None and len(wheels)>=3 and ground_force>=.8*pre['board_mass_kg']*9.81 else 0.
                if wheel_stable>=.5:placement_pass=True;stop='real placement3wheel stable500ms';
            tauerr=float(np.max(abs(b.data.qfrc_applied[b.robot_dof_adr]-desired))) if k>=25 else 0.
            rows.append(dict(time=float(b.data.time),pad_normal_N=padforce.tolist(),inner_pad_normal_N=innerforce.tolist(),gripper_back_contacts=back_contacts,other_robot_contacts=robot_other,nose_material_slip_m=nose_slip,hold_relative_drift_m=hold_drift,pure_pad_strict_valid=purepad_valid,other_board_support=other,all_board_contacts=allcontacts,
                phase=phase,ground_board_normal_N=ground_force,wheel_ground_contact_count=len(wheels),pad_stable_seconds=pad_stable,wheel_stable_seconds=wheel_stable,relative_drift_m=drift,root_tilt_rad=tilt,root_z_m=float(b.data.xpos[root,2]),stable_hold_seconds=stable,actual_aperture_m=measured_opening(b.model,b.data,'right'),
                torque_error_Nm=tauerr,torque_desired_Nm=desired.tolist(),torque_actual_Nm=b.last_tau.tolist(),raw_command=raw))
            if np.any(b.data.qfrc_applied[:6]) or np.any(b.data.ctrl[:6]) or np.any(b.data.xfrc_applied):raise RuntimeError('Forbidden root/external input')
            if placement_pass or (hold_at is not None and not ground_entry):break
            if tilt>.25 or b.data.xpos[root,2]<.62:stop='actual upright guard';break
            if k>=25 and tauerr>1e-3:stop='torque execution guard';break
            if t>=(7.5 if ground_entry else 1.5):break
    finally:
        rec.close();b.recorder=None;b.close()
    report=dict(close=close,stop_reason=stop,error=error,holding_1s_pass=hold_at is not None,max_true_hold_seconds=maxstable,simulated_seconds=rows[-1]['time'] if rows else 0.,
        placement_executed=lower_at is not None,placement_pass=placement_pass,ground_build_grasp_at_s=lift_at,lower_at_s=lower_at,release_at_s=release_at,placement_status='real wheel-down release passed' if placement_pass else 'not completed; inspect actual achieved stage',preflight=pre)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'metrics.json').write_text(json.dumps(rows)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--open-control',action='store_true');p.add_argument('--branch',type=int,choices=[-1,1],default=1);p.add_argument('--ground-entry',action='store_true');p.add_argument('--seed-record',type=Path);a=p.parse_args();run(a.output,not a.open_control,a.branch,a.ground_entry,a.seed_record)
