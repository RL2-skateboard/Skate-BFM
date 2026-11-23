"""Native full-static zero-residual hover followed by bounded contact approach."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,copy
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from skate_skill_ppo import HoverResidual,PARAMS
import skate_skill_search as skill
from skate_sim.validation.balance import SupportGeometryGuard
from skate_sim.models.gripper import pad_tcp
from skate_sim.validation.clearance import contact_top_triangle_certificate

def quintic(p,v,a,end,T):
 c=np.zeros((6,3));c[:3]=[p,v,a/2];A=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]]);c[3:]=np.linalg.solve(A,np.array([end-p-v*T-a*T*T/2,-v-a*T,-a]));return c

def poly(c,t):return np.array([1,t,t*t,t**3,t**4,t**5])@c,np.array([0,1,2*t,3*t*t,4*t**3,5*t**4])@c,np.array([0,0,2,6*t,12*t*t,20*t**3])@c


def preposition_hand(b,cache,rec,report):
 source=Path(__file__).resolve().parents[1]/'logs/2026-11-09_17-00-09-edge-retract-pulse/events.jsonl'
 nearest=None
 with source.open() as f:
  for line in f:
   row=json.loads(line)
   if row.get('event')=='tick' and (nearest is None or abs(row['state']['time']-10.55)<abs(nearest['state']['time']-10.55)):nearest=row
 source_kw=copy.deepcopy(cache['kwargs']);source_kw['scene_config']['style']='standard'
 with skill.contextlib.redirect_stdout(skill.io.StringIO()):source_b=skill.create_backend('mujoco',**source_kw)
 scratch=mujoco.MjData(source_b.model);scratch.qpos[:]=nearest['state']['qpos'];mujoco.mj_forward(source_b.model,scratch);deck=source_b.model.body('deck').id;D=scratch.xmat[deck].reshape(3,3).copy();source_deck_position=scratch.xpos[deck].copy();source_chassis=source_b.model.body('chassis').id;source_chassis_position=scratch.xpos[source_chassis].copy();source_chassis_rotation=scratch.xmat[source_chassis].reshape(3,3).copy();source_roll=float(scratch.joint('deck_roll').qpos[0]);source_b.close()
 from skate_sim.models.board import board_styles
 selected=board_styles(b.scene_config['style'])[b.scene_config['style']];nose_x=.45625*selected.deck_length;nose_z=selected.kick_height*((nose_x-(selected.deck_length/2-selected.kick_length))/selected.kick_length)**2
 # The TCP centres the PLATE THICKNESS, not the previously hit top face.
 goal=source_deck_position+D@np.array([nose_x,0,nose_z]);normal=D@np.array([-.39,0,1.]);normal/=np.linalg.norm(normal);inward=D@np.array([-1.,0,-.39]);inward/=np.linalg.norm(inward)
 p0,R0=pad_tcp(b.model,b.data,'right');options=[np.column_stack([np.cross(sign*normal,inward),sign*normal,inward]) for sign in [-1,1]];Rg=min(options,key=lambda R:np.linalg.norm(Rotation.from_matrix(R@R0.T).as_rotvec()));rvgoal=Rotation.from_matrix(Rg@R0.T).as_rotvec()
 C=skill.ContactBalance(b);C.__dict__.update(copy.deepcopy(cache['controller']));C.b=b;C.m=b.model;C.numeric_backend='clarabel_lifted_ls';C.allow_board_support=False;C.toe_force_target=None
 foot=b.model.body('left_ankle_roll_link').id;FR=b.data.xmat[foot].reshape(3,3).copy();local=np.array([.12,0,-.03]);fp=b.data.xpos[foot]+FR@local;swing=dict(local=local,pos=fp,vel=np.zeros(3),acc=np.zeros(3),rotation=FR,omega=np.zeros(3),alpha=np.zeros(3));com=b.data.subtree_com[C.root].copy();tau=C.previous_tau.copy();start=float(b.data.time);root=b.model.body('pelvis').id;maxerr=0.;passed=False;stop='bounded hand preposition tracking guard';last={}
 entity_good=0.;entity={};virtual=mujoco.MjData(b.model);grasp_x=.46875*selected.deck_length;grasp_h=selected.kick_height*((grasp_x-(selected.deck_length/2-selected.kick_length))/selected.kick_length)**2;grasp_slope=2*selected.kick_height*(grasp_x-(selected.deck_length/2-selected.kick_length))/selected.kick_length**2;entity_P=source_deck_position+D@np.array([grasp_x,0.,grasp_h]);entity_N=D@np.array([-grasp_slope,0.,1.]);entity_N/=np.linalg.norm(entity_N);entity_half=selected.deck_thickness/2/np.sqrt(1+grasp_slope**2);all_board=set(skill.body_geoms(b,'chassis'));all_robot=set(skill.body_geoms(b,'pelvis'));padids={b.model.geom('right_2f85_'+side+'_pad'+str(j)).id for side in ['left','right'] for j in [1,2]}
 for tick in range(3000):
  elapsed=float(b.data.time)-start;ss,vv,aa=skill.ToeStages.ramp(elapsed,2.);hand=dict(pos=p0+ss*(goal-p0),vel=vv*(goal-p0),acc=aa*(goal-p0),rotation=Rotation.from_rotvec(ss*rvgoal).as_matrix()@R0,omega=vv*rvgoal,alpha=aa*rvgoal)
  kp=b.policy_values['kp'];kd=b.policy_values['kd'];q=b._robot_q();dq=b._robot_dq();lim=b.policy_values['joint_effort_limit'];lo=np.maximum(-lim,kp*(b.policy_values['joint_pos_lower_limit']-q)-kd*dq);hi=np.minimum(lim,kp*(b.policy_values['joint_pos_upper_limit']-q)-kd*dq)
  if tick%5==0 or np.maximum(lo-tau,tau-hi).max()>1e-3:
   try:tau,diag=C.solve(com,np.zeros(3),np.zeros(3),swing=None if cache['audit'].get('prefix_support_requirement')=='double' else swing,hand=hand)
   except SupportGeometryGuard as e:stop='measured support geometry guard: '+str(e);break
   rec.write('control_update',time=float(b.data.time),phase='C2_hand_preposition_before_toe_raise',torque_desired_Nm=tau,qp=diag,hand_reference=hand)
  if elapsed>=2. and not getattr(C,'arm_compensation_enabled',False):C.arm_compensation_enabled=True;C.arm_compensation_start=float(b.data.time)
  total,arm_detail=C.compensate_arm(tau,hand);C.apply(total);b.set_gripper_command(0.);rec.phase='C2_hand_preposition_before_toe_raise';rec.desired=total.copy();rec.time_before=float(b.data.time);b.step(update_policy=False)
  actual_error=float(np.max(abs(b.data.qfrc_applied[b.robot_dof_adr]-total)));rec.write('arm_compensation',time=float(b.data.time),baseline_tau29=tau,total_tau29=total,compensation=arm_detail);maxerr=max(maxerr,actual_error);ap,aR=pad_tcp(b.model,b.data,'right');tilt=float(np.arccos(np.clip(b.data.xmat[root].reshape(3,3)[2,2],-1,1)));last=dict(time_s=float(b.data.time),position_error_m=float(np.linalg.norm(ap-goal)),orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(Rg@aR.T).as_rotvec())),root_tilt_rad=tilt,root_z_m=float(b.data.xpos[root,2]),actual_tau_error_Nm=actual_error,actual_tcp_world_m=ap.tolist());rec.write('hand_preposition',**last)
  if actual_error>1e-3:stop='original actual torque execution guard';break
  if tilt>.25 or b.data.xpos[root,2]<.62:stop='original upright guard';break
  if np.any(b.data.qfrc_applied[:6]) or np.any(b.data.ctrl[:6]) or np.any(b.data.xfrc_applied):raise RuntimeError('Forbidden base/board force')
  sides=[]
  for side in ['left','right']:
   candidates=[]
   for j in [1,2]:
    g=b.model.geom('right_2f85_'+side+'_pad'+str(j)).id;other=b.model.geom('right_2f85_'+('right' if side=='left' else 'left')+'_pad'+str(j)).id;PR=b.data.geom_xmat[g].reshape(3,3);pg=b.data.geom_xpos[g];size=b.model.geom_size[g];normal=PR[:,1].copy();normal*=np.sign(normal@(b.data.geom_xpos[other]-pg));surface=entity_P+np.sign((pg-entity_P)@entity_N)*entity_half*entity_N;loc=PR.T@(surface-pg);gap=float(normal@(surface-(pg+size[1]*normal)));candidates.append(dict(pad=j,margin_m=min(gap,size[0]-abs(loc[0]),size[2]-abs(loc[2])),opening_normal_alignment=float(abs(normal@entity_N))))
   sides.append(max(candidates,key=lambda x:x['margin_m']))
  entity=dict(prediction_only=True,source='A10.55 world board-pose template; not actual C motion',grasp_material_x_m=grasp_x,min_margin_m=min(x['margin_m'] for x in sides),sides=sides)
  ready=elapsed>=2. and entity['min_margin_m']>=.003 and min(x['opening_normal_alignment'] for x in sides)>=np.cos(np.deg2rad(15))
  if ready:
   mujoco.mj_copyData(virtual,b.model,b.data);free=virtual.joint('deck_free').qpos;free[:3]=source_chassis_position;quat=Rotation.from_matrix(source_chassis_rotation).as_quat();free[3:]=quat[[3,0,1,2]];virtual.joint('deck_roll').qpos[:]=source_roll;mujoco.mj_forward(b.model,virtual);bad=[]
   for ci in range(virtual.ncon):
    co=virtual.contact[ci];pair={int(co.geom1),int(co.geom2)}
    if pair&all_board and pair&(all_robot-padids) and co.dist< -1e-6:bad.append([int(co.geom1),int(co.geom2),float(co.dist)])
   entity['forecast_nonpad_penetrations']=bad;ready=not bad
  entity_good=entity_good+.002 if ready else 0.;rec.write('predicted_pregrasp_entity',time=float(b.data.time),prediction=entity,continuous_ready_s=entity_good)
  if entity_good>=.1:passed=True;stop='predicted finite-pad entity window prepared; ideal tracking result separately reported';break
 report['hand_preposition']=dict(passed=passed,stop_reason=stop,source_events=str(source),source_sha256=skill.sha(source),source_time_s=nearest['state']['time'],target_position_world_m=goal.tolist(),target_rotation_world=Rg.tolist(),orientation_change_rad=float(np.linalg.norm(rvgoal)),initial_tcp_world_m=p0.tolist(),max_tau_error_Nm=maxerr,last=last,only_29_joint_actuators=True,ideal_tracking_gate_passed=bool(last and last['position_error_m']<=.02 and last['orientation_error_rad']<=.15),predicted_entity_window=entity,predicted_entity_ready_s=entity_good)
 if not passed:raise SupportGeometryGuard(stop)
 C.q0=b._robot_q().copy();rec.write('preposed_posture_reference',time=float(b.data.time),q29=C.q0,hand_reference_target_position=goal,hand_reference_target_rotation=Rg)
 cache=copy.copy(cache);cache['state']=b.get_state();cache['runtime']=b.export_replay_state();cache['controller']={key:copy.deepcopy(C.__dict__[key]) for key in cache['controller']};return cache,dict(pos=goal,vel=np.zeros(3),acc=np.zeros(3),rotation=Rg,omega=np.zeros(3),alpha=np.zeros(3))

def transfer_after_preposition(b,cache,hand,rec,report,actual_load_correction=False):
 from scipy.spatial import ConvexHull
 C=skill.ContactBalance(b);C.__dict__.update(copy.deepcopy(cache['controller']));C.b=b;C.m=b.model;C.feedback_enabled=True;C.numeric_backend='clarabel_lifted_ls';C.q0=b._robot_q().copy();root=C.root
 reference0=b.data.subtree_com[root].copy();right_geoms=list(C.feet[1]);left_geoms=list(C.feet[0]);right_xy=b.data.geom_xpos[right_geoms,:2].copy();hull=ConvexHull(right_xy);goal=reference0.copy();goal[:2]=right_xy[hull.vertices].mean(axis=0);goal[2]-=.08
 rec.write('preposition_to_transfer',time=float(b.data.time),posture_reference_q29=C.q0,com_reference_start=reference0,com_reference_goal=goal,hand_reference=hand,actual_state=b.get_state(),transition='no state write; only local controller reference updated')
 start=float(b.data.time);tau=C.previous_tau.copy();lift_start=None;lift_hold=0.;unload_good=0.;last={};passed=False;stop='actual unloading deadline';force=np.zeros(6);foot=b.model.body('left_ankle_roll_link').id
 correction_start=None;correction_delta=None
 for tick in range(5000 if actual_load_correction else 3000):
  elapsed=float(b.data.time)-start;ss,vv,aa=skill.ToeStages.ramp(elapsed,2.25);delta=goal-reference0;ref=reference0+ss*delta;vref=vv*delta;aref=aa*delta;swing=None;phase='preposed_hand_actual_transfer'
  if correction_start is not None:
   cc,cvv,caa=skill.ToeStages.ramp(float(b.data.time)-correction_start,1.);ref=goal+cc*correction_delta;vref=cvv*correction_delta;aref=caa*correction_delta;phase='single_measured_load_reference_correction'
  if lift_start is not None:
   phase='preposed_hand_actual_left_lift';ss,vv,aa=skill.ToeStages.ramp(float(b.data.time)-lift_start,.5);offset=np.array([0.,0.,.03]);swing=dict(local=np.zeros(3),pos=lift_p+ss*offset,vel=vv*offset,acc=aa*offset,rotation=lift_R,omega=np.zeros(3),alpha=np.zeros(3))
  kp=b.policy_values['kp'];kd=b.policy_values['kd'];q=b._robot_q();dq=b._robot_dq();lim=b.policy_values['joint_effort_limit'];lo=np.maximum(-lim,kp*(b.policy_values['joint_pos_lower_limit']-q)-kd*dq);hi=np.minimum(lim,kp*(b.policy_values['joint_pos_upper_limit']-q)-kd*dq)
  if tick%5==0 or np.maximum(lo-tau,tau-hi).max()>1e-3:
   try:tau,diag=C.solve(ref,vref,aref,swing=swing,hand=hand)
   except SupportGeometryGuard as e:stop='actual support guard: '+str(e);break
   rec.write('control_update',time=float(b.data.time),phase=phase,torque_desired_Nm=tau,qp=diag,reference_com=ref,hand_reference=hand)
  total,arm_detail=C.compensate_arm(tau,hand);C.apply(total);b.set_gripper_command(0.);rec.phase=phase;rec.desired=total.copy();rec.write('arm_compensation',time=float(b.data.time),baseline_tau29=tau,total_tau29=total,compensation=arm_detail);rec.time_before=float(b.data.time);b.step(update_policy=False)
  actual_error=float(np.max(abs(b.data.qfrc_applied[b.robot_dof_adr]-total)));loads=np.zeros(2);side_points=[[],[]];side_forces=[[],[]]
  for j in range(b.data.ncon):
   c=b.data.contact[j];pair={int(c.geom1),int(c.geom2)}
   if C.ground in pair:
    mujoco.mj_contactForce(b.model,b.data,j,force)
    for side in [0,1]:
     if pair&C.feet[side]:
      ff=max(float(force[0]),0.);loads[side]+=ff;side_points[side].append(c.pos[:2].copy());side_forces[side].append(ff)
  com=b.data.subtree_com[root];J=np.zeros((3,b.model.nv));mujoco.mj_jacSubtreeCom(b.model,b.data,J,root);cv=J@b.data.qvel;xi=com[:2]+cv[:2]/np.sqrt(9.81/max(C.com0[2],.1));hull=ConvexHull(b.data.geom_xpos[right_geoms,:2]);equations=hull.equations;den=np.linalg.norm(equations[:,:2],axis=1);margin=float(np.min(-(equations[:,:2]@com[:2]+equations[:,2])/den));ximargin=float(np.min(-(equations[:,:2]@xi+equations[:,2])/den));tilt=float(np.arccos(np.clip(b.data.xmat[root].reshape(3,3)[2,2],-1,1)));air=min(float(b.data.geom_xpos[g,2]-b.model.geom_size[g,0]) for g in left_geoms)
  if actual_load_correction and correction_start is None and lift_start is None and elapsed>=5.998:
   if min(loads)<=1e-5 or np.linalg.norm(cv[:2])>.04 or tilt>.1:stop='load correction actual stable-support precondition not met';break
   centres=[np.average(points,axis=0,weights=weights) for points,weights in zip(side_points,side_forces)];fraction=float(loads[0]/loads.sum());correction_delta=np.r_[(fraction-.025)*(centres[1]-centres[0]),0.];newgoal=goal+correction_delta;newmargin=float(np.min(-(equations[:,:2]@newgoal[:2]+equations[:,2])/den))
   if newmargin<.01:stop='single load correction reference outside1cm right support';break
   correction_start=float(b.data.time);rec.write('single_actual_load_reference_correction',time=correction_start,measured_left_fraction=fraction,target_left_fraction=.025,actual_foot_cop_xy_m=np.asarray(centres),reference_delta_xyz_m=correction_delta,new_reference_com=newgoal,right_support_margin_m=newmargin,transition_seconds=1.,max_wait_after_transition_seconds=1.)
  if correction_start is not None and lift_start is None and float(b.data.time)-correction_start>=2.:stop='actual unload guard not reached after single reference correction';break
  if lift_start is not None and float(b.data.time)-lift_start>1.5:stop='actual lift hold deadline';break
  ready=loads[0]<.05*C.mg and loads[1]>.8*C.mg and margin>=.01 and ximargin>=.01 and abs(cv[1])<=.04 and tilt<.1
  if lift_start is None:
   unload_good=unload_good+.002 if ready and tick else (.002 if ready else 0.)
   if unload_good>=.05:lift_start=float(b.data.time);lift_p=b.data.xpos[foot].copy();lift_R=b.data.xmat[foot].reshape(3,3).copy()
  lift_hold=lift_hold+.002 if lift_start is not None and air>=.02 and loads[0]<1e-5 else 0.
  last=dict(time_s=float(b.data.time),phase=phase,foot_ground_normal_N=loads.tolist(),com_margin_m=margin,dcm_margin_m=ximargin,com_velocity_y_m_s=float(cv[1]),left_foot_clearance_m=air,left_air_hold_s=lift_hold,root_tilt_rad=tilt,root_z_m=float(b.data.xpos[root,2]),actual_tau_error_Nm=actual_error);rec.write('preposed_transfer',**last)
  if np.any(b.data.qfrc_applied[:6]) or np.any(b.data.ctrl[:6]) or np.any(b.data.xfrc_applied):raise RuntimeError('Forbidden base/board force')
  if actual_error>1e-3:stop='original actual torque execution guard';break
  if tilt>.25 or b.data.xpos[root,2]<.62:stop='original upright guard';break
  if lift_hold>=.25:passed=True;stop='actual single-support lift250ms passed';break
 report['preposed_transfer']=dict(passed=passed,stop_reason=stop,last=last,lift_start_time_s=lift_start,load_reference_correction_time_s=correction_start,load_reference_correction_delta_xyz_m=None if correction_delta is None else correction_delta.tolist())
 if not passed:raise SupportGeometryGuard(stop)
 cache=copy.copy(cache);cache['state']=b.get_state();cache['runtime']=b.export_replay_state();cache['controller']={key:copy.deepcopy(C.__dict__[key]) for key in cache['controller']};cache['right_xy']=b.data.geom_xpos[right_geoms,:2].copy();return cache

def run(out,pulse=False,control_kind="pulse",pregrasp=False,double_support=False,style=None,actual_load_correction=False):
 if control_kind=="raised":pulse=False
 if (out/'events.jsonl').exists():raise FileExistsError('Do not overwrite an existing physical experiment')
 out.mkdir(parents=True,exist_ok=True)
 cache,audit,b,rec=skill.prepare_prefix(skill.LAYOUT,keep_live=True,record=out,metadata={'continuation_experiment':'Full-static zero residual hover then actual contact approach; no runtime state reset','geometry_guard_version':'v2'},style=style,end_time=5. if double_support else None,require_single_support=not double_support)
 if cache is None:raise RuntimeError('Actual static prefix failed')
 report={'prefix':audit,'contact_classifier_version':'board_side_witness_actual_mesh_v5_closed_triangles','residual_policy':'zero','post_t0_state_writes':0,'approach_duration_s':.8,'approach_normal_travel_m':.10,'pulse_executed':False,'control_kind':control_kind,'force_soft_target_enabled':bool(pulse and control_kind=='pulse'),'runner_sha256':skill.sha(__file__),'comparison_scope':'Only compare variants over shared actual absolute simulated time; goal hovering is not pop/grasp success'}
 rows=[];started=time.perf_counter();force=np.zeros(6);impulse=0.;first=None;stop='bounded approach deadline';errmax=0.;edge_impulse=0.;edge_first=None;edge_initial_force=None
 try:
  hand_goal=None
  if pregrasp:cache,hand_goal=preposition_hand(b,cache,rec,report)
  if double_support:cache=transfer_after_preposition(b,cache,hand_goal,rec,report,actual_load_correction)
  env=HoverResidual(cache,backend=b,restore_state=False);env.hand_target=hand_goal
  for _ in range(300):
   _,_,done,info=env.step(np.zeros(8),record=rec)
   if done:break
  report['hover']=info
  if not info['success']:raise RuntimeError('Zero actual hover failed: '+str(info))
  t0=float(b.data.time);report['actual_hover_time_s']=t0
  # Match the previous reference position/velocity/acceleration at the switch.
  old=env.swing;delta=t0-float(rec.time_before)
  p0=np.asarray(old['pos'])+delta*np.asarray(old['vel'])+.5*delta**2*np.asarray(old['acc']);v0=np.asarray(old['vel'])+delta*np.asarray(old['acc']);a0=np.asarray(old['acc'])
  R0=(Rotation.from_rotvec(delta*np.asarray(old['omega'])+.5*delta**2*np.asarray(old['alpha'])).as_matrix()@old['rotation']);omega0=np.asarray(old['omega'])+delta*np.asarray(old['alpha']);alpha0=np.asarray(old['alpha']);orientation_coef=quintic(np.zeros(3),omega0,alpha0,Rotation.from_matrix(env.stages.rotation_goal@R0.T).as_rotvec(),.8);report['rotation_handoff']='C2 rotation-vector reference from prior actual controller reference omega/alpha to the same frozen geometric goal';report['handoff_omega_rad_s']=omega0.tolist();DR=b.data.xmat[env.deck].reshape(3,3);normal=DR@env.stages.normal
  T=.8;end=p0 if control_kind=="raised" else p0-.10*normal
  coef=np.zeros((6,3));coef[0]=p0;coef[1]=v0;coef[2]=a0/2
  A=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
  coef[3:]=np.linalg.solve(A,np.array([end-p0-v0*T-a0*T*T/2,-v0-a0*T,-a0]))
  control=env.control;control.allow_board_support=True;control.toe_force_target=None;base=env.base.copy();target=env.reference.copy();target[:2]=env.preload
  pulse_start=None;retract_start=None;pulse_local=None
  for tick in range(900 if pulse else 400):
   elapsed=float(b.data.time)-t0;t=min(elapsed,T)
   pos=np.array([1,t,t*t,t**3,t**4,t**5])@coef;vel=np.array([0,1,2*t,3*t*t,4*t**3,5*t**4])@coef;acc=np.array([0,0,2,6*t,12*t*t,20*t**3])@coef
   phase='descend_until_valid_top'
   if pulse_start is not None:
    phase='actual_contact_timed_pulse';u=float(b.data.time)-pulse_start;DR=b.data.xmat[env.deck].reshape(3,3);dp=b.data.xpos[env.deck];fraction=max(0.,min(u/.03,(PARAMS['pulse_time']-u)/.03,1.));control.toe_force_target=PARAMS['pulse_peak_N']*fraction if control_kind=='pulse' else None
    localp,localv,locala=poly(pulse_coef,min(u,PARAMS['pulse_time']));point=dp+DR@localp;jp=np.zeros((3,b.model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(b.model,b.data,jp,jr,point,env.deck);pos=point;vel=jp@b.data.qvel+DR@localv;acc=DR@locala
    if u>=PARAMS['pulse_time'] or (rows and rows[-1]['nose_angle_rad']>.10):
     if retract_start is None:retract_start=float(b.data.time);retract_p=pos.copy();retract_R=R0.copy();retract_coef=quintic(pos,vel,acc,pos+np.array([0.,0.,.12]),.6)
   if retract_start is not None:
    phase='retract_after_pulse';control.toe_force_target=None;u=float(b.data.time)-retract_start;pos,vel,acc=poly(retract_coef,min(u,.6))
   rv,rvv,rva=poly(orientation_coef,min(elapsed,.8));theta=np.linalg.norm(rv);cross=np.array([[0.,-rv[2],rv[1]],[rv[2],0.,-rv[0]],[-rv[1],rv[0],0.]]);Jl=np.eye(3)+(.5 if theta<1e-7 else (1-np.cos(theta))/theta**2)*cross+(1/6 if theta<1e-7 else (theta-np.sin(theta))/theta**3)*(cross@cross);rotation=Rotation.from_rotvec(rv).as_matrix()@R0;omega=Jl@rvv;alpha=Jl@rva
   def left_jac(x):
    th=np.linalg.norm(x);X=np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]]);return np.eye(3)+(.5 if th<1e-7 else (1-np.cos(th))/th**2)*X+(1/6 if th<1e-7 else (th-np.sin(th))/th**3)*(X@X)
   alpha+=((left_jac(rv+1e-6*rvv)-left_jac(rv-1e-6*rvv))/2e-6)@rvv
   swing=dict(local=env.stages.local,pos=pos,vel=vel,acc=acc,rotation=rotation,omega=omega,alpha=alpha)
   kp=b.policy_values['kp'];kd=b.policy_values['kd'];q=b._robot_q();dq=b._robot_dq();limit=b.policy_values['joint_effort_limit'];lo=np.maximum(-limit,kp*(b.policy_values['joint_pos_lower_limit']-q)-kd*dq);hi=np.minimum(limit,kp*(b.policy_values['joint_pos_upper_limit']-q)-kd*dq)
   if tick%5==0 or np.maximum(lo-base,base-hi).max()>1e-3:
    base,diag=control.solve(target,np.zeros(3),np.zeros(3),swing=swing,hand=hand_goal)
    rec.write('control_update',time=float(b.data.time),phase=phase,torque_desired_Nm=base,qp=diag)
   total,arm_detail=control.compensate_arm(base,hand_goal);control.apply(total);b.set_gripper_command(0.);rec.phase=phase;rec.desired=total.copy();rec.write('arm_compensation',time=float(b.data.time),baseline_tau29=base,total_tau29=total,compensation=arm_detail);rec.time_before=float(b.data.time);b.step(update_policy=False)
   err=float(np.max(abs(b.data.qfrc_applied[b.robot_dof_adr]-total)));errmax=max(errmax,err)
   DR=b.data.xmat[env.deck].reshape(3,3);dp=b.data.xpos[env.deck];topforce=0.;bad=0.;edgeforce=0.;contacts=[]
   for i in range(b.data.ncon):
    c=b.data.contact[i];pair={int(c.geom1),int(c.geom2)}
    if not(pair&env.board and pair&env.robot):continue
    mujoco.mj_contactForce(b.model,b.data,i,force)
    if force[0]<=1e-5:continue
    fg=next(iter(pair&set(env.stages.front)),None);sign=1 if fg is not None and c.geom2==fg else -1;reaction=sign*c.frame.reshape(3,3)[0]
    midpoint=DR.T@(c.pos-dp);board_witness=c.pos-.5*float(c.dist)*reaction;local=DR.T@(board_witness-dp);surface=env.stages.surface(local[:2])
    # MuJoCo's contact.pos is the two witness points' midpoint; classify the
    # actual board-side witness, not an expanded nominal tail rectangle.
    triangle_certificate=None
    if surface is None and fg is not None and reaction@(DR@env.stages.normal)>.5:
     participating=next(iter(pair&env.board));triangle_certificate=contact_top_triangle_certificate(b.model,b.data,participating,board_witness,DR@env.stages.normal,env.stages.board_style.deck_length)
    top_height_valid=(surface is not None and abs(local[2]-surface)<=1e-4) or (triangle_certificate is not None and triangle_certificate['accepted'])
    valid=fg is not None and local[0]<=-(env.stages.board_style.deck_length/2-env.stages.board_style.kick_length) and top_height_valid and reaction@(DR@env.stages.normal)>.5
    known_edge=bool(not valid and phase=='retract_after_pulse' and fg is not None and surface is not None and abs(local[2]-surface)<=1e-4 and local[0]<=-(env.stages.board_style.deck_length/2-env.stages.board_style.kick_length) and reaction@(DR@env.stages.normal)>0.)
    if valid:topforce+=float(force[0])
    elif known_edge:edgeforce+=float(force[0])
    else:bad+=float(force[0])
    contacts.append(dict(geom_ids=[int(c.geom1),int(c.geom2)],geoms=[b.model.geom(g).name for g in [int(c.geom1),int(c.geom2)]],reaction_dot_top=float(reaction@(DR@env.stages.normal)),contact_distance_m=float(c.dist),contact_midpoint_deck_local_m=midpoint.tolist(),board_side_witness_deck_local_m=local.tolist(),actual_ray_surface_z_m=surface,triangle_contact_certificate=triangle_certificate,deck_local_m=local.tolist(),force_N=float(force[0]),valid_top=bool(valid),retract_tail_edge=known_edge))
   impulse+=topforce*.002;edge_impulse+=edgeforce*.002
   if edgeforce>0 and edge_first is None:edge_first=float(b.data.time);edge_initial_force=edgeforce
   tilt=float(np.arccos(np.clip(b.data.xmat[env.root].reshape(3,3)[2,2],-1,1)));angle=float(np.arctan2(DR[2,0],np.linalg.norm(DR[:2,0])))
   row=dict(time_s=float(b.data.time),elapsed_s=float(b.data.time)-t0,phase=phase,valid_top_force_N=topforce,valid_top_impulse_Ns=impulse,invalid_contact_force_N=bad,edge_residual_force_N=edgeforce,edge_residual_impulse_Ns=edge_impulse,nose_angle_rad=angle,root_tilt_rad=tilt,root_z_m=float(b.data.xpos[env.root,2]),toe_world_m=env.stages.actual()[0].tolist(),toe_target_world_m=pos.tolist(),measurements=env.stages.measurements(),torque_error_Nm=err,contacts=contacts);rows.append(row);rec.write('contact_approach',**row)
   if np.any(b.data.qfrc_applied[:6]) or np.any(b.data.ctrl[:6]) or np.any(b.data.xfrc_applied):raise RuntimeError('Forbidden external/base force')
   if err>1e-3:stop='original actual torque guard';break
   if edgeforce>0 and (float(b.data.time)-edge_first>.05 or edgeforce>2*edge_initial_force):stop='bounded/growing tail-edge residual contact during retract';break
   if bad>1e-5:stop='unexpected side/underside/non-toe contact';break
   if topforce>1e-5 and first is None:
    first=row
    if not pulse:stop='first valid top contact reached; pulse not yet executed';break
    pulse_start=float(b.data.time);pulse_local=DR.T@(pos-dp);jp=np.zeros((3,b.model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(b.model,b.data,jp,jr,pos,env.deck);pulse_coef=quintic(pulse_local,DR.T@(vel-jp@b.data.qvel),DR.T@acc,pulse_local-.012*env.stages.normal,PARAMS['pulse_time']);report['pulse_executed']=True;report['pulse_start_time_s']=pulse_start
   if retract_start is not None and float(b.data.time)-retract_start>=.65:stop='bounded pulse and retract completed';break
   if tilt>.25 or b.data.xpos[env.root,2]<.62:stop='original upright guard';break
  report.update(stop_reason=stop,first_valid_top_contact=first,valid_top_impulse_Ns=impulse,peak_top_force_N=max([r['valid_top_force_N'] for r in rows],default=0.),actual_approach_ticks=len(rows),max_actual_torque_error_Nm=errmax,last=rows[-1] if rows else None)
 except SupportGeometryGuard as e:report.update(stop_reason='measured support geometry guard',error=repr(e),numerical_invalid=False)
 except Exception as e:report.update(stop_reason='runtime/numerical stop',error=repr(e),numerical_invalid=True);raise
 finally:
  report['retract_edge_impulse_Ns']=edge_impulse;report['edge_first_time_s']=edge_first;report['edge_rule']='known toe/tail upper-edge sliding only during force-off retraction, max50ms and max2x initial edge force; separate from valid top impulse';report.setdefault('valid_top_impulse_Ns',impulse);report.setdefault('first_valid_top_contact',first);report.setdefault('last',rows[-1] if rows else None);report.setdefault('actual_approach_ticks',len(rows));report.setdefault('peak_top_force_N',max([r['valid_top_force_N'] for r in rows],default=0.));report.setdefault('max_actual_torque_error_Nm',errmax);report['wall_seconds']=time.perf_counter()-started;skill.dump(out/'report.json',report);rec.close();b.recorder=None;b.close();print(json.dumps(report,default=skill.plain),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pulse',action='store_true');p.add_argument('--pregrasp',action='store_true');p.add_argument('--double-support',action='store_true');p.add_argument('--style');p.add_argument('--actual-load-correction',action='store_true');p.add_argument('--control',choices=['pulse','pulse-off','raised'],default='pulse');args=p.parse_args();run(args.output,args.pulse,args.control,args.pregrasp,args.double_support,args.style,args.actual_load_correction)
