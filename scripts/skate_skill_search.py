"""Small native29 Robotiq pop-control CEM experiment; no learned policy/PPO claim."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,contextlib,copy,hashlib,io,json,multiprocessing as mp,pickle,resource,sys,time
from datetime import datetime
from dataclasses import asdict,is_dataclass
from pathlib import Path
import numpy as np
SIM=Path(__file__).resolve().parents[2]/'skate-sim';sys.path.insert(0,str(SIM/'src'))
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from skate_sim.backends import create_backend
from skate_sim.replay import read_record,_backend_kwargs
from skate_sim.record import compiled_model_hash,numerics_manifest
from skate_sim.validation.balance import ContactBalance,TorqueRecorder,inset_support,project_support,toe_reference
from skate_sim.validation.tail_catch import body_geoms
from skate_sim.validation.toe_stages import ToeStages
from skate_sim.validation.place import geom_bottom

PREFIX=SIM/'runs/tail_catch/contact_balance/qp_dcm_actual'
LAYOUT=[.3609061980297052,-.03543082218342743,.7193153508938036]
NAMES=['h_hover_world','translate_time','cop_preload','pulse_peak_N','pulse_time','yaw_offset']
LOW=np.array([.20,.8,0.,80.,.14,-.35]);HIGH=np.array([.29,2.,.03,220.,.26,.35])
MEAN=np.array([.24,1.3,.015,150.,.22,0.]);STD=(HIGH-LOW)*.2
_WORKER=None
_WORKER_ERROR=None

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def binary_model_hash(b):
 buffer=np.zeros(mujoco.mj_sizeModel(b.model),dtype=np.uint8)
 mujoco.mj_saveModel(b.model,buffer=buffer)
 return hashlib.sha256(buffer.tobytes()).hexdigest()

def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,default=plain)+'\n')
def descendant(model,body,ancestor):
 while body and body!=ancestor:body=int(model.body_parentid[body])
 return body==ancestor

def prepare_prefix(layout,keep_live=False,record=None,metadata=None,style=None,end_time=None,require_single_support=True):
 start=time.perf_counter();rows=read_record(PREFIX);kw,initial=_backend_kwargs(rows,rows[0])
 with contextlib.redirect_stdout(io.StringIO()):original=create_backend('mujoco',**kw)
 kwargs=copy.deepcopy(kw)
 if style is not None:kwargs['scene_config']['style']=style
 if layout is not None:
  quat=Rotation.from_euler('z',layout[2]).as_quat();kwargs['scene_config']['board_pose']=dict(pos=[layout[0],layout[1],.1125],quat=quat[[3,0,1,2]].tolist())
 with contextlib.redirect_stdout(io.StringIO()):b=create_backend('mujoco',**kwargs)
 if layout is None and style is None:
  b.set_state(initial['state']);b.import_replay_state(initial['runtime_state'])
 else:
  root=original.model.body('pelvis').id
  # Fresh new-layout initial state. Only named robot posture is reused; the
  # skateboard stays at its new model qpos0 and starts at rest.
  for j in range(original.model.njnt):
   if not descendant(original.model,int(original.model.jnt_bodyid[j]),root):continue
   name=original.model.joint(j).name;source=int(original.model.jnt_qposadr[j]);count=7 if original.model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1
   b.data.joint(name).qpos[:]=np.asarray(initial['state']['qpos'])[source:source+count]
  b.data.qvel[:]=0;b.pose_target=np.asarray(initial['runtime_state']['runtime']['pose_target']);mujoco.mj_forward(b.model,b.data)
 original.close()
 prefix_recorder=None
 if record is not None:
  prefix_recorder=TorqueRecorder(b,record,dict(controller='pose',numerics=numerics_manifest(b),experiment='Fresh static prefix plus fixed CEM-parameter closed-loop evaluation',**(metadata or {})))
  recordkw=copy.deepcopy(kwargs)
  if is_dataclass(recordkw.get('config')):recordkw['config']=asdict(recordkw['config'])
  prefix_recorder.start(b.get_state(),backend_kwargs=recordkw,runtime_state=b.export_replay_state());b.recorder=prefix_recorder
  prefix_recorder.write('compiled_physics_model',binary_sha256=binary_model_hash(b),exported_xml_sha256=compiled_model_hash(b))
 controller=None;root=b.model.body('pelvis').id;ground=b.model.geom('ground').id
 feet=[set(body_geoms(b,s+'_ankle_roll_link')) for s in ['left','right']];board=set(body_geoms(b,'chassis'));robot=set(body_geoms(b,'pelvis'))
 f=np.zeros(6);last_window=[];max_cross=0;body_error=0.;count=0
 for row in rows:
  if end_time is not None and float(b.data.time)>=end_time-1e-10:break
  if row['event']=='control_update':
   if controller is None:
    controller=ContactBalance(b);controller.feedback_enabled=True;controller.solve(controller.com0,np.zeros(3),np.zeros(3))
    right_xy=b.data.geom_xpos[list(feet[1]),:2].copy()
   controller.previous_tau=np.asarray(row['torque_desired_Nm'])
  elif row['event']=='tick':
   if prefix_recorder:prefix_recorder.desired=controller.previous_tau.copy() if controller else np.zeros(29);prefix_recorder.phase='fixed body-command prefix in fresh layout';prefix_recorder.time_before=float(b.data.time)
   b.step_recorded(row['action'],row['command'],row['inputs']);count+=1
   loads=np.zeros(2);cross=0
   for i in range(b.data.ncon):
    c=b.data.contact[i];pair={int(c.geom1),int(c.geom2)}
    if pair&board and pair&robot and c.dist< -1e-6:cross+=1
    if ground not in pair:continue
    mujoco.mj_contactForce(b.model,b.data,i,f)
    for side in [0,1]:
     if pair&feet[side]:loads[side]+=max(float(f[0]),0.)
   max_cross=max(max_cross,cross)
   tilt=float(np.arccos(np.clip(b.data.xmat[root].reshape(3,3)[2,2],-1,1)))
   clearance=min(float(b.data.geom_xpos[g,2]-b.model.geom_size[g,0]) for g in feet[0])
   last_window.append(dict(time=float(b.data.time),loads=loads.tolist(),tilt=tilt,clearance=clearance))
   if len(last_window)>125:last_window.pop(0)
   body_error=max(body_error,float(np.max(abs(b.data.qpos[b.robot_qpos_adr]-np.asarray(row['state']['qpos'])[original.robot_qpos_adr]))))
 # Revalidate the actual final125 physical ticks, not the old report boolean.
 valid=bool(len(last_window)==125 and max_cross==0 and all(r['clearance']>=.02 and r['loads'][0]<1e-5 and r['loads'][1]>.8*controller.mg and r['tilt']<.1 for r in last_window))
 if not require_single_support:valid=bool(len(last_window)==125 and max_cross==0 and all(min(r['loads'])>.1*controller.mg and r['tilt']<.1 for r in last_window))
 audit=dict(style=kwargs.get('scene_config',{}).get('style','standard'),prefix_support_requirement='single' if require_single_support else 'double',double_support_actual_last025s_pass=valid if not require_single_support else None,layout=layout,new_layout_initialized_fresh=layout is not None or style is not None,source_commands=str(PREFIX),source_commands_sha256=sha(PREFIX/'events.jsonl'),physics_ticks=count,simulated_seconds=float(b.data.time),robot_command_replay_qpos_difference=body_error,max_robot_board_cross_contacts=max_cross,single_support_actual_last025s_pass=valid if require_single_support else None,last025s=last_window,wall_seconds=time.perf_counter()-start,model_sha256=compiled_model_hash(b),model_binary_sha256=binary_model_hash(b),model_identity_contract='exact MJB bytes, MuJoCo version and asset provenance; XML MIME metadata retained separately',numerics=numerics_manifest(b))
 if not valid:b.close();return None,audit
 excluded={'b','m','debug_numeric','debug_problem','previous_solution'}
 memory={k:copy.deepcopy(v) for k,v in controller.__dict__.items() if k not in excluded}
 board_top=max(-geom_bottom(b.model,b.data,g) if False else float(b.data.geom_xpos[g,2]+b.model.geom_rbound[g]) for g in board)
 # Exact top support from the same compiled collision geometries, using the
 # bottom helper under a reflected observation frame without changing data.
 tops=[]
 for g in board:
  typ=int(b.model.geom_type[g]);R=b.data.geom_xmat[g].reshape(3,3);pos=b.data.geom_xpos[g];size=b.model.geom_size[g]
  if typ==mujoco.mjtGeom.mjGEOM_MESH:
   mid=int(b.model.geom_dataid[g]);a=int(b.model.mesh_vertadr[mid]);n=int(b.model.mesh_vertnum[mid]);tops.append(float(np.max(b.model.mesh_vert[a:a+n]@R[2]+pos[2])))
  elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:tops.append(float(pos[2]+size[0]))
  elif typ==mujoco.mjtGeom.mjGEOM_BOX:tops.append(float(pos[2]+abs(R[2])@size))
  elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:tops.append(float(pos[2]+size[0]*np.linalg.norm(R[2,:2])+size[1]*abs(R[2,2])))
  elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:tops.append(float(pos[2]+size[0]+size[1]*abs(R[2,2])))
  else:raise RuntimeError('Unvalidated collision geometry in search height bound')
 board_top=max(tops);audit['board_collision_top_world_m']=board_top;audit['toe_radius_m']=.005
 cache=dict(kwargs=kwargs,state=b.get_state(),runtime=b.export_replay_state(),controller=memory,right_xy=right_xy,audit=audit)
 if keep_live:return cache,audit,b,prefix_recorder
 if prefix_recorder:prefix_recorder.close();b.recorder=None
 b.close();return cache,audit

def init_worker(cache):
 global _WORKER,_WORKER_ERROR
 try:
  with contextlib.redirect_stdout(io.StringIO()):b=create_backend('mujoco',**cache['kwargs'])
  if binary_model_hash(b)!=cache['audit']['model_binary_sha256']:raise RuntimeError('Search worker exact MJB identity mismatch')
  _WORKER=(b,cache);_WORKER_ERROR=None
 except Exception as error:
  _WORKER=(None,cache);_WORKER_ERROR=repr(error)

def probe_worker(cache,queue):
 init_worker(cache)
 if _WORKER_ERROR is not None:queue.put(dict(ok=False,error=_WORKER_ERROR));return
 b,_=_WORKER;queue.put(dict(ok=True,model_binary_sha256=binary_model_hash(b),exported_xml_sha256=compiled_model_hash(b),mujoco=mujoco.__version__));b.close()

def evaluate_candidate(task,record=None,restore_state=True,existing_recorder=None,guard_version='v1',press=True):
 if _WORKER_ERROR is not None:raise RuntimeError('Worker initialization rejected: '+_WORKER_ERROR)
 index,vector=task;parameters=dict(zip(NAMES,map(float,vector)));parameters['geometry_guard_version']=guard_version;b,cache=_WORKER
 start=time.perf_counter()
 if restore_state:b.set_state(cache['state']);b.import_replay_state(cache['runtime'])
 else:
  error=np.max(abs(np.array(b.export_replay_state()['integration'])-np.array(cache['runtime']['integration'])))
  if error>1e-12:raise RuntimeError('Fresh live prefix differs from cached continuation boundary')
 controller=ContactBalance(b);controller.__dict__.update(copy.deepcopy(cache['controller']));controller.b=b;controller.m=b.model;controller.numeric_backend='clarabel_lifted_ls';controller.allow_board_support=True
 root=b.model.body('pelvis').id;deck=b.model.body('deck').id;DR=b.data.xmat[deck].reshape(3,3);deckp=b.data.xpos[deck].copy()
 goal=deckp+DR@np.array([-.36,0,.23-deckp[2]])
 try:rotation,geometry=toe_reference(b,goal,prefer_flat=cache['audit']['layout'] is not None)
 except RuntimeError as e:return dict(index=index,parameters=parameters,score=-100.,stop_reason='offline geometry rejected',error=str(e),wall_seconds=time.perf_counter()-start,events={})
 stages=ToeStages(b,press,rotation,parameters)
 right_vertices=inset_support(cache['right_xy'],.01);reference=b.data.subtree_com[root].copy()
 direction=goal[:2]-reference[:2];direction/=max(float(np.linalg.norm(direction)),1e-12)
 preload=project_support(reference[:2]+parameters['cop_preload']*direction,right_vertices)
 board=set(body_geoms(b,'chassis'));robot=set(body_geoms(b,'pelvis'));front=set(stages.front);pads=[set(b.model.geom('right_2f85_'+side+'_pad'+str(j)).id for j in [1,2]) for side in ['left','right']]
 normal_local=np.array([.39,0,1.]);normal_local/=np.linalg.norm(normal_local)
 recorder=existing_recorder;states=[]
 if record is not None and recorder is None:
  recorder=TorqueRecorder(b,record,dict(controller='pose',numerics=numerics_manifest(b),experiment='CEM candidate continuation; source initial physical prefix separately referenced',parameters=parameters,prefix=cache['audit']))
  record_kwargs=copy.deepcopy(cache['kwargs'])
  if is_dataclass(record_kwargs.get('config')):record_kwargs['config']=asdict(record_kwargs['config'])
  recorder.start(b.get_state(),backend_kwargs=record_kwargs,runtime_state=b.export_replay_state());b.recorder=recorder
  recorder.write('asset_model',files=[dict(sha256=compiled_model_hash(b),kind='compiled_native_model')])
 start_time=float(b.data.time);desired=controller.previous_tau.copy();swing=None;checks={};force=np.zeros(6);updates=0;peak_tau_error=0.;gap=0.;certificate=0.;valid_impulse=0.;invalid_impulse=0.;valid_peak=0.;nose_peak=0.;omega_peak=0.;max_tilt=0.;min_z=9.;minimum_error=np.inf;initial_error=None;reached_hover=False;stage_index=0;unexpected=False;minimum_xy=np.inf;minimum_normal_gap=np.inf;highest_front=-np.inf;initial_front=None;stop='candidate time budget';failure=None;max_cross=0;actual_steps=0
 try:
  for step in range(3500):
   elapsed=float(b.data.time)-start_time
   if step%5==0:
    swing,checks=stages.command(valid_impulse,nose_peak)
    if stages.phase!='raise_outside_board' and not unexpected:stage_index=max(stage_index,1)
    if stages.phase=='press_pulse' and not unexpected:reached_hover=True;stage_index=max(stage_index,2)
    if swing is None:stop=stages.reason;break
    fraction=min(elapsed/1.5,1.);smooth=10*fraction**3-15*fraction**4+6*fraction**5
    target=reference.copy();target[:2]=reference[:2]+smooth*(preload-reference[:2])
    controller.toe_force_target=parameters['pulse_peak_N']*max(stages.pulse_fraction,0.) if stages.phase=='press_pulse' else None
   kp=b.policy_values['kp'];kd=b.policy_values['kd'];q=b._robot_q();v=b._robot_dq();limits=b.policy_values['joint_effort_limit']
   lo=np.maximum(-limits,kp*(b.policy_values['joint_pos_lower_limit']-q)-kd*v);hi=np.minimum(limits,kp*(b.policy_values['joint_pos_upper_limit']-q)-kd*v)
   if step%5==0 or np.maximum(lo-desired,desired-hi).max()>1e-3:
    try:desired,diag=controller.solve(target,np.zeros(3),np.zeros(3),swing=swing,hand=None)
    except RuntimeError as error:stop='QP certificate rejected';failure=str(error);break
    updates+=1;gap=max(gap,diag.get('relative_dual_gap') or 0.);certificate=max(certificate,diag['kkt_physical_inf'])
    if recorder:recorder.write('control_update',time=float(b.data.time),phase=stages.phase,torque_desired_Nm=desired,qp=diag,search_parameters=parameters)
   controller.apply(desired);b.set_gripper_command(0.)
   if recorder:recorder.phase=stages.phase;recorder.desired=desired.copy();recorder.time_before=float(b.data.time)
   b.step(update_policy=False);actual_steps+=1
   tau_error=float(np.max(abs(b.data.qfrc_applied[b.robot_dof_adr]-desired)));peak_tau_error=max(peak_tau_error,tau_error)
   if np.any(b.data.qfrc_applied[:6]) or np.any(b.data.ctrl[:6]) or np.any(b.data.xfrc_applied):raise RuntimeError('Forbidden root or external board force')
   DR=b.data.xmat[deck].reshape(3,3);dp=b.data.xpos[deck];surface=dp+DR@np.array([-.36,0,.0175+.03*.6**2+.04])
   actual_toe=stages.actual()[0];front_heights=[float(b.data.geom_xpos[g,2]) for g in front];minimum_front=min(front_heights);highest_front=max(highest_front,minimum_front)
   if initial_front is None:initial_front=minimum_front
   stage_measurement=stages.measurements()
   if stage_measurement['all_foot_above_board']:minimum_xy=min(minimum_xy,float(np.linalg.norm(actual_toe[:2]-surface[:2])))
   if all(stage_measurement['front_inside']):minimum_normal_gap=min(minimum_normal_gap,max(min(stage_measurement['front_clearance_m']),0.))
   distance=float(np.linalg.norm(actual_toe-surface));minimum_error=min(minimum_error,distance)
   if initial_error is None:initial_error=distance
   tilt=float(np.arccos(np.clip(b.data.xmat[root].reshape(3,3)[2,2],-1,1)));root_z=float(b.data.xpos[root,2]);max_tilt=max(max_tilt,tilt);min_z=min(min_z,root_z)
   nose=dp+DR@np.array([.4,0,.03]);tail=dp+DR@np.array([-.4,0,.03]);angle=float(np.arctan2(nose[2]-tail[2],np.linalg.norm(nose[:2]-tail[:2])));nose_peak=max(nose_peak,angle)
   velocity=np.zeros(6);mujoco.mj_objectVelocity(b.model,b.data,mujoco.mjtObj.mjOBJ_BODY,deck,velocity,0);nose_omega=float(-velocity[:3]@DR[:,1]);omega_peak=max(omega_peak,nose_omega)
   topforce=0.;badforce=0.;cross=0;contact_rows=[]
   for i in range(b.data.ncon):
    contact=b.data.contact[i];pair={int(contact.geom1),int(contact.geom2)}
    if not(pair&board and pair&robot):continue
    mujoco.mj_contactForce(b.model,b.data,i,force)
    if force[0]<=1e-5:continue
    local=DR.T@(contact.pos-dp);foot_geom=next(iter(pair&front),None)
    sign=1. if foot_geom is not None and contact.geom2==foot_geom else -1.
    reaction=sign*contact.frame.reshape(3,3)[0]
    top=float(reaction@(DR@normal_local))>.5 and local[2]>=stages.top(local[0])-.005
    valid=foot_geom is not None and local[0]<=-.30 and abs(local[1])<.10 and top
    if valid and stage_index>=2 and not unexpected:topforce+=float(force[0])
    else:badforce+=float(force[0]);cross+=1
    if recorder:contact_rows.append(dict(geom_ids=[int(contact.geom1),int(contact.geom2)],body_pair=[b.model.body(int(b.model.geom_bodyid[g])).name for g in [contact.geom1,contact.geom2]],deck_local_m=local.tolist(),normal_N=float(force[0]),valid_top_toe=valid))
   if badforce>1e-5:unexpected=True
   valid_impulse+=topforce*.002;
   if valid_impulse>1e-5 and not unexpected:stage_index=max(stage_index,3)
   invalid_impulse+=badforce*.002;valid_peak=max(valid_peak,topforce);max_cross=max(max_cross,cross)
   if recorder:states.append(dict(time=float(b.data.time),phase=stages.phase,parameters=parameters,toe_world_m=actual_toe.tolist(),toe_goal_m=swing['pos'].tolist(),toe_error_m=float(np.linalg.norm(actual_toe-swing['pos'])),measurements=stages.measurements(),valid_top_force_N=topforce,invalid_robot_board_force_N=badforce,contacts=contact_rows,nose_angle_rad=angle,nose_omega_rad_s=nose_omega,root_tilt_rad=tilt,root_z_m=root_z,torque_execution_error_Nm=tau_error))
   if tau_error>1e-3:stop='original torque execution guard';break
   if tilt>.25 or root_z<.62:stop='original upright guard';break
   if not np.isfinite(b.data.qpos).all() or not np.isfinite(b.data.qvel).all():raise RuntimeError('Nonfinite physical state')
 except Exception as error:stop='runtime error';failure=repr(error)
 finally:
  if recorder:recorder.close();b.recorder=None
 events=dict(outside_clearance=stage_index>=1,toe_hover_captured=stage_index>=2,valid_top_contact=stage_index>=3,nose_above20deg=bool(nose_peak>np.deg2rad(20) and valid_impulse>0),end_grip=False,unsupported_hold_1s=False,placement=False)
 if stage_index==0:
  needed=cache['audit']['board_collision_top_world_m']+.03+.005-(initial_front or 0.)
  progress=float(np.clip((highest_front-(initial_front or 0.))/max(needed,1e-6),0,1))
 elif stage_index==1:progress=float(np.exp(-minimum_xy/.1)) if np.isfinite(minimum_xy) else 0.
 elif stage_index==2:progress=float(np.exp(-minimum_normal_gap/.03)) if np.isfinite(minimum_normal_gap) else 0.
 else:progress=float(np.clip(nose_peak/np.deg2rad(80),0,1))
 numerical_invalid=failure is not None
 components=dict(prefix_stage=2.*stage_index,progress_to_next_stage=progress,posture_guard_penalty=-.5 if stop=='original upright guard' else 0.,unexpected_board_contact_penalty=-.5 if unexpected else 0.)
 score=float(sum(components.values())) if not numerical_invalid else -100.
 result=dict(index=index,parameters=parameters,score=score,score_version='ordered_events_v1',geometry_guard_version=guard_version,press_control=press,stage_index=stage_index,numerical_invalid=numerical_invalid,reward_components=components,events=events,stop_reason=stop,failure=failure,actual_final_phase=stages.phase,simulated_control_steps=actual_steps,simulated_seconds=float(b.data.time)-start_time,wall_seconds=time.perf_counter()-start,qp_updates=updates,max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,min_actual_hover_error_m=float(minimum_error),valid_top_tail_impulse_Ns=valid_impulse,invalid_robot_board_impulse_Ns=invalid_impulse,peak_valid_top_force_N=valid_peak,max_nose_angle_rad=nose_peak,max_nose_omega_rad_s=omega_peak,max_root_tilt_rad=max_tilt,min_root_z_m=min_z,max_tau_execution_error_Nm=peak_tau_error,max_physical_stationarity=certificate,max_relative_LS_dual_gap=gap,hand_controller_executed=False)
 if record is not None:dump(Path(record)/'report.json',result);dump(Path(record)/'metrics.json',states)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--population',type=int,default=12);p.add_argument('--generations',type=int,default=3);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=7);args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter();cache,audit=prepare_prefix(LAYOUT);dump(args.output/'prefix-audit.json',audit)
 if cache is None:
  print(json.dumps(dict(stage='new-layout prefix guard failed; using old verified layout next',audit={k:v for k,v in audit.items() if k!='last025s'})),flush=True)
  cache,audit=prepare_prefix(None);dump(args.output/'prefix-audit-fallback.json',audit)
 if cache is None:raise RuntimeError('No actually valid physical prefix; no search run')
 LOW[0]=audit['board_collision_top_world_m']+.03+.005+.005;HIGH[0]=LOW[0]+.09;MEAN[0]=np.clip(.24,LOW[0],HIGH[0]);STD[0]=(HIGH[0]-LOW[0])*.2
 report=dict(score_version='ordered_events_v1',height_semantics='h_hover_world is two front sphere centre goal world Z(m); lower bound compiled board top + .03 clearance + .005 radius + .005 margin',method='CEM low-dimensional model-based trajectory control search; NOT PPO or a learned neural policy',seed=args.seed,layout=audit['layout'],population=args.population,generations=args.generations,workers=args.workers,parameter_names=NAMES,lower=LOW,upper=HIGH,initial_mean=MEAN,initial_std=STD,source_sha256=sha(__file__),native_sources={str(p.relative_to(SIM)):sha(p) for p in [SIM/'src/skate_sim/validation/balance.py',SIM/'src/skate_sim/validation/qp_solver.py',SIM/'src/skate_sim/validation/toe_stages.py']},prefix=audit,models=dict(robot_body_dof=29,gripper='official Robotiq2F85; unchanged',board='original standard; native hooks unchanged'),limitations='Pop-stage search; no hand/grasp controller is executed until genuine toe/top/nose stages are achieved. Component prefix cached only after actual new-layout full integration.')
 dump(args.output/'report.json',report);init_worker(cache)
 baseline=evaluate_candidate((0,MEAN),record=args.output/'baseline');print('SMOKE '+json.dumps(baseline,default=plain),flush=True)
 dump(args.output/'smoke.json',baseline)
 if args.smoke:return
 rng=np.random.default_rng(args.seed);mean=MEAN.copy();std=STD.copy();best=baseline;all_rows=[]
 context=mp.get_context('spawn');queue=context.Queue();probe=context.Process(target=probe_worker,args=(cache,queue));probe.start();probe.join(30)
 if probe.is_alive():probe.terminate();probe.join();raise RuntimeError('Single worker identity preflight timed out')
 receipt=queue.get(timeout=3) if probe.exitcode==0 else dict(ok=False,exitcode=probe.exitcode)
 report['single_spawn_identity']=receipt;dump(args.output/'report.json',report)
 if not receipt['ok']:raise RuntimeError(f'Single worker preflight failed: {receipt}')
 with context.Pool(args.workers,initializer=init_worker,initargs=(cache,)) as pool,open(args.output/'search.jsonl','a',buffering=1) as log:
  for generation in range(args.generations):
   vectors=np.clip(rng.normal(mean,std,(args.population,len(mean))),LOW,HIGH);vectors[0]=mean
   jobs=[(generation*args.population+i+1,vector) for i,vector in enumerate(vectors)]
   candidates=list(pool.imap(evaluate_candidate,jobs))
   for row in candidates:row['generation']=generation+1;log.write(json.dumps(row,default=plain)+'\n');all_rows.append(row)
   elite=sorted(candidates,key=lambda r:r['score'],reverse=True)[:3];values=np.array([[r['parameters'][name] for name in NAMES] for r in elite]);mean=.5*mean+.5*values.mean(axis=0);std=np.maximum(.5*std+.5*values.std(axis=0),(HIGH-LOW)*.05)
   winner=elite[0]
   if winner['score']>best['score']:best=winner
   summary=dict(generation=generation+1,mean=mean,std=std,best_score=best['score'],elite_indices=[r['index'] for r in elite],event_counts={name:sum(bool(r['events'].get(name)) for r in candidates) for name in baseline['events']},median_wall_seconds=float(np.median([r['wall_seconds'] for r in candidates])))
   log.write(json.dumps(dict(record_type='generation',**summary),default=plain)+'\n');print('GENERATION '+json.dumps(summary,default=plain),flush=True)
 report.update(best=best,final_mean=mean,final_std=std,total_search_candidates=len(all_rows),search_wall_seconds=time.perf_counter()-start)
 # A deterministic recorded continuation for audit. Full t0 verification is a
 # separate required step and is not claimed by this component replay.
 best_result=evaluate_candidate((best['index'],np.array([best['parameters'][n] for n in NAMES])),record=args.output/'best')
 report['best_recorded_continuation']=best_result;report['full_static_reproduction']='pending independent t0 validation';dump(args.output/'report.json',report);print('FINAL '+json.dumps(best_result,default=plain),flush=True)
if __name__=='__main__':main()
