"""RSL-RL PPO on8 real-body torque residuals; native Robotiq hover lesson."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,copy,json,multiprocessing as mp,sys,time,traceback,resource
from pathlib import Path
import numpy as np
import skate_skill_search as skill
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from skate_sim.validation.balance import SupportGeometryGuard
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tqdm import tqdm

AXES=['left_hip_pitch_joint','left_hip_roll_joint','left_knee_joint','right_hip_pitch_joint','right_hip_roll_joint','right_knee_joint','right_ankle_roll_joint','waist_roll_joint']
PARAMS=dict(h_hover_world=.23029645229701157,translate_time=1.2420239170913996,cop_preload=.007525448672422479,pulse_peak_N=144.019968286459,pulse_time=.2173205844417036,yaw_offset=-.0011302674115758297,geometry_guard_version='v2')
PHASES=['raise_outside_board','translate_above_tail','capture_hover','hover_control']

class HoverResidual:
 def __init__(self,cache,backend=None,restore_state=True):
  self.cache=cache;self.hand_target=None
  if backend is None:
   with skill.contextlib.redirect_stdout(skill.io.StringIO()):self.b=skill.create_backend('mujoco',**cache['kwargs'])
  else:self.b=backend
  if skill.binary_model_hash(self.b)!=cache['audit']['model_binary_sha256']:raise RuntimeError('PPO worker exact MJB identity mismatch')
  self.axis=np.array([self.b.robot_joint_names.index(n) for n in AXES]);self.joints=np.array([self.b.model.joint(n).id for n in AXES]);self.root=self.b.model.body('pelvis').id;self.deck=self.b.model.body('deck').id
  self.feet=[set(skill.body_geoms(self.b,s+'_ankle_roll_link')) for s in ['left','right']];self.board=set(skill.body_geoms(self.b,'chassis'));self.robot=set(skill.body_geoms(self.b,'pelvis'));self.ground=self.b.model.geom('ground').id
  effort=self.b.policy_values['joint_effort_limit'][self.axis];hardware=np.min(abs(self.b.model.jnt_actfrcrange[self.joints]),axis=1)
  if not np.all(self.b.model.jnt_actfrclimited[self.joints]):raise RuntimeError('Residual action hardware effort not explicitly limited')
  self.scale=.12*np.minimum(effort,hardware);self.force=np.zeros(6);self.reset(restore_state)
 def reset(self,restore_state=True):
  b=self.b
  if restore_state:b.set_state(self.cache['state']);b.import_replay_state(self.cache['runtime'])
  elif np.max(abs(np.asarray(b.export_replay_state()['integration'])-np.asarray(self.cache['runtime']['integration'])))>1e-12:raise RuntimeError('Live physical prefix differs from audited integration boundary')
  self.control=skill.ContactBalance(b);self.control.__dict__.update(copy.deepcopy(self.cache['controller']));self.control.b=b;self.control.m=b.model;self.control.numeric_backend='clarabel_lifted_ls';self.control.allow_board_support=False;self.control.toe_force_target=None
  DR=b.data.xmat[self.deck].reshape(3,3);p=b.data.xpos[self.deck]
  from skate_sim.models.board import board_styles
  selected=board_styles(b.scene_config['style'])[b.scene_config['style']];self.tail_x=-.45*selected.deck_length
  goal=p+DR@np.array([self.tail_x,0,.23-p[2]]);R,_=skill.toe_reference(b,goal,prefer_flat=True)
  self.stages=skill.ToeStages(b,False,R,PARAMS);self.swing,self.check=self.stages.command();self.base=self.control.previous_tau.copy();self.previous=np.zeros(8);self.raw_previous=np.zeros(8);self.steps=0;self.ticks=0;self.total_return=0.;self.terms_total={};self.hold=0.;self.max_hold=0.;self.clear_awarded=False;self.max_stage=0;self.episode_start=float(b.data.time);self.solved=0
  self.reference=b.data.subtree_com[self.root].copy();target=goal[:2]-self.reference[:2];target/=max(np.linalg.norm(target),1e-12);hull=skill.inset_support(self.cache['right_xy'],.01);self.preload=skill.project_support(self.reference[:2]+PARAMS['cop_preload']*target,hull)
  self.prev_distance=self.distance();self.last_delivery=np.zeros(8);self.last_clip=np.zeros(8);return self.observation()
 def distance(self):
  b=self.b;R=b.data.xmat[self.deck].reshape(3,3);goal=b.data.xpos[self.deck]+R@np.array([self.tail_x,0,.0175+.03*.6**2+.005+.04]);return float(np.linalg.norm(self.stages.actual()[0]-goal))
 def loads(self):
  result=np.zeros(2)
  for i in range(self.b.data.ncon):
   c=self.b.data.contact[i];pair={int(c.geom1),int(c.geom2)}
   if self.ground not in pair:continue
   mujoco.mj_contactForce(self.b.model,self.b.data,i,self.force)
   for side in [0,1]:
    if pair&self.feet[side]:result[side]+=max(float(self.force[0]),0.)
  return result
 def observation(self):
  b=self.b;d=b.data;m=b.model;R=d.xmat[self.root].reshape(3,3);DR=d.xmat[self.deck].reshape(3,3);p,footR=self.stages.actual();jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,p,self.stages.foot);velocity=jp@d.qvel
  J=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,J,self.root);cv=J@d.qvel;com=d.subtree_com[self.root];centre=self.cache['right_xy'].mean(axis=0);omega=np.sqrt(9.81/max(com[2],.1));xi=com[:2]+cv[:2]/omega
  rootv=np.zeros(6);boardv=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.root,rootv,0);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.deck,boardv,0)
  phase=np.zeros(4);phase[PHASES.index(self.stages.phase) if self.stages.phase in PHASES else 3]=1
  check=self.stages.measurements();goal=self.swing['pos'] if self.swing is not None else p;vgoal=self.swing['vel'] if self.swing is not None else np.zeros(3)
  footerr=Rotation.from_matrix((self.swing['rotation'] if self.swing is not None else footR).T@footR).as_rotvec()
  observation=np.r_[b._robot_q()-self.control.q0,b._robot_dq()/10,rootv[:3]/3,R.T@np.array([0,0,-1.]),self.previous,
    R.T@(d.xpos[self.deck]-d.xpos[self.root]),Rotation.from_matrix(R.T@DR).as_rotvec(),boardv/3,
    com-np.r_[centre,0.],cv[:2],xi-centre,self.loads()/self.control.mg,
    p-goal,(velocity-vgoal)/2,footerr,check['foot_normal_dot_tail'],np.clip(check['front_clearance_m'],-.2,.5),phase,
    R.T@(goal-d.xpos[self.root]),min((float(d.time)-self.episode_start)/6.,1.)]
  if not np.isfinite(observation).all():raise RuntimeError('Nonfinite physical observation')
  return observation.astype(np.float32)
 def step(self,raw,record=None):
  raw=np.asarray(raw,float)
  if raw.shape!=(8,) or not np.isfinite(raw).all():raise ValueError('Residual policy action must be finite8')
  action=np.tanh(raw);b=self.b;terms={k:0. for k in ['xy_tracking','z_tracking','clearance_progress','upright','action_cost','action_rate_cost','clearance_event','hover_event','failure']};reason=None;numeric=False;success=False;executed=0;max_error=0.;delivery=np.zeros(8);clips=np.zeros(8);rawtrace=[]
  for sub in range(10):
   if self.ticks%5==0:
    self.swing,self.check=self.stages.command()
    if self.swing is None:reason=self.stages.reason;break
   kp=b.policy_values['kp'];kd=b.policy_values['kd'];q=b._robot_q();dq=b._robot_dq();limit=b.policy_values['joint_effort_limit']
   lo=np.maximum(-limit,kp*(b.policy_values['joint_pos_lower_limit']-q)-kd*dq);hi=np.minimum(limit,kp*(b.policy_values['joint_pos_upper_limit']-q)-kd*dq)
   if self.ticks%5==0 or np.maximum(lo-self.base,self.base-hi).max()>1e-3:
    elapsed=float(b.data.time)-self.episode_start;s=min(elapsed/1.5,1.);s=10*s**3-15*s**4+6*s**5;target=self.reference.copy();target[:2]+=s*(self.preload-self.reference[:2])
    try:self.base,diag=self.control.solve(target,np.zeros(3),np.zeros(3),swing=self.swing,hand=self.hand_target)
    except SupportGeometryGuard as error:reason='measured support geometry guard: '+str(error);break
    except RuntimeError as error:reason=str(error);numeric=True;break
    self.solved+=1
   headroom=np.where(action>=0,hi[self.axis]-self.base[self.axis],self.base[self.axis]-lo[self.axis]);available=np.minimum(self.scale,np.maximum(headroom,0.));residual=action*available;total=self.base.copy();total[self.axis]+=residual
   total,arm_detail=self.control.compensate_arm(total,self.hand_target);self.control.apply(total);b.set_gripper_command(0.)
   if record:record.desired=total.copy();record.phase='PPO '+self.stages.phase;record.time_before=float(b.data.time)
   before=float(b.data.time);b.step(update_policy=False);self.ticks+=1;executed+=1
   actual=b.data.qfrc_applied[b.robot_dof_adr];error=float(np.max(abs(actual-total)));max_error=max(max_error,error);delivery+=residual;clips+=(available<self.scale-1e-9)
   if record:record.write('residual_control',time_before=before,time_after=float(b.data.time),raw_policy8=raw,tanh_policy8=action,axis_names=AXES,baseline_tau29=self.base,arm_compensation=arm_detail,requested_residual8_Nm=action*self.scale,residual8_Nm=residual,total_tau29=total,actual_tau29=actual,available_headroom8_Nm=available,action_scale8_Nm=self.scale,torque_error_inf_Nm=error)
   if error>1e-3:reason='original actual torque execution guard';break
   if np.any(b.data.qfrc_applied[:6]) or np.any(b.data.ctrl[:6]) or np.any(b.data.xfrc_applied):raise RuntimeError('Forbidden base/board external force')
   tilt=float(np.arccos(np.clip(b.data.xmat[self.root].reshape(3,3)[2,2],-1,1)));z=float(b.data.xpos[self.root,2]);check=self.stages.measurements();p=self.stages.actual()[0];DR=b.data.xmat[self.deck].reshape(3,3);goal=b.data.xpos[self.deck]+DR@np.array([self.tail_x,0,.19-b.data.xpos[self.deck,2]])
   xy=float(np.linalg.norm(p[:2]-goal[:2]));ze=float(p[2]-goal[2]);clear=float(np.clip(min(check['front_clearance_m'])/.03,0,1))
   terms['xy_tracking']+=.002*1.5*np.exp(-xy*xy/.1**2);terms['z_tracking']+=.002*1.5*np.exp(-ze*ze/.06**2);terms['clearance_progress']+=.002*.5*clear;terms['upright']+=.002*.5*np.exp(-tilt*tilt/.12**2);terms['action_cost']-=.002*.02*np.mean(action**2);terms['action_rate_cost']-=.002*.03*np.mean((action-self.previous)**2)
   actual_hover=all(check['front_inside']) and min(check['front_clearance_m'])>=.03 and check['foot_normal_dot_tail']>=0 and check['whole_foot_separated']
   self.hold=self.hold+.002 if actual_hover else 0.;self.max_hold=max(self.max_hold,self.hold)
   if self.stages.phase!='raise_outside_board' and not self.clear_awarded:terms['clearance_event']+=1.;self.clear_awarded=True;self.max_stage=1
   if self.hold>=.15:terms['hover_event']+=2.;success=True;self.max_stage=2;reason='actual hover held150ms';break
   bad=False
   for i in range(b.data.ncon):
    contact=b.data.contact[i];pair={int(contact.geom1),int(contact.geom2)}
    if pair&self.board and pair&self.robot:
     mujoco.mj_contactForce(b.model,b.data,i,self.force)
     if self.force[0]>1e-5:bad=True;break
   if bad:reason='unexpected robot-board contact in hover lesson';break
   if tilt>.25 or z<.62:reason='original upright guard';break
   if self.stages.phase!='raise_outside_board' and not check['whole_foot_separated']:reason='actual30mm whole-foot clearance guard';break
  self.steps+=1;self.last_delivery=delivery/max(executed,1);self.last_clip=clips/max(executed,1);self.previous=action.copy();self.raw_previous=raw.copy()
  if self.steps>=300 and reason is None:reason='bounded hover task deadline'
  if reason and not success and not numeric:terms['failure']-=1.
  reward=float(sum(terms.values()));self.total_return+=reward
  for key,value in terms.items():self.terms_total[key]=self.terms_total.get(key,0.)+value
  info=dict(reward_terms=terms,reason=reason,numerical_invalid=numeric,success=success,stage=self.max_stage,max_hover_seconds=self.max_hold,torque_error_inf_Nm=max_error,mean_residual8_Nm=self.last_delivery.tolist(),headroom_limited_fraction8=self.last_clip.tolist(),physics_ticks=executed)
  done=reason is not None
  if done:info['episode']=dict(return_sum=self.total_return,policy_steps=self.steps,physics_steps=self.ticks,success=success,reason=reason,numerical_invalid=numeric,reward_terms=self.terms_total.copy(),max_hover_seconds=self.max_hold,stage=self.max_stage)
  obs=self.observation()
  return obs,reward,done,info
 def close(self):self.b.close()


def worker(connection,cache):
 try:
  env=HoverResidual(cache);connection.send(('ready',env.observation()))
  while True:
   command,payload=connection.recv()
   if command=='close':break
   if command=='reset':connection.send(('ok',env.reset()));continue
   if command=='step':
    obs,reward,done,info=env.step(payload)
    if done:obs=env.reset()
    connection.send(('ok',(obs,reward,done,info)))
 except Exception:connection.send(('error',traceback.format_exc()))
 finally:
  if 'env' in locals():env.close()
  connection.close()

class Parallel:
 def __init__(self,cache,num=4):
  self.connections=[];self.processes=[];ctx=mp.get_context('spawn')
  for _ in range(num):
   a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,cache));p.start();b.close();self.connections.append(a);self.processes.append(p)
  self.obs=np.stack([self.receive(c) for c in self.connections])
 def receive(self,c):
  status,data=c.recv()
  if status=='error':raise RuntimeError(data)
  return data
 def step(self,actions):
  for c,a in zip(self.connections,actions):c.send(('step',a))
  outputs=[self.receive(c) for c in self.connections];self.obs=np.stack([x[0] for x in outputs]);return self.obs,np.array([x[1] for x in outputs],np.float32),np.array([x[2] for x in outputs],bool),[x[3] for x in outputs]
 def close(self):
  for c in self.connections:
   try:c.send(('close',None))
   except (BrokenPipeError,EOFError):pass
  for p in self.processes:
   p.join(10)
   if p.is_alive():p.terminate();p.join()
  for c in self.connections:c.close()

def sensitivity(cache,out):
 env=HoverResidual(cache);zero=env.reset();_,_,_,zinfo=env.step(np.zeros(8));baseline=env.b.get_state();rows=[]
 for i,name in enumerate(AXES):
  branches=[]
  for sign in [-1.,1.]:
   env.reset();action=np.zeros(8);action[i]=sign*.5;_,reward,done,info=env.step(action);state=env.b.get_state();p,footR=env.stages.actual();J,_,D,_=env.control.jac(env.b.data,p,env.stages.foot);extra=dict(toe_p=p.copy(),toe_v=J@env.b.data.qvel,toe_acc=J@env.b.data.qacc+D@env.b.data.qvel,root_R=env.b.data.xmat[env.root].copy());branches.append((state,info,extra))
  minus,plus=branches;row=dict(toe_position_plus_minus_m=float(np.linalg.norm(plus[2]['toe_p']-minus[2]['toe_p'])),toe_velocity_plus_minus_m_s=float(np.linalg.norm(plus[2]['toe_v']-minus[2]['toe_v'])),toe_acceleration_plus_minus_m_s2=float(np.linalg.norm(plus[2]['toe_acc']-minus[2]['toe_acc'])),root_orientation_matrix_plus_minus_inf=float(np.max(abs(plus[2]['root_R']-minus[2]['root_R']))),joint=name,raw_probe=.5,mean_actual_residual_minus_Nm=minus[1]['mean_residual8_Nm'][i],mean_actual_residual_plus_Nm=plus[1]['mean_residual8_Nm'][i],qpos_plus_minus_inf=float(np.max(abs(np.asarray(plus[0]['qpos'])-np.asarray(minus[0]['qpos'])))),qvel_plus_minus_inf=float(np.max(abs(np.asarray(plus[0]['qvel'])-np.asarray(minus[0]['qvel'])))),tau_execution_error_max_Nm=max(minus[1]['torque_error_inf_Nm'],plus[1]['torque_error_inf_Nm']),headroom_limited_minus=minus[1]['headroom_limited_fraction8'][i],headroom_limited_plus=plus[1]['headroom_limited_fraction8'][i]);rows.append(row)
 env.close();skill.dump(out/'action-probe.json',dict(axis_order=AXES,branches=rows,action_mapping='raw Gaussian -> tanh -> asymmetric12% hardware/PD headroom; actual29torque only'))
 print('ACTION_PROBE '+json.dumps(rows),flush=True);return rows

def td(array):return TensorDict({'policy':torch.as_tensor(array,dtype=torch.float32)},batch_size=[len(array)])
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--updates',type=int,default=5);parser.add_argument('--seed',type=int,default=7);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
 cache,audit=skill.prepare_prefix(skill.LAYOUT)
 if cache is None:raise RuntimeError('Fresh-layout actual single-support prefix invalid; not starting PPO')
 skill.dump(args.output/'prefix-audit.json',audit);probes=sensitivity(cache,args.output)
 if any(r['qvel_plus_minus_inf']<=1e-10 or max(abs(r['mean_actual_residual_minus_Nm']),abs(r['mean_actual_residual_plus_Nm']))<=1e-6 for r in probes):raise RuntimeError('Residual axis ineffective; not starting learning')
 envs=Parallel(cache);obs=td(envs.obs);groups={'actor':['policy'],'critic':['policy']}
 actor=MLPModel(obs,groups,'actor',8,hidden_dims=[128,128],activation='elu',obs_normalization=False,distribution_cfg=dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',init_std=.1,std_range=(.01,.3),std_type='log'))
 final_linear=[module for module in actor.mlp.modules() if isinstance(module,torch.nn.Linear)][-1];torch.nn.init.zeros_(final_linear.weight);torch.nn.init.zeros_(final_linear.bias)
 critic=MLPModel(obs,groups,'critic',1,hidden_dims=[128,128],activation='elu',obs_normalization=False)
 storage=RolloutStorage('rl',4,64,obs,(8,),device='cpu');algorithm=PPO(actor,critic,storage,num_learning_epochs=4,num_mini_batches=4,clip_param=.2,gamma=.99,lam=.95,value_loss_coef=.5,entropy_coef=.005,learning_rate=3e-4,max_grad_norm=.5,schedule='fixed',desired_kl=None,device='cpu')
 initial=torch.cat([p.detach().flatten().clone() for p in actor.parameters()]);identity=dict(observation_normalization='fixed physical scales; no online running normalization',initial_residual_policy='zero output mean, Gaussian std.1; tanh; hardware scale unchanged',method='rsl_rl5.4.2 PPO real native29 plus actual Robotiq residual hover lesson',seed=args.seed,num_envs=4,steps_per_env=64,updates=args.updates,actor_critic_hidden=[128,128],action_axes=AXES,raw_action='Gaussian8; environment tanh; deliveredNm≤12%effort and current asymmetric PD/headroom; baseline plusresidual actualqfrc only',model_binary_sha256=audit['model_binary_sha256'],source_sha256=skill.sha(__file__),baseline_parameters=PARAMS,physics_backend='native MuJoCo3.11,500Hz; WBC100Hz plus PD-boundary refresh; residual policy50Hz',learning=dict(gamma=.99,lam=.95,clip=.2,lr=.0003,epochs=4,minibatches=4,value_coefficient=.5,entropy_coefficient=.005,grad_clip=.5),no_success_claim=True)
 skill.dump(args.output/'report.json',identity);torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),config=identity),args.output/'initial.pt')
 start=time.perf_counter();transitions=0
 try:
  progress=tqdm(range(args.updates),desc='residual PPO updates',disable=not sys.stdout.isatty())
  with open(args.output/'updates.jsonl','a',buffering=1) as logfile:
   for update in progress:
    learner_boundary=dict(actor=copy.deepcopy(actor.state_dict()),critic=copy.deepcopy(critic.state_dict()),optimizer=copy.deepcopy(algorithm.optimizer.state_dict()),rng=torch.get_rng_state().clone())
    summaries=[];rewards=[]
    for step in range(64):
     with torch.no_grad():actions=algorithm.act(obs)
     array,reward,done,infos=envs.step(actions.numpy());next_obs=td(array)
     if any(info['numerical_invalid'] for info in infos):
      actor.load_state_dict(learner_boundary['actor']);critic.load_state_dict(learner_boundary['critic']);algorithm.optimizer.load_state_dict(learner_boundary['optimizer']);torch.set_rng_state(learner_boundary['rng']);storage.clear()
      skill.dump(args.output/'numeric-stop.json',dict(update=update+1,learner_rollback=True,discarded_entire_rollout=True,invalid_transition_not_added=True,infos=infos));raise RuntimeError('Numerically invalid rollout discarded BEFORE PPO update; learner restored')
     algorithm.process_env_step(next_obs,torch.from_numpy(reward),torch.from_numpy(done),{})
     summaries.extend(infos);rewards.extend(reward.tolist());obs=next_obs;transitions+=4
    with torch.no_grad():algorithm.compute_returns(obs)
    old_obs=storage.observations.clone().reshape(-1);old_actions=storage.actions.reshape(-1,8).clone();old_logs=storage.actions_log_prob.reshape(-1).clone();old_params=tuple(p.reshape(-1,p.shape[-1]).clone() for p in storage.distribution_params)
    losses=algorithm.update()
    with torch.no_grad():
     actor(old_obs,stochastic_output=True);newlogs=actor.get_output_log_prob(old_actions).flatten();ratio=torch.exp(newlogs-old_logs);clip_fraction=float((abs(ratio-1)>.2).float().mean());kl=float(actor.get_kl_divergence(old_params,actor.output_distribution_params).mean())
    parameters=torch.cat([p.detach().flatten() for p in actor.parameters()]);change=float(torch.linalg.norm(parameters-initial));finite=all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in list(actor.parameters())+list(critic.parameters()))
    episodes=[r['episode'] for r in summaries if 'episode' in r];terms={key:float(np.mean([r['reward_terms'][key] for r in summaries])) for key in summaries[0]['reward_terms']}
    row=dict(update=update+1,policy_transitions=transitions,physics_steps=sum(r['physics_ticks'] for r in summaries),wall_seconds=time.perf_counter()-start,losses=losses,mean_step_reward=float(np.mean(rewards)),reward_terms_mean=terms,episodes=episodes,completed_hover_count=sum(r['success'] for r in episodes),max_hover_seconds=max(r['max_hover_seconds'] for r in summaries),numeric_invalid_count=sum(r['numerical_invalid'] for r in summaries),approx_kl=kl,clip_fraction=clip_fraction,actor_parameter_change_l2=change,finite=bool(finite),max_actual_tau_error_Nm=max(r['torque_error_inf_Nm'] for r in summaries),headroom_limited_fraction8=np.mean([r['headroom_limited_fraction8'] for r in summaries],axis=0).tolist())
    logfile.write(json.dumps(row,default=skill.plain)+'\n');tqdm.write('PPO_UPDATE '+json.dumps(row,default=skill.plain));progress.set_postfix(reward=row['mean_step_reward'],hover=row['completed_hover_count'])
    if not finite or row['numeric_invalid_count']:raise RuntimeError('PPO finite/numerical validity smoke failed; stopped honestly')
    torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=algorithm.optimizer.state_dict(),config=identity,update=update+1,policy_transitions=transitions,torch_rng=torch.get_rng_state()),args.output/'latest.pt')
  identity.update(status='short run complete',policy_transitions=transitions,wall_seconds=time.perf_counter()-start,actor_parameter_change_l2=change);skill.dump(args.output/'report.json',identity)
 finally:envs.close()
if __name__=='__main__':main()
