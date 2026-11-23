"""Batched control/observation boundary for external 29-DoF learners.

The learner and replay buffer live outside skate-sim. Latents are never part of
this environment's observations. All returned tensors own their storage.
"""
from __future__ import annotations

import numpy as np
import torch
import mujoco

from .robot.contract import G1_29_JOINTS
from .backends.base import validate_env_ids
from .models.scene import normalize_scene_config
from .record import compiled_model_hash

# Official 30 physical bodies (torso is index 15), followed by a virtual head.
BFM_BODIES = ("pelvis",) + tuple(
    "torso_link" if name == "waist_pitch" else name + "_link"
    for name in G1_29_JOINTS
)
AUX_REWARD_KEYS = (
    "penalty_torques", "penalty_action_rate", "limits_dof_pos", "limits_torque",
    "penalty_undesired_contact", "penalty_feet_ori", "penalty_ankle_roll", "penalty_slippage",
)


def rotate(q, v):
    v = torch.broadcast_to(v, q.shape[:-1] + (3,))
    u = q[..., 1:]
    return v + 2 * q[..., :1] * torch.cross(u, v, dim=-1) + 2 * torch.cross(u, torch.cross(u, v, dim=-1), dim=-1)


def multiply(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((w, v), -1)


def privileged_state(pos, quat, vel, omega):
    """Official max-local-self state, including the torso-offset virtual head."""
    offset = rotate(quat[:, 15], pos.new_tensor((0., 0., .35)))
    pos = torch.cat((pos, (pos[:, 15] + offset)[:, None]), 1)
    quat = torch.cat((quat, quat[:, 15:16]), 1)
    vel = torch.cat((vel, (vel[:, 15] + torch.cross(omega[:, 15], offset, dim=-1))[:, None]), 1)
    omega = torch.cat((omega, omega[:, 15:16]), 1)
    forward = rotate(quat[:, 0], pos.new_tensor((1., 0., 0.)))
    angle = -.5 * torch.atan2(forward[:, 1], forward[:, 0])
    heading = torch.zeros_like(quat[:, 0]); heading[:, 0] = angle.cos(); heading[:, 3] = angle.sin()
    heading = heading[:, None].expand_as(quat)
    localq = multiply(heading, quat)
    tangent = rotate(localq, pos.new_tensor((1., 0., 0.)))
    normal = rotate(localq, pos.new_tensor((0., 0., 1.)))
    return torch.cat((pos[:, 0, 2:3], rotate(heading, pos - pos[:, :1])[:, 1:].flatten(1),
                      torch.cat((tangent, normal), -1).flatten(1), rotate(heading, vel).flatten(1),
                      rotate(heading, omega).flatten(1)), -1)


class SkateEnv:
    """One backend, N worlds. reset never accepts legacy HUSKY raw states."""
    def __init__(self, *, backend="mujoco", num_envs=1, preset="steer_start", seed=0, scene_config=None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        scene_config = normalize_scene_config({"preset":preset} if scene_config is None else scene_config)
        if not scene_config["robot"] or not scene_config["board"]:
            raise ValueError("training requires both robot and board in scene_config")
        preset = scene_config["preset"]
        if backend == "mujoco":
            if num_envs != 1:
                raise ValueError("MuJoCo training supports one world; use mjlab for batches")
            from .backends.mujoco import MujocoBackend
            self.backend = MujocoBackend(scene="rider", controller="pose", scene_config=scene_config, seed=seed)
            self.model = self.backend.model
            self.device = torch.device("cpu")
        elif backend == "mjlab":
            from .backends.mjlab import MjlabBackend
            self.backend = MjlabBackend(scene="rider", controller="pose", scene_config=scene_config, seed=seed, num_envs=num_envs)
            self.model = self.backend.sim.mj_model
            self.device = self.backend.sim.data.qpos._tensor.device
        else:
            raise ValueError(f"unknown backend {backend!r}")
        try:
            self.num_envs = num_envs
            self.preset = preset
            self.seed = seed
            self.body_ids = [self.model.body(name).id for name in BFM_BODIES]
            self.foot_ids = [self.model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
            self.foot_bodies = {}
            for foot in self.foot_ids:
                descendants=[]
                for body in range(self.model.nbody):
                    ancestor=body
                    while ancestor and ancestor != foot:
                        ancestor=int(self.model.body_parentid[ancestor])
                    if ancestor == foot: descendants.append(body)
                self.foot_bodies[foot]=torch.as_tensor(descendants,device=self.device)
            self.deck = self.model.body("deck").id
            self.root = self.model.body("pelvis").id
            self.values = {k: self.tensor(v) for k, v in self.backend.policy_values.items()}
            self.history = {k: torch.zeros((num_envs, 4, n), device=self.device) for k,n in
                            (("actions",29), ("base_ang_vel",3), ("dof_pos",29), ("dof_vel",29), ("projected_gravity",3))}
            self.last_action = torch.zeros((num_envs,29), device=self.device)
            self.steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
            self._observation = None
            self._done = torch.ones(num_envs, device=self.device, dtype=torch.bool)
            self.model_sha256 = compiled_model_hash(self.backend)
            self.reset()
        except BaseException:
            self.backend.close()
            raise

    def tensor(self, value):
        return torch.as_tensor(np.asarray(value).copy(), device=self.device, dtype=torch.float32)

    def _fields(self):
        b = self.backend
        if self.device.type == "cuda":
            d = b.sim.data
            pos, quat, cvel = d.xpos._tensor, d.xquat._tensor, d.cvel._tensor
            roots = torch.as_tensor(self.model.body_rootid, device=self.device, dtype=torch.long)
            com = d.subtree_com._tensor[:, roots]
            omega = cvel[..., :3]
            vel = cvel[..., 3:] + torch.cross(omega, pos - com, dim=-1)
            q = d.qpos._tensor[:, b.robot_qpos_index]
            dq = d.qvel._tensor[:, b.robot_dof_index]
            return q, dq, pos, quat, vel, omega
        d = b.data
        spatial = np.empty((self.model.nbody, 6))
        for i in range(self.model.nbody):
            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, i, spatial[i], 0)
        return (self.tensor(b._robot_q()[None]), self.tensor(b._robot_dq()[None]), self.tensor(d.xpos[None]),
                self.tensor(d.xquat[None]), self.tensor(spatial[None, :, 3:]), self.tensor(spatial[None, :, :3]))

    def _observe(self, fields):
        q,dq,pos,quat,vel,omega = fields
        inv = quat[:,self.root].clone(); inv[:,1:] *= -1
        gravity = rotate(inv, q.new_tensor((0.,0.,-1.)))
        angvel = .25 * rotate(inv, omega[:,self.root])
        relq = q - self.values["default"]
        obs = {"state":torch.cat((relq,dq,gravity,angvel),-1),
               "privileged_state":privileged_state(pos[:,self.body_ids],quat[:,self.body_ids],vel[:,self.body_ids],omega[:,self.body_ids]),
               "last_action":self.last_action * self.values["action_rescale"],
               "history_actor":torch.cat([self.history[k].flatten(1) for k in sorted(self.history)],-1)}
        current = {"actions":obs["last_action"], "base_ang_vel":angvel,"dof_pos":relq,"dof_vel":dq,"projected_gravity":gravity}
        for k,v in current.items():
            self.history[k] = torch.cat((v[:,None],self.history[k][:,:-1]),1)
        return {k:v.to(torch.float32).clone() for k,v in obs.items()}

    def reset(self, *, env_ids=None, seed=None):
        raw_ids = validate_env_ids(env_ids, self.num_envs)
        ids = torch.as_tensor(raw_ids,device=self.device,dtype=torch.long)
        self.backend.reset(seed=seed, env_ids=None if env_ids is None else raw_ids)
        for v in self.history.values(): v[ids] = 0
        self.last_action[ids] = 0; self.steps[ids] = 0; self._done[ids] = False
        # Observe only reset worlds without advancing histories in active worlds.
        old_history = {k:v.clone() for k,v in self.history.items()}
        fields = self._fields()
        obs = self._observe(fields)
        active = torch.ones(self.num_envs,device=self.device,dtype=torch.bool); active[ids]=False
        for k in self.history: self.history[k][active] = old_history[k][active]
        if self._observation is None: self._observation = obs
        else:
            for k in obs: self._observation[k][ids] = obs[k][ids]
        return {k:v.clone() for k,v in self._observation.items()}

    def _contacts(self):
        b = self.backend
        if self.device.type == "cpu":
            d=b.data
            force=np.empty((d.ncon,6))
            for i in range(d.ncon): mujoco.mj_contactForce(self.model,d,i,force[i])
            return (torch.zeros(d.ncon,device=self.device,dtype=torch.long),
                    torch.as_tensor(d.contact.geom.copy(),device=self.device,dtype=torch.long),
                    self.tensor(d.contact.pos),self.tensor(force[:,:3]),self.tensor(d.contact.frame.reshape(-1,3,3)[:,0]))
        import warp as wp
        import mujoco_warp as mw
        d=b.sim.wp_data
        if not hasattr(self,"_contact_ids"):
            self._contact_ids = wp.array(np.arange(d.naconmax,dtype=np.int32),device=b.sim.wp_device)
            self._contact_force = wp.zeros(d.naconmax,dtype=wp.spatial_vector,device=b.sim.wp_device)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            mw.contact_force(b.sim.wp_model,d,self._contact_ids,False,self._contact_force)
        n = wp.to_torch(d.nacon)[0]
        valid = torch.arange(d.naconmax,device=self.device) < n
        return (torch.where(valid,wp.to_torch(d.contact.worldid),0).long(),
                torch.where(valid[:,None],wp.to_torch(d.contact.geom),0).long(),
                wp.to_torch(d.contact.pos),torch.where(valid[:,None],wp.to_torch(self._contact_force)[:,:3],0.),wp.to_torch(d.contact.frame)[:,0])

    def _rewards(self, fields, previous):
        q,dq,pos,quat,vel,omega=fields
        b=self.backend
        tau=b._external_tau if self.device.type == "cuda" else self.tensor(b.last_tau[None])
        lower,upper=self.values["joint_pos_lower_limit"],self.values["joint_pos_upper_limit"]
        mid=(lower+upper)/2; radius=.95*(upper-lower)/2
        rewards={"penalty_torques":tau.square().sum(-1),"penalty_action_rate":(self.last_action-previous).square().sum(-1),
                 "limits_dof_pos":((mid-radius-q).clamp_min(0)+(q-mid-radius).clamp_min(0)).sum(-1),
                 "limits_torque":(tau.abs()-.95*self.values["joint_effort_limit"]).clamp_min(0).sum(-1),
                 "penalty_ankle_roll":q[:,[G1_29_JOINTS.index(s+"_ankle_roll") for s in ("left","right")]].square().sum(-1)}
        world,geoms,point,force,contact_normal=self._contacts()
        geom_body=torch.as_tensor(self.model.geom_bodyid,device=self.device,dtype=torch.long)
        bodies=geom_body[geoms]
        board_bodies=torch.as_tensor(np.flatnonzero(self.model.body_rootid==self.model.body_rootid[self.deck]),device=self.device)
        touches_board=torch.isin(bodies,board_bodies).any(-1)
        welded=torch.as_tensor(self.model.body_weldid,device=self.device,dtype=torch.long)
        surface=touches_board | (welded[bodies]==0).any(-1)
        penalized=torch.as_tensor([i for i in self.body_ids if any(t in self.model.body(i).name for t in ("pelvis","shoulder","hip"))],device=self.device)
        undesired=surface & torch.isin(bodies,penalized).any(-1) & (force.abs()>1.).any(-1)
        contact_cost=q.new_zeros(self.num_envs)
        contact_cost.scatter_reduce_(0,world,undesired.to(q.dtype),reduce="amax",include_self=True)
        slip=q.new_zeros(self.num_envs); orientation=slip.clone()
        normal=rotate(quat,q.new_tensor((0.,0.,1.)))
        magnitude=force.norm(dim=-1)
        for foot in self.foot_ids:
            valid=torch.isin(bodies,self.foot_bodies[foot]).any(-1)&surface
            values=torch.where(valid,magnitude,torch.full_like(magnitude,-1))
            # Select strongest foot contact per world, matching legacy semantics.
            maximum=q.new_full((self.num_envs,),-1.)
            maximum.scatter_reduce_(0,world,values,reduce="amax",include_self=True)
            candidates=valid & (values==maximum[world])
            idx=torch.arange(world.numel(),device=self.device,dtype=torch.long)
            chosen=torch.full((self.num_envs,),world.numel(),device=self.device,dtype=torch.long)
            chosen.scatter_reduce_(0,world,torch.where(candidates,idx,world.numel()),reduce="amin",include_self=True)
            if world.numel()==0: continue
            safe=chosen.clamp_max(world.numel()-1)
            active=(chosen<world.numel()) & (maximum>1.)
            p=point[safe]
            footvel=vel[:,foot]+torch.cross(omega[:,foot],p-pos[:,foot],dim=-1)
            pair=bodies[safe]
            other=torch.where(torch.isin(pair[:,0],self.foot_bodies[foot]),pair[:,1],pair[:,0])
            env_ids=torch.arange(self.num_envs,device=self.device)
            surfacevel=vel[env_ids,other]+torch.cross(omega[env_ids,other],p-pos[env_ids,other],dim=-1)
            relative=footvel-surfacevel
            n=contact_normal[safe]
            n=n/n.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            tangential=relative-n*(relative*n).sum(-1,keepdim=True)
            slip+=torch.where(active,tangential.norm(dim=-1),0.)
            orientation+=torch.where(active,(1-normal[:,foot,2].square()).clamp_min(0).sqrt(),0.)
        rewards.update(penalty_undesired_contact=contact_cost,penalty_feet_ori=orientation,penalty_slippage=slip)
        return {k:rewards[k].to(torch.float32).clone() for k in AUX_REWARD_KEYS}

    def step(self, action):
        action=torch.as_tensor(action,device=self.device,dtype=torch.float32).detach()
        if action.shape != (self.num_envs,29) or not bool(torch.isfinite(action).all()):
            raise ValueError(f"action must be finite [{self.num_envs},29]")
        if bool(self._done.any()): raise RuntimeError("reset done worlds before stepping")
        observation={k:v.clone() for k,v in self._observation.items()}
        previous=self.last_action.clone(); self.last_action=action.clamp(-1,1).clone()
        self.backend.control_step(action if self.device.type=="cuda" else action[0].cpu().numpy())
        self.steps+=1
        fields=self._fields()
        next_obs=self._observe(fields)
        rewards=self._rewards(fields,previous)
        q,dq,pos,quat,vel,omega=fields
        terminated=(pos[:,self.root,2]<.35) | ~torch.isfinite(q).all(-1) | ~torch.isfinite(dq).all(-1)
        terminated|=((q<self.values["joint_pos_lower_limit"]-.05)|(q>self.values["joint_pos_upper_limit"]+.05)).any(-1)
        if hasattr(self.backend, "_external_terminated"):
            terminated |= torch.as_tensor(self.backend._external_terminated,device=self.device).reshape(self.num_envs)
        truncated=torch.zeros_like(terminated)
        if self.backend.episode_time_limit is not None:
            truncated=(self.steps*self.backend.config.control_dt>=self.backend.episode_time_limit) & ~terminated
        self._done=terminated|truncated
        self._observation={k:v.clone() for k,v in next_obs.items()}
        return {"observation":observation,"action":self.last_action.clone(),"step_count":(self.steps-1)[:,None].clone(),
                "aux_rewards":{k:v[:,None] for k,v in rewards.items()},
                "next":{"observation":next_obs,"terminated":terminated[:,None].clone(),"truncated":truncated[:,None].clone()}}

    def close(self):
        self.backend.close()
