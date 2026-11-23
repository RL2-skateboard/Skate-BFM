"""Measured staged toe trial; references only, no writes to simulated state."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .tail_catch import body_geoms


class ToeStages:
    def __init__(self,b,press,rotation,parameters=None):
        parameters=parameters or {}
        self.guard_version=parameters.get('geometry_guard_version','v1');self.parameters=parameters;self.hover_world=parameters.get('h_hover_world');self.raise_height=parameters.get('raise_height',.18);self.translate_time=parameters.get('translate_time',1.3);self.pulse_time=parameters.get('pulse_time',.20);self.pulse_fraction=0.
        from skate_sim.models.board import board_styles
        selected=board_styles(b.scene_config['style'])[b.scene_config['style']]
        self.board_style=selected;self.tail_x=-.45*selected.deck_length;self.grasp_x=.45625*selected.deck_length
        self.b=b;self.press=press;self.phase='raise_outside_board';self.start=float(b.data.time)
        self.foot=b.model.body('left_ankle_roll_link').id;self.deck=b.model.body('deck').id
        self.geoms=body_geoms(b,'left_ankle_roll_link');self.all_foot_geoms=body_geoms(b,'left_ankle_pitch_link');
        from .clearance import FootBoardClearance
        self.clearance=FootBoardClearance(b.model,self.all_foot_geoms,body_geoms(b,'chassis'))
        self.front=[g for g in self.geoms if b.model.geom_pos[g,0]>.05]
        self.local=np.array([.12,0,-.03]);normal=b.data.xmat[self.deck].reshape(3,3)@np.array([0.,0.,1.]);self.rotation_goal=Rotation.from_rotvec(parameters.get('yaw_offset',0.)*normal).as_matrix()@rotation;self.original=self.actual()[0].copy()
        self.p0,self.R0=self.actual();self.capture_z=.19;self.good=0.;self.reason=None;self.finished=False
        slope=2*selected.kick_height*((self.grasp_x-(selected.deck_length/2-selected.kick_length))/selected.kick_length)/selected.kick_length
        self.normal=np.array([slope,0,1.]);self.normal/=np.linalg.norm(self.normal)
        self.released=False;self.last_time=self.start
    def actual(self):
        d=self.b.data;R=d.xmat[self.foot].reshape(3,3).copy()
        return d.xpos[self.foot]+R@self.local,R
    def set_phase(self,name):
        self.phase=name;self.start=float(self.b.data.time);self.p0,self.R0=self.actual();self.good=0.
        if name=='press_pulse':
            R=self.b.data.xmat[self.deck].reshape(3,3)
            self.press_local=R.T@(self.p0-self.b.data.xpos[self.deck]);self.rotation_local=R.T@self.R0
    def surface(self,xy):
        b=self.b;R=b.data.xmat[self.deck].reshape(3,3);origin=b.data.xpos[self.deck]+R@np.r_[xy,.5];direction=-R[:,2];hits=[]
        for g in body_geoms(b,'deck'):
            if b.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            distance=mujoco.mj_rayMesh(b.model,b.data,g,origin,direction)
            if distance>=0:hits.append(distance)
        return .5-min(hits) if hits else None
    def top(self,x):return self.board_style.deck_thickness/2+self.board_style.kick_height*max((abs(float(x))-(self.board_style.deck_length/2-self.board_style.kick_length))/self.board_style.kick_length,0.)**2
    def measurements(self):
        b=self.b;d=b.data;DR=d.xmat[self.deck].reshape(3,3);centre=d.xpos[self.deck]
        local=np.array([DR.T@(d.geom_xpos[g]-centre) for g in self.geoms]);radii=np.array([b.model.geom_size[g,0] for g in self.geoms])
        front_indices=[self.geoms.index(g) for g in self.front]
        inside=[];clear=[]
        for i in front_indices:
            x,y,z=local[i];halfwidth=self.board_style.deck_width/2;half=self.board_style.deck_length/2;corner=self.board_style.corner_radius
            if abs(x)>half-corner:
                dx=abs(x)-(half-corner);halfwidth=self.board_style.deck_width/2-corner+np.sqrt(max(corner**2-dx**2,0.))
            inside.append(-half+radii[i]<=x<=-(half-self.board_style.kick_length)-radii[i] and abs(y)<=halfwidth-radii[i]-.003)
            actual_top=self.surface([x,y]) if self.guard_version=='v2' else self.top(x)
            if actual_top is None:inside[-1]=False;actual_top=self.board_style.deck_thickness/2+self.board_style.kick_height
            clear.append(z-radii[i]-actual_top)
        p,R=self.actual();dot=float(R[:,2]@(DR@self.normal))
        max_deck_z=float(max(d.geom_xpos[g,2]+b.model.geom_rbound[g] for g in body_geoms(b,'deck')))
        # Bounding spheres are intentionally conservative for the outside lift.
        all_high=bool(min(d.geom_xpos[g,2]-b.model.geom_size[g,0] for g in self.geoms)>=centre[2]+self.board_style.deck_thickness/2+self.board_style.kick_height+.03)
        certificate=self.clearance.measure(d) if self.guard_version=='v2' else {}
        actual_clear=certificate.get('separation_at_least_30mm',all_high)
        above_side=bool(all(p[2]-.005>=self.board_style.deck_thickness/2 for p in local[front_indices]) and dot>=0)
        if self.guard_version=='v2':all_high=certificate['exact_world_vertical_clearance_m']>=.03
        return dict(geometry_guard_version=self.guard_version,whole_geometry_certificate=certificate,whole_foot_separated=bool(actual_clear and above_side),front_deck_local_m=local[front_indices].tolist(),front_clearance_m=clear,front_inside=inside,foot_normal_dot_tail=dot,all_foot_above_board=all_high,toe_actual_world_m=p.tolist())
    @staticmethod
    def ramp(t,T):
        u=float(np.clip(t/T,0,1));s=10*u**3-15*u**4+6*u**5
        v=(30*u**2-60*u**3+30*u**4)/T if u<1 else 0.
        a=(60*u-180*u**2+120*u**3)/T**2 if u<1 else 0.
        return s,v,a
    def command(self,valid_impulse=0.,nose_angle=0.):
        b=self.b;d=b.data;now=float(d.time);elapsed=now-self.start;dt=now-self.last_time;self.last_time=now
        check=self.measurements();DR=d.xmat[self.deck].reshape(3,3);deckp=d.xpos[self.deck]
        if self.phase=='raise_outside_board':
            goal=self.original.copy();goal[2]=self.hover_world if self.hover_world is not None else self.original[2]+self.raise_height
            T=1.;targetR=self.R0
            ready=elapsed>=T and check['all_foot_above_board']
            self.good=self.good+dt if ready else 0.
            if self.good>=.05:self.set_phase('translate_above_tail');return self.command(valid_impulse,nose_angle)
            if elapsed>1.8:self.reason='actual full-foot vertical clearance guard';return None,check
        elif self.phase=='translate_above_tail':
            goal=deckp+DR@np.array([self.tail_x,0,.23-deckp[2]]);T=self.translate_time;targetR=self.rotation_goal
            ready=elapsed>=T and all(check['front_inside']) and (check['whole_foot_separated'] if self.guard_version=='v2' else check['all_foot_above_board']) and np.linalg.norm(self.actual()[0]-goal)<.02
            self.good=self.good+dt if ready else 0.
            if self.good>=.05:self.set_phase('capture_hover');return self.command(valid_impulse,nose_angle)
            if elapsed>T+.7:self.reason='actual horizontal toe capture guard';return None,check
            if not (check['whole_foot_separated'] if self.guard_version=='v2' else check['all_foot_above_board']):self.reason='lost full-foot clearance during horizontal motion';return None,check
        elif self.phase=='capture_hover':
            goal=deckp+DR@np.array([self.tail_x,0,self.capture_z-deckp[2]]);T=.6;targetR=self.rotation_goal
            ready=elapsed>=T and all(check['front_inside']) and min(check['front_clearance_m'])>=.03 and check['foot_normal_dot_tail']>=0 and np.linalg.norm(self.actual()[0]-goal)<.02
            self.good=self.good+dt if ready else 0.
            if self.good>=.05:
                if self.press:self.set_phase('press_pulse')
                else:self.set_phase('hover_control')
                return self.command(valid_impulse,nose_angle)
            if elapsed>1.3:self.reason='actual over-tail hover before press guard';return None,check
        elif self.phase=='press_pulse':
            s,v,a=self.ramp(elapsed,self.pulse_time);self.pulse_fraction=float(min(elapsed/.03,(self.pulse_time-elapsed)/.03,1.));local=self.press_local-.04*s*self.normal
            point=deckp+DR@local
            jp=np.zeros((3,b.model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(b.model,d,jp,jr,point,self.deck)
            vel=jp@d.qvel-.04*v*(DR@self.normal);acc=-.04*a*(DR@self.normal)
            if elapsed>=self.pulse_time or (valid_impulse>0 and nose_angle>.10):self.released=True;self.set_phase('retract');return self.command(valid_impulse,nose_angle)
            return dict(local=self.local,pos=point,vel=vel,acc=acc,rotation=DR@self.rotation_local,omega=jr@d.qvel,alpha=np.zeros(3)),check
        elif self.phase in ('retract','hover_control'):
            goal=self.original.copy() if self.phase=='retract' else self.p0.copy();goal[2]=max(.23,self.p0[2]+.10) if self.phase=='retract' else self.p0[2];T=.6;targetR=self.rotation_goal
            if elapsed>.70:self.finished=True;self.reason='bounded staged toe trial completed';return None,check
        else:raise RuntimeError(self.phase)
        s,v,a=self.ramp(elapsed,T);delta=goal-self.p0;rv=Rotation.from_matrix(targetR@self.R0.T).as_rotvec()
        return dict(local=self.local,pos=self.p0+s*delta,vel=v*delta,acc=a*delta,rotation=Rotation.from_rotvec(s*rv).as_matrix()@self.R0,omega=v*rv,alpha=a*rv),check
