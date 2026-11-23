"""Experimental full-model contact inverse dynamics. All simulated forces stay native."""
from __future__ import annotations

import numpy as np
import mujoco
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation


class SupportGeometryGuard(RuntimeError):
    """Measured support cannot satisfy the predeclared safe geometry."""

def inset_support(points,margin=.005):
    """Measured planar support, inset without enlarging degenerate contacts."""
    from scipy.spatial import ConvexHull
    points=np.unique(np.asarray(points,float),axis=0)
    if len(points)<2:raise SupportGeometryGuard('Insufficient measured support points')
    if len(points)==2 or np.linalg.matrix_rank(points-points[0])<2:
        direction=points[-1]-points[0];length=np.linalg.norm(direction)
        if length<=2*margin:raise SupportGeometryGuard('Measured support line too short')
        direction/=length
        return np.array([points[0]+margin*direction,points[-1]-margin*direction])
    hull=ConvexHull(points);eq=hull.equations.copy();eq[:,2]+=margin
    vertices=[]
    for j,a in enumerate(eq):
        for b in eq[j+1:]:
            pair=np.vstack([a[:2],b[:2]])
            if abs(np.linalg.det(pair))<1e-12:continue
            point=np.linalg.solve(pair,-np.array([a[2],b[2]]))
            if np.all(eq[:,:2]@point+eq[:,2]<=1e-9):vertices.append(point)
    vertices=np.unique(np.asarray(vertices),axis=0)
    if len(vertices)<3:raise SupportGeometryGuard('Empty inset measured support')
    vertices=vertices[ConvexHull(vertices).vertices]
    return vertices


def project_support(point,vertices):
    point=np.asarray(point,float);vertices=np.asarray(vertices,float)
    if len(vertices)>2:
        from scipy.spatial import ConvexHull
        eq=ConvexHull(vertices).equations
        if np.all(eq[:,:2]@point+eq[:,2]<=0):return point.copy()
    choices=[]
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        delta=b-a;t=np.clip((point-a)@delta/(delta@delta),0,1)
        choices.append(a+t*delta)
    return min(choices,key=lambda p:np.linalg.norm(p-point)).copy()


def toe_reference(backend,goal,prefer_flat=False):
    """Bounded offline toe pose query on current actual state; never writes it."""
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from .tail_catch import body_geoms
    b=backend;m=b.model;d=mujoco.MjData(m);mujoco.mj_copyData(d,m,b.data)
    original=d.qpos.copy();left=m.body('left_ankle_roll_link').id;right=m.body('right_ankle_roll_link').id;root=m.body('pelvis').id
    rp=d.xpos[right].copy();rr=d.xmat[right].reshape(3,3).copy()
    xy=d.geom_xpos[body_geoms(b,'right_ankle_roll_link'),:2];hull=ConvexHull(xy);cg=xy[hull.vertices].mean(axis=0)
    ids=b.robot_qpos_adr[:15];jids=[m.joint(n).id for n in b.robot_joint_names[:15]];lo=m.jnt_range[jids,0];hi=m.jnt_range[jids,1];local=np.array([.12,0,-.03])
    deck_rotation=b.data.xmat[m.body('deck').id].reshape(3,3).copy()
    tail_normal=deck_rotation@np.array([.39,0,1.]);tail_normal/=np.linalg.norm(tail_normal)
    def residual(q):
        d.qpos[:]=original;d.qpos[ids]=q;mujoco.mj_forward(m,d)
        point=d.xpos[left]+d.xmat[left].reshape(3,3)@local
        return np.r_[100*(point-goal),100*(d.xpos[right]-rp),10*Rotation.from_matrix(rr.T@d.xmat[right].reshape(3,3)).as_rotvec(),10*d.xmat[left].reshape(3,3)[2,1],30*(d.subtree_com[root,:2]-cg),100*min(float(d.xmat[left].reshape(3,3)[:,2]@tail_normal)-.05,0),10*Rotation.from_matrix(deck_rotation.T@d.xmat[left].reshape(3,3)).as_rotvec() if prefer_flat else np.zeros(3),.001*(q-original[ids])]
    result=least_squares(residual,np.clip(original[ids],lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    residual(result.x);rotation=d.xmat[left].reshape(3,3).copy();actual=d.xpos[left]+rotation@local
    margin=float(np.min(-(hull.equations[:,:2]@d.subtree_com[root,:2]+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
    audit=dict(foot_normal_dot_tail=float(rotation[:,2]@tail_normal),toe_error_m=float(np.linalg.norm(actual-goal)),right_foot_error_m=float(np.linalg.norm(d.xpos[right]-rp)),right_rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(rr.T@d.xmat[right].reshape(3,3)).as_rotvec())),com_margin_m=margin,q_reference=result.x.tolist(),left_rotation_world=rotation.tolist())
    if audit['foot_normal_dot_tail']<0 or audit['toe_error_m']>.005 or audit['right_foot_error_m']>.005 or audit['right_rotation_error_rad']>np.deg2rad(5) or margin<.01:raise RuntimeError(f'Toe reference infeasible: {audit}')
    return rotation,audit


class ContactBalance:
    """100Hz QP: full accelerations,29 physical joint torques, contact/equality reactions.

    Contact and equality forces below are prediction variables, never applied to
    MuJoCo. Execution uses only original29 limited PD torques and Robotiq ctrl.
    """
    def __init__(self,b):
        from .tail_catch import body_geoms
        self.b=b;self.m=b.model;self.nv=self.m.nv
        self.dofs=b.robot_dof_adr.copy();self.qadr=b.robot_qpos_adr.copy()
        self.feet=[set(body_geoms(b,s+'_ankle_roll_link')) for s in ['left','right']]
        self.ground=self.m.geom('ground').id
        self.root=self.m.body('pelvis').id
        self.mass=float(self.m.body_subtreemass[self.root]);self.mg=self.mass*9.81
        self.q0=b.data.qpos[self.qadr].copy()
        self.com0=b.data.subtree_com[self.root].copy()
        self.orientation0=b.data.xmat[self.root].reshape(3,3).copy()
        self.footposes=[b.data.xpos[self.m.body(s+'_ankle_roll_link').id].copy() for s in ['left','right']]
        self.footrot=[b.data.xmat[self.m.body(s+'_ankle_roll_link').id].reshape(3,3).copy() for s in ['left','right']]
        self.numeric_backend='clarabel_lifted_ls';self.toe_force_target=None
        self.feedback_enabled=False;self.allow_board_support=False;self.board_geoms=set(body_geoms(b,'chassis'))
        self.anchors={};self.previous_tau=np.zeros(29);self.previous_solution=None
        self.robot_bodies=[]
        for i in range(self.m.nbody):
            parent=i
            while parent and parent!=self.root:parent=int(self.m.body_parentid[parent])
            if parent==self.root:self.robot_bodies.append(i)

    def jac(self,data,point,body):
        p=np.zeros((3,self.nv));r=np.zeros_like(p)
        dp=np.zeros_like(p);dr=np.zeros_like(p)
        mujoco.mj_jac(self.m,data,p,r,point,body)
        mujoco.mj_jacDot(self.m,data,dp,dr,point,body)
        return p,r,dp,dr

    def constraint_jacobian(self,data):
        if not mujoco.mj_isSparse(self.m):return np.asarray(data.efc_J).reshape(data.nefc,self.nv)
        result=np.zeros((data.nefc,self.nv))
        for i in range(data.nefc):
            start=int(data.efc_J_rowadr[i]);end=start+int(data.efc_J_rownnz[i])
            result[i,np.asarray(data.efc_J_colind[start:end],int)]=data.efc_J[start:end]
        return result

    def solve(self,com_ref,com_velocity,com_acceleration,left_load=0.5,swing=None,hand=None):
        b=self.b;d=b.data;nv=self.nv
        # Actual current command/state actuation, including original passive linkage.
        from skate_sim.models.gripper import command_grippers
        command_grippers(self.m,d,b.gripper_command)
        mujoco.mj_forward(self.m,d)
        M=np.zeros((nv,nv));mujoco.mj_fullM(self.m,d,M)
        known=d.qfrc_actuator.copy()+d.qfrc_passive
        applied=d.qfrc_applied.copy();applied[self.dofs]=0
        if np.any(applied[:6]) or np.any(d.xfrc_applied):raise RuntimeError('Unexpected external base/body force')
        all_J=self.constraint_jacobian(d)
        kinds=np.asarray(d.efc_type)
        equality=np.flatnonzero(kinds==mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        contact_kinds=[int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)]
        passive_constraints=np.flatnonzero((kinds!=mujoco.mjtConstraint.mjCNSTR_EQUALITY)&~np.isin(kinds,contact_kinds))
        measured_other=all_J[passive_constraints].T@np.asarray(d.efc_force)[passive_constraints] if len(passive_constraints) else np.zeros(nv)
        h=d.qfrc_bias-known-applied-measured_other
        Je=all_J[equality]
        contacts=[]
        for i in range(d.ncon):
            c=d.contact[i]
            if c.efc_address<0:continue
            first=int(self.m.geom_bodyid[c.geom1]);second=int(self.m.geom_bodyid[c.geom2])
            j1,_,_,_=self.jac(d,c.pos,first);j2,_,_,_=self.jac(d,c.pos,second)
            frame=c.frame.reshape(3,3)
            group=None;pair={int(c.geom1),int(c.geom2)}
            if self.ground in pair:
                for side,geoms in enumerate(self.feet):
                    if pair&geoms:group=side
            if self.allow_board_support and pair&self.board_geoms:
                for side,geoms in enumerate(self.feet):
                    if pair&geoms:group=side
            sign=1. if group is not None and int(c.geom2) in self.feet[group] else -1.
            contacts.append(dict(index=i,c=c,body1=first,body2=second,B=(j2-j1).T@frame.T,vertical=sign*frame[:,2],
                                 mu=float(min(c.friction[:2])),group=group))
        grasp_contacts=[];measured_grasp_force=np.zeros(3);measured_grasp_moment=np.zeros(3)
        if getattr(self,'grasp_wrench_tracking',False):
            from .tail_catch import body_geoms
            from ..models.gripper import pad_tcp
            grip=set(body_geoms(b,'right_2f85_base_mount'));tcp,_=pad_tcp(self.m,d,'right');cf=np.zeros(6)
            for i,c in enumerate(contacts):
                pair={int(c['c'].geom1),int(c['c'].geom2)}
                if not(pair&grip and pair&self.board_geoms):continue
                sign=1. if int(c['c'].geom2) in grip else -1.;direction=sign*c['c'].frame.reshape(3,3).T;mujoco.mj_contactForce(self.m,d,c['index'],cf);F=direction@cf[:3];lever=c['c'].pos-tcp
                measured_grasp_force+=F;measured_grasp_moment+=np.cross(lever,F);grasp_contacts.append((i,c,direction,lever,cf[:3].copy()))
        support_load=self.mg-measured_grasp_force[2]
        nc=len(contacts);ne=len(equality);n=nv+29+3*nc+ne
        ts=slice(nv,nv+29);fs=slice(nv+29,nv+29+3*nc);es=slice(nv+29+3*nc,n)
        P=np.eye(n)*2e-8;q=np.zeros(n)
        ls_rows=[np.eye(n)*1e-4];ls_targets=[np.zeros(n)]
        def cost(A,target,weight,scale=1.):
            nonlocal P,q
            A=np.asarray(A,dtype=float);target=np.asarray(target,dtype=float)
            if A.ndim==1:A=A[None,:];target=np.atleast_1d(target)
            divisor=np.asarray(scale)
            A=A/(divisor[:,None] if divisor.ndim else divisor);target=target/divisor
            P+=2*weight*A.T@A;q-=2*weight*A.T@target
            ls_rows.append(np.sqrt(weight)*A);ls_targets.append(np.sqrt(weight)*target)
        def task(J,goal,weight,scale=1.):
            A=np.zeros((len(J),n));A[:,:nv]=J;cost(A,goal,weight,scale)
        # Exact full dynamics; predicted reactions are not written to simulator.
        Aeq=np.zeros((nv,n));Aeq[:,:nv]=M;Aeq[self.dofs,nv+np.arange(29)]=-1
        if nc:Aeq[:,fs]=-np.column_stack([c['B'] for c in contacts])
        if ne:Aeq[:,es]=-Je.T
        constraints=[Aeq];lower=[-h];upper=[-h]
        if ne:
            scales=np.array([.5 if self.m.eq_type[int(d.efc_id[i])]==mujoco.mjtEq.mjEQ_CONNECT else 2. for i in equality])
            task(Je/scales[:,None],np.asarray(d.efc_aref)[equality]/scales,100)
        Jcom=np.zeros((3,nv));mujoco.mj_jacSubtreeCom(self.m,d,Jcom,self.root)
        Dcom=np.zeros((3,nv))
        for body in self.robot_bodies:
            _,_,dot,_=self.jac(d,d.xipos[body],body)
            Dcom+=self.m.body_mass[body]/self.mass*dot
        goal=np.clip(com_acceleration+25*(com_ref-d.subtree_com[self.root])+10*(com_velocity-Jcom@d.qvel),-3,3)
        dcm_info={};cop_target=None
        if self.feedback_enabled:
            measured=[];measured_force=[];side_points=[[],[]];side_force=[[],[]]
            cf=np.zeros(6)
            for contact in contacts:
                if contact['group'] is None:continue
                mujoco.mj_contactForce(self.m,d,contact['index'],cf)
                vertical=float(contact['vertical']@cf[:3])
                if vertical<=1e-5:continue
                pos=contact['c'].pos[:2].copy();side=contact['group']
                measured.append(pos);measured_force.append(vertical)
                side_points[side].append(pos);side_force[side].append(vertical)
            hull=inset_support(measured)
            omega=np.sqrt(9.81/self.com0[2]);position=d.subtree_com[self.root];velocity=Jcom@d.qvel
            xi=position[:2]+velocity[:2]/omega
            xi_ref=np.asarray(com_ref)[:2]+np.asarray(com_velocity)[:2]/omega
            xi_dot_ref=np.asarray(com_velocity)[:2]+np.asarray(com_acceleration)[:2]/omega
            raw=xi-xi_dot_ref/omega+.7*(xi-xi_ref);cop_target=project_support(raw,hull)
            goal[:2]=np.clip(omega**2*(position[:2]-cop_target),-3,3)
            if all(side_points):
                centres=np.array([np.average(p,axis=0,weights=f) for p,f in zip(side_points,side_force)])
                delta=centres[1]-centres[0]
                fraction=np.clip((cop_target-centres[0])@delta/(delta@delta),0,1)
                left_load=float(1-fraction)
            elif side_points[1]:left_load=0.
            else:left_load=1.
            dcm_info=dict(dcm_xy_m=xi.tolist(),com_velocity_xy_m_s=velocity[:2].tolist(),cop_raw_xy_m=raw.tolist(),cop_target_xy_m=cop_target.tolist(),cop_saturation_m=(cop_target-raw).tolist(),
                actual_cop_xy_m=np.average(measured,axis=0,weights=measured_force).tolist(),actual_support_force_over_mg=float(sum(measured_force)/self.mg),support_inset_vertices_xy_m=hull.tolist(),target_left_load_fraction=left_load)
        task(Jcom,goal-Dcom@d.qvel,10)
        p,Jrot,dp,Drot=self.jac(d,d.xpos[self.root],self.root)
        error=Rotation.from_matrix(self.orientation0@d.xmat[self.root].reshape(3,3).T).as_rotvec()
        goal=np.clip(25*error-10*(Jrot@d.qvel),-8,8)
        task(Jrot,goal-Drot@d.qvel,8,2.)
        for side,geoms in enumerate(self.feet):
            foot=self.m.body(('left' if side==0 else 'right')+'_ankle_roll_link').id
            if swing is not None and side==0:
                if isinstance(swing,dict):
                    location=d.xpos[foot]+d.xmat[foot].reshape(3,3)@swing['local']
                    pos,vel,acc=swing['pos'],swing['vel'],swing['acc']
                    rotation=swing['rotation'];omega=swing.get('omega',np.zeros(3));alpha=swing.get('alpha',np.zeros(3))
                else:
                    location=d.xpos[foot];pos,vel,acc=swing;rotation=self.footrot[side];omega=np.zeros(3);alpha=np.zeros(3)
                J,R,D,DR=self.jac(d,location,foot)
                desired=np.clip(acc+25*(pos-location)+10*(vel-J@d.qvel),-3,3)
                task(J,desired-D@d.qvel,6)
                er=Rotation.from_matrix(rotation@d.xmat[foot].reshape(3,3).T).as_rotvec()
                task(R,np.clip(alpha+25*er+10*(omega-R@d.qvel),-8,8)-DR@d.qvel,6,2)
                continue
            # Material support points: do not pin wheel horizontal motion.
            for geom in geoms:
                candidates=[c for c in contacts if c['group']==side and geom in (c['c'].geom1,c['c'].geom2)]
                if not candidates:continue
                body=int(self.m.geom_bodyid[geom]);rotation=d.xmat[body].reshape(3,3)
                if geom not in self.anchors:
                    world=candidates[0]['c'].pos.copy()
                    self.anchors[geom]=(rotation.T@(world-d.xpos[body]),world)
                local,anchor=self.anchors[geom];world=d.xpos[body]+rotation@local
                J,_,D,_=self.jac(d,world,body)
                desired=np.clip(60*(anchor-world)-15*(J@d.qvel),-3,3)
                task(J,desired-D@d.qvel,25,.5)
        if hand is not None:
            from skate_sim.models.gripper import pad_tcp
            point,rotation=pad_tcp(self.m,d,'right')
            hand_body=self.m.body('right_2f85_base').id
            J,R,D,DR=self.jac(d,point,hand_body)
            if isinstance(hand,dict):
                goal_position=hand['pos'];goal_rotation=hand['rotation'];goal_velocity=hand['vel'];goal_acceleration=hand['acc'];goal_omega=hand['omega'];goal_alpha=hand['alpha']
            else:
                goal_position,goal_rotation=hand;goal_velocity=np.zeros(3);goal_acceleration=np.zeros(3);goal_omega=np.zeros(3);goal_alpha=np.zeros(3)
            desired=np.clip(goal_acceleration+25*(goal_position-point)+10*(goal_velocity-J@d.qvel),-3,3)
            task(J,desired-D@d.qvel,6)
            error=Rotation.from_matrix(goal_rotation@rotation.T).as_rotvec()
            task(R,np.clip(goal_alpha+25*error+10*(goal_omega-R@d.qvel),-8,8)-DR@d.qvel,6,2.)
        # Wheels can roll: constrain normal acceleration softly, never horizontal.
        for contact in contacts:
            c=contact['c']
            if self.ground not in (c.geom1,c.geom2) or contact['group'] is not None:continue
            normal=c.frame.reshape(3,3)[0]
            j1,_,d1,_=self.jac(d,c.pos,contact['body1']);j2,_,d2,_=self.jac(d,c.pos,contact['body2'])
            J=normal@(j2-j1);D=normal@(d2-d1)
            task(J[None,:],np.array([-15*float(J@d.qvel)-float(D@d.qvel)]),25,.5)
        J=np.zeros((29,nv));J[np.arange(29),self.dofs]=1
        posture=np.clip(5*(self.q0-d.qpos[self.qadr])-3*d.qvel[self.dofs],-30,30)
        task(J,posture,.2,10)
        limits=b.policy_values['joint_effort_limit']
        T=np.zeros((29,n));T[:,ts]=np.eye(29)
        cost(T,np.zeros(29),.02,limits);cost(T,self.previous_tau,.1,limits)
        if nc:
            F=np.zeros((3*nc,n));F[:,fs]=np.eye(3*nc)
            cost(F,np.zeros(3*nc),.01,self.mg)
            if cop_target is not None:
                C=np.zeros((2,n))
                for i,c in enumerate(contacts):
                    if c['group'] is not None:C[:,nv+29+3*i:nv+29+3*i+3]=np.outer(c['c'].pos[:2]-cop_target,c['vertical'])
                cost(C,np.zeros(2),10,self.mg*.03)
            if self.toe_force_target is not None:
                L=np.zeros(n)
                for i,c in enumerate(contacts):
                    if c['group']==0 and {int(c['c'].geom1),int(c['c'].geom2)}&self.board_geoms:L[nv+29+3*i:nv+29+3*i+3]=c['vertical']
                cost(L,np.array([self.toe_force_target]),2,100.)
            if grasp_contacts:
                G=np.zeros((6,n))
                for i,c,direction,lever,cf in grasp_contacts:
                    columns=slice(nv+29+3*i,nv+29+3*i+3);G[:3,columns]=direction;cross=np.array([[0.,-lever[2],lever[1]],[lever[2],0.,-lever[0]],[-lever[1],lever[0],0.]]);G[3:,columns]=cross@direction
                    j1,_,d1,_=self.jac(d,c['c'].pos,c['body1']);j2,_,d2,_=self.jac(d,c['c'].pos,c['body2']);relative=j2-j1;rate=(d2-d1)@d.qvel
                    task(relative,-15*(relative@d.qvel)-rate,6,1.)
                cost(G,np.r_[measured_grasp_force,measured_grasp_moment],1.,np.array([10.,10.,10.,1.,1.,1.]))
            for side,portion in enumerate([left_load,1-left_load]):
                L=np.zeros(n)
                for i,c in enumerate(contacts):
                    if c['group']==side:L[nv+29+3*i:nv+29+3*i+3]=c['vertical']
                cost(L,np.array([portion*support_load]),2,self.mg)
        lo=np.full(n,-np.inf);hi=np.full(n,np.inf)
        kp=b.policy_values['kp'];kd=b.policy_values['kd'];joint=d.qpos[self.qadr];vel=d.qvel[self.dofs]
        lo[ts]=np.maximum(-limits,kp*(b.policy_values['joint_pos_lower_limit']-joint)-kd*vel)
        hi[ts]=np.minimum(limits,kp*(b.policy_values['joint_pos_upper_limit']-joint)-kd*vel)
        for i,c in enumerate(contacts):
            j=nv+29+3*i;lo[j]=0;hi[j]=2*self.mg
            for a in [-1,1]:
                for z in [-1,1]:
                    row=np.zeros(n);row[j]=-c['mu'];row[j+1]=a;row[j+2]=z
                    constraints.append(row[None,:]);lower.append(np.array([-np.inf]));upper.append(np.array([0.]))
        constraints.append(np.eye(n));lower.append(lo);upper.append(hi)
        A=np.vstack(constraints);l=np.concatenate(lower);u=np.concatenate(upper)
        # Algebraically eliminate accelerations using the actual SPD full mass.
        # x = offset + transform*y, y=(tau,contact reactions,equality reactions).
        # This preserves all costs/bounds, and enforces all56 dynamics equations.
        from scipy.linalg import cho_factor, cho_solve
        factor=cho_factor(M,lower=True,check_finite=False)
        free_acc=cho_solve(factor,-h,check_finite=False)
        response=cho_solve(factor,-Aeq[:,nv:],check_finite=False)
        ny=n-nv
        transform=np.vstack([response,np.eye(ny)])
        offset=np.r_[free_acc,np.zeros(ny)]
        K=np.vstack(ls_rows);target=np.concatenate(ls_targets)
        W=K@transform;v=target-K@offset
        nmain=ny-ne
        # Householder QR removes unconstrained equality reactions in residual
        # space; never subtract nearly equal normal-equation matrices.
        from scipy.linalg.lapack import dgeqrf,dormqr
        from scipy.linalg import solve_triangular
        if ne:
            qr,house,_,info=dgeqrf(np.asfortranarray(W[:,nmain:]))
            if info:raise RuntimeError(f'Equality QR failed: {info}')
            C=np.asfortranarray(np.column_stack([W[:,:nmain],v]))
            _,work,info=dormqr('L','T',qr,house,C,lwork=-1)
            C,_,info=dormqr('L','T',qr,house,C,lwork=int(work[0]))
            if info:raise RuntimeError(f'Equality QR apply failed: {info}')
            R=np.triu(qr[:ne,:]);top=C[:ne];tail=C[ne:]
        else:
            R=np.empty((0,0));top=np.empty((0,nmain+1))
            tail=np.column_stack([W,v])
        reduced_P=2*tail[:,:nmain].T@tail[:,:nmain]
        reduced_q=-2*tail[:,:nmain].T@tail[:,-1]
        bounded=np.isfinite(l[nv:])|np.isfinite(u[nv:])
        remaining=A[nv:][bounded]
        full_reduced_A=remaining@transform
        if ne and np.any(full_reduced_A[:,nmain:]!=0):
            raise RuntimeError('Equality reactions unexpectedly have hard constraints')
        reduced_A=full_reduced_A[:,:nmain]
        reduced_l=l[nv:][bounded]-remaining@offset
        reduced_u=u[nv:][bounded]-remaining@offset
        def recover(main):
            lam=solve_triangular(R,top[:,-1]-top[:,:nmain]@main) if ne else np.empty(0)
            return np.r_[main,lam]
        units=1./np.maximum(np.linalg.norm(tail[:,:nmain],axis=0),1e-12)
        scaled_A=reduced_A*units[None,:]
        row_units=np.maximum(np.max(abs(scaled_A),axis=1),1.)
        scaled_A/=row_units[:,None]
        scaled_P=reduced_P*units[:,None]*units[None,:]
        objective_unit=1.
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(np.triu(scaled_P/objective_unit)),q=reduced_q*units/objective_unit,
            A=sparse.csc_matrix(scaled_A),l=reduced_l/row_units,u=reduced_u/row_units,
            verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polish=True,warm_start=True)
        seed=np.zeros(nmain);seed[:29]=d.qfrc_applied[self.dofs]
        contact_force=np.zeros(6)
        for i,c in enumerate(contacts):
            mujoco.mj_contactForce(self.m,d,c['index'],contact_force)
            seed[29+3*i:29+3*i+3]=contact_force[:3]
        solver.warm_start(x=seed/units)
        self.debug_numeric=dict(K=K,target=target,W=W,v=v,qr_top=top,qr_tail=tail,qr_R=R,recover=recover,P=P,q=q,A=A,l=l,u=u,transform=transform,offset=offset,units=units,
            reduced_P=reduced_P,reduced_q=reduced_q,reduced_A=reduced_A,reduced_l=reduced_l,reduced_u=reduced_u,
            scaled_P=scaled_P/objective_unit,scaled_q=reduced_q*units/objective_unit,scaled_A=scaled_A,
            scaled_l=reduced_l/row_units,scaled_u=reduced_u/row_units,objective_unit=objective_unit)
        import time
        solve_started=time.process_time()
        self.debug_problem=dict(Aeq=Aeq,h=h,contacts=contacts,equality=equality,lo=lo,hi=hi)
        if self.numeric_backend=='clarabel_lifted_ls':
            from .qp_solver import clarabel_least_squares
            result=clarabel_least_squares(tail[:,:nmain]*units[None,:],tail[:,-1],scaled_A,reduced_l/row_units,reduced_u/row_units)
            total_iterations=int(result.info.iter);restarts=0;solver_name='clarabel_lifted_ls';osqp_status='not run: direct exact residual lifting'
        else:
            result=solver.solve();total_iterations=int(result.info.iter);restarts=0;solver_name='osqp';osqp_status=result.info.status
            candidate_valid=False
            if result.info.status=='solved' and np.isfinite(result.x).all():
                candidate=offset+transform@recover(result.x*units)
                candidate_valid=float(np.max(np.maximum(l-A@candidate,A@candidate-u)))<=1e-3
            if not candidate_valid:
                from .qp_solver import clarabel_least_squares
                result=clarabel_least_squares(tail[:,:nmain]*units[None,:],tail[:,-1],scaled_A,reduced_l/row_units,reduced_u/row_units)
                total_iterations+=int(result.info.iter);solver_name='clarabel_lifted_after_osqp'
        self.debug_problem=dict(Aeq=Aeq,h=h,contacts=contacts,equality=equality,lo=lo,hi=hi)
        if result.info.status!='solved':
            raise RuntimeError(f'Inverse-dynamics QP {result.info.status}; pri={result.info.pri_res}, dual={result.info.dua_res}, n={ny}, contacts={nc}')
        main=result.x*units
        y=recover(main);x=offset+transform@y
        ls_cost=float(np.sum((K@x-target)**2))
        qr_cost=float(np.sum((tail[:,:nmain]@main-tail[:,-1])**2))
        if abs(ls_cost-qr_cost)>1e-7*max(1.,ls_cost):raise RuntimeError('QR objective mismatch')
        self.previous_solution=x.copy();self.previous_tau=x[ts].copy()
        dyn=float(np.max(abs(Aeq@x+h)))
        physical_violation=float(np.max(np.maximum(l-A@x,A@x-u)))
        if not np.isfinite(x).all() or dyn>1. or physical_violation>1e-3:
            raise RuntimeError(f'QP physical residual dynamics={dyn}, bounds={physical_violation}')
        gradient=scaled_P@(result.x)+reduced_q*units+scaled_A.T@result.y
        physical_gradient=gradient/units
        gradient_denominator=max(1.,float(np.max(abs(reduced_P@main))),float(np.max(abs(reduced_q))),float(np.max(abs(reduced_A.T@(result.y/row_units)))))
        relative_stationarity=float(np.max(abs(physical_gradient))/gradient_denominator)
        if np.max(abs(physical_gradient))>1e-3 or relative_stationarity>1e-6 or (getattr(result,'relative_dual_gap',0.)>1e-6):raise RuntimeError(f'QP original-coordinate optimality certificate failed: gradient={np.max(abs(physical_gradient))}, relative={relative_stationarity}')
        diagnostics=dict(internal_gap_tolerance=getattr(result,'internal_gap_tolerance',None),numeric_retries=getattr(result,'numeric_retries',0),first_solver_status=getattr(result,'first_status',None),relative_stationarity=relative_stationarity,stationarity_denominator=gradient_denominator,dual_complementarity_inf=getattr(result,'dual_complementarity_inf',None),relative_dual_gap=getattr(result,'relative_dual_gap',None),lifted_equation_inf=getattr(result,'residual_equation_inf',None),solver_module=getattr(result,'module_file',osqp.__file__),solver_version=getattr(result,'version',osqp.__version__),solver_backend=solver_name,osqp_status=osqp_status,kkt_scaled_inf=float(np.max(abs(gradient))),kkt_physical_inf=float(np.max(abs(gradient/units))),status=result.info.status,ls_cost=ls_cost,qr_cost=qr_cost,qr_objective_error=abs(ls_cost-qr_cost),dynamics_residual_inf=dyn,physical_bound_violation=physical_violation,
            equality_acceleration_residual_inf=float(np.max(abs(Je@x[:nv]-np.asarray(d.efc_aref)[equality]))) if ne else 0.,
            contacts=nc,equality_rows=ne,measured_other_constraint_rows=len(passive_constraints),iterations=total_iterations,solver_restarts=restarts,solver_cpu_seconds=time.process_time()-solve_started,predicted_com_acceleration=(Jcom@x[:nv]+Dcom@d.qvel).tolist(),
            actual_com=d.subtree_com[self.root].copy().tolist(),target_com=np.asarray(com_ref).tolist(),
            predicted_foot_normal_N=[float(sum(c['vertical']@x[nv+29+3*i:nv+29+3*i+3] for i,c in enumerate(contacts) if c['group']==side)) for side in [0,1]])
        diagnostics.update(dcm_info)
        if grasp_contacts:diagnostics.update(measured_grasp_force_world_N=measured_grasp_force.tolist(),measured_grasp_moment_about_TCP_Nm=measured_grasp_moment.tolist(),foot_load_target_N=float(support_load),grasp_contact_count=len(grasp_contacts),grasp_prediction="measured wrench cost and sticking velocity soft task on actual contact variables only; no simulated weld/force")
        if cop_target is not None:
            supported=[(c['c'].pos[:2],c['vertical']@x[nv+29+3*i:nv+29+3*i+3]) for i,c in enumerate(contacts) if c['group'] is not None]
            total=sum(f for _,f in supported)
            diagnostics['predicted_cop_xy_m']=(sum(p*f for p,f in supported)/total).tolist() if total>1e-5 else None
        return x[ts].copy(),diagnostics

    def compensate_arm(self,tau,hand):
        """Explicit experimental seven-arm residual, separate from WBC prediction.

        Use a name-mapped, bounded task-space spring. Only original
        body torques are delivered; no waist/leg/base or board force is added.
        """
        total=np.asarray(tau,float).copy()
        if hand is None or not getattr(self,'arm_compensation_enabled',False):return total,{}
        from ..models.gripper import pad_tcp
        b=self.b;now=float(b.data.time)
        position,rotation=(hand['pos'],hand['rotation']) if isinstance(hand,dict) else hand
        p,R=pad_tcp(self.m,b.data,'right');body=self.m.body('right_2f85_base').id
        Jp,Jr,_,_=self.jac(b.data,p,body);position_error=np.asarray(position)-p;rotation_error=Rotation.from_matrix(np.asarray(rotation)@R.T).as_rotvec()
        force=100.*position_error;force*=min(1.,8./max(np.linalg.norm(force),1e-12));moment=2.*rotation_error;moment*=min(1.,.6/max(np.linalg.norm(moment),1e-12))
        kp=b.policy_values['kp'];kd=b.policy_values['kd'];q=b._robot_q();dq=b._robot_dq();lim=b.policy_values['joint_effort_limit'];lo=np.maximum(-lim,kp*(b.policy_values['joint_pos_lower_limit']-q)-kd*dq);hi=np.minimum(lim,kp*(b.policy_values['joint_pos_upper_limit']-q)-kd*dq)
        u=float(np.clip((now-self.arm_compensation_start)/.5,0.,1.));blend=10*u**3-15*u**4+6*u**5
        requested=blend*np.clip((Jp.T@force+Jr.T@moment)[b.robot_dof_adr[22:29]],-.8,.8)
        delivered=np.clip(requested,lo[22:29]-total[22:29],hi[22:29]-total[22:29]);total[22:29]+=delivered
        return total,dict(method='right-arm-only TCP J-transpose100N/m capped8N +2Nm/rad capped0.6Nm; each axis0.8Nm and PD/effort headroom; replaces IKbias; NOT WBC-predicted torque',axis_names=b.robot_joint_names[22:29],position_error_world_m=position_error.tolist(),rotation_error_world_rad=rotation_error.tolist(),requested_task_force_world_N=force.tolist(),requested_task_moment_world_Nm=moment.tolist(),requested_Nm=requested.tolist(),delivered_Nm=delivered.tolist(),headroom_clipped=(abs(requested-delivered)>1e-9).tolist(),blend=blend)

    def apply(self,tau):
        """Reverse original PD; only b.step performs the integration and torque write."""
        b=self.b;kp=b.policy_values['kp'];kd=b.policy_values['kd'];q=b._robot_q();v=b._robot_dq()
        target=q+(tau+kd*v)/kp
        b.pose_target=target
        return target


class TorqueRecorder:
    """Wrap existing Recorder; action29 is actual joint Nm, never a dummy BFM action."""
    def __init__(self,backend,path,metadata):
        from skate_sim.record import Recorder
        self.backend=backend;self.inner=Recorder(str(path),backend='mujoco',scene='rider',seed=0,metadata=metadata)
        self.desired=np.zeros(29);self.phase='contact_settle';self.time_before=0.
    def __getattr__(self,name):return getattr(self.inner,name)
    def tick(self,state,**kwargs):
        kwargs['action']=self.backend.last_tau.copy()
        kwargs['action_source']='actual_joint_torque_Nm_via_original_PD'
        kwargs['command']=dict(kwargs['command'],torque_desired_Nm=self.desired.tolist(),
            experimental_phase=self.phase,time_before=self.time_before,time_after=float(state['time']),
            actual_qfrc_body_Nm=self.backend.data.qfrc_applied[self.backend.robot_dof_adr].copy().tolist())
        self.inner.tick(state,**kwargs)


def run_balance_trial(output,mode='qp',feedback=False,tail=None,physical_prefix=None,board_se2=None):
    import contextlib,io,json,time
    from pathlib import Path
    from dataclasses import asdict
    from scipy.spatial import ConvexHull
    from skate_sim.models.scene import normalize_scene_config
    from skate_sim.models.gripper import manifest
    from skate_sim.backends.mujoco import MujocoBackend
    from skate_sim.record import numerics_manifest,compiled_model_hash
    from .tail_catch import static_pd_balance,body_geoms,point,ik,analytic_support_shift,robot_com
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    scene_values=dict(gripper='right',robot_pose=dict(pos=[-.12,-.35,.793],quat=[1,0,0,0]))
    if board_se2 is not None:
        if physical_prefix:raise ValueError('A changed initial board layout cannot reuse a different-layout physical prefix')
        xyzw=Rotation.from_euler('z',board_se2[2]).as_quat();scene_values['board_pose']=dict(pos=[board_se2[0],board_se2[1],.1125],quat=xyzw[[3,0,1,2]].tolist())
    scene=normalize_scene_config(scene_values)
    with contextlib.redirect_stdout(io.StringIO()):
        b=MujocoBackend(scene_config=scene,controller='pose',episode_time_limit=None,
                       out_of_bounds_distance=None,offboard_distance=None)
    static_target,static=static_pd_balance(b);b.pose_target=static_target.copy()
    prefix_rows=None;prefix_audit=None
    if physical_prefix is not None:
        if not tail:raise ValueError('A verified physical prefix is only a toe-continuation input')
        from skate_sim.replay import read_record
        prefix_rows=read_record(physical_prefix)
        initial=next(r for r in prefix_rows if r['event']=='initial')
        b.set_state(initial['state']);b.import_replay_state(initial['runtime_state'])
    rec=TorqueRecorder(b,output,dict(numerics=numerics_manifest(b),gripper_attachment=manifest(),mode=mode,
        action_semantics='action29 actual Nm, command.torque_desired_Nm requested; pose_target29 is exact inverse-PD. State-before=initial or previous tick.state; state-after=this tick.state.',
        controller='full56 inverse dynamics QP,100Hz; physical500Hz; reactions predicted only',
        known_other='native limit/frictionloss reaction frozen at measured actual state for local prediction; not applied externally',
        initialization='root lowering to foot contact once, followed by explicit50ms native-PD settle shared with baseline'))
    kwargs=dict(scene_config=scene,controller='pose',episode_time_limit=None,out_of_bounds_distance=None,
                offboard_distance=None,config=asdict(b.config))
    rec.start(b.get_state(),backend_kwargs=kwargs,runtime_state=b.export_replay_state());b.recorder=rec
    rec.write('asset_model',files=[dict(sha256=compiled_model_hash(b),kind='compiled_native_model')])
    toe_stages=None;toe_stage_checks={};tail_start=None;toe_plan=None;toe_initial=None;toe_rotation=None;tail_impulse=0.;toe_release=None
    controller=None;swing=None;desired=np.zeros(29);diagnostics={};rows=[];qp_rows=[];phase='contact_settle'
    support_geoms=[set(body_geoms(b,s+'_ankle_roll_link')) for s in ['left','right']]
    footids=[b.model.body(s+'_ankle_roll_link').id for s in ['left','right']]
    root=b.model.body('pelvis').id;ground=b.model.geom('ground').id
    force=np.zeros(6);body_velocity=np.zeros(6);unloaded=None;unload_good=0;lift_start=None
    longest_clear=clear=0.;stop='time budget';failure=None;start=time.perf_counter()
    initial_foot_positions=None;right_hull=None;prefix_state=None;last_qp_time=None;posture_ref=None
    boundary_refreshes=0;max_held_violation=0.
    k_start=0
    if prefix_rows is not None:
        import hashlib
        source_path=Path(physical_prefix)/'events.jsonl'
        digest=hashlib.sha256(source_path.read_bytes()).hexdigest()
        maximum_error=0.;prefix_ticks=0
        for source in prefix_rows:
            if source['event']=='control_update':
                if controller is None:
                    controller=ContactBalance(b);controller.feedback_enabled=feedback
                    controller.solve(controller.com0,np.zeros(3),np.zeros(3))
                    right_xy=np.array([b.data.geom_xpos[g,:2] for g in support_geoms[1]])
                    right_hull=ConvexHull(right_xy);target_com=controller.com0.copy();target_com[:2]=right_xy[right_hull.vertices].mean(axis=0);target_com[2]-=.08
                    initial_foot_positions=controller.footposes.copy()
                desired=np.array(source['torque_desired_Nm']);controller.previous_tau=desired.copy()
                rec.write('control_update',**{k:v for k,v in source.items() if k not in ('event','wall_time')})
            elif source['event']=='tick':
                rec.desired=desired.copy();rec.phase='VERIFIED ORIGINAL CONTROL PREFIX';rec.time_before=float(b.data.time)
                b.step_recorded(source['action'],source['command'],source['inputs'])
                error=max(float(np.max(abs(b.data.qpos-np.array(source['state']['qpos'])))),float(np.max(abs(b.data.qvel-np.array(source['state']['qvel'])))))
                maximum_error=max(maximum_error,error);prefix_ticks+=1
                if error>1e-7:raise RuntimeError(f'Prefix independent integration mismatch{error}')
        k_start=prefix_ticks
        original_report=json.loads((Path(physical_prefix)/'report.json').read_text())
        if not original_report['single_support_lift_pass']:raise RuntimeError('Prefix did not pass actual single-support guard')
        lift_start=original_report['actual_unload']['time'];unloaded=original_report['actual_unload'];longest_clear=original_report['left_airborne_hold_seconds']
        reference=target_com.copy();vref=np.zeros(3);aref=np.zeros(3);fraction=1.
        foot=b.model.body('left_ankle_roll_link').id;toe_rotation=b.data.xmat[foot].reshape(3,3).copy();toe_initial=b.data.xpos[foot]+toe_rotation@np.array([.12,0,-.03])
        deck_z=float(b.data.xpos[b.model.body('deck').id,2]);hover=np.array([-.36,0,.21])
        hover_R,hover_audit=toe_reference(b,hover);press_R,press_audit=toe_reference(b,np.array([-.36,0,.15]))
        toe_plan=dict(hover=hover,hover_R=hover_R,press_R=press_R,deck_z=deck_z,hover_audit=hover_audit,press_audit=press_audit);tail_start=float(b.data.time)
        prefix_audit=dict(source=str(physical_prefix),events_sha256=digest,physics_ticks=prefix_ticks,max_qpos_qvel_error=maximum_error,control_semantics='original recorded control prefix plus new column-equilibrated closed-loop continuation; not a fresh new-solver transfer')
        rec.write('verified_prefix',**prefix_audit);rec.write('toe_plan',time=tail_start,plan=toe_plan)
    try:
        for k in range(k_start,6500 if tail else 4500):
            t=k*.002;rec.time_before=float(b.data.time)
            if k==25:
                controller=ContactBalance(b);controller.feedback_enabled=feedback
                initial_foot_positions=controller.footposes.copy()
                right_xy=np.array([b.data.geom_xpos[g,:2] for g in support_geoms[1]])
                right_hull=ConvexHull(right_xy);target_com=controller.com0.copy()
                target_com[:2]=right_xy[right_hull.vertices].mean(axis=0);target_com[2]-=.08
                initial_robot_q=b.data.qpos.copy();ankles={s:point(b,s+'_ankle_roll_link',[0,0,0]) for s in ['left','right']}
                if mode=='pd':pd_root_y,_=analytic_support_shift(b,initial_robot_q,ankles)
            if k<25:
                b.pose_target=static_target.copy();desired=b.policy_values['kp']*(b.pose_target-b._robot_q())-b.policy_values['kd']*b._robot_dq()
            elif k%5==0:
                elapsed=t-.05
                if elapsed<5.:
                    phase='stand';reference=controller.com0.copy();vref=np.zeros(3);aref=np.zeros(3);fraction=0.
                else:
                    phase='transfer';duration=2.25 if feedback else 1.5;u=float(np.clip((elapsed-5.)/duration,0,1))
                    fraction=10*u**3-15*u**4+6*u**5
                    speed=(30*u**2-60*u**3+30*u**4)/duration if u<1 else 0.
                    acc=(60*u-180*u**2+120*u**3)/duration**2 if u<1 else 0.
                    delta=target_com-controller.com0;reference=controller.com0+fraction*delta;vref=speed*delta;aref=acc*delta
                swing=None
                if lift_start is not None:
                    phase='left_lift_hold';u=float(np.clip((t-lift_start)/.5,0,1));s=10*u**3-15*u**4+6*u**5
                    sv=(30*u**2-60*u**3+30*u**4)/.5 if u<1 else 0.
                    sa=(60*u-180*u**2+120*u**3)/.5**2 if u<1 else 0.
                    offset=np.array([0,0,.03]);swing=(controller.footposes[0]+s*offset,sv*offset,sa*offset)
                if tail_start is not None and toe_stages is None:
                    elapsed_toe=t-tail_start;phase='toe_approach'
                    local_toe=np.array([.12,0,-.03]);duration_toe=1.5
                    u=float(np.clip(elapsed_toe/duration_toe,0,1));ss=10*u**3-15*u**4+6*u**5
                    sv=(30*u**2-60*u**3+30*u**4)/duration_toe if u<1 else 0.
                    sa=(60*u-180*u**2+120*u**3)/duration_toe**2 if u<1 else 0.
                    target=toe_plan['hover'];delta=target-toe_initial
                    pos=toe_initial+ss*delta;vel=sv*delta;acc=sa*delta
                    rotvec=Rotation.from_matrix(toe_plan['hover_R']@toe_rotation.T).as_rotvec()
                    rotation=Rotation.from_rotvec(ss*rotvec).as_matrix()@toe_rotation;omega=sv*rotvec;alpha=sa*rotvec
                    if elapsed_toe>=duration_toe:
                        phase='toe_press_pulse' if tail=='press' else 'toe_hover_control'
                        pulse_elapsed=elapsed_toe-duration_toe
                        if tail=='press' and toe_release is None:
                            u=float(np.clip(pulse_elapsed/.16,0,1));ss=10*u**3-15*u**4+6*u**5
                            sv=(30*u**2-60*u**3+30*u**4)/.16 if u<1 else 0.;sa=(60*u-180*u**2+120*u**3)/.16**2 if u<1 else 0.
                            deck=b.model.body('deck').id;DR=b.data.xmat[deck].reshape(3,3)
                            local=np.array([-.36,0,.21-toe_plan['deck_z']+ss*(.15-.21)])
                            pos=b.data.xpos[deck]+DR@local
                            J,_,_,_=controller.jac(b.data,pos,deck)
                            vel=J@b.data.qvel+DR@np.array([0,0,sv*(.15-.21)]);acc=DR@np.array([0,0,sa*(.15-.21)])
                            rdelta=Rotation.from_matrix(toe_plan['press_R']@toe_plan['hover_R'].T).as_rotvec()
                            rotation=Rotation.from_rotvec(ss*rdelta).as_matrix()@toe_plan['hover_R'];omega=sv*rdelta;alpha=sa*rdelta
                            if pulse_elapsed>=.16:toe_release=t
                        if toe_release is not None:
                            phase='toe_retract';u=float(np.clip((t-toe_release)/.30,0,1));ss=10*u**3-15*u**4+6*u**5
                            pos=toe_plan['hover']+np.array([0,0,.06*ss]);vel=np.zeros(3);acc=np.zeros(3);rotation=toe_plan['hover_R'];omega=np.zeros(3);alpha=np.zeros(3)
                    controller.allow_board_support=True
                    if phase=='toe_press_pulse' and rows and rows[-1].get('toe_tail_normal_N',0)>0:
                        reference=reference.copy();reference[:2]=.8*right_xy[right_hull.vertices].mean(axis=0)+.2*pos[:2]
                    swing=dict(local=local_toe,pos=pos,vel=vel,acc=acc,rotation=rotation,omega=omega,alpha=alpha)
                if toe_stages is not None:
                    swing,toe_stage_checks=toe_stages.command(tail_impulse,nose_angle if rows else 0.)
                    phase=toe_stages.phase;controller.allow_board_support=True
                    if swing is None:stop=toe_stages.reason;break
                if mode=='qp':
                    try:desired,diagnostics=controller.solve(reference,vref,aref,left_load=.5*(1-fraction),swing=swing)
                    except RuntimeError as e:failure=str(e);stop='QP rejected';break
                    qp_rows.append(dict(time=t,phase=phase,torque_desired_Nm=desired.tolist(),**diagnostics))
                    rec.write('control_update',time=t,phase=phase,torque_desired_Nm=desired,qp=diagnostics,
                              state_time=float(b.data.time),reference_com=reference)
                else:
                    if phase=='stand':b.pose_target=static_target.copy()
                    else:
                        q=initial_robot_q.copy();q[1]=-.35+fraction*(pd_root_y+.35);q[2]=initial_robot_q[2]+fraction*(.70-initial_robot_q[2])
                        q,_=ik(b,q,b.robot_joint_names[:12],[(s+'_ankle_roll_link',np.zeros(3),ankles[s],np.eye(3)) for s in ['left','right']],regularization=.0001)
                        b.pose_target=q[b.robot_qpos_adr]+np.array(static['pd_target_offset_rad'])
            if mode=='qp' and k>=25:
                kp=b.policy_values['kp'];kd=b.policy_values['kd'];qnow=b._robot_q();vnow=b._robot_dq()
                limit=b.policy_values['joint_effort_limit']
                reachable_lo=np.maximum(-limit,kp*(b.policy_values['joint_pos_lower_limit']-qnow)-kd*vnow)
                reachable_hi=np.minimum(limit,kp*(b.policy_values['joint_pos_upper_limit']-qnow)-kd*vnow)
                violation=float(np.maximum(reachable_lo-desired,desired-reachable_hi).max())
                max_held_violation=max(max_held_violation,violation)
                if violation>1e-3:
                    boundary_refreshes+=1
                    try:desired,diagnostics=controller.solve(reference,vref,aref,left_load=.5*(1-fraction),swing=swing)
                    except RuntimeError as e:failure=str(e);stop='boundary QP rejected';break
                    qp_rows.append(dict(time=t,phase=phase,reason='PD reachable box changed',held_violation_Nm=violation,torque_desired_Nm=desired.tolist(),**diagnostics))
                    rec.write('control_update',time=t,phase=phase,reason='PD reachable box changed',held_violation_Nm=violation,torque_desired_Nm=desired,qp=diagnostics)
                controller.apply(desired)
            rec.phase=phase;rec.desired=desired.copy();b.set_gripper_command(0.)
            b.step(update_policy=False)
            actual=b.data.qfrc_applied[b.robot_dof_adr].copy()
            error=float(np.max(abs(actual-desired))) if k>=25 and mode=='qp' else 0.
            if np.any(b.data.xfrc_applied) or np.any(b.data.qfrc_applied[:6]) or np.any(b.data.ctrl[:6]):raise RuntimeError('Forbidden root/body input')
            if not np.isfinite(b.data.qpos).all() or not np.isfinite(b.data.qvel).all():raise RuntimeError('Nonfinite physical state')
            loads=np.zeros(2);contacts=[0,0]
            for i in range(b.data.ncon):
                c=b.data.contact[i];pair={int(c.geom1),int(c.geom2)}
                if ground not in pair:continue
                mujoco.mj_contactForce(b.model,b.data,i,force)
                if force[0]<=1e-5:continue
                for side,geoms in enumerate(support_geoms):
                    if pair&geoms:loads[side]+=force[0];contacts[side]+=1
            deck_id=b.model.body('deck').id;deck_R=b.data.xmat[deck_id].reshape(3,3)
            nose=b.data.xpos[deck_id]+deck_R@np.array([.4,0,.03]);tail_point=b.data.xpos[deck_id]+deck_R@np.array([-.4,0,.03])
            nose_angle=float(np.arctan2(nose[2]-tail_point[2],np.linalg.norm(nose[:2]-tail_point[:2])))
            toe_force=0.;toe_contact_points=[];toe_contact_geom_ids=[]
            for contact_index in range(b.data.ncon):
                contact=b.data.contact[contact_index];pair={int(contact.geom1),int(contact.geom2)}
                if controller is None:continue
                if not(pair&support_geoms[0] and pair&controller.board_geoms):continue
                local=deck_R.T@(contact.pos-b.data.xpos[deck_id])
                if local[0]>-.30:continue
                mujoco.mj_contactForce(b.model,b.data,contact_index,force)
                if force[0]<=1e-5:continue
                toe_force+=float(force[0]);toe_contact_points.append(local.tolist());toe_contact_geom_ids.append([int(contact.geom1),int(contact.geom2)])
            tail_impulse+=toe_force*.002
            if tail_start is not None and tail=='press' and toe_release is None and tail_impulse>0 and nose_angle>.10:toe_release=float(b.data.time)
            left_foot=b.model.body('left_ankle_roll_link').id;foot_R=b.data.xmat[left_foot].reshape(3,3);toe_actual=b.data.xpos[left_foot]+foot_R@np.array([.12,0,-.03])
            toe_error=float(np.linalg.norm(toe_actual-swing['pos'])) if isinstance(swing,dict) else None
            com=b.data.subtree_com[root].copy();tilt=float(np.arccos(np.clip(b.data.xmat[root].reshape(3,3)[2,2],-1,1)))
            mujoco.mj_objectVelocity(b.model,b.data,mujoco.mjtObj.mjOBJ_BODY,root,body_velocity,0)
            margin=None;clearance=float(min(b.data.geom_xpos[g,2]-b.model.geom_size[g,0] for g in support_geoms[0]))
            if right_hull is not None:
                margin=float(np.min(-(right_hull.equations[:,:2]@com[:2]+right_hull.equations[:,2])/np.linalg.norm(right_hull.equations[:,:2],axis=1)))
                if feedback:
                    Jcom=np.zeros((3,b.model.nv));mujoco.mj_jacSubtreeCom(b.model,b.data,Jcom,root)
                    cv=Jcom@b.data.qvel;xi=com[:2]+cv[:2]/np.sqrt(9.81/controller.com0[2])
                    xi_margin=float(np.min(-(right_hull.equations[:,:2]@xi+right_hull.equations[:,2])/np.linalg.norm(right_hull.equations[:,:2],axis=1)))
                ready=loads[0]<.05*controller.mg and loads[1]>.8*controller.mg and margin>=.01 and tilt<.1
                ready=ready and (not feedback or (xi_margin>=.01 and np.linalg.norm(cv[:2])<=.04))
                unload_good=unload_good+1 if ready and phase=='transfer' else 0
                if unload_good>=25 and lift_start is None:lift_start=t;unloaded=dict(time=t,left_N=loads[0],right_N=loads[1],margin_m=margin)
                if lift_start is not None and clearance>=.02 and contacts[0]==0:
                    clear+=.002;longest_clear=max(longest_clear,clear)
                else:clear=0.
            rows.append(dict(time=float(b.data.time),phase=phase,toe_tail_normal_N=toe_force,toe_tail_impulse_Ns=tail_impulse,toe_stage_measurements=toe_stage_checks,toe_position_world_m=toe_actual.tolist(),toe_target_error_m=toe_error,foot_normal_dot_tail=float(foot_R[:,2]@(deck_R@np.array([.39,0,1.]))/np.sqrt(1+.39**2)),toe_tail_contact_geom_ids=toe_contact_geom_ids,toe_tail_contacts_deck_local_m=toe_contact_points,nose_z_m=float(nose[2]),nose_angle_rad=nose_angle,com=com.tolist(),root_z_m=float(b.data.xpos[root,2]),
                root_tilt_rad=tilt,root_linear_m_s=float(np.linalg.norm(body_velocity[3:])),root_angular_rad_s=float(np.linalg.norm(body_velocity[:3])),
                foot_normal_N=loads.tolist(),foot_ground_contacts=contacts,right_support_margin_m=margin,
                left_min_surface_z_m=clearance,torque_error_inf_Nm=error,torque_desired_Nm=desired.tolist(),torque_actual_Nm=actual.tolist(),
                pose_target_rad=b.pose_target.copy().tolist(),gripper_raw=0.,root_qfrc_zero=True,root_ctrl_zero=True,xfrc_zero=True))
            if k%250==0:print(f'{mode} t={b.data.time:.3f} {phase} tilt={tilt:.4g} load={loads.tolist()} tauerr={error:.3g}',flush=True)
            if k>=25 and error>1e-3 and mode=='qp':stop='PD inverse torque realizability guard';break
            if tilt>.25 or b.data.xpos[root,2]<.62:stop='actual upright guard';break
            if longest_clear>=.25 and tail_start is None:
                if tail is None:stop='single-support lift gate achieved';break
                deck=b.model.body('deck').id;deck_z=float(b.data.xpos[deck,2]);deck_R=b.data.xmat[deck].reshape(3,3);hover=b.data.xpos[deck]+deck_R@np.array([-.36,0,.23-deck_z]);press_goal=b.data.xpos[deck]+deck_R@np.array([-.36,0,.15-deck_z])
                try:
                    hover_R,hover_audit=toe_reference(b,hover,prefer_flat=board_se2 is not None);press_R,press_audit=toe_reference(b,press_goal,prefer_flat=board_se2 is not None)
                except RuntimeError as e:failure=str(e);stop='toe offline reference guard';break
                toe_plan=dict(hover=hover,hover_R=hover_R,press_R=press_R,deck_z=deck_z,hover_audit=hover_audit,press_audit=press_audit)
                toe_start_body=b.model.body('left_ankle_roll_link').id;toe_rotation=b.data.xmat[toe_start_body].reshape(3,3).copy();toe_initial=b.data.xpos[toe_start_body]+toe_rotation@np.array([.12,0,-.03]);tail_start=float(b.data.time)
                if board_se2 is not None:
                    from .toe_stages import ToeStages
                    toe_stages=ToeStages(b,tail=='press',hover_R)
                rec.write('toe_plan',time=tail_start,plan=toe_plan,semantics='kinematic references only; no simulated state writes')
            if toe_stages is None and tail_start is not None and ((tail=='raised' and t>=tail_start+1.96) or (toe_release is not None and t>=toe_release+.35)):
                stop='bounded toe trial completed';break
            if t>=(8.30 if feedback else 7.55) and lift_start is None:stop='actual unloading deadline';break
        import hashlib
        report=dict(initial_board_se2=board_se2,toe_final_stage=toe_stages.phase if toe_stages else None,verified_physical_prefix=prefix_audit,toe_tail_impulse_Ns=tail_impulse,max_nose_angle_rad=max((r['nose_angle_rad'] for r in rows),default=0),max_nose_z_m=max((r['nose_z_m'] for r in rows),default=0),tail_trial=tail,tail_start_time=tail_start,toe_reference=toe_plan,mode=mode,feedback_dcm=feedback,transfer_seconds=2.25 if feedback else 1.5,stop_reason=stop,failure=failure,elapsed_wall_seconds=time.perf_counter()-start,
            simulated_seconds=float(b.data.time),physics_ticks=len(rows)+(prefix_audit['physics_ticks'] if prefix_audit else 0),continuation_physics_ticks=len(rows),metrics_scope='continuation after verified prefix' if prefix_audit else 'entire experiment',qp_updates=len(qp_rows),
            native_full_nv=b.model.nv,body_torque_dim=29,gripper_original_dofs=8,
            actual_unload=unloaded,left_airborne_hold_seconds=longest_clear,
            single_support_lift_pass=longest_clear>=.25,
            stand_5s_completed=bool(rows and rows[-1]['time']>=5.05),
            max_torque_error_Nm=max((r['torque_error_inf_Nm'] for r in rows),default=None),
            max_QP_dynamics_residual=max((r['dynamics_residual_inf'] for r in qp_rows),default=None),
            max_QP_equality_acc_residual=max((r['equality_acceleration_residual_inf'] for r in qp_rows),default=None),
            action_semantics='Actual29 generalized joint torque Nm through unchanged Backend PD; no normalized action. Predicted reactions never applied.',
            control_hz=100,physics_hz=500,boundary_QP_refreshes=boundary_refreshes,max_held_reachable_violation_Nm=max_held_violation,gripper=manifest(),static_initialization=static,
            model_source='native G1+standard board+official right Robotiq; unmodified contact/asset parameters')
        (output/'report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist())+'\n')
        (output/'metrics.json').write_text(json.dumps(rows)+'\n');(output/'qp.json').write_text(json.dumps(qp_rows)+'\n')
        print(json.dumps({k:report[k] for k in ['mode','stop_reason','failure','simulated_seconds','qp_updates','stand_5s_completed','single_support_lift_pass','max_torque_error_Nm','max_QP_dynamics_residual']}),flush=True)
        return report
    finally:rec.close();b.close()


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['qp','pd'],default='qp')
    parser.add_argument('--dcm',action='store_true',help='fixed measured-CoP/DCM feedback variant')
    parser.add_argument('--tail',choices=['press','raised'])
    parser.add_argument('--physical-prefix',type=Path)
    parser.add_argument('--board-se2',nargs=3,type=float)
    a=parser.parse_args()
    run_balance_trial(a.output,a.mode,a.dcm,a.tail,a.physical_prefix,a.board_se2)
