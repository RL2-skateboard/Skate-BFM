"""Conservative full-geometry clearance certificates for the toe experiment."""
import numpy as np
import mujoco


class FootBoardClearance:
    def __init__(self,model,foot_geoms,board_geoms):
        self.model=model;self.foot=sorted(foot_geoms);self.board=sorted(board_geoms)
        self.mesh={}
        for g in self.foot+self.board:
            if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);self.mesh[g]=model.mesh_vert[a:a+n].copy()
    def bounds(self,data,g):
        m=self.model;position=data.geom_xpos[g];R=data.geom_xmat[g].reshape(3,3);size=m.geom_size[g];kind=int(m.geom_type[g])
        if g in self.mesh:
            points=self.mesh[g]@R.T+position
            return points.min(axis=0),points.max(axis=0),points
        if kind==mujoco.mjtGeom.mjGEOM_SPHERE:extent=np.full(3,size[0])
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:extent=abs(R)@size
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:extent=size[0]*np.linalg.norm(R[:,:2],axis=1)+size[1]*abs(R[:,2])
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=size[0]+size[1]*abs(R[:,2])
        else:raise RuntimeError(f'No certified AABB for foot/board geometry{kind}')
        return position-extent,position+extent,None
    def measure(self,data,threshold=.03):
        bounds={g:self.bounds(data,g) for g in self.foot+self.board}
        minimum=np.inf;method_counts=dict(aabb=0,sphere_distance=0,mesh_projection=0,uncertified=0)
        fromto=np.zeros(6)
        for foot in self.foot:
            flo,fhi,fpoints=bounds[foot]
            for board in self.board:
                blo,bhi,bpoints=bounds[board]
                gap=np.maximum(np.maximum(blo-fhi,flo-bhi),0.)
                lower=float(np.linalg.norm(gap));method='aabb'
                if lower<threshold:
                    if self.model.geom_type[foot]==mujoco.mjtGeom.mjGEOM_SPHERE:
                        lower=float(mujoco.mj_geomDistance(self.model,data,foot,board,.2,fromto));method='sphere_distance'
                    elif fpoints is not None and bpoints is not None:
                        # Projection intervals on unit axes provide independent
                        # separating lower bounds even when mesh distance is0.
                        axes=np.vstack([np.eye(3),data.geom_xmat[foot].reshape(3,3).T,data.geom_xmat[board].reshape(3,3).T])
                        ap=fpoints@axes.T;bp=bpoints@axes.T
                        projection=np.maximum(np.maximum(bp.min(axis=0)-ap.max(axis=0),ap.min(axis=0)-bp.max(axis=0)),0.)
                        lower=max(lower,float(projection.max()));method='mesh_projection'
                    else:method='uncertified'
                method_counts[method]+=1;minimum=min(minimum,lower)
        foot_min_z=min(bounds[g][0][2] for g in self.foot);board_max_z=max(bounds[g][1][2] for g in self.board)
        contacts=[]
        foot_set=set(self.foot);board_set=set(self.board)
        for c in data.contact[:data.ncon]:
            pair={int(c.geom1),int(c.geom2)}
            if pair&foot_set and pair&board_set:contacts.append(dict(geoms=[int(c.geom1),int(c.geom2)],distance_m=float(c.dist)))
        return dict(whole_foot_board_separation_lower_bound_m=float(minimum),separation_at_least_30mm=bool(minimum>=threshold and not contacts),actual_native_foot_board_contacts=contacts,
            exact_world_vertical_clearance_m=float(foot_min_z-board_max_z),certificate_methods=method_counts)


def contact_top_triangle_certificate(model, data, geom, point_world, outward_world, board_length):
    """Contact-only closed top-triangle witness; four float32 geometry ULPs.

    This is not a broad ray/clearance tolerance. Only the actual participating
    collision mesh's outward top faces are considered. No simulation state changes.
    """
    if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return dict(accepted=False, reason='participating geometry is not mesh')
    mesh=int(model.geom_dataid[geom]);va=int(model.mesh_vertadr[mesh]);vn=int(model.mesh_vertnum[mesh]);fa=int(model.mesh_faceadr[mesh]);fn=int(model.mesh_facenum[mesh])
    verts=model.mesh_vert[va:va+vn].astype(float)@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
    point=np.asarray(point_world,float);out=np.asarray(outward_world,float);out=out/max(np.linalg.norm(out),1e-15)
    tolerance=float(4*np.spacing(np.float32(board_length)));best=None
    for index,face in enumerate(model.mesh_face[fa:fa+fn]):
        if np.any(face<0) or np.any(face>=vn):raise RuntimeError('Unexpected nonlocal compiled mesh face index')
        a,b,c=verts[face];ab=b-a;ac=c-a;normal=np.cross(ab,ac);length=np.linalg.norm(normal)
        if length<1e-15:continue
        normal/=length
        if normal@out<=.5:continue
        projected=point-normal*((point-a)@normal);gram=np.array([[ab@ab,ab@ac],[ab@ac,ac@ac]])
        if abs(np.linalg.det(gram))<1e-24:continue
        uv=np.linalg.solve(gram,np.array([(projected-a)@ab,(projected-a)@ac]));candidates=[]
        if min(uv)>=0 and uv.sum()<=1:candidates.append(projected)
        for first,second in [(a,b),(b,c),(c,a)]:
            edge=second-first;u=float(np.clip((point-first)@edge/max(edge@edge,1e-30),0,1));candidates.append(first+u*edge)
        nearest=min(candidates,key=lambda q:np.linalg.norm(point-q));distance=float(np.linalg.norm(nearest-point))
        if best is None or distance<best['distance_m']:best=dict(face_index=index,distance_m=distance,normal_alignment=float(normal@out),closest_world_m=nearest.tolist())
    return dict(accepted=bool(best is not None and best['distance_m']<=tolerance),tolerance_m=tolerance,method='participating-mesh outward closed triangle,4float32ULP(board_length)',nearest=best)
