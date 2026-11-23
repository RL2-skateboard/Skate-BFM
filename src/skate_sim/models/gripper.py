"""Official Menagerie Robotiq2F-85 assembly; no mechanical parameter retuning."""
from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ASSET = Path(__file__).parents[3]/'assets/robot/grippers/robotiq_2f85/2f85.xml'
MODEL = 'robotiq_2f85'
RAW_RANGE = (0.,255.)
# Assembly only: flange local+z becomes G1 hand+x. Opening is measured from pads.
FLANGE_POS = '.015 0 0'
FLANGE_QUAT = '0 0.7071067811865476 0 0.7071067811865476'


def sides(selection):
    return () if selection == 'none' else ('left','right') if selection == 'both' else (selection,)


def prefix(side):
    return side+'_2f85_'


def add_grippers(robot, selection):
    for side in sides(selection):
        source=ET.parse(ASSET).getroot()
        # MJCF otherwise infers mesh name from file; retain it before changing path.
        for mesh in source.findall('asset/mesh'):
            mesh.set('name',mesh.get('name',Path(mesh.get('file')).stem))
            mesh.set('file',str((ASSET.parent/'assets'/mesh.get('file')).resolve()))
        names={e.get('name') for e in source.iter() if e.get('name')}
        classes={e.get('class') for e in source.findall('.//default') if e.get('class')}
        references={'class','childclass','mesh','material','texture','joint','joint1','joint2',
                    'body1','body2','site','site1','site2','geom','geom1','geom2','tendon','target'}
        for element in source.iter():
            if element.get('name'):
                element.set('name',prefix(side)+element.get('name'))
            for key in references:
                value=element.get(key)
                if value in names or value in classes:
                    element.set(key,prefix(side)+value)
        hand=robot.find(f".//body[@name='{side}_rubber_hand']")
        if hand is None:raise ValueError(f'Missing wrist attachment: {side}')
        for geom in hand.findall('geom'):
            geom.set('contype','0')
            geom.set('conaffinity','0')
            geom.set('rgba','0 0 0 0')
        flange=ET.SubElement(hand,'body',name=side+'_gripper',pos=FLANGE_POS,quat=FLANGE_QUAT)
        flange.extend(list(source.find('worldbody')))
        for section in source:
            if section.tag in ('compiler','option','worldbody'):continue
            destination=robot.find(section.tag)
            if destination is None:destination=ET.SubElement(robot,section.tag)
            destination.extend(list(section))


def initialize_grippers(model,data,selection):
    for side in sides(selection):
        data.ctrl[model.actuator(prefix(side)+'fingers_actuator').id]=0.


def command_grippers(model,data,commands):
    for side,command in commands.items():
        data.ctrl[model.actuator(prefix(side)+'fingers_actuator').id]=command


def measured_opening(model,data,side):
    """Geometric pad-centre separation minus inward box support, not a ctrl scale."""
    p=prefix(side)
    first=model.geom(p+'right_pad1').id
    second=model.geom(p+'left_pad1').id
    delta=data.geom_xpos[second]-data.geom_xpos[first]
    normal=delta/np.linalg.norm(delta)
    extent=sum(float(np.abs(data.geom_xmat[g].reshape(3,3).T@normal)@model.geom_size[g]) for g in (first,second))
    return float(np.linalg.norm(delta)-extent)


def tcp_pose(model,data,side):
    """Official pinch site's world pose (not the old synthetic pad frame)."""
    sid=model.site(prefix(side)+'pinch').id
    return data.site_xpos[sid].copy(),data.site_xmat[sid].reshape(3,3).copy()


def manifest():
    import hashlib
    import json
    source=json.loads((ASSET.parent/'provenance.json').read_text())
    return dict(model=MODEL,repository=source['repository'],revision=source['revision'],
        source_xml_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        flange_pos_m=[.015,0,0],flange_quat_wxyz=[0,2**-.5,0,2**-.5],
        assembly_status='project wrist adapter, not manufacturer-certified mounting',
        command='official tendon raw0=open255=close; not metres or per-finger force',
        global_options='project scene solver/cone/impratio retained to avoid changing skateboard physics; standalone asset cone=elliptic/impratio10 differs')


def compare_mechanics(model, side):
    """Name-matched compiled properties; IDs/whole-world placement may differ."""
    import mujoco
    source=mujoco.MjModel.from_xml_path(str(ASSET))
    diffs=[];counts={}
    fields=dict(body=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat'],
                joint=['jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_stiffness'],
                geom=['geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity','geom_condim','geom_priority'],
                actuator=['actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','actuator_gaintype','actuator_biastype'])
    sizes=dict(body=source.nbody,joint=source.njnt,geom=source.ngeom,actuator=source.nu)
    # Several upstream geoms have no name; pair by their body-local ordinal.
    for kind in fields:
        checked=0
        for i in range(sizes[kind]):
            if kind=='body' and i==0:continue
            obj={'body':mujoco.mjtObj.mjOBJ_BODY,'joint':mujoco.mjtObj.mjOBJ_JOINT,'geom':mujoco.mjtObj.mjOBJ_GEOM,'actuator':mujoco.mjtObj.mjOBJ_ACTUATOR}[kind]
            name=mujoco.mj_id2name(source,obj,i)
            if name:j=mujoco.mj_name2id(model,obj,prefix(side)+name)
            else:
                parent=int(source.geom_bodyid[i]);bn=mujoco.mj_id2name(source,mujoco.mjtObj.mjOBJ_BODY,parent)
                source_ids=np.flatnonzero(source.geom_bodyid==parent)
                dest_ids=np.flatnonzero(model.geom_bodyid==model.body(prefix(side)+bn).id)
                j=int(dest_ids[list(source_ids).index(i)])
            for field in fields[kind]:
                if not np.allclose(getattr(source,field)[i],getattr(model,field)[j],rtol=0,atol=1e-12):diffs.append(f'{kind}:{name or i}:{field}')
            if kind=='joint':
                a=int(source.jnt_dofadr[i]);b=int(model.jnt_dofadr[j])
                for field in ['dof_armature','dof_damping','dof_frictionloss']:
                    if not np.allclose(getattr(source,field)[a],getattr(model,field)[b],rtol=0,atol=1e-12):diffs.append(f'joint:{name}:{field}')
            checked+=1
        counts[kind]=checked
    ids=[model.body(prefix(side)+source.body(i).name).id for i in range(1,source.nbody)]
    return dict(property_differences=diffs,checked_objects=counts,asset_total_mass_kg=float(source.body_mass.sum()),
        mounted_asset_mass_kg=float(model.body_mass[ids].sum()),source_joint_count=source.njnt,
        source_equality_count=source.neq,source_tendon_count=source.ntendon,
        source_cone=int(source.opt.cone),project_cone=int(model.opt.cone),
        source_impratio=float(source.opt.impratio),project_impratio=float(model.opt.impratio))


def pad_tcp(model,data,side):
    """Actual inner-pad midpoint and official base axes, including linkage motion."""
    p=prefix(side)
    a=model.geom(p+'right_pad1').id;b=model.geom(p+'left_pad1').id
    return .5*(data.geom_xpos[a]+data.geom_xpos[b]),tcp_pose(model,data,side)[1]
