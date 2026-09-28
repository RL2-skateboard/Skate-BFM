from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BoardStyle:
    name: str
    deck_length: float
    deck_width: float
    deck_thickness: float
    wheelbase: float
    track: float
    wheel_radius: float
    wheel_width: float
    deck_mass: float
    truck_mass: float
    wheel_mass: float
    truck_stiffness: float
    truck_damping: float
    truck_angle: float = math.pi / 4
    truck_limit: float = .35
    corner_radius: float = .045
    kick_length: float = .10
    kick_height: float = .035

    @property
    def total_mass(self) -> float:
        return self.deck_mass + 2 * self.truck_mass + 4 * self.wheel_mass

    def as_dict(self) -> dict:
        return asdict(self) | {"total_mass": self.total_mass, "units": "SI", "angle_unit": "radian"}


_STYLES = {
    "standard": BoardStyle("standard", .80, .20, .035, .48, .16, .035, .022, 2.2, .22, .09, 2.0, .08, truck_angle=.785398, truck_limit=.35, corner_radius=.045, kick_length=.10, kick_height=.030),
    "cruiser": BoardStyle("cruiser", .72, .22, .014, .42, .18, .038, .024, 1.3, .24, .10, 2.2, .09, corner_radius=.09, kick_length=.11, kick_height=.035),
    "longboard": BoardStyle("longboard", 1.00, .24, .035, .62, .19, .040, .025, 3.0, .28, .11, 2.4, .10, truck_angle=.785398, truck_limit=.35, corner_radius=.050, kick_length=.12, kick_height=.028),
    "street": BoardStyle("street", .80, .205, .032, .44, .16, .030, .020, 1.9, .20, .075, 2.0, .08, truck_angle=.785398, truck_limit=.35, corner_radius=.040, kick_length=.09, kick_height=.025),
    "downhill": BoardStyle("downhill", 1.08, .255, .042, .72, .21, .045, .028, 3.8, .34, .14, 2.8, .12, truck_angle=.785398, truck_limit=.35, corner_radius=.055, kick_length=.14, kick_height=.022),
}


def board_styles(style: str = "all") -> dict[str, BoardStyle]:
    if style == "all":
        return dict(_STYLES)
    if style not in _STYLES:
        raise ValueError(f"unknown board style {style!r}; choose all or one of {', '.join(_STYLES)}")
    return {style: _STYLES[style]}


def board_hash(style: BoardStyle) -> str:
    payload = json.dumps({'generator': 2, **style.as_dict()}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _wheel_xml(name: str, y: float, z: float, s: BoardStyle) -> str:
    # MuJoCo cylinders use a local Z axis; fromto makes the wheel axis explicit Y.
    return f'''<body name="{name}" pos="0 {y:.6f} {z:.6f}">
      <joint name="{name}_spin" type="hinge" axis="0 1 0" damping=".02"/>
      <geom name="{name}_collision" type="cylinder" fromto="0 {-s.wheel_width/2:.6f} 0 0 {s.wheel_width/2:.6f} 0" size="{s.wheel_radius:.6f}" mass="{s.wheel_mass:.6f}" rgba=".08 .08 .08 1"/>
      <site name="{name}_axle" pos="0 0 0" size=".006" rgba="1 1 0 1"/>
      <site name="{name}_axis" type="capsule" fromto="0 -.04 0 0 .04 0" size=".002" rgba="1 1 0 1" group="3"/>
      <site name="{name}_marker" pos="{s.wheel_radius:.6f} 0 0" size=".004" rgba="1 .2 .1 1"/>
    </body>'''


def deck_meshes(s):
    """Convex strips form a rounded, thin deck without bridging the kick concavity."""
    half = s.deck_length/2
    raw_xs = sorted([-half, half, -half+s.kick_length, half-s.kick_length] +
                    [-half+s.deck_length*i/48 for i in range(49)])
    xs = []
    for x in raw_xs:
        if not xs or abs(x - xs[-1]) > 1e-8:
            xs.append(x)
    widths = []
    heights = []
    for x in xs:
        dx = max(0., abs(x)-(half-s.corner_radius))
        widths.append(s.deck_width/2-s.corner_radius+math.sqrt(max(0.,s.corner_radius**2-dx**2)))
        u = max(0., (abs(x)-(half-s.kick_length))/s.kick_length)
        heights.append(s.kick_height*u*u)
    areas = [(xs[i+1]-xs[i])*(widths[i]+widths[i+1]) for i in range(len(xs)-1)]
    meshes, geoms = [], []
    for i, area in enumerate(areas):
        vertices = []
        for j in (i,i+1):
            for y in (-widths[j], widths[j]):
                for dz in (-s.deck_thickness/2,s.deck_thickness/2):
                    vertices.extend((xs[j],y,heights[j]+dz))
        ident = f'deck_strip_{i}'
        meshes.append(f'<mesh name="{ident}" vertex="{" ".join(format(v,".9g") for v in vertices)}"/>')
        geoms.append(f'<geom name="{ident}" type="mesh" mesh="{ident}" mass="{s.deck_mass*area/sum(areas):.12g}" rgba=".18 .27 .38 1"/>')
    return ''.join(meshes), ''.join(geoms)


def build_board_xml(style: BoardStyle, *, gallery: bool = False, dynamics=None, initial_roll: float = 0.0) -> str:
    dynamics = dynamics or {}
    model = dynamics.get("model", "truck")
    if model not in ("truck", "reduced"):
        raise ValueError("Unknown dynamics model")
    truck_k1 = float(dynamics.get("truck_k1", style.truck_stiffness))
    truck_c = float(dynamics.get("truck_c", style.truck_damping))
    meshes, deck_geoms = deck_meshes(style)
    x_front, x_rear = style.wheelbase / 2, -style.wheelbase / 2
    wheel_z = style.wheel_radius
    deck_z = wheel_z + .060 + style.deck_thickness / 2
    deck_half = style.deck_length / 2
    kick_start = deck_half - style.kick_length
    kick_angle = math.atan2(style.kick_height, style.kick_length)
    trucks = []
    for prefix, x in (("front", x_front), ("rear", x_rear)):
        wheel_local_z = wheel_z - deck_z + style.deck_thickness / 2
        wheels = _wheel_xml(f"{prefix}_left", style.track / 2, wheel_local_z, style)
        wheels += _wheel_xml(f"{prefix}_right", -style.track / 2, wheel_local_z, style)
        trucks.append(f'''<body name="{prefix}_truck" pos="{x:.6f} 0 {-style.deck_thickness/2:.6f}">
          <joint name="{prefix}_truck_tilt" type="hinge" axis="{math.cos(style.truck_angle)} 0 {math.sin(style.truck_angle)*(1 if prefix == 'front' else -1)}" range="{-style.truck_limit} {style.truck_limit}" stiffness="{truck_k1}" damping="{truck_c}"/>
          <geom name="{prefix}_truck_collision" type="box" pos="0 0 -.030" size=".025 .012 .030" mass="{style.truck_mass*.6:.6f}" rgba=".65 .65 .68 1"/>
          <geom name="{prefix}_axle_collision" type="box" pos="0 0 -.060" size=".012 {style.track/2-style.wheel_width/2:.6f} .010" mass="{style.truck_mass*.4:.6f}" rgba=".65 .65 .68 1"/>
          <site name="{prefix}_truck_pivot" size=".005" rgba="1 .3 1 1"/>
           <site name="{prefix}_truck_axis" type="capsule" fromto="0 0 0 {math.cos(style.truck_angle)*.09} 0 {math.sin(style.truck_angle)*.09*(1 if prefix == 'front' else -1)}" size=".002" rgba="1 .3 1 1" group="3"/>
          {wheels}
        </body>''')
    obstacles = '''
      <geom name="gallery_ramp" type="mesh" mesh="ramp" pos="1.3 0 0" rgba=".75 .35 .12 1"/>
      <geom name="gallery_low_platform" type="box" pos="-1.1 .9 .10" size=".35 .35 .10" rgba=".25 .55 .85 1"/>
      <geom name="gallery_rail" type="cylinder" fromto="-.4 -1.1 .15 .4 -1.1 .15" size=".025" rgba=".85 .75 .12 1"/>
    ''' if gallery else ""
    xml = f'''<mujoco model="skate-sim-board-{style.name}-{model}">
      <compiler angle="radian" coordinate="local"/>
      <option gravity="0 0 -9.81" integrator="implicitfast" timestep=".002"/>
      <visual><headlight diffuse=".8 .8 .8" ambient=".35 .35 .35"/></visual>
      <asset>
        {meshes}
        <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1=".2 .23 .26" rgb2=".4 .43 .46"/>
        <material name="floor" texture="grid" texrepeat="10 10" texuniform="true"/>
        <mesh name="ramp" vertex="-.35 -.55 0 .35 -.55 0 -.35 .55 0 .35 .55 0 .35 -.55 .24 .35 .55 .24"/>
      </asset>
      <worldbody>
        <light name="key" pos="0 -2 4" directional="true"/>
        <geom name="ground" type="plane" size="5 5 .1" material="floor"/>
        <geom name="axis_x" type="box" pos="1 0 .002" size="1 .008 .002" rgba=".9 .1 .1 1" contype="0" conaffinity="0"/>
        <geom name="axis_y" type="box" pos="0 1 .002" size=".008 1 .002" rgba=".1 .8 .1 1" contype="0" conaffinity="0"/>
        {obstacles}
        <body name="deck" pos="0 0 {deck_z:.6f}" euler="{initial_roll:.8g} 0 0">
          <freejoint name="deck_free"/>
          <geom name="deck_collision" type="box" size="{(style.deck_length-2*style.kick_length)/2:.6f} {style.deck_width/2:.6f} {style.deck_thickness/2:.6f}" mass="0" contype="0" conaffinity="0" rgba="0 0 0 0" group="5"/>
          {deck_geoms}
          <site name="deck_top" pos="0 0 {style.deck_thickness/2:.6f}" size=".012" rgba="1 .2 .1 1"/>
          <site name="left_foot_anchor" pos="{style.wheelbase*.22:.6f} {style.deck_width*.28:.6f} {style.deck_thickness/2:.6f}" size=".01" rgba="1 .8 .1 1"/>
          <site name="right_foot_anchor" pos="{-style.wheelbase*.22:.6f} {-style.deck_width*.28:.6f} {style.deck_thickness/2:.6f}" size=".01" rgba="1 .8 .1 1"/>
          {"".join(trucks)}
        </body>
      </worldbody>
    </mujoco>'''
    if model == "reduced":
        root = ET.fromstring(xml)
        chassis = root.find("worldbody/body[@name='deck']")
        chassis.set('name', 'chassis')
        # 10% of the deck mass becomes a physical load-bearing central spine.
        ET.SubElement(chassis, 'geom', name='chassis_spine', type='box',
                      pos='0 0 -.02', size=f'{style.wheelbase/2} .01 .005',
                      mass=str(style.deck_mass*.1), rgba='.6 .6 .6 1')
        deck = ET.SubElement(chassis, 'body', name='deck')
        ET.SubElement(deck, 'joint', name='deck_roll', type='hinge', axis='1 0 0',
                      range='-.2 .2', stiffness=str(truck_k1), damping=str(truck_c))
        for child in list(chassis):
            if child.tag in ('geom','site') and child.get('name') != 'chassis_spine':
                chassis.remove(child)
                if child.tag == 'geom' and child.get('mass'):
                    child.set('mass', str(float(child.get('mass'))*.9))
                deck.append(child)
        equality = ET.SubElement(root, 'equality')
        for p, sign in (('front',1),('rear',-1)):
            joint = chassis.find(f"body[@name='{p}_truck']/joint")
            joint.set('axis',f'0 0 {sign}')
            joint.set('range','-.4 .4')
            chassis.find(f"body[@name='{p}_truck']/site[@name='{p}_truck_axis']").set('fromto', f'0 0 0 0 0 {sign*.09}')
            # Source: tilt(axis=-X)=-.577*steer(front=-Z,rear=+Z).
            # Our axes are reversed, preserving q_roll=-.577*q_steer.
            ET.SubElement(equality,'joint',joint1='deck_roll',joint2=p+'_truck_tilt',
                          polycoef='0 -.577 0 0 0', solref='.005 1')
        xml = ET.tostring(root, encoding='unicode')
    return xml
