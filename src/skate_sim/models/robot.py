from __future__ import annotations

import math
import tempfile
from pathlib import Path

from ..robot.contract import G1_29_JOINTS


def robot_xml() -> str:
    joints = []
    for i, name in enumerate(G1_29_JOINTS):
        side = -1 if name.startswith("left") else 1 if name.startswith("right") else 0
        x = -.18 if side < 0 else .18 if side > 0 else 0.0
        z = 1.1 - .025 * min(i, 10)
        joints.append(f'''<body name="{name}_link" pos="{x:.3f} 0 {z:.3f}">
          <joint name="{name}" type="hinge" axis="0 1 0" range="-1.5 1.5" damping=".2"/>
          <geom name="{name}_collision" type="capsule" fromto="0 0 -.06 0 0 .06" size=".025" mass=".15" rgba=".35 .48 .62 1"/>
        </body>''')
    return f'''<mujoco model="skate-sim-robot-ground">
      <compiler angle="radian"/>
      <option gravity="0 0 -9.81" timestep=".002" integrator="implicitfast"/>
      <worldbody>
        <light name="key" pos="0 -2 4" directional="true"/>
        <geom name="ground" type="plane" size="5 5 .1" rgba=".18 .2 .22 1"/>
        <body name="robot" pos="0 0 1.2">
          <freejoint name="robot_free"/>
          <geom name="pelvis" type="box" size=".18 .12 .12" mass="5" rgba=".2 .4 .7 1"/>
          {''.join(joints)}
        </body>
      </worldbody>
    </mujoco>'''


def official_robot_scene() -> Path:
    """Return the vendored official deploy-branch G1 scene."""
    path = Path(__file__).parents[3] / "assets" / "robot" / "g1" / "scene_29dof_freebase.xml"
    if not path.exists():
        raise FileNotFoundError(f"official G1 scene missing: {path}")
    return path


def inspection_scene() -> Path:
    """Official asset with an explicit, releasable pelvis/world weld.

    Retain the free joint and its observation addresses. The weld is an
    inspection fixture, never used for free-base pose or policy evaluation.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(official_robot_scene()).getroot()
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="inspection_base", body1="pelvis",
                  solref=".002 1", solimp=".999 .999 .001")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".xml",
                                     prefix=".inspection-", dir=official_robot_scene().parent,
                                     delete=False) as file:
        file.write(ET.tostring(root, encoding="unicode"))
        return Path(file.name)


def rider_scene(style: str = "standard", preset: str = "separate", dynamics=None) -> Path:
    """Compose the official G1 free-base asset with a real movable board.

    The board and the G1 remain independent bodies. Presets only choose their
    initial state; no support geometry or equality constraint is generated.
    """
    import xml.etree.ElementTree as ET
    from .board import board_styles, build_board_xml
    robot_root = ET.parse(official_robot_scene()).getroot()
    board_root = ET.fromstring(build_board_xml(board_styles(style)[style], dynamics=dynamics or {}))
    robot_asset = robot_root.find("asset")
    board_asset = board_root.find("asset")
    for asset_child in list(board_asset):
        if asset_child.tag in ("mesh", "material", "texture"):
            robot_asset.append(asset_child)
    board_world = board_root.find("worldbody")
    ground = board_world.find("geom[@name='ground']")
    old_floor = robot_root.find("worldbody/geom[@name='floor']")
    if old_floor is not None:
        robot_root.find("worldbody").remove(old_floor)
    if ground is not None:
        robot_root.find("worldbody").append(ground)
    deck = board_world.find("body[@name='deck']")
    if deck is None: raise ValueError("board XML has no deck")
    robot_root.find("worldbody").append(deck)
    if preset not in ("separate", "push_start", "steer_start"):
        raise ValueError("unknown rider preset; choose separate, push_start, or steer_start")
    # The deploy MJCF has finite foot contact proxies attached below each
    # ankle-roll body (the four named dummy_l{f,r}_* bodies). Keep those
    # contacts and add no auxiliary support geometry.
    asset_dir = Path(__file__).parents[3] / "assets" / "robot" / "g1"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".xml",
                                     prefix=".rider-", dir=asset_dir, delete=False) as file:
        file.write(ET.tostring(robot_root, encoding="unicode"))
        return Path(file.name)
