from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .board import BoardStyle, board_styles, build_board_xml, build_motion_board_xml


def board_scene(style: BoardStyle) -> str:
    return build_board_xml(style, gallery=False)


def gallery_scene(style: BoardStyle) -> str:
    return build_board_xml(style, gallery=True)


def _vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must have {length} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{label} must contain finite numbers")
    values = [float(x) for x in value]
    if length == 4:
        norm = math.sqrt(sum(x*x for x in values))
        if norm < 1e-12:
            raise ValueError(f"{label} must be a nonzero quaternion")
        values = [x/norm for x in values]
    return values


def _pose(value, label):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"pos", "quat"}:
        raise ValueError(f"{label} requires pos and quat")
    return {"pos": _vector(value["pos"], 3, label + ".pos"),
            "quat": _vector(value["quat"], 4, label + ".quat")}


def normalize_scene_config(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("scene configuration must be an object")
    config = dict(version=1, robot=True, board=True, style="standard", preset="separate",
                  robot_pose=None, board_pose=None, obstacles=[], gripper="none")
    if set(value) - set(config):
        raise ValueError(f"unknown scene fields: {sorted(set(value)-set(config))}")
    config.update(copy.deepcopy(value))
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("scene version must be 1")
    if config["gripper"] not in ("none", "left", "right", "both"):
        raise ValueError("gripper must be none/left/right/both")
    if config["gripper"] != "none" and not config["robot"]:
        raise ValueError("gripper requires a robot")
    for key in ("robot", "board"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    try:
        board_styles(config["style"])
    except ValueError:
        raise ValueError("unknown board style") from None
    if config["preset"] not in ("separate", "push_start", "steer_start"):
        raise ValueError("unknown scene preset")
    for key in ("robot_pose", "board_pose"):
        config[key] = _pose(config[key], key)
    if not isinstance(config["obstacles"], list):
        raise ValueError("obstacles must be a list")
    seen = set()
    for obstacle in config["obstacles"]:
        if not isinstance(obstacle, dict) or set(obstacle) != {"id", "kind", "pos", "quat"}:
            raise ValueError("obstacles require id, kind, pos, quat")
        ident = obstacle["id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", ident) or ident in seen:
            raise ValueError("obstacle ids must be unique identifiers")
        seen.add(ident)
        if obstacle["kind"] not in ("ramp", "platform", "rail"):
            raise ValueError("unknown obstacle kind")
        obstacle.update(_pose({k: obstacle[k] for k in ("pos", "quat")}, ident))
    return config


def scene_kind(config: dict) -> str:
    c = normalize_scene_config(config)
    return "rider" if c["robot"] and c["board"] else "robot" if c["robot"] else "board" if c["board"] else "bench"


def build_scene_xml(config: dict) -> str:
    """Compile one common MJCF for both backends, without runtime file includes."""
    c = normalize_scene_config(config)
    root = ET.Element("mujoco", model="skate-sim-scene")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", gravity="0 0 -9.81", timestep=".002", integrator="implicitfast")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", pos="0 -2 4", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="5 5 .1", rgba=".2 .23 .26 1")

    def merge(source):
        for section in source:
            if section.tag in ("compiler", "option", "statistic", "visual"):
                continue
            dest = root.find(section.tag)
            if dest is None:
                dest = ET.SubElement(root, section.tag)
            dest.extend(list(section))

    if c["robot"]:
        path = Path(__file__).parents[3] / "assets/robot/g1/g1_29dof_freebase.xml"
        robot = ET.parse(path).getroot()
        for mesh in robot.findall("asset/mesh"):
            mesh.set("file", str((path.parent / "meshes" / mesh.get("file")).resolve()))
            # Keep serialized model identity stable across MuJoCo's asset cache.
            mesh.set("content_type", "model/stl")
        if c["gripper"] != "none":
            from .gripper import add_grippers
            add_grippers(robot, c["gripper"])
        merge(robot)
    if c["board"]:
        merge(ET.fromstring(build_motion_board_xml(c["style"])))
    for obstacle in c["obstacles"]:
        body = ET.SubElement(world, "body", name="obstacle_" + obstacle["id"],
                             pos=" ".join(map(str, obstacle["pos"])), quat=" ".join(map(str, obstacle["quat"])))
        colors = {"ramp": ".75 .35 .12 1", "platform": ".25 .55 .85 1", "rail": ".85 .75 .12 1"}
        attrs = dict(name="obstacle_" + obstacle["id"] + "_collision", rgba=colors[obstacle["kind"]])
        if obstacle["kind"] == "platform":
            ET.SubElement(body, "geom", type="box", pos="0 0 .1", size=".35 .35 .1", **attrs)
        elif obstacle["kind"] == "rail":
            ET.SubElement(body, "geom", type="cylinder", fromto="-.4 0 .15 .4 0 .15", size=".025", **attrs)
        else:
            asset = root.find("asset")
            if asset is None:
                asset = ET.SubElement(root, "asset")
            name = "obstacle_" + obstacle["id"] + "_mesh"
            ET.SubElement(asset, "mesh", name=name, vertex="-.35 -.55 0 .35 -.55 0 -.35 .55 0 .35 .55 0 .35 -.55 .24 .35 .55 .24")
            ET.SubElement(body, "geom", type="mesh", mesh=name, **attrs)
    return ET.tostring(root, encoding="unicode")


def initialize_scene(model, data, config: dict) -> None:
    """Name-based initial state shared by MuJoCo and mjlab CPU reset staging."""
    import mujoco
    import numpy as np
    c = normalize_scene_config(config)
    if c["robot"]:
        if c["board"]:
            if c["preset"] == "separate":
                data.joint("floating_base_joint").qpos[:3] = (0, -1, model.qpos0[int(model.joint("floating_base_joint").qposadr[0]) + 2])
            else:
                from ..robot.rider_pose import initialize_rider_pose
                initialize_rider_pose(model, data, c["preset"])
    for enabled, key, joint in ((c["robot"], "robot_pose", "floating_base_joint"),
                                (c["board"], "board_pose", "deck_free")):
        if enabled and c[key] is not None:
            data.joint(joint).qpos[:] = c[key]["pos"] + c[key]["quat"]
    if c["board"] and c["preset"] == "steer_start":
        velocity = np.zeros(3)
        mujoco.mju_rotVecQuat(velocity, np.array([1., 0., 0.]), data.joint("deck_free").qpos[3:7])
        data.joint("deck_free").qvel[:3] = velocity
        if c["robot"]:
            data.joint("floating_base_joint").qvel[:3] = velocity
        for end in ("front", "rear"):
            for side in ("left", "right"):
                data.joint(f"{end}_{side}_spin").qvel[:] = 1 / float(model.geom("front_left_collision").size[0])
    if c["gripper"] != "none":
        from .gripper import initialize_grippers
        initialize_grippers(model, data, c["gripper"])
    mujoco.mj_forward(model, data)


def load_scene_config(path) -> dict:
    import json
    with open(path, encoding="utf-8") as file:
        return normalize_scene_config(json.load(file))


def save_scene_config(config: dict, path) -> None:
    import json
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(normalize_scene_config(config), indent=2) + "\n", encoding="utf-8")


initialize_scene_state = initialize_scene
