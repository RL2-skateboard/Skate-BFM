"""Phase-wise, read-only auxiliary-reward audit for recorded HUSKY rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import mujoco
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_collection.rollout_split import randomize_husky_play_physics  # noqa: E402

from skate_bfm.integration.actions import BFM0_JOINTS  # noqa: E402

PHASE_GROUPS = ("push", "push2steer", "steer", "steer2push")
PHASE_COLORS = {
    "push": "#15803d",
    "push2steer": "#d97706",
    "steer_left": "#2563eb",
    "steer_right": "#7c3aed",
    "steer_forward": "#0891b2",
    "steer2push": "#dc2626",
    "fall": "#475569",
}
REWARD_COLUMNS = (
    "penalty_torques",
    "penalty_action_rate_29d",
    "penalty_action_rate_23d",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori_world",
    "penalty_feet_ori_surface",
    "penalty_ankle_roll",
    "penalty_slippage_world",
    "penalty_slippage_board",
    "penalty_slippage_ground",
    "original_weighted_aux",
    "skate_candidate_aux",
    "skate_candidate_aux_world_ori",
    "skate_candidate_aux_surface_ori",
)
CSV_COLUMNS = (
    "rollout",
    "frame",
    "time",
    "phase",
    "phase_group",
    *REWARD_COLUMNS,
    "left_contact",
    "right_contact",
    "nonfoot_ground_contacts",
    "nonfoot_board_contacts",
    "board_roll_rad",
    "left_ankle_roll_sq",
    "right_ankle_roll_sq",
    "ignored_wrist_action_rate_contribution",
)


@dataclass
class RawRollout:
    path: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def name(self) -> str:
        return str(self.metadata["episode_id"])

    @property
    def phase_mapping(self) -> dict[int, str]:
        return {
            int(key): str(value)
            for key, value in self.metadata["phase_mapping"].items()
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout",
        type=Path,
        action="append",
        required=True,
        help="Raw rollout NPZ; repeat for independent phase-rich rollouts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "m2.4b-1-reward-audit",
    )
    parser.add_argument(
        "--fidelity-root-rmse-max",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--fidelity-joint-rmse-max",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--fidelity-board-rmse-max",
        type=float,
        default=0.01,
    )
    return parser.parse_args()


def read_raw_rollout(path: Path) -> RawRollout:
    path = path.expanduser().resolve()
    metadata_path = path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing raw rollout metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "action",
        "body_pos",
        "body_quat",
        "board_root_lin_vel",
        "board_root_pos",
        "board_root_quat",
        "dof_pos",
        "phase_id",
        "qpos",
        "qvel",
        "root_pos",
        "root_quat",
        "sim_time",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{path} is missing required raw fields: {missing}")
    frame_count = int(metadata["num_frames"])
    for name, value in arrays.items():
        if value.ndim and value.shape[0] != frame_count:
            raise ValueError(
                f"{path}:{name} has {value.shape[0]} frames, expected {frame_count}"
            )
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"{path}:{name} contains NaN or Inf")
    if arrays["action"].shape != (frame_count, 23):
        raise ValueError(f"{path}: expected 23D action, got {arrays['action'].shape}")
    return RawRollout(path=path, metadata=metadata, arrays=arrays)


def phase_runs(rollout: RawRollout) -> list[dict[str, Any]]:
    phase_ids = rollout.arrays["phase_id"]
    mapping = rollout.phase_mapping
    runs = []
    start = 0
    for end in range(1, len(phase_ids) + 1):
        if end == len(phase_ids) or phase_ids[end] != phase_ids[start]:
            phase = mapping[int(phase_ids[start])]
            runs.append(
                {
                    "phase": phase,
                    "start": int(start),
                    "end": int(end),
                    "frames": int(end - start),
                    "duration_s": float(
                        rollout.arrays["sim_time"][end - 1]
                        - rollout.arrays["sim_time"][start]
                        + rollout.metadata["dt"]
                    ),
                }
            )
            start = end
    return runs


def phase_group(phase: str) -> str:
    return "steer" if phase.startswith("steer_") else phase


def quat_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(left * right, axis=-1)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def rmse_and_max(value: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(np.square(value)))),
        "max_error": float(np.max(np.abs(value))),
    }


def configure_model(rollout: RawRollout) -> mujoco.MjModel:
    model_path = Path(rollout.metadata["robot_xml"]).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Recorded robot XML is unavailable: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = 0.005
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.ccd_iterations = 50
    physics = rollout.metadata.get("physics_randomization", {})
    if physics.get("enabled", False):
        replay_report, _ = randomize_husky_play_physics(
            model,
            str(rollout.metadata["rollout_id"]),
            int(physics["seed"]),
        )
        if replay_report["seed"] != physics["seed"]:
            raise RuntimeError("Physics randomization seed was not reproduced.")
        mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


def control_spec() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    defaults = np.asarray(
        (
            0.0,
            0.0,
            0.0,
            0.23,
            -0.20,
            0.0,
            -0.7,
            0.0,
            0.0,
            1.17,
            -0.45,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.03,
            0.45,
            -0.21,
            1.32,
            -0.7,
            -0.845,
            0.83,
            1.19,
        ),
        dtype=np.float64,
    )
    scales = np.asarray(
        (
            0.5475,
            0.3507,
            0.5475,
            0.3507,
            0.4386,
            0.4386,
            0.5475,
            0.3507,
            0.5475,
            0.3507,
            0.4386,
            0.4386,
            0.5475,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
            0.4386,
        ),
        dtype=np.float64,
    )
    reindex = np.asarray(
        (
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            0,
            2,
            6,
            8,
            12,
            1,
            3,
            7,
            9,
            13,
            14,
            4,
            5,
            10,
            11,
        ),
        dtype=np.int64,
    )
    return defaults, scales, reindex


def set_recorded_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rollout: RawRollout,
    frame: int,
) -> None:
    defaults, scales, reindex = control_spec()
    data.qpos[:] = rollout.arrays["qpos"][frame]
    data.qvel[:] = rollout.arrays["qvel"][frame]
    data.time = float(rollout.arrays["sim_time"][frame])
    target = defaults + scales * rollout.arrays["action"][frame]
    data.ctrl[:23] = target[reindex]
    mujoco.mj_forward(model, data)


def replay_fidelity(
    rollout: RawRollout,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    defaults, scales, reindex = control_spec()
    recorded = rollout.arrays
    thresholds = {
        "joint_position": args.fidelity_joint_rmse_max,
        "root_position": args.fidelity_root_rmse_max,
        "board_position": args.fidelity_board_rmse_max,
        "board_velocity": 0.05,
        "root_orientation_rad": 0.05,
        "board_orientation_rad": 0.05,
    }
    phase_reports = []
    all_errors: dict[str, list[np.ndarray]] = defaultdict(list)
    phase_occurrence: Counter[str] = Counter()
    for run in phase_runs(rollout):
        phase = run["phase"]
        phase_occurrence[phase] += 1
        model = configure_model(rollout)
        data = mujoco.MjData(model)
        start, end = run["start"], run["end"]
        data.qpos[:] = recorded["qpos"][start]
        data.qvel[:] = recorded["qvel"][start]
        data.time = float(recorded["sim_time"][start])
        mujoco.mj_forward(model, data)
        replay: dict[str, list[np.ndarray]] = defaultdict(list)
        for frame in range(start, end):
            replay["dof_pos"].append(
                np.asarray(
                    [
                        data.qpos[model.joint(name).qposadr[0]]
                        for name in rollout.metadata["joint_order"]
                    ],
                    dtype=np.float64,
                )
            )
            replay["root_pos"].append(data.qpos[:3].copy())
            replay["root_quat"].append(data.qpos[3:7].copy())
            board_joint = model.joint("skateboard/floating_base_joint_skateboard")
            board_qpos = board_joint.qposadr[0]
            board_qvel = board_joint.dofadr[0]
            replay["board_pos"].append(
                data.qpos[board_qpos : board_qpos + 3].copy()
            )
            replay["board_quat"].append(
                data.qpos[board_qpos + 3 : board_qpos + 7].copy()
            )
            replay["board_vel"].append(
                data.qvel[board_qvel : board_qvel + 3].copy()
            )
            if frame == end - 1:
                break
            target = defaults + scales * recorded["action"][frame + 1]
            data.ctrl[:23] = target[reindex]
            for _ in range(4):
                mujoco.mj_step(model, data)
        replay_arrays = {key: np.asarray(value) for key, value in replay.items()}
        errors = {
            "joint_position": replay_arrays["dof_pos"] - recorded["dof_pos"][start:end],
            "root_position": replay_arrays["root_pos"] - recorded["root_pos"][start:end],
            "root_orientation_rad": quat_angle(
                replay_arrays["root_quat"],
                recorded["root_quat"][start:end],
            ),
            "board_position": replay_arrays["board_pos"]
            - recorded["board_root_pos"][start:end],
            "board_orientation_rad": quat_angle(
                replay_arrays["board_quat"],
                recorded["board_root_quat"][start:end],
            ),
            "board_velocity": replay_arrays["board_vel"]
            - recorded["board_root_lin_vel"][start:end],
        }
        metrics = {
            name: rmse_and_max(value)
            for name, value in errors.items()
        }
        passed = all(
            metrics[name]["rmse"] <= threshold
            for name, threshold in thresholds.items()
        )
        phase_reports.append(
            {
                **run,
                "occurrence": phase_occurrence[phase] - 1,
                "metrics": metrics,
                "status": "PASS" if passed else "FAILED",
            }
        )
        for name, value in errors.items():
            all_errors[name].append(value.reshape(-1))
    aggregate = {
        name: rmse_and_max(np.concatenate(values))
        for name, values in all_errors.items()
    }
    passed = all(item["status"] == "PASS" for item in phase_reports)
    return passed, {
        "aggregate": aggregate,
        "phase_reports": phase_reports,
        "alignment": (
            "Each recorded phase is replayed from its recorded start state; "
            "action[t] is applied to reconstruct state[t+1]. This prevents "
            "contact-solver drift in an earlier phase from contaminating later phases."
        ),
        "status": "PASS" if passed else "FAILED",
    }


def bfm_limits() -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    path = (
        REPOSITORY_ROOT
        / "train"
        / "scripts"
        / "isaac_env"
        / "humanoidverse"
        / "config"
        / "robot"
        / "g1"
        / "g1_29dof_hard_waist.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["robot"]
    names = config["dof_names"]
    lower = config["dof_pos_lower_limit_list"]
    upper = config["dof_pos_upper_limit_list"]
    effort = config["dof_effort_limit_list"]
    position_limits = {
        name: (
            float((minimum + maximum) / 2 - 0.95 * (maximum - minimum) / 2),
            float((minimum + maximum) / 2 + 0.95 * (maximum - minimum) / 2),
        )
        for name, minimum, maximum in zip(names, lower, upper, strict=True)
    }
    effort_limits = dict(zip(names, effort, strict=True))
    return position_limits, effort_limits


def body_velocity_at_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    point: np.ndarray,
) -> np.ndarray:
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        0,
    )
    return velocity[3:] + np.cross(velocity[:3], point - data.xpos[body_id])


def body_normal(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3)[:, 2]


def contact_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[dict[str, Any], list[str]]:
    foot_geoms = {
        "left": {
            index
            for index in range(model.ngeom)
            if (model.geom(index).name or "").startswith("robot/left_foot")
        },
        "right": {
            index
            for index in range(model.ngeom)
            if (model.geom(index).name or "").startswith("robot/right_foot")
        },
    }
    board_geoms = {
        index
        for index in range(model.ngeom)
        if (model.geom(index).name or "").startswith("skateboard/")
    }
    terrain_geoms = {
        index
        for index in range(model.ngeom)
        if (model.geom(index).name or "") == "terrain"
    }
    board_body = model.body("skateboard/board_tilt_body").id
    feet = {
        "left": model.body("robot/left_ankle_roll_link").id,
        "right": model.body("robot/right_ankle_roll_link").id,
    }
    candidates: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    pairs = []
    undesired = False
    nonfoot_ground = 0
    nonfoot_board = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        first_name = model.geom(first).name or f"geom_{first}"
        second_name = model.geom(second).name or f"geom_{second}"
        force = np.empty(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        magnitude = float(np.linalg.norm(force[:3]))
        pair = f"{first_name} <-> {second_name}"
        pairs.append(pair)
        robot_names = (first_name, second_name)
        if any(
            name.startswith("robot/")
            and any(token in name for token in ("pelvis", "shoulder", "hip"))
            for name in robot_names
        ) and np.any(np.abs(force[:3]) > 1.0):
            undesired = True

        for side, geom_ids in foot_geoms.items():
            foot_geom = first if first in geom_ids else second if second in geom_ids else None
            if foot_geom is None:
                continue
            other = second if foot_geom == first else first
            if other in terrain_geoms:
                contact_type = "ground"
            elif other in board_geoms:
                contact_type = "skateboard"
            else:
                contact_type = "other"
            candidates[side].append(
                {
                    "type": contact_type,
                    "force": magnitude,
                    "point": np.asarray(contact.pos, dtype=np.float64).copy(),
                    "normal": np.asarray(contact.frame[:3], dtype=np.float64).copy(),
                    "foot_body": feet[side],
                    "other_body": int(model.geom_bodyid[other]),
                }
            )
            break
        else:
            touches_terrain = first in terrain_geoms or second in terrain_geoms
            touches_board = first in board_geoms or second in board_geoms
            touches_robot = first_name.startswith("robot/") or second_name.startswith("robot/")
            if touches_robot and touches_terrain:
                nonfoot_ground += 1
            if touches_robot and touches_board:
                nonfoot_board += 1

    metrics = {
        "penalty_undesired_contact": float(undesired),
        "nonfoot_ground_contacts": nonfoot_ground,
        "nonfoot_board_contacts": nonfoot_board,
        "left_contact": "none",
        "right_contact": "none",
        "penalty_slippage_world": 0.0,
        "penalty_slippage_ground": 0.0,
        "penalty_slippage_board": 0.0,
        "penalty_feet_ori_world": 0.0,
        "penalty_feet_ori_surface": 0.0,
    }
    for side, values in candidates.items():
        if not values:
            continue
        selected = max(values, key=lambda item: item["force"])
        if selected["force"] <= 1.0:
            continue
        contact_type = selected["type"]
        metrics[f"{side}_contact"] = contact_type
        point = selected["point"]
        foot_velocity = body_velocity_at_point(
            model,
            data,
            selected["foot_body"],
            point,
        )
        metrics["penalty_slippage_world"] += float(np.linalg.norm(foot_velocity))
        surface_normal = selected["normal"]
        if contact_type == "ground":
            relative_velocity = foot_velocity
            surface_normal = np.asarray((0.0, 0.0, 1.0))
            metrics["penalty_slippage_ground"] += float(
                np.linalg.norm(
                    relative_velocity
                    - surface_normal * np.dot(relative_velocity, surface_normal)
                )
            )
        elif contact_type == "skateboard":
            board_velocity = body_velocity_at_point(model, data, board_body, point)
            relative_velocity = foot_velocity - board_velocity
            surface_normal = body_normal(data, board_body)
            metrics["penalty_slippage_board"] += float(
                np.linalg.norm(
                    relative_velocity
                    - surface_normal * np.dot(relative_velocity, surface_normal)
                )
            )
        else:
            continue
        foot_normal = body_normal(data, selected["foot_body"])
        metrics["penalty_feet_ori_world"] += float(
            math.sqrt(max(0.0, 1.0 - float(np.dot(foot_normal, (0.0, 0.0, 1.0))) ** 2))
        )
        metrics["penalty_feet_ori_surface"] += float(
            math.sqrt(
                max(
                    0.0,
                    1.0 - float(np.dot(foot_normal, surface_normal)) ** 2,
                )
            )
        )
    return metrics, sorted(set(pairs))


def board_roll(data: mujoco.MjData, body_id: int) -> float:
    rotation = data.xmat[body_id].reshape(3, 3)
    return float(math.atan2(rotation[2, 1], rotation[2, 2]))


def evaluate_rollout(
    rollout: RawRollout,
    fidelity_passed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = configure_model(rollout)
    data = mujoco.MjData(model)
    position_limits, effort_limits = bfm_limits()
    joint_names = [name.split("/")[-1] for name in rollout.metadata["joint_order"]]
    joint_index = {name: index for index, name in enumerate(joint_names)}
    fixed_wrist = set(rollout.metadata["fixed_bfm_joints"])
    if set(BFM0_JOINTS) - set(joint_names) != fixed_wrist:
        raise RuntimeError("Raw HUSKY joint contract does not match BFM fixed wrists.")
    board_body = model.body("skateboard/board_tilt_body").id
    trace = []
    phase_pairs: dict[str, Counter[str]] = defaultdict(Counter)
    previous_action = np.zeros(23, dtype=np.float64)
    for frame in range(len(rollout.arrays["phase_id"])):
        phase = rollout.phase_mapping[int(rollout.arrays["phase_id"][frame])]
        action = rollout.arrays["action"][frame].astype(np.float64)
        action_delta = action - previous_action
        previous_action = action
        action_rate_29d = float(np.sum(np.square(5.0 * action_delta)))
        _, action_scales, _ = control_spec()
        action_rate_23d = float(np.sum(np.square(action_delta * action_scales)))
        dof = {
            name: float(rollout.arrays["dof_pos"][frame, index])
            for name, index in joint_index.items()
        }
        limit_penalty = 0.0
        for name in BFM0_JOINTS:
            position = dof.get(name, 0.0)
            lower, upper = position_limits[name]
            limit_penalty += max(lower - position, 0.0) + max(position - upper, 0.0)
        left_roll = float(dof["left_ankle_roll_joint"] ** 2)
        right_roll = float(dof["right_ankle_roll_joint"] ** 2)
        row: dict[str, Any] = {
            "rollout": rollout.name,
            "frame": int(frame),
            "time": float(rollout.arrays["sim_time"][frame]),
            "phase": phase,
            "phase_group": phase_group(phase),
            "penalty_action_rate_29d": action_rate_29d,
            "penalty_action_rate_23d": action_rate_23d,
            "limits_dof_pos": float(limit_penalty),
            "penalty_ankle_roll": left_roll + right_roll,
            "left_ankle_roll_sq": left_roll,
            "right_ankle_roll_sq": right_roll,
            "ignored_wrist_action_rate_contribution": 0.0,
            "board_roll_rad": math.nan,
            "left_contact": "unavailable",
            "right_contact": "unavailable",
        }
        if fidelity_passed:
            set_recorded_state(model, data, rollout, frame)
            contact, pairs = contact_diagnostics(model, data)
            phase_pairs[phase].update(pairs)
            torques = np.asarray(
                [
                    data.qfrc_actuator[model.joint(f"robot/{name}").dofadr[0]]
                    for name in joint_names
                ],
                dtype=np.float64,
            )
            torque_map = dict(zip(joint_names, torques, strict=True))
            torque_29d = np.asarray(
                [torque_map.get(name, 0.0) for name in BFM0_JOINTS],
                dtype=np.float64,
            )
            effort_29d = np.asarray(
                [effort_limits[name] * 0.95 for name in BFM0_JOINTS],
                dtype=np.float64,
            )
            row.update(contact)
            row["penalty_torques"] = float(np.sum(np.square(torque_29d)))
            row["limits_torque"] = float(
                np.sum(np.maximum(np.abs(torque_29d) - effort_29d, 0.0))
            )
            row["board_roll_rad"] = board_roll(data, board_body)
        else:
            for name in (
                "penalty_torques",
                "limits_torque",
                "penalty_undesired_contact",
                "penalty_feet_ori_world",
                "penalty_feet_ori_surface",
                "penalty_slippage_world",
                "penalty_slippage_board",
                "penalty_slippage_ground",
            ):
                row[name] = math.nan

        original = (
            -0.1 * row["penalty_action_rate_29d"]
            - 10.0 * row["limits_dof_pos"]
            - 1.0 * row["penalty_undesired_contact"]
            - 0.4 * row["penalty_feet_ori_world"]
            - 4.0 * row["penalty_ankle_roll"]
            - 2.0 * row["penalty_slippage_world"]
        )
        candidate_world = (
            -0.1 * row["penalty_action_rate_29d"]
            - 10.0 * row["limits_dof_pos"]
            - 1.0 * row["penalty_undesired_contact"]
            - 0.4 * row["penalty_feet_ori_world"]
            - 4.0 * row["penalty_ankle_roll"]
            - 2.0
            * (row["penalty_slippage_ground"] + row["penalty_slippage_board"])
        )
        candidate_surface = (
            -0.1 * row["penalty_action_rate_29d"]
            - 10.0 * row["limits_dof_pos"]
            - 1.0 * row["penalty_undesired_contact"]
            - 0.4 * row["penalty_feet_ori_surface"]
            - 4.0 * row["penalty_ankle_roll"]
            - 2.0
            * (row["penalty_slippage_ground"] + row["penalty_slippage_board"])
        )
        row["original_weighted_aux"] = original
        row["skate_candidate_aux"] = candidate_surface
        row["skate_candidate_aux_world_ori"] = candidate_world
        row["skate_candidate_aux_surface_ori"] = candidate_surface
        trace.append(row)
    return trace, {
        phase: dict(counter.most_common())
        for phase, counter in sorted(phase_pairs.items())
    }


def numeric_stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "max": None,
            "nonzero_fraction": None,
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
        "nonzero_fraction": float(np.mean(np.abs(finite) > 1e-8)),
    }


def phase_statistics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace:
        groups[row["phase"]].append(row)
        if row["phase_group"] != row["phase"]:
            groups[row["phase_group"]].append(row)
    return {
        phase: {
            reward: numeric_stats(
                np.asarray([float(row[reward]) for row in rows], dtype=np.float64)
            )
            for reward in REWARD_COLUMNS
        }
        for phase, rows in sorted(groups.items())
    }


def save_csv(path: Path, trace: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(trace)


def phase_spans(axis: plt.Axes, trace: list[dict[str, Any]]) -> None:
    start = 0
    while start < len(trace):
        rollout = trace[start]["rollout"]
        phase = trace[start]["phase"]
        end = start + 1
        while (
            end < len(trace)
            and trace[end]["rollout"] == rollout
            and trace[end]["phase"] == phase
        ):
            end += 1
        axis.axvspan(
            start,
            end - 1,
            color=PHASE_COLORS.get(phase, "#94a3b8"),
            alpha=0.10,
            linewidth=0,
        )
        start = end


def plot_traces(output_dir: Path, trace: list[dict[str, Any]]) -> None:
    x = np.arange(len(trace))

    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    for axis, reward in zip(axes.ravel(), REWARD_COLUMNS[:8], strict=True):
        axis.plot(x, [row[reward] for row in trace], linewidth=1.1)
        phase_spans(axis, trace)
        axis.set_ylabel(reward.replace("penalty_", ""))
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("concatenated recorded frame")
    axes[-1, 1].set_xlabel("concatenated recorded frame")
    fig.suptitle("M2.4b-1 auxiliary reward traces with recorded phase spans")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_traces.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 5))
    for name, color in (
        ("penalty_slippage_world", "#dc2626"),
        ("penalty_slippage_ground", "#15803d"),
        ("penalty_slippage_board", "#2563eb"),
    ):
        axis.plot(x, [row[name] for row in trace], label=name, color=color)
    phase_spans(axis, trace)
    axis.legend()
    axis.set_xlabel("concatenated recorded frame")
    axis.set_ylabel("contact-gated velocity")
    axis.set_title("World versus surface-relative slippage")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "slippage_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 5))
    for name, color in (
        ("penalty_feet_ori_world", "#dc2626"),
        ("penalty_feet_ori_surface", "#2563eb"),
    ):
        axis.plot(x, [row[name] for row in trace], label=name, color=color)
    phase_spans(axis, trace)
    axis.legend()
    axis.set_xlabel("concatenated recorded frame")
    axis.set_ylabel("contact-gated orientation penalty")
    axis.set_title("World-horizontal versus contact-surface foot orientation")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "feet_orientation_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(
        x,
        [row["penalty_ankle_roll"] for row in trace],
        label="ankle roll squared sum",
        color="#7c3aed",
    )
    twin = axis.twinx()
    twin.plot(
        x,
        [row["board_roll_rad"] for row in trace],
        label="board roll rad",
        color="#d97706",
    )
    phase_spans(axis, trace)
    axis.set_xlabel("concatenated recorded frame")
    axis.set_ylabel("ankle roll penalty")
    twin.set_ylabel("board roll (rad)")
    axis.set_title("Ankle roll penalty and board roll")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "ankle_roll_board_roll.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = [read_raw_rollout(path) for path in args.rollout]
    fidelities = {}
    trace = []
    contact_pairs = {}
    for rollout in rollouts:
        passed, report = replay_fidelity(rollout, args)
        fidelities[rollout.name] = report
        rollout_trace, pairs = evaluate_rollout(rollout, passed)
        trace.extend(rollout_trace)
        contact_pairs[rollout.name] = pairs

    save_csv(output_dir / "expert_reward_trace.csv", trace)
    plot_traces(output_dir, trace)
    summary = {
        "milestone": "M2.4b-1 Phase-wise Expert Reward Audit",
        "source_rollouts": [
            {
                "path": str(rollout.path),
                "metadata": str(rollout.path.with_suffix(".json")),
                "episode_id": rollout.name,
                "frame_count": int(len(rollout.arrays["frame_idx"])),
                "phase_mapping": {
                    str(key): value for key, value in rollout.phase_mapping.items()
                },
                "phase_runs": phase_runs(rollout),
            }
            for rollout in rollouts
        ],
        "replay_fidelity": fidelities,
        "phase_statistics": phase_statistics(trace),
        "contact_pairs": contact_pairs,
        "unavailable_reason": {
            "formal_expert_phase_coverage": (
                "The tracked MotionLib expert contains only one 50-frame push. "
                "This audit uses separate phase-rich raw HUSKY policy rollouts "
                "for diagnostics and does not add them to the formal expert set."
            ),
            "video": "No full raw-expert video was retained; none was generated.",
        },
        "formal_replay_modified": False,
        "aux_reward_training_semantics_changed": False,
        "training_performed": False,
        "next_milestone": "M2.4b-2 — Skate Aux Reward Contract",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    statuses = {report["status"] for report in fidelities.values()}
    print(f"Reward trace: {output_dir / 'expert_reward_trace.csv'}")
    print(f"Replay fidelity: {', '.join(sorted(statuses))}")
    return 0 if statuses == {"PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
