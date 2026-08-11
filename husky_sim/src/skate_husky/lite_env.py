"""Lightweight MuJoCo runtime for Skate-BFM integration tests."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np

AUX_REWARD_KEYS = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
)
CONTACT_FORCE_THRESHOLD = 1.0
SOFT_LIMIT_RATIO = 0.95
WORLD_UP = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)


def contact_tangential_speed(
    relative_velocity: np.ndarray,
    surface_normal: np.ndarray,
) -> float:
    """Return contact-surface-relative tangential speed."""

    normal = np.asarray(surface_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    velocity = np.asarray(relative_velocity, dtype=np.float64)
    return float(np.linalg.norm(velocity - normal * np.dot(velocity, normal)))


def world_horizontal_orientation_penalty(foot_normal: np.ndarray) -> float:
    """Return the original BFM world-horizontal foot-orientation penalty."""

    normal = np.asarray(foot_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return float(np.sqrt(max(0.0, 1.0 - np.dot(normal, WORLD_UP) ** 2)))


class HuskyLiteEnv:
    """Small runtime around HUSKY's official generated MuJoCo scene."""

    robot_action_dim = 23

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        control_dt: float = 0.02,
        action_scale: float = 0.1,
        viewer: bool = False,
        realtime: bool = False,
    ) -> None:
        husky_root = Path(__file__).resolve().parents[2]
        default_xml = husky_root / "upstream/test_scene/mjlab_scene.xml"
        self.xml_path = Path(xml_path) if xml_path else default_xml
        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"HUSKY scene not found at {self.xml_path}. "
                "Run: git submodule update --init --depth 1"
            )
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.decimation = max(1, round(control_dt / self.model.opt.timestep))
        self.action_scale = float(action_scale)
        self.realtime = bool(realtime)
        self._viewer_requested = bool(viewer)
        self._viewer = None
        self._neutral_control = self.model.key_ctrl[0, : self.robot_action_dim].copy()
        self._last_action = np.zeros(self.robot_action_dim, dtype=np.float32)
        self._previous_action = np.zeros(self.robot_action_dim, dtype=np.float32)
        self._last_aux_rewards = self._zero_aux_rewards()
        self._reset_joint_offsets: dict[str, float] = {}
        self._robot_joints = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.robot_action_dim)
        ]
        self._board_dof = self.model.joint(
            "skateboard/floating_base_joint_skateboard"
        ).dofadr[0]
        self._board_body = self.model.body("skateboard/skateboard_deck").id
        self._board_surface_body = self.model.body(
            "skateboard/board_tilt_body"
        ).id
        self._pelvis_body = self.model.body("robot/pelvis").id
        self._robot_body_ids = np.asarray(
            [
                body_id
                for body_id in range(self.model.nbody)
                if (
                    (name := mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        body_id,
                    ))
                    and name.startswith("robot/")
                )
            ],
            dtype=np.int32,
        )
        self._configure_aux_reward_contract()

    def set_control_mapping(
        self,
        neutral_control: np.ndarray,
        action_scale: np.ndarray,
    ) -> None:
        neutral_control = np.asarray(neutral_control, dtype=np.float64)
        action_scale = np.asarray(action_scale, dtype=np.float64)
        expected = (self.robot_action_dim,)
        if neutral_control.shape != expected or action_scale.shape != expected:
            raise ValueError(
                "Control mapping must contain one value per HUSKY actuator, "
                f"got neutral={neutral_control.shape}, scale={action_scale.shape}."
            )
        self._neutral_control = neutral_control.copy()
        self.action_scale = action_scale.copy()

    def set_reset_joint_offsets(
        self,
        joint_offsets: Mapping[str, float] | None,
    ) -> None:
        offsets = dict(joint_offsets or {})
        for joint_name, offset in offsets.items():
            joint = self.model.joint(joint_name)
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Reset offset cannot target free joint {joint_name}.")
            if not np.isfinite(offset):
                raise ValueError(f"Reset offset for {joint_name} must be finite.")
        self._reset_joint_offsets = {
            name: float(offset)
            for name, offset in offsets.items()
        }

    def reset(self) -> dict[str, np.ndarray | float]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for joint_name, offset in self._reset_joint_offsets.items():
            joint = self.model.joint(joint_name)
            self.data.qpos[joint.qposadr[0]] += offset
        self._last_action.fill(0.0)
        self._previous_action.fill(0.0)
        self._last_aux_rewards = self._zero_aux_rewards()
        mujoco.mj_forward(self.model, self.data)
        if self._viewer_requested and self._viewer is None:
            self._launch_viewer()
        self._sync_viewer()
        return self._observation()

    def step(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        start_time = time.perf_counter()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.robot_action_dim,):
            raise ValueError(f"Expected HUSKY action shape (23,), got {action.shape}")
        self._last_action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[: self.robot_action_dim] = (
            self._neutral_control + self.action_scale * self._last_action
        )
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self._last_aux_rewards = self._compute_aux_rewards()
        self._previous_action = self._last_action.copy()
        self._sync_viewer()
        if self.realtime:
            remaining = self.control_dt - (time.perf_counter() - start_time)
            if remaining > 0.0:
                time.sleep(remaining)
        return self._observation()

    @property
    def is_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def close(self) -> None:
        viewer = self._viewer
        self._viewer = None
        if viewer is not None and viewer.is_running():
            viewer.close()
            time.sleep(0.1)

    @property
    def last_aux_rewards(self) -> dict[str, float]:
        """Return raw post-step auxiliary penalties for the last transition."""

        return dict(self._last_aux_rewards)

    @property
    def physical_actuator_report(self) -> tuple[dict[str, object], ...]:
        """Return the validated HUSKY actuator-to-joint constraint mapping."""

        return tuple(dict(item) for item in self._physical_actuator_report)

    def _zero_aux_rewards(self) -> dict[str, float]:
        return {name: 0.0 for name in AUX_REWARD_KEYS}

    def _configure_aux_reward_contract(self) -> None:
        robot_actuator_ids = tuple(
            actuator_id
            for actuator_id in range(self.model.nu)
            if (self.model.actuator(actuator_id).name or "").startswith("robot/")
        )
        if robot_actuator_ids != tuple(range(self.robot_action_dim)):
            raise RuntimeError(
                "HUSKY auxiliary rewards require exactly the 23 mapped "
                "robot actuators."
            )
        actuator_joint_ids = []
        report = []
        joint_qposadr = []
        joint_dofadr = []
        joint_lower_limits = []
        joint_upper_limits = []
        joint_torque_limits = []

        for actuator_id, actuator_name in enumerate(self._robot_joints):
            if actuator_name is None:
                raise RuntimeError(f"HUSKY actuator {actuator_id} has no name.")
            if int(self.model.actuator_trntype[actuator_id]) != int(
                mujoco.mjtTrn.mjTRN_JOINT
            ):
                raise RuntimeError(
                    f"{actuator_name} must use a joint transmission for "
                    "auxiliary rewards."
                )
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint = self.model.joint(joint_id)
            joint_name = joint.name
            if joint_name is None or actuator_name != joint_name:
                raise RuntimeError(
                    f"{actuator_name} does not map one-to-one to its joint "
                    f"({joint_name})."
                )
            if int(self.model.jnt_type[joint_id]) != int(
                mujoco.mjtJoint.mjJNT_HINGE
            ):
                raise RuntimeError(
                    f"{actuator_name} must control one hinge joint, got {joint_name}."
                )
            if not bool(self.model.jnt_limited[joint_id]):
                raise RuntimeError(f"{joint_name} has no physical position limit.")
            if not bool(self.model.actuator_forcelimited[actuator_id]):
                raise RuntimeError(f"{actuator_name} has no physical force limit.")

            gear = self.model.actuator_gear[actuator_id].astype(np.float64)
            if not np.isfinite(gear).all() or abs(gear[0]) <= 1e-12:
                raise RuntimeError(f"{actuator_name} has an invalid actuator gear.")
            if not np.allclose(gear[1:], 0.0, atol=1e-12):
                raise RuntimeError(
                    f"{actuator_name} has an ambiguous multi-axis gear: {gear}."
                )
            force_range = self.model.actuator_forcerange[actuator_id].astype(
                np.float64
            )
            if (
                not np.isfinite(force_range).all()
                or force_range[0] >= 0.0
                or force_range[1] <= 0.0
                or not np.isclose(abs(force_range[0]), abs(force_range[1]))
            ):
                raise RuntimeError(
                    f"{actuator_name} has an ambiguous force range: {force_range}."
                )
            torque_limit = abs(float(gear[0])) * float(force_range[1])
            joint_range = self.model.jnt_range[joint_id].astype(np.float64)
            if not np.isfinite(joint_range).all() or joint_range[0] >= joint_range[1]:
                raise RuntimeError(f"{joint_name} has an invalid position range.")

            actuator_joint_ids.append(joint_id)
            joint_qposadr.append(int(joint.qposadr[0]))
            joint_dofadr.append(int(joint.dofadr[0]))
            joint_lower_limits.append(float(joint_range[0]))
            joint_upper_limits.append(float(joint_range[1]))
            joint_torque_limits.append(torque_limit)
            report.append(
                {
                    "actuator_name": actuator_name,
                    "joint_name": joint_name,
                    "transmission_type": "mjTRN_JOINT",
                    "gear": gear.tolist(),
                    "force_limited": True,
                    "force_range": force_range.tolist(),
                    "derived_joint_torque_limit": torque_limit,
                }
            )

        if len(set(actuator_joint_ids)) != self.robot_action_dim:
            raise RuntimeError("HUSKY actuator-to-joint mapping is not one-to-one.")

        self._robot_joint_qposadr = np.asarray(joint_qposadr, dtype=np.int32)
        self._robot_joint_dofadr = np.asarray(joint_dofadr, dtype=np.int32)
        self._robot_joint_lower_limits = np.asarray(
            joint_lower_limits,
            dtype=np.float64,
        )
        self._robot_joint_upper_limits = np.asarray(
            joint_upper_limits,
            dtype=np.float64,
        )
        self._robot_joint_torque_limits = np.asarray(
            joint_torque_limits,
            dtype=np.float64,
        )
        self._physical_actuator_report = tuple(report)

        self._foot_geoms = {
            side: {
                geom_id
                for geom_id in range(self.model.ngeom)
                if (self.model.geom(geom_id).name or "").startswith(
                    f"robot/{side}_foot"
                )
            }
            for side in ("left", "right")
        }
        if not all(self._foot_geoms.values()):
            raise RuntimeError("HUSKY foot collision geoms are unavailable.")
        self._ground_geoms = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (self.model.geom(geom_id).name or "") == "terrain"
        }
        self._board_geoms = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (self.model.geom(geom_id).name or "").startswith("skateboard/")
        }
        if not self._ground_geoms or not self._board_geoms:
            raise RuntimeError("HUSKY ground or skateboard collision geoms are unavailable.")
        self._foot_body_ids = {
            "left": self.model.body("robot/left_ankle_roll_link").id,
            "right": self.model.body("robot/right_ankle_roll_link").id,
        }
        penalized_tokens = ("pelvis", "shoulder", "hip")
        self._penalized_body_ids = {
            body_id
            for body_id in range(self.model.nbody)
            if (
                (name := self.model.body(body_id).name)
                and name.startswith("robot/")
                and any(token in name for token in penalized_tokens)
            )
        }
        for token in penalized_tokens:
            if not any(
                token in (self.model.body(body_id).name or "")
                for body_id in self._penalized_body_ids
            ):
                raise RuntimeError(
                    f"HUSKY has no reliable penalized-body mapping for {token}."
                )

    def _body_velocity_at_point(self, body_id: int, point: np.ndarray) -> np.ndarray:
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        return velocity[3:] + np.cross(
            velocity[:3],
            np.asarray(point, dtype=np.float64) - self.data.xpos[body_id],
        )

    def _body_normal(self, body_id: int) -> np.ndarray:
        return self.data.xmat[body_id].reshape(3, 3)[:, 2]

    def _contact_penalties(self) -> dict[str, float]:
        candidates: dict[str, list[tuple[str, float, np.ndarray]]] = {
            "left": [],
            "right": [],
        }
        undesired_contact = False

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            first_geom, second_geom = int(contact.geom1), int(contact.geom2)
            contact_force = np.empty(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_id, contact_force)
            force_magnitude = float(np.linalg.norm(contact_force[:3]))
            first_body = int(self.model.geom_bodyid[first_geom])
            second_body = int(self.model.geom_bodyid[second_geom])
            touches_surface = (
                first_geom in self._ground_geoms
                or second_geom in self._ground_geoms
                or first_geom in self._board_geoms
                or second_geom in self._board_geoms
            )
            if touches_surface and (
                first_body in self._penalized_body_ids
                or second_body in self._penalized_body_ids
            ) and np.any(np.abs(contact_force[:3]) > CONTACT_FORCE_THRESHOLD):
                undesired_contact = True

            for side, foot_geoms in self._foot_geoms.items():
                if first_geom in foot_geoms:
                    other_geom = second_geom
                elif second_geom in foot_geoms:
                    other_geom = first_geom
                else:
                    continue
                if other_geom in self._ground_geoms:
                    contact_type = "ground"
                elif other_geom in self._board_geoms:
                    contact_type = "board"
                else:
                    continue
                candidates[side].append(
                    (
                        contact_type,
                        force_magnitude,
                        np.asarray(contact.pos, dtype=np.float64).copy(),
                    )
                )
                break

        slippage = 0.0
        feet_orientation = 0.0
        for side, contacts in candidates.items():
            if not contacts:
                continue
            contact_type, force_magnitude, point = max(
                contacts,
                key=lambda item: item[1],
            )
            if force_magnitude <= CONTACT_FORCE_THRESHOLD:
                continue
            foot_body = self._foot_body_ids[side]
            foot_velocity = self._body_velocity_at_point(foot_body, point)
            if contact_type == "ground":
                relative_velocity = foot_velocity
                surface_normal = WORLD_UP
            else:
                relative_velocity = foot_velocity - self._body_velocity_at_point(
                    self._board_surface_body,
                    point,
                )
                surface_normal = self._body_normal(self._board_surface_body)
            slippage += contact_tangential_speed(relative_velocity, surface_normal)
            feet_orientation += world_horizontal_orientation_penalty(
                self._body_normal(foot_body)
            )

        return {
            "penalty_undesired_contact": float(undesired_contact),
            "penalty_feet_ori": feet_orientation,
            "penalty_slippage": slippage,
        }

    def _compute_aux_rewards(self) -> dict[str, float]:
        joint_positions = self.data.qpos[self._robot_joint_qposadr]
        midpoint = (
            self._robot_joint_lower_limits + self._robot_joint_upper_limits
        ) / 2.0
        radius = self._robot_joint_upper_limits - self._robot_joint_lower_limits
        lower_soft_limit = midpoint - SOFT_LIMIT_RATIO * radius / 2.0
        upper_soft_limit = midpoint + SOFT_LIMIT_RATIO * radius / 2.0
        joint_torques = self.data.qfrc_actuator[self._robot_joint_dofadr]

        rewards = {
            "penalty_torques": float(np.sum(np.square(joint_torques))),
            "penalty_action_rate": float(
                np.sum(np.square(self._last_action - self._previous_action))
            ),
            "limits_dof_pos": float(
                np.sum(
                    np.maximum(lower_soft_limit - joint_positions, 0.0)
                    + np.maximum(joint_positions - upper_soft_limit, 0.0)
                )
            ),
            "limits_torque": float(
                np.sum(
                    np.maximum(
                        np.abs(joint_torques)
                        - SOFT_LIMIT_RATIO * self._robot_joint_torque_limits,
                        0.0,
                    )
                )
            ),
            "penalty_ankle_roll": float(
                sum(
                    self.data.qpos[self.model.joint(joint_name).qposadr[0]] ** 2
                    for joint_name in (
                        "robot/left_ankle_roll_joint",
                        "robot/right_ankle_roll_joint",
                    )
                )
            ),
        }
        rewards.update(self._contact_penalties())
        rewards = {name: rewards[name] for name in AUX_REWARD_KEYS}
        if tuple(rewards) != AUX_REWARD_KEYS:
            raise RuntimeError("HUSKY auxiliary reward contract keys changed.")
        if not all(np.isfinite(value) and value >= 0.0 for value in rewards.values()):
            raise RuntimeError("HUSKY auxiliary rewards must be finite raw penalties.")
        return rewards

    def _launch_viewer(self) -> None:
        import mujoco.viewer

        self._viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self._viewer.cam.distance = 4.0
        self._viewer.cam.azimuth = 210.0
        self._viewer.cam.elevation = -10.0

    def _sync_viewer(self) -> None:
        if self._viewer is None or not self._viewer.is_running():
            return
        self._viewer.cam.lookat[:] = self.data.xpos[self._pelvis_body]
        self._viewer.sync()

    def _observation(self) -> dict[str, np.ndarray | float]:
        joint_position = np.empty(self.robot_action_dim, dtype=np.float32)
        joint_velocity = np.empty(self.robot_action_dim, dtype=np.float32)
        for index, actuator_name in enumerate(self._robot_joints):
            joint = self.model.joint(actuator_name)
            joint_position[index] = self.data.qpos[joint.qposadr[0]]
            joint_velocity[index] = self.data.qvel[joint.dofadr[0]]

        qw, qx, qy, qz = self.data.qpos[3:7]
        gravity = np.array(
            (
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ),
            dtype=np.float32,
        )
        body_velocity = np.empty((len(self._robot_body_ids), 6), dtype=np.float64)
        for index, body_id in enumerate(self._robot_body_ids):
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                body_velocity[index],
                0,
            )
        board_velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._board_body,
            board_velocity,
            0,
        )
        return {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "last_action": self._last_action.copy(),
            "projected_gravity": gravity,
            "angular_velocity": self.data.qvel[3:6].astype(np.float32).copy(),
            "root_position": self.data.qpos[:3].astype(np.float32).copy(),
            "root_quaternion": self.data.qpos[3:7].astype(np.float32).copy(),
            "root_linear_velocity": self.data.qvel[:3].astype(np.float32).copy(),
            "root_angular_velocity": self.data.qvel[3:6].astype(np.float32).copy(),
            "root_height": float(self.data.qpos[2]),
            "board_speed": float(self.data.qvel[self._board_dof]),
            "body_position": self.data.xpos[self._robot_body_ids].astype(np.float32).copy(),
            "body_quaternion": self.data.xquat[self._robot_body_ids].astype(np.float32).copy(),
            "body_linear_velocity": body_velocity[:, 3:].astype(np.float32),
            "body_angular_velocity": body_velocity[:, :3].astype(np.float32),
            "board_position": self.data.xpos[self._board_body].astype(np.float32).copy(),
            "board_quaternion": self.data.xquat[self._board_body].astype(np.float32).copy(),
            "board_linear_velocity": board_velocity[3:].astype(np.float32),
            "board_angular_velocity": board_velocity[:3].astype(np.float32),
        }
