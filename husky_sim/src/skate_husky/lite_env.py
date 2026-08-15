"""Lightweight MuJoCo runtime for Skate-BFM integration tests."""

from __future__ import annotations

import hashlib
import math
import re
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
DEFAULT_FALL_ORIENTATION_LIMIT_DEG = 70.0
DEFAULT_FALL_ROOT_HEIGHT_MIN = 0.45
DEFAULT_FALL_CONFIRM_TIME = 0.2
HUSKY_ROBOT_COM_RANGES = ((-0.025, 0.025), (-0.025, 0.025), (-0.03, 0.03))
HUSKY_SKATEBOARD_COM_RANGES = ((-0.02, 0.02), (-0.02, 0.02), (-0.01, 0.01))
HUSKY_ROBOT_FRICTION_SCALE_RANGE = (0.3, 1.6)
HUSKY_DECK_FRICTION_SCALE_RANGE = (0.8, 2.0)
HUSKY_FOOT_FRICTION_RANGE = (0.3, 1.8)
HUSKY_WHEEL_FRICTION_SCALE_RANGE = (0.8, 1.6)
HUSKY_JOINT_POSITION_OFFSET_RANGE = (-0.01, 0.01)


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


def fall_confirmation_steps(confirm_time: float, control_dt: float) -> int:
    if confirm_time <= 0.0 or control_dt <= 0.0:
        raise ValueError("Fall confirmation time and control_dt must be positive.")
    return max(1, round(confirm_time / control_dt))


def resolve_physics_seed(rollout_id: str, requested_seed: int | None) -> int:
    if requested_seed is not None:
        return requested_seed
    digest = hashlib.sha256(f"husky-play-dr-v1:{rollout_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def randomize_husky_play_physics(
    model: mujoco.MjModel,
    rollout_id: str,
    requested_seed: int | None,
) -> tuple[dict[str, object], dict[str, float]]:
    """Apply HUSKY's official play-time randomization once per rollout."""

    seed = resolve_physics_seed(rollout_id, requested_seed)
    rng = np.random.default_rng(seed)

    def uniform(bounds: tuple[float, float]) -> float:
        return float(rng.uniform(*bounds))

    robot_torso = model.body("robot/torso_link").id
    skateboard_deck = model.body("skateboard/skateboard_deck").id
    robot_com_offset = np.asarray(
        [uniform(bounds) for bounds in HUSKY_ROBOT_COM_RANGES], dtype=np.float64
    )
    skateboard_com_offset = np.asarray(
        [uniform(bounds) for bounds in HUSKY_SKATEBOARD_COM_RANGES],
        dtype=np.float64,
    )
    model.body_ipos[robot_torso] += robot_com_offset
    model.body_ipos[skateboard_deck] += skateboard_com_offset

    robot_friction_scales: dict[str, float] = {}
    deck_friction_scales: dict[str, float] = {}
    foot_friction: dict[str, float] = {}
    wheel_friction_scales: dict[str, float] = {}
    foot_pattern = re.compile(r"robot/(left|right)_foot[1-7]_collision$")
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if name.startswith("robot/"):
            scale = uniform(HUSKY_ROBOT_FRICTION_SCALE_RANGE)
            model.geom_friction[geom_id, 0] *= scale
            robot_friction_scales[name] = scale
        if name == "skateboard/skateboard_deck_collision":
            scale = uniform(HUSKY_DECK_FRICTION_SCALE_RANGE)
            model.geom_friction[geom_id, 0] *= scale
            deck_friction_scales[name] = scale
        if foot_pattern.fullmatch(name):
            value = uniform(HUSKY_FOOT_FRICTION_RANGE)
            model.geom_friction[geom_id, 0] = value
            foot_friction[name] = value
        if name.startswith("skateboard/") and name.endswith("_wheel_collision"):
            scale = uniform(HUSKY_WHEEL_FRICTION_SCALE_RANGE)
            model.geom_friction[geom_id, 2] *= scale
            wheel_friction_scales[name] = scale

    joint_offsets = {
        name: uniform(HUSKY_JOINT_POSITION_OFFSET_RANGE)
        for joint_id in range(model.njnt)
        if (name := model.joint(joint_id).name or "").startswith("robot/")
        and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    }
    return {
        "enabled": True,
        "mode": "official_husky_play_startup_and_reset",
        "lifecycle": "sampled_once_per_rollout",
        "seed": seed,
        "ranges": {
            "robot_torso_com_offset_m": HUSKY_ROBOT_COM_RANGES,
            "skateboard_com_offset_m": HUSKY_SKATEBOARD_COM_RANGES,
            "robot_sliding_friction_scale": HUSKY_ROBOT_FRICTION_SCALE_RANGE,
            "deck_sliding_friction_scale": HUSKY_DECK_FRICTION_SCALE_RANGE,
            "foot_sliding_friction": HUSKY_FOOT_FRICTION_RANGE,
            "wheel_rolling_friction_scale": HUSKY_WHEEL_FRICTION_SCALE_RANGE,
            "joint_position_offset_rad": HUSKY_JOINT_POSITION_OFFSET_RANGE,
        },
        "robot_torso_com_offset_m": robot_com_offset.tolist(),
        "skateboard_com_offset_m": skateboard_com_offset.tolist(),
        "robot_sliding_friction_scale": robot_friction_scales,
        "deck_sliding_friction_scale": deck_friction_scales,
        "foot_sliding_friction": foot_friction,
        "wheel_rolling_friction_scale": wheel_friction_scales,
        "joint_position_offset_rad": joint_offsets,
        "external_push": False,
        "observation_corruption": False,
    }, joint_offsets


class LiveFallDetector:
    """Detect a persistent fall without treating foot lift-off as a fall."""

    _illegal_geom = re.compile(
        r"(left|right)_(shin|linkage_brace|shoulder_yaw|elbow_yaw|wrist|hand)_collision"
        r"|robot/pelvis_collision$"
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        orientation_limit_deg: float,
        root_height_min: float,
        confirm_frames: int,
    ) -> None:
        self.model = model
        self.orientation_limit_deg = float(orientation_limit_deg)
        self.root_height_min = float(root_height_min)
        self.confirm_frames = max(1, int(confirm_frames))
        self.bad_frames = 0
        self.foot_geoms: set[int] = set()
        self.board_geoms: set[int] = set()
        self.illegal_geoms: set[int] = set()

        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ""
            if re.search(r"robot/(left|right)_foot[0-9]+_collision$", name):
                self.foot_geoms.add(geom_id)
            if name == "skateboard/skateboard_deck_collision":
                self.board_geoms.add(geom_id)
            if self._illegal_geom.search(name):
                self.illegal_geoms.add(geom_id)

    def reset(self) -> None:
        self.bad_frames = 0

    def check(
        self,
        data: mujoco.MjData,
    ) -> tuple[bool, list[str], dict[str, float | bool | int]]:
        quaternion = np.asarray(data.qpos[3:7], dtype=np.float64)
        norm = np.linalg.norm(quaternion)
        if norm <= 0.0:
            tilt_deg = 180.0
        else:
            _, qx, qy, qz = quaternion / norm
            gravity_z = 1.0 - 2.0 * (qx * qx + qy * qy)
            tilt_deg = math.degrees(math.acos(np.clip(gravity_z, -1.0, 1.0)))

        feet_on_board = False
        illegal_contact = False
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            first_geom, second_geom = int(contact.geom1), int(contact.geom2)
            if (
                first_geom in self.foot_geoms and second_geom in self.board_geoms
            ) or (
                second_geom in self.foot_geoms and first_geom in self.board_geoms
            ):
                feet_on_board = True
            if first_geom in self.illegal_geoms or second_geom in self.illegal_geoms:
                illegal_contact = True

        root_height = float(data.qpos[2])
        severe_tilt = tilt_deg > self.orientation_limit_deg
        low_contact_fall = root_height < self.root_height_min and illegal_contact
        candidate = severe_tilt or low_contact_fall
        self.bad_frames = self.bad_frames + 1 if candidate else 0
        reasons = []
        if severe_tilt:
            reasons.append(f"tilt>{self.orientation_limit_deg:.0f}deg")
        if low_contact_fall:
            reasons.append(f"height<{self.root_height_min:.2f}+illegal_contact")
        return self.bad_frames >= self.confirm_frames, reasons, {
            "tilt_deg": tilt_deg,
            "root_height": root_height,
            "feet_on_board": feet_on_board,
            "illegal_contact": illegal_contact,
            "fall_candidate": candidate,
            "confirm_frames": self.bad_frames,
        }


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
        fall_orientation_limit_deg: float = DEFAULT_FALL_ORIENTATION_LIMIT_DEG,
        fall_root_height_min: float = DEFAULT_FALL_ROOT_HEIGHT_MIN,
        fall_confirm_time: float = DEFAULT_FALL_CONFIRM_TIME,
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
        self._last_fall = False
        self._last_fall_reasons: list[str] = []
        self._last_fall_diagnostics: dict[str, float | bool | int] = {}
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
        self.fall_detector = LiveFallDetector(
            self.model,
            fall_orientation_limit_deg,
            fall_root_height_min,
            fall_confirmation_steps(fall_confirm_time, self.control_dt),
        )

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

    def reset(
        self,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float]:
        if (qpos is None) != (qvel is None):
            raise ValueError("qpos and qvel must be provided together.")
        if qpos is None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            for joint_name, offset in self._reset_joint_offsets.items():
                joint = self.model.joint(joint_name)
                self.data.qpos[joint.qposadr[0]] += offset
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            qpos = np.asarray(qpos, dtype=np.float64)
            qvel = np.asarray(qvel, dtype=np.float64)
            if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,):
                raise ValueError(
                    f"Expected qpos [{self.model.nq}] and qvel [{self.model.nv}], "
                    f"got {qpos.shape} and {qvel.shape}."
                )
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError("Reset qpos and qvel must be finite.")
            for joint_id in range(self.model.njnt):
                if self.model.jnt_type[joint_id] not in (
                    mujoco.mjtJoint.mjJNT_FREE,
                    mujoco.mjtJoint.mjJNT_BALL,
                ):
                    continue
                width = 7 if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE else 4
                quaternion = qpos[
                    self.model.jnt_qposadr[joint_id] + width - 4 :
                    self.model.jnt_qposadr[joint_id] + width
                ]
                if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-4):
                    raise ValueError("Reset quaternion is not normalized.")
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            self.data.ctrl[: self.robot_action_dim] = self._neutral_control
        self._last_action.fill(0.0)
        self._previous_action.fill(0.0)
        self._last_aux_rewards = self._zero_aux_rewards()
        mujoco.mj_forward(self.model, self.data)
        self._require_valid_state()
        self.fall_detector.reset()
        self._last_fall = False
        self._last_fall_reasons = []
        self._last_fall_diagnostics = {}
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
        self._require_valid_state()
        self._last_aux_rewards = self._compute_aux_rewards()
        (
            self._last_fall,
            self._last_fall_reasons,
            self._last_fall_diagnostics,
        ) = self.fall_detector.check(self.data)
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
    def fallen(self) -> bool:
        return self._last_fall

    @property
    def last_fall_diagnostics(self) -> dict[str, float | bool | int | str]:
        diagnostics: dict[str, float | bool | int | str] = dict(
            self._last_fall_diagnostics
        )
        diagnostics["fall_reason"] = ",".join(self._last_fall_reasons)
        return diagnostics

    @property
    def physical_actuator_report(self) -> tuple[dict[str, object], ...]:
        """Return the validated HUSKY actuator-to-joint constraint mapping."""

        return tuple(dict(item) for item in self._physical_actuator_report)

    def _zero_aux_rewards(self) -> dict[str, float]:
        return {name: 0.0 for name in AUX_REWARD_KEYS}

    def _require_valid_state(self) -> None:
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(
            self.data.qvel
        ).all():
            raise RuntimeError("HUSKY MuJoCo state contains NaN or Inf.")

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
