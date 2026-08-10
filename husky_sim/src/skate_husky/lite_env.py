"""Lightweight MuJoCo runtime for Skate-BFM integration tests."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np


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
        self._robot_joints = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.robot_action_dim)
        ]
        self._board_dof = self.model.joint("skateboard/floating_base_joint_skateboard").dofadr[0]
        self._board_body = self.model.body("skateboard/skateboard_deck").id
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

    def reset(self) -> dict[str, np.ndarray | float]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._last_action.fill(0.0)
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
