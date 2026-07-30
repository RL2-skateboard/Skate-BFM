"""Headless MuJoCo runtime for Skate-BFM integration smoke tests."""

from __future__ import annotations

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
        self.decimation = max(1, round(control_dt / self.model.opt.timestep))
        self.action_scale = float(action_scale)
        self._neutral_control = self.model.key_ctrl[0, : self.robot_action_dim].copy()
        self._last_action = np.zeros(self.robot_action_dim, dtype=np.float32)
        self._robot_joints = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.robot_action_dim)
        ]
        self._board_dof = self.model.joint("skateboard/floating_base_joint_skateboard").dofadr[0]

    def reset(self) -> dict[str, np.ndarray | float]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._last_action.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def step(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.robot_action_dim,):
            raise ValueError(f"Expected HUSKY action shape (23,), got {action.shape}")
        self._last_action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[: self.robot_action_dim] = (
            self._neutral_control + self.action_scale * self._last_action
        )
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        return self._observation()

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
        return {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "last_action": self._last_action.copy(),
            "projected_gravity": gravity,
            "angular_velocity": self.data.qvel[3:6].astype(np.float32).copy(),
            "root_height": float(self.data.qpos[2]),
            "board_speed": float(self.data.qvel[self._board_dof]),
        }

