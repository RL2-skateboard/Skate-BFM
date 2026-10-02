from __future__ import annotations

from typing import Any
from queue import SimpleQueue
from pathlib import Path

import mujoco
import numpy as np
import mujoco.viewer

from .base import Backend
from ..config import SimConfig
from ..models.board import BoardStyle, board_styles, build_board_xml, board_hash
from ..robot.bfm import (BFMOnnxPolicy, DEPLOY_JOINTS, build_official_actor_input,
                         load_policy_context, deploy_joint_values)
from ..robot.contract import HistoryState
from ..physics.contact import contact_xml, contact_metrics


class MujocoBackend(Backend):
    name = "mujoco"
    KEY_SPACE, KEY_N, KEY_M, KEY_R = 32, 78, 77, 82
    KEY_F, KEY_E, KEY_ESC = 70, 69, 256

    def __init__(self, *, seed: int = 0, config: SimConfig | None = None,
                 scene: str = "bench", style: str = "standard", inspect: bool = False,
                 dynamics: dict | None = None, initial_roll: float = 0.0,
                 experiment: str | None = None, contact_model: str = "native",
                 controller: str = "none", checkpoint: str | None = None,
                 policy_task: str = "reward", latent_name: str | None = None,
                 preset: str = "separate"):
        self.config = config or SimConfig()
        self.scene = scene
        self.style = style
        self.inspect = inspect
        self.experiment = experiment
        self.contact_model = contact_model
        self.controller = controller
        self.policy_task = policy_task
        self.latent_name = latent_name
        self.policy = None
        self.policy_counter = 0
        self.policy_context_index = 0
        self.robot_joint_names = []
        self.robot_history = HistoryState()
        self.initial_roll = initial_roll
        self.dynamics = dynamics or {}
        self.events = SimpleQueue()
        self.contact_scene = scene == "contact"
        self.preset = preset
        self.robot_scene = scene in ("robot", "rider")
        self.rider_scene = scene == "rider"
        if self.rider_scene and preset not in ("separate", "push_start", "steer_start"):
            raise ValueError("unknown rider preset; choose separate, push_start, or steer_start")
        robot_path = None
        if self.robot_scene:
            from ..models.robot import official_robot_scene, rider_scene
            # Every robot controller is evaluated on the same official 29-DoF
            # free-base asset. The procedural model is only retained for the
            # old non-robot workbench and is never a policy/pose fallback.
            robot_path = rider_scene(style, preset) if self.rider_scene else official_robot_scene()
            xml = robot_path.read_text(encoding="utf-8")
            self.board_style = None
        elif self.contact_scene:
            xml = contact_xml(freeboard=experiment == "freeboard")
            self.board_style = None
        elif scene in ("board", "gallery"):
            self.board_style = board_styles(style)[style]
            xml = build_board_xml(self.board_style, gallery=scene == "gallery", dynamics=dynamics, initial_roll=initial_roll)
        else:
            self.board_style = None
            xml = self._xml()
        self.model = mujoco.MjModel.from_xml_path(str(robot_path)) if robot_path else mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = self.config.physics_dt
        self.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        if self.contact_scene:
            for gid in range(self.model.ngeom):
                geom_name = self.model.geom(gid).name or ""
                if geom_name == "deck_collision" or geom_name.startswith("deck_strip_"):
                    self.model.geom_friction[gid] = (.7, .01, .001)
            self.model.geom_friction[self.model.geom("foot_contact").id] = (.8, .02, .002)
        if experiment and self.board_style is not None:
            for prefix in ("front", "rear"):
                for side in ("left", "right"):
                    j = self.model.joint(f"{prefix}_{side}_spin").id
                    self.model.dof_damping[self.model.jnt_dofadr[j]] = self.dynamics["wheel_spin_damping"]
        self.data = mujoco.MjData(self.model)
        if self.robot_scene:
            self.robot_joint_names = list(DEPLOY_JOINTS)
            if len(self.robot_joint_names) != 29:
                raise ValueError(f"official G1 asset contract mismatch: robot_hinges={len(self.robot_joint_names)}")
            if any(self.model.joint(name).id < 0 for name in self.robot_joint_names):
                raise ValueError("official G1 asset is missing one or more deploy joints")
            self.robot_joint_ids = [self.model.joint(name).id for name in self.robot_joint_names]
            self.robot_qpos_adr = np.asarray([self.model.joint(name).qposadr[0] for name in self.robot_joint_names], dtype=int)
            self.robot_dof_adr = np.asarray([self.model.joint(name).dofadr[0] for name in self.robot_joint_names], dtype=int)
            if controller == "policy":
                if not checkpoint:
                    raise ValueError("policy controller requires --checkpoint")
                root = Path(checkpoint)
                onnx = root / "model" / "exported" / "FBcprAuxModel.onnx" if root.is_dir() else root
                self.policy = BFMOnnxPolicy(onnx)
                self.policy_z, self.policy_context_name = load_policy_context(root, policy_task, latent_name)
                self.policy_values = deploy_joint_values(Path(__file__).parents[3]/"configs"/"bfm"/"motivo_newG1.yaml")
            elif controller == "pose":
                self.policy_values = deploy_joint_values(Path(__file__).parents[3]/"configs"/"bfm"/"motivo_newG1.yaml")
                self.pose_target = self.policy_values["default"].copy()
        self.viewer = None
        self.paused = inspect
        self.closed = False
        self.force = np.zeros(3, dtype=np.float64)
        self.push_force = 20.
        self.push_duration = .2
        self.mouse_force = np.zeros(3, dtype=np.float64)
        body_name = "cube" if scene == "bench" else "foot" if self.contact_scene else "pelvis" if self.robot_scene else "deck"
        self._body_id = self.model.body(body_name).id
        self._deck_id = self.model.body("deck").id if self.contact_scene else None
        self.normal_load = 8.0 if self.contact_scene else 0.0
        self.pull_force = 0.0
        self.twist_torque = 0.0
        self.policy_values = getattr(self, "policy_values", None)
        self.q_target = np.zeros(29, dtype=np.float64)
        self.policy_action = np.zeros(29, dtype=np.float64)
        self.raw_action = np.zeros(29, dtype=np.float64)
        self.last_tau = np.zeros(29, dtype=np.float64)
        self.last_qfrc_applied = np.zeros(29, dtype=np.float64)
        self.last_q = np.zeros(29, dtype=np.float64)
        self.last_dq = np.zeros(29, dtype=np.float64)
        self.physics_step_counter = 0
        self.inference_time_ms = 0.0
        self._first_policy_debug = False
        self.last_obs = np.zeros(721, dtype=np.float32)
        self.reset(seed)

    @staticmethod
    def _xml() -> str:
        return '''<mujoco model="skate-sim-bench">
          <compiler angle="radian" coordinate="local"/>
          <option gravity="0 0 -9.81" integrator="implicitfast"/>
          <visual><headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/></visual>
          <worldbody>
            <light name="key" pos="0 -2 4" directional="true"/>
            <geom name="ground" type="plane" size="5 5 .1" rgba=".18 .20 .22 1"/>
            <geom name="axis_x" type="box" pos="1 0 .002" size="1 .008 .002" rgba=".9 .1 .1 1" contype="0" conaffinity="0"/>
            <geom name="axis_y" type="box" pos="0 1 .002" size=".008 1 .002" rgba=".1 .8 .1 1" contype="0" conaffinity="0"/>
            <geom name="axis_z" type="box" pos="0 0 .5" size=".008 .008 .5" rgba=".1 .3 1 1" contype="0" conaffinity="0"/>
            <geom name="grid_x" type="box" pos="0 0 .001" size="2.5 .002 .001" rgba=".45 .45 .45 1" contype="0" conaffinity="0"/>
            <geom name="grid_y" type="box" pos="0 0 .001" size=".002 2.5 .001" rgba=".45 .45 .45 1" contype="0" conaffinity="0"/>
            <body name="cube" pos="0 0 1">
              <freejoint name="cube_free"/>
              <geom name="cube_collision" type="box" size=".1 .1 .1" mass="1" rgba=".12 .48 .85 1"/>
              <site name="cube_center" size=".025" rgba="1 0.2 0.1 1"/>
            </body>
          </worldbody>
        </mujoco>'''

    def reset(self, seed: int = 0, env_ids: Any = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.force[:] = 0
        self.force_point = None
        self.mouse_selection = None
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.perturb.active = 0
                self.viewer.perturb.active2 = 0
        self.mouse_force[:] = 0
        self._pulse_steps = 0
        if self.robot_scene:
            # MuJoCo free joints use wxyz. The source asset has the official
            # pelvis height in qpos0; explicitly restore it on every reset.
            self.data.qpos[:7] = self.model.qpos0[:7]
            self.data.qpos[0:3] = (0.0, 0.0, float(self.model.qpos0[2]))
            self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            self.data.qpos[self.robot_qpos_adr] = self.policy_values["default"] if self.policy_values is not None else 0.0
            self.data.qvel[:] = 0.0
            if self.rider_scene:
                deck_q = self.model.joint("deck_free")
                deck_qa = int(deck_q.qposadr[0]); deck_da = int(deck_q.dofadr[0])
                # Use the procedural board's wheel-supported height.
                self.data.qpos[deck_qa:deck_qa + 7] = self.model.qpos0[deck_qa:deck_qa + 7]
                if self.preset == "separate":
                    self.data.qpos[0:3] = (0.0, -1.0, float(self.model.qpos0[2]))
                else:
                    from ..robot.rider_pose import initialize_rider_pose
                    initialize_rider_pose(self.model, self.data, self.preset)
                    if self.preset == "steer_start":
                        self.data.qvel[deck_da] = 1.0
                        self.data.qvel[0] = 1.0
                        style = board_styles(self.style)[self.style]
                        for prefix in ("front", "rear"):
                            for side in ("left", "right"):
                                self.data.joint(f"{prefix}_{side}_spin").qvel[:] = 1.0 / style.wheel_radius
                if self.controller == "pose":
                    self.pose_target = self.data.qpos[self.robot_qpos_adr].copy()
            self.robot_history.reset()
            self.policy_counter = 0
            self.policy_context_index = 0
            self.policy_action[:] = 0.0
            self.raw_action[:] = 0.0
            self.q_target[:] = self.policy_values["default"] if self.policy_values is not None else 0.0
            self.last_tau[:] = 0.0
            self.last_qfrc_applied[:] = 0.0
            self.last_q[:] = 0.0
            self.last_dq[:] = 0.0
            self.inference_time_ms = 0.0
            self.physics_step_counter = 0
            self._last_q_change = 0.0
            self.last_obs[:] = 0.0
        if self.board_style is not None:
            self.data.qpos[3:7] = (np.cos(self.initial_roll/2), np.sin(self.initial_roll/2), 0, 0)
        self.external_work = 0.0
        self.fixture = self.experiment in ('lean','turn')
        if self.rider_scene:
            self.fixture = False
        self.path = []
        if self.board_style is not None and self.experiment in ("coast", "turn"):
            root = self.model.joint("deck_free").dofadr[0]
            self.data.qvel[root] = self.dynamics.get('initial_speed',1.)
            for prefix in ("front", "rear"):
                for side in ("left", "right"):
                    self.data.joint(f"{prefix}_{side}_spin").qvel[:] = self.data.qvel[root] / self.board_style.wheel_radius
        if self.board_style is not None and self.experiment == "release":
            self.data.joint("front_truck_tilt").qpos[:] = .08
            self.data.joint("rear_truck_tilt").qpos[:] = .08
            if self.dynamics.get('model') == 'reduced':
                self.data.joint('deck_roll').qpos[:] = -.577*.08
            # Lift the whole assembly just enough to remove initial wheel penetration.
            mujoco.mj_forward(self.model, self.data)
            low = min(self.data.xpos[self.model.body(f"{p}_{s}").id, 2]
                      for p in ("front", "rear") for s in ("left", "right"))
            self.data.joint("deck_free").qpos[2] += max(0., self.board_style.wheel_radius-low)
        if self.experiment == 'drop':
            self.data.joint('deck_free').qpos[2] += .08
        mujoco.mj_forward(self.model, self.data)
        if self.rider_scene:
            self._print_rider_initial_state()

    def step(self, action: Any = None) -> None:
        del action
        q_before = self._robot_q() if self.robot_scene else None
        if self.robot_scene and self.controller == "policy" and self.physics_step_counter % self.config.substeps == 0:
            import time as _time
            base = self.model.body("pelvis").id
            q = self._robot_q().astype(np.float32)
            dq = self._robot_dq().astype(np.float32)
            context = self.policy_z[self.policy_context_index] if self.policy_task == "tracking" else self.policy_z
            t0 = _time.perf_counter()
            obs = build_official_actor_input(q, dq, self.data.xquat[base], self.data.cvel[base, :3],
                                             self.policy_action, self.robot_history, context)
            self.last_obs = obs.copy()
            raw = np.asarray(self.policy(obs), dtype=np.float64)
            self.inference_time_ms = (_time.perf_counter() - t0) * 1000.0
            if raw.shape != (29,) or not np.isfinite(raw).all():
                raise RuntimeError(f"BFM action contract failed: shape={raw.shape} finite={np.isfinite(raw).all()}")
            self.raw_action = raw.copy()
            self.policy_action = np.clip(raw, -1.0, 1.0)
            self.q_target = np.clip(
                self.policy_values["default"] + float(self.policy_values["action_rescale"]) *
                self.policy_action * self.policy_values["action_scale"],
                self.policy_values["joint_pos_lower_limit"], self.policy_values["joint_pos_upper_limit"])
            if self.policy_task == "tracking":
                self.policy_context_index = (self.policy_context_index + 1) % len(self.policy_z)
            self.policy_counter += 1
            self._save_policy_debug(obs)
        elif self.robot_scene and self.controller == "pose":
            self.q_target = np.clip(self.pose_target, self.policy_values["joint_pos_lower_limit"],
                                    self.policy_values["joint_pos_upper_limit"])
        robot_torques = []
        if self.robot_scene and self.controller in ("policy", "pose"):
            q = self._robot_q()
            dq = self._robot_dq()
            tau = self.policy_values["kp"] * (self.q_target - q) - self.policy_values["kd"] * dq
            tau = np.clip(tau, -self.policy_values["joint_effort_limit"], self.policy_values["joint_effort_limit"])
            self.last_q, self.last_dq, self.last_tau = q.copy(), dq.copy(), tau.copy()
            robot_torques = list(zip(self.robot_dof_adr, tau))
        self.data.xfrc_applied[:] = 0
        self.mouse_selection = None
        if self.viewer is not None:
            with self.viewer.lock():
                pert = self.viewer.perturb
                mujoco.mjv_applyPerturbForce(self.model, self.data, pert)
                if pert.select > 0:
                    bid = pert.select
                    position = self.data.xpos[bid] + self.data.xmat[bid].reshape(3,3) @ pert.localpos
                    self.mouse_selection = dict(body=self.model.body(bid).name,
                        local_point_m=pert.localpos.copy().tolist(), world_point_m=position.tolist(),
                        force_N=self.data.xfrc_applied[bid,:3].copy().tolist(),
                        torque_at_com_Nm=self.data.xfrc_applied[bid,3:].copy().tolist(),
                        active=bool(pert.active | pert.active2))
        self.mouse_force[:] = self.data.xfrc_applied[self._body_id, :3]
        self.data.xfrc_applied[self._body_id, :3] += self.force
        if self.contact_scene:
            # Positive load is a commanded downward clamp force in the UI.
            self.data.xfrc_applied[self._body_id, 2] -= self.normal_load
            self.data.xfrc_applied[self._body_id, 0] += self.pull_force
            self.data.xfrc_applied[self._body_id, 3] += self.twist_torque
        if self.force_point is not None:
            self.data.xfrc_applied[self._body_id, 3:] += np.cross(
                self.force_point-self.data.xipos[self._body_id], self.force)
        if self.fixture and not self.rider_scene:
            rotation = self.data.xmat[self._body_id].reshape(3,3)
            roll = np.arctan2(rotation[2,1],rotation[2,2])
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self._body_id,velocity,0)
            self.data.xfrc_applied[self._body_id,3] += self.dynamics['fixture_k']*(self.dynamics['lean_target']-roll)-self.dynamics['fixture_c']*velocity[0]
        self.data.qfrc_applied[:] = 0
        if robot_torques:
            # Use one indexed write so the MuJoCo ndarray is not repeatedly
            # traversed through scalar numpy views.
            self.data.qfrc_applied[self.robot_dof_adr] = self.last_tau
        if self.robot_scene and self.controller in ("policy", "pose"):
            self.last_qfrc_applied = self.data.qfrc_applied[self.robot_dof_adr].copy()
            self.torque_injection_error = float(np.max(np.abs(self.last_qfrc_applied - self.last_tau)))
            if self.controller == "policy" and np.linalg.norm(self.policy_action) > 1e-7 and np.linalg.norm(self.last_qfrc_applied) < 1e-9:
                raise RuntimeError("nonzero policy action produced zero qfrc_applied")
        if self.experiment and self.board_style is not None:
            for prefix in ("front", "rear"):
                joint = self.model.joint(prefix + "_truck_tilt")
                q = self.data.qpos[joint.qposadr[0]]
                self.data.qfrc_applied[joint.dofadr[0]] = -self.dynamics["truck_k3"] * q**3
                for side in ("left", "right"):
                    adr = self.model.joint(f"{prefix}_{side}_spin").dofadr[0]
                    self.data.qfrc_applied[adr] = -self.dynamics["bearing_torque"] * np.tanh(self.data.qvel[adr]/.1)
        power_before = self._external_power()
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.physics_step_counter += 1
        if self.robot_scene:
            q_after = self._robot_q()
            self._last_q_change = float(np.linalg.norm(q_after - q_before))
            if (self.controller == "policy" and np.linalg.norm(self.policy_action) > 1e-7
                    and np.linalg.norm(self.last_tau) > 1e-7 and self._last_q_change == 0.0):
                raise RuntimeError("robot q did not change after nonzero policy torque")
        self.external_work += .5*(power_before+self._external_power())*self.config.physics_dt
        if self.board_style and len(self.path) < 10000:
            if not self.path or np.linalg.norm(self.data.xpos[self._body_id]-self.path[-1]) > .01:
                self.path.append(self.data.xpos[self._body_id].copy())
        if self._pulse_steps > 0:
            self._pulse_steps -= 1
            if self._pulse_steps == 0:
                self.force[:] = 0
                self.data.xfrc_applied[:] = 0
                print("[bench] pulse finished; force=0 N", flush=True)

    def _robot_q(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.robot_qpos_adr], dtype=np.float64).copy()

    def _robot_dq(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self.robot_dof_adr], dtype=np.float64).copy()

    def _rider_contact_info(self) -> dict[str, Any]:
        """Inspect compiled contact pairs by actual G1/board entity names."""
        left_body = self.model.body("left_ankle_roll_link").id
        right_body = self.model.body("right_ankle_roll_link").id
        def is_descendant(body_id, ancestor_id):
            while body_id:
                if body_id == ancestor_id: return True
                body_id = int(self.model.body_parentid[body_id])
            return False
        deck_body = self.model.body("deck").id
        ground_geom = self.model.geom("ground").id
        foot_deck, foot_ground, pairs = [], [], []
        penetration = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1]); b2 = int(self.model.geom_bodyid[c.geom2])
            pair = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
            pairs.append(pair)
            left_hit = is_descendant(b1, left_body) or is_descendant(b2, left_body)
            right_hit = is_descendant(b1, right_body) or is_descendant(b2, right_body)
            if (left_hit or right_hit) and deck_body in (b1, b2):
                foot_deck.append(pair)
            if (left_hit or right_hit) and ground_geom in (c.geom1, c.geom2):
                foot_ground.append(pair)
            penetration = min(penetration, float(c.dist))
        def body_contacts(body_id, contacts):
            return [p for p in contacts if p and (self.model.geom_bodyid[self.model.geom(p[0]).id] == body_id if p[0] else False)]
        # Categorize by body IDs rather than nullable generated geom names.
        left_deck = right_deck = left_ground = right_ground = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]; b1 = int(self.model.geom_bodyid[c.geom1]); b2 = int(self.model.geom_bodyid[c.geom2])
            foot = left_body if is_descendant(b1, left_body) or is_descendant(b2, left_body) else right_body if is_descendant(b1, right_body) or is_descendant(b2, right_body) else None
            if foot is None: continue
            if deck_body in (b1, b2):
                if foot == left_body: left_deck += 1
                else: right_deck += 1
            if ground_geom in (c.geom1, c.geom2):
                if foot == left_body: left_ground += 1
                else: right_ground += 1
        deck_q = self.model.joint("deck_free")
        return {"left_foot_body": "left_ankle_roll_link", "right_foot_body": "right_ankle_roll_link",
                "left_foot_position": self.data.xpos[left_body].tolist(),
                "right_foot_position": self.data.xpos[right_body].tolist(),
                "deck_position": self.data.xpos[deck_body].tolist(),
                "deck_top_height": float(self.data.site("deck_top").xpos[2]),
                "pelvis_position": self.data.xpos[self.model.body("pelvis").id].tolist(),
                "pelvis_quaternion_wxyz": self.data.xquat[self.model.body("pelvis").id].tolist(),
                "left_foot_deck_contact_count": left_deck, "right_foot_deck_contact_count": right_deck,
                "left_foot_ground_contact_count": left_ground, "right_foot_ground_contact_count": right_ground,
                "foot_deck_contacts": foot_deck, "foot_ground_contacts": foot_ground,
                "all_contact_pairs": pairs, "penetration_min_m": penetration,
                "board_linear_velocity": self.data.qvel[int(deck_q.dofadr[0]):int(deck_q.dofadr[0]) + 3].tolist(),
                "board_angular_velocity": self.data.qvel[int(deck_q.dofadr[0]) + 3:int(deck_q.dofadr[0]) + 6].tolist(),
                "pelvis_deck_horizontal_distance": float(np.linalg.norm(self.data.xpos[self.model.body("pelvis").id][:2] - self.data.xpos[deck_body][:2])),
                "support_force_N": 0.0, "has_support_fixture": False}

    def rider_initial_state(self) -> dict[str, Any]:
        if not self.rider_scene:
            raise ValueError("rider_initial_state requires scene=rider")
        info = self._rider_contact_info()
        info.update({"preset": self.preset,
                     "robot_root_position": self.data.qpos[:3].tolist(),
                     "robot_root_quaternion_wxyz": self.data.qpos[3:7].tolist(),
                     "board_root_position": self.data.qpos[int(self.model.joint("deck_free").qposadr[0]):int(self.model.joint("deck_free").qposadr[0]) + 3].tolist(),
                     "board_root_quaternion_wxyz": self.data.qpos[int(self.model.joint("deck_free").qposadr[0]) + 3:int(self.model.joint("deck_free").qposadr[0]) + 7].tolist(),
                     "ground_geom": "ground", "deck_body": "deck",
                     "deck_contact_geoms": [self.model.geom(i).name for i in range(self.model.ngeom)
                                             if self.model.geom_bodyid[i] == self.model.body("deck").id and self.model.geom_contype[i]],
                     "termination": self._rider_termination(info)})
        return info

    def _print_rider_initial_state(self) -> None:
        info = self.rider_initial_state()
        print("RIDER_INIT " + str({k: info[k] for k in (
            "preset", "pelvis_position", "pelvis_quaternion_wxyz", "left_foot_position",
            "right_foot_position", "deck_position", "deck_top_height",
            "pelvis_deck_horizontal_distance", "board_linear_velocity",
            "board_angular_velocity", "left_foot_deck_contact_count",
            "right_foot_deck_contact_count", "left_foot_ground_contact_count",
            "right_foot_ground_contact_count", "penetration_min_m", "all_contact_pairs")}), flush=True)

    def _rider_termination(self, info: dict[str, Any]) -> dict[str, Any]:
        pelvis = np.asarray(info["pelvis_position"])
        q = self._robot_q(); finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        reason = None
        if not finite: reason = "nonfinite"
        elif pelvis[2] < 0.35: reason = "fall"
        elif info["penetration_min_m"] < -0.02: reason = "penetration"
        else:
            lower = self.policy_values["joint_pos_lower_limit"] if self.policy_values is not None else np.asarray([self.model.jnt_range[j, 0] for j in self.robot_joint_ids])
            upper = self.policy_values["joint_pos_upper_limit"] if self.policy_values is not None else np.asarray([self.model.jnt_range[j, 1] for j in self.robot_joint_ids])
            if np.any(q < lower) or np.any(q > upper): reason = "joint_limit"
        return {"terminated": reason is not None, "truncated": False, "termination_reason": reason}

    def _save_policy_debug(self, obs: np.ndarray) -> None:
        if self._first_policy_debug:
            return
        out = Path("runs/05_debug")
        out.mkdir(parents=True, exist_ok=True)
        import json
        (out / "obs.json").write_text(json.dumps({
            "obs_shape": list(obs.shape), "obs_dtype": str(obs.dtype),
            "obs_finite": bool(np.isfinite(obs).all()), "obs_min": float(obs.min()),
            "obs_max": float(obs.max()), "current_state_shape": [93],
            "history_field_shapes": {"prev_actions": [116], "base_ang_vel": [12],
                                     "dof_pos_minus_default": [116], "dof_vel": [116],
                                     "projected_gravity": [12]}, "latent_shape": [256]}, indent=2))
        (out / "action.json").write_text(json.dumps({"action_shape": [29],
            "raw_action_min": float(self.raw_action.min()), "raw_action_max": float(self.raw_action.max()),
            "clipped_action_min": float(self.policy_action.min()), "clipped_action_max": float(self.policy_action.max())}, indent=2))
        mapping = []
        for i, name in enumerate(self.robot_joint_names):
            mapping.append({"policy_index": i, "official_joint_name": name,
                            "mujoco_joint_id": int(self.robot_joint_ids[i]),
                            "qpos_address": int(self.robot_qpos_adr[i]), "dof_address": int(self.robot_dof_adr[i]),
                            "default_position": float(self.policy_values["default"][i]),
                            "action_scale": float(self.policy_values["action_scale"][i]),
                            "kp": float(self.policy_values["kp"][i]), "kd": float(self.policy_values["kd"][i]),
                            "position_lower": float(self.policy_values["joint_pos_lower_limit"][i]),
                            "position_upper": float(self.policy_values["joint_pos_upper_limit"][i]),
                            "velocity_limit": float(self.policy_values["joint_velocity_limit"][i]),
                            "effort_limit": float(self.policy_values["joint_effort_limit"][i])})
        (out / "mapping.json").write_text(json.dumps(mapping, indent=2))
        self._first_policy_debug = True

    def get_state(self) -> dict[str, Any]:
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "time": float(self.data.time)}

    def _external_power(self):
        power = 0.
        for bid in range(1,self.model.nbody):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,bid,velocity,0)
            power += self.data.xfrc_applied[bid,:3] @ velocity[3:] + self.data.xfrc_applied[bid,3:] @ velocity[:3]
        return float(power)

    def set_state(self, state: dict[str, Any]) -> None:
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.time = state["time"]
        mujoco.mj_forward(self.model, self.data)

    def apply_wrench(self, body: str, force: tuple[float, float, float], point: tuple[float, float, float] | None = None) -> None:
        """World force at a world point; None means body COM. Persists until release."""
        expected = {"cube"} if self.scene == "bench" else {"deck"}
        if self.rider_scene:
            expected = {"deck", "pelvis"}
        if body not in expected:
            raise ValueError(f"unknown body {body!r}; available: {', '.join(sorted(expected))}")
        self.force[:] = force
        self.force_point = None if point is None else np.asarray(point, dtype=float).copy()
        self._pulse_steps = 0

    def diagnostics(self) -> dict[str, Any]:
        body_name = "cube" if self.scene == "bench" else ("foot" if self.contact_scene else "pelvis" if self.robot_scene else "deck")
        body_id = self.model.body(body_name).id
        result = {"backend": self.name, "version": mujoco.__version__, "sim_time": float(self.data.time),
                "object": body_name, "object_pos": self.data.xpos[body_id].tolist(),
                "object_vel": self.data.cvel[body_id, 3:].tolist(),
                "force_world_N": (self.mouse_force + self.force).tolist(),
                "keyboard_force_N": self.force.tolist(), "mouse_force_N": self.mouse_force.tolist(),
                "mouse_body": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self._body_id),
                "contacts": int(self.data.ncon)}
        result['mouse_selection'] = self.mouse_selection
        result['force_point_world_m'] = None if self.force_point is None else self.force_point.tolist()
        if self.board_style is not None:
            rotation = self.data.xmat[body_id].reshape(3,3)
            elastic_extra = sum(.25*self.dynamics.get("truck_k3",0)*float(self.data.joint(p+"_truck_tilt").qpos[0])**4 for p in ("front", "rear"))
            steer = {}
            for p in ('front','rear'):
                values=[]
                for s in ('left','right'):
                    wheel=self.model.body(f'{p}_{s}').id
                    axle=self.data.xmat[wheel].reshape(3,3)[:,1]
                    yaw=np.arctan2(rotation[1,0],rotation[0,0])
                    angle=np.arctan2(-axle[0],axle[1])-yaw
                    values.append(float(np.arctan2(np.sin(angle),np.cos(angle))))
                steer[p] = float(np.mean(values))
            result.update({"experiment": self.experiment, "model": self.dynamics.get('model','truck'),
                           "fixture": self.fixture, "external_work_J": self.external_work,
                           "steer_rad": steer,
                           "deck_roll_rad": float(np.arctan2(rotation[2,1], rotation[2,2])),
                           "mechanical_energy_J": float(self.data.energy.sum()+elastic_extra),
                           "wheel_speed_rad_s": {f"{p}_{s}": float(self.data.joint(f"{p}_{s}_spin").qvel[0]) for p in ("front", "rear") for s in ("left", "right")}})
        if self.contact_scene:
            result.update({"experiment": self.experiment, "contact_model": self.contact_model,
                           "normal_load_N": self.normal_load, "pull_force_N": self.pull_force,
                           "twist_torque_Nm": self.twist_torque,
                           **contact_metrics(self.model, self.data, self._body_id, self._deck_id)})
        if self.robot_scene:
            q = self._robot_q()
            dq = self._robot_dq()
            tau = self.last_tau
            action = self.policy_action
            result.update({"controller": self.controller, "policy_task": self.policy_task,
                           "latent_name": getattr(self, "policy_context_name", None),
                           "obs_shape": [721] if self.controller == "policy" else None,
                           "obs_norm": float(getattr(self, "last_obs", np.zeros(1)).dot(getattr(self, "last_obs", np.zeros(1)))) ** .5 if hasattr(self, "last_obs") else 0.0,
                           "raw_action_norm": float(np.linalg.norm(self.raw_action)),
                           "raw_action_min": float(self.raw_action.min()), "raw_action_max": float(self.raw_action.max()),
                           "q_target_norm": float(np.linalg.norm(self.q_target)),
                           "q_target_min": float(self.q_target.min()), "q_target_max": float(self.q_target.max()),
                           "q_norm": float(np.linalg.norm(q)), "dq_norm": float(np.linalg.norm(dq)),
                           "tau_norm": float(np.linalg.norm(tau)), "tau_min": float(tau.min()),
                           "tau_max": float(tau.max()),
                           "number_of_saturated_torques": int(np.count_nonzero(np.isclose(np.abs(tau), self.policy_values["joint_effort_limit"], rtol=0, atol=1e-6))) if self.policy_values is not None else 0,
                           "mean_abs_tau": float(np.mean(np.abs(tau))),
                           "mean_abs_q_error": float(np.mean(np.abs(self.q_target - q))),
                           "inference_time_ms": self.inference_time_ms,
                           "policy_counter": self.policy_counter,
                           "physics_step_counter": self.physics_step_counter,
                           "qfrc_applied_norm": float(np.linalg.norm(self.last_qfrc_applied)),
                           "torque_injection_error": float(getattr(self, "torque_injection_error", 0.0)),
                           "q_change_after_mj_step": float(getattr(self, "_last_q_change", 0.0)),
                           "base_position": self.data.xpos[self.model.body("pelvis").id].tolist(),
                           "base_orientation_wxyz": self.data.xquat[self.model.body("pelvis").id].tolist(),
                           "velocity_saturated": int(np.count_nonzero(np.abs(dq) > self.policy_values["joint_velocity_limit"])) if self.policy_values is not None else 0})
        if self.rider_scene:
            result["rider_init"] = self.rider_initial_state()
            result["termination"] = result["rider_init"]["termination"]
            result["board_velocity"] = result["rider_init"]["board_linear_velocity"]
            result["left_foot_contact"] = result["rider_init"]["left_foot_deck_contact_count"] + result["rider_init"]["left_foot_ground_contact_count"]
            result["right_foot_contact"] = result["rider_init"]["right_foot_deck_contact_count"] + result["rider_init"]["right_foot_ground_contact_count"]
            result["support_force_N"] = 0.0
            result["has_support_fixture"] = False
        return result

    def render(self) -> bool:
        while not self.events.empty():
            self._handle_key(self.events.get())
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._key_callback)
            self.viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
            self.viewer.cam.distance = 3.2
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25
            if self.board_style:
                self.viewer.cam.distance = 1.8 if self.scene == "board" else 4.5
                self.viewer.cam.lookat[:] = (0, 0, .1)
        if self.experiment:
            with self.viewer.lock():
                scn = self.viewer.user_scn
                scn.ngeom = 0
                points = self.path[-200:]
                for start,end in zip(points, points[1:]):
                    if np.linalg.norm(end-start)<1e-8: continue
                    g=scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array([1.,.6,0.,1.]))
                    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.002,start,end)
                    scn.ngeom+=1
                if self.mouse_selection:
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([.008,.008,.008]),np.array(self.mouse_selection['world_point_m']),
                        np.eye(3).flatten(),np.array([1.,.1,.8,1.]))
                    scn.ngeom += 1
            d=self.diagnostics()
            if self.contact_scene:
                self.viewer.sync()
                return not self.viewer.is_running()
            self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'skate-sim | '+d['model']+' | '+str(self.experiment)+'\n'+('EXTERNAL ROLL FIXTURE' if self.fixture else 'FREE')+'\nTime / roll / steer F,R\nEnergy / external work\nF push; +/- amplitude; [ ] duration\nA/D lean; E release; R reset',
                f"\n{'PAUSED' if self.paused else 'RUNNING'}\n{self.data.time:.3f}s / {d['deck_roll_rad']:.3f} / {d['steer_rad']['front']:.3f},{d['steer_rad']['rear']:.3f}\n{d['mechanical_energy_J']:.4f}J / {self.external_work:.4f}J\n{self.push_force:.1f}N / {self.push_duration:.2f}s\nSelected: {self.mouse_selection['body'] + ' ' + str(np.round(self.mouse_selection['local_point_m'],3)) if self.mouse_selection else 'double-click a surface point'}"))
        self.viewer.sync()
        return not self.viewer.is_running()

    def _key_callback(self, key: int) -> None:
        self.events.put(key)

    def _handle_key(self, key: int) -> None:
        if key == self.KEY_SPACE:
            self.paused = not self.paused
        elif key == self.KEY_N:
            if self.paused:
                self.step()
        elif key == self.KEY_M:
            if self.paused:
                for _ in range(self.config.substeps):
                    self.step()
        elif key == self.KEY_R:
            self.reset()
        elif self.contact_scene and key == 65:
            self.pull_force += 1.0
        elif self.contact_scene and key == 68:
            self.pull_force -= 1.0
        elif self.contact_scene and key == 84:
            self.twist_torque += .1
        elif self.contact_scene and key == 89:
            self.twist_torque -= .1
        elif key == self.KEY_F:
            body = "cube" if self.scene == "bench" else "deck"
            self.apply_wrench(body, (self.push_force, 0.0, 0.0))
            self._pulse_steps = round(self.push_duration / self.config.physics_dt)
            print(f"[bench] F received: {body} +X 20 N for 0.2 sim seconds; paused={self.paused}", flush=True)
        elif key == self.KEY_E:
            self.force_point = None
            self.fixture = False
            if self.viewer:
                with self.viewer.lock():
                    self.viewer.perturb.active=0
                    self.viewer.perturb.active2=0
            self.force[:] = 0.0
            self._pulse_steps = 0
            self.data.xfrc_applied[:] = 0
            print("[bench] E received: force released", flush=True)
        elif key in (65,68) and self.experiment:
            self.dynamics['lean_target'] = abs(self.dynamics['lean_target'])*(1 if key == 65 else -1)
            self.fixture = True
            print('EXTERNAL ROLL FIXTURE',self.dynamics['lean_target'],flush=True)
        elif key in (61,45):
            self.push_force = max(0., self.push_force+(5 if key == 61 else -5))
        elif key in (91,93):
            self.push_duration=max(.02,self.push_duration+(.02 if key==93 else -.02))
        elif key == 256:  # GLFW Escape
            self.closed = True
        elif key == 86 and self.viewer:
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] ^= 1
        elif key == 74 and self.viewer:
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] ^= 1
        elif self.board_style and self.paused and key in (49, 50, 51, 52):
            wheel = ("front_left", "front_right", "rear_left", "rear_right")[key-49]
            address = self.model.joint(wheel + "_spin").qposadr[0]
            self.data.qpos[address] += .1
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            print(f"INSPECTION POSITION EDIT: {wheel} +0.1 rad; R restores nominal", flush=True)

    def is_running(self) -> bool:
        return not self.closed and (self.viewer is None or self.viewer.is_running())

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
