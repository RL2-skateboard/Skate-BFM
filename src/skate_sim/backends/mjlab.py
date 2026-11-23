from __future__ import annotations

from typing import Any
import time

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

from .base import Backend, validate_env_ids
from ..config import SimConfig
from ..models.board import board_styles, build_board_xml
from ..robot.bfm import (BFMOnnxPolicy, DEPLOY_JOINTS,
                         build_official_actor_input, deploy_joint_values,
                         load_policy_context, tracking_context, next_tracking_cursor)
from ..robot.contract import HistoryState
from ..physics.randomize import (effective_dynamics, nominal_parameters,
                                  parameter_metadata, parameter_summary,
                                  resolve_parameters, sample_parameter_batch)


class MjlabBackend(Backend):
    """A minimal mjlab Simulation wrapper for the stage-01 bench.

    The physics state is owned by mjlab's ``Simulation`` (MuJoCo Warp), while
    the native MuJoCo viewer is used only as a display of a synchronized copy.
    No MuJoCo ``mj_step`` is used by this backend.
    """

    name = "mjlab"
    KEY_SPACE, KEY_N, KEY_M, KEY_R = 32, 78, 77, 82
    KEY_F, KEY_E = 70, 69

    def __init__(self, *, seed: int = 0, config: SimConfig | None = None,
                 scene: str = "bench", style: str = "standard", inspect: bool = False,
                 num_envs: int = 1, view_env: int = 0, controller: str = "none",
                 checkpoint: str | None = None, policy_task: str = "reward",
                 preset: str = "separate", params: str | dict | None = None,
                 randomize: bool = False, dynamics: dict | None = None,
                 episode_time_limit: float | None = 30.0, out_of_bounds_distance: float | None = 5.0,
                 offboard_distance: float | None = 1.0, scene_config: dict | None = None,
                 _validate_gpu_gripper: bool = False, **_: Any):
        import torch
        from mjlab.sim import Simulation, SimulationCfg
        import warp as wp

        if scene_config is not None and (randomize or (params is not None and params != "nominal") or dynamics or _.get("experiment")):
            raise ValueError("composed scene_config does not support legacy dynamics parameters or randomization; use an explicit legacy scene")
        self.scene_config = None
        if scene_config is not None:
            from ..models.scene import normalize_scene_config, scene_kind
            self.scene_config = normalize_scene_config(scene_config)
            if self.scene_config.get("gripper", "none") != "none" and not _validate_gpu_gripper:
                raise ValueError("Robotiq gripper control is currently native MuJoCo only; use --backend mujoco")
            scene = scene_kind(self.scene_config)
            style, preset = self.scene_config["style"], self.scene_config["preset"]
        self.has_board = bool(self.scene_config["board"]) if self.scene_config is not None else scene in ("board", "gallery", "rider")
        self.config = config or SimConfig()
        for name,value in (("episode_time_limit",episode_time_limit),("out_of_bounds_distance",out_of_bounds_distance),("offboard_distance",offboard_distance)):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive or None")
            setattr(self,name,value)
        self._episode_origin = None
        self._board_approached = np.zeros(num_envs,dtype=bool)
        self.scene = scene
        self.style = style
        self.num_envs = int(num_envs)
        self.view_env = int(view_env)
        if self.num_envs < 1 or not 0 <= self.view_env < self.num_envs:
            raise ValueError("view_env must be within [0, num_envs)")
        if controller not in ("none", "pose", "policy"):
            raise ValueError("controller must be none, pose, or policy")
        if controller == "policy" and not checkpoint:
            raise ValueError("policy controller requires --checkpoint")
        if scene not in ("rider", "robot") and controller != "none":
            raise ValueError("pose/policy controllers require scene=rider or scene=robot")
        if scene == "rider" and preset not in ("separate", "push_start", "steer_start"):
            raise ValueError("unknown rider preset")
        self.controller, self.preset, self.policy_task = controller, preset, policy_task
        self.seed = int(seed)
        self.randomize = bool(randomize)
        self.parameter_spec = params
        self.parameter_source = ("nominal" if params is None else
                                 "mapping" if isinstance(params, dict) else str(params))
        self.base_dynamics = dict(dynamics or {})
        self.parameters = resolve_parameters(params, randomize=self.randomize, seed=self.seed)
        self.nominal_parameters = nominal_parameters()
        self.parameter_batch = sample_parameter_batch(
            self.num_envs, seed=self.seed, randomize=self.randomize, source=self.parameter_spec)
        self.dynamics = effective_dynamics(self.base_dynamics, self.parameters)
        self.compile_dynamics = effective_dynamics(self.base_dynamics, self.nominal_parameters)
        self.board_style = board_styles(style)[style] if self.scene_config is None and scene in ("board", "gallery") else None
        self._torch = torch
        self._wp = wp
        scene_path = None
        if self.scene_config is not None:
            from ..models.scene import build_scene_xml
            xml = build_scene_xml(self.scene_config)
        elif self.board_style:
            xml = build_board_xml(self.board_style, gallery=scene == "gallery", dynamics=self.compile_dynamics)
        elif scene == "contact":
            from ..models.contact_scene import contact_xml
            xml = contact_xml()
        elif scene in ("robot", "rider"):
            from ..models.robot import official_robot_scene, rider_scene
            scene_path = rider_scene(style, preset, dynamics=self.compile_dynamics) if scene == "rider" else official_robot_scene()
            xml = scene_path.read_text(encoding="utf-8")
        else:
            xml = self._xml()
        generated_scene = self.scene_config is None and scene == "rider"
        try:
            self.spec = mujoco.MjSpec.from_file(str(scene_path)) if scene_path else mujoco.MjSpec.from_string(xml)
        finally:
            if generated_scene and scene_path is not None:
                Path(scene_path).unlink(missing_ok=True)
        from .warp import configure_launch_order, install_contact_order
        configure_launch_order()
        sim_cfg = SimulationCfg()
        sim_cfg.mujoco.timestep = self.config.physics_dt
        if _validate_gpu_gripper:
            # mjlab defaults to 50, whereas the native MuJoCo asset uses 35.
            # Preserve native collision quality for the fidelity experiment.
            sim_cfg.mujoco.ccd_iterations = int(self.spec.compile().opt.ccd_iterations)
        sim_cfg.nconmax = 512
        sim_cfg.njmax = 1024
        self.sim = Simulation(num_envs=self.num_envs, cfg=sim_cfg, spec=self.spec, device="cuda:0")
        install_contact_order(self.sim)
        self._validate_gpu_gripper = bool(_validate_gpu_gripper)
        from ..models.gripper import sides, prefix
        self._gripper_ids = {side: self.sim.mj_model.actuator(prefix(side) + "fingers_actuator").id
                             for side in sides((self.scene_config or {}).get("gripper", "none"))}
        self._setup_robot_controller(checkpoint)
        self.viewer = None
        self.paused = inspect
        self._force = np.zeros(3, dtype=np.float64)
        self._mouse_force = np.zeros(3, dtype=np.float64)
        self.physics_step_counter = 0
        self.policy_counter = 0
        if self.scene not in ("robot", "rider"):
            self.last_tau = None
            self.last_qfrc_applied = None
        self._viewer_data = mujoco.MjData(self.sim.mj_model)
        body_name = "deck" if self.board_style else ("pelvis" if scene in ("robot", "rider") else "foot" if scene == "contact" else "cube")
        if self.scene_config is not None:
            body_name = "pelvis" if scene in ("robot", "rider") else "deck" if self.has_board else "world"
        self._body_id = self.sim.mj_model.body(body_name).id
        self._parameter_metadata = parameter_metadata(
            self.parameter_batch[self.view_env], source=self.parameter_source,
            seed=self.seed, randomize=self.randomize)
        self._board_wheel_dof = []
        self._board_truck_joints = []
        self._board_root_dof = None
        self._board_deck_body_id = None
        self._board_root_qpos = None
        if self.board_style:
            model = self.sim.mj_model
            self._board_wheel_dof = [int(model.joint(f"{prefix}_{side}_spin").dofadr[0])
                                     for prefix in ("front", "rear") for side in ("left", "right")]
            self._board_truck_joints = [
                (int(model.joint(f"{prefix}_truck_tilt").qposadr[0]),
                 int(model.joint(f"{prefix}_truck_tilt").dofadr[0]))
                for prefix in ("front", "rear")]
            self._board_root_dof = int(model.joint("deck_free").dofadr[0])
            self._board_root_qpos = int(model.joint("deck_free").qposadr[0])
            self._board_deck_body_id = int(model.body("deck").id)
        self._nominal_qpos = self._nominal_qpos.to(self.sim.data.qpos.device) if hasattr(self, "_nominal_qpos") else self.sim.data.qpos.clone()
        self._nominal_qvel = self._nominal_qvel.to(self.sim.data.qvel.device) if hasattr(self, "_nominal_qvel") else self.sim.data.qvel.clone()
        self._state_snapshot = None
        self.reset(seed)
        self._viser_server = None
        self._viser_viewer = None
        self._viser_thread = None
        self._requested_style = None
        self._style_select = None

    def validation_torque_step(self, body_torque, gripper_raw=None, *, board_force_hooks=False) -> None:
        """Private fidelity probe: native29 Nm + official raw tendon command.

        No PD/QP, viewer, observations, reward or reset is included in this
        physics-only path. Optional board hooks are evaluated every 2 ms;
        current native composed scenes do not enable these experimental hooks.
        """
        if not self._validate_gpu_gripper:
            raise RuntimeError("GPU torque probe requires explicit validation opt-in")
        t = self._torch
        qfrc = self.sim.data.qfrc_applied._tensor
        tau = t.as_tensor(body_torque, device=qfrc.device, dtype=qfrc.dtype)
        if tau.shape != (self.num_envs, 29) or not bool(t.isfinite(tau).all()):
            raise ValueError("body_torque requires a finite (num_envs,29) batch")
        if bool((tau.abs() > t.as_tensor(self.policy_values['joint_effort_limit'], device=tau.device)).any()):
            raise ValueError("body torque exceeds original G1 effort limits")
        raw = None
        if gripper_raw is not None:
            raw = t.as_tensor(gripper_raw, device=qfrc.device, dtype=qfrc.dtype)
            if raw.shape != (self.num_envs, len(self._gripper_ids)) or not bool(t.isfinite(raw).all()) or bool(((raw < 0) | (raw > 255)).any()):
                raise ValueError("gripper_raw requires finite (num_envs,num_grippers) values in [0,255]")
        self._validation_torque_tick(tau, raw, board_force_hooks=board_force_hooks)

    def _validation_torque_tick(self, tau, raw, *, board_force_hooks=False):
        """Already checked device tensors; used by the physics-only profiler."""
        t = self._torch
        qfrc = self.sim.data.qfrc_applied._tensor
        with self._wp.ScopedStream(self._wp.stream_from_torch(t.cuda.current_stream())):
            qfrc.zero_()
            qfrc[:, self.robot_dof_index] = tau
            self.sim.data.ctrl._tensor.zero_()
            if raw is not None:
                self.sim.data.ctrl._tensor[:, list(self._gripper_ids.values())] = raw
            if self.has_board and board_force_hooks:
                from ..physics.dynamics import DynamicsConfig
                cfg = DynamicsConfig()
                model = self.sim.mj_model
                q, dq = self.sim.data.qpos._tensor, self.sim.data.qvel._tensor
                for end in ('front', 'rear'):
                    joint = model.joint(end + '_truck_tilt')
                    qfrc[:, int(joint.dofadr[0])] += -cfg.truck_k3 * q[:, int(joint.qposadr[0])] ** 3
                    for side in ('left', 'right'):
                        adr = int(model.joint(f'{end}_{side}_spin').dofadr[0])
                        qfrc[:, adr] += -cfg.bearing_torque * t.tanh(dq[:, adr] / .1)
            self.sim.step()
        self.physics_step_counter += 1
        self._state_snapshot = None

    @classmethod
    def for_asset(cls, scene: str, style: str, *, seed: int = 0, config: SimConfig | None = None):
        return cls(scene=scene, style=style, seed=seed, config=config, inspect=True)

    @staticmethod
    def _xml() -> str:
        return '''<mujoco model="skate-sim-mjlab-bench">
          <compiler angle="radian"/>
          <option gravity="0 0 -9.81" integrator="implicitfast"/>
          <worldbody>
            <geom name="ground" type="plane" size="5 5 .1" rgba=".18 .20 .22 1"/>
            <geom name="axis_x" type="box" pos="1 0 .002" size="1 .008 .002" rgba=".9 .1 .1 1" contype="0" conaffinity="0"/>
            <geom name="axis_y" type="box" pos="0 1 .002" size=".008 1 .002" rgba=".1 .8 .1 1" contype="0" conaffinity="0"/>
            <geom name="axis_z" type="box" pos="0 0 .5" size=".008 .008 .5" rgba=".1 .3 1 1" contype="0" conaffinity="0"/>
            <geom name="grid_x" type="box" pos="0 0 .001" size="2.5 .002 .001" rgba=".45 .45 .45 1" contype="0" conaffinity="0"/>
            <geom name="grid_y" type="box" pos="0 0 .001" size=".002 2.5 .001" rgba=".45 .45 .45 1" contype="0" conaffinity="0"/>
            <body name="cube" pos="0 0 1">
              <freejoint name="cube_free"/>
              <geom name="cube_collision" type="box" size=".1 .1 .1" mass="1" rgba=".12 .48 .85 1"/>
            </body>
          </worldbody>
        </mujoco>'''

    def _sync_viewer(self) -> None:
        self._synchronize()
        qpos = self.sim.data.qpos[self.view_env].cpu().numpy()
        qvel = self.sim.data.qvel[self.view_env].cpu().numpy()
        self._viewer_data.qpos[:] = qpos
        self._viewer_data.qvel[:] = qvel
        self._viewer_data.time = float(self.sim.data.time[self.view_env].cpu().item())
        if self.viewer is None or getattr(self, "visual_replay", False) or getattr(self, "_replaying", False):
            self._viewer_data.xfrc_applied[:] = self.sim.data.xfrc_applied[self.view_env].cpu().numpy()
        else:
            # The live passive viewer feeds this field back as mouse input in step().
            # Copying the total simulated wrench here would apply it a second time.
            self._viewer_data.xfrc_applied[:] = 0
            self._viewer_data.xfrc_applied[self._body_id, :3] = self._mouse_force
        mujoco.mj_forward(self.sim.mj_model, self._viewer_data)

    def _setup_robot_controller(self, checkpoint: str | None) -> None:
        """Compile the native asset contract used by both mjlab controllers."""
        if self.scene not in ("robot", "rider"):
            if self.scene_config is not None:
                from ..models.scene import initialize_scene
                nominal = mujoco.MjData(self.sim.mj_model)
                initialize_scene(self.sim.mj_model, nominal, self.scene_config)
                self._nominal_qpos = self._torch.as_tensor(np.repeat(nominal.qpos[None, :], self.num_envs, axis=0), device=self.sim.data.qpos.device)
                self._nominal_qvel = self._torch.as_tensor(np.repeat(nominal.qvel[None, :], self.num_envs, axis=0), device=self.sim.data.qvel.device)
                return
            self._nominal_qpos = self.sim.data.qpos.clone()
            self._nominal_qvel = self.sim.data.qvel.clone()
            return

        model = self.sim.mj_model
        self.robot_qpos_adr = np.asarray(
            [model.joint(name).qposadr[0] for name in DEPLOY_JOINTS], dtype=np.int64)
        self.robot_dof_adr = np.asarray(
            [model.joint(name).dofadr[0] for name in DEPLOY_JOINTS], dtype=np.int64)
        self.robot_qpos_index = self._torch.as_tensor(
            self.robot_qpos_adr, dtype=self._torch.int64, device=self.sim.data.qpos.device)
        self.robot_dof_index = self._torch.as_tensor(
            self.robot_dof_adr, dtype=self._torch.int64, device=self.sim.data.qvel.device)
        self.pelvis_body_id = model.body("pelvis").id
        self.policy_values = deploy_joint_values(
            Path(__file__).parents[3] / "configs" / "bfm" / "motivo_newG1.yaml")
        self.pose_target = np.tile(self.policy_values["default"], (self.num_envs, 1))
        self.policy_action = np.zeros((self.num_envs, 29), dtype=np.float64)
        self.raw_action = np.zeros((self.num_envs, 29), dtype=np.float64)
        self._action_queues: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self.q_target = np.tile(self.policy_values["default"], (self.num_envs, 1))
        self.last_tau = np.zeros((self.num_envs, 29), dtype=np.float64)
        self.last_qfrc_applied = np.zeros((self.num_envs, 29), dtype=np.float64)
        self.last_obs = np.zeros((self.num_envs, 721), dtype=np.float32)
        self.inference_time_ms = np.zeros(self.num_envs)
        self.robot_history = [HistoryState() for _ in range(self.num_envs)]
        self.policy_counter = 0
        self.physics_step_counter = 0
        self.policy_context_index = np.zeros(self.num_envs, dtype=np.int64)
        self.policy_tracking_active = np.ones(self.num_envs, dtype=bool)
        self.policy = None
        self.policy_z = None
        self.policy_context_name = None
        if self.controller == "policy":
            root = Path(checkpoint)
            onnx = root / "model" / "exported" / "FBcprAuxModel.onnx" if root.is_dir() else root
            self.policy = BFMOnnxPolicy(onnx)
            self.policy_z, self.policy_context_name = load_policy_context(
                root if root.is_dir() else root.parent, self.policy_task, None)

        nominal = mujoco.MjData(model)
        nominal.qpos[:] = model.qpos0
        nominal.qvel[:] = 0
        nominal.qpos[self.robot_qpos_adr] = self.policy_values["default"]
        if self.scene == "rider" and self.scene_config is None:
            deck_q = model.joint("deck_free")
            deck_qa = int(deck_q.qposadr[0])
            nominal.qpos[deck_qa:deck_qa + 7] = model.qpos0[deck_qa:deck_qa + 7]
            if self.preset == "separate":
                nominal.qpos[:3] = (0.0, -1.0, float(model.qpos0[2]))
                nominal.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            else:
                from ..robot.rider_pose import initialize_rider_pose
                initialize_rider_pose(model, nominal, self.preset)
                if self.preset == "steer_start":
                    deck_da = int(deck_q.dofadr[0])
                    nominal.qvel[deck_da] = 1.0
                    nominal.qvel[0] = 1.0
                    radius = board_styles(self.style)[self.style].wheel_radius
                    for prefix in ("front", "rear"):
                        for side in ("left", "right"):
                            nominal.joint(f"{prefix}_{side}_spin").qvel[:] = 1.0 / radius
        if self.scene_config is not None:
            from ..models.scene import initialize_scene
            initialize_scene(model, nominal, self.scene_config)
        self.pose_target[:] = nominal.qpos[self.robot_qpos_adr]
        mujoco.mj_forward(model, nominal)
        qpos = self._torch.as_tensor(np.repeat(nominal.qpos[None, :], self.num_envs, axis=0), device=self.sim.data.qpos.device)
        qvel = self._torch.as_tensor(np.repeat(nominal.qvel[None, :], self.num_envs, axis=0), device=self.sim.data.qvel.device)
        self._nominal_qpos = qpos.clone()
        self._nominal_qvel = qvel.clone()

    def _write_field(self, name: str, value) -> None:
        """Synchronously update a Warp-backed tensor before the next kernel."""
        field = getattr(self.sim.data, name)
        tensor = field._tensor
        source = value if isinstance(value, self._torch.Tensor) else self._torch.as_tensor(value, device=tensor.device, dtype=tensor.dtype)
        if source.numel() == 0 and tensor.numel() == 0:
            source = source.reshape(tensor.shape)
        tensor.copy_(source)
        if not hasattr(self, "_external_target"):
            self._synchronize()

    def _synchronize(self) -> None:
        self._wp.synchronize_device(self.sim.wp_device)
        self._torch.cuda.synchronize()

    def _capture_state_snapshot(self) -> None:
        """Capture the public simulation state after Warp has completed."""
        self._synchronize()
        self._state_snapshot = {
            "qpos": self.sim.data.qpos._tensor.detach().cpu().numpy().copy(),
            "qvel": self.sim.data.qvel._tensor.detach().cpu().numpy().copy(),
            "time": self.sim.data.time._tensor.detach().cpu().numpy().copy(),
            "xfrc_applied": self.sim.data.xfrc_applied._tensor.detach().cpu().numpy().copy(),
            "qfrc_applied": self.sim.data.qfrc_applied._tensor.detach().cpu().numpy().copy(),
        }

    def _read_state_snapshot(self) -> dict[str, np.ndarray]:
        if self._state_snapshot is None:
            self._capture_state_snapshot()
        return {name: value.copy() for name, value in self._state_snapshot.items()}

    def _robot_state_cpu(self):
        self._synchronize()
        qpos = self.sim.data.qpos._tensor
        qvel = self.sim.data.qvel._tensor
        quat = self.sim.data.xquat._tensor[:, self.pelvis_body_id, :]
        world_angvel = self.sim.data.cvel._tensor[:, self.pelvis_body_id, :3]
        rotation = self.sim.data.xmat._tensor[:, self.pelvis_body_id, :].reshape(self.num_envs, 3, 3)
        angvel = self._torch.bmm(rotation.transpose(1, 2), world_angvel.unsqueeze(-1)).squeeze(-1)
        return (qpos[:, self.robot_qpos_index].detach().cpu().numpy(),
                qvel[:, self.robot_dof_index].detach().cpu().numpy(),
                quat.detach().cpu().numpy(), angvel.detach().cpu().numpy())

    def _set_external_action(self, action):
        if self.scene not in ("robot", "rider"):
            raise ValueError("external actions require a robot scene")
        t = self._torch
        raw = t.as_tensor(action, device=self.sim.data.qpos.device, dtype=self.sim.data.qpos._tensor.dtype)
        if raw.shape != (self.num_envs, 29) or not bool(t.isfinite(raw).all()):
            raise ValueError(f"external action must be finite with shape ({self.num_envs}, 29)")
        if not hasattr(self, "_external_queue"):
            delay = max(int(row["control_delay_steps"]) for row in self.parameter_batch)
            self._external_queue = t.zeros((delay + 1, self.num_envs, 29), device=raw.device, dtype=raw.dtype)
        self._external_raw = raw.clone()
        self._external_queue = t.roll(self._external_queue, 1, 0)
        self._external_queue[0] = raw.clamp(-1, 1)
        delays = t.as_tensor([int(row["control_delay_steps"]) for row in self.parameter_batch], device=raw.device)
        applied = self._external_queue[delays, t.arange(self.num_envs, device=raw.device)]
        if not hasattr(self, "_external_values"):
            self._external_values = {k:t.as_tensor(v,device=raw.device,dtype=raw.dtype) for k,v in self.policy_values.items()}
        values = self._external_values
        self._external_target = (values["default"] + values["action_rescale"] * applied * values["action_scale"]).clamp(values["joint_pos_lower_limit"], values["joint_pos_upper_limit"])
        self.policy_counter += 1

    def control_step(self, action):
        """One batch control interval, with feedback at every physics tick."""
        self.step(action, update_policy=False)
        for _ in range(self.config.substeps - 1):
            self.step(update_policy=False)
    def _update_controller(self, update_policy=True) -> None:
        qfrc = self._torch.zeros_like(self.sim.data.qfrc_applied._tensor)
        if self.board_style:
            qvel = self.sim.data.qvel._tensor
            qpos = self.sim.data.qpos._tensor
            wheel_scale = self._torch.as_tensor(
                [row["wheel_spin_damping_scale"] for row in self.parameter_batch],
                device=qvel.device, dtype=qvel.dtype)
            bearing_scale = self._torch.as_tensor(
                [row["bearing_torque_scale"] for row in self.parameter_batch],
                device=qvel.device, dtype=qvel.dtype)
            friction_scale = self._torch.as_tensor(
                [row["friction_scale"] for row in self.parameter_batch],
                device=qvel.device, dtype=qvel.dtype)
            stiffness_scale = self._torch.as_tensor(
                [row["truck_stiffness_scale"] for row in self.parameter_batch],
                device=qvel.device, dtype=qvel.dtype)
            damping_scale = self._torch.as_tensor(
                [row["truck_damping_scale"] for row in self.parameter_batch],
                device=qvel.device, dtype=qvel.dtype)
            for index in self._board_wheel_dof:
                qfrc[:, index] += -(wheel_scale - 1.0) * 0.02 * qvel[:, index]
                qfrc[:, index] += -(bearing_scale - 1.0) * 0.0004 * self._torch.tanh(qvel[:, index] / 0.1)
            truck_k1 = float(self.compile_dynamics.get("truck_k1", self.board_style.truck_stiffness))
            truck_c = float(self.compile_dynamics.get("truck_c", self.board_style.truck_damping))
            for qpos_index, qvel_index in self._board_truck_joints:
                qfrc[:, qvel_index] += -(stiffness_scale - 1.0) * truck_k1 * qpos[:, qpos_index]
                qfrc[:, qvel_index] += -(damping_scale - 1.0) * truck_c * qvel[:, qvel_index]
            if self._board_root_dof is not None:
                qfrc[:, self._board_root_dof] += -(friction_scale - 1.0) * 0.5 * qvel[:, self._board_root_dof]
                style = self.board_style
                nominal_mass = style.total_mass
                effective_mass = (
                    style.deck_mass * self._torch.as_tensor(
                        [row["deck_mass_scale"] for row in self.parameter_batch], device=qvel.device, dtype=qvel.dtype)
                    + 2.0 * style.truck_mass * self._torch.as_tensor(
                        [row["truck_mass_scale"] for row in self.parameter_batch], device=qvel.device, dtype=qvel.dtype)
                        + 4.0 * style.wheel_mass * self._torch.as_tensor(
                        [row["wheel_mass_scale"] for row in self.parameter_batch], device=qvel.device, dtype=qvel.dtype))
                qfrc[:, self._board_root_dof + 2] += -(effective_mass - nominal_mass) * 9.81
                offsets = self._torch.as_tensor(
                    [[row["deck_com_offset_x"], row["deck_com_offset_y"], 0.0]
                     for row in self.parameter_batch], device=qvel.device, dtype=qvel.dtype)
                rotation = self.sim.data.xmat._tensor[:, self._board_deck_body_id, :].reshape(
                    self.num_envs, 3, 3)
                offset_world = self._torch.bmm(rotation, offsets.unsqueeze(-1)).squeeze(-1)
                gravity_force = self._torch.zeros_like(offset_world)
                gravity_force[:, 2] = -effective_mass * 9.81
                qfrc[:, self._board_root_dof + 3:self._board_root_dof + 6] += self._torch.cross(
                    offset_world, gravity_force, dim=1)
        if hasattr(self, "_external_target"):
            t = self._torch
            q = self.sim.data.qpos._tensor[:, self.robot_qpos_index]
            dq = self.sim.data.qvel._tensor[:, self.robot_dof_index]
            values = self._external_values
            if not hasattr(self,"_external_pd_scales"):
                self._external_pd_scales = tuple(t.as_tensor([row[key] for row in self.parameter_batch],device=q.device,dtype=q.dtype)[:,None] for key in ("pd_kp_scale","pd_kd_scale"))
            kp,kd = self._external_pd_scales
            tau = (values["kp"] * kp * (self._external_target - q) - values["kd"] * kd * dq).clamp(-values["joint_effort_limit"], values["joint_effort_limit"])
            self._external_tau = tau
            qfrc[:, self.robot_dof_index] = tau
            self._write_field("qfrc_applied", qfrc)
            return
        if self.scene not in ("robot", "rider") or self.controller == "none":
            self._write_field("qfrc_applied", qfrc)
            return
        q, dq, quat, angvel = self._robot_state_cpu()
        kp_scale = np.asarray([row["pd_kp_scale"] for row in self.parameter_batch])[:, None]
        kd_scale = np.asarray([row["pd_kd_scale"] for row in self.parameter_batch])[:, None]
        if self.controller == "policy" and update_policy and self.physics_step_counter % self.config.substeps == 0:
            if not hasattr(self,"inference_time_ms"): self.inference_time_ms=np.zeros(self.num_envs)
            for env_id in range(self.num_envs):
                context = tracking_context(self.policy_z, self.policy_context_index[env_id]) if self.policy_task == "tracking" else self.policy_z
                obs = build_official_actor_input(
                    q[env_id], dq[env_id], quat[env_id], angvel[env_id],
                    self.robot_history[env_id].last_action, self.robot_history[env_id], context,
                    sensor_noise_std=self.parameter_batch[env_id]["observation_noise_std"],
                    rng=np.random.default_rng(self.seed + self.physics_step_counter + env_id))
                inference_started=time.perf_counter()
                raw = np.asarray(self.policy(obs), dtype=np.float64)
                self.inference_time_ms[env_id]=(time.perf_counter()-inference_started)*1000
                if raw.shape != (29,) or not np.isfinite(raw).all():
                    raise RuntimeError(f"BFM action contract failed: shape={raw.shape} finite={np.isfinite(raw).all()}")
                self.last_obs[env_id] = obs
                self.raw_action[env_id] = raw
                candidate_action = np.clip(raw, -1.0, 1.0)
                self.robot_history[env_id].last_action[:] = float(self.policy_values["action_rescale"]) * candidate_action
                queue = self._action_queues[env_id]
                queue.append(candidate_action.copy())
                delay = int(self.parameter_batch[env_id]["control_delay_steps"])
                self.policy_action[env_id] = (queue.pop(0)
                                              if len(queue) > delay else np.zeros(29, dtype=np.float64))
                self.q_target[env_id] = np.clip(
                    self.policy_values["default"] + self.policy_values["action_rescale"] *
                    self.policy_action[env_id] * self.policy_values["action_scale"],
                    self.policy_values["joint_pos_lower_limit"],
                    self.policy_values["joint_pos_upper_limit"])
            if self.policy_task == "tracking":
                for env_id in range(self.num_envs):
                    self.policy_context_index[env_id], self.policy_tracking_active[env_id] = next_tracking_cursor(
                        self.policy_context_index[env_id], self.policy_tracking_active[env_id], len(self.policy_z))
            self.policy_counter += 1
        elif self.controller == "pose":
            self.q_target[:] = self.pose_target
        self._last_robot_q_pre_step=q.copy()
        tau = self.policy_values["kp"] * kp_scale * (self.q_target - q) - self.policy_values["kd"] * kd_scale * dq
        tau = np.clip(tau, -self.policy_values["joint_effort_limit"], self.policy_values["joint_effort_limit"])
        self.last_tau = tau
        qfrc[:, self.robot_dof_index] = self._torch.as_tensor(tau, device=qfrc.device, dtype=qfrc.dtype)
        self._write_field("qfrc_applied", qfrc)
        self.last_qfrc_applied = tau.copy()

    def _resample_parameters(self, env_ids: np.ndarray | None, seed: int | None) -> None:
        if seed is not None:
            self.seed = int(seed)
        if env_ids is None:
            self.parameter_batch = sample_parameter_batch(
                self.num_envs, seed=self.seed, randomize=self.randomize, source=self.parameter_spec)
        else:
            for env_id in env_ids.tolist():
                self.parameter_batch[env_id] = resolve_parameters(
                    self.parameter_spec,
                    randomize=self.randomize, seed=self.seed + int(env_id) * 1009)
        self.parameters = self.parameter_batch[self.view_env]
        self.dynamics = effective_dynamics(self.base_dynamics, self.parameters)
        self._parameter_metadata = parameter_metadata(
            self.parameters, source=self.parameter_source, seed=self.seed,
            randomize=self.randomize)

    def reset(self, seed: int | None = None, env_ids: Any = None) -> np.ndarray | None:
        ids = None if env_ids is None else self._env_ids(env_ids)
        if getattr(self,"recorder",None):
            if not hasattr(self,"_pending_inputs"): self._pending_inputs=[]
            self._pending_inputs.append({"event":"reset","seed":seed,"env_ids":None if ids is None else ids.tolist()})
        self._resample_parameters(ids, seed)
        if hasattr(self,"_external_pd_scales"): del self._external_pd_scales
        snapshot = self._read_state_snapshot()
        if ids is None:
            qpos = self._nominal_qpos.detach().cpu().numpy().copy()
            qvel = self._nominal_qvel.detach().cpu().numpy().copy()
            time = np.zeros_like(snapshot["time"])
            xfrc = np.zeros_like(snapshot["xfrc_applied"])
            qfrc = np.zeros_like(snapshot["qfrc_applied"])
            self._force[:] = 0
            self._mouse_force[:] = 0
        else:
            qpos = snapshot["qpos"]
            qvel = snapshot["qvel"]
            time = snapshot["time"]
            xfrc = snapshot["xfrc_applied"]
            qfrc = snapshot["qfrc_applied"]
            nominal_qpos = self._nominal_qpos.detach().cpu().numpy()
            nominal_qvel = self._nominal_qvel.detach().cpu().numpy()
            reset_set = set(ids.tolist())
            qpos = np.stack([nominal_qpos[i] if i in reset_set else snapshot["qpos"][i]
                             for i in range(self.num_envs)])
            qvel = np.stack([nominal_qvel[i] if i in reset_set else snapshot["qvel"][i]
                             for i in range(self.num_envs)])
            time = np.asarray([0 if i in reset_set else snapshot["time"][i]
                               for i in range(self.num_envs)], dtype=snapshot["time"].dtype)
            xfrc = np.stack([np.zeros_like(snapshot["xfrc_applied"][i]) if i in reset_set else snapshot["xfrc_applied"][i]
                             for i in range(self.num_envs)])
            qfrc = np.stack([np.zeros_like(snapshot["qfrc_applied"][i]) if i in reset_set else snapshot["qfrc_applied"][i]
                             for i in range(self.num_envs)])
            if self.view_env in reset_set:
                self._force[:] = 0
                self._mouse_force[:] = 0
        if self._board_root_qpos is not None:
            reset_indices = range(self.num_envs) if ids is None else ids.tolist()
            for env_id in reset_indices:
                qpos[env_id, self._board_root_qpos] += self.parameter_batch[env_id]["deck_com_offset_x"]
                qpos[env_id, self._board_root_qpos + 1] += self.parameter_batch[env_id]["deck_com_offset_y"]
        device = self.sim.data.qpos._tensor.device
        self._write_field("qpos", qpos)
        self._write_field("qvel", qvel)
        self._write_field("time", time)
        self._write_field("xfrc_applied", xfrc)
        self._write_field("qfrc_applied", qfrc)
        if self._gripper_ids:
            ctrl = self.sim.data.ctrl._tensor
            reset_lanes = slice(None) if ids is None else self._torch.as_tensor(ids, device=ctrl.device)
            ctrl[reset_lanes] = 0
        if self.scene in ("robot", "rider"):
            reset_ids = range(self.num_envs) if ids is None else ids.tolist()
            for env_id in reset_ids:
                self.robot_history[env_id].reset()
                self.policy_context_index[env_id] = 0
                self.policy_tracking_active[env_id] = True
                self.policy_action[env_id] = 0
                self.raw_action[env_id] = 0
                self._action_queues[env_id].clear()
                self.last_obs[env_id] = 0
                self.last_tau[env_id] = 0
                self.last_qfrc_applied[env_id] = 0
                self.q_target[env_id] = self.policy_values["default"]
                self.pose_target[env_id] = self._nominal_qpos[env_id, self.robot_qpos_index].detach().cpu().numpy()
            if ids is None:
                self.policy_counter = 0
                self.policy_context_index = np.zeros(self.num_envs, dtype=np.int64)
                self.physics_step_counter = 0
        for key in list(getattr(self, "_wrench_steps", {})):
            if ids is None or key[0] in ids:
                del self._wrench_steps[key]
        if ids is None or self.view_env in ids:
            self._pulse_steps = 0
        if hasattr(self, "_external_queue"):
            reset_ids = slice(None) if ids is None else self._torch.as_tensor(ids, device=device)
            self._external_queue[:, reset_ids] = 0
            self._external_raw[reset_ids] = 0
            if hasattr(self,"_external_terminated"): self._external_terminated[reset_ids]=False
            if hasattr(self,"_external_board_approached"): self._external_board_approached[reset_ids]=self.preset!="separate"
            self._external_target[reset_ids] = self._torch.as_tensor(self.policy_values["default"], device=device, dtype=self._external_target.dtype)
        self._synchronize()
        self.sim.forward()
        self._sync_viewer()
        self._capture_state_snapshot()
        root = (self.sim.data.xpos._tensor[:,self._body_id,:].cpu().numpy()
                if hasattr(self.sim.data,"xpos") else qpos[:,:3])
        if getattr(self,"_episode_origin",None) is None: self._episode_origin=root.copy()
        reset_indices = np.arange(self.num_envs) if ids is None else ids
        self._episode_origin[reset_indices]=root[reset_indices]
        if self.scene == "rider":
            deck = self.sim.data.xpos._tensor[:,self.sim.mj_model.body("deck").id,:].cpu().numpy()
            if not hasattr(self,"_deck_episode_origin"): self._deck_episode_origin=deck.copy()
            self._deck_episode_origin[reset_indices]=deck[reset_indices]
        if not hasattr(self,"_board_approached"): self._board_approached=np.zeros(self.num_envs,dtype=bool)
        self._board_approached[reset_indices] = getattr(self,"preset","separate") != "separate"
        if self.scene in ("robot", "rider") and self.controller == "policy":
            q, dq, quat, angvel = self._robot_state_cpu()
            reset_ids = np.arange(self.num_envs) if ids is None else ids
            for env_id in reset_ids:
                context = tracking_context(self.policy_z, 0) if self.policy_task == "tracking" else self.policy_z
                self.last_obs[env_id] = build_official_actor_input(
                    q[env_id], dq[env_id], quat[env_id], angvel[env_id],
                    np.zeros(29), HistoryState(), context)
            return self.last_obs[reset_ids].copy()

    def step(self, action: Any = None, *, update_policy=True) -> None:
        if action is not None:
            self._set_external_action(action)
        record_command = None
        record_inputs = list(getattr(self,"_pending_inputs",[]))
        self._pending_inputs=[]
        if self.viewer is not None and not getattr(self,"_replaying",False) and not getattr(self,"visual_replay",False):
            with self.viewer.lock():
                self.viewer.sync()
        if self.viewer is not None and not getattr(self,"_replaying",False) and not getattr(self,"visual_replay",False):
            self._mouse_force[:] = self._viewer_data.xfrc_applied[self._body_id, :3]
        if getattr(self, "recorder", None):
            record_command = {"update_policy":update_policy,"controller": self.controller, "force": self._force.copy(),
                              "mouse_force": self._mouse_force.copy(), "pulse_steps": self._pulse_steps,
                              "pose_target": getattr(self, "pose_target", None),
                              "xfrc_applied": self._read_state_snapshot()["xfrc_applied"],
                              "wrench_steps": [[e,b,n] for (e,b),n in getattr(self,"_wrench_steps",{}).items()],
                              "external_control": hasattr(self,"_external_target"), "external_action_update": action is not None}
        total_force = self._mouse_force + self._force
        xfrc = self.sim.data.xfrc_applied._tensor.clone()
        xfrc[self.view_env, self._body_id, :3] += self._torch.as_tensor(total_force, device=xfrc.device, dtype=xfrc.dtype)
        self._write_field("xfrc_applied", xfrc)
        self._update_controller(update_policy=update_policy)
        if hasattr(self, "_external_target"):
            with self._wp.ScopedStream(self._wp.stream_from_torch(self._torch.cuda.current_stream())):
                self.sim.step()
        else:
            self.sim.step()
        self.physics_step_counter += 1
        if hasattr(self,"_external_target") and self.physics_step_counter % self.config.substeps == 0:
            with self._wp.ScopedStream(self._wp.stream_from_torch(self._torch.cuda.current_stream())):
                self.sim.forward()
        # Remove the per-step interactive contribution without clearing other env/body forces.
        xfrc[self.view_env, self._body_id, :3] -= self._torch.as_tensor(total_force, device=xfrc.device, dtype=xfrc.dtype)
        for key, remaining in list(getattr(self, "_wrench_steps", {}).items()):
            if remaining is not None:
                remaining -= 1
                if remaining == 0:
                    xfrc[key[0], key[1]] = 0
                    del self._wrench_steps[key]
                else: self._wrench_steps[key] = remaining
        self._write_field("xfrc_applied", xfrc)
        if self._pulse_steps > 0:
            self._pulse_steps -= 1
            if self._pulse_steps == 0:
                self._force[:] = 0
                self.apply_wrench(self.sim.mj_model.body(self._body_id).name, (0, 0, 0))
                print("[bench] pulse finished; force=0 N", flush=True)
        if self.scene == "rider":
            positions = self.sim.data.xpos._tensor
            distance = self._torch.linalg.vector_norm(positions[:,self.pelvis_body_id,:2]-positions[:,self.sim.mj_model.body("deck").id,:2],dim=1)
            if hasattr(self, "_external_target"):
                if not hasattr(self, "_external_board_approached"):
                    self._external_board_approached = self._torch.as_tensor(self._board_approached, device=distance.device).clone()
                self._external_board_approached |= distance <= .5
                if not hasattr(self,"_external_terminated"):
                    self._external_terminated = self._torch.zeros(self.num_envs,device=distance.device,dtype=self._torch.bool)
                root=positions[:,self.pelvis_body_id]
                deck=positions[:,self.sim.mj_model.body("deck").id]
                q=self.sim.data.qpos._tensor
                dq=self.sim.data.qvel._tensor
                bad=~self._torch.isfinite(q).all(1) | ~self._torch.isfinite(dq).all(1) | (root[:,2]<.35)
                joint=q[:,self.robot_qpos_index]
                bad |= ((joint<self._external_values["joint_pos_lower_limit"]-.05)|(joint>self._external_values["joint_pos_upper_limit"]+.05)).any(1)
                if self.offboard_distance is not None:
                    bad |= self._external_board_approached & (distance>self.offboard_distance)
                if self.out_of_bounds_distance is not None:
                    origin=self._torch.as_tensor(self._episode_origin,device=root.device,dtype=root.dtype)
                    deck_origin=self._torch.as_tensor(self._deck_episode_origin,device=root.device,dtype=root.dtype)
                    bad |= self._torch.maximum((root-origin).norm(dim=1),(deck-deck_origin).norm(dim=1))>self.out_of_bounds_distance
                self._external_terminated |= bad
            else:
                self._board_approached |= distance.cpu().numpy() <= .5
        if self.viewer is not None:
            self._sync_viewer()
        self._state_snapshot = None
        if record_command is not None:
            recorded_action = self._external_raw.detach().cpu().numpy().copy() if hasattr(self,"_external_raw") else self.raw_action.copy() if self.scene in ("robot", "rider") else None
            self.recorder.tick(self.get_state(), action=recorded_action,
                               action_source="external" if hasattr(self,"_external_target") else self.controller, command=record_command, inputs=record_inputs)

    def export_replay_state(self):
        from dataclasses import asdict
        import copy
        names = ("controller", "physics_step_counter", "policy_counter", "policy_context_index",
                 "policy_tracking_active", "q_target", "pose_target", "policy_action", "raw_action",
                 "last_tau", "last_qfrc_applied", "last_obs", "_action_queues", "_force", "_mouse_force",
                 "_pulse_steps", "seed", "parameter_batch", "parameters", "dynamics", "_episode_origin", "_deck_episode_origin", "_board_approached", "episode_time_limit", "out_of_bounds_distance", "offboard_distance", "_parameter_metadata")
        integration = {name:getattr(self.sim.data,name)._tensor.cpu().numpy().copy()
                       for name in ("qacc_warmstart", "qacc", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active") if hasattr(self.sim.data,name)}
        return {"external": {name:getattr(self,name).detach().cpu().numpy().copy() for name in
                             ("_external_queue", "_external_target", "_external_raw", "_external_tau", "_external_board_approached", "_external_terminated") if hasattr(self,name)},
                "integration": integration, "runtime": copy.deepcopy({n:getattr(self,n) for n in names if hasattr(self,n)}),
                "history": [asdict(h) for h in getattr(self,"robot_history",[])],
                "wrench_steps": [[e,b,n] for (e,b),n in getattr(self,"_wrench_steps",{}).items()]}

    def import_replay_state(self, state):
        for name,value in state.get("external",{}).items():
            setattr(self,name,self._torch.as_tensor(value,device=self.sim.data.qpos.device).clone())
        if hasattr(self,"_external_target"):
            self._external_values={k:self._torch.as_tensor(v,device=self.sim.data.qpos.device,dtype=self.sim.data.qpos._tensor.dtype) for k,v in self.policy_values.items()}
        for name,value in state.get("integration",{}).items(): self._write_field(name,value)
        for name,value in state["runtime"].items():
            if name == "_action_queues": value = [[np.asarray(v) for v in queue] for queue in value]
            elif isinstance(getattr(self,name,None),np.ndarray): value=np.asarray(value,dtype=getattr(self,name).dtype)
            setattr(self,name,value)
        for history,values in zip(getattr(self,"robot_history",[]),state.get("history",[])):
            for name,value in values.items():
                setattr(history,name,np.asarray(value,dtype=np.float32) if name == "last_action" else [np.asarray(v,dtype=np.float32) for v in value])
        self._wrench_steps = {(int(e),int(b)):n for e,b,n in state.get("wrench_steps",[])}

    def step_recorded(self, action, command, inputs):
        if not command: raise ValueError("mjlab replay requires physics inputs")
        for event in inputs or []:
            if event.get("event") == "reset": self.reset(event.get("seed"),env_ids=event.get("env_ids"))
        self.controller = command["controller"]
        self._force[:] = command["force"]
        self._mouse_force[:] = command["mouse_force"]
        self._pulse_steps = int(command["pulse_steps"])
        if command.get("pose_target") is not None: self.pose_target=np.asarray(command["pose_target"])
        self._write_field("xfrc_applied",command["xfrc_applied"])
        self._wrench_steps={(int(e),int(b)):n for e,b,n in command.get("wrench_steps",[])}
        policy=getattr(self,"policy",None)
        if self.controller == "policy":
            actions=iter(np.asarray(action))
            self.policy=lambda obs: next(actions).copy()
        self._replaying=True
        try:
            self.step(action if command.get("external_action_update") else None,
                      update_policy=False if command.get("external_control") else command.get("update_policy",True))
        finally:
            self.policy=policy
            self._replaying=False

    def _env_ids(self, env_ids=None, *, default_view=False):
        if env_ids is None:
            return np.asarray([self.view_env]) if default_view else np.arange(self.num_envs)
        return validate_env_ids(env_ids, self.num_envs)

    def get_state(self, env_ids=None) -> dict[str, Any]:
        ids = self._env_ids(env_ids)
        return {key: value[ids].copy() for key, value in self._read_state_snapshot().items()}

    def rider_initial_state(self) -> dict[str, Any]:
        """Return the selected GPU environment's compiled rider summary."""
        if self.scene != "rider":
            raise ValueError("rider_initial_state requires scene=rider")
        model = self.sim.mj_model
        data = self._viewer_data
        pelvis = model.body("pelvis").id
        deck = model.body("deck").id
        left = model.body("left_ankle_roll_link").id
        right = model.body("right_ankle_roll_link").id
        deck_joint = model.joint("deck_free")
        dq = int(deck_joint.dofadr[0])
        return {
            "preset": self.preset,
            "backend": self.name,
            "physics_owner": "mjlab.mujoco_warp",
            "device": str(self.sim.device),
            "pelvis_position": data.xpos[pelvis].tolist(),
            "left_foot_position": data.xpos[left].tolist(),
            "right_foot_position": data.xpos[right].tolist(),
            "deck_position": data.xpos[deck].tolist(),
            "board_linear_velocity": self.sim.data.qvel[self.view_env, dq:dq + 3].cpu().tolist(),
            "board_angular_velocity": self.sim.data.qvel[self.view_env, dq + 3:dq + 6].cpu().tolist(),
            "contact_count": int(data.ncon),
            "termination": self.termination()[self.view_env],
        }

    def set_state(self, state: dict[str, Any], env_ids=None) -> None:
        ids = self._env_ids(env_ids)
        snapshot = self._read_state_snapshot()
        if not {"qpos", "qvel"}.issubset(state):
            raise ValueError("state requires qpos and qvel")
        updates = {}
        for key, value in state.items():
            if key not in snapshot:
                raise ValueError(f"unknown state field {key!r}")
            array = np.asarray(value, dtype=snapshot[key].dtype)
            expected = (len(ids), *snapshot[key].shape[1:])
            if key == "time" and array.ndim == 0:
                array = np.full(expected, array, dtype=snapshot[key].dtype)
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"{key} requires finite shape {expected}")
            updates[key] = array.copy()
        qpos = updates["qpos"]
        for joint in range(self.sim.mj_model.njnt):
            kind = self.sim.mj_model.jnt_type[joint]
            if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                adr = int(self.sim.mj_model.jnt_qposadr[joint]) + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
                norm = np.linalg.norm(qpos[:, adr:adr+4], axis=1)
                if np.any(np.abs(norm - 1) > 1e-4):
                    raise ValueError("state quaternion must be unit length")
        # Validate every field before touching any GPU state.
        for key, array in updates.items():
            snapshot[key][ids] = array
            self._write_field(key, snapshot[key])
        self._synchronize()
        self.sim.forward()
        self._sync_viewer()
        self._capture_state_snapshot()

    def apply_wrench(self, body: str, force, point=None, *, env_ids=None,
                     frame="world", duration_steps=None) -> None:
        """Replace selected body forces; points/forces use frame, torque is about COM.

        No point means a COM force. None duration persists; a positive duration
        counts physics steps. A zero force clears only this body/selection.
        """
        ids = self._env_ids(env_ids, default_view=True)
        try:
            body_id = int(self.sim.mj_model.body(body).id)
        except KeyError as exc:
            raise ValueError(f"unknown body {body!r}") from exc
        if body_id == 0 or frame not in ("world", "body"):
            raise ValueError("wrench requires a dynamic body and world/body frame")
        if duration_steps is not None and (not isinstance(duration_steps, int) or isinstance(duration_steps, bool) or duration_steps < 1):
            raise ValueError("duration_steps must be a positive integer or None")
        def vectors(value, name):
            array = np.asarray(value, dtype=float)
            if array.shape == (3,): array = np.tile(array, (len(ids), 1))
            if array.shape != (len(ids), 3) or not np.isfinite(array).all():
                raise ValueError(f"{name} requires finite 3-vector or (selected environments,3)")
            return array
        forces = vectors(force, "force")
        points = None if point is None else vectors(point, "point")
        self._synchronize()
        rotation = self.sim.data.xmat._tensor[ids, body_id].cpu().numpy().reshape(-1,3,3)
        origin = self.sim.data.xpos._tensor[ids, body_id].cpu().numpy()
        com = self.sim.data.xipos._tensor[ids, body_id].cpu().numpy()
        if frame == "body":
            forces = np.einsum("nij,nj->ni", rotation, forces)
            if points is not None: points = origin + np.einsum("nij,nj->ni", rotation, points)
        values = np.zeros((len(ids), 6))
        values[:, :3] = forces
        if points is not None: values[:, 3:] = np.cross(points-com, forces)
        if not hasattr(self, "_wrench_steps"):
            self._wrench_steps = {}
        xfrc = self._read_state_snapshot()["xfrc_applied"]
        xfrc[ids, body_id] = values
        for env in ids:
            self._wrench_steps[(int(env), body_id)] = duration_steps
        self._write_field("xfrc_applied", xfrc)
        self._capture_state_snapshot()

    def clear_wrenches(self, env_ids=None) -> None:
        """Clear all body wrenches for the selected environments."""
        ids = self._env_ids(env_ids)
        xfrc = self._read_state_snapshot()["xfrc_applied"]
        xfrc[ids] = 0
        for key in list(getattr(self, "_wrench_steps", {})):
            if key[0] in ids:
                del self._wrench_steps[key]
        if self.view_env in ids:
            self._force[:] = 0
            self._mouse_force[:] = 0
            self._pulse_steps = 0
            self._viewer_data.xfrc_applied[:] = 0
            if self.viewer is not None:
                with self.viewer.lock():
                    self.viewer.perturb.active = 0
                    self.viewer.perturb.active2 = 0
        self._write_field("xfrc_applied", xfrc)
        self._capture_state_snapshot()
        if getattr(self, "recorder", None):
            if not hasattr(self, "_pending_inputs"):
                self._pending_inputs = []
            self._pending_inputs.append({"event": "clear_wrenches", "env_ids": ids.tolist()})

    def termination(self):
        state = self._read_state_snapshot()
        finite = np.isfinite(state["qpos"]).all(axis=1) & np.isfinite(state["qvel"]).all(axis=1)
        reasons = [None if ok else "nonfinite" for ok in finite]
        if self.scene in ("robot", "rider"):
            self._synchronize()
            height = self.sim.data.xpos._tensor[:, self.pelvis_body_id, 2].cpu().numpy()
            q = state["qpos"][:, self.robot_qpos_adr]
            limits = (q < self.policy_values["joint_pos_lower_limit"] - .05) | (q > self.policy_values["joint_pos_upper_limit"] + .05)
            for env in range(self.num_envs):
                if reasons[env] is None and height[env] < .35: reasons[env] = "fall"
                if reasons[env] is None and limits[env].any(): reasons[env] = "joint_limit"
        root = self.sim.data.xpos._tensor[:,self._body_id,:].cpu().numpy()
        distance = np.linalg.norm(root-self._episode_origin,axis=1)
        if self.scene == "rider":
            deck = self.sim.data.xpos._tensor[:,self.sim.mj_model.body("deck").id,:].cpu().numpy()
            offboard = np.linalg.norm(root[:,:2]-deck[:,:2],axis=1)
            distance = np.maximum(distance,np.linalg.norm(deck-self._deck_episode_origin,axis=1))
        for env in range(self.num_envs):
            if reasons[env] is None and self.out_of_bounds_distance is not None and distance[env] > self.out_of_bounds_distance:
                reasons[env]="out_of_bounds"
            if self.scene == "rider" and reasons[env] is None and self.offboard_distance is not None and self._board_approached[env] and offboard[env] > self.offboard_distance:
                reasons[env]="offboard"
        return [{"terminated": reason is not None,
                 "truncated": bool(getattr(self, "episode_time_limit", None) is not None and state["time"][env] >= self.episode_time_limit),
                 "termination_reason": reason,
                 "truncation_reason": "time_limit" if getattr(self,"episode_time_limit",None) is not None and state["time"][env] >= self.episode_time_limit else None}
                for env, reason in enumerate(reasons)]

    def numerics_manifest(self) -> dict:
        from .warp import numerics_manifest
        return numerics_manifest(self.sim)

    def diagnostics(self) -> dict[str, Any]:
        body_name = self.sim.mj_model.body(self._body_id).name
        result = {"backend": self.name, "version": mujoco.__version__, "object": body_name,
                "sim_time": float(self.sim.data.time[self.view_env].cpu().numpy()),
                "num_envs": self.num_envs, "view_env": self.view_env,
                "cube_pos": self._viewer_data.xpos[self._body_id].tolist(),
                "cube_vel": self._viewer_data.cvel[self._body_id, 3:].tolist(),
                "force_world_N": (self._mouse_force + self._force).tolist(),
                "keyboard_force_N": self._force.tolist(), "mouse_force_N": self._mouse_force.tolist(),
                "mouse_body": body_name, "contacts": int(self._viewer_data.ncon),
                "device": str(self.sim.device), "physics_owner": "mjlab.mujoco_warp",
                "controller": self.controller, "physics_step_counter": self.physics_step_counter,
                "policy_counter": self.policy_counter, "parameter_source": self.parameter_source,
                "randomized": self.randomize,
                "parameters": dict(self.parameter_batch[self.view_env]),
                "parameter_summary": parameter_summary(self.parameter_batch),
                "parameter_metadata": self._parameter_metadata}
        done = self.termination()
        result.update({"guard_thresholds": {"episode_time_limit":self.episode_time_limit,
                       "out_of_bounds_distance":self.out_of_bounds_distance,"offboard_distance":self.offboard_distance},
                       "termination": done[self.view_env], "termination_batch": done,
                       "health": {"finite": bool(np.isfinite(self.get_state()["qpos"]).all() and np.isfinite(self.get_state()["qvel"]).all()),
                                  "physics_device": str(self.sim.device), "policy_device": "CPUExecutionProvider" if self.controller == "policy" else None}})
        if self.scene in ("robot", "rider") and self.last_tau is not None:
            env = self.view_env
            result.update({
                "obs_shape": [721] if self.controller == "policy" else None,
                "raw_action_norm": float(np.linalg.norm(self.raw_action[env])),
                "qfrc_applied_norm": float(np.linalg.norm(self.last_qfrc_applied[env])),
                "inference_time_ms": float(self.inference_time_ms[env]),
                "torque_injection_error": float(np.max(np.abs(self.get_state()["qfrc_applied"][env,self.robot_dof_adr]-self.last_tau[env]))),
                "q_change_after_mj_step": float(np.max(np.abs(self.get_state()["qpos"][env,self.robot_qpos_adr]-self._last_robot_q_pre_step[env]))) if hasattr(self,"_last_robot_q_pre_step") else None,
            })
        return result

    def render(self) -> bool:
        if self.viewer is None:
            # Switch the display-only wrench field to the live mouse input channel.
            if not getattr(self, "visual_replay", False):
                self._viewer_data.xfrc_applied[:] = 0
                self._viewer_data.xfrc_applied[self._body_id, :3] = self._mouse_force
            self.viewer = mujoco.viewer.launch_passive(self.sim.mj_model, self._viewer_data,
                                                       key_callback=self._key_callback)
            self.viewer.cam.lookat[:] = (0.0, 0.0, 0.1 if self.board_style else 0.35)
            self.viewer.cam.distance = 1.8 if self.board_style else 3.2
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25
        self.viewer.sync()
        return not self.viewer.is_running()

    def render_web(self, style_hash: str = "") -> bool:
        if self._viser_viewer is None:
            import viser
            import mjviser
            self._viser_server=viser.ViserServer(label="skate-sim-mjlab", port=0)
            style_select = self._viser_server.gui.add_dropdown("Board asset", options=["street", "standard", "cruiser", "longboard", "downhill"], initial_value=self.style)
            self._style_select = style_select
            style_select.disabled = bool(not self.board_style or getattr(self,"visual_replay",False) or getattr(self,"recorder",None))
            @style_select.on_update
            def _(_):
                if getattr(self,"visual_replay",False) or getattr(self,"recorder",None):
                    style_select.value=self.style
                    return
                if style_select.value != self.style:
                    self._requested_style = style_select.value
            fx = self._viser_server.gui.add_slider("Force X (N)", min=-30, max=30, step=.5, initial_value=0)
            fy = self._viser_server.gui.add_slider("Force Y (N)", min=-30, max=30, step=.5, initial_value=0)
            fz = self._viser_server.gui.add_slider("Force Z (N)", min=-30, max=30, step=.5, initial_value=0)
            release = self._viser_server.gui.add_button("Release Force")
            for handle in (fx,fy,fz,release): handle.disabled=bool(getattr(self,"visual_replay",False))
            @fx.on_update
            def _(_):
                if not getattr(self,"visual_replay",False): self._force[:] = (fx.value, fy.value, fz.value)
            @fy.on_update
            def _(_):
                if not getattr(self,"visual_replay",False): self._force[:] = (fx.value, fy.value, fz.value)
            @fz.on_update
            def _(_):
                if not getattr(self,"visual_replay",False): self._force[:] = (fx.value, fy.value, fz.value)
            @release.on_click
            def _(_):
                if getattr(self,"visual_replay",False): return
                fx.value = fy.value = fz.value = 0
                self._force[:] = 0
            def step_fn(model,data):
                # The built-in Step button calls this only while paused.
                if self._viser_viewer is not None and self._viser_viewer._paused and not getattr(self,"visual_replay",False):
                    self.step(update_policy=False)
            def render_fn(scene):
                self._sync_viewer()
                scene.update_from_mjdata(self._viewer_data)
            def reset_fn(model,data):
                if not getattr(self,"visual_replay",False): self.reset()
            self._viser_viewer=mjviser.Viewer(
                self.sim.mj_model,
                self._viewer_data,
                step_fn=step_fn,
                render_fn=render_fn,
                reset_fn=reset_fn,
                server=self._viser_server,
            )
            self._viser_server.gui.add_markdown("## skate-sim · mjlab / MuJoCo Warp\nWhite Viser viewer. Physics: CUDA Warp.\nAsset hash: `"+style_hash+"`\nUse X/Y/Z sliders for a world-frame force on the deck. Release Force clears it.")
            self._viser_viewer._setup_gui()
            self._viser_viewer._paused = self.paused
            print(f"mjlab Viser viewer: http://localhost:{self._viser_server.get_port()}",flush=True)
        self._sync_viewer()
        if self._viser_viewer:
            self._viser_viewer._tick()
            self.paused = bool(self._viser_viewer._paused)
        return False

    def switch_style(self, style: str) -> None:
        """Replace the active board simulation while keeping the Viser server alive."""
        if getattr(self,"visual_replay",False) or getattr(self,"recorder",None):
            raise RuntimeError("asset switching is disabled during recording or visual replay")
        if not self.board_style:
            raise RuntimeError("asset switching is available for board/gallery scenes")
        if style == self.style:
            return
        import torch
        from mjlab.sim import Simulation, SimulationCfg

        old_style = self.style
        new_style = board_styles(style)[style]
        new_spec = mujoco.MjSpec.from_string(build_board_xml(
            new_style, gallery=self.scene == "gallery", dynamics=self.compile_dynamics))
        from .warp import configure_launch_order, install_contact_order
        configure_launch_order()
        sim_cfg = SimulationCfg(nconmax=512, njmax=1024)
        sim_cfg.mujoco.timestep = self.config.physics_dt
        new_sim = Simulation(num_envs=self.num_envs, cfg=sim_cfg, spec=new_spec, device="cuda:0")

        if self._viser_server is not None:
            self._viser_server.scene.reset()
        install_contact_order(new_sim)
        self.sim = new_sim
        self.spec = new_spec
        self.style = style
        self.board_style = new_style
        self._viewer_data = mujoco.MjData(self.sim.mj_model)
        self._body_id = self.sim.mj_model.body("deck").id
        self._board_truck_joints = [
            (int(self.sim.mj_model.joint(f"{prefix}_truck_tilt").qposadr[0]),
             int(self.sim.mj_model.joint(f"{prefix}_truck_tilt").dofadr[0]))
            for prefix in ("front", "rear")]
        self._board_deck_body_id = int(self.sim.mj_model.body("deck").id)
        self._nominal_qpos = self.sim.data.qpos.clone()
        self._nominal_qvel = self.sim.data.qvel.clone()
        self._state_snapshot = None
        self._force[:] = 0
        self._mouse_force[:] = 0
        self._pulse_steps = 0
        self.reset()

        if self._viser_server is not None:
            import mjviser
            def step_fn(model, data):
                # The built-in Step button calls this only while paused.
                if self._viser_viewer is not None and self._viser_viewer._paused and not getattr(self,"visual_replay",False):
                    self.step(update_policy=False)
            def render_fn(scene):
                self._sync_viewer()
                scene.update_from_mjdata(self._viewer_data)
            def reset_fn(model, data):
                if not getattr(self,"visual_replay",False): self.reset()
            self._viser_viewer = mjviser.Viewer(
                self.sim.mj_model, self._viewer_data,
                step_fn=step_fn, render_fn=render_fn, reset_fn=reset_fn,
                server=self._viser_server,
            )
            self._viser_viewer._setup_gui()
            self._viser_viewer._paused = self.paused
            if self._style_select is not None:
                self._style_select.value = style
            print(f"mjlab asset switched in-place: {old_style} -> {style}", flush=True)

    def _key_callback(self, key: int) -> None:
        if getattr(self,"visual_replay",False): return
        if key == self.KEY_SPACE:
            self.paused = not self.paused
        elif key == self.KEY_N and self.paused:
            self.step(update_policy=False)
        elif key == self.KEY_M and self.paused:
            for _ in range(self.config.substeps):
                self.step()
        elif key == self.KEY_R:
            self.reset()
        elif key == self.KEY_F:
            self.apply_wrench(self.sim.mj_model.body(self._body_id).name, (20.0, 0.0, 0.0))
            self._pulse_steps = round(0.2 / self.config.physics_dt)
            print(f"[bench] F received: cube +X 20 N for 0.2 sim seconds; paused={self.paused}", flush=True)
        elif key == self.KEY_E:
            self._force[:] = 0.0
            self._pulse_steps = 0
            self.apply_wrench(self.sim.mj_model.body(self._body_id).name, (0, 0, 0))
            print("[bench] E received: force released", flush=True)

    def close(self) -> None:
        if self._viser_server is not None:
            self._viser_server.stop()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def take_requested_style(self):
        style = self._requested_style
        self._requested_style = None
        return style
