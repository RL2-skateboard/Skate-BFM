from __future__ import annotations

from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

from .base import Backend
from ..config import SimConfig
from ..models.board import board_styles, build_board_xml
from ..robot.bfm import (BFMOnnxPolicy, DEPLOY_JOINTS,
                         build_official_actor_input, deploy_joint_values,
                         load_policy_context)
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
                 randomize: bool = False, dynamics: dict | None = None, **_: Any):
        import torch
        from mjlab.sim import Simulation, SimulationCfg
        import warp as wp

        self.config = config or SimConfig()
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
        self.board_style = board_styles(style)[style] if scene in ("board", "gallery") else None
        self._torch = torch
        self._wp = wp
        scene_path = None
        if self.board_style:
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
        generated_scene = scene == "rider"
        try:
            self.spec = mujoco.MjSpec.from_file(str(scene_path)) if scene_path else mujoco.MjSpec.from_string(xml)
        finally:
            if generated_scene and scene_path is not None:
                Path(scene_path).unlink(missing_ok=True)
        sim_cfg = SimulationCfg()
        sim_cfg.mujoco.timestep = self.config.physics_dt
        sim_cfg.nconmax = 512
        sim_cfg.njmax = 1024
        self.sim = Simulation(num_envs=self.num_envs, cfg=sim_cfg, spec=self.spec, device="cuda:0")
        self._setup_robot_controller(checkpoint)
        self.viewer = None
        self.paused = inspect
        self._force = np.zeros(3, dtype=np.float64)
        self._mouse_force = np.zeros(3, dtype=np.float64)
        self.physics_step_counter = 0
        self.policy_counter = 0
        self.last_tau = None
        self.last_qfrc_applied = None
        self._viewer_data = mujoco.MjData(self.sim.mj_model)
        body_name = "deck" if self.board_style else ("pelvis" if scene in ("robot", "rider") else "foot" if scene == "contact" else "cube")
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

    @classmethod
    def for_asset(cls, scene: str, style: str, *, seed: int = 0, config: SimConfig | None = None):
        self = cls.__new__(cls)
        import torch
        import warp as wp
        from mjlab.sim import Simulation, SimulationCfg
        self.config=config or SimConfig(); self._torch=torch; self._wp=wp; self.scene=scene; self.style=style
        self.seed=int(seed); self.randomize=False; self.parameter_spec=None; self.parameter_source="nominal"
        self.base_dynamics={}; self.parameters=nominal_parameters(); self.nominal_parameters=nominal_parameters()
        self.parameter_batch=[dict(self.parameters)]
        self.dynamics=effective_dynamics({}, self.parameters); self.compile_dynamics=effective_dynamics({}, self.nominal_parameters)
        self.board_style=board_styles(style)[style]
        self.spec=mujoco.MjSpec.from_string(build_board_xml(self.board_style,gallery=scene=="gallery",dynamics=self.compile_dynamics))
        sim_cfg=SimulationCfg(nconmax=512, njmax=1024); sim_cfg.mujoco.timestep=self.config.physics_dt
        self.sim=Simulation(num_envs=1,cfg=sim_cfg,spec=self.spec,device="cuda:0")
        self.viewer=None; self.paused=True; self._force=np.zeros(3); self._mouse_force=np.zeros(3)
        self.physics_step_counter=0; self.policy_counter=0; self.last_tau=None; self.last_qfrc_applied=None
        self.controller="none"; self.preset="separate"; self.policy_task="reward"; self._state_snapshot=None
        self._pulse_steps=0; self._viewer_data=mujoco.MjData(self.sim.mj_model)
        self._body_id=self.sim.mj_model.body("deck").id
        self._parameter_metadata = parameter_metadata(self.parameters, source="nominal", seed=seed, randomize=False)
        self._board_wheel_dof = [int(self.sim.mj_model.joint(f"{prefix}_{side}_spin").dofadr[0])
                                 for prefix in ("front", "rear") for side in ("left", "right")]
        self._board_truck_joints = [
            (int(self.sim.mj_model.joint(f"{prefix}_truck_tilt").qposadr[0]),
             int(self.sim.mj_model.joint(f"{prefix}_truck_tilt").dofadr[0]))
            for prefix in ("front", "rear")]
        self._board_root_dof = int(self.sim.mj_model.joint("deck_free").dofadr[0])
        self._board_deck_body_id = int(self.sim.mj_model.body("deck").id)
        self._sync_viewer(); self._viser_server=None; self._viser_viewer=None; self._viser_thread=None
        self._requested_style = None
        self._style_select = None
        self.reset(seed)
        return self

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
        self._viewer_data.xfrc_applied[:] = 0
        self._viewer_data.xfrc_applied[self._body_id, :3] = self._force + self._mouse_force
        mujoco.mj_forward(self.sim.mj_model, self._viewer_data)

    def _setup_robot_controller(self, checkpoint: str | None) -> None:
        """Compile the native asset contract used by both mjlab controllers."""
        if self.scene not in ("robot", "rider"):
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
        self.robot_history = [HistoryState() for _ in range(self.num_envs)]
        self.policy_counter = 0
        self.physics_step_counter = 0
        self.policy_context_index = 0
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
        if self.scene == "rider":
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
        tensor.copy_(source)
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
        angvel = self.sim.data.cvel._tensor[:, self.pelvis_body_id, :3]
        return (qpos[:, self.robot_qpos_index].detach().cpu().numpy(),
                qvel[:, self.robot_dof_index].detach().cpu().numpy(),
                quat.detach().cpu().numpy(), angvel.detach().cpu().numpy())

    def _update_controller(self) -> None:
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
        if self.scene not in ("robot", "rider") or self.controller == "none":
            self._write_field("qfrc_applied", qfrc)
            return
        q, dq, quat, angvel = self._robot_state_cpu()
        kp_scale = np.asarray([row["pd_kp_scale"] for row in self.parameter_batch])[:, None]
        kd_scale = np.asarray([row["pd_kd_scale"] for row in self.parameter_batch])[:, None]
        if self.controller == "policy" and self.physics_step_counter % self.config.substeps == 0:
            for env_id in range(self.num_envs):
                context = self.policy_z[self.policy_context_index] if self.policy_task == "tracking" else self.policy_z
                obs = build_official_actor_input(
                    q[env_id], dq[env_id], quat[env_id], angvel[env_id],
                    self.policy_action[env_id], self.robot_history[env_id], context)
                noise_std = self.parameter_batch[env_id]["observation_noise_std"]
                if noise_std > 0:
                    obs = obs + np.random.default_rng(self.seed + self.physics_step_counter + env_id).normal(
                        0.0, noise_std, size=obs.shape).astype(np.float32)
                raw = np.asarray(self.policy(obs), dtype=np.float64)
                if raw.shape != (29,) or not np.isfinite(raw).all():
                    raise RuntimeError(f"BFM action contract failed: shape={raw.shape} finite={np.isfinite(raw).all()}")
                self.last_obs[env_id] = obs
                self.raw_action[env_id] = raw
                candidate_action = np.clip(raw, -1.0, 1.0)
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
                self.policy_context_index = (self.policy_context_index + 1) % len(self.policy_z)
            self.policy_counter += 1
        elif self.controller == "pose":
            self.q_target[:] = self.pose_target
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

    def reset(self, seed: int | None = None, env_ids: Any = None) -> None:
        ids = None if env_ids is None else np.asarray(env_ids, dtype=np.int64).reshape(-1)
        self._resample_parameters(ids, seed)
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
        if self.scene in ("robot", "rider"):
            reset_ids = range(self.num_envs) if ids is None else ids.tolist()
            for env_id in reset_ids:
                self.robot_history[env_id].reset()
                self.policy_action[env_id] = 0
                self.raw_action[env_id] = 0
                self._action_queues[env_id].clear()
                self.last_obs[env_id] = 0
                self.q_target[env_id] = self.policy_values["default"]
                self.pose_target[env_id] = self.policy_values["default"]
            if ids is None:
                self.policy_counter = 0
                self.policy_context_index = 0
                self.physics_step_counter = 0
        self._pulse_steps = 0
        self._sync_viewer()
        self._capture_state_snapshot()

    def step(self, action: Any = None) -> None:
        del action
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.sync()
        self._mouse_force[:] = self._viewer_data.xfrc_applied[self._body_id, :3]
        total_force = self._mouse_force + self._force
        xfrc = self.sim.data.xfrc_applied._tensor.clone()
        xfrc[self.view_env, self._body_id, :3] = self._torch.as_tensor(total_force, device=xfrc.device, dtype=xfrc.dtype)
        self._write_field("xfrc_applied", xfrc)
        self._update_controller()
        self.sim.step()
        self.physics_step_counter += 1
        if self._pulse_steps > 0:
            self._pulse_steps -= 1
            if self._pulse_steps == 0:
                self._force[:] = 0
                self._write_field("xfrc_applied", self._torch.zeros_like(self.sim.data.xfrc_applied._tensor))
                print("[bench] pulse finished; force=0 N", flush=True)
        self._sync_viewer()
        self._capture_state_snapshot()

    def get_state(self) -> dict[str, Any]:
        state = self._read_state_snapshot()
        return {"qpos": state["qpos"], "qvel": state["qvel"],
                "time": float(state["time"][self.view_env])}

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
            "termination": False,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._write_field("qpos", self._torch.as_tensor(state["qpos"], device=self.sim.data.qpos.device))
        self._write_field("qvel", self._torch.as_tensor(state["qvel"], device=self.sim.data.qvel.device))
        self.sim.forward()
        self._sync_viewer()

    def apply_wrench(self, body: str, force: tuple[float, float, float], point=None) -> None:
        expected = "deck" if self.board_style else ("pelvis" if self.scene in ("robot", "rider") else "cube")
        if body != expected:
            raise ValueError(f"unknown body {body!r}; available: {expected}")
        self._force[:] = force
        self._pulse_steps = 0

    def diagnostics(self) -> dict[str, Any]:
        body_name = "deck" if self.board_style else ("pelvis" if self.scene in ("robot", "rider") else "foot" if self.scene == "contact" else "cube")
        result = {"backend": self.name, "version": mujoco.__version__, "object": body_name,
                "sim_time": float(self.sim.data.time[self.view_env].cpu().numpy()),
                "num_envs": self.num_envs, "view_env": self.view_env,
                "cube_pos": self._viewer_data.xpos[1].tolist(),
                "cube_vel": self._viewer_data.cvel[1, 3:].tolist(),
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
        if self.scene in ("robot", "rider") and self.last_tau is not None:
            env = self.view_env
            result.update({
                "obs_shape": [721] if self.controller == "policy" else None,
                "raw_action_norm": float(np.linalg.norm(self.raw_action[env])),
                "qfrc_applied_norm": float(np.linalg.norm(self.last_qfrc_applied[env])),
                "torque_injection_error": 0.0,
                "q_change_after_mj_step": 0.0,
            })
        return result

    def render(self) -> bool:
        if self.viewer is None:
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
            @style_select.on_update
            def _(_):
                if style_select.value != self.style:
                    self._requested_style = style_select.value
            fx = self._viser_server.gui.add_slider("Force X (N)", min=-30, max=30, step=.5, initial_value=0)
            fy = self._viser_server.gui.add_slider("Force Y (N)", min=-30, max=30, step=.5, initial_value=0)
            fz = self._viser_server.gui.add_slider("Force Z (N)", min=-30, max=30, step=.5, initial_value=0)
            release = self._viser_server.gui.add_button("Release Force")
            @fx.on_update
            def _(_): self._force[:] = (fx.value, fy.value, fz.value)
            @fy.on_update
            def _(_): self._force[:] = (fx.value, fy.value, fz.value)
            @fz.on_update
            def _(_): self._force[:] = (fx.value, fy.value, fz.value)
            @release.on_click
            def _(_):
                fx.value = fy.value = fz.value = 0
                self._force[:] = 0
            def step_fn(model,data):
                self.step()
            def render_fn(scene):
                self._sync_viewer()
                scene.update_from_mjdata(self._viewer_data)
            def reset_fn(model,data):
                self.reset()
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
            self._viser_viewer._is_paused = self.paused
            print(f"mjlab Viser viewer: http://localhost:{self._viser_server.get_port()}",flush=True)
        self._sync_viewer()
        if self._viser_viewer:
            self._viser_viewer._tick()
        return False

    def switch_style(self, style: str) -> None:
        """Replace the active board simulation while keeping the Viser server alive."""
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
        sim_cfg = SimulationCfg(nconmax=512, njmax=1024)
        sim_cfg.mujoco.timestep = self.config.physics_dt
        new_sim = Simulation(num_envs=self.num_envs, cfg=sim_cfg, spec=new_spec, device="cuda:0")

        if self._viser_server is not None:
            self._viser_server.scene.reset()
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
                self.step()
            def render_fn(scene):
                self._sync_viewer()
                scene.update_from_mjdata(self._viewer_data)
            def reset_fn(model, data):
                self.reset()
            self._viser_viewer = mjviser.Viewer(
                self.sim.mj_model, self._viewer_data,
                step_fn=step_fn, render_fn=render_fn, reset_fn=reset_fn,
                server=self._viser_server,
            )
            self._viser_viewer._setup_gui()
            self._viser_viewer._is_paused = self.paused
            if self._style_select is not None:
                self._style_select.value = style
            print(f"mjlab asset switched in-place: {old_style} -> {style}", flush=True)

    def _key_callback(self, key: int) -> None:
        if key == self.KEY_SPACE:
            self.paused = not self.paused
        elif key == self.KEY_N:
            self.step()
        elif key == self.KEY_M:
            for _ in range(self.config.substeps):
                self.step()
        elif key == self.KEY_R:
            self.reset()
        elif key == self.KEY_F:
            self.apply_wrench("deck" if self.board_style else "cube", (20.0, 0.0, 0.0))
            self._pulse_steps = round(0.2 / self.config.physics_dt)
            print(f"[bench] F received: cube +X 20 N for 0.2 sim seconds; paused={self.paused}", flush=True)
        elif key == self.KEY_E:
            self._force[:] = 0.0
            self._pulse_steps = 0
            self._write_field("xfrc_applied", self._torch.zeros_like(self.sim.data.xfrc_applied._tensor))
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
