from __future__ import annotations

from typing import Any
from queue import SimpleQueue

import mujoco
import numpy as np
import mujoco.viewer

from .base import Backend
from ..config import SimConfig
from ..models.board import BoardStyle, board_styles, build_board_xml, board_hash


class MujocoBackend(Backend):
    name = "mujoco"
    KEY_SPACE, KEY_N, KEY_M, KEY_R = 32, 78, 77, 82
    KEY_F, KEY_E, KEY_ESC = 70, 69, 256

    def __init__(self, *, seed: int = 0, config: SimConfig | None = None,
                 scene: str = "bench", style: str = "standard", inspect: bool = False,
                 dynamics: dict | None = None, initial_roll: float = 0.0,
                 experiment: str | None = None):
        self.config = config or SimConfig()
        self.scene = scene
        self.style = style
        self.inspect = inspect
        self.experiment = experiment
        self.initial_roll = initial_roll
        self.dynamics = dynamics or {}
        self.events = SimpleQueue()
        if scene in ("board", "gallery"):
            self.board_style = board_styles(style)[style]
            xml = build_board_xml(self.board_style, gallery=scene == "gallery", dynamics=dynamics, initial_roll=initial_roll)
        else:
            self.board_style = None
            xml = self._xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = self.config.physics_dt
        self.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        if experiment:
            for prefix in ("front", "rear"):
                for side in ("left", "right"):
                    j = self.model.joint(f"{prefix}_{side}_spin").id
                    self.model.dof_damping[self.model.jnt_dofadr[j]] = self.dynamics["wheel_spin_damping"]
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.paused = inspect
        self.closed = False
        self.force = np.zeros(3, dtype=np.float64)
        self.push_force = 20.
        self.push_duration = .2
        self.mouse_force = np.zeros(3, dtype=np.float64)
        self._body_id = self.model.body("cube").id if scene == "bench" else self.model.body("deck").id
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
        if self.board_style is not None:
            self.data.qpos[3:7] = (np.cos(self.initial_roll/2), np.sin(self.initial_roll/2), 0, 0)
        self.external_work = 0.0
        self.fixture = self.experiment in ('lean','turn')
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

    def step(self, action: Any = None) -> None:
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
        if self.force_point is not None:
            self.data.xfrc_applied[self._body_id, 3:] += np.cross(
                self.force_point-self.data.xipos[self._body_id], self.force)
        if self.fixture:
            rotation = self.data.xmat[self._body_id].reshape(3,3)
            roll = np.arctan2(rotation[2,1],rotation[2,2])
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self._body_id,velocity,0)
            self.data.xfrc_applied[self._body_id,3] += self.dynamics['fixture_k']*(self.dynamics['lean_target']-roll)-self.dynamics['fixture_c']*velocity[0]
        self.data.qfrc_applied[:] = 0
        if self.experiment:
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
        expected = "cube" if self.scene == "bench" else "deck"
        if body != expected:
            raise ValueError(f"unknown body {body!r}; available: {expected}")
        self.force[:] = force
        self.force_point = None if point is None else np.asarray(point, dtype=float).copy()
        self._pulse_steps = 0

    def diagnostics(self) -> dict[str, Any]:
        body_name = "cube" if self.board_style is None else "deck"
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
        elif key == self.KEY_F:
            body = "cube" if self.board_style is None else "deck"
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
