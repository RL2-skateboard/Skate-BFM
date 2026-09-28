from __future__ import annotations

from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from .base import Backend
from ..config import SimConfig
from ..models.board import board_styles, build_board_xml


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
                 scene: str = "bench", style: str = "standard", inspect: bool = False):
        import torch
        from mjlab.sim import Simulation, SimulationCfg

        self.config = config or SimConfig()
        self.scene = scene
        self.style = style
        self.board_style = board_styles(style)[style] if scene in ("board", "gallery") else None
        self._torch = torch
        xml = build_board_xml(self.board_style, gallery=scene == "gallery") if self.board_style else self._xml()
        self.spec = mujoco.MjSpec.from_string(xml)
        sim_cfg = SimulationCfg()
        sim_cfg.mujoco.timestep = self.config.physics_dt
        sim_cfg.nconmax = 256
        self.sim = Simulation(num_envs=1, cfg=sim_cfg, spec=self.spec, device="cuda:0")
        self.viewer = None
        self.paused = inspect
        self._force = np.zeros(3, dtype=np.float64)
        self._mouse_force = np.zeros(3, dtype=np.float64)
        self._viewer_data = mujoco.MjData(self.sim.mj_model)
        body_name = "deck" if self.board_style else "cube"
        self._body_id = self.sim.mj_model.body(body_name).id
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
        from mjlab.sim import Simulation, SimulationCfg
        self.config=config or SimConfig(); self._torch=torch; self.scene=scene; self.style=style
        self.board_style=board_styles(style)[style]
        self.spec=mujoco.MjSpec.from_string(build_board_xml(self.board_style,gallery=scene=="gallery"))
        sim_cfg=SimulationCfg(nconmax=256); sim_cfg.mujoco.timestep=self.config.physics_dt
        self.sim=Simulation(num_envs=1,cfg=sim_cfg,spec=self.spec,device="cuda:0")
        self.viewer=None; self.paused=True; self._force=np.zeros(3); self._mouse_force=np.zeros(3)
        self._pulse_steps=0; self._viewer_data=mujoco.MjData(self.sim.mj_model)
        self._body_id=self.sim.mj_model.body("deck").id
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
        qpos = self.sim.data.qpos.cpu().numpy()
        qvel = self.sim.data.qvel.cpu().numpy()
        self._viewer_data.qpos[:] = qpos
        self._viewer_data.qvel[:] = qvel
        self._viewer_data.xfrc_applied[:] = 0
        self._viewer_data.xfrc_applied[self._body_id, :3] = self._force + self._mouse_force
        mujoco.mj_forward(self.sim.mj_model, self._viewer_data)

    def reset(self, seed: int = 0, env_ids: Any = None) -> None:
        del seed
        ids = None if env_ids is None else self._torch.as_tensor(env_ids, dtype=self._torch.int64)
        self.sim.reset(ids)
        self._force[:] = 0
        self._mouse_force[:] = 0
        self._pulse_steps = 0
        self._sync_viewer()

    def step(self, action: Any = None) -> None:
        del action
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.sync()
        self._mouse_force[:] = self._viewer_data.xfrc_applied[self._body_id, :3]
        total_force = self._mouse_force + self._force
        self.sim.data.xfrc_applied[0, 1, :3] = self._torch.as_tensor(total_force, device=self.sim.data.xfrc_applied.device)
        self.sim.step()
        if self._pulse_steps > 0:
            self._pulse_steps -= 1
            if self._pulse_steps == 0:
                self._force[:] = 0
                self.sim.data.xfrc_applied[:] = 0
                print("[bench] pulse finished; force=0 N", flush=True)
        self._sync_viewer()

    def get_state(self) -> dict[str, Any]:
        return {"qpos": self.sim.data.qpos.cpu().numpy().copy(),
                "qvel": self.sim.data.qvel.cpu().numpy().copy(),
                "time": float(self.sim.data.time.cpu().numpy()[0])}

    def set_state(self, state: dict[str, Any]) -> None:
        self.sim.data.qpos[:] = self._torch.as_tensor(state["qpos"])
        self.sim.data.qvel[:] = self._torch.as_tensor(state["qvel"])
        self.sim.forward()
        self._sync_viewer()

    def apply_wrench(self, body: str, force: tuple[float, float, float], point=None) -> None:
        expected = "deck" if self.board_style else "cube"
        if body != expected:
            raise ValueError(f"unknown body {body!r}; available: {expected}")
        self._force[:] = force
        self._pulse_steps = 0

    def diagnostics(self) -> dict[str, Any]:
        body_name = "deck" if self.board_style else "cube"
        return {"backend": self.name, "version": mujoco.__version__, "object": body_name,
                "sim_time": float(self.sim.data.time.cpu().numpy()[0]),
                "cube_pos": self._viewer_data.xpos[1].tolist(),
                "cube_vel": self._viewer_data.cvel[1, 3:].tolist(),
                "force_world_N": (self._mouse_force + self._force).tolist(),
                "keyboard_force_N": self._force.tolist(), "mouse_force_N": self._mouse_force.tolist(),
                "mouse_body": body_name, "contacts": int(self._viewer_data.ncon),
                "device": str(self.sim.device)}

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
        new_spec = mujoco.MjSpec.from_string(build_board_xml(new_style, gallery=self.scene == "gallery"))
        sim_cfg = SimulationCfg(nconmax=256)
        sim_cfg.mujoco.timestep = self.config.physics_dt
        new_sim = Simulation(num_envs=1, cfg=sim_cfg, spec=new_spec, device="cuda:0")

        if self._viser_server is not None:
            self._viser_server.scene.reset()
        self.sim = new_sim
        self.spec = new_spec
        self.style = style
        self.board_style = new_style
        self._viewer_data = mujoco.MjData(self.sim.mj_model)
        self._body_id = self.sim.mj_model.body("deck").id
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
            self.sim.data.xfrc_applied[:] = 0
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
