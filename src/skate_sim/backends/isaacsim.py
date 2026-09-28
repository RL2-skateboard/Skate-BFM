from __future__ import annotations

from typing import Any

import numpy as np
from importlib.util import find_spec
from pathlib import Path

from .base import Backend
from ..config import SimConfig


class IsaacSimBackend(Backend):
    """Isaac Sim 5.1 native USD/PhysX bench backend.

    Isaac imports intentionally happen only after ``SimulationApp`` starts. The
    World and DynamicCuboid objects below own the physics state; this wrapper
    never mirrors a MuJoCo simulation.
    """

    name = "isaacsim"

    def __init__(self, *, seed: int = 0, config: SimConfig | None = None,
                 headless: bool = False):
        del seed
        from isaacsim import SimulationApp

        self.config = config or SimConfig()
        package_spec = find_spec("isaacsim")
        if package_spec is None or package_spec.origin is None:
            raise RuntimeError("Isaac Sim package path cannot be resolved")
        experience = Path(package_spec.origin).parent / "apps" / "isaacsim.exp.full.kit"
        self.app = SimulationApp({
            "headless": headless,
            "physics_dt": self.config.physics_dt,
            "rendering_dt": self.config.control_dt,
            "multi_gpu": False,
            "active_gpu": 0,
            "experience": str(experience),
            "extra_args": ["--/app/fileWatcher/enabled=false", "--/app/fileSystemWatcher/enabled=false"],
        })
        self._closed = False
        try:
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.core.api")
            self.app.update()
            from omni.isaac.core import World
            from omni.isaac.core.objects import DynamicCuboid
            from omni.isaac.core.objects.ground_plane import GroundPlane

            self.world = World(stage_units_in_meters=1.0,
                               physics_dt=self.config.physics_dt,
                               rendering_dt=self.config.control_dt)
            self.world.scene.add(GroundPlane("/World/Ground", size=10.0))
            self.cube = self.world.scene.add(
                DynamicCuboid("/World/Cube", name="cube", position=np.array([0.0, 0.0, 1.0]),
                              size=0.2, mass=1.0, color=np.array([0.12, 0.48, 0.85])))
            self.paused = False
            self._force = np.zeros(3, dtype=np.float32)
            self.world.reset()
            self.cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
            self.cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
            self._install_keys()
        except Exception:
            self.app.close()
            raise

    def _install_keys(self) -> None:
        try:
            import carb
            import omni.appwindow
            self._input = carb.input.acquire_input()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._key_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._key_callback)
        except Exception:
            self._input = self._keyboard = self._key_sub = None

    def _key_callback(self, event) -> bool:
        import carb
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input
        name = getattr(key, "name", str(key)).upper()
        if name.endswith("SPACE"):
            self.paused = not self.paused
        elif name.endswith("N"):
            self.step()
        elif name.endswith("M"):
            for _ in range(self.config.substeps):
                self.step()
        elif name.endswith("R"):
            self.reset()
        elif name.endswith("F"):
            self._force[:] = (1.0, 0.0, 0.0)
        elif name.endswith("E"):
            self._force[:] = 0.0
        elif name.endswith("ESCAPE"):
            self.close()
        return True

    def reset(self, seed: int = 0, env_ids: Any = None) -> None:
        del seed, env_ids
        self._force[:] = 0.0
        self.world.reset()
        self.cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.cube.set_angular_velocity(np.zeros(3, dtype=np.float32))

    def step(self, action: Any = None) -> None:
        del action
        if np.any(self._force):
            self.cube._rigid_prim_view.apply_forces(self._force.reshape(1, 3))
        self.world.step(render=False)

    def get_state(self) -> dict[str, Any]:
        state = self.cube.get_current_dynamic_state()
        return {"position": np.asarray(state.position).copy(), "orientation": np.asarray(state.orientation).copy(),
                "linear_velocity": np.asarray(state.linear_velocity).copy(),
                "angular_velocity": np.asarray(state.angular_velocity).copy(),
                "time": float(self.world.current_time)}

    def set_state(self, state: dict[str, Any]) -> None:
        self.cube.set_world_pose(position=state["position"], orientation=state["orientation"])
        self.cube.set_linear_velocity(state["linear_velocity"])
        self.cube.set_angular_velocity(state["angular_velocity"])
        self.world.step(render=False)

    def apply_wrench(self, body: str, force: tuple[float, float, float], point=None) -> None:
        if body != "cube":
            raise ValueError(f"unknown body {body!r}; available: cube")
        self._force[:] = force

    def diagnostics(self) -> dict[str, Any]:
        state = self.get_state()
        return {"backend": self.name, "version": "5.1.0", "sim_time": state["time"],
                "cube_pos": state["position"].tolist(),
                "cube_vel": state["linear_velocity"].tolist(),
                "force_world_N": self._force.tolist(), "contacts": "PhysX contact API pending",
                "device": "cuda:0"}

    def render(self) -> bool:
        if self._closed:
            return True
        self.app.update()
        return not self.app.is_running()

    def is_running(self) -> bool:
        return not self._closed and self.app.is_running()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._key_sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._key_sub)
            except Exception:
                pass
        self.world.clear()
        self.app.close()
