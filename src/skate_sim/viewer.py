"""One scene editor and one physics owner for both simulation backends."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import numpy as np

from .backends import create_backend
from .models.scene import build_scene_xml, load_scene_config, normalize_scene_config, save_scene_config
from .record import Recorder, checkpoint_manifest, recording_inputs, compiled_model_xml, numerics_manifest


class SceneWorkbench:
    """UI callbacks enqueue requests; only ``tick`` touches simulation state."""

    def __init__(self, backend_name, backend_kwargs, *, record_path=None, running=False):
        import viser
        self.backend_name = backend_name
        self.kwargs = dict(backend_kwargs)
        self.config = normalize_scene_config(self.kwargs["scene_config"])
        self.pending = copy.deepcopy(self.config)
        self.pending_controller = self.kwargs.get("controller", "none")
        self.backend = create_backend(backend_name, **self.kwargs)
        self._model_xml = None
        self.backend.paused = not running
        self.requests = SimpleQueue()
        self.recorder = None
        self.message = "拖动只预览；点击“应用场景”后碰撞体才会更新。"
        self.selected = "无"
        self.gizmo = None
        self.preview = None
        self.obstacle_previews = {}
        self._updating = False
        self._status_time = 0.
        self.server = None
        try:
            self._model_xml = self._capture_model_xml()
            self.server = viser.ViserServer(label="skate-sim", host="127.0.0.1", port=0)
            self._make_gui(record_path)
            self._bind_scene()
            if record_path:
                self.start_recording(record_path)
        except Exception:
            if self.recorder is not None:
                self.recorder.close()
            try:
                self.backend.close()
            finally:
                if self.server is not None:
                    self.server.stop()
            raise
        print(f"skate-sim | {backend_name} | http://localhost:{self.server.get_port()}", flush=True)

    def _capture_model_xml(self):
        return compiled_model_xml(self.backend)

    @property
    def model(self):
        return self.backend.model if self.backend_name == "mujoco" else self.backend.sim.mj_model

    def enqueue(self, operation, value=None):
        self.requests.put((operation, copy.deepcopy(value)))

    def _make_gui(self, record_path):
        gui = self.server.gui
        self.status = gui.add_markdown("")
        self.gripper_sliders = {}
        if getattr(self.backend, "gripper_command", None):
            with gui.add_folder("Robotiq2F-85（独立于29DoF动作）"):
                for side, command in self.backend.gripper_command.items():
                    slider = gui.add_slider(f"{side} raw command (0=open,255=close)", min=0., max=255., step=1., initial_value=command)
                    self.gripper_sliders[side] = slider
                    slider.on_update(lambda event, side=side: self.enqueue("gripper", (side, event.target.value)) if not self._updating else None)
        self.play_button = gui.add_button("继续 / 暂停")
        self.play_button.on_click(lambda _: self.enqueue("pause"))
        gui.add_button("单个物理步").on_click(lambda _: self.enqueue("step"))
        gui.add_button("单个控制步").on_click(lambda _: self.enqueue("control_step"))
        gui.add_button("重置").on_click(lambda _: self.enqueue("reset"))
        with gui.add_folder("场景资产"):
            self.robot_box = gui.add_checkbox("机器人", initial_value=self.pending["robot"])
            self.board_box = gui.add_checkbox("滑板", initial_value=self.pending["board"])
            self.style_box = gui.add_dropdown("滑板样式", options=("standard", "cruiser", "longboard", "street", "downhill"), initial_value=self.pending["style"])
            for handle, key in ((self.robot_box, "robot"), (self.board_box, "board"), (self.style_box, "style")):
                handle.on_update(lambda event, key=key: self.enqueue("field", (key, event.target.value)) if not self._updating else None)
            self.kind_box = gui.add_dropdown("障碍类型", options=("ramp", "platform", "rail"))
            self.add_button = gui.add_button("添加障碍")
            self.add_button.on_click(lambda _: self.enqueue("add", self.kind_box.value))
            self.selection = gui.add_dropdown("选择摆放对象", options=self._selections(), initial_value="无")
            self.selection.on_update(lambda event: self.enqueue("select", event.target.value) if not self._updating else None)
            self.remove_button = gui.add_button("移除选中障碍")
            self.remove_button.on_click(lambda _: self.enqueue("remove"))
            self.apply_button = gui.add_button("应用场景（暂停并重置）")
            self.apply_button.on_click(lambda _: self.enqueue("apply"))
            options = ("none", "pose", "policy") if self.kwargs.get("checkpoint") else ("none", "pose")
            self.controller_box = gui.add_dropdown("机器人控制", options=options, initial_value=self.pending_controller)
            self.controller_box.on_update(lambda event: self.enqueue("controller", event.target.value) if not self._updating else None)
        with gui.add_folder("场景预设"):
            self.path_box = gui.add_text("场景文件", initial_value="runs/scenes/scene.json")
            self.save_button = gui.add_button("保存摆放预设")
            self.save_button.on_click(lambda _: self.enqueue("save", self.path_box.value))
            self.load_button = gui.add_button("载入预设（待应用）")
            self.load_button.on_click(lambda _: self.enqueue("load", self.path_box.value))
            self.preset_box = gui.add_dropdown("初始姿态", options=("separate", "push_start", "steer_start"), initial_value=self.pending["preset"])
            self.preset_box.on_update(lambda event: self.enqueue("field", ("preset", event.target.value)) if not self._updating else None)
        with gui.add_folder("记录与施力"):
            self.record_box = gui.add_text("记录目录", initial_value=record_path or self._new_record_path())
            gui.add_button("开始记录").on_click(lambda _: self.enqueue("record", self.record_box.value))
            gui.add_button("停止记录").on_click(lambda _: self.enqueue("stop_record"))
            self.force_box = gui.add_vector3("世界坐标力 (N)", initial_value=(20., 0., 0.), step=1.)
            gui.add_button("施力 0.2 秒").on_click(lambda _: self.enqueue("force", self.force_box.value))
            gui.add_button("释放外力").on_click(lambda _: self.enqueue("release"))
            gui.add_button("关闭").on_click(lambda _: self.enqueue("close"))
        self.edit_handles = (self.robot_box, self.board_box, self.style_box, self.kind_box,
                             self.add_button, self.selection, self.remove_button,
                             self.apply_button, self.load_button, self.preset_box, self.controller_box)
        self.closed = False

    @staticmethod
    def _new_record_path():
        return "runs/scene_" + time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"

    def _selections(self):
        return ["无"] + (["机器人"] if self.pending["robot"] else []) + (["滑板"] if self.pending["board"] else []) + [o["id"] for o in self.pending["obstacles"]]

    def _bind_scene(self):
        from mjviser import ViserMujocoScene
        self.server.scene.reset()
        self._display_model = copy.copy(self.model)
        self.renderer = ViserMujocoScene(self.server, self._display_model, num_envs=1)
        self.renderer.set_refresh_handler(lambda: self.enqueue("refresh"))
        self.gizmo = self.preview = None
        self.obstacle_previews = {}
        self._refresh_gui()
        self.render()

    def _refresh_gui(self):
        self._updating = True
        try:
            self.robot_box.value = self.pending["robot"]
            self.board_box.value = self.pending["board"]
            self.style_box.value = self.pending["style"]
            self.preset_box.value = self.pending["preset"]
            self.controller_box.value = self.pending_controller
            for side, slider in self.gripper_sliders.items():
                slider.visible = side in self.backend.gripper_command
                if slider.visible:
                    slider.value = self.backend.gripper_command[side]
            options = self._selections()
            if self.selected not in options:
                self.selected = "无"
            self.selection.options = options
            self.selection.value = self.selected
            for handle in self.edit_handles:
                handle.disabled = self.recorder is not None
            self._show_gizmo()
        finally:
            self._updating = False

    def _selected_pose(self):
        if self.selected in ("机器人", "滑板"):
            key, joint = ("robot_pose", "floating_base_joint") if self.selected == "机器人" else ("board_pose", "deck_free")
            pose = self.pending[key]
            if pose is None:
                try:
                    address = int(self.model.joint(joint).qposadr[0])
                    q = self.backend.get_state()["qpos"]
                    if np.asarray(q).ndim == 2:
                        q = q[self.backend.view_env]
                    pose = {"pos": list(q[address:address+3]), "quat": list(q[address+3:address+7])}
                except KeyError:
                    pose = {"pos": [0., -1., .79] if key == "robot_pose" else [0., 0., .1], "quat": [1., 0., 0., 0.]}
            return pose
        return next((o for o in self.pending["obstacles"] if o["id"] == self.selected), None)

    @staticmethod
    def _obstacle_meshes(obstacles):
        """Use the compiled physics geometry, in each body's local frame."""
        import mujoco
        from mjviser.conversions import merge_geoms
        model = mujoco.MjModel.from_xml_string(build_scene_xml(
            dict(robot=False, board=False, obstacles=obstacles)))
        return {o["id"]: merge_geoms(model, np.flatnonzero(
            model.geom_bodyid == model.body("obstacle_" + o["id"]).id).tolist())
            for o in obstacles}

    @staticmethod
    def _same_pose(pose, previous):
        if pose is None or previous is None:
            return pose is previous
        q, old_q = np.asarray(pose["quat"]), np.asarray(previous["quat"])
        return (np.allclose(pose["pos"], previous["pos"], rtol=0., atol=1e-12)
                and min(np.linalg.norm(q - old_q), np.linalg.norm(q + old_q)) < 1e-12)

    def _scene_dirty(self):
        pending, applied = normalize_scene_config(self.pending), normalize_scene_config(self.config)
        for key in ("robot_pose", "board_pose"):
            if not self._same_pose(pending.pop(key), applied.pop(key)):
                return True
        obstacles, old_obstacles = pending.pop("obstacles"), applied.pop("obstacles")
        if pending != applied or len(obstacles) != len(old_obstacles):
            return True
        for obstacle, old in zip(obstacles, old_obstacles):
            if (obstacle["id"] != old["id"] or obstacle["kind"] != old["kind"]
                    or not self._same_pose(obstacle, old)):
                return True
        return self.config["robot"] and self.pending_controller != self.kwargs.get("controller", "none")

    def _show_obstacle_previews(self):
        applied = {o["id"]: o for o in normalize_scene_config(self.config)["obstacles"]}
        changed = [o for o in normalize_scene_config(self.pending)["obstacles"]
                   if (o["id"] not in applied or o["kind"] != applied[o["id"]]["kind"]
                       or not self._same_pose(o, applied[o["id"]]))]
        if self.recorder:
            changed = []
        kinds = getattr(self, "_preview_kinds", {})
        cache = getattr(self, "_preview_mesh_cache", {})
        for obstacle in changed:
            kind = obstacle["kind"]
            if kind not in cache:
                canonical = dict(id="preview", kind=kind, pos=[0., 0., 0.], quat=[1., 0., 0., 0.])
                cache[kind] = self._obstacle_meshes([canonical])["preview"]
        self._preview_mesh_cache = cache
        keep = {o["id"]: o for o in changed}
        for ident in list(self.obstacle_previews):
            if ident not in keep or kinds.get(ident) != keep[ident]["kind"]:
                self.obstacle_previews.pop(ident).remove()
                kinds.pop(ident, None)
        for obstacle in changed:
            ident = obstacle["id"]
            selected = ident == self.selected
            color = (255, 190, 65) if selected else (80, 190, 255)
            opacity = .55 if selected else .35
            if ident not in self.obstacle_previews:
                mesh = cache[obstacle["kind"]]
                self.obstacle_previews[ident] = self.server.scene.add_mesh_simple(
                    "/pending_obstacles/" + ident, mesh.vertices, mesh.faces,
                    position=obstacle["pos"], wxyz=obstacle["quat"],
                    color=color, opacity=opacity, side="double",
                    cast_shadow=False, receive_shadow=False)
                kinds[ident] = obstacle["kind"]
            else:
                handle = self.obstacle_previews[ident]
                handle.position, handle.wxyz = obstacle["pos"], obstacle["quat"]
                handle.color, handle.opacity = color, opacity
        self._preview_kinds = kinds

    def _new_obstacle_position(self, obstacle):
        # Conservative world-space spheres cover arbitrary existing orientations.
        obstacles = self.pending["obstacles"] + [obstacle]
        meshes = self._obstacle_meshes(obstacles)
        radii = {ident: float(np.linalg.norm(mesh.vertices, axis=1).max())
                 for ident, mesh in meshes.items()}
        x = max(1.3, radii[obstacle["id"]] + .65)
        while any(np.linalg.norm(np.array([x, 0.]) - np.asarray(o["pos"][:2]))
                  < radii[obstacle["id"]] + radii[o["id"]] + .2
                  for o in self.pending["obstacles"]):
            x += .25
        return [x, 0., 0.]

    def _show_gizmo(self):
        self._show_obstacle_previews()
        for handle in (self.gizmo, self.preview):
            if handle is not None:
                handle.remove()
        self.gizmo = self.preview = None
        pose = self._selected_pose()
        if pose is None or self.recorder:
            return
        self.gizmo = self.server.scene.add_transform_controls("/placement", position=pose["pos"], wxyz=pose["quat"], scale=.4)
        self.gizmo.on_update(lambda event, target=self.selected: self.enqueue("move", (target, list(event.target.position), list(event.target.wxyz))))
        if self.selected not in ("机器人", "滑板"):
            return
        self.preview = self.server.scene.add_box("/placement_preview", dimensions=(.2, .2, .2), opacity=.25,
                                                 position=pose["pos"], wxyz=pose["quat"], wireframe=True)

    def apply_scene(self):
        if self.recorder:
            raise ValueError("请先停止记录，再编辑场景。")
        self.backend.paused = True
        config = normalize_scene_config(self.pending)
        kwargs = dict(self.kwargs, scene_config=config, controller=self.pending_controller)
        if not config["robot"]:
            kwargs.update(controller="none")
        replacement = create_backend(self.backend_name, **kwargs)
        replacement.paused = True
        old = self.backend
        old_kwargs, old_config, old_pending, old_xml = self.kwargs, self.config, self.pending, self._model_xml
        self.backend = replacement
        self.kwargs, self.config, self.pending = kwargs, config, copy.deepcopy(config)
        try:
            self._model_xml = self._capture_model_xml()
            self._bind_scene()
        except Exception:
            self.backend = old
            self._model_xml = old_xml
            self.kwargs, self.config, self.pending = old_kwargs, old_config, old_pending
            replacement.close()
            self._bind_scene()
            raise
        old.close()
        self.message = "场景已应用；初态、控制历史和外力已重置。"

    def start_recording(self, path):
        if self.recorder:
            raise ValueError("当前正在记录。")
        if self._scene_dirty():
            raise ValueError("请先应用场景，再开始记录。")
        b = self.backend
        inputs = recording_inputs(self.kwargs.get("checkpoint"), self.kwargs.get("policy_task", "reward"))
        recorder = Recorder(path, backend=self.backend_name, scene=b.scene, seed=b.seed,
                            metadata={"physics_dt": b.config.physics_dt, "control_dt": b.config.control_dt,
                                      "checkpoint": inputs, "scene_config": self.config,
                                      "gripper_attachment": getattr(b, "gripper_metadata", None),
                                      "numerics": numerics_manifest(b)})
        try:
            recorder.start(b.get_state(), backend_kwargs=dict(self.kwargs, config=asdict(b.config)),
                           runtime_state=b.export_replay_state())
            import mujoco
            model_path = Path(path).resolve() / "model.xml"
            model_path.write_text(self._model_xml, encoding="utf-8")
            recorder.write("asset_model", files=checkpoint_manifest(str(model_path)))
        except Exception:
            recorder.close()
            raise
        self.recorder = b.recorder = recorder
        self.record_box.value = str(path)
        self._refresh_gui()
        self.message = "正在记录，场景编辑已锁定。"

    def stop_recording(self):
        if self.recorder:
            self.recorder.close()
            self.recorder = None
            self.backend.recorder = None
            self.record_box.value = self._new_record_path()
            self._refresh_gui()
            self.message = "记录已保存，可以编辑场景。"

    def _process(self, operation, value):
        if operation == "gripper":
            self.backend.set_gripper_command(value[1], side=value[0])
            return
        if self.recorder and operation in ("field", "controller", "add", "remove", "move", "load", "apply"):
            raise ValueError("请先停止记录，再编辑场景。")
        if operation == "controller":
            self.pending_controller = value
            self.backend.paused = True
        elif operation == "field":
            key, item = value
            self.pending[key] = item
            if key == "preset":
                self.pending["robot_pose"] = self.pending["board_pose"] = None
            self.backend.paused = True
            self._refresh_gui()
        elif operation == "add":
            index = 1
            ids = {o["id"] for o in self.pending["obstacles"]}
            while f"item_{index}" in ids:
                index += 1
            self.selected = f"item_{index}"
            obstacle = dict(id=self.selected, kind=value, pos=[0., 0., 0.], quat=[1., 0., 0., 0.])
            obstacle["pos"] = self._new_obstacle_position(obstacle)
            self.pending["obstacles"].append(obstacle)
            self.message = "半透明障碍为待应用预览（无碰撞）；原场景碰撞保持到点击应用。"
            self.backend.paused = True
            self._refresh_gui()
        elif operation == "select":
            self.selected = value
            self._show_gizmo()
        elif operation == "move":
            ident, pos, quat = value
            if ident in ("机器人", "滑板"):
                self.pending["robot_pose" if ident == "机器人" else "board_pose"] = dict(pos=pos, quat=quat)
            else:
                target = next((o for o in self.pending["obstacles"] if o["id"] == ident), None)
                if target is None:
                    return
                target.update(pos=pos, quat=quat)
            self.backend.paused = True
            if self.preview is not None:
                self.preview.position, self.preview.wxyz = pos, quat
            self._show_obstacle_previews()
        elif operation == "remove":
            self.pending["obstacles"] = [o for o in self.pending["obstacles"] if o["id"] != self.selected]
            self.selected = "无"
            self.backend.paused = True
            self._refresh_gui()
        elif operation == "apply":
            self.apply_scene()
        elif operation == "save":
            save_scene_config(self.pending, value)
            self.message = "摆放预设已保存（不包含实时运动状态）。"
        elif operation == "load":
            self.pending = load_scene_config(value)
            self.backend.paused = True
            self._refresh_gui()
            self.message = "预设已载入，请点击“应用场景”。"
        elif operation == "pause":
            self.backend.paused = not self.backend.paused
        elif operation == "step" and self.backend.paused:
            self.backend.step(update_policy=False)
        elif operation == "control_step" and self.backend.paused:
            for _ in range(self.backend.config.substeps):
                self.backend.step()
        elif operation == "reset":
            self.backend.reset()
            self._refresh_gui()
        elif operation == "record":
            self.start_recording(value)
        elif operation == "stop_record":
            self.stop_recording()
        elif operation == "force":
            if self.selected not in ("机器人", "滑板"):
                raise ValueError("请先选择机器人或滑板，再施力。")
            body = "pelvis" if self.selected == "机器人" else "chassis"
            if body:
                kwargs = {"env_ids": [self.backend.view_env]} if self.backend_name == "mjlab" else {}
                self.backend.apply_wrench(body, tuple(value), duration_steps=round(.2/self.backend.config.physics_dt), **kwargs)
        elif operation == "release":
            kwargs = {"env_ids": [self.backend.view_env]} if self.backend_name == "mjlab" else {}
            self.backend.clear_wrenches(**kwargs)
        elif operation == "close":
            self.closed = True

    def tick(self):
        while True:
            try:
                operation, value = self.requests.get_nowait()
            except Empty:
                break
            try:
                self._process(operation, value)
            except Exception as error:
                self.message = f"操作未生效：{error}"
                print(self.message, flush=True)
        if not self.backend.paused and not self.closed:
            self.backend.step()

    def render(self):
        if self.backend_name == "mjlab":
            self.backend._sync_viewer()
            data = self.backend._viewer_data
        else:
            data = self.backend.data
        self.renderer.update_from_mjdata(data)
        now = time.monotonic()
        if now - self._status_time >= .25:
            self._status_time = now
            dirty = self._scene_dirty()
            gripper_status = ""
            if getattr(self.backend, "gripper_command", None):
                from .models.gripper import measured_opening
                gripper_status = "\n\nRobotiq2F-85 · " + " · ".join(
                    f"{side}: raw {value:.0f}, measured opening {1000*measured_opening(self.model, data, side):.2f} mm"
                    for side, value in self.backend.gripper_command.items())
            self.status.content = (f"### skate-sim · {self.backend_name}\n"
                f"{'暂停' if self.backend.paused else '运行'} · {'有待应用的摆放' if dirty else '场景已应用'} · "
                f"{'记录中' if self.recorder else '未记录'}\n\n{self.message}\n\n"
                f"仿真时间：{float(data.time):.3f} s\n\n"
                f"物理步：{self.backend.physics_step_counter} · 策略更新：{self.backend.policy_counter}{gripper_status}")

    def close(self):
        try:
            self.stop_recording()
        finally:
            try:
                self.backend.close()
            finally:
                self.server.stop()


def run_scene(backend_name, backend_kwargs, *, record_path=None, duration=None):
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive and finite")
    workbench = SceneWorkbench(backend_name, backend_kwargs, record_path=record_path, running=duration is not None)
    next_frame = time.perf_counter()
    try:
        while not workbench.closed:
            workbench.tick()
            if duration is not None and np.all(np.asarray(workbench.backend.get_state()["time"]) >= duration):
                break
            if time.perf_counter() >= next_frame:
                workbench.render()
                next_frame = time.perf_counter() + 1/30
            time.sleep(workbench.backend.config.physics_dt)
    except KeyboardInterrupt:
        pass
    finally:
        workbench.close()
    return 0
