from __future__ import annotations
from collections import deque
from math import pi, sin
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from panda3d.core import (
    loadPrcFileData,
    CardMaker,
    Texture,
    DirectionalLight,
    AmbientLight,
    Vec4,
    PythonTask,
    KeyboardButton,
    Vec3,
    ClockObject,
    NodePath,
)
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from teleop_bus import TeleopBusSub
from utils import apply_opencv_intrinsics_to_lens, ros_orientation_to_panda_hpr
from avatar import Avatar
from ui import UI

# ---- config before ShowBase ----
loadPrcFileData("", "window-title SpaceBotLink")
loadPrcFileData("", "framebuffer-srgb true")
loadPrcFileData("", "transparency-sort off")

# -- try to enable glTF loader if present --
try:
    import importlib

    _gltf_mod = importlib.import_module("panda3d_gltf")
    getattr(_gltf_mod, "GLTFLoader").register_loader()
except Exception:
    pass

# Key mappings
forward_button = KeyboardButton.ascii_key("w")
backward_button = KeyboardButton.ascii_key("s")
left_button = KeyboardButton.ascii_key("a")
right_button = KeyboardButton.ascii_key("d")
up_button = KeyboardButton.ascii_key("e")
down_button = KeyboardButton.ascii_key("q")
up_button_alt = KeyboardButton.space()
down_button_alt = KeyboardButton.lshift()
pitch_up_button = KeyboardButton.ascii_key("i")
pitch_down_button = KeyboardButton.ascii_key("k")
yaw_left_button = KeyboardButton.ascii_key("u")
yaw_right_button = KeyboardButton.ascii_key("o")
roll_left_button = KeyboardButton.ascii_key("j")
roll_right_button = KeyboardButton.ascii_key("l")
reset_orient_button = KeyboardButton.ascii_key("r")

# Topics used by your ROS→ZMQ bridge
TOPIC_IMAGE = "/main_camera/image"
TOPIC_CAMINFO = "/main_camera/camera_info"
TOPIC_IMU = "/imu/data"
TOPIC_POSE = "pose"  # optional custom topic if you publish it

MOVE_SPEED = 0.8
ROTATE_SPEED = 1.5


class SpacebotLinkApp(ShowBase):
    def __init__(
        self,
        endpoint: str = "tcp://localhost:5556",
        gltf_model: str = "../assets/cobot4.glb",
    ):
        super().__init__()
        self.disableMouse()
        self.render.setShaderAuto()

        # lights
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1, 1, 1, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self.render.setLight(sun_np)
        amb = AmbientLight("ambient")
        amb.setColor(Vec4(0.35, 0.35, 0.35, 1))
        amb_np = self.render.attachNewNode(amb)
        self.render.setLight(amb_np)

        # Single SUB for everything
        self.bus = TeleopBusSub(endpoint)

        # background card
        self._make_bg_card(initial_aspect=9 / 16)

        # avatar
        model_path = Path(gltf_model)
        if not model_path.exists():
            model_path = (
                Path(__file__).resolve().parent
                / ".."
                / "assets"
                / Path(gltf_model).name
            )
        self.avatar = Avatar(self.render, self.loader, str(model_path))

        # ui
        self.ui: UI = UI(self)
        self._fps_samples = deque(maxlen=120)

        # reasonable default intrinsics (updated once we see cam_info)
        self._init_default_lens()

        # tasks
        self.taskMgr.add(self._bus_task, "BusTask")
        self.taskMgr.add(self._camera_task, "CameraTask")
        self.taskMgr.add(self._sensor_task, "SensorTask")
        self.taskMgr.add(self._pool_keyboard, "PoolKeyboard")
        self.taskMgr.add(self._hud_task, "HUDTask")

        # cleanup
        self.exitFunc: Optional[Callable[[], None]] = self._cleanup

    # ---- lens / bg helpers ----
    def _init_default_lens(self) -> None:
        w, h = 1280, 720
        fx = fy = 900.0
        cx, cy = w / 2, h / 2
        apply_opencv_intrinsics_to_lens(self.camLens, w, h, fx, fy, cx, cy)
        self.camLens.setNear(0.1)
        self.camLens.setFar(5000.0)
        self._update_bg_scale()

    def _make_bg_card(self, initial_aspect: float) -> None:
        if self.camera is None:
            return
        cm = CardMaker("background")
        cm.setFrame(-1, 1, -initial_aspect, initial_aspect)
        self.bg_card: NodePath = self.camera.attachNewNode(cm.generate())
        self.bg_card.setScale(50)
        self.bg_card.setPos(0, 100, 0)
        self.bg_card.setBin("background", 0)
        self.setBackgroundColor(0, 0, 0, 1)
        self.bg_card.setDepthWrite(False)
        self.bg_card.setDepthTest(False)
        self.bg_tex: Texture = Texture("background")
        self.bg_tex.setup2dTexture(2, 2, Texture.T_unsigned_byte, Texture.F_rgb)
        self.bg_card.setTexture(self.bg_tex)
        self._bg_aspect = float(initial_aspect)
        self._update_bg_scale()

    def _update_bg_scale(self) -> None:
        if not hasattr(self, "bg_card") or self.camLens is None:
            return
        d = abs(self.bg_card.getY())
        fov_x, fov_y = self.camLens.getFov()
        fov_y_rad = fov_y * (pi / 180.0)
        half_h = d * (
            sin(fov_y_rad / 2.0) / (1e-9 + (1.0 - 0.5 * (fov_y_rad**2) / 3.0))
        )  # tan via series; stable enough
        if getattr(self, "_bg_aspect", 0) > 0:
            s = half_h / self._bg_aspect
            self.bg_card.setScale(s)

    # ---- tasks ----
    def _bus_task(self, task: "PythonTask"):
        # Pump messages into cache
        self.bus.poll(100)
        return Task.cont

    def _camera_task(self, task: "PythonTask"):
        rgb = self.bus.get_image_rgb(TOPIC_IMAGE)
        if rgb is not None:
            # Frames arrive with origin at bottom-left; flip so the texture matches screen space.
            rgb = np.flipud(rgb).copy()
            h, w = rgb.shape[:2]
            if self.bg_tex.getXSize() != w or self.bg_tex.getYSize() != h:
                self.bg_tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
            self.bg_tex.setRamImageAs(rgb.tobytes(), "RGB")
        return Task.cont

    def _sensor_task(self, task: "PythonTask"):
        # Pose (optional custom topic)
        pose = self.bus.get(TOPIC_POSE)
        if isinstance(pose, dict):
            try:
                x = pose.get("x")
                y = pose.get("y")
                z = pose.get("z")
                if any(v is not None for v in (x, y, z)):
                    curr_x, curr_y, curr_z = (
                        self.avatar._front.getX(),
                        self.avatar._front.getY(),
                        self.avatar._front.getZ(),
                    )
                    self.avatar.set_pos(
                        float(x) if x is not None else float(curr_x),
                        float(y) if y is not None else float(curr_y),
                        float(z) if z is not None else float(curr_z),
                    )
            except Exception:
                pass
            try:
                h_in = pose.get("h")
                p_in = pose.get("p")
                r_in = pose.get("r")
                if any(v is not None for v in (h_in, p_in, r_in)):
                    ch, cp, cr = self.avatar.get_hpr()
                    self.avatar.set_hpr(
                        float(h_in) if h_in is not None else ch,
                        float(p_in) if p_in is not None else cp,
                        float(r_in) if r_in is not None else cr,
                    )
            except Exception:
                pass

        # IMU → orientation (fallback if pose.hpr absent)
        imu = self.bus.get(TOPIC_IMU)
        if isinstance(imu, dict):
            orientation = imu.get("orientation")
            if isinstance(orientation, dict):
                hpr = ros_orientation_to_panda_hpr(orientation)
                if hpr:
                    self.avatar.set_hpr(*hpr)

        # Camera intrinsics
        caminfo = self.bus.get(TOPIC_CAMINFO)
        if isinstance(caminfo, dict):
            try:
                w = int(caminfo.get("width", 0))
                h = int(caminfo.get("height", 0))
                k = caminfo.get("k")
                if isinstance(k, list) and len(k) >= 6 and w > 0 and h > 0:
                    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
                    apply_opencv_intrinsics_to_lens(self.camLens, w, h, fx, fy, cx, cy)
                    self.camLens.setNear(0.1)
                    self.camLens.setFar(5000.0)
                    self._bg_aspect = h / max(1.0, float(w))
                    self._update_bg_scale()
            except (TypeError, ValueError):
                pass

        return Task.cont

    def _pool_keyboard(self, task: "PythonTask"):
        dt = ClockObject.getGlobalClock().getDt()
        mw = self.mouseWatcherNode
        if not mw:
            return Task.cont

        move = Vec3(0, 0, 0)
        if mw.is_button_down(forward_button):
            move.y += MOVE_SPEED * dt
        if mw.is_button_down(backward_button):
            move.y -= MOVE_SPEED * dt
        if mw.is_button_down(left_button):
            move.x -= MOVE_SPEED * dt
        if mw.is_button_down(right_button):
            move.x += MOVE_SPEED * dt
        if mw.is_button_down(up_button) or mw.is_button_down(up_button_alt):
            move.z += MOVE_SPEED * dt
        if mw.is_button_down(down_button) or mw.is_button_down(down_button_alt):
            move.z -= MOVE_SPEED * dt
        if move.length_squared() > 0:
            self.avatar.move_world(move.x, move.y, move.z)

        dh = dp = dr = 0.0
        step = ROTATE_SPEED * 60.0 * dt
        if mw.is_button_down(yaw_left_button):
            dh += step
        if mw.is_button_down(yaw_right_button):
            dh -= step
        if mw.is_button_down(pitch_up_button):
            dp += step
        if mw.is_button_down(pitch_down_button):
            dp -= step
        if mw.is_button_down(roll_left_button):
            dr += step
        if mw.is_button_down(roll_right_button):
            dr -= step
        if dh or dp or dr:
            self.avatar.add_hpr(dh, dp, dr)
        if mw.is_button_down(reset_orient_button):
            self.avatar.reset_hpr()
        return Task.cont

    def _hud_task(self, task: "PythonTask"):
        dt = ClockObject.getGlobalClock().getDt()
        if dt > 1e-6:
            self._fps_samples.append(1.0 / dt)
        avg_fps = (
            (sum(self._fps_samples) / len(self._fps_samples))
            if self._fps_samples
            else 0.0
        )
        rgb_w = rgb_h = None
        img = self.bus.get_image_rgb(TOPIC_IMAGE)
        if img is not None:
            rgb_h, rgb_w = img.shape[:2]
            self.ui.update(f"Video {rgb_w}x{rgb_h} | FPS {avg_fps:.1f}")
        else:
            self.ui.update(f"Waiting for video… | FPS {avg_fps:.1f}")
        return Task.cont

    def _cleanup(self):
        try:
            self.bus.close()
        except Exception:
            pass


if __name__ == "__main__":
    app = SpacebotLinkApp()
    app.run()
