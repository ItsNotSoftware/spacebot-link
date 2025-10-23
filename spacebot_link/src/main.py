from __future__ import annotations
from collections import deque
from math import pi, sin
from pathlib import Path
from typing import Optional, Callable, Tuple

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

from teleop_bus import TeleopBusSub, TeleopBusPub
from utils import (
    apply_opencv_intrinsics_to_lens,
    ros_pose_to_panda_pos_hpr,
)
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
TOPIC_POSE = "/space_cobot/pose"
TOPIC_CMD_VEL = "/space_cobot/cmd_vel"

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

        # Single SUB for sensors/pose
        self.bus = TeleopBusSub(endpoint)

        # PUB for commands (UI -> robot)
        self.cmd_pub = TeleopBusPub("tcp://localhost:5557")

        # Toggle: move avatar (default) or robot (send cmd_vel)
        self._move_robot: bool = False
        self.accept("t", self._toggle_move_mode)

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
        self.taskMgr.add(self._pose_task, "PoseTask")
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
        self.camLens.setNear(0.1)  # type: ignore
        self.camLens.setFar(5000.0)  # type: ignore
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

    def _pose_task(self, task: "PythonTask"):
        """Update camera NodePath from world-frame pose messages.

        Applies transforms explicitly in the world/render space to avoid any
        ambiguity from prior parenting or default-local ops.
        """
        if self.camera is None:
            return Task.cont
        payload = self.bus.get(TOPIC_POSE)
        if isinstance(payload, dict):
            parsed = ros_pose_to_panda_pos_hpr(payload)
            if parsed is not None:
                pos, hpr = parsed
                print(pos)
                # Explicitly set in world (render) space
                self.camera.setPos(self.render, pos[0], pos[1], pos[2])
                self.camera.setHpr(self.render, hpr[0], hpr[1], hpr[2])
                # cache for HUD
                self._last_cam_pos_hpr = (pos, hpr)
        return Task.cont

    def _pool_keyboard(self, task: "PythonTask"):
        dt = ClockObject.getGlobalClock().getDt()
        mw = self.mouseWatcherNode
        if not mw:
            return Task.cont

        if not self._move_robot:
            # Avatar movement (as before)
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
        else:
            # Robot movement (publish full 6-DOF cmd_vel). Keep zero when no keys pressed.
            lin_x = lin_y = lin_z = 0.0
            ang_x = ang_y = ang_z = 0.0

            # Translational:
            #   W/S -> forward/back (x)
            if mw.is_button_down(forward_button):
                lin_x = +MOVE_SPEED
            elif mw.is_button_down(backward_button):
                lin_x = -MOVE_SPEED
            #   A/D -> strafe left/right (y)
            if mw.is_button_down(left_button):
                lin_y = +MOVE_SPEED
            elif mw.is_button_down(right_button):
                lin_y = -MOVE_SPEED
            #   Q/E or Shift/Space -> down/up (z)
            if mw.is_button_down(up_button) or mw.is_button_down(up_button_alt):
                lin_z = +MOVE_SPEED
            if mw.is_button_down(down_button) or mw.is_button_down(down_button_alt):
                lin_z = -MOVE_SPEED

            # Rotational:
            #   J/L -> roll (x)
            if mw.is_button_down(roll_left_button):
                ang_x = +ROTATE_SPEED
            elif mw.is_button_down(roll_right_button):
                ang_x = -ROTATE_SPEED
            #   I/K -> pitch (y)
            if mw.is_button_down(pitch_up_button):
                ang_y = +ROTATE_SPEED
            elif mw.is_button_down(pitch_down_button):
                ang_y = -ROTATE_SPEED
            #   U/O -> yaw (z)
            if mw.is_button_down(yaw_left_button):
                ang_z = +ROTATE_SPEED
            elif mw.is_button_down(yaw_right_button):
                ang_z = -ROTATE_SPEED

            self._publish_cmd_vel(lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)
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
            pose_txt = ""
            pos_hpr = getattr(self, "_last_cam_pos_hpr", None)
            if pos_hpr is not None:
                (x, y, z), (h, p, r) = pos_hpr
                pose_txt = (
                    f" | Cam pos (m) [{x:.2f}, {y:.2f}, {z:.2f}]"
                    f" HPR (deg) [{h:.1f}, {p:.1f}, {r:.1f}]"
                )
            self.ui.update(f"Video {rgb_w}x{rgb_h} | FPS {avg_fps:.1f}{pose_txt}")
        else:
            pose_txt = ""
            pos_hpr = getattr(self, "_last_cam_pos_hpr", None)
            if pos_hpr is not None:
                (x, y, z), (h, p, r) = pos_hpr
                pose_txt = (
                    f" | Cam pos (m) [{x:.2f}, {y:.2f}, {z:.2f}]"
                    f" HPR (deg) [{h:.1f}, {p:.1f}, {r:.1f}]"
                )
            self.ui.update(f"Waiting for video… | FPS {avg_fps:.1f}{pose_txt}")
        return Task.cont

    def _cleanup(self):
        try:
            self.bus.close()
        except Exception:
            pass
        try:
            self.cmd_pub.close()
        except Exception:
            pass

    # ---- control helpers ----
    def _toggle_move_mode(self) -> None:
        self._move_robot = not self._move_robot
        self.ui.set_move_target("Robot" if self._move_robot else "Avatar")
        if not self._move_robot:
            # ensure robot is stopped when exiting robot mode
            self._publish_cmd_vel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _publish_cmd_vel(
        self,
        lin_x: float,
        lin_y: float,
        lin_z: float,
        ang_x: float,
        ang_y: float,
        ang_z: float,
    ) -> None:
        data = {
            "linear": {"x": float(lin_x), "y": float(lin_y), "z": float(lin_z)},
            "angular": {"x": float(ang_x), "y": float(ang_y), "z": float(ang_z)},
        }
        try:
            self.cmd_pub.publish(TOPIC_CMD_VEL, data)
        except Exception:
            pass

    # ---- pose getters ----
    def get_camera_pose(
        self,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Return camera (pos, hpr) in world coordinates, or None if unavailable."""
        if self.camera is None:
            return None
        pos_v = self.camera.getPos(self.render)
        hpr_v = self.camera.getHpr(self.render)
        pos = (float(pos_v[0]), float(pos_v[1]), float(pos_v[2]))
        hpr = (float(hpr_v[0]), float(hpr_v[1]), float(hpr_v[2]))
        return pos, hpr

    def get_avatar_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return avatar (pos, hpr) in world coordinates."""
        return self.avatar.get_pose()


if __name__ == "__main__":
    app = SpacebotLinkApp()
    app.run()
