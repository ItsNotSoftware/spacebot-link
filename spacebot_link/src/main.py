from __future__ import annotations
import argparse
import cProfile
import io
import pstats
from collections import deque
from math import pi, sin
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np

from panda3d.core import (
    loadPrcFileData,
    CardMaker,
    Texture,
    TextureStage,
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
        sensor_endpoint: str = "tcp://localhost:5556",
        image_endpoint: str = "tcp://localhost:5560",
        gltf_model: str = "../assets/cobot4.glb",
        cmd_endpoint: str = "tcp://localhost:5557",
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

        # Separate SUBs for sensors and images
        self.bus_sensors = TeleopBusSub(sensor_endpoint, rcv_hwm=200)
        self.bus_images = TeleopBusSub(image_endpoint, rcv_hwm=1, conflate=True)

        # PUB for commands (UI -> robot)
        self.cmd_pub = TeleopBusPub(cmd_endpoint)

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
        # Flip via UVs instead of flipping the pixel buffer
        ts = TextureStage.getDefault()
        self.bg_card.setTexScale(ts, 1, -1)
        self.bg_card.setTexOffset(ts, 0, 1)
        self._update_bg_scale()

    def _update_bg_scale(self) -> None:
        if not hasattr(self, "bg_card") or self.camLens is None:
            return
        d = abs(self.bg_card.getY())
        fov_x, fov_y = self.camLens.getFov()
        fov_y_rad = fov_y * (pi / 180.0)
        half_h = d * (
            sin(fov_y_rad / 2.0) / (1e-9 + (1.0 - 0.5 * (fov_y_rad**2) / 3.0))
        )
        if getattr(self, "_bg_aspect", 0) > 0:
            s = half_h / self._bg_aspect
            self.bg_card.setScale(s)

    # ---- tasks ----
    def _bus_task(self, task: "PythonTask"):
        self.bus_sensors.poll(100)
        self.bus_images.poll(100)
        return Task.cont

    def _camera_task(self, task: "PythonTask"):
        rgb = self.bus_images.get_image_rgb(TOPIC_IMAGE)
        if rgb is not None:
            h, w = rgb.shape[:2]
            if self.bg_tex.getXSize() != w or self.bg_tex.getYSize() != h:
                self.bg_tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
            self.bg_tex.setRamImageAs(rgb.tobytes(), "RGB")
        return Task.cont

    def _pose_task(self, task: "PythonTask"):
        if self.camera is None:
            return Task.cont
        payload = self.bus_sensors.get(TOPIC_POSE)
        if isinstance(payload, dict):
            parsed = ros_pose_to_panda_pos_hpr(payload)
            if parsed is not None:
                pos, hpr = parsed
                self.camera.setPos(self.render, pos[0], pos[1], pos[2])
                self.camera.setHpr(self.render, hpr[0], hpr[1], hpr[2])
                self._last_cam_pos_hpr = (pos, hpr)
        return Task.cont

    def _pool_keyboard(self, task: "PythonTask"):
        dt = ClockObject.getGlobalClock().getDt()
        mw = self.mouseWatcherNode
        if not mw:
            return Task.cont

        if not self._move_robot:
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
            lin_x = lin_y = lin_z = 0.0
            ang_x = ang_y = ang_z = 0.0

            if mw.is_button_down(forward_button):
                lin_x = +MOVE_SPEED
            elif mw.is_button_down(backward_button):
                lin_x = -MOVE_SPEED
            if mw.is_button_down(left_button):
                lin_y = +MOVE_SPEED
            elif mw.is_button_down(right_button):
                lin_y = -MOVE_SPEED
            if mw.is_button_down(up_button) or mw.is_button_down(up_button_alt):
                lin_z = +MOVE_SPEED
            if mw.is_button_down(down_button) or mw.is_button_down(down_button_alt):
                lin_z = -MOVE_SPEED

            if mw.is_button_down(roll_left_button):
                ang_x = +ROTATE_SPEED
            elif mw.is_button_down(roll_right_button):
                ang_x = -ROTATE_SPEED
            if mw.is_button_down(pitch_up_button):
                ang_y = +ROTATE_SPEED
            elif mw.is_button_down(pitch_down_button):
                ang_y = -ROTATE_SPEED
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
        img = self.bus_images.get_image_rgb(TOPIC_IMAGE)
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
            self.bus_sensors.close()
        except Exception:
            pass
        try:
            self.bus_images.close()
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
        if self.camera is None:
            return None
        pos_v = self.camera.getPos(self.render)
        hpr_v = self.camera.getHpr(self.render)
        pos = (float(pos_v[0]), float(pos_v[1]), float(pos_v[2]))
        hpr = (float(hpr_v[0]), float(hpr_v[1]), float(hpr_v[2]))
        return pos, hpr

    def get_avatar_pose(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return self.avatar.get_pose()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SpacebotLink teleoperation UI")
    parser.add_argument(
        "--sensor-endpoint",
        default="tcp://localhost:5556",
        help="ZMQ SUB endpoint for telemetry topics (pose, imu, etc).",
    )
    parser.add_argument(
        "--image-endpoint",
        default="tcp://localhost:5560",
        help="ZMQ SUB endpoint for image frames (typically conflated).",
    )
    parser.add_argument(
        "--cmd-endpoint",
        default="tcp://localhost:5557",
        help="ZMQ PUB endpoint for command bus (UI connects).",
    )
    parser.add_argument(
        "--gltf-model",
        default="../assets/cobot4.glb",
        help="Path to the glTF avatar model to load.",
    )
    parser.add_argument(
        "--profile",
        nargs="?",
        const="spacebot_profile.prof",
        help=(
            "Enable cProfile; optionally provide a destination file. Use '-' to "
            "print a summary to stdout."
        ),
    )
    parser.add_argument(
        "--profile-sort",
        default="cumtime",
        help="Sort key for profile stats (e.g. cumtime, tottime, ncalls).",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=30,
        help="How many rows to show when printing stats (<=0 prints all).",
    )
    return parser.parse_args(argv)


def _run_with_profile(
    app: SpacebotLinkApp, destination: str, sort_key: str, top: int
) -> None:
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        app.run()
    finally:
        profiler.disable()
        lines = top if isinstance(top, int) and top > 0 else None
        is_stdout = destination in ("-", "stdout", None)
        if is_stdout:
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            try:
                stats.sort_stats(sort_key)
            except Exception:
                stats.sort_stats("cumtime")
            stats.print_stats(lines)
            print(stream.getvalue())
        else:
            try:
                profiler.dump_stats(destination)
                print(f"[profile] Wrote stats to {destination}")
            except Exception as exc:
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats("cumtime")
                stats.print_stats(lines)
                print(
                    f"[profile] Failed to write stats to {destination}: {exc}\n"
                    f"Falling back to stdout:\n{stream.getvalue()}"
                )


def main(argv=None) -> None:
    args = _parse_args(argv)
    app = SpacebotLinkApp(
        sensor_endpoint=args.sensor_endpoint,
        image_endpoint=args.image_endpoint,
        gltf_model=args.gltf_model,
        cmd_endpoint=args.cmd_endpoint,
    )
    if args.profile:
        _run_with_profile(app, args.profile, args.profile_sort, args.profile_top)
    else:
        app.run()


if __name__ == "__main__":
    main()
