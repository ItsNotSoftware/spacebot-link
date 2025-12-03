from __future__ import annotations
from collections import deque
import time
from math import asin, atan2, degrees, pi, sin, sqrt
from pathlib import Path
from typing import Optional, Callable, Tuple, Sequence, Dict, List

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

import p3dimgui
from imgui_bundle import imgui

from teleop_bus import TeleopBusSub, TeleopBusPub
from utils import (
    apply_opencv_intrinsics_to_lens,
    ros_pose_to_panda_pos_hpr,
    panda_pose_to_ros,
)
from avatar import Avatar
from ui import UI

WINDOW_TITLE = "SpaceBotLink"
FRAMEBUFFER_SRGB_CFG = "framebuffer-srgb true"
TRANSPARENCY_SORT_CFG = "transparency-sort off"

# Key mappings
FORWARD_BUTTON = KeyboardButton.ascii_key("w")
BACKWARD_BUTTON = KeyboardButton.ascii_key("s")
LEFT_BUTTON = KeyboardButton.ascii_key("a")
RIGHT_BUTTON = KeyboardButton.ascii_key("d")
UP_BUTTON = KeyboardButton.ascii_key("e")
DOWN_BUTTON = KeyboardButton.ascii_key("q")
UP_BUTTON_ALT = KeyboardButton.space()
DOWN_BUTTON_ALT = KeyboardButton.lshift()
PITCH_UP_BUTTON = KeyboardButton.ascii_key("i")
PITCH_DOWN_BUTTON = KeyboardButton.ascii_key("k")
YAW_LEFT_BUTTON = KeyboardButton.ascii_key("u")
YAW_RIGHT_BUTTON = KeyboardButton.ascii_key("o")
ROLL_LEFT_BUTTON = KeyboardButton.ascii_key("j")
ROLL_RIGHT_BUTTON = KeyboardButton.ascii_key("l")
RESET_ORIENT_BUTTON = KeyboardButton.ascii_key("r")
RESET_TO_ROBOT_ORIENT_BUTTON = KeyboardButton.backspace()

# Topics used by your ROS→ZMQ bridge
TOPIC_IMAGE = "/main_camera/image"
TOPIC_CAMINFO = "/main_camera/camera_info"
TOPIC_IMU = "/imu/data"
TOPIC_POSE = "/space_cobot/pose"
TOPIC_CMD_VEL = "/space_cobot/cmd_vel"
TOPIC_GOAL = "/nav6d/goal"
TOPIC_PATH = "/nav6d/planner/path"
TOPIC_CMD_PATH = "/nav6d/planner/path"

MOVE_SPEED = 0.8
ROTATE_SPEED = 1.5

# ---- config before ShowBase ----
loadPrcFileData("", f"window-title {WINDOW_TITLE}")
loadPrcFileData("", FRAMEBUFFER_SRGB_CFG)
loadPrcFileData("", TRANSPARENCY_SORT_CFG)

try:
    import importlib

    _gltf_mod = importlib.import_module("panda3d_gltf")
    getattr(_gltf_mod, "GLTFLoader").register_loader()
except Exception:
    pass


class SpacebotLinkApp(ShowBase):
    def __init__(
        self,
        sensor_endpoint: str = "tcp://localhost:5556",
        image_endpoint: str = "tcp://localhost:5560",
        gltf_model: str = "../assets/cobot_ghost.glb",
        cmd_endpoint: str = "tcp://localhost:5557",
    ) -> None:
        """Initialize app wiring, assets, networking, and tasks."""
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

        # path ghosts
        self._path_markers: List[NodePath] = []
        self._path_proto: Optional[NodePath] = None
        self._path_proto_failed: bool = False
        self._last_path_data: Optional[Dict] = None

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
        self.ui: UI = UI(self, on_abort=self._abort_to_robot_pose)
        self._fps_samples = deque(maxlen=120)
        self._avg_fps: float = 0.0
        self._last_ros_pose: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None
        self._last_ros_orientation: Optional[Dict[str, float]] = None
        self._last_robot_pose_panda: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None
        self._last_goal_pub_time: float = 0.0
        self._last_goal_pose: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None
        self._last_robot_hpr: Optional[Tuple[float, float, float]] = None
        self._pending_abort_goal: bool = False
        self._abort_restore_mode: Optional[str] = None
        self._abort_restore_task: Optional[Task] = None
        self._robot_stopped_last: bool = False
        self._nav_publishing_enabled: bool = True
        self._follow_path_points: List[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = []
        self._follow_pos_eps: float = 0.02
        self._follow_hpr_eps: float = 1.0
        self._follow_reached_thresh: float = 0.2
        self._follow_sample_period: float = 0.15
        self._imgui_ready: bool = False
        self._imgui_ini_path: Path = Path(__file__).resolve().parent.parent / "imgui.ini"

        # reasonable default intrinsics (updated once we see cam_info)
        self._init_default_lens()

        # tasks
        self.taskMgr.add(self._bus_task, "BusTask")
        self.taskMgr.add(self._camera_task, "CameraTask")
        self.taskMgr.add(self._pose_task, "PoseTask")
        self.taskMgr.add(self._pool_keyboard, "PoolKeyboard")
        self.taskMgr.add(self._metrics_task, "MetricsTask")
        self.taskMgr.add(self._goal_publish_task, "GoalPublishTask")
        self.taskMgr.add(self._path_task, "PathTask")
        self.taskMgr.doMethodLater(
            self._follow_sample_period, self._follow_mode_tick, "FollowModeTick"
        )

        self._init_imgui()

        # cleanup
        self.exitFunc: Optional[Callable[[], None]] = self._cleanup

    # ---- lens / bg helpers ----
    def _init_default_lens(self) -> None:
        """Seed a reasonable default lens configuration until camera info arrives."""
        w, h = 1280, 720
        fx = fy = 900.0
        cx, cy = w / 2, h / 2
        apply_opencv_intrinsics_to_lens(self.camLens, w, h, fx, fy, cx, cy)
        self.camLens.setNear(0.1)  # type: ignore
        self.camLens.setFar(5000.0)  # type: ignore
        self._update_bg_scale()

    def _make_bg_card(self, initial_aspect: float) -> None:
        """Create a textured card behind the scene to display the camera feed."""
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
        """Scale the background card to fill the current camera frustum."""
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

    def _resolve_asset_path(self, gltf_path: str) -> Optional[Path]:
        """Return an existing path for a GLTF asset (local or ../assets)."""
        path = Path(gltf_path)
        if path.exists():
            return path
        fallback = (
            Path(__file__).resolve().parent / ".." / "assets" / Path(gltf_path).name
        ).resolve()
        return fallback if fallback.exists() else None

    # ---- tasks ----
    def _bus_task(self, task: "PythonTask") -> int:
        """Pump sensor and image sockets so they stay current."""
        self.bus_sensors.poll(100)
        self.bus_images.poll(100)
        return Task.cont

    def _goal_publish_task(self, task: "PythonTask") -> int:
        """Publish nav goal if avatar pose changed since the last send."""
        if self.ui.mode != "Goal Mode":
            return Task.cont
        if not self._nav_publishing_enabled:
            return Task.cont

        pose = self.avatar.get_pose()
        if not self._pose_changed_since_last_goal(pose):
            return Task.cont

        ros_pose = panda_pose_to_ros(pose)
        if ros_pose is None:
            return Task.cont

        msg = {"header": {"frame_id": "map"}, "pose": ros_pose}
        try:
            self.cmd_pub.publish(TOPIC_GOAL, msg)
            self._last_goal_pose = pose
        except Exception:
            pass

        return Task.cont

    def _follow_mode_tick(self, task: "PythonTask") -> int:
        """Sample avatar pose and publish a path while in Follow Mode."""
        if self.ui.mode == "Follow Mode":
            self._prune_follow_path()
            pose = self.avatar.get_pose()
            if self._should_append_follow_pose(pose):
                self._follow_path_points.append(pose)
            if self._nav_publishing_enabled:
                self._publish_follow_path()
        task.delayTime = self._follow_sample_period
        return Task.again

    def _camera_task(self, task: "PythonTask") -> int:
        """Update the background texture with the latest camera frame."""
        rgb = self.bus_images.get_image_rgb(TOPIC_IMAGE)
        if rgb is not None:
            h, w = rgb.shape[:2]
            if self.bg_tex.getXSize() != w or self.bg_tex.getYSize() != h:
                self.bg_tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
            self.bg_tex.setRamImageAs(rgb.tobytes(), "RGB")
        return Task.cont

    def _should_append_follow_pose(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> bool:
        """Return True if pose is far enough from last sample to append."""
        if not self._follow_path_points:
            return True
        (x1, y1, z1), (h1, p1, r1) = self._follow_path_points[-1]
        (x2, y2, z2), (h2, p2, r2) = pose
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        dh = abs(h2 - h1)
        dp = abs(p2 - p1)
        dr = abs(r2 - r1)
        return (
            dx > self._follow_pos_eps
            or dy > self._follow_pos_eps
            or dz > self._follow_pos_eps
            or dh > self._follow_hpr_eps
            or dp > self._follow_hpr_eps
            or dr > self._follow_hpr_eps
        )

    def _prune_follow_path(self) -> None:
        """Drop leading waypoints once the robot reaches them."""
        if self._last_robot_pose_panda is None or not self._follow_path_points:
            return
        (rx, ry, rz), _ = self._last_robot_pose_panda
        while len(self._follow_path_points) > 1:
            (px, py, pz), _ = self._follow_path_points[0]
            dist = sqrt((px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2)
            if dist <= self._follow_reached_thresh:
                self._follow_path_points.pop(0)
            else:
                break

    def _publish_follow_path(self) -> None:
        """Publish the current follow-mode path as a nav_msgs/Path-like dict."""
        if not self._follow_path_points:
            return
        points = list(self._follow_path_points)
        if len(points) == 1 and self._last_robot_pose_panda is not None:
            points = [self._last_robot_pose_panda] + points
        poses = []
        for pos_hpr in points:
            ros_pose = panda_pose_to_ros(pos_hpr)
            if ros_pose is None:
                continue
            poses.append({"pose": ros_pose})
        if not poses:
            return
        msg = {"header": {"frame_id": "map"}, "poses": poses}
        try:
            self.cmd_pub.publish(TOPIC_CMD_PATH, msg)
        except Exception:
            pass

    def _publish_hold_path(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        ros_pose_override: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Publish a tiny 2-point path at the current pose to keep the controller stable."""
        ros_pose = ros_pose_override or panda_pose_to_ros(pose)
        if ros_pose is None:
            return
        poses = [{"pose": ros_pose}, {"pose": ros_pose}]
        msg = {"header": {"frame_id": "map"}, "poses": poses}
        try:
            self.cmd_pub.publish(TOPIC_CMD_PATH, msg)
        except Exception:
            pass

    def _panda_pose_to_ros_tuple(
        self, pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Convert panda (pos,hpr) tuple to ROS-frame (pos,rpy deg)."""
        ros_pose = panda_pose_to_ros(pose)
        if ros_pose is None:
            return None
        pos = ros_pose["position"]
        ori = ros_pose["orientation"]
        rpy = self._quat_to_rpy_deg(ori["x"], ori["y"], ori["z"], ori["w"])
        if rpy is None:
            return None
        return (pos["x"], pos["y"], pos["z"]), rpy

    def _pose_task(self, task: "PythonTask") -> int:
        """Track robot pose and drive the camera to follow it."""
        payload = self.bus_sensors.get(TOPIC_POSE)
        if isinstance(payload, dict):
            ros_pose = self._extract_ros_pose(payload)
            if ros_pose is not None:
                pos, rpy, ori = ros_pose
                self._last_ros_pose = (pos, rpy)
                self._last_ros_orientation = ori
            if self.camera is not None:
                parsed = ros_pose_to_panda_pos_hpr(payload)
                if parsed is not None:
                    pos, hpr = parsed
                    self._last_robot_hpr = hpr
                    self._last_robot_pose_panda = (pos, hpr)
                    self.camera.setPos(self.render, pos[0], pos[1], pos[2])
                    self.camera.setHpr(self.render, hpr[0], hpr[1], hpr[2])
                    self._last_cam_pos_hpr = (pos, hpr)
                    self._maybe_finalize_abort()
            self._maybe_sync_avatar_on_stop()
        return Task.cont

    def _pool_keyboard(self, task: "PythonTask") -> int:
        """Poll keyboard to either move the avatar or send cmd_vel."""
        dt = ClockObject.getGlobalClock().getDt()
        mw = self.mouseWatcherNode
        if not mw:
            return Task.cont

        if not self._move_robot:
            move = Vec3(0, 0, 0)
            if mw.is_button_down(FORWARD_BUTTON):
                move.y += MOVE_SPEED * dt
            if mw.is_button_down(BACKWARD_BUTTON):
                move.y -= MOVE_SPEED * dt
            if mw.is_button_down(LEFT_BUTTON):
                move.x -= MOVE_SPEED * dt
            if mw.is_button_down(RIGHT_BUTTON):
                move.x += MOVE_SPEED * dt
            if mw.is_button_down(UP_BUTTON) or mw.is_button_down(UP_BUTTON_ALT):
                move.z += MOVE_SPEED * dt
            if mw.is_button_down(DOWN_BUTTON) or mw.is_button_down(DOWN_BUTTON_ALT):
                move.z -= MOVE_SPEED * dt
            if move.length_squared() > 0:
                # Apply translation in the current camera (robot) frame.
                frame = self.camera if self.camera is not None else self.render
                q = frame.getQuat(self.render)
                delta = q.xform(move)
                self.avatar.move_world(delta.x, delta.y, delta.z)

            dh = dp = dr = 0.0
            step = ROTATE_SPEED * 60.0 * dt
            if mw.is_button_down(YAW_LEFT_BUTTON):
                dh += step
            if mw.is_button_down(YAW_RIGHT_BUTTON):
                dh -= step
            if mw.is_button_down(PITCH_UP_BUTTON):
                dp += step
            if mw.is_button_down(PITCH_DOWN_BUTTON):
                dp -= step
            if mw.is_button_down(ROLL_LEFT_BUTTON):
                dr += step
            if mw.is_button_down(ROLL_RIGHT_BUTTON):
                dr -= step
            if dh or dp or dr:
                self.avatar.add_hpr(dh, dp, dr)
            if (
                mw.is_button_down(RESET_TO_ROBOT_ORIENT_BUTTON)
                and self._last_robot_hpr is not None
            ):
                self.avatar.set_hpr(*self._last_robot_hpr)
            if mw.is_button_down(RESET_ORIENT_BUTTON):
                self.avatar.reset_hpr()
        else:
            lin_x = lin_y = lin_z = 0.0
            ang_x = ang_y = ang_z = 0.0

            if mw.is_button_down(FORWARD_BUTTON):
                lin_x = +MOVE_SPEED
            elif mw.is_button_down(BACKWARD_BUTTON):
                lin_x = -MOVE_SPEED
            if mw.is_button_down(LEFT_BUTTON):
                lin_y = +MOVE_SPEED
            elif mw.is_button_down(RIGHT_BUTTON):
                lin_y = -MOVE_SPEED
            if mw.is_button_down(UP_BUTTON) or mw.is_button_down(UP_BUTTON_ALT):
                lin_z = +MOVE_SPEED
            if mw.is_button_down(DOWN_BUTTON) or mw.is_button_down(DOWN_BUTTON_ALT):
                lin_z = -MOVE_SPEED

            if mw.is_button_down(ROLL_LEFT_BUTTON):
                ang_x = +ROTATE_SPEED
            elif mw.is_button_down(ROLL_RIGHT_BUTTON):
                ang_x = -ROTATE_SPEED
            if mw.is_button_down(PITCH_UP_BUTTON):
                ang_y = +ROTATE_SPEED
            elif mw.is_button_down(PITCH_DOWN_BUTTON):
                ang_y = -ROTATE_SPEED
            if mw.is_button_down(YAW_LEFT_BUTTON):
                ang_z = +ROTATE_SPEED
            elif mw.is_button_down(YAW_RIGHT_BUTTON):
                ang_z = -ROTATE_SPEED

            self._publish_cmd_vel(lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)
        return Task.cont

    def _metrics_task(self, task: "PythonTask") -> int:
        """Accumulate FPS samples for display."""
        dt = ClockObject.getGlobalClock().getDt()
        if dt > 1e-6:
            self._fps_samples.append(1.0 / dt)
        self._avg_fps = (
            (sum(self._fps_samples) / len(self._fps_samples))
            if self._fps_samples
            else 0.0
        )
        return Task.cont

    def _cleanup(self) -> None:
        """Close all bus publishers/subscribers on exit."""
        self._save_imgui_settings()
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
        self._clear_path_markers()

    def _save_imgui_settings(self) -> None:
        """Persist ImGui layout/settings to disk."""
        if not self._imgui_ready:
            return
        try:
            self._imgui_ini_path.parent.mkdir(parents=True, exist_ok=True)
            imgui.save_ini_settings_to_disk(str(self._imgui_ini_path))
        except Exception:
            pass

    # ---- control helpers ----
    def _set_move_mode(self, move_robot: bool) -> None:
        """Switch between avatar movement and sending cmd_vel to the robot."""
        move_robot = bool(move_robot)
        if move_robot == self._move_robot:
            return
        self._move_robot = move_robot
        self.ui.set_move_target("Robot" if self._move_robot else "Avatar")
        if not self._move_robot:
            self._publish_cmd_vel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _toggle_move_mode(self) -> None:
        """Toggle between avatar and robot control modes."""
        self._set_move_mode(not self._move_robot)

    def _publish_cmd_vel(
        self,
        lin_x: float,
        lin_y: float,
        lin_z: float,
        ang_x: float,
        ang_y: float,
        ang_z: float,
    ) -> None:
        """Publish a geometry_msgs/Twist-style command."""
        data = {
            "linear": {"x": float(lin_x), "y": float(lin_y), "z": float(lin_z)},
            "angular": {"x": float(ang_x), "y": float(ang_y), "z": float(ang_z)},
        }
        try:
            self.cmd_pub.publish(TOPIC_CMD_VEL, data)
        except Exception:
            pass

    def _extract_ros_pose(
        self, payload: dict
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], dict]]:
        """Normalize incoming ROS pose payload into position, rpy, and orientation."""
        pose = payload.get("pose") if isinstance(payload, dict) else None
        if isinstance(pose, dict):
            position = pose.get("position")
            orientation = pose.get("orientation")
        else:
            position = payload.get("position") if isinstance(payload, dict) else None
            orientation = (
                payload.get("orientation") if isinstance(payload, dict) else None
            )

        if not isinstance(position, dict) or not isinstance(orientation, dict):
            return None

        try:
            pos = (
                float(position.get("x")),
                float(position.get("y")),
                float(position.get("z")),
            )
        except (TypeError, ValueError):
            return None

        rpy = self._quat_to_rpy_deg(
            orientation.get("x"),
            orientation.get("y"),
            orientation.get("z"),
            orientation.get("w"),
        )
        if rpy is None:
            return None
        ori = {
            "x": float(orientation.get("x")),
            "y": float(orientation.get("y")),
            "z": float(orientation.get("z")),
            "w": float(orientation.get("w")),
        }
        return pos, rpy, ori

    def _quat_to_rpy_deg(
        self, qx: float, qy: float, qz: float, qw: float
    ) -> Optional[Tuple[float, float, float]]:
        """Convert quaternion components into roll, pitch, yaw in degrees."""
        try:
            qx = float(qx)
            qy = float(qy)
            qz = float(qz)
            qw = float(qw)
        except (TypeError, ValueError):
            return None

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if sinp >= 1.0:
            pitch = pi / 2.0
        elif sinp <= -1.0:
            pitch = -pi / 2.0
        else:
            pitch = asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = atan2(siny_cosp, cosy_cosp)
        return (degrees(roll), degrees(pitch), degrees(yaw))

    def _pose_changed_since_last_goal(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        pos_eps: float = 1e-4,
        hpr_eps: float = 1e-3,
    ) -> bool:
        """Check if pose moved enough to warrant publishing a new goal."""
        if self._last_goal_pose is None:
            return True
        (x1, y1, z1), (h1, p1, r1) = self._last_goal_pose
        (x2, y2, z2), (h2, p2, r2) = pose
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        dh = abs(h2 - h1)
        dp = abs(p2 - p1)
        dr = abs(r2 - r1)
        return (
            (dx > pos_eps)
            or (dy > pos_eps)
            or (dz > pos_eps)
            or (dh > hpr_eps)
            or (dp > hpr_eps)
            or (dr > hpr_eps)
        )

    # ---- pose getters ----
    def get_camera_pose(
        self,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Return the camera pose in world coordinates, if available."""
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
        """Return the avatar pose in world coordinates."""
        return self.avatar.get_pose()

    # ---- path visualizer ----
    def _load_path_proto(self) -> Optional[NodePath]:
        """Load the reusable prototype model for path markers."""
        if self._path_proto_failed:
            return None
        resolved = self._resolve_asset_path("../assets/path_ghost.glb")
        if resolved is None:
            self._path_proto_failed = True
            return None
        proto = self.loader.loadModel(str(resolved))
        if proto is None or proto.isEmpty():
            self._path_proto_failed = True
            return None
        return proto

    def _clear_path_markers(self) -> None:
        """Remove any existing path marker nodes from the scene graph."""
        for np_ in self._path_markers:
            try:
                np_.removeNode()
            except Exception:
                pass
        self._path_markers.clear()

    def _parse_ros_path(
        self, payload: Dict
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Convert a ROS Path-like dict into Panda3D (pos, hpr) tuples."""
        poses = payload.get("poses") if isinstance(payload, dict) else None
        if not isinstance(poses, list):
            return []

        parsed: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        for entry in poses:
            pose_dict = entry.get("pose") if isinstance(entry, dict) else None
            pose_obj = pose_dict if isinstance(pose_dict, dict) else entry
            if not isinstance(pose_obj, dict):
                continue
            pos_hpr = ros_pose_to_panda_pos_hpr(pose_obj)
            if pos_hpr is not None:
                parsed.append(pos_hpr)
        return parsed

    def _path_task(self, task: "PythonTask") -> int:
        """Lay down path ghost markers for every other pose in the latest path."""
        if self.ui.mode == "Follow Mode":
            self._render_path_markers(self._follow_path_points)
            return Task.cont

        payload = self.bus_sensors.get(TOPIC_PATH)
        if isinstance(payload, dict) and payload != self._last_path_data:
            self._last_path_data = payload
            poses = self._parse_ros_path(payload)
            self._render_path_markers(poses)
        return Task.cont

    def _render_path_markers(
        self,
        poses: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    ) -> None:
        """Render path markers from a list of Panda3D (pos, hpr) tuples."""
        self._clear_path_markers()
        if not poses:
            return
        if self._path_proto is None:
            self._path_proto = self._load_path_proto()
        proto = self._path_proto
        if proto is None:
            return
        for idx, (pos, hpr) in enumerate(poses):
            if idx % 4 != 0 or idx == 0:
                continue
            ghost = proto.copyTo(self.render)
            ghost.setPos(self.render, pos[0], pos[1], pos[2])
            ghost.setHpr(self.render, hpr[0], hpr[1], hpr[2])
            ghost.setBin("fixed", 5)
            ghost.setDepthWrite(False)
            self._path_markers.append(ghost)

    # ---- debug UI ----
    def _init_imgui(self) -> None:
        """Initialize the ImGui overlay if available."""
        try:
            p3dimgui.init()
            if self._imgui_ini_path.exists():
                imgui.load_ini_settings_from_disk(str(self._imgui_ini_path))
            style = imgui.get_style()
            style.font_size_base = 22.0  # request larger base font size
            style.font_scale_main = 1.25
            style.scale_all_sizes(1.25)
            style.window_rounding = 8.0
            style.child_rounding = 8.0
            style.frame_rounding = 6.0
            style.window_padding = (12, 12)
            style.frame_padding = (10, 8)
            style.item_spacing = (10, 8)
            imgui.style_colors_dark()  # basic palette that works across bindings
        except Exception as exc:
            print(f"[imgui] Failed to initialize ImGui overlay: {exc}")
            self._imgui_ready = False
            return

        self._imgui_ready = True
        self.accept("imgui-new-frame", self._draw_debug_ui)

    def _draw_debug_ui(self) -> None:
        """Render an on-screen debug window with state and controls."""
        if not self._imgui_ready:
            return

        pad = 14.0
        io = imgui.get_io()
        scr_w = io.display_size.x or 1920.0
        scr_h = io.display_size.y or 1080.0

        imgui.set_next_window_pos((pad, pad), imgui.Cond_.once)
        imgui.set_next_window_size((860, 520), imgui.Cond_.once)
        imgui.set_next_window_bg_alpha(0.92)
        imgui.begin("Debug")

        fps_text = self._avg_fps if self._avg_fps > 0.0 else imgui.get_io().framerate
        imgui.text(f"FPS: {fps_text:.1f}")

        imgui.separator()
        follow_count = len(self._follow_path_points)
        goal_path_count = 0
        if isinstance(self._last_path_data, dict):
            goal_path_count = len(self._parse_ros_path(self._last_path_data))

        imgui.text(f"Mode: {self.ui.mode}")
        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()
        imgui.text(
            f"Waypoints: {follow_count if self.ui.mode == 'Follow Mode' else goal_path_count}"
        )

        changed_nav, publish_nav = imgui.checkbox(
            "Publish goals/paths", self._nav_publishing_enabled
        )
        imgui.same_line()
        changed_move, move_robot = imgui.checkbox(
            "Control robot (cmd_vel)", self._move_robot
        )
        if changed_nav:
            self._nav_publishing_enabled = publish_nav
        if changed_move:
            self._set_move_mode(move_robot)

        imgui.spacing()
        imgui.begin_child("Nav", (0, 140), True)
        if self.ui.mode == "Goal Mode":
            imgui.text("Goal")
            ros_goal = self._panda_pose_to_ros_tuple(self._last_goal_pose)
            if ros_goal is not None:
                (gx, gy, gz), (gh, gp, gr) = ros_goal
                imgui.text(f"  pos (m, ROS): {gx:.2f}, {gy:.2f}, {gz:.2f}")
                imgui.text(f"  rpy (deg, ROS): {gh:.1f}, {gp:.1f}, {gr:.1f}")
            else:
                (gx, gy, gz), (gh, gp, gr) = self._last_goal_pose
                imgui.text(f"  pos (m): {gx:.2f}, {gy:.2f}, {gz:.2f}")
                imgui.text(f"  hpr (deg): {gh:.1f}, {gp:.1f}, {gr:.1f}")
        else:
            imgui.text("Follow path")
            imgui.text(f"  buffered poses: {follow_count}")
            if follow_count:
                fx, fy, fz = self._follow_path_points[-1][0]
                imgui.text(f"  tip (m): {fx:.2f}, {fy:.2f}, {fz:.2f}")
        imgui.end_child()

        imgui.spacing()
        imgui.columns(2, "pose_columns", False)
        imgui.text("Avatar pose (ROS frame)")
        av_ros_tuple = self._panda_pose_to_ros_tuple(self.avatar.get_pose())
        if av_ros_tuple is not None:
            (ax, ay, az), (ar, ap, ayaw) = av_ros_tuple
            imgui.text(f"  pos (m): {ax:.2f}, {ay:.2f}, {az:.2f}")
            imgui.text(f"  rpy (deg): {ar:.1f}, {ap:.1f}, {ayaw:.1f}")
        else:
            imgui.text("  waiting for pose")

        imgui.next_column()
        imgui.text("Robot pose (ROS frame)")
        if self._last_ros_pose is not None:
            (x, y, z), rpy = self._last_ros_pose
            imgui.text(f"  pos (m): {x:.2f}, {y:.2f}, {z:.2f}")
            if rpy is not None:
                roll, pitch, yaw = rpy
                imgui.text(f"  rpy (deg): {roll:.1f}, {pitch:.1f}, {yaw:.1f}")
        else:
            imgui.text("  waiting for /space_cobot/pose")
        imgui.columns(1)

        if av_ros_tuple is not None and self._last_ros_pose is not None:
            (ax, ay, az), _ = av_ros_tuple
            (rx, ry, rz), _ = self._last_ros_pose
            dx = ax - rx
            dy = ay - ry
            dz = az - rz
            dist = sqrt(dx * dx + dy * dy + dz * dz)
            imgui.spacing()
            imgui.text(f"Avatar-robot position error: {dist:.3f} m")

        imgui.end()

        # Top-right control window
        ctrl_w = 420.0
        ctrl_h = 320.0
        ctrl_x = max(pad, scr_w - ctrl_w - pad)
        ctrl_y = pad
        imgui.set_next_window_pos((ctrl_x, ctrl_y), imgui.Cond_.always)
        imgui.set_next_window_size((ctrl_w, ctrl_h), imgui.Cond_.once)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize,
        )

        avail = imgui.get_content_region_avail().x
        half = (avail - imgui.get_style().item_spacing.x) * 0.5
        btn_h = 56

        def _button(label: str, size, base_col, hover_col, active_col) -> bool:
            imgui.push_style_color(imgui.Col_.button, base_col)
            imgui.push_style_color(imgui.Col_.button_hovered, hover_col)
            imgui.push_style_color(imgui.Col_.button_active, active_col)
            pressed = imgui.button(label, size)
            imgui.pop_style_color(3)
            return pressed

        is_follow = self.ui.mode == "Follow Mode"

        # Active mode gets a brighter tint instead of a trailing check mark
        def _mode_colors(active: bool):
            if active:
                return (
                    (0.25, 0.55, 0.92, 1.0),
                    (0.30, 0.60, 0.98, 1.0),
                    (0.22, 0.48, 0.80, 1.0),
                )
            return (
                (0.24, 0.34, 0.48, 1.0),
                (0.28, 0.40, 0.56, 1.0),
                (0.22, 0.32, 0.44, 1.0),
            )

        f_base, f_hover, f_active = _mode_colors(is_follow)
        g_base, g_hover, g_active = _mode_colors(not is_follow)

        if _button("Follow", (half, btn_h), f_base, f_hover, f_active):
            self._activate_follow_mode()
        imgui.same_line()
        if _button("Goal", (half, btn_h), g_base, g_hover, g_active):
            self._activate_goal_mode()

        imgui.spacing()
        imgui.text_colored((0.82, 0.90, 1.00, 1.0), f"Active Mode: {self.ui.mode}")
        imgui.separator()

        btn_w = avail
        if _button(
            "Reset Orientation",
            (btn_w, btn_h),
            (0.26, 0.46, 0.68, 1.0),
            (0.30, 0.52, 0.78, 1.0),
            (0.24, 0.40, 0.60, 1.0),
        ):
            self._reset_avatar_orientation_to_robot()
        if _button(
            "Abort",
            (btn_w, btn_h),
            (0.70, 0.22, 0.22, 1.0),
            (0.78, 0.28, 0.28, 1.0),
            (0.60, 0.18, 0.18, 1.0),
        ):
            self._abort_to_robot_pose()

        imgui.end()

    def _reset_avatar_orientation_to_robot(self) -> None:
        """Align avatar orientation to the latest robot heading."""
        if self._last_robot_hpr is None:
            return
        h, p, r = self._last_robot_hpr
        self.avatar.set_hpr(h, p, r)

    def _activate_goal_mode(self) -> None:
        """Switch to goal mode and clear any pending follow path."""
        if self.ui.mode == "Goal Mode":
            return
        self.ui.set_mode("Goal Mode")
        self._follow_path_points.clear()
        # Seed last goal pose so we don't immediately publish until the avatar moves.
        self._last_goal_pose = self.avatar.get_pose()

    def _activate_follow_mode(self) -> None:
        """Switch to follow mode and seed the path with the current avatar pose."""
        if self.ui.mode == "Follow Mode":
            return
        self.ui.set_mode("Follow Mode")
        self._follow_path_points.clear()
        self._follow_path_points.append(self.avatar.get_pose())

    def _publish_goal_for_pose(
        self, pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ) -> None:
        """Publish a single goal for the provided Panda3D pose."""
        ros_pose = panda_pose_to_ros(pose)
        if ros_pose is None:
            return
        self._publish_ros_goal(ros_pose)
        self._last_goal_pose = pose

    def _publish_ros_goal(self, ros_pose: Dict[str, Dict[str, float]]) -> None:
        """Publish a goal when already in ROS coordinates."""
        msg = {"header": {"frame_id": "map"}, "pose": ros_pose}
        try:
            self.cmd_pub.publish(TOPIC_GOAL, msg)
        except Exception:
            pass

    def _set_avatar_to_robot_pose(
        self,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Snap avatar to latest robot pose without publishing."""
        if self._last_robot_pose_panda is None:
            return None
        pos, hpr = self._last_robot_pose_panda
        self.avatar.set_pos(pos[0], pos[1], pos[2])
        self.avatar.set_hpr(hpr[0], hpr[1], hpr[2])
        return self.avatar.get_pose()

    def _abort_to_robot_pose(self) -> None:
        """Stop the robot and align avatar to the freshest robot pose."""
        if self._abort_restore_task is not None:
            self.taskMgr.remove(self._abort_restore_task)
            self._abort_restore_task = None
        # Temporarily switch to follow mode to reuse the hold-path abort flow.
        if self.ui.mode != "Follow Mode":
            self._abort_restore_mode = self.ui.mode
            self._activate_follow_mode()
        else:
            self._abort_restore_mode = None
        # Immediately command zero twist to halt motion.
        self._publish_cmd_vel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # Mark for goal publish on the next pose sample (or now if available).
        self._pending_abort_goal = True
        self._maybe_finalize_abort()

    def _maybe_finalize_abort(self) -> None:
        """If an abort is pending and pose is available, snap and publish goal."""
        if not self._pending_abort_goal:
            return
        pose = self._set_avatar_to_robot_pose()
        if pose is None:
            return
        # Prefer publishing using the last ROS pose to avoid frame drift.
        ros_pose = None
        if self._last_ros_pose is not None and self._last_ros_orientation is not None:
            pos = self._last_ros_pose[0]
            ros_pose = {
                "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "orientation": {
                    "x": float(self._last_ros_orientation.get("x", 0.0)),
                    "y": float(self._last_ros_orientation.get("y", 0.0)),
                    "z": float(self._last_ros_orientation.get("z", 0.0)),
                    "w": float(self._last_ros_orientation.get("w", 1.0)),
                },
            }
            self._publish_ros_goal(ros_pose)
        else:
            self._publish_goal_for_pose(pose)

        if self.ui.mode == "Follow Mode":
            self._follow_path_points = []
        self._publish_hold_path(pose, ros_pose_override=ros_pose)
        # Ensure stop command accompanies the goal.
        self._publish_cmd_vel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._pending_abort_goal = False
        if self._abort_restore_mode is not None:
            self._abort_restore_task = self.taskMgr.doMethodLater(
                0.5, self._restore_abort_mode, "AbortRestoreMode"
            )

    def _restore_abort_mode(self, task: "PythonTask") -> int:
        """Return to the original UI mode after an abort delay."""
        if self._abort_restore_mode is not None:
            restored_mode = self._abort_restore_mode
            self.ui.set_mode(restored_mode)
            if restored_mode == "Goal Mode":
                # Seed last goal pose to avoid an immediate publish until movement.
                self._last_goal_pose = self.avatar.get_pose()
        self._abort_restore_mode = None
        self._abort_restore_task = None
        return Task.done

    def _maybe_sync_avatar_on_stop(self) -> None:
        """Teleport avatar to robot pose when incoming cmd_vel is zero."""
        payload = self.bus_sensors.get(TOPIC_CMD_VEL)
        is_zero = self._is_zero_cmd_vel(payload)
        if is_zero and not self._robot_stopped_last:
            self._set_avatar_to_robot_pose()
        self._robot_stopped_last = is_zero

    def _is_zero_cmd_vel(self, payload: dict, eps: float = 1e-4) -> bool:
        if not isinstance(payload, dict):
            return False
        lin = payload.get("linear") if isinstance(payload, dict) else None
        ang = payload.get("angular") if isinstance(payload, dict) else None
        try:
            lx = float(lin.get("x", 0.0))
            ly = float(lin.get("y", 0.0))
            lz = float(lin.get("z", 0.0))
            ax = float(ang.get("x", 0.0))
            ay = float(ang.get("y", 0.0))
            az = float(ang.get("z", 0.0))
        except Exception:
            return False
        return (
            abs(lx) <= eps
            and abs(ly) <= eps
            and abs(lz) <= eps
            and abs(ax) <= eps
            and abs(ay) <= eps
            and abs(az) <= eps
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Create and run the SpacebotLink application."""
    app = SpacebotLinkApp()
    app.run()


if __name__ == "__main__":
    main()
