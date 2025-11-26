from __future__ import annotations
from collections import deque
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

# Topics used by your ROS→ZMQ bridge
TOPIC_IMAGE = "/main_camera/image"
TOPIC_CAMINFO = "/main_camera/camera_info"
TOPIC_IMU = "/imu/data"
TOPIC_POSE = "/space_cobot/pose"
TOPIC_CMD_VEL = "/space_cobot/cmd_vel"
TOPIC_GOAL = "/nav6d/goal"
TOPIC_PATH = "/nav6d/planner/path"

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
        self._goal_publishing_enabled: bool = True

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
        self.ui: UI = UI(self)
        self._fps_samples = deque(maxlen=120)
        self._avg_fps: float = 0.0
        self._last_ros_pose: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None
        self._last_ros_orientation: Optional[Dict[str, float]] = None
        self._last_goal_pub_time: float = 0.0
        self._last_goal_pose: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None
        self._imgui_ready: bool = False

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
        if not self._goal_publishing_enabled:
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

    def _camera_task(self, task: "PythonTask") -> int:
        """Update the background texture with the latest camera frame."""
        rgb = self.bus_images.get_image_rgb(TOPIC_IMAGE)
        if rgb is not None:
            h, w = rgb.shape[:2]
            if self.bg_tex.getXSize() != w or self.bg_tex.getYSize() != h:
                self.bg_tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
            self.bg_tex.setRamImageAs(rgb.tobytes(), "RGB")
        return Task.cont

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
                    self.camera.setPos(self.render, pos[0], pos[1], pos[2])
                    self.camera.setHpr(self.render, hpr[0], hpr[1], hpr[2])
                    self._last_cam_pos_hpr = (pos, hpr)
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
                self.avatar.move_world(move.x, move.y, move.z)

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
        payload = self.bus_sensors.get(TOPIC_PATH)
        if not isinstance(payload, dict):
            return Task.cont
        if payload == self._last_path_data:
            return Task.cont
        self._last_path_data = payload

        poses = self._parse_ros_path(payload)
        self._clear_path_markers()
        if not poses:
            return Task.cont

        if self._path_proto is None:
            self._path_proto = self._load_path_proto()
        proto = self._path_proto
        if proto is None:
            return Task.cont

        for idx, (pos, hpr) in enumerate(poses):
            if idx % 4 != 0 or idx == 0:
                continue
            ghost = proto.copyTo(self.render)
            ghost.setPos(self.render, pos[0], pos[1], pos[2])
            ghost.setHpr(self.render, hpr[0], hpr[1], hpr[2])
            # Draw markers behind the avatar overlays.
            ghost.setBin("fixed", 5)
            ghost.setDepthWrite(False)
            self._path_markers.append(ghost)
        return Task.cont

    # ---- debug UI ----
    def _init_imgui(self) -> None:
        """Initialize the ImGui overlay if available."""
        try:
            p3dimgui.init()
            style = imgui.get_style()
            style.font_size_base = 22.0  # request larger base font size
            style.font_scale_main = 1.25
            style.scale_all_sizes(1.25)
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

        imgui.set_next_window_size((520, 260), imgui.Cond_.once)
        imgui.set_next_window_bg_alpha(0.92)
        imgui.begin("Debug")

        fps_text = self._avg_fps if self._avg_fps > 0.0 else imgui.get_io().framerate
        imgui.text(f"FPS: {fps_text:.1f}")

        imgui.spacing()
        imgui.text("Avatar pose")
        av_pos, av_hpr = self.avatar.get_pose()
        av_ros = panda_pose_to_ros((av_pos, av_hpr))
        if av_ros is not None:
            pos_ros = av_ros["position"]
            imgui.text(
                f"  position (m):  {pos_ros['x']:.2f}, {pos_ros['y']:.2f}, {pos_ros['z']:.2f}"
            )
            rpy_ros = self._quat_to_rpy_deg(
                av_ros["orientation"]["x"],
                av_ros["orientation"]["y"],
                av_ros["orientation"]["z"],
                av_ros["orientation"]["w"],
            )
            if rpy_ros is not None:
                roll, pitch, yaw = rpy_ros
                imgui.text(
                    f"  orientation (deg): roll {roll:.1f}, pitch {pitch:.1f}, yaw {yaw:.1f}"
                )
        else:
            imgui.text("  waiting for avatar pose")

        imgui.spacing()
        imgui.text("Robot pose (/space_cobot/pose)")
        if self._last_ros_pose is not None:
            (x, y, z), rpy = self._last_ros_pose
            imgui.text(f"  position (m):  {x:.2f}, {y:.2f}, {z:.2f}")
            if rpy is not None:
                roll, pitch, yaw = rpy
                imgui.text(
                    f"  orientation (deg): roll {roll:.1f}, pitch {pitch:.1f}, yaw {yaw:.1f}"
                )
        else:
            imgui.text("  waiting for /space_cobot/pose")

        if av_ros is not None and self._last_ros_pose is not None:
            dx = pos_ros["x"] - self._last_ros_pose[0][0]
            dy = pos_ros["y"] - self._last_ros_pose[0][1]
            dz = pos_ros["z"] - self._last_ros_pose[0][2]
            dist = sqrt(dx * dx + dy * dy + dz * dz)
            imgui.spacing()
            imgui.text(f"Avatar-robot position error: {dist:.3f} m")

        imgui.separator()
        changed_move, move_robot = imgui.checkbox(
            "Control robot (publish cmd_vel)", self._move_robot
        )
        if changed_move:
            self._set_move_mode(move_robot)

        changed_goal, publish_goals = imgui.checkbox(
            "Publish goals", self._goal_publishing_enabled
        )
        if changed_goal:
            self._goal_publishing_enabled = publish_goals

        imgui.end()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Create and run the SpacebotLink application."""
    app = SpacebotLinkApp()
    app.run()


if __name__ == "__main__":
    main()
