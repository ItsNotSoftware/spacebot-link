from __future__ import annotations

from collections import deque
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence, List, Tuple

from panda3d.core import loadPrcFileData, PythonTask
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

import p3dimgui
from imgui_bundle import imgui

from config import (
    FRAMEBUFFER_SRGB_CFG,
    TRANSPARENCY_SORT_CFG,
    WINDOW_TITLE,
    TOPIC_CMD_VEL,
    TOPIC_GOAL,
    TOPIC_IMAGE,
    TOPIC_PATH,
    TOPIC_POSE,
    TOPIC_PATH_QUALITY,
    TOPIC_PATH_EXEC_SUMMARY_VEL,
    TOPIC_PATH_EXEC_SUMMARY_FORCE,
    PATH_QUALITY_CSV_ENABLED,
    default_cmd_endpoint,
    default_gltf_model,
    default_image_endpoint,
    default_sensor_endpoint,
    OCTOMAP_SERVER_BIN,
    OCTOMAP_SERVER_ENDPOINT,
    OCTOMAP_SERVER_MAP,
    OCTOMAP_SERVER_MAX_RANGE,
    OCTOMAP_QUERY_PERIOD_S,
    AVATAR_COLOR_VISIBLE,
    AVATAR_COLOR_OCCLUDED,
    AVATAR_COLOR_IN_OBSTACLE,
    FLOOR_PROJECTION_ENABLED,
    AVATAR_AUTO_RESET_DISTANCE,
    AVATAR_AUTO_RESET_DELAY_S,
    AVATAR_HIDE_DISTANCE,
    GAMEPAD_REMOTE_AUTOSTART,
    GAMEPAD_REMOTE_ENDPOINT,
    GAMEPAD_REMOTE_TOPIC,
)
from utils import (
    extract_ros_pose,
    is_zero_cmd_vel,
    panda_pose_to_ros_tuple,
    panda_pose_to_ros,
    parse_ros_path,
    ros_position_to_panda_pos,
    ros_pose_to_panda_pos_hpr,
    ros_vector_to_panda,
)
from input_controller import InputController
from navigation import Navigation
from renderer import Renderer
from teleop_bus import TeleopBusPub, TeleopBusSub
from ui import UI
import zmq

# ---- config before ShowBase ----
loadPrcFileData("", f"window-title {WINDOW_TITLE}")
loadPrcFileData("", FRAMEBUFFER_SRGB_CFG)
loadPrcFileData("", TRANSPARENCY_SORT_CFG)

# Register GLTF loader if available
try:
    import importlib

    _gltf_mod = importlib.import_module("panda3d_gltf")
    getattr(_gltf_mod, "GLTFLoader").register_loader()
except Exception:
    pass


class SpacebotLinkApp(ShowBase):
    def __init__(
        self,
        sensor_endpoint: str = default_sensor_endpoint,
        image_endpoint: str = default_image_endpoint,
        gltf_model: str = default_gltf_model,
        cmd_endpoint: str = default_cmd_endpoint,
    ) -> None:
        """Initialize app wiring, assets, networking, and tasks."""
        super().__init__()

        # buses
        self.bus_sensors = TeleopBusSub(sensor_endpoint, rcv_hwm=200)
        self.bus_images = TeleopBusSub(image_endpoint, rcv_hwm=1, conflate=True)
        self.cmd_pub = TeleopBusPub(cmd_endpoint)

        # rendering + navigation + input
        self.renderer = Renderer(self, gltf_model)
        # Expose for UI FOV bumps
        self._update_bg_scale = self.renderer.update_bg_scale
        self.nav = Navigation(self.cmd_pub, self.taskMgr)
        self.input = InputController(
            self,
            self.renderer,
            self.nav,
            self.cmd_pub,
            on_abort=self._abort_to_robot_pose,
            on_toggle_mode=self._toggle_ui_mode,
        )

        # metrics
        self._fps_samples = deque(maxlen=120)
        self._avg_fps: float = 0.0
        self._robot_stopped_last: bool = False
        self._avatar_auto_reset_pending_since: Optional[float] = None
        self._avatar_auto_reset_done: bool = False
        self._avatar_spawned_from_pose: bool = False
        self._last_path_poses: List = []
        self._last_floor_height: Optional[str] = None
        self._floor_projection_enabled: bool = bool(FLOOR_PROJECTION_ENABLED)
        self._last_path_quality: Optional[Dict[str, Any]] = None
        self._last_exec_summary_by_topic: Dict[str, Dict[str, Any]] = {}
        self._octomap_proc: Optional[subprocess.Popen] = None
        self._octomap_socket: Optional[zmq.Socket] = None
        self._gamepad_proc: Optional[subprocess.Popen] = None
        self._octomap_context = zmq.Context.instance()
        self._last_octomap: Optional[Dict[str, Any]] = None
        self._last_octomap_raw: Optional[str] = None
        self._last_avatar_state: Optional[Tuple[Optional[bool], Optional[bool]]] = None

        # UI + status
        self.ui = UI(self, self._collect_status, on_abort=self._abort_to_robot_pose)

        # tasks
        self.taskMgr.add(self._bus_task, "BusTask")
        self.taskMgr.add(self._camera_task, "CameraTask")
        self.taskMgr.add(self._pose_task, "PoseTask")
        self.taskMgr.add(self._keyboard_task, "KeyboardTask")
        self.taskMgr.add(self._orientation_preview_task, "OrientationPreviewTask")
        self.taskMgr.add(self._metrics_task, "MetricsTask")
        self.taskMgr.add(self._goal_publish_task, "GoalPublishTask")
        self.taskMgr.add(self._path_task, "PathTask")
        self.taskMgr.doMethodLater(
            OCTOMAP_QUERY_PERIOD_S, self._octomap_task, "OctomapTask"
        )
        self.taskMgr.doMethodLater(
            self.nav.follow_sample_period, self._follow_mode_tick, "FollowModeTick"
        )

        # cleanup
        self.exitFunc: Optional[callable] = self._cleanup

        self._start_octomap_process()
        self._init_octomap_socket()
        self._start_gamepad_remote_process()
        self.renderer.set_avatar_color(AVATAR_COLOR_VISIBLE)

    # ---- tasks ----
    def _bus_task(self, task: PythonTask) -> int:
        """Poll ZMQ sockets to keep sensor/image caches current."""
        self.bus_sensors.poll(100)
        self.bus_images.poll(100)
        self._maybe_log_path_quality_csv()
        return Task.cont

    def _camera_task(self, task: PythonTask) -> int:
        """Upload latest camera frame to the background texture."""
        rgb = self.bus_images.get_image_rgb(TOPIC_IMAGE)
        if rgb is not None:
            self.renderer.update_bg_frame(rgb)
        return Task.cont

    def _pose_task(self, task: PythonTask) -> int:
        """Track robot pose and drive camera/avatar sync logic."""
        payload = self.bus_sensors.get(TOPIC_POSE)
        if not isinstance(payload, dict):
            return Task.cont

        ori = None
        ros_pose = extract_ros_pose(payload)
        if ros_pose is not None:
            pos_r, rpy, ori = ros_pose
            self.nav.state.last_ros_pose = (pos_r, rpy)
            self.nav.state.last_ros_orientation = ori

        parsed_pos_hpr = ros_pose_to_panda_pos_hpr(payload)
        if parsed_pos_hpr is not None:
            pos, hpr = parsed_pos_hpr
            self.nav.update_robot_pose(
                parsed_pos_hpr, self.nav.state.last_ros_pose, ori
            )
            self.renderer.set_camera_pose(pos, hpr)
            if not self._avatar_spawned_from_pose:
                self.renderer.sync_avatar_to_robot(parsed_pos_hpr)
                self._avatar_spawned_from_pose = True
            self._update_avatar_visibility(parsed_pos_hpr)
        self._maybe_sync_avatar_on_stop()
        return Task.cont

    def _start_octomap_process(self) -> None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        bin_path = os.path.abspath(os.path.join(root_dir, OCTOMAP_SERVER_BIN))
        map_path = os.path.abspath(os.path.join(root_dir, OCTOMAP_SERVER_MAP))
        if not os.path.isfile(bin_path):
            print(f"[octomap] Binary not found: {bin_path}")
            return
        if not os.path.isfile(map_path):
            print(f"[octomap] Map not found: {map_path}")
            return

        args = [
            bin_path,
            map_path,
            OCTOMAP_SERVER_ENDPOINT,
            str(OCTOMAP_SERVER_MAX_RANGE),
        ]
        if self._octomap_proc is not None and self._octomap_proc.poll() is None:
            return
        try:
            self._octomap_proc = subprocess.Popen(
                args,
                cwd=os.path.dirname(bin_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as ex:
            print(f"[octomap] Failed to start process: {ex}")

    def _start_gamepad_remote_process(self) -> None:
        if not (
            GAMEPAD_REMOTE_AUTOSTART or os.getenv("GAMEPAD_REMOTE_AUTOSTART") == "1"
        ):
            return
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script_path = os.path.abspath(
            os.path.join(root_dir, "src", "gamepad_daemon.py")
        )
        if not os.path.isfile(script_path):
            print(f"[gamepad] daemon not found: {script_path}")
            return
        if self._gamepad_proc is not None and self._gamepad_proc.poll() is None:
            return
        env = os.environ.copy()
        env.setdefault("GAMEPAD_REMOTE", "1")
        env.setdefault("GAMEPAD_REMOTE_ENDPOINT", GAMEPAD_REMOTE_ENDPOINT)
        env.setdefault("GAMEPAD_REMOTE_TOPIC", GAMEPAD_REMOTE_TOPIC)
        try:
            self._gamepad_proc = subprocess.Popen(
                [sys.executable, script_path],
                cwd=os.path.dirname(script_path),
                env=env,
            )
        except Exception as ex:
            print(f"[gamepad] Failed to start remote process: {ex}")

    def _init_octomap_socket(self) -> None:
        sock = self._octomap_context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 60)
        sock.setsockopt(zmq.SNDTIMEO, 60)
        sock.setsockopt(zmq.REQ_RELAXED, 1)
        sock.setsockopt(zmq.REQ_CORRELATE, 1)
        sock.connect(OCTOMAP_SERVER_ENDPOINT)
        self._octomap_socket = sock

    def _octomap_task(self, task: PythonTask) -> int:
        if self._octomap_proc is not None:
            rc = self._octomap_proc.poll()
            if rc is not None:
                stdout, stderr = self._octomap_proc.communicate()
                if stdout:
                    print(f"[octomap] stdout:\n{stdout.strip()}")
                if stderr:
                    print(f"[octomap] stderr:\n{stderr.strip()}")
                print(f"[octomap] process exited with code {rc}")
                self._octomap_proc = None
                return Task.again
        if self._octomap_socket is None:
            return Task.again

        robot_pose_panda = self.nav.state.last_robot_pose_panda
        if robot_pose_panda is None:
            return Task.again

        robot_pose_ros = panda_pose_to_ros(robot_pose_panda)
        avatar_pose_panda = self.renderer.get_avatar_pose()
        avatar_pose_ros = panda_pose_to_ros(avatar_pose_panda)
        if robot_pose_ros is None or avatar_pose_ros is None:
            return Task.again

        payload = {
            "type": "avatar_query",
            "robot_pose": robot_pose_ros,
            "avatar_pose": avatar_pose_ros,
        }

        try:
            self._octomap_socket.send_string(json.dumps(payload))
            response_json = self._octomap_socket.recv_string()
        except zmq.Again:
            return Task.again
        except Exception:
            return Task.again

        try:
            response = json.loads(response_json)
        except Exception:
            return Task.again

        self._last_octomap = response
        self._last_octomap_raw = response_json
        self._apply_octomap_response(response, avatar_pose_ros, robot_pose_panda)
        return Task.again

    def _apply_octomap_response(
        self,
        response: Dict[str, Any],
        avatar_pose_ros: Dict[str, Dict[str, float]],
        robot_pose_panda: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> None:
        axis_map = {
            "x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
            "-z": (0.0, 0.0, -1.0),
        }

        axis_name = response.get("ground_axis")
        axis_ros = axis_map.get(axis_name)
        try:
            ground_distance = float(response.get("ground_distance"))
        except Exception:
            ground_distance = float("nan")

        if axis_ros is None or not math.isfinite(ground_distance):
            self.renderer.clear_floor_indicator()
            self._last_floor_height = None
            return

        avatar_pos_ros = avatar_pose_ros.get("position", {})
        try:
            ax = float(avatar_pos_ros.get("x"))
            ay = float(avatar_pos_ros.get("y"))
            az = float(avatar_pos_ros.get("z"))
        except Exception:
            self.renderer.clear_floor_indicator()
            self._last_floor_height = None
            return

        shadow_pos_ros = (
            ax + axis_ros[0] * ground_distance,
            ay + axis_ros[1] * ground_distance,
            az + axis_ros[2] * ground_distance,
        )

        avatar_pos_panda = ros_position_to_panda_pos({"x": ax, "y": ay, "z": az})
        shadow_pos_panda = ros_position_to_panda_pos(
            {"x": shadow_pos_ros[0], "y": shadow_pos_ros[1], "z": shadow_pos_ros[2]}
        )
        if avatar_pos_panda is None or shadow_pos_panda is None:
            self.renderer.clear_floor_indicator()
            self._last_floor_height = None
            return

        axis_panda = ros_vector_to_panda(axis_ros)
        if self._floor_projection_enabled:
            (rx, ry, rz), _ = robot_pose_panda
            dx = avatar_pos_panda[0] - rx
            dy = avatar_pos_panda[1] - ry
            dz = avatar_pos_panda[2] - rz
            dist_to_robot = (dx * dx + dy * dy + dz * dz) ** 0.5
            self.renderer.update_floor_indicator(
                avatar_pos_panda, shadow_pos_panda, axis_panda, dist_to_robot
            )
        else:
            self.renderer.clear_floor_indicator()

        self._last_floor_height = ground_distance

        def _parse_bool(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "false"):
                    return lowered == "true"
            return None

        occluded_bool = _parse_bool(response.get("avatar_occluded"))
        in_obstacle_bool = _parse_bool(response.get("avatar_in_obstacle"))

        desired_color = None
        if in_obstacle_bool is True:
            desired_color = AVATAR_COLOR_IN_OBSTACLE
        elif occluded_bool is True:
            desired_color = AVATAR_COLOR_OCCLUDED
        elif in_obstacle_bool is False or occluded_bool is False:
            desired_color = AVATAR_COLOR_VISIBLE

        state_key = (in_obstacle_bool, occluded_bool)
        if desired_color is not None and state_key != self._last_avatar_state:
            self.renderer.set_avatar_color(desired_color)
            self._last_avatar_state = state_key

    def _keyboard_task(self, task: PythonTask) -> int:
        """Handle keyboard/gamepad-driven avatar/robot control."""
        self.input.poll()
        return Task.cont

    def _orientation_preview_task(self, task: PythonTask) -> int:
        """Update the orientation preview models."""
        robot_ros_orientation = self.nav.state.last_ros_orientation
        avatar_pose = self.renderer.get_avatar_pose()
        avatar_hpr = avatar_pose[1] if avatar_pose else None
        self.renderer.update_orientation_preview(robot_ros_orientation, avatar_hpr)
        return Task.cont

    def _metrics_task(self, task: PythonTask) -> int:
        """Accumulate FPS samples for display."""
        dt = self.taskMgr.globalClock.getDt()
        if dt > 1e-6:
            self._fps_samples.append(1.0 / dt)
        self._avg_fps = (
            (sum(self._fps_samples) / len(self._fps_samples))
            if self._fps_samples
            else 0.0
        )
        return Task.cont

    def _goal_publish_task(self, task: PythonTask) -> int:
        """Publish nav goal when in Goal Mode and pose changed."""
        if self.ui.mode != "Goal Mode":
            return Task.cont
        if not self.nav.state.nav_publishing_enabled:
            return Task.cont
        if self.input.is_robot_mode():
            return Task.cont

        pose = self.renderer.get_avatar_pose()
        if not self.nav.pose_changed_since_last_goal(pose):
            return Task.cont
        self.nav.publish_goal_for_pose(pose)
        return Task.cont

    def _follow_mode_tick(self, task: PythonTask) -> int:
        """Sample avatar pose and publish follow path while in Follow Mode."""
        if self.input.is_robot_mode():
            task.delayTime = self.nav.follow_sample_period
            return Task.again
        self.nav.follow_tick(self.renderer)
        task.delayTime = self.nav.follow_sample_period
        return Task.again

    def _path_task(self, task: PythonTask) -> int:
        """Render path markers from nav path or follow buffer."""
        if self.ui.mode == "Follow Mode":
            self.renderer.render_path_markers(self.nav.state.follow_path_points)
            return Task.cont

        payload = self.bus_sensors.get(TOPIC_PATH)
        if isinstance(payload, dict) and payload != getattr(
            self, "_last_path_data", None
        ):
            self._last_path_data = payload
            poses = parse_ros_path(payload)
            poses = self._prepend_robot_pose_if_needed(poses)
            self.renderer.render_path_markers(poses)
            self._last_path_poses = poses
        return Task.cont

    def _maybe_log_path_quality_csv(self) -> None:
        """Print CSV line when execution summary arrives, using cached quality."""
        if not PATH_QUALITY_CSV_ENABLED:
            return

        payload = self.bus_sensors.get(TOPIC_PATH_QUALITY)
        if isinstance(payload, dict) and payload != self._last_path_quality:
            self._last_path_quality = payload

        for topic in (
            TOPIC_PATH_EXEC_SUMMARY_VEL,
            TOPIC_PATH_EXEC_SUMMARY_FORCE,
        ):
            summary = self.bus_sensors.get(topic)
            if not isinstance(summary, dict):
                continue
            if self._last_exec_summary_by_topic.get(topic) == summary:
                continue
            self._last_exec_summary_by_topic[topic] = summary
            if not self._last_path_quality:
                continue
            header, line = self._format_path_quality_csv(
                self._last_path_quality, summary
            )
            if line:
                print(header)
                print(line)

    @staticmethod
    def _format_path_quality_csv(
        quality: Dict[str, Any], summary: Dict[str, Any]
    ) -> tuple[str, str]:
        def _float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        clearance = _float(quality.get("clearance_score"))
        narrow = _float(quality.get("narrow_score"))
        turn = _float(quality.get("turn_score"))
        efficiency = _float(quality.get("efficiency_score"))
        planned = _float(summary.get("planned_length"))
        executed = _float(summary.get("executed_length"))
        rms = _float(summary.get("rms_tracking_error"))

        def _int(value: Any) -> int:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

        nrc = _int(summary.get("nrc", quality.get("nrc", 0)))
        mev = _float(summary.get("mev", quality.get("mev", 0.0)))

        width = 12
        labels = ("CLR", "NAR", "TRN", "EFF", "PLN", "EXE", "RMS", "NRC", "MEV")
        header = "".join(f"{label:>{width}}" for label in labels)
        values = (
            f"{clearance:>{width}.5f}",
            f"{narrow:>{width}.5f}",
            f"{turn:>{width}.5f}",
            f"{efficiency:>{width}.5f}",
            f"{planned:>{width}.5f}",
            f"{executed:>{width}.5f}",
            f"{rms:>{width}.5f}",
            f"{nrc:>{width}d}",
            f"{mev:>{width}.5f}",
        )
        line = "".join(values)
        return header, line

    def _prepend_robot_pose_if_needed(self, poses: List) -> List:
        """Ensure path lines originate at the current robot pose."""
        if not poses:
            return poses
        robot_pose = self.nav.state.last_robot_pose_panda
        if robot_pose is None:
            return poses
        (rx, ry, rz), _ = robot_pose
        (px, py, pz), _ = poses[0]
        dist = ((px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2) ** 0.5
        if dist > 1e-3:
            return [robot_pose] + poses
        return poses

    # ---- helpers ----
    def _collect_status(self) -> Dict[str, Any]:
        """Assemble a status snapshot for the overlay."""
        avatar_pose = self.renderer.get_avatar_pose()
        avatar_ros = panda_pose_to_ros_tuple(avatar_pose)
        robot_ros = self.nav.state.last_ros_pose
        pos_err = None
        if avatar_ros is not None and robot_ros is not None:
            (ax, ay, az), _ = avatar_ros
            (rx, ry, rz), _ = robot_ros
            dx, dy, dz = ax - rx, ay - ry, az - rz
            pos_err = (dx * dx + dy * dy + dz * dz) ** 0.5

        follow_tip = None
        if self.nav.state.follow_path_points:
            follow_tip = self.nav.state.follow_path_points[-1][0]

        return {
            "fps": self._avg_fps if self._avg_fps > 0.0 else imgui.get_io().framerate,
            "mode": self.ui.mode,
            "move_robot": self.input.is_robot_mode(),
            "nav_enabled": self.nav.state.nav_publishing_enabled,
            "waypoint_count": len(self.nav.state.follow_path_points),
            "follow_tip": follow_tip,
            "path_pose_count": len(self._last_path_poses),
            "last_goal_ros": (
                panda_pose_to_ros_tuple(self.nav.state.last_goal_pose)
                if self.nav.state.last_goal_pose
                else None
            ),
            "last_goal_panda": self.nav.state.last_goal_pose,
            "avatar_ros_pose": avatar_ros,
            "robot_ros_pose": robot_ros,
            "avatar_robot_error": pos_err,
            "floor_height": self._last_floor_height,
            "octomap_json": (
                self._last_octomap_raw[:2000] if self._last_octomap_raw else None
            ),
            "octomap_ground_distance": (
                self._last_octomap.get("ground_distance")
                if self._last_octomap
                else None
            ),
            "octomap_ground_axis": (
                self._last_octomap.get("ground_axis") if self._last_octomap else None
            ),
            "octomap_occluded": (
                self._last_octomap.get("avatar_occluded")
                if self._last_octomap
                else None
            ),
            "octomap_in_obstacle": (
                self._last_octomap.get("avatar_in_obstacle")
                if self._last_octomap
                else None
            ),
            "floor_projection_enabled": self._floor_projection_enabled,
            "set_floor_projection_enabled": self._set_floor_projection_enabled,
            "set_nav_enabled": self._set_nav_enabled,
            "set_move_mode": self.input.set_move_mode,
            "activate_follow": self._activate_follow_mode,
            "activate_goal": self._activate_goal_mode,
            "reset_orientation": self.renderer.reset_avatar_to_camera_hpr,
            "nudge_avatar_hpr": self.renderer.add_avatar_hpr,
            "path_mode": self.renderer.path_mode,
            "pose_stride": self.renderer.pose_stride,
            "line_stride": self.renderer.line_stride,
            "anim_speed": self.renderer.anim_speed,
            "anim_instances": self.renderer.anim_instances,
            "anim_line_enabled": self.renderer.anim_line_enabled,
            "set_path_mode": self._set_path_mode,
            "set_pose_stride": self._set_pose_stride,
            "set_line_stride": self._set_line_stride,
            "set_anim_speed": self._set_anim_speed,
            "set_anim_instances": self._set_anim_instances,
            "set_anim_line_enabled": self._set_anim_line_enabled,
        }

    def _set_nav_enabled(self, enabled: bool) -> None:
        """Toggle publishing of goals/paths."""
        self.input.set_nav_publish_preference(enabled)

    def _set_floor_projection_enabled(self, enabled: bool) -> None:
        """Toggle floor projection rendering."""
        self._floor_projection_enabled = bool(enabled)
        if not self._floor_projection_enabled:
            self.renderer.clear_floor_indicator()

    def _activate_goal_mode(self) -> None:
        """Switch to goal mode and seed last goal pose."""
        if self.ui.mode == "Goal Mode":
            return
        self.ui.set_mode("Goal Mode")
        self.nav.set_mode("Goal Mode")
        self.nav.state.last_goal_pose = self.renderer.get_avatar_pose()
        self._rerender_path()

    def _activate_follow_mode(self) -> None:
        """Switch to follow mode and seed path with avatar pose."""
        if self.ui.mode == "Follow Mode":
            return
        self.ui.set_mode("Follow Mode")
        self.nav.set_mode("Follow Mode")

    def _toggle_ui_mode(self) -> None:
        """Toggle between Goal and Follow modes."""
        if self.ui.mode == "Goal Mode":
            self._activate_follow_mode()
        else:
            self._activate_goal_mode()
        self.nav.set_follow_seed(self.renderer.get_avatar_pose())
        self._last_path_poses = []
        self._rerender_path()

    def _abort_to_robot_pose(self) -> None:
        """Abort motion by snapping avatar to robot pose and sending hold path."""
        self._publish_cmd_vel_zero()
        self.nav.abort_to_robot_pose(self.renderer)
        self._publish_cmd_vel_zero()
        self._rerender_path()

    def _publish_cmd_vel_zero(self) -> None:
        """Publish a zero-velocity command."""
        try:
            self.cmd_pub.publish(
                TOPIC_CMD_VEL,
                {
                    "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
            )
        except Exception:
            pass

    def _maybe_sync_avatar_on_stop(self) -> None:
        """Teleport avatar to robot pose when incoming cmd_vel hits zero."""
        payload = self.bus_sensors.get(TOPIC_CMD_VEL)
        is_zero = is_zero_cmd_vel(payload)
        if self.input.is_robot_mode():
            self._avatar_auto_reset_pending_since = None
            self._avatar_auto_reset_done = False
            self._robot_stopped_last = is_zero
            return

        if not is_zero:
            self._avatar_auto_reset_pending_since = None
            self._avatar_auto_reset_done = False
            self._robot_stopped_last = False
            return

        if self._avatar_auto_reset_done:
            self._robot_stopped_last = True
            return

        if self.nav.state.last_robot_pose_panda is None:
            self._avatar_auto_reset_pending_since = None
            self._robot_stopped_last = True
            return

        if self.input.is_any_input_active():
            self._avatar_auto_reset_pending_since = None
            self._robot_stopped_last = True
            return

        avatar_pos, _ = self.renderer.get_avatar_pose()
        (rx, ry, rz), _ = self.nav.state.last_robot_pose_panda
        dx = avatar_pos[0] - rx
        dy = avatar_pos[1] - ry
        dz = avatar_pos[2] - rz
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist > AVATAR_AUTO_RESET_DISTANCE:
            self._avatar_auto_reset_pending_since = None
            self._robot_stopped_last = True
            return

        now = self.taskMgr.globalClock.getFrameTime()
        if self._avatar_auto_reset_pending_since is None:
            self._avatar_auto_reset_pending_since = now
            self._robot_stopped_last = True
            return

        if (now - self._avatar_auto_reset_pending_since) >= AVATAR_AUTO_RESET_DELAY_S:
            self.renderer.sync_avatar_to_robot(self.nav.state.last_robot_pose_panda)
            self._avatar_auto_reset_done = True
            self._avatar_auto_reset_pending_since = None
        self._robot_stopped_last = True

    def _update_avatar_visibility(self, robot_pose: Tuple) -> None:
        """Hide avatar when it's very close to the robot pose."""
        if self.input.is_robot_mode():
            return
        if robot_pose is None:
            return
        avatar_pos, _ = self.renderer.get_avatar_pose()
        (rx, ry, rz), _ = robot_pose
        dx = avatar_pos[0] - rx
        dy = avatar_pos[1] - ry
        dz = avatar_pos[2] - rz
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        self.renderer.set_avatar_visible(dist > AVATAR_HIDE_DISTANCE)

    def _rerender_path(self) -> None:
        """Force a re-render of the current path with latest viz settings."""
        if self.ui.mode == "Follow Mode":
            self.renderer.render_path_markers(self.nav.state.follow_path_points)
        elif self._last_path_poses:
            self.renderer.render_path_markers(self._last_path_poses)

    # ---- path viz setters ----
    def _set_path_mode(self, mode: str) -> None:
        self.renderer.set_path_mode(mode)
        self._rerender_path()

    def _set_pose_stride(self, stride: int) -> None:
        self.renderer.set_pose_stride(stride)
        self._rerender_path()

    def _set_line_stride(self, stride: int) -> None:
        self.renderer.set_line_stride(stride)
        self._rerender_path()

    def _set_anim_speed(self, speed: float) -> None:
        self.renderer.set_anim_speed(speed)
        self._rerender_path()

    def _set_anim_instances(self, count: int) -> None:
        self.renderer.set_anim_instances(count)
        self._rerender_path()

    def _set_anim_line_enabled(self, enabled: bool) -> None:
        self.renderer.set_anim_line_enabled(enabled)
        self._rerender_path()

    # ---- cleanup ----
    def _cleanup(self) -> None:
        self.ui.save_imgui_settings()
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
        try:
            if self._octomap_socket is not None:
                self._octomap_socket.close(0)
        except Exception:
            pass
        try:
            if self._octomap_proc is not None:
                self._octomap_proc.terminate()
        except Exception:
            pass
        try:
            if self._gamepad_proc is not None:
                self._gamepad_proc.terminate()
        except Exception:
            pass
        self.renderer.clear_path_markers()


def main(argv: Optional[Sequence[str]] = None) -> None:
    app = SpacebotLinkApp()
    app.run()


if __name__ == "__main__":
    main()
