from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import p3dimgui
import zmq
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from imgui_bundle import imgui
from panda3d.core import PythonTask, Vec3, loadPrcFileData

from config import (
    AVATAR_AUTO_RESET_DELAY_S,
    AVATAR_AUTO_RESET_DISTANCE,
    AVATAR_COLOR_IN_OBSTACLE,
    AVATAR_COLOR_OCCLUDED,
    AVATAR_COLOR_VISIBLE,
    AVATAR_HIDE_DISTANCE,
    FLOOR_PROJECTION_ENABLED,
    FRAMEBUFFER_SRGB_CFG,
    GAMEPAD_REMOTE_ENDPOINT,
    GAMEPAD_REMOTE_TOPIC,
    ISS_MODULE_DETECT_PERIOD_S,
    ISS_MODULE_POINTS_YAML,
    OCTOMAP_QUERY_PERIOD_S,
    OCTOMAP_SERVER_BIN,
    OCTOMAP_SERVER_ENDPOINT,
    OCTOMAP_SERVER_MAP,
    OCTOMAP_SERVER_MAX_RANGE,
    REMOTE_INPUT_DAEMON,
    SPACEMOUSE_REMOTE_ENDPOINT,
    SPACEMOUSE_REMOTE_TOPIC,
    TOPIC_CMD_PATH,
    TOPIC_CMD_VEL,
    TOPIC_GOAL,
    TOPIC_IMAGE,
    TOPIC_PATH,
    TOPIC_PATH_EXEC_SUMMARY_FORCE,
    TOPIC_PATH_EXEC_SUMMARY_VEL,
    TOPIC_PATH_QUALITY,
    TOPIC_POSE,
    TRANSPARENCY_SORT_CFG,
    UI_RESPONSE_DELAY_FILL_S,
    WINDOW_SIZE_PX,
    WINDOW_TITLE,
    default_cmd_endpoint,
    default_gltf_model,
    default_image_endpoint,
    default_sensor_endpoint,
)
from input_controller import InputController
from navigation import Navigation
from renderer import Renderer
from teleop_bus import TeleopBusPub, TeleopBusSub
from ui import UI
from utils import (
    extract_ros_pose,
    is_zero_cmd_vel,
    panda_pose_to_ros,
    panda_pose_to_ros_tuple,
    parse_ros_path,
    ros_pose_to_panda_pos_hpr,
    ros_position_to_panda_pos,
    ros_vector_to_panda,
)

# ---- config before ShowBase ----
loadPrcFileData("", f"window-title {WINDOW_TITLE}")
loadPrcFileData("", f"win-size {int(WINDOW_SIZE_PX[0])} {int(WINDOW_SIZE_PX[1])}")
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
    _STATUS_CACHE_PERIOD_S = 0.05

    @staticmethod
    def _select_remote_input_daemon() -> str:
        value = os.getenv("REMOTE_INPUT_DAEMON", REMOTE_INPUT_DAEMON)
        selected = str(value).strip().lower()
        if selected in {"gamepad", "spacemouse", "both", "none"}:
            return selected
        print(
            f"[input] invalid REMOTE_INPUT_DAEMON={value!r}; "
            "expected one of: gamepad, spacemouse, both, none. Using 'gamepad'."
        )
        return "gamepad"

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
        self._fps_sum: float = 0.0
        self._avg_fps: float = 0.0
        self._robot_stopped_last: bool = False
        self._avatar_auto_reset_pending_since: Optional[float] = None
        self._avatar_auto_reset_pending_pose: Optional[Tuple[float, float, float]] = (
            None
        )
        self._avatar_auto_reset_done: bool = False
        self._avatar_spawned_from_pose: bool = False
        self._last_path_poses: List = []
        self._last_floor_height: Optional[str] = None
        self._floor_projection_enabled: bool = bool(FLOOR_PROJECTION_ENABLED)
        self._last_path_quality: Optional[Dict[str, Any]] = None
        self._octomap_proc: Optional[subprocess.Popen] = None
        self._octomap_socket: Optional[zmq.Socket] = None
        self._gamepad_proc: Optional[subprocess.Popen] = None
        self._spacemouse_proc: Optional[subprocess.Popen] = None
        self._cleaned_up: bool = False
        self._octomap_context = zmq.Context.instance()
        self._last_octomap: Optional[Dict[str, Any]] = None
        self._last_octomap_raw: Optional[str] = None
        self._last_avatar_state: Optional[Tuple[Optional[bool], Optional[bool]]] = None
        self._last_path_line_style_refresh_s: float = 0.0
        self._avatar_delay_bar_pose: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None
        self._avatar_delay_bar_reset_s: float = time.monotonic()
        self._pending_cmd_sent_s: Dict[str, float] = {}
        self._last_cmd_latency_s: Optional[float] = None
        self._last_cmd_latency_label: Optional[str] = None
        self._last_cmd_sent_topic: Optional[str] = None
        self._last_cmd_sent_s: Optional[float] = None
        self._last_cmd_ack_s: Optional[float] = None
        self._last_cmd_vel_echo_payload: Any = None
        self._last_exec_summary_vel: Any = None
        self._last_exec_summary_force: Any = None
        self._total_flight_length_m: float = 0.0
        self._last_distance_pose: Optional[Tuple[float, float, float]] = None
        self._operational_start_s: float = time.monotonic()
        self._iss_module_points: Dict[str, List[Tuple[float, float]]] = (
            self._load_iss_module_points()
        )
        self._current_iss_module: Optional[str] = None
        self._current_iss_module_dist_m: Optional[float] = None
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_next_s: float = 0.0
        self._direct_mode: bool = bool(self.input.is_robot_mode())
        self._floor_projection_before_direct_mode: bool = bool(
            self._floor_projection_enabled
        )
        self._install_cmd_publish_tracking()

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
            ISS_MODULE_DETECT_PERIOD_S,
            self._iss_module_task,
            "IssModuleTask",
        )
        self.taskMgr.doMethodLater(
            self.nav.follow_sample_period, self._follow_mode_tick, "FollowModeTick"
        )

        # cleanup
        self.exitFunc: Optional[callable] = self._cleanup

        self._start_octomap_process()
        self._init_octomap_socket()
        selected_daemon = self._select_remote_input_daemon()
        if selected_daemon in {"gamepad", "both"}:
            self._start_gamepad_remote_process()
        if selected_daemon in {"spacemouse", "both"}:
            self._start_spacemouse_remote_process()
        self.renderer.set_avatar_color(AVATAR_COLOR_VISIBLE)

    @staticmethod
    def _extract_path_goodness(payload: Dict[str, Any]) -> Optional[float]:
        """Read path-goodness from payload, with backward-compatible fallback."""
        raw = payload.get("path_goodness", payload.get("heuristic"))
        try:
            return max(0.0, min(1.0, float(raw)))
        except Exception:
            return None

    def _load_iss_module_points(self) -> Dict[str, List[Tuple[float, float]]]:
        """Load ROS XY reference points by ISS module from YAML."""
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cfg_path = os.path.abspath(os.path.join(root_dir, ISS_MODULE_POINTS_YAML))
        if not os.path.isfile(cfg_path):
            print(f"[module] module points file not found: {cfg_path}")
            return {}

        try:
            import yaml
        except Exception as exc:
            print(f"[module] PyYAML not available; module detection disabled: {exc}")
            return {}

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"[module] failed to read module points YAML: {exc}")
            return {}

        modules_obj = data.get("modules") if isinstance(data, dict) else None
        if not isinstance(modules_obj, dict):
            return {}

        out: Dict[str, List[Tuple[float, float]]] = {}
        for module_name, module_data in modules_obj.items():
            if not isinstance(module_name, str):
                continue
            points_obj = None
            if isinstance(module_data, dict):
                points_obj = module_data.get("points")
            elif isinstance(module_data, list):
                points_obj = module_data
            if not isinstance(points_obj, list):
                continue

            parsed_points: List[Tuple[float, float]] = []
            for entry in points_obj:
                x = None
                y = None
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    x, y = entry[0], entry[1]
                elif isinstance(entry, dict):
                    x, y = entry.get("x"), entry.get("y")
                try:
                    xf = float(x)
                    yf = float(y)
                except Exception:
                    continue
                if math.isfinite(xf) and math.isfinite(yf):
                    parsed_points.append((xf, yf))
            if parsed_points:
                out[module_name] = parsed_points

        if not out:
            print(f"[module] no valid module points found in: {cfg_path}")
        return out

    def _update_current_iss_module(self) -> None:
        """Classify robot module by nearest configured ROS XY point."""
        robot_pose = self.nav.state.last_ros_pose
        if robot_pose is None or not self._iss_module_points:
            self._current_iss_module = None
            self._current_iss_module_dist_m = None
            return

        (rx, ry, _rz), _ = robot_pose
        best_module = None
        best_dist = float("inf")
        for module_name, points in self._iss_module_points.items():
            for px, py in points:
                dist = math.hypot(float(rx) - px, float(ry) - py)
                if dist < best_dist:
                    best_dist = dist
                    best_module = module_name

        if best_module is None or not math.isfinite(best_dist):
            self._current_iss_module = None
            self._current_iss_module_dist_m = None
            return

        self._current_iss_module = best_module
        self._current_iss_module_dist_m = float(best_dist)

    def _install_cmd_publish_tracking(self) -> None:
        """Wrap the command publisher to timestamp outgoing commands for UI latency display."""
        orig_publish = self.cmd_pub.publish

        def _tracked_publish(topic: str, data: Dict[str, Any]) -> None:
            sent_s = time.monotonic()
            orig_publish(topic, data)
            self._record_cmd_sent(topic, sent_s)

        # Instance-level override so all existing users (nav/input/main) are tracked.
        self.cmd_pub.publish = _tracked_publish  # type: ignore[method-assign]

    def _record_cmd_sent(self, topic: str, sent_s: float) -> None:
        """Record an outgoing command send attempt."""
        key = self._cmd_pending_key(topic)
        self._pending_cmd_sent_s[key] = float(sent_s)
        self._last_cmd_sent_topic = topic
        self._last_cmd_sent_s = float(sent_s)

    def _cmd_pending_key(self, topic: str) -> str:
        if topic == TOPIC_CMD_VEL:
            return "cmd_vel"
        if topic == TOPIC_GOAL:
            return "nav_goal"
        if topic == TOPIC_CMD_PATH:
            return "nav_path"
        return topic

    def _ack_pending_cmds(self, keys: Sequence[str], source_label: str) -> None:
        """Mark the newest pending command among `keys` as acknowledged and store latency."""
        newest_key = None
        newest_sent = -1.0
        for key in keys:
            sent = self._pending_cmd_sent_s.get(key)
            if sent is None:
                continue
            if sent > newest_sent:
                newest_sent = sent
                newest_key = key
        if newest_key is None or newest_sent < 0.0:
            return
        now_s = time.monotonic()
        self._last_cmd_latency_s = max(0.0, now_s - newest_sent)
        self._last_cmd_latency_label = source_label
        self._last_cmd_ack_s = now_s
        self._pending_cmd_sent_s.pop(newest_key, None)

    def _update_command_latency_tracking(self) -> None:
        """Detect command echoes/summaries and convert them into UI latency samples."""
        cmd_vel_echo = self.bus_sensors.get(TOPIC_CMD_VEL)
        if cmd_vel_echo is not None and cmd_vel_echo != self._last_cmd_vel_echo_payload:
            self._last_cmd_vel_echo_payload = cmd_vel_echo
            self._ack_pending_cmds(("cmd_vel",), "cmd_vel echo")

        exec_vel = self.bus_sensors.get(TOPIC_PATH_EXEC_SUMMARY_VEL)
        if exec_vel is not None and exec_vel != self._last_exec_summary_vel:
            self._last_exec_summary_vel = exec_vel
            self._ack_pending_cmds(("nav_goal", "nav_path"), "path exec")

        exec_force = self.bus_sensors.get(TOPIC_PATH_EXEC_SUMMARY_FORCE)
        if exec_force is not None and exec_force != self._last_exec_summary_force:
            self._last_exec_summary_force = exec_force
            self._ack_pending_cmds(("nav_goal", "nav_path"), "path exec")

        planner_path = self.bus_sensors.get(TOPIC_PATH)
        if planner_path is not None and planner_path != getattr(
            self, "_last_path_data", None
        ):
            # Planner path updates are often the earliest visible response to goal/path commands.
            self._ack_pending_cmds(("nav_goal", "nav_path"), "planner path")

    def _last_pending_command_age_s(self) -> Optional[float]:
        """Return age of newest still-pending command, if any."""
        if not self._pending_cmd_sent_s:
            return None
        newest = max(self._pending_cmd_sent_s.values())
        return max(0.0, time.monotonic() - newest)

    def _avatar_delay_fill_progress(
        self, avatar_pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ) -> float:
        """Return [0,1] progress that resets on avatar movement and fills over configured duration."""
        now_s = time.monotonic()
        prev = self._avatar_delay_bar_pose
        moved = False
        if prev is None:
            moved = True
        else:
            (px, py, pz), (ph, pp, pr) = prev
            (x, y, z), (h, p, r) = avatar_pose
            moved = (
                abs(x - px) > 1e-4
                or abs(y - py) > 1e-4
                or abs(z - pz) > 1e-4
                or abs(h - ph) > 1e-3
                or abs(p - pp) > 1e-3
                or abs(r - pr) > 1e-3
            )
        if moved:
            self._avatar_delay_bar_reset_s = now_s
            self._avatar_delay_bar_pose = avatar_pose
            return 0.0
        elapsed = max(0.0, now_s - self._avatar_delay_bar_reset_s)
        duration_s = max(0.1, float(UI_RESPONSE_DELAY_FILL_S))
        return max(0.0, min(1.0, elapsed / duration_s))

    # ---- tasks ----
    def _bus_task(self, task: PythonTask) -> int:
        """Poll ZMQ sockets to keep sensor/image caches current."""
        self.bus_sensors.poll(100)
        self.bus_images.poll(100)
        self._update_command_latency_tracking()
        payload = self.bus_sensors.get(TOPIC_PATH_QUALITY)
        if isinstance(payload, dict) and payload != self._last_path_quality:
            self._last_path_quality = payload
            try:
                self.renderer.set_path_goodness(self._extract_path_goodness(payload))
                self.renderer.set_path_local_risks(payload.get("local_risks"))
                self._refresh_path_line_style_if_needed()
            except Exception:
                pass
        return Task.cont

    def _camera_task(self, task: PythonTask) -> int:
        """Upload latest camera frame to the background texture."""
        rgb = self.bus_images.get_image_rgb(TOPIC_IMAGE)
        if rgb is not None:
            self.renderer.update_bg_frame(rgb)
        return Task.cont

    def _iss_module_task(self, task: PythonTask) -> int:
        """Refresh nearest ISS module classification at a fixed cadence."""
        self._update_current_iss_module()
        return Task.again

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
            last_pos = self._last_distance_pose
            if last_pos is not None:
                dx = float(pos[0]) - last_pos[0]
                dy = float(pos[1]) - last_pos[1]
                dz = float(pos[2]) - last_pos[2]
                step_m = (dx * dx + dy * dy + dz * dz) ** 0.5
                if math.isfinite(step_m):
                    self._total_flight_length_m += max(0.0, step_m)
            self._last_distance_pose = (float(pos[0]), float(pos[1]), float(pos[2]))
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

    def _start_spacemouse_remote_process(self) -> None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script_path = os.path.abspath(
            os.path.join(root_dir, "src", "spacemouse_daemon.py")
        )
        if not os.path.isfile(script_path):
            print(f"[spacemouse] daemon not found: {script_path}")
            return
        if self._spacemouse_proc is not None and self._spacemouse_proc.poll() is None:
            return
        env = os.environ.copy()
        env.setdefault("SPACEMOUSE_REMOTE_ENDPOINT", SPACEMOUSE_REMOTE_ENDPOINT)
        env.setdefault("SPACEMOUSE_REMOTE_TOPIC", SPACEMOUSE_REMOTE_TOPIC)
        # Backward-compatible env keys.
        env.setdefault("GAMEPAD_REMOTE_ENDPOINT", SPACEMOUSE_REMOTE_ENDPOINT)
        env.setdefault("GAMEPAD_REMOTE_TOPIC", SPACEMOUSE_REMOTE_TOPIC)
        try:
            self._spacemouse_proc = subprocess.Popen(
                [sys.executable, script_path],
                cwd=os.path.dirname(script_path),
                env=env,
            )
        except Exception as ex:
            print(f"[spacemouse] Failed to start remote process: {ex}")

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
        if self._direct_mode:
            self.renderer.clear_floor_indicator()
            return Task.again
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
        move_robot = bool(self.input.is_robot_mode())
        if move_robot != self._direct_mode:
            self._set_direct_mode(move_robot)
        return Task.cont

    def _orientation_preview_task(self, task: PythonTask) -> int:
        """Update the orientation preview models."""
        if self._direct_mode:
            return Task.cont
        robot_ros_orientation = self.nav.state.last_ros_orientation
        avatar_pose = self.renderer.get_avatar_pose()
        self.renderer.update_orientation_preview_motion_hint(avatar_pose)
        avatar_hpr = avatar_pose[1] if avatar_pose else None
        self.renderer.update_orientation_preview(robot_ros_orientation, avatar_hpr)
        return Task.cont

    def _metrics_task(self, task: PythonTask) -> int:
        """Accumulate FPS samples for display."""
        dt = self.taskMgr.globalClock.getDt()
        if dt > 1e-6:
            fps = 1.0 / dt
            if len(self._fps_samples) == self._fps_samples.maxlen:
                self._fps_sum -= self._fps_samples[0]
            self._fps_samples.append(fps)
            self._fps_sum += fps
        self._avg_fps = (
            (self._fps_sum / len(self._fps_samples)) if self._fps_samples else 0.0
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
        if self._direct_mode:
            self.renderer.clear_path_markers()
            return Task.cont
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

    @staticmethod
    def _normalize_vec3_tuple(
        vec: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        x, y, z = vec
        mag = (x * x + y * y + z * z) ** 0.5
        if not math.isfinite(mag) or mag <= 1e-6:
            return None
        return (x / mag, y / mag, z / mag)

    @staticmethod
    def _panda_vec_to_ros(vec: Vec3) -> Tuple[float, float, float]:
        """Convert Panda3D vector to ROS map frame vector."""
        return (float(vec.y), float(-vec.x), float(vec.z))

    def _control_axes_ros(
        self,
    ) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """Return camera-relative control axes expressed in ROS map frame."""
        frame = self.camera if self.camera is not None else self.render
        if frame is None or self.render is None:
            return None
        try:
            q = frame.getQuat(self.render)
        except Exception:
            return None

        forward = self._normalize_vec3_tuple(
            self._panda_vec_to_ros(q.xform(Vec3(0.0, 1.0, 0.0)))
        )
        right = self._normalize_vec3_tuple(
            self._panda_vec_to_ros(q.xform(Vec3(1.0, 0.0, 0.0)))
        )
        up = self._normalize_vec3_tuple(
            self._panda_vec_to_ros(q.xform(Vec3(0.0, 0.0, 1.0)))
        )
        if forward is None or right is None or up is None:
            return None
        return {"forward": forward, "right": right, "up": up}

    # ---- helpers ----
    def _collect_status(self) -> Dict[str, Any]:
        """Assemble a status snapshot for the overlay."""
        now_s = time.monotonic()
        if self._status_cache is not None and now_s < self._status_cache_next_s:
            status_cached = dict(self._status_cache)
            status_cached["operational_time_s"] = max(
                0.0, float(now_s - self._operational_start_s)
            )
            return status_cached

        avatar_pose = self.renderer.get_avatar_pose()
        avatar_ros = panda_pose_to_ros_tuple(avatar_pose)
        control_axes_ros = self._control_axes_ros()
        response_delay_fill = self._avatar_delay_fill_progress(avatar_pose)
        robot_ros = self.nav.state.last_ros_pose
        pos_err = None
        goal_delta_ros: Optional[Tuple[float, float, float]] = None
        goal_vertical_relation: Optional[str] = None
        if avatar_ros is not None and robot_ros is not None:
            (ax, ay, az), _ = avatar_ros
            (rx, ry, rz), _ = robot_ros
            dx, dy, dz = ax - rx, ay - ry, az - rz
            pos_err = (dx * dx + dy * dy + dz * dz) ** 0.5
            goal_delta_ros = (float(dx), float(dy), float(dz))
            if dz > 0.05:
                goal_vertical_relation = "above"
            elif dz < -0.05:
                goal_vertical_relation = "below"
            else:
                goal_vertical_relation = "level"

        follow_tip = None
        if self.nav.state.follow_path_points:
            follow_tip = self.nav.state.follow_path_points[-1][0]

        status = {
            "fps": self._avg_fps if self._avg_fps > 0.0 else imgui.get_io().framerate,
            "mode": self.ui.mode,
            "direct_mode": self._direct_mode,
            "move_robot": self.input.is_robot_mode(),
            "nav_enabled": self.nav.state.nav_publishing_enabled,
            "waypoint_count": len(self.nav.state.follow_path_points),
            "follow_tip": follow_tip,
            "path_pose_count": len(self._last_path_poses),
            "response_delay_fill": response_delay_fill,
            "response_delay_fill_s": float(UI_RESPONSE_DELAY_FILL_S),
            "total_flight_length_m": float(self._total_flight_length_m),
            "operational_time_s": max(
                0.0, float(time.monotonic() - self._operational_start_s)
            ),
            "current_iss_module": self._current_iss_module,
            "current_iss_module_distance_m": self._current_iss_module_dist_m,
            "last_cmd_latency_s": self._last_cmd_latency_s,
            "last_cmd_latency_label": self._last_cmd_latency_label,
            "last_cmd_pending_age_s": self._last_pending_command_age_s(),
            "path_quality": self._last_path_quality,
            "path_goodness": (
                self._extract_path_goodness(self._last_path_quality)
                if isinstance(self._last_path_quality, dict)
                else None
            ),
            "last_goal_ros": (
                panda_pose_to_ros_tuple(self.nav.state.last_goal_pose)
                if self.nav.state.last_goal_pose
                else None
            ),
            "last_goal_panda": self.nav.state.last_goal_pose,
            "avatar_ros_pose": avatar_ros,
            "robot_ros_pose": robot_ros,
            "avatar_robot_error": pos_err,
            "goal_delta_ros": goal_delta_ros,
            "goal_vertical_relation": goal_vertical_relation,
            "control_forward_ros": (
                control_axes_ros.get("forward") if control_axes_ros else None
            ),
            "control_right_ros": (
                control_axes_ros.get("right") if control_axes_ros else None
            ),
            "control_up_ros": control_axes_ros.get("up") if control_axes_ros else None,
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
            "set_direct_mode": self._set_direct_mode,
            "activate_follow": self._activate_follow_mode,
            "activate_goal": self._activate_goal_mode,
            "reset_orientation": self.renderer.reset_avatar_to_camera_hpr,
            "nudge_avatar_hpr": self.renderer.add_avatar_hpr,
            "path_mode": self.renderer.path_mode,
            "marker_spacing_m": self.renderer.marker_spacing_m,
            "anim_speed": self.renderer.anim_speed,
            "anim_instances": self.renderer.anim_instances,
            "anim_line_enabled": self.renderer.anim_line_enabled,
            "set_path_mode": self._set_path_mode,
            "set_marker_spacing": self._set_marker_spacing,
            "set_anim_speed": self._set_anim_speed,
            "set_anim_instances": self._set_anim_instances,
            "set_anim_line_enabled": self._set_anim_line_enabled,
            "reset_session_metrics": self._reset_session_metrics,
        }
        self._status_cache = status
        self._status_cache_next_s = now_s + float(self._STATUS_CACHE_PERIOD_S)
        return status

    def _reset_session_metrics(self) -> None:
        """Reset session meters: integrated distance and operational time."""
        self._total_flight_length_m = 0.0
        self._operational_start_s = time.monotonic()
        self._status_cache_next_s = 0.0
        robot_pose = self.nav.state.last_robot_pose_panda
        if robot_pose is None:
            self._last_distance_pose = None
            return
        (x, y, z), _ = robot_pose
        self._last_distance_pose = (float(x), float(y), float(z))

    def _set_nav_enabled(self, enabled: bool) -> None:
        """Toggle publishing of goals/paths."""
        self.input.set_nav_publish_preference(enabled)

    def _set_floor_projection_enabled(self, enabled: bool) -> None:
        """Toggle floor projection rendering."""
        self._floor_projection_enabled = bool(enabled)
        if not self._floor_projection_enabled:
            self.renderer.clear_floor_indicator()

    def _set_direct_mode(self, enabled: bool) -> None:
        """Toggle raw direct teleop mode and strip/restore auxiliary visuals."""
        enabled = bool(enabled)
        if enabled == self._direct_mode:
            return
        self.input.set_move_mode(enabled)
        self._direct_mode = enabled
        if enabled:
            self._floor_projection_before_direct_mode = bool(
                self._floor_projection_enabled
            )
            self._floor_projection_enabled = False
            self.renderer.clear_floor_indicator()
            self.renderer.clear_path_markers()
            self.renderer.set_orientation_preview_visible(False)
        else:
            self._floor_projection_enabled = bool(
                self._floor_projection_before_direct_mode
            )
            self.renderer.set_orientation_preview_visible(True)
            self._rerender_path()
        self._status_cache_next_s = 0.0

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
            self._avatar_auto_reset_pending_pose = None
            self._avatar_auto_reset_done = False
            self._robot_stopped_last = is_zero
            return

        if not is_zero:
            self._avatar_auto_reset_pending_since = None
            self._avatar_auto_reset_pending_pose = None
            self._avatar_auto_reset_done = False
            self._robot_stopped_last = False
            return

        if self._avatar_auto_reset_done:
            self._robot_stopped_last = True
            return

        if self.nav.state.last_robot_pose_panda is None:
            self._avatar_auto_reset_pending_since = None
            self._avatar_auto_reset_pending_pose = None
            self._robot_stopped_last = True
            return

        if self.input.is_any_input_active():
            self._avatar_auto_reset_pending_since = None
            self._avatar_auto_reset_pending_pose = None
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
            self._avatar_auto_reset_pending_pose = None
            self._robot_stopped_last = True
            return

        now = self.taskMgr.globalClock.getFrameTime()
        if self._avatar_auto_reset_pending_since is None:
            self._avatar_auto_reset_pending_since = now
            self._avatar_auto_reset_pending_pose = (
                float(avatar_pos[0]),
                float(avatar_pos[1]),
                float(avatar_pos[2]),
            )
            self._robot_stopped_last = True
            return

        pending_pose = self._avatar_auto_reset_pending_pose
        if pending_pose is not None:
            pdx = float(avatar_pos[0]) - pending_pose[0]
            pdy = float(avatar_pos[1]) - pending_pose[1]
            pdz = float(avatar_pos[2]) - pending_pose[2]
            # Cancel auto-reset as soon as avatar moves during the pending window.
            if (pdx * pdx + pdy * pdy + pdz * pdz) > 1e-6:
                self._avatar_auto_reset_pending_since = None
                self._avatar_auto_reset_pending_pose = None
                self._avatar_auto_reset_done = False
                self._robot_stopped_last = True
                return

        if (now - self._avatar_auto_reset_pending_since) >= AVATAR_AUTO_RESET_DELAY_S:
            self.renderer.sync_avatar_to_robot(self.nav.state.last_robot_pose_panda)
            self._avatar_auto_reset_done = True
            self._avatar_auto_reset_pending_since = None
            self._avatar_auto_reset_pending_pose = None
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
        if self._direct_mode:
            self.renderer.clear_path_markers()
            return
        if self.ui.mode == "Follow Mode":
            self.renderer.render_path_markers(self.nav.state.follow_path_points)
        elif self._last_path_poses:
            self.renderer.render_path_markers(self._last_path_poses)

    def _refresh_path_line_style_if_needed(self, min_interval_s: float = 0.5) -> None:
        """Refresh only line style/heatmap for path-quality updates (throttled)."""
        if self.ui.mode == "Follow Mode":
            return
        now_s = time.monotonic()
        if (now_s - self._last_path_line_style_refresh_s) < max(0.0, min_interval_s):
            return
        if not self._last_path_poses:
            return
        self.renderer.refresh_path_line(self._last_path_poses)
        self._last_path_line_style_refresh_s = now_s

    # ---- path viz setters ----
    def _set_path_mode(self, mode: str) -> None:
        self.renderer.set_path_mode(mode)
        self._rerender_path()

    def _set_marker_spacing(self, spacing_m: float) -> None:
        self.renderer.set_marker_spacing(spacing_m)
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
        if self._cleaned_up:
            return
        self._cleaned_up = True

        def _stop_process(proc: Optional[subprocess.Popen], name: str) -> None:
            if proc is None or proc.poll() is not None:
                return
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                print(f"[{name}] terminate timed out, killing process")
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            except Exception:
                pass

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
                _stop_process(self._octomap_proc, "octomap")
                self._octomap_proc = None
        except Exception:
            pass
        try:
            if self._gamepad_proc is not None:
                _stop_process(self._gamepad_proc, "gamepad")
                self._gamepad_proc = None
        except Exception:
            pass
        try:
            if self._spacemouse_proc is not None:
                _stop_process(self._spacemouse_proc, "spacemouse")
                self._spacemouse_proc = None
        except Exception:
            pass
        self.renderer.clear_path_markers()


def main(argv: Optional[Sequence[str]] = None) -> None:
    app = SpacebotLinkApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt received, shutting down")
    finally:
        app._cleanup()
        try:
            app.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
