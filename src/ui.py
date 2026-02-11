"""UI state holder and ImGui overlay rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import p3dimgui
from imgui_bundle import imgui
from panda3d.core import PerspectiveLens
from direct.showbase.ShowBase import ShowBase


class UI:
    """Lightweight UI state + ImGui overlay."""

    def __init__(
        self,
        base: ShowBase,
        status_provider: Callable[[], Dict[str, Any]],
        on_abort: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize UI state and ImGui overlay plumbing."""
        self.base = base
        self.mode: str = "Goal Mode"
        self.move_target: str = "Avatar"
        self._last_status: str = ""
        self._on_abort = on_abort
        self._status_provider = status_provider
        self._imgui_ready: bool = False
        self._imgui_ini_path: Path = Path(__file__).resolve().parent.parent / "imgui.ini"
        self._show_nav: bool = True
        self._show_pathviz: bool = True
        self._show_poses: bool = True
        self._show_octomap: bool = True
        self._show_octomap_raw: bool = False

        self._init_imgui()

    def _bump_fov(self, delta: float) -> None:
        """Nudge camera FOV by delta degrees on both axes."""
        lens: PerspectiveLens = self.base.camLens
        fx, fy = lens.getFov()
        lens.setFov(max(10.0, fx + delta), max(10.0, fy + delta))
        update = getattr(self.base, "_update_bg_scale", None)
        if callable(update):
            update()

    def set_mode(self, mode: str) -> None:
        """Set active UI mode label."""
        self.mode = mode
        self._last_status = ""

    def set_move_target(self, target: str) -> None:
        """Update the move target label in the HUD ("Avatar" or "Robot")."""
        self.move_target = target
        self._last_status = ""

    def update(self, extra: str = "") -> None:
        """Store a status message for later display."""
        self._last_status = extra

    def trigger_abort(self) -> None:
        """Invoke abort callback if configured."""
        if callable(self._on_abort):
            try:
                self._on_abort()
            except Exception:
                pass

    # ---- ImGui ----
    def _init_imgui(self) -> None:
        """Set up ImGui styling and hook render callback."""
        try:
            p3dimgui.init()
            if self._imgui_ini_path.exists():
                imgui.load_ini_settings_from_disk(str(self._imgui_ini_path))
            style = imgui.get_style()
            style.font_size_base = 23.0
            style.font_scale_main = 1.3
            style.scale_all_sizes(1.3)
            style.window_rounding = 8.0
            style.child_rounding = 8.0
            style.frame_rounding = 6.0
            style.window_padding = (12, 12)
            style.frame_padding = (10, 8)
            style.item_spacing = (10, 8)
            imgui.style_colors_dark()
        except Exception as exc:
            print(f"[imgui] Failed to initialize ImGui overlay: {exc}")
            self._imgui_ready = False
            return

        self._imgui_ready = True
        self.base.accept("imgui-new-frame", self._draw_overlay)

    def _draw_overlay(self) -> None:
        """Render debug + control overlay each frame."""
        if not self._imgui_ready:
            return
        status = self._status_provider()
        pad = 14.0
        io = imgui.get_io()
        scr_w = io.display_size.x or 1920.0
        scr_h = io.display_size.y or 1080.0

        imgui.set_next_window_pos((pad, pad), imgui.Cond_.once)
        imgui.set_next_window_size((1000, 620), imgui.Cond_.once)
        imgui.set_next_window_bg_alpha(0.92)
        imgui.begin("Debug")

        def _parse_bool_flag(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "false"):
                    return lowered == "true"
            return None

        fps_text = status.get("fps", 0.0)
        imgui.text(f"FPS: {fps_text:.1f}")

        imgui.separator()
        imgui.text(f"Mode: {status.get('mode', '')}")
        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()
        imgui.text(f"Waypoints: {status.get('waypoint_count', 0)}")
        occluded_bool = _parse_bool_flag(status.get("octomap_occluded"))
        if occluded_bool is not None:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            occ_color = (
                (1.0, 0.6, 0.2, 1.0)
                if occluded_bool
                else (0.7, 1.0, 0.7, 1.0)
            )
            imgui.text_colored(
                occ_color, f"Occluded: {'yes' if occluded_bool else 'no'}"
            )
        in_obstacle_bool = _parse_bool_flag(status.get("octomap_in_obstacle"))
        if in_obstacle_bool is not None:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            inside_color = (
                (1.0, 0.35, 0.35, 1.0)
                if in_obstacle_bool
                else (0.7, 1.0, 0.7, 1.0)
            )
            imgui.text_colored(
                inside_color,
                f"Inside obstacle: {'yes' if in_obstacle_bool else 'no'}",
            )

        changed_nav, publish_nav = imgui.checkbox(
            "Publish goals/paths", bool(status.get("nav_enabled", True))
        )
        imgui.same_line()
        changed_move, move_robot = imgui.checkbox(
            "Control robot (cmd_vel)", bool(status.get("move_robot", False))
        )
        imgui.same_line()
        changed_floor, floor_on = imgui.checkbox(
            "Floor projection", bool(status.get("floor_projection_enabled", True))
        )
        if changed_nav:
            status.get("set_nav_enabled", lambda _v: None)(publish_nav)
        if changed_move:
            status.get("set_move_mode", lambda _v: None)(move_robot)
        if changed_floor:
            status.get("set_floor_projection_enabled", lambda _v: None)(floor_on)

        imgui.spacing()
        imgui.separator()
        imgui.text("Widget visibility")
        imgui.text("Show sections")
        imgui.same_line()
        _, self._show_nav = imgui.checkbox("Nav", self._show_nav)
        imgui.same_line()
        _, self._show_pathviz = imgui.checkbox("PathViz", self._show_pathviz)
        imgui.same_line()
        _, self._show_poses = imgui.checkbox("Poses", self._show_poses)
        imgui.same_line()
        _, self._show_octomap = imgui.checkbox("OctoMap", self._show_octomap)
        imgui.spacing()
        imgui.columns(2, "debug_columns", False)
        if self._show_nav:
            imgui.begin_child("Nav", (0, 170), True)
            if status.get("mode") == "Goal Mode":
                imgui.text("Goal")
                ros_goal = status.get("last_goal_ros")
                if ros_goal is not None:
                    (gx, gy, gz), (gh, gp, gr) = ros_goal
                    imgui.text(f"  pos (m, ROS): {gx:.2f}, {gy:.2f}, {gz:.2f}")
                    imgui.text(f"  rpy (deg, ROS): {gh:.1f}, {gp:.1f}, {gr:.1f}")
                else:
                    last_goal = status.get("last_goal_panda")
                    if last_goal is not None:
                        (gx, gy, gz), (gh, gp, gr) = last_goal
                        imgui.text(f"  pos (m): {gx:.2f}, {gy:.2f}, {gz:.2f}")
                        imgui.text(f"  hpr (deg): {gh:.1f}, {gp:.1f}, {gr:.1f}")
                    else:
                        imgui.text("  waiting for goal")
                imgui.text(f"  path poses: {status.get('path_pose_count', 0)}")
            else:
                imgui.text("Follow path")
                imgui.text(f"  buffered poses: {status.get('waypoint_count', 0)}")
                tip = status.get("follow_tip")
                if tip is not None:
                    fx, fy, fz = tip
                    imgui.text(f"  tip (m): {fx:.2f}, {fy:.2f}, {fz:.2f}")
            imgui.end_child()

        imgui.spacing()
        if self._show_pathviz:
            imgui.begin_child("PathViz", (0, 220), True)
            imgui.text("Path visualization")
            modes = ["poses", "poses_line", "planes", "animated"]
            current_mode = status.get("path_mode", modes[0])
            try:
                idx = modes.index(current_mode)
            except ValueError:
                idx = 0
            changed_mode, new_idx = imgui.combo("Mode", idx, modes)
            if changed_mode:
                status.get("set_path_mode", lambda _m: None)(modes[new_idx])

            selected_mode = modes[new_idx]
            if selected_mode == "poses":
                pose_stride = int(status.get("pose_stride", 4))
                imgui.text("Ghost stride (every Nth pose)")
                imgui.set_next_item_width(140)
                changed, pose_stride = imgui.input_int("##pose_stride", pose_stride)
                if changed:
                    status.get("set_pose_stride", lambda _v: None)(max(1, pose_stride))
            elif selected_mode == "poses_line":
                line_stride = int(status.get("line_stride", 8))
                imgui.text("Ghost stride (every Nth pose)")
                imgui.set_next_item_width(140)
                changed, line_stride = imgui.input_int("##line_stride", line_stride)
                if changed:
                    status.get("set_line_stride", lambda _v: None)(max(1, line_stride))
            elif selected_mode == "planes":
                plane_stride = int(status.get("pose_stride", 4))
                imgui.text("Plane stride (every Nth pose)")
                imgui.set_next_item_width(140)
                changed, plane_stride = imgui.input_int("##plane_stride", plane_stride)
                if changed:
                    status.get("set_pose_stride", lambda _v: None)(max(1, plane_stride))
            else:
                anim_speed = float(status.get("anim_speed", 1.0))
                imgui.text("Anim speed (m/s)")
                imgui.set_next_item_width(140)
                changed, anim_speed = imgui.input_float(
                    "##anim_speed", anim_speed, step=0.0
                )
                if changed:
                    status.get("set_anim_speed", lambda _v: None)(max(0.01, anim_speed))
                anim_instances = int(status.get("anim_instances", 1))
                imgui.text("Ghost instances")
                imgui.set_next_item_width(140)
                changed_count, anim_instances = imgui.input_int(
                    "##anim_instances", anim_instances
                )
                if changed_count:
                    status.get("set_anim_instances", lambda _v: None)(
                        max(1, anim_instances)
                    )
                anim_line_enabled = bool(status.get("anim_line_enabled", True))
                changed_line, anim_line_enabled = imgui.checkbox(
                    "Show path line", anim_line_enabled
                )
                if changed_line:
                    status.get("set_anim_line_enabled", lambda _v: None)(
                        anim_line_enabled
                    )
            imgui.end_child()

        imgui.next_column()
        if self._show_poses:
            imgui.begin_child("Poses", (0, 170), True)
            imgui.columns(2, "pose_columns", False)
            imgui.text("Avatar pose (ROS frame)")
            av_ros_tuple = status.get("avatar_ros_pose")
            if av_ros_tuple is not None:
                (ax, ay, az), (ar, ap, ayaw) = av_ros_tuple
                imgui.text(f"  pos (m): {ax:.2f}, {ay:.2f}, {az:.2f}")
                imgui.text(f"  rpy (deg): {ar:.1f}, {ap:.1f}, {ayaw:.1f}")
            else:
                imgui.text("  waiting for pose")

            imgui.next_column()
            imgui.text("Robot pose (ROS frame)")
            robot_pose = status.get("robot_ros_pose")
            if robot_pose is not None:
                (x, y, z), rpy = robot_pose
                imgui.text(f"  pos (m): {x:.2f}, {y:.2f}, {z:.2f}")
                if rpy is not None:
                    roll, pitch, yaw = rpy
                    imgui.text(f"  rpy (deg): {roll:.1f}, {pitch:.1f}, {yaw:.1f}")
            else:
                imgui.text("  waiting for /space_cobot/pose")
            imgui.columns(1)
            imgui.end_child()

        pos_err = status.get("avatar_robot_error")
        if pos_err is not None:
            imgui.spacing()
            imgui.text(f"Avatar-robot position error: {pos_err:.3f} m")
        if self._show_octomap:
            imgui.begin_child("OctoMap", (0, 220), True)
            imgui.text("OctoMap Raycast")
            imgui.separator()

            occluded_display = _parse_bool_flag(status.get("octomap_occluded"))
            if occluded_display is None:
                imgui.text("avatar_occluded=unknown")
            else:
                occ_color = (
                    (1.0, 0.6, 0.2, 1.0)
                    if occluded_display
                    else (0.7, 1.0, 0.7, 1.0)
                )
                imgui.text_colored(
                    occ_color,
                    f"avatar_occluded={str(occluded_display).lower()}",
                )

            in_obstacle_display = _parse_bool_flag(status.get("octomap_in_obstacle"))
            if in_obstacle_display is None:
                imgui.text("avatar_in_obstacle=unknown")
            else:
                inside_color = (
                    (1.0, 0.35, 0.35, 1.0)
                    if in_obstacle_display
                    else (0.7, 1.0, 0.7, 1.0)
                )
                imgui.text_colored(
                    inside_color,
                    f"avatar_in_obstacle={str(in_obstacle_display).lower()}",
                )

            imgui.spacing()
            imgui.columns(2, "octo_kv", False)

            ground_axis = status.get("octomap_ground_axis")
            ground_distance = status.get("octomap_ground_distance")
            imgui.text("ground_axis")
            imgui.next_column()
            imgui.text("unknown" if ground_axis is None else f"{ground_axis}")
            imgui.next_column()

            imgui.text("ground_distance_m")
            imgui.next_column()
            if ground_distance is None:
                imgui.text("unknown")
            else:
                try:
                    dist_val = float(ground_distance)
                    imgui.text(f"{dist_val:.2f}")
                except Exception:
                    imgui.text(str(ground_distance))
            imgui.columns(1)

            octomap_json = status.get("octomap_json")
            if octomap_json:
                imgui.spacing()
                _, self._show_octomap_raw = imgui.checkbox(
                    "Show raw JSON", self._show_octomap_raw
                )
                if self._show_octomap_raw:
                    imgui.spacing()
                    imgui.begin_child("octomap_json", (0, 90), True)
                    imgui.text_wrapped(octomap_json)
                    imgui.end_child()
            imgui.end_child()

        imgui.columns(1)

        imgui.end()

        # Top-right control window
        ctrl_w = 420.0
        ctrl_h = 250.0
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

        def _button(
            label: str,
            size: tuple[float, float],
            base_col: tuple[float, float, float, float],
            hover_col: tuple[float, float, float, float],
            active_col: tuple[float, float, float, float],
        ) -> bool:
            """Render a styled button and return whether it was pressed."""
            imgui.push_style_color(imgui.Col_.button, base_col)
            imgui.push_style_color(imgui.Col_.button_hovered, hover_col)
            imgui.push_style_color(imgui.Col_.button_active, active_col)
            pressed = imgui.button(label, size)
            imgui.pop_style_color(3)
            return pressed

        is_follow = status.get("mode") == "Follow Mode"

        def _mode_colors(active: bool) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]:
            """Return base/hover/active colors based on mode state."""
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
            status.get("activate_follow", lambda: None)()
        imgui.same_line()
        if _button("Goal", (half, btn_h), g_base, g_hover, g_active):
            status.get("activate_goal", lambda: None)()

        imgui.spacing()
        imgui.text_colored((0.82, 0.90, 1.00, 1.0), f"Active Mode: {status.get('mode')}")
        imgui.separator()

        btn_w = avail
        if _button(
            "Abort",
            (btn_w, btn_h),
            (0.70, 0.22, 0.22, 1.0),
            (0.78, 0.28, 0.28, 1.0),
            (0.60, 0.18, 0.18, 1.0),
        ):
            self.trigger_abort()

        imgui.end()

    def save_imgui_settings(self) -> None:
        """Persist ImGui layout to disk if initialized."""
        if not self._imgui_ready:
            return
        try:
            self._imgui_ini_path.parent.mkdir(parents=True, exist_ok=True)
            imgui.save_ini_settings_to_disk(str(self._imgui_ini_path))
        except Exception:
            pass
