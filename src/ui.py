"""UI state holder and ImGui overlay rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import p3dimgui
from direct.showbase.ShowBase import ShowBase
from imgui_bundle import imgui
from panda3d.core import PerspectiveLens

from config import (
    ORIENT_PREVIEW_CROP_TOP,
    ORIENT_PREVIEW_ENABLED,
    ORIENT_PREVIEW_REGION,
    PATH_QUALITY_LABEL_CRITICAL,
    PATH_QUALITY_LABEL_EXCELLENT,
    PATH_QUALITY_LABEL_GOOD,
    PATH_QUALITY_LABEL_RISKY,
    PATH_QUALITY_THRESH_EXCELLENT,
    PATH_QUALITY_THRESH_GOOD,
    PATH_QUALITY_THRESH_RISKY,
    UI_FONT_PATH,
    UI_FONT_SIZE_PX,
)


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
        self._imgui_ini_path: Path = (
            Path(__file__).resolve().parent.parent / "imgui.ini"
        )
        self._show_nav: bool = True
        self._show_pathviz: bool = True
        self._show_poses: bool = True
        self._show_octomap: bool = True
        self._show_octomap_raw: bool = False
        self._advanced_debug: bool = False

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
            io = imgui.get_io()
            # Load a configurable TTF font (bold by default) when available.
            try:
                font_path = Path(str(UI_FONT_PATH)).expanduser()
                if font_path.is_file():
                    font = io.fonts.add_font_from_file_ttf(
                        str(font_path), float(UI_FONT_SIZE_PX)
                    )
                    if font is not None:
                        io.font_default = font
                else:
                    print(f"[imgui] UI font not found, using default: {font_path}")
            except Exception as exc:
                print(f"[imgui] Failed to load UI font, using default: {exc}")
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

        def _draw_attitude_preview_frame() -> None:
            """Draw a subtle framed widget around the Panda3D attitude preview region."""
            if bool(status.get("direct_mode", False)):
                return
            if not ORIENT_PREVIEW_ENABLED:
                return
            try:
                x0, x1, y0, y1 = ORIENT_PREVIEW_REGION
            except Exception:
                return
            if x1 <= x0 or y1 <= y0:
                return

            # Match renderer crop logic so the frame aligns with the display region.
            crop = max(0.0, float(ORIENT_PREVIEW_CROP_TOP))
            if crop > 1e-6:
                base_height = y1 - y0
                base_width = x1 - x0
                if base_height > 1e-6:
                    new_y1 = y1 - crop
                    new_height = new_y1 - y0
                    if new_height > 1e-6:
                        aspect = base_width / base_height
                        new_width = aspect * new_height
                        mid_x = 0.5 * (x0 + x1)
                        x0 = mid_x - 0.5 * new_width
                        x1 = mid_x + 0.5 * new_width
                        y1 = new_y1
                        x0 = max(0.0, x0)
                        x1 = min(1.0, x1)

            # Normalize/clamp because region intentionally may overscan screen edges.
            x0 = max(0.0, min(1.0, float(x0)))
            x1 = max(0.0, min(1.0, float(x1)))
            y0 = max(0.0, min(1.0, float(y0)))
            y1 = max(0.0, min(1.0, float(y1)))
            if x1 <= x0 or y1 <= y0:
                return

            px0 = x0 * scr_w
            px1 = x1 * scr_w
            py0 = (1.0 - y1) * scr_h
            py1 = (1.0 - y0) * scr_h
            if px1 - px0 < 20.0 or py1 - py0 < 20.0:
                return

            draw = imgui.get_foreground_draw_list()
            if draw is None:
                return

            outer_pad = 8.0
            title_h = 38.0
            rounding = 10.0
            fx0 = px0 - outer_pad
            fx1 = px1 + outer_pad
            fy0 = py0 - title_h - 14.0
            fy1 = py1 + outer_pad

            shadow_col = imgui.get_color_u32((0.0, 0.0, 0.0, 0.34))
            panel_col = imgui.get_color_u32((0.06, 0.08, 0.11, 0.28))
            border_col = imgui.get_color_u32((0.40, 0.78, 0.98, 0.75))
            inner_border_col = imgui.get_color_u32((0.95, 0.98, 1.0, 0.20))
            title_bg_col = imgui.get_color_u32((0.10, 0.16, 0.22, 0.92))
            title_text_col = imgui.get_color_u32((0.88, 0.96, 1.0, 1.0))

            draw.add_rect_filled(
                (fx0 + 3.0, fy0 + 4.0),
                (fx1 + 3.0, fy1 + 4.0),
                shadow_col,
                rounding + 1.0,
            )
            draw.add_rect_filled((fx0, fy0), (fx1, fy1), panel_col, rounding)
            draw.add_rect((fx0, fy0), (fx1, fy1), border_col, rounding, 0, 2.0)
            draw.add_rect(
                (px0 - 2.0, py0 - 2.0),
                (px1 + 2.0, py1 + 2.0),
                inner_border_col,
                8.0,
                0,
                1.0,
            )

            label = "MOTION PREVIEW"
            text_sz = imgui.calc_text_size(label)
            chip_w = max(154.0, text_sz.x + 24.0)
            chip_x0 = fx0 + 10.0
            chip_y0 = fy0 + 4.0
            chip_x1 = chip_x0 + chip_w
            chip_y1 = chip_y0 + title_h
            draw.add_rect_filled(
                (chip_x0, chip_y0), (chip_x1, chip_y1), title_bg_col, 7.0
            )
            draw.add_rect(
                (chip_x0, chip_y0), (chip_x1, chip_y1), border_col, 7.0, 0, 1.0
            )
            draw.add_text((chip_x0 + 10.0, chip_y0 + 7.0), title_text_col, label)

        _draw_attitude_preview_frame()

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

        def _quality_badge(value):
            try:
                q = max(0.0, min(1.0, float(value)))
            except Exception:
                return None
            excellent_thr = float(PATH_QUALITY_THRESH_EXCELLENT)
            good_thr = float(PATH_QUALITY_THRESH_GOOD)
            risky_thr = float(PATH_QUALITY_THRESH_RISKY)
            thresholds = sorted([risky_thr, good_thr, excellent_thr])
            risky_thr, good_thr, excellent_thr = (
                thresholds[0],
                thresholds[1],
                thresholds[2],
            )
            if q >= excellent_thr:
                label = PATH_QUALITY_LABEL_EXCELLENT
            elif q >= good_thr:
                label = PATH_QUALITY_LABEL_GOOD
            elif q >= risky_thr:
                label = PATH_QUALITY_LABEL_RISKY
            else:
                label = PATH_QUALITY_LABEL_CRITICAL
            if q < 0.5:
                t = q / 0.5
                c0 = (0.95, 0.22, 0.20, 1.0)
                c1 = (1.00, 0.78, 0.10, 1.0)
            else:
                t = (q - 0.5) / 0.5
                c0 = (1.00, 0.78, 0.10, 1.0)
                c1 = (0.22, 0.92, 0.38, 1.0)
            color = tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(4))
            return q, label, color

        def _draw_quality_bar(
            q: float, color: tuple[float, float, float, float], width: float = 0.0
        ) -> None:
            """Draw a compact horizontal bar showing path goodness."""
            draw = imgui.get_window_draw_list()
            if draw is None:
                return
            pos = imgui.get_cursor_screen_pos()
            avail_w = imgui.get_content_region_avail().x
            bar_w = max(120.0, width if width > 0.0 else avail_w)
            bar_h = 12.0
            x0, y0 = pos.x, pos.y
            x1, y1 = x0 + bar_w, y0 + bar_h
            rounding = 6.0
            bg_col = imgui.get_color_u32((0.16, 0.17, 0.20, 1.0))
            fill_col = imgui.get_color_u32(color)
            border_col = imgui.get_color_u32((0.55, 0.58, 0.65, 0.55))
            fill_x = x0 + max(0.0, min(1.0, q)) * bar_w
            draw.add_rect_filled((x0, y0), (x1, y1), bg_col, rounding)
            if fill_x > x0 + 1.0:
                draw.add_rect_filled((x0, y0), (fill_x, y1), fill_col, rounding)
            draw.add_rect((x0, y0), (x1, y1), border_col, rounding)
            imgui.dummy((bar_w, bar_h))

        def _format_hhmmss(seconds: Any) -> str:
            try:
                total = max(0, int(float(seconds)))
            except Exception:
                return "00:00:00"
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        fps_text = status.get("fps", 0.0)
        imgui.text(f"FPS: {fps_text:.1f}")

        imgui.separator()
        imgui.text(f"Mode: {status.get('mode', '')}")
        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()
        imgui.text(f"Waypoints: {status.get('waypoint_count', 0)}")
        current_module = status.get("current_iss_module")
        if current_module:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            imgui.text(f"Module: {current_module}")
        total_len = status.get("total_flight_length_m")
        if total_len is not None:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            try:
                imgui.text(f"Distance: {float(total_len):.2f} m")
            except Exception:
                imgui.text(f"Distance: {total_len}")
        op_time_s = status.get("operational_time_s")
        if op_time_s is not None:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            imgui.text(f"Operational: {_format_hhmmss(op_time_s)}")
        path_goodness = status.get("path_goodness")
        quality_badge = _quality_badge(path_goodness)
        if quality_badge is not None:
            try:
                q, quality_label, quality_color = quality_badge
                imgui.same_line()
                imgui.text_disabled("|")
                imgui.same_line()
                imgui.text_colored(quality_color, f"Path: {quality_label} ({q:.2f})")
            except Exception:
                pass
        occluded_bool = _parse_bool_flag(status.get("octomap_occluded"))
        if occluded_bool is not None:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            occ_color = (1.0, 0.6, 0.2, 1.0) if occluded_bool else (0.7, 1.0, 0.7, 1.0)
            imgui.text_colored(
                occ_color, f"Occluded: {'yes' if occluded_bool else 'no'}"
            )
        in_obstacle_bool = _parse_bool_flag(status.get("octomap_in_obstacle"))
        if in_obstacle_bool is not None:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            inside_color = (
                (1.0, 0.35, 0.35, 1.0) if in_obstacle_bool else (0.7, 1.0, 0.7, 1.0)
            )
            imgui.text_colored(
                inside_color,
                f"Inside obstacle: {'yes' if in_obstacle_bool else 'no'}",
            )

        changed_nav, publish_nav = imgui.checkbox(
            "Publish goals/paths", bool(status.get("nav_enabled", True))
        )
        imgui.same_line()
        changed_direct, direct_mode = imgui.checkbox(
            "Direct mode", bool(status.get("direct_mode", False))
        )
        if changed_nav:
            status.get("set_nav_enabled", lambda _v: None)(publish_nav)
        if changed_direct:
            status.get("set_direct_mode", lambda _v: None)(direct_mode)

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
            q = status.get("path_goodness")
            if q is not None:
                try:
                    qf = float(q)
                    badge = _quality_badge(qf)
                    label = badge[1] if badge is not None else "?"
                    imgui.text(f"  path_goodness (MC): {qf:.3f} [{label}]")
                except Exception:
                    imgui.text(f"  path_goodness (MC): {q}")
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
            if selected_mode in ("poses", "poses_line", "planes"):
                marker_spacing = float(status.get("marker_spacing_m", 0.40))
                imgui.text("Projection spacing (m)")
                imgui.set_next_item_width(140)
                changed, marker_spacing = imgui.input_float(
                    "##marker_spacing_m", marker_spacing, step=0.05
                )
                if changed:
                    status.get("set_marker_spacing", lambda _v: None)(
                        max(0.05, marker_spacing)
                    )
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
        total_len = status.get("total_flight_length_m")
        if total_len is not None:
            imgui.spacing()
            try:
                imgui.text(f"Total flight length: {float(total_len):.3f} m")
            except Exception:
                imgui.text(f"Total flight length: {total_len}")
        op_time_s = status.get("operational_time_s")
        if op_time_s is not None:
            imgui.spacing()
            imgui.text(f"Operational time: {_format_hhmmss(op_time_s)}")
        if total_len is not None or op_time_s is not None:
            imgui.spacing()
            if imgui.button("Reset"):
                status.get("reset_session_metrics", lambda: None)()
        if self._show_octomap:
            imgui.begin_child("OctoMap", (0, 220), True)
            imgui.text("OctoMap Raycast")
            imgui.separator()

            occluded_display = _parse_bool_flag(status.get("octomap_occluded"))
            if occluded_display is None:
                imgui.text("avatar_occluded=unknown")
            else:
                occ_color = (
                    (1.0, 0.6, 0.2, 1.0) if occluded_display else (0.7, 1.0, 0.7, 1.0)
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
        if bool(status.get("direct_mode", False)):
            return

        # Top-right control window
        ctrl_w = 540.0
        ctrl_h = 450.0
        ctrl_x = max(pad, scr_w - ctrl_w - pad)
        ctrl_y = pad
        imgui.set_next_window_pos((ctrl_x, ctrl_y), imgui.Cond_.always)
        imgui.set_next_window_size((ctrl_w, ctrl_h), imgui.Cond_.once)
        imgui.begin(
            "Dashboard",
            flags=imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize,
        )
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (14.0, 10.0))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (16.0, 14.0))

        avail = imgui.get_content_region_avail().x
        half = (avail - imgui.get_style().item_spacing.x) * 0.5
        btn_h = 76

        def _text_bold(color, text: str) -> None:
            """Draw text with a simple double-pass to mimic a bolder weight."""
            pos = imgui.get_cursor_screen_pos()
            draw = imgui.get_window_draw_list()
            if draw is None:
                imgui.text_colored(color, text)
                return
            shadow = (0.02, 0.02, 0.02, color[3] * 0.95)
            shadow_u32 = imgui.get_color_u32(shadow)
            color_u32 = imgui.get_color_u32(color)
            # Multi-pass offsets create a thicker/faux-bold appearance.
            for ox, oy in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.5)):
                draw.add_text((pos.x + ox, pos.y + oy), shadow_u32, text)
            draw.add_text((pos.x, pos.y), color_u32, text)
            imgui.dummy(imgui.calc_text_size(text))

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

        def _mode_colors(
            active: bool,
        ) -> tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ]:
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

        # imgui.text_disabled("Navigation Mode")
        if _button("FOLLOW", (half, btn_h), f_base, f_hover, f_active):
            status.get("activate_follow", lambda: None)()
        imgui.same_line()
        if _button("GOAL", (half, btn_h), g_base, g_hover, g_active):
            status.get("activate_goal", lambda: None)()
        btn_w = avail
        imgui.spacing()
        if _button(
            "ABORT",
            (btn_w, btn_h),
            (0.70, 0.22, 0.22, 1.0),
            (0.78, 0.28, 0.28, 1.0),
            (0.60, 0.18, 0.18, 1.0),
        ):
            self.trigger_abort()

        module_name = status.get("current_iss_module")
        if module_name:
            imgui.spacing()
            label = str(module_name).strip()
            if label.lower() == "unknown":
                module_color = (1.0, 0.72, 0.36, 1.0)
            else:
                module_color = (0.70, 0.92, 1.0, 1.0)
            _text_bold(module_color, f"CURRENT MODULE: {label.upper()}")

        quality_badge = _quality_badge(status.get("path_goodness"))
        if quality_badge is not None:
            q, quality_label, quality_color = quality_badge
            imgui.spacing()
            # imgui.text_disabled("Path Quality")
            _text_bold(quality_color, f"PATH {quality_label.upper()} ({q:.2f})")
            _draw_quality_bar(q, quality_color, width=avail)
        bar_progress = 0.0
        try:
            bar_progress = max(
                0.0, min(1.0, float(status.get("response_delay_fill", 0.0)))
            )
        except Exception:
            bar_progress = 0.0
        if bar_progress < 0.999:
            imgui.spacing()
            try:
                delay_fill_s = max(0.1, float(status.get("response_delay_fill_s", 2.5)))
            except Exception:
                delay_fill_s = 2.5
            imgui.text_disabled("Robot Response Delay")
            imgui.push_style_color(imgui.Col_.frame_bg, (0.10, 0.14, 0.18, 1.0))
            imgui.push_style_color(imgui.Col_.plot_histogram, (0.20, 0.72, 0.95, 1.0))
            imgui.push_style_color(
                imgui.Col_.plot_histogram_hovered, (0.30, 0.80, 1.0, 1.0)
            )
            imgui.progress_bar(bar_progress, (avail, 12.0), "")
            imgui.pop_style_color(3)
        imgui.separator()

        imgui.pop_style_var(2)
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
