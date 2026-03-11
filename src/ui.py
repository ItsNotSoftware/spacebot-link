"""UI state holder and ImGui overlay rendering."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import p3dimgui
from direct.showbase.ShowBase import ShowBase
from imgui_bundle import imgui
from panda3d.core import PerspectiveLens, Texture

from config import (
    ISS_MAP_HEADING_LEN_PX,
    ISS_MAP_MARKER_RADIUS_PX,
    ISS_MAP_ROBOT_OFFSET_XY_M,
    ISS_MAP_ROBOT_SCALE_PX_PER_M,
    ISS_MAP_YAW_OFFSET_DEG,
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
        self._iss_map_path: Path = (
            Path(__file__).resolve().parent.parent / "assets/iss.jpeg"
        )
        self._iss_map_texture: Optional[Texture] = None
        self._iss_map_tex_id: Optional[int] = None
        self._show_iss_map_window: bool = True
        self._advanced_debug: bool = False
        self._iss_map_offset_x_m: float = float(ISS_MAP_ROBOT_OFFSET_XY_M[0])
        self._iss_map_offset_y_m: float = float(ISS_MAP_ROBOT_OFFSET_XY_M[1])
        self._iss_map_scale_x_px_per_m: float = float(ISS_MAP_ROBOT_SCALE_PX_PER_M[0])
        self._iss_map_scale_y_px_per_m: float = float(ISS_MAP_ROBOT_SCALE_PX_PER_M[1])
        self._iss_map_yaw_offset_deg: float = float(ISS_MAP_YAW_OFFSET_DEG)
        self._iss_map_marker_radius_px: float = float(ISS_MAP_MARKER_RADIUS_PX)
        self._iss_map_heading_len_px: float = float(ISS_MAP_HEADING_LEN_PX)

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

        self._init_iss_map_texture()
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

        def _draw_iss_environment_map_window() -> None:
            """Render the ISS environment map in a movable/closable ImGui window."""
            if not self._show_iss_map_window:
                return
            if self._iss_map_tex_id is None or self._iss_map_texture is None:
                return

            map_w = 860.0
            map_h = 490.0
            dashboard_w = 540.0
            dashboard_x = max(pad, scr_w - dashboard_w - pad)
            map_x = max(pad, dashboard_x - map_w - 12.0)
            map_y = max(pad, 180.0)
            try:
                x0, x1, y0, y1 = ORIENT_PREVIEW_REGION
                if x1 > x0 and y1 > y0:
                    map_y = max(pad, (1.0 - float(y0)) * scr_h + 52.0)
            except Exception:
                pass

            imgui.set_next_window_pos((map_x, map_y), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size((map_w, map_h), imgui.Cond_.first_use_ever)
            visible, self._show_iss_map_window = imgui.begin(
                "ISS Environment Map",
                self._show_iss_map_window,
            )
            if visible:
                avail = imgui.get_content_region_avail()
                tex_w = max(1, int(self._iss_map_texture.get_x_size()))
                tex_h = max(1, int(self._iss_map_texture.get_y_size()))
                scale = min(avail.x / float(tex_w), avail.y / float(tex_h))
                if scale <= 0.0:
                    scale = 1.0
                draw_w = max(1.0, float(tex_w) * scale)
                draw_h = max(1.0, float(tex_h) * scale)
                image_pos = imgui.get_cursor_screen_pos()
                imgui.image(
                    imgui.ImTextureRef(int(self._iss_map_tex_id)),
                    (draw_w, draw_h),
                    (0.0, 1.0),
                    (1.0, 0.0),
                )
                draw = imgui.get_window_draw_list()
                path_xy = status.get("map_path_ros_xy")
                if draw is not None and isinstance(path_xy, list) and len(path_xy) >= 2:
                    off_x_m = self._iss_map_offset_x_m
                    off_y_m = self._iss_map_offset_y_m
                    sx = self._iss_map_scale_x_px_per_m * scale
                    sy = self._iss_map_scale_y_px_per_m * scale
                    prev_screen = None
                    traj_col = imgui.get_color_u32((0.06, 0.52, 0.20, 0.95))
                    end_col = imgui.get_color_u32((0.80, 1.0, 0.84, 0.98))
                    for item in path_xy:
                        try:
                            px_m = float(item[0])
                            py_m = float(item[1])
                        except Exception:
                            continue
                        sxp = image_pos.x + draw_w * 0.5 + (px_m - off_x_m) * sx
                        syp = image_pos.y + draw_h * 0.5 - (py_m - off_y_m) * sy
                        sxp = max(image_pos.x, min(image_pos.x + draw_w, sxp))
                        syp = max(image_pos.y, min(image_pos.y + draw_h, syp))
                        if prev_screen is not None:
                            draw.add_line(prev_screen, (sxp, syp), traj_col, 5.0)
                        prev_screen = (sxp, syp)
                    if prev_screen is not None:
                        draw.add_circle_filled(prev_screen, 3.2, end_col, 14)

                def _draw_entity_marker(
                    cx: float,
                    cy: float,
                    yaw_rad: float,
                    r: float,
                    head_len: float,
                    fill_rgba: tuple,
                ) -> None:
                    """Clean directional marker: circle body + heading arrow + text label."""
                    if draw is None:
                        return
                    glow_c = imgui.get_color_u32((fill_rgba[0], fill_rgba[1], fill_rgba[2], 0.18))
                    fill_c = imgui.get_color_u32(fill_rgba)
                    white_c = imgui.get_color_u32((1.0, 1.0, 1.0, 0.92))
                    shadow_c = imgui.get_color_u32((0.0, 0.0, 0.0, 0.72))

                    # Soft glow halo
                    draw.add_circle_filled((cx, cy), r * 2.8, glow_c, 32)

                    # Heading arrow — drawn first so body sits on top
                    tip_x = cx + math.cos(yaw_rad) * (r + head_len)
                    tip_y = cy - math.sin(yaw_rad) * (r + head_len)
                    edge_x = cx + math.cos(yaw_rad) * r
                    edge_y = cy - math.sin(yaw_rad) * r
                    bw = r * 0.72
                    perp = yaw_rad + math.pi / 2
                    al_x = edge_x + math.cos(perp) * bw
                    al_y = edge_y - math.sin(perp) * bw
                    ar_x = edge_x - math.cos(perp) * bw
                    ar_y = edge_y + math.sin(perp) * bw
                    so = 1.5
                    draw.add_triangle_filled(
                        (tip_x + so, tip_y + so),
                        (al_x + so, al_y + so),
                        (ar_x + so, ar_y + so),
                        shadow_c,
                    )
                    draw.add_triangle_filled((tip_x, tip_y), (al_x, al_y), (ar_x, ar_y), fill_c)
                    draw.add_line((tip_x, tip_y), (al_x, al_y), white_c, 1.2)
                    draw.add_line((al_x, al_y), (ar_x, ar_y), white_c, 1.2)
                    draw.add_line((ar_x, ar_y), (tip_x, tip_y), white_c, 1.2)

                    # Body circle with drop shadow
                    draw.add_circle_filled((cx + 1.0, cy + 1.5), r, shadow_c, 32)
                    draw.add_circle_filled((cx, cy), r, fill_c, 32)
                    draw.add_circle((cx, cy), r, white_c, 32, 2.0)


                r = max(8.0, self._iss_map_marker_radius_px)
                head_len = max(14.0, self._iss_map_heading_len_px * 0.65)
                off_x_m = self._iss_map_offset_x_m
                off_y_m = self._iss_map_offset_y_m
                sx = self._iss_map_scale_x_px_per_m * scale
                sy = self._iss_map_scale_y_px_per_m * scale

                robot_pose = status.get("robot_ros_pose")
                if isinstance(robot_pose, (tuple, list)) and len(robot_pose) >= 2:
                    try:
                        pos = robot_pose[0]
                        rpy = robot_pose[1]
                        robot_x_m = float(pos[0])
                        robot_y_m = float(pos[1])
                        robot_yaw_deg = float(rpy[2])
                    except Exception:
                        robot_x_m = robot_y_m = robot_yaw_deg = None
                    if robot_x_m is not None and robot_y_m is not None and robot_yaw_deg is not None:
                        robot_cx = image_pos.x + draw_w * 0.5 + (robot_x_m - off_x_m) * sx
                        robot_cy = image_pos.y + draw_h * 0.5 - (robot_y_m - off_y_m) * sy
                        robot_cx = max(image_pos.x, min(image_pos.x + draw_w, robot_cx))
                        robot_cy = max(image_pos.y, min(image_pos.y + draw_h, robot_cy))
                        robot_yaw = math.radians(robot_yaw_deg + self._iss_map_yaw_offset_deg)
                        _draw_entity_marker(
                            robot_cx, robot_cy, robot_yaw, r, head_len,
                            (0.18, 0.62, 1.0, 1.0),
                        )

                avatar_pose = status.get("avatar_ros_pose")
                if isinstance(avatar_pose, (tuple, list)) and len(avatar_pose) >= 2:
                    try:
                        pos = avatar_pose[0]
                        rpy = avatar_pose[1]
                        rx_m = float(pos[0])
                        ry_m = float(pos[1])
                        yaw_deg = float(rpy[2])
                    except Exception:
                        rx_m = ry_m = yaw_deg = None
                    if rx_m is not None and ry_m is not None and yaw_deg is not None:
                        av_cx = image_pos.x + draw_w * 0.5 + (rx_m - off_x_m) * sx
                        av_cy = image_pos.y + draw_h * 0.5 - (ry_m - off_y_m) * sy
                        av_cx = max(image_pos.x, min(image_pos.x + draw_w, av_cx))
                        av_cy = max(image_pos.y, min(image_pos.y + draw_h, av_cy))
                        av_yaw = math.radians(yaw_deg + self._iss_map_yaw_offset_deg)
                        _draw_entity_marker(
                            av_cx, av_cy, av_yaw, r, head_len,
                            (1.0, 0.55, 0.10, 1.0),
                        )

                map_avail = imgui.get_content_region_avail().x
                quality_badge = _quality_badge(status.get("path_goodness"))
                if quality_badge is not None:
                    q, quality_label, quality_color = quality_badge
                    imgui.spacing()
                    imgui.text_colored(quality_color, f"PATH {quality_label.upper()} ({q:.2f})")
                    _draw_quality_bar(q, quality_color, width=map_avail)
                bar_progress = 0.0
                try:
                    bar_progress = max(
                        0.0, min(1.0, float(status.get("response_delay_fill", 0.0)))
                    )
                except Exception:
                    bar_progress = 0.0
                if bar_progress < 0.999:
                    imgui.spacing()
                    imgui.text_disabled("Robot Response Delay")
                    imgui.push_style_color(imgui.Col_.frame_bg, (0.10, 0.14, 0.18, 1.0))
                    imgui.push_style_color(imgui.Col_.plot_histogram, (0.20, 0.72, 0.95, 1.0))
                    imgui.push_style_color(
                        imgui.Col_.plot_histogram_hovered, (0.30, 0.80, 1.0, 1.0)
                    )
                    imgui.progress_bar(bar_progress, (map_avail, 12.0), "")
                    imgui.pop_style_color(3)
            imgui.end()

        _draw_iss_environment_map_window()

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
        imgui.columns(2, "debug_columns", False)
        imgui.begin_child("NavSummary", (0, 170), True)
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
        imgui.begin_child("MapCalibration", (0, 260), True)
        imgui.text("ISS map calibration")
        imgui.separator()
        imgui.text_disabled("Robot marker transform")
        imgui.text(
            "offset_xy_m=(%.3f, %.3f)  scale_px_per_m=(%.3f, %.3f)"
            % (
                self._iss_map_offset_x_m,
                self._iss_map_offset_y_m,
                self._iss_map_scale_x_px_per_m,
                self._iss_map_scale_y_px_per_m,
            )
        )
        imgui.text(
            "yaw_offset_deg=%.3f  marker_radius_px=%.3f  heading_len_px=%.3f"
            % (
                self._iss_map_yaw_offset_deg,
                self._iss_map_marker_radius_px,
                self._iss_map_heading_len_px,
            )
        )
        imgui.set_next_item_width(220.0)
        changed, self._iss_map_offset_x_m = imgui.input_float(
            "Offset X (m)", self._iss_map_offset_x_m, step=0.1, format="%.3f"
        )
        if changed:
            self._iss_map_offset_x_m = float(self._iss_map_offset_x_m)
        imgui.set_next_item_width(220.0)
        changed, self._iss_map_offset_y_m = imgui.input_float(
            "Offset Y (m)", self._iss_map_offset_y_m, step=0.1, format="%.3f"
        )
        if changed:
            self._iss_map_offset_y_m = float(self._iss_map_offset_y_m)
        imgui.set_next_item_width(220.0)
        changed, self._iss_map_scale_x_px_per_m = imgui.input_float(
            "Scale X (px/m)",
            self._iss_map_scale_x_px_per_m,
            step=0.1,
            format="%.3f",
        )
        if changed:
            self._iss_map_scale_x_px_per_m = max(
                0.01, float(self._iss_map_scale_x_px_per_m)
            )
        imgui.set_next_item_width(220.0)
        changed, self._iss_map_scale_y_px_per_m = imgui.input_float(
            "Scale Y (px/m)",
            self._iss_map_scale_y_px_per_m,
            step=0.1,
            format="%.3f",
        )
        if changed:
            self._iss_map_scale_y_px_per_m = max(
                0.01, float(self._iss_map_scale_y_px_per_m)
            )
        imgui.set_next_item_width(220.0)
        changed, self._iss_map_yaw_offset_deg = imgui.input_float(
            "Yaw offset (deg)",
            self._iss_map_yaw_offset_deg,
            step=0.5,
            format="%.3f",
        )
        if changed:
            self._iss_map_yaw_offset_deg = float(self._iss_map_yaw_offset_deg)
        imgui.set_next_item_width(220.0)
        changed, self._iss_map_marker_radius_px = imgui.input_float(
            "Marker radius (px)",
            self._iss_map_marker_radius_px,
            step=0.5,
            format="%.3f",
        )
        if changed:
            self._iss_map_marker_radius_px = max(
                2.0, float(self._iss_map_marker_radius_px)
            )
        imgui.set_next_item_width(220.0)
        changed, self._iss_map_heading_len_px = imgui.input_float(
            "Heading len (px)",
            self._iss_map_heading_len_px,
            step=1.0,
            format="%.3f",
        )
        if changed:
            self._iss_map_heading_len_px = max(8.0, float(self._iss_map_heading_len_px))
        if imgui.button("Reset map defaults"):
            self._iss_map_offset_x_m = float(ISS_MAP_ROBOT_OFFSET_XY_M[0])
            self._iss_map_offset_y_m = float(ISS_MAP_ROBOT_OFFSET_XY_M[1])
            self._iss_map_scale_x_px_per_m = float(ISS_MAP_ROBOT_SCALE_PX_PER_M[0])
            self._iss_map_scale_y_px_per_m = float(ISS_MAP_ROBOT_SCALE_PX_PER_M[1])
            self._iss_map_yaw_offset_deg = float(ISS_MAP_YAW_OFFSET_DEG)
            self._iss_map_marker_radius_px = float(ISS_MAP_MARKER_RADIUS_PX)
            self._iss_map_heading_len_px = float(ISS_MAP_HEADING_LEN_PX)
        imgui.end_child()

        imgui.next_column()
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
        imgui.columns(1)

        imgui.end()
        if bool(status.get("direct_mode", False)):
            return

        # Top-center mode indicator
        is_follow = status.get("mode") == "Follow Mode"
        mode_label = "FOLLOW MODE" if is_follow else "GOAL MODE"
        mode_rgb = (0.25, 0.55, 0.92) if is_follow else (0.22, 0.82, 0.48)
        draw = imgui.get_foreground_draw_list()
        if draw is not None:
            text_sz = imgui.calc_text_size(mode_label)
            chip_w = text_sz.x * 1.6 + 64.0
            chip_h = text_sz.y * 1.6 + 32.0
            chip_x0 = (scr_w - chip_w) * 0.5
            chip_y0 = pad
            chip_x1 = chip_x0 + chip_w
            chip_y1 = chip_y0 + chip_h
            shadow_col = imgui.get_color_u32((0.0, 0.0, 0.0, 0.45))
            bg_col = imgui.get_color_u32((0.06, 0.08, 0.12, 0.92))
            border_col = imgui.get_color_u32((*mode_rgb, 0.85))
            text_col = imgui.get_color_u32((*mode_rgb, 1.0))
            draw.add_rect_filled(
                (chip_x0 + 2.0, chip_y0 + 3.0),
                (chip_x1 + 2.0, chip_y1 + 3.0),
                shadow_col, 8.0,
            )
            draw.add_rect_filled((chip_x0, chip_y0), (chip_x1, chip_y1), bg_col, 8.0)
            draw.add_rect((chip_x0, chip_y0), (chip_x1, chip_y1), border_col, 8.0, 0, 1.5)
            draw.add_text(
                (chip_x0 + (chip_w - text_sz.x) * 0.5, chip_y0 + (chip_h - text_sz.y) * 0.5),
                text_col, mode_label,
            )

    def save_imgui_settings(self) -> None:
        """Persist ImGui layout to disk if initialized."""
        if not self._imgui_ready:
            return
        try:
            self._imgui_ini_path.parent.mkdir(parents=True, exist_ok=True)
            imgui.save_ini_settings_to_disk(str(self._imgui_ini_path))
        except Exception:
            pass

    def _init_iss_map_texture(self) -> None:
        """Load and register the ISS map texture for ImGui image rendering."""
        if not self._iss_map_path.is_file():
            return
        try:
            texture = self.base.loader.loadTexture(str(self._iss_map_path))
            if texture is None:
                return
            existing_ids = {
                int(k)
                for k in getattr(self.base.imgui, "textures", {}).keys()
                if isinstance(k, int) and 0 < k < 2_147_483_647
            }
            # Keep app-provided texture IDs in a high reserved range so they never
            # collide with imgui internal IDs (font atlas and dynamic uploads).
            tex_id = 2_000_000_000
            while tex_id in existing_ids and tex_id > 1_500_000_000:
                tex_id -= 1
            if tex_id in existing_ids:
                return
            self.base.imgui.textures[tex_id] = texture
            self._iss_map_texture = texture
            self._iss_map_tex_id = tex_id
        except Exception:
            self._iss_map_texture = None
            self._iss_map_tex_id = None
