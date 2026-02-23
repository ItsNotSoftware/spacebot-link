"""Rendering utilities: lights, camera lens, background, and avatar setup."""

from __future__ import annotations

from math import pi, sin, cos
from pathlib import Path
from typing import Optional, Tuple, List, Any

from panda3d.core import (
    AmbientLight,
    Camera,
    CardMaker,
    DirectionalLight,
    LineSegs,
    NodePath,
    PerspectiveLens,
    Quat,
    Texture,
    TextureStage,
    Vec3,
    Vec4,
    TransparencyAttrib,
)
from direct.showbase.ShowBase import ShowBase

from avatar import Avatar
from config import (
    CAMERA_UP_OFFSET_M,
    PATH_GHOST_MODEL,
    PATH_ANIM_INSTANCES,
    PATH_ANIM_LINE_ENABLED,
    PATH_ANIM_SPEED,
    PATH_LINE_COLOR,
    PATH_LINE_SAMPLE_SPACING_M,
    PATH_LINE_STRIDE,
    PATH_LINE_THICKNESS,
    PATH_MARKER_SPACING_M,
    PATH_MODE_DEFAULT,
    PATH_POSE_STRIDE,
    PATH_GHOST_SKIP_START,
    PATH_PLANE_SIZE,
    PATH_PLANE_OUTLINE_COLOR,
    PATH_PLANE_FILL_ALPHA,
    PATH_PLANE_THICKNESS,
    AVATAR_CAMERA_OFFSET,
    FLOOR_SHADOW_BASE_RADIUS,
    FLOOR_SHADOW_INNER_RATIO,
    FLOOR_SHADOW_MIN_SCALE,
    FLOOR_SHADOW_MAX_SCALE,
    FLOOR_SHADOW_NEAR_DIST,
    FLOOR_SHADOW_FAR_DIST,
    FLOOR_SHADOW_COLOR,
    FLOOR_SHADOW_THICKNESS,
    FLOOR_LINE_COLOR,
    FLOOR_LINE_THICKNESS,
    CAMERA_WIDTH_PX,
    CAMERA_HEIGHT_PX,
    CAMERA_FX,
    CAMERA_FY,
    CAMERA_CX,
    CAMERA_CY,
    ORIENT_PREVIEW_ENABLED,
    ORIENT_PREVIEW_REGION,
    ORIENT_PREVIEW_BG,
    ORIENT_PREVIEW_CROP_TOP,
    ORIENT_PREVIEW_MODEL,
    ORIENT_PREVIEW_TARGET_SIZE,
    ORIENT_PREVIEW_CAMERA_DISTANCE,
    ORIENT_PREVIEW_CAMERA_HEIGHT,
    ORIENT_PREVIEW_AVATAR_COLOR,
    ORIENT_PREVIEW_EXTRA_YAW_DEG,
)
from utils import (
    apply_opencv_intrinsics_to_lens,
    panda_pose_to_ros,
    ros_orientation_to_panda_hpr,
    ros_orientation_to_panda_quat,
)

PoseTuple = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


class Renderer:
    """Owns scene setup and avatar/camera helpers."""

    def __init__(self, base: ShowBase, gltf_model: str) -> None:
        """Initialize rendering pipeline, background card, and avatar."""
        self.base = base
        self.base.disableMouse()
        self.base.render.setShaderAuto()

        self._path_markers: List[NodePath] = []
        self._path_proto: Optional[NodePath] = None
        self._path_proto_failed: bool = False
        self._plane_proto: Optional[NodePath] = None
        self._bg_aspect: float = 0.0
        self._path_line: Optional[NodePath] = None
        self._anim_nps: List[NodePath] = []
        self._anim_task_name = "PathGhostAnim"
        self._anim_path: List[PoseTuple] = []
        self._anim_dist: List[float] = []
        self._anim_t: float = 0.0
        self._anim_length: float = 0.0
        self.anim_instances: int = max(1, int(PATH_ANIM_INSTANCES))
        self.anim_line_enabled: bool = bool(PATH_ANIM_LINE_ENABLED)
        self.path_mode: str = (
            PATH_MODE_DEFAULT  # poses | poses_line | planes | animated
        )
        self.pose_stride: int = PATH_POSE_STRIDE
        self.line_stride: int = PATH_LINE_STRIDE
        self.marker_spacing_m: float = max(0.05, float(PATH_MARKER_SPACING_M))
        self.line_sample_spacing_m: float = max(0.05, float(PATH_LINE_SAMPLE_SPACING_M))
        self.anim_speed: float = PATH_ANIM_SPEED  # units per second
        self._path_goodness: Optional[float] = None
        self._path_local_risks: Optional[List[float]] = None

        self._init_lights()
        self._make_bg_card(initial_aspect=9 / 16)

        model_path = self._resolve_asset_path(gltf_model)
        if model_path is None:
            raise FileNotFoundError(f"Could not resolve GLTF model: {gltf_model}")
        self.avatar = Avatar(self.base.render, self.base.loader, str(model_path))
        self.set_avatar_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self._floor_shadow: Optional[NodePath] = None
        self._floor_line: Optional[NodePath] = None
        self._init_floor_indicator()
        self._orient_region = None
        self._orient_scene: Optional[NodePath] = None
        self._orient_cam: Optional[NodePath] = None
        self._orient_preview: Optional[NodePath] = None
        self._orient_enabled: bool = False
        self._orient_ros_axes_hpr = (90.0, 0.0, 0.0)  # ROS X->Panda Y, ROS Y->Panda -X

        # reasonable default intrinsics (updated once we see cam_info)
        self._init_default_lens()
        preview_model = self._resolve_asset_path(ORIENT_PREVIEW_MODEL)
        self._init_orientation_preview(preview_model or model_path)

    # ---- lights / lens / background ----
    def _init_lights(self) -> None:
        """Add a simple directional + ambient light setup."""
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1, 1, 1, 1))
        sun_np = self.base.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self.base.render.setLight(sun_np)
        amb = AmbientLight("ambient")
        amb.setColor(Vec4(0.35, 0.35, 0.35, 1))
        amb_np = self.base.render.attachNewNode(amb)
        self.base.render.setLight(amb_np)

    def _init_default_lens(self) -> None:
        """Seed a reasonable default lens configuration until camera info arrives."""
        apply_opencv_intrinsics_to_lens(
            self.base.camLens,
            CAMERA_WIDTH_PX,
            CAMERA_HEIGHT_PX,
            CAMERA_FX,
            CAMERA_FY,
            CAMERA_CX,
            CAMERA_CY,
        )
        self.base.camLens.setNear(0.1)  # type: ignore
        self.base.camLens.setFar(5000.0)  # type: ignore
        self._update_bg_scale()

    def _make_bg_card(self, initial_aspect: float) -> None:
        """Create a textured quad behind the scene for the camera feed."""
        if self.base.camera is None:
            return
        cm = CardMaker("background")
        cm.setFrame(-1, 1, -initial_aspect, initial_aspect)
        self.bg_card: NodePath = self.base.camera.attachNewNode(cm.generate())
        self.bg_card.setScale(50)
        self.bg_card.setPos(0, 100, 0)
        self.bg_card.setBin("background", 0)
        self.base.setBackgroundColor(0, 0, 0, 1)
        self.bg_card.setDepthWrite(False)
        self.bg_card.setDepthTest(False)
        self.bg_tex: Texture = Texture("background")
        self.bg_tex.setup2dTexture(2, 2, Texture.T_unsigned_byte, Texture.F_rgb)
        self.bg_card.setTexture(self.bg_tex)
        self._bg_aspect = float(initial_aspect)
        ts = TextureStage.getDefault()
        self.bg_card.setTexScale(ts, 1, -1)
        self.bg_card.setTexOffset(ts, 0, 1)
        self._update_bg_scale()

    # ---- orientation preview ----
    def _init_orientation_preview(self, model_path: Path) -> None:
        """Create a small Panda3D preview window for avatar orientation."""
        if not ORIENT_PREVIEW_ENABLED:
            return
        if self.base.win is None:
            return
        try:
            x0, x1, y0, y1 = ORIENT_PREVIEW_REGION
        except Exception:
            return
        if x1 <= x0 or y1 <= y0:
            return

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

        try:
            dr = self.base.win.makeDisplayRegion(x0, x1, y0, y1)
        except Exception:
            return
        dr.setSort(20)
        dr.setClearDepthActive(True)
        dr.setClearColorActive(True)
        dr.setClearColor(Vec4(*ORIENT_PREVIEW_BG))
        self._orient_region = dr

        scene = NodePath("orientation_scene")
        self._orient_scene = scene
        scene.setShaderAuto()

        lens = PerspectiveLens()
        lens.setFov(30.0)
        lens.setNearFar(0.05, 50.0)
        cam = Camera("orientation_cam", lens)
        cam_np = scene.attachNewNode(cam)
        dr.setCamera(cam_np)
        self._orient_cam = cam_np

        amb = AmbientLight("orientation_ambient")
        amb.setColor(Vec4(0.55, 0.55, 0.55, 1.0))
        amb_np = scene.attachNewNode(amb)
        scene.setLight(amb_np)
        sun = DirectionalLight("orientation_sun")
        sun.setColor(Vec4(0.8, 0.8, 0.8, 1.0))
        sun_np = scene.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        scene.setLight(sun_np)

        try:
            proto_model = self.base.loader.loadModel(str(model_path))
        except Exception:
            proto_model = None
        if proto_model is None or proto_model.isEmpty():
            return

        container = NodePath("orientation_proto")
        proto_model.reparentTo(container)
        proto_model.setHpr(float(ORIENT_PREVIEW_EXTRA_YAW_DEG), 0.0, 0.0)

        max_dim = 1.0
        try:
            bounds = proto_model.getTightBounds()
            if bounds is not None and bounds[0] is not None and bounds[1] is not None:
                mn, mx = bounds
                cx = 0.5 * (mn.x + mx.x)
                cy = 0.5 * (mn.y + mx.y)
                cz = 0.5 * (mn.z + mx.z)
                proto_model.setPos(-cx, -cy, -cz)
                sx = float(mx.x - mn.x)
                sy = float(mx.y - mn.y)
                sz = float(mx.z - mn.z)
                max_dim = max(1e-6, max(sx, sy, sz))
        except Exception:
            max_dim = 1.0

        target = max(0.05, float(ORIENT_PREVIEW_TARGET_SIZE))
        container.setScale(target / max_dim)
        preview_anchor = scene.attachNewNode("orientation_preview_anchor")
        preview_axes = preview_anchor.attachNewNode("orientation_preview_axes")
        preview_axes.setHpr(*self._orient_ros_axes_hpr)

        preview_offset = preview_axes.attachNewNode("orientation_preview_offset")
        # Rotate preview frame to align pitch/roll axes with avatar controls.
        preview_offset.setHpr(90.0, 0.0, 0.0)

        preview_pose = preview_offset.attachNewNode("orientation_preview_pose")
        container.copyTo(preview_pose)
        preview_pose.setColorScale(*ORIENT_PREVIEW_AVATAR_COLOR)

        self._orient_preview = preview_pose

        cam_dist = float(ORIENT_PREVIEW_CAMERA_DISTANCE) * target
        cam_height = float(ORIENT_PREVIEW_CAMERA_HEIGHT) * target
        center = Vec3(0.0, 0.0, 0.0)
        try:
            bounds = container.getTightBounds()
            if bounds is not None and bounds[0] is not None and bounds[1] is not None:
                mn, mx = bounds
                center = Vec3(
                    0.5 * (mn.x + mx.x),
                    0.5 * (mn.y + mx.y),
                    0.5 * (mn.z + mx.z),
                )
        except Exception:
            center = Vec3(0.0, 0.0, 0.0)
        cam_np.setPos(center.x, center.y - cam_dist, center.z + cam_height)
        cam_np.lookAt(center)

        self._orient_enabled = True

    def update_orientation_preview(
        self,
        robot_ros_orientation: Optional[dict],
        avatar_hpr: Optional[Tuple[float, float, float]],
    ) -> None:
        """Update preview to show avatar orientation in the robot frame (ROS)."""
        if not self._orient_enabled:
            return
        if self._orient_preview is None:
            return
        # Models are parented under a fixed ROS-axes alignment node. To make the
        # preview match the main scene's Panda-world HPR, we must compensate for
        # that parent transform.
        axes_quat = Quat()
        axes_quat.setHpr(self._orient_ros_axes_hpr)
        axes_inv = axes_quat.conjugate()
        if robot_ros_orientation is None or avatar_hpr is None:
            self._orient_preview.hide()
            return
        avatar_ros = panda_pose_to_ros(((0.0, 0.0, 0.0), avatar_hpr))
        if not isinstance(avatar_ros, dict):
            self._orient_preview.hide()
            return
        avatar_ori = avatar_ros.get("orientation")
        if not isinstance(avatar_ori, dict):
            self._orient_preview.hide()
            return

        def _quat_from_ros(ori: dict) -> Optional[Tuple[float, float, float, float]]:
            try:
                return (
                    float(ori.get("w")),
                    float(ori.get("x")),
                    float(ori.get("y")),
                    float(ori.get("z")),
                )
            except (TypeError, ValueError):
                return None

        def _quat_multiply(
            lhs: Tuple[float, float, float, float],
            rhs: Tuple[float, float, float, float],
        ) -> Tuple[float, float, float, float]:
            w1, x1, y1, z1 = lhs
            w2, x2, y2, z2 = rhs
            return (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )

        def _quat_conjugate(
            q: Tuple[float, float, float, float],
        ) -> Tuple[float, float, float, float]:
            w, x, y, z = q
            return (w, -x, -y, -z)

        robot_q = _quat_from_ros(robot_ros_orientation)
        avatar_q = _quat_from_ros(avatar_ori)
        if robot_q is None or avatar_q is None:
            self._orient_preview.hide()
            return

        relative_q = _quat_multiply(_quat_conjugate(robot_q), avatar_q)
        rel_quat = ros_orientation_to_panda_quat(
            {
                "x": relative_q[1],
                "y": relative_q[2],
                "z": relative_q[3],
                "w": relative_q[0],
            }
        )
        if rel_quat is None:
            self._orient_preview.hide()
            return
        desired = rel_quat
        self._orient_preview.show()
        rel = axes_inv * desired
        h, p, r = rel.getHpr()
        fixed = Quat()
        fixed.setHpr((float(h), float(-p), float(-r)))
        self._orient_preview.setQuat(fixed)

    def _init_floor_indicator(self) -> None:
        """Initialize the floor height shadow and line."""
        self._floor_shadow = self._make_floor_shadow()
        if self._floor_shadow is not None:
            self._floor_shadow.hide()

    def _make_floor_shadow(self) -> Optional[NodePath]:
        """Create a donut-style floor shadow marker (two concentric circles)."""
        outer_r = max(0.05, float(FLOOR_SHADOW_BASE_RADIUS))
        inner_r = max(0.02, float(FLOOR_SHADOW_INNER_RATIO) * outer_r)
        segments = 64

        def _make_ring(radius: float, name: str) -> Optional[NodePath]:
            segs = LineSegs(name)
            segs.setThickness(FLOOR_SHADOW_THICKNESS)
            segs.setColor(*FLOOR_SHADOW_COLOR)
            first = True
            for i in range(segments + 1):
                t = (2.0 * 3.14159265) * (i / segments)
                x = radius * float(cos(t))
                z = radius * float(sin(t))
                if first:
                    segs.moveTo(x, 0.0, z)
                    first = False
                else:
                    segs.drawTo(x, 0.0, z)
            node = segs.create()
            if node is None:
                return None
            return NodePath(node)

        from math import cos, sin

        container = NodePath("floor_shadow")
        outer = _make_ring(outer_r, "floor_shadow_outer")
        inner = _make_ring(inner_r, "floor_shadow_inner")
        if outer is None or inner is None:
            return None
        outer.reparentTo(container)
        inner.reparentTo(container)
        container.reparentTo(self.base.render)
        container.setTransparency(TransparencyAttrib.MAlpha)
        container.setBin("fixed", 8)
        container.setDepthWrite(False)
        container.setDepthTest(False)
        return container

    def _update_bg_scale(self) -> None:
        """Compute card scale so it fills the current camera frustum."""
        if not hasattr(self, "bg_card") or self.base.camLens is None:
            return
        d = abs(self.bg_card.getY())
        fov_x, fov_y = self.base.camLens.getFov()
        fov_y_rad = fov_y * (pi / 180.0)
        half_h = d * (
            sin(fov_y_rad / 2.0) / (1e-9 + (1.0 - 0.5 * (fov_y_rad**2) / 3.0))
        )
        if getattr(self, "_bg_aspect", 0) > 0:
            s = half_h / self._bg_aspect
            self.bg_card.setScale(s)

    def update_bg_scale(self) -> None:
        """Public entrypoint to rescale the background card after FOV changes."""
        self._update_bg_scale()

    def _resolve_asset_path(self, gltf_path: str) -> Optional[Path]:
        path = Path(gltf_path)
        if path.exists():
            return path
        fallback = (
            Path(__file__).resolve().parent / ".." / "assets" / Path(gltf_path).name
        ).resolve()
        return fallback if fallback.exists() else None

    def _draw_path_line(
        self,
        poses: List[PoseTuple],
        color: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Draw a layered line strip to improve readability and depth cues."""
        line_poses = self._line_poses_by_distance(poses)
        if len(line_poses) < 2:
            return
        base_color = color or self._path_line_color()
        seg_count = max(0, len(line_poses) - 1)
        segment_risks = self._segment_risks_for_poses(line_poses)
        if len(segment_risks) < seg_count:
            segment_risks.extend([0.0] * (seg_count - len(segment_risks)))

        def _make_layer(name: str, thickness: float, color_fn) -> Optional[NodePath]:
            segs = LineSegs(name)
            segs.setThickness(max(1.0, thickness))
            for idx in range(seg_count):
                (x1, y1, z1), _ = line_poses[idx]
                (x2, y2, z2), _ = line_poses[idx + 1]
                segs.setColor(*color_fn(idx, seg_count))
                segs.moveTo(x1, y1, z1)
                segs.drawTo(x2, y2, z2)
            node = segs.create()
            if node is None:
                return None
            np_line = self.base.render.attachNewNode(node)
            np_line.setDepthWrite(False)
            np_line.setDepthTest(False)
            return np_line

        def _make_endpoint_markers() -> Optional[NodePath]:
            """Draw start/end markers to improve path readability."""
            if len(line_poses) < 2:
                return None
            segs = LineSegs("path_line_endpoints")
            segs.setThickness(max(2.0, PATH_LINE_THICKNESS - 0.5))
            (sx, sy, sz), _ = line_poses[0]
            (ex, ey, ez), _ = line_poses[-1]
            ring_r_start = 0.09
            ring_r_end = 0.13
            ring_steps = 18
            # Start marker: cyan ring.
            segs.setColor(0.15, 0.92, 1.0, 0.95)
            for i in range(ring_steps + 1):
                t = (2.0 * pi * i) / ring_steps
                x = sx + ring_r_start * cos(t)
                y = sy + ring_r_start * sin(t)
                z = sz + 0.015
                if i == 0:
                    segs.moveTo(x, y, z)
                else:
                    segs.drawTo(x, y, z)
            # End marker: larger ring + crosshair.
            segs.setColor(1.0, 0.96, 0.92, 0.98)
            for i in range(ring_steps + 1):
                t = (2.0 * pi * i) / ring_steps
                x = ex + ring_r_end * cos(t)
                y = ey + ring_r_end * sin(t)
                z = ez + 0.02
                if i == 0:
                    segs.moveTo(x, y, z)
                else:
                    segs.drawTo(x, y, z)
            arm = 0.07
            segs.moveTo(ex - arm, ey, ez + 0.02)
            segs.drawTo(ex + arm, ey, ez + 0.02)
            segs.moveTo(ex, ey - arm, ez + 0.02)
            segs.drawTo(ex, ey + arm, ez + 0.02)
            node = segs.create()
            if node is None:
                return None
            np_markers = self.base.render.attachNewNode(node)
            np_markers.setDepthWrite(False)
            np_markers.setDepthTest(False)
            return np_markers

        def _make_direction_chevrons() -> Optional[NodePath]:
            """Draw small chevrons along the line to indicate travel direction."""
            if len(line_poses) < 2:
                return None
            # Build cumulative distances on the rendered line.
            cumulative = [0.0]
            total = 0.0
            for i in range(1, len(line_poses)):
                (x1, y1, z1), _ = line_poses[i - 1]
                (x2, y2, z2), _ = line_poses[i]
                d = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
                total += d
                cumulative.append(total)
            if total < 0.35:
                return None

            def _sample_at(dist: float) -> Tuple[Vec3, Vec3]:
                dist = max(0.0, min(total, dist))
                seg_idx = 0
                while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < dist:
                    seg_idx += 1
                if seg_idx + 1 >= len(line_poses):
                    seg_idx = max(0, len(line_poses) - 2)
                d0 = cumulative[seg_idx]
                d1 = cumulative[seg_idx + 1]
                span = max(1e-6, d1 - d0)
                u = (dist - d0) / span
                (x1, y1, z1), _ = line_poses[seg_idx]
                (x2, y2, z2), _ = line_poses[seg_idx + 1]
                p = Vec3(
                    x1 + (x2 - x1) * u,
                    y1 + (y2 - y1) * u,
                    z1 + (z2 - z1) * u,
                )
                tangent = Vec3(x2 - x1, y2 - y1, z2 - z1)
                if tangent.length_squared() < 1e-8:
                    tangent = Vec3(0, 1, 0)
                tangent.normalize()
                return p, tangent

            segs = LineSegs("path_line_chevrons")
            segs.setThickness(max(1.5, PATH_LINE_THICKNESS - 1.0))
            spacing = 0.55
            start = min(0.35, 0.2 * total)
            end = max(start, total - 0.25)
            d = start
            while d < end:
                p, tangent = _sample_at(d)
                progress = d / max(1e-6, total)
                # Use a stable frame for the chevron plane.
                up = Vec3(0, 0, 1)
                side = tangent.cross(up)
                if side.length_squared() < 1e-6:
                    side = tangent.cross(Vec3(1, 0, 0))
                if side.length_squared() < 1e-6:
                    d += spacing
                    continue
                side.normalize()
                chevron_len = 0.11
                chevron_w = 0.07
                apex = p + tangent * (0.5 * chevron_len) + Vec3(0, 0, 0.01)
                tail = p - tangent * (0.5 * chevron_len) + Vec3(0, 0, 0.01)
                left = tail + side * chevron_w
                right = tail - side * chevron_w
                alpha = max(0.35, 0.95 - 0.40 * progress)
                segs.setColor(1.0, 1.0, 1.0, alpha)
                segs.moveTo(left)
                segs.drawTo(apex)
                segs.drawTo(right)
                d += spacing
            node = segs.create()
            if node is None:
                return None
            np_chev = self.base.render.attachNewNode(node)
            np_chev.setDepthWrite(False)
            np_chev.setDepthTest(False)
            return np_chev

        def _risk_to_color(risk: float) -> Tuple[float, float, float, float]:
            risk = max(0.0, min(1.0, float(risk)))
            red = (0.95, 0.22, 0.20, 1.0)
            yellow = (1.00, 0.78, 0.10, 1.0)
            green = (0.22, 0.92, 0.38, 1.0)
            # low risk -> green, high risk -> red
            if risk < 0.5:
                t = risk / 0.5
                c0, c1 = green, yellow
            else:
                t = (risk - 0.5) / 0.5
                c0, c1 = yellow, red
            return tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(4))  # type: ignore[return-value]

        container = self.base.render.attachNewNode("path_line_layers")
        r, g, b, a = base_color
        outer = _make_layer(
            "path_line_outer",
            PATH_LINE_THICKNESS + 2.5,
            lambda _idx, _n: (0.0, 0.0, 0.0, min(0.55, a)),
        )
        if any(v > 1e-6 for v in segment_risks):
            def _inner_color(idx: int, nseg: int) -> Tuple[float, float, float, float]:
                progress = (idx + 0.5) / max(1, nseg)
                rr, gg, bb, _ = _risk_to_color(segment_risks[idx])
                # Add subtle progression-based fade to improve depth/readability.
                alpha = max(0.35, a * (0.70 + 0.30 * (1.0 - 0.35 * progress)))
                brighten = 1.0 - 0.12 * progress
                return (rr * brighten, gg * brighten, bb * brighten, alpha)
        else:
            def _inner_color(idx: int, nseg: int) -> Tuple[float, float, float, float]:
                progress = (idx + 0.5) / max(1, nseg)
                alpha = max(0.35, a * (0.70 + 0.30 * (1.0 - 0.35 * progress)))
                brighten = 1.0 - 0.12 * progress
                return (r * brighten, g * brighten, b * brighten, alpha)
        inner = _make_layer("path_line_inner", PATH_LINE_THICKNESS, _inner_color)
        endpoint_markers = _make_endpoint_markers()
        chevrons = _make_direction_chevrons()
        for idx, child in enumerate((outer, inner, endpoint_markers, chevrons)):
            if child is None:
                continue
            child.reparentTo(container)
            child.setBin("fixed", 4 + idx)
        self._path_line = container

    def _path_line_color(self) -> Tuple[float, float, float, float]:
        """Return line color derived from path goodness, or the default if unavailable."""
        if self._path_goodness is None:
            return PATH_LINE_COLOR
        r, g, b, _ = self._path_tint_rgba()
        return (r, g, b, PATH_LINE_COLOR[3])

    def _path_tint_rgba(self) -> Tuple[float, float, float, float]:
        """Map path goodness [0,1] to a red-yellow-green tint."""
        if self._path_goodness is None:
            return (1.0, 1.0, 1.0, 1.0)
        goodness = max(0.0, min(1.0, float(self._path_goodness)))
        red = (0.95, 0.22, 0.20, 1.0)
        yellow = (1.00, 0.78, 0.10, 1.0)
        green = (0.22, 0.92, 0.38, 1.0)
        if goodness < 0.5:
            t = goodness / 0.5
            c0, c1 = red, yellow
        else:
            t = (goodness - 0.5) / 0.5
            c0, c1 = yellow, green
        return tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(4))  # type: ignore[return-value]

    def set_path_goodness(self, goodness: Optional[float]) -> None:
        """Store latest path goodness score used to tint path visuals."""
        if goodness is None:
            self._path_goodness = None
            return
        try:
            self._path_goodness = max(0.0, min(1.0, float(goodness)))
        except Exception:
            self._path_goodness = None

    def set_path_local_risks(self, local_risks: Optional[List[Any]]) -> None:
        """Store MC local risk samples (0..1) used for heatmapped path line rendering."""
        if not isinstance(local_risks, list):
            self._path_local_risks = None
            return
        cleaned: List[float] = []
        for v in local_risks:
            try:
                cleaned.append(max(0.0, min(1.0, float(v))))
            except Exception:
                continue
        self._path_local_risks = cleaned or None

    def _line_poses_by_distance(self, poses: List[PoseTuple]) -> List[PoseTuple]:
        """Decimate line rendering by distance to reduce draw cost while preserving shape."""
        if len(poses) <= 2:
            return list(poses)
        spacing = max(0.05, float(self.line_sample_spacing_m))
        out: List[PoseTuple] = [poses[0]]
        dist_acc = 0.0
        next_target = spacing
        for idx in range(1, len(poses)):
            (x1, y1, z1), _ = poses[idx - 1]
            (x2, y2, z2), _ = poses[idx]
            seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
            dist_acc += seg_len
            if dist_acc + 1e-6 >= next_target:
                out.append(poses[idx])
                while next_target <= dist_acc:
                    next_target += spacing
        if out[-1] != poses[-1]:
            out.append(poses[-1])
        return out

    def _segment_risks_for_poses(self, poses: List[PoseTuple]) -> List[float]:
        """Map local risk samples to rendered path segments."""
        if len(poses) < 2:
            return []
        risks = self._path_local_risks
        seg_count = len(poses) - 1
        if not risks:
            return [0.0] * seg_count
        if len(risks) == 1:
            return [risks[0]] * seg_count
        mapped: List[float] = []
        last = len(risks) - 1
        for idx in range(seg_count):
            t = (idx + 0.5) / max(1, seg_count)
            src_f = t * last
            i0 = int(src_f)
            i1 = min(last, i0 + 1)
            frac = src_f - i0
            val = risks[i0] + (risks[i1] - risks[i0]) * frac
            mapped.append(max(0.0, min(1.0, val)))
        return mapped

    def _resample_poses_by_distance(self, poses: List[PoseTuple]) -> List[PoseTuple]:
        """Resample path poses at fixed arc-length spacing for consistent preview density."""
        if len(poses) <= 2:
            return list(poses)
        spacing = max(0.05, float(self.marker_spacing_m))

        cumulative = [0.0]
        total = 0.0
        for i in range(1, len(poses)):
            (x1, y1, z1), _ = poses[i - 1]
            (x2, y2, z2), _ = poses[i]
            d = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
            total += d
            cumulative.append(total)

        if total <= 1e-6:
            return [poses[0], poses[-1]] if poses[0] != poses[-1] else [poses[0]]

        targets: List[float] = [0.0]
        t = spacing
        while t < total - 1e-6:
            targets.append(t)
            t += spacing
        if targets[-1] != total:
            targets.append(total)

        out: List[PoseTuple] = []
        seg_idx = 0
        for target in targets:
            while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < target:
                seg_idx += 1
            if seg_idx + 1 >= len(poses):
                out.append(poses[-1])
                continue
            s0 = cumulative[seg_idx]
            s1 = cumulative[seg_idx + 1]
            span = max(1e-6, s1 - s0)
            u = max(0.0, min(1.0, (target - s0) / span))
            (x1, y1, z1), (h1, p1, r1) = poses[seg_idx]
            (x2, y2, z2), (h2, p2, r2) = poses[seg_idx + 1]
            pos = (
                x1 + (x2 - x1) * u,
                y1 + (y2 - y1) * u,
                z1 + (z2 - z1) * u,
            )
            hpr = (
                h1 + (h2 - h1) * u,
                p1 + (p2 - p1) * u,
                r1 + (r2 - r1) * u,
            )
            out.append((pos, hpr))
        return out

    def _make_plane_proto(self) -> Optional[NodePath]:
        """Build an outlined + translucent plane for pose visualization."""
        w, h = PATH_PLANE_SIZE
        half_w = 0.5 * max(0.05, float(w))
        half_h = 0.5 * max(0.05, float(h))

        outline = LineSegs("path_plane_outline")
        outline.setThickness(PATH_PLANE_THICKNESS)
        outline.setColor(*PATH_PLANE_OUTLINE_COLOR)
        outline.moveTo(-half_w, 0.0, -half_h)
        outline.drawTo(half_w, 0.0, -half_h)
        outline.drawTo(half_w, 0.0, half_h)
        outline.drawTo(-half_w, 0.0, half_h)
        outline.drawTo(-half_w, 0.0, -half_h)
        outline_node = outline.create()
        if outline_node is None:
            return None

        container = NodePath("path_plane_proto")
        outline_np = container.attachNewNode(outline_node)
        outline_np.setBin("fixed", 6)
        outline_np.setDepthWrite(False)
        outline_np.setDepthTest(False)

        cm = CardMaker("path_plane_fill")
        cm.setFrame(-half_w, half_w, -half_h, half_h)
        fill_np = container.attachNewNode(cm.generate())
        fill_np.setP(-90.0)  # rotate into the XZ plane so normal points forward (+Y)
        fill_np.setTransparency(TransparencyAttrib.MAlpha)
        fill_np.setTwoSided(True)
        fill_np.setColor(
            PATH_PLANE_OUTLINE_COLOR[0],
            PATH_PLANE_OUTLINE_COLOR[1],
            PATH_PLANE_OUTLINE_COLOR[2],
            PATH_PLANE_FILL_ALPHA,
        )
        fill_np.setBin("fixed", 6)
        fill_np.setDepthWrite(False)
        fill_np.setDepthTest(False)

        container.setTransparency(TransparencyAttrib.MAlpha)
        container.setDepthWrite(False)
        container.setDepthTest(False)
        container.setBin("fixed", 6)
        return container

    # ---- animated ghost ----
    def _place_anim_at_distance(self, np_node: NodePath, dist: float) -> None:
        """Move animated ghost to a specific distance along the path."""
        if np_node is None or len(self._anim_path) < 2 or not self._anim_dist:
            return
        dist = max(0.0, min(dist, self._anim_length))
        lo, hi = 0, len(self._anim_dist) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._anim_dist[mid] <= dist:
                lo = mid + 1
            else:
                hi = mid
        idx = max(1, lo)
        prev_idx = idx - 1
        d0 = self._anim_dist[prev_idx]
        d1 = self._anim_dist[idx]
        seg_span = max(1e-6, d1 - d0)
        seg_t = (dist - d0) / seg_span
        (x1, y1, z1), hpr1 = self._anim_path[prev_idx]
        (x2, y2, z2), hpr2 = self._anim_path[idx]
        pos = (
            x1 + (x2 - x1) * seg_t,
            y1 + (y2 - y1) * seg_t,
            z1 + (z2 - z1) * seg_t,
        )
        h = hpr1[0] + (hpr2[0] - hpr1[0]) * seg_t
        p = hpr1[1] + (hpr2[1] - hpr1[1]) * seg_t
        r = hpr1[2] + (hpr2[2] - hpr1[2]) * seg_t
        np_node.setPos(self.base.render, *pos)
        np_node.setHpr(self.base.render, h, p, r)

    def _update_animation_path(self, poses: List[PoseTuple]) -> None:
        """Update animated ghost path without resetting its progress."""
        if len(poses) < 2:
            self._stop_path_animation()
            return
        if self._path_proto is None:
            self._path_proto = self.load_path_proto()
        proto = self._path_proto
        if proto is None:
            return
        progress = 0.0
        if getattr(self, "_anim_length", 0.0) > 1e-6:
            progress = min(1.0, self._anim_t / self._anim_length)

        required = max(1, int(self.anim_instances))
        while len(self._anim_nps) < required:
            ghost = proto.copyTo(self.base.render)
            ghost.setBin("fixed", 5)
            ghost.setDepthWrite(False)
            self._anim_nps.append(ghost)
        if len(self._anim_nps) > required:
            extras = self._anim_nps[required:]
            for np_ in extras:
                try:
                    np_.removeNode()
                except Exception:
                    pass
            self._anim_nps = self._anim_nps[:required]
        self._anim_path = poses
        self._anim_dist = [0.0]
        total = 0.0
        for i in range(1, len(poses)):
            (x1, y1, z1), _ = poses[i - 1]
            (x2, y2, z2), _ = poses[i]
            d = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
            total += d
            self._anim_dist.append(total)
        self._anim_length = max(1e-6, total)
        target_dist = progress * self._anim_length
        spacing = self._anim_length / max(1, len(self._anim_nps))
        for idx, ghost in enumerate(self._anim_nps):
            offset = spacing * idx
            dist = target_dist + offset
            if self._anim_length > 1e-6:
                dist = dist % self._anim_length
            self._place_anim_at_distance(ghost, dist)
        if not self.base.taskMgr.hasTaskNamed(self._anim_task_name):
            self.base.taskMgr.add(self._anim_task, self._anim_task_name)

    def _stop_path_animation(self) -> None:
        """Stop path animation and remove animated ghost."""
        try:
            self.base.taskMgr.remove(self._anim_task_name)
        except Exception:
            pass
        for np_ in self._anim_nps:
            try:
                np_.removeNode()
            except Exception:
                pass
        self._anim_nps = []
        self._anim_path = []
        self._anim_dist = []
        self._anim_t = 0.0
        self._anim_length = 0.0

    def _anim_task(self, task) -> int:
        """Advance animated ghost along the path with constant linear speed."""
        if not self._anim_nps or len(self._anim_path) < 2:
            return task.done
        dt = self.base.taskMgr.globalClock.getDt()
        speed = max(0.01, self.anim_speed)
        # advance distance along path
        self._anim_t += speed * dt
        if self._anim_t > self._anim_length:
            self._anim_t = self._anim_t % self._anim_length
        spacing = self._anim_length / max(1, len(self._anim_nps))
        for idx, ghost in enumerate(self._anim_nps):
            dist = self._anim_t + spacing * idx
            if self._anim_length > 1e-6:
                dist = dist % self._anim_length
            self._place_anim_at_distance(ghost, dist)
        return task.cont

    # ---- avatar helpers ----
    def get_avatar_pose(self) -> PoseTuple:
        """Return avatar pose (pos, hpr) in world coordinates."""
        return self.avatar.get_pose()

    def set_avatar_pose(
        self, pos: Tuple[float, float, float], hpr: Tuple[float, float, float]
    ) -> PoseTuple:
        """Set avatar pose in world coordinates and return the updated pose."""
        self.avatar.set_pos(*pos)
        self.avatar.set_hpr(*hpr)
        return self.avatar.get_pose()

    def reset_avatar_hpr(self) -> None:
        """Reset avatar orientation to its initial rotation."""
        self.avatar.reset_hpr()

    def reset_avatar_to_camera_hpr(self) -> None:
        """Align avatar orientation to current camera heading/pitch/roll."""
        if self.base.camera is None:
            return
        hpr = self.base.camera.getHpr(self.base.render)
        self.avatar.set_hpr(float(hpr[0]), float(hpr[1]), float(hpr[2]))

    def set_avatar_visible(self, visible: bool) -> None:
        """Toggle avatar visibility in the scene."""
        self.avatar.set_visible(visible)

    def set_avatar_opacity(self, alpha: float) -> None:
        """Adjust avatar transparency."""
        self.avatar.set_opacity(alpha)

    def set_avatar_color(self, rgba: Tuple[float, float, float, float]) -> None:
        """Set avatar color scale (RGBA)."""
        self.avatar.set_color(*rgba)

    def move_avatar(self, dx: float, dy: float, dz: float) -> None:
        """Translate avatar in world space by the given deltas."""
        self.avatar.move_world(dx, dy, dz)

    def add_avatar_hpr(self, dh: float, dp: float, dr: float) -> None:
        """Increment avatar orientation in its local frame."""
        self.avatar.add_hpr(dh, dp, dr)

    def update_floor_indicator(
        self,
        avatar_pos: Tuple[float, float, float],
        floor_pos: Tuple[float, float, float],
        axis: Tuple[float, float, float],
        distance_to_robot: float,
    ) -> None:
        """Update floor shadow + line using avatar position and floor cast."""
        if self._floor_shadow is None:
            return
        ax, ay, az = axis
        axis_len = (ax * ax + ay * ay + az * az) ** 0.5
        if axis_len < 1e-6:
            self.clear_floor_indicator()
            return
        ax /= axis_len
        ay /= axis_len
        az /= axis_len

        fx, fy, fz = floor_pos
        self._floor_shadow.setPos(fx, fy, fz)
        self._floor_shadow.lookAt(self._floor_shadow.getPos() + Vec3(ax, ay, az))
        if FLOOR_SHADOW_FAR_DIST <= FLOOR_SHADOW_NEAR_DIST:
            scale = 1.0
        else:
            t = (distance_to_robot - FLOOR_SHADOW_NEAR_DIST) / (
                FLOOR_SHADOW_FAR_DIST - FLOOR_SHADOW_NEAR_DIST
            )
            t = max(0.0, min(1.0, t))
            scale = FLOOR_SHADOW_MAX_SCALE + t * (
                FLOOR_SHADOW_MIN_SCALE - FLOOR_SHADOW_MAX_SCALE
            )
        self._floor_shadow.setScale(scale)
        self._floor_shadow.show()

        if self._floor_line is not None:
            try:
                self._floor_line.removeNode()
            except Exception:
                pass
            self._floor_line = None

        segs = LineSegs("floor_line")
        segs.setThickness(FLOOR_LINE_THICKNESS)
        segs.setColor(*FLOOR_LINE_COLOR)
        segs.moveTo(avatar_pos[0], avatar_pos[1], avatar_pos[2])
        segs.drawTo(fx, fy, fz)
        node = segs.create()
        if node is None:
            return
        self._floor_line = self.base.render.attachNewNode(node)
        self._floor_line.setBin("fixed", 9)
        self._floor_line.setDepthWrite(False)
        self._floor_line.setDepthTest(False)

    def clear_floor_indicator(self) -> None:
        """Hide floor shadow + line when no valid reading exists."""
        if self._floor_shadow is not None:
            self._floor_shadow.hide()
        if self._floor_line is not None:
            try:
                self._floor_line.removeNode()
            except Exception:
                pass
            self._floor_line = None

    # ---- camera helpers ----
    def set_camera_pose(
        self, pos: Tuple[float, float, float], hpr: Tuple[float, float, float]
    ) -> None:
        """Place the camera at the given world pose plus the camera offset."""
        if self.base.camera is None:
            return
        offset_x, offset_y, offset_z = AVATAR_CAMERA_OFFSET
        offset = Vec3(offset_x, offset_y, offset_z)
        rot = Quat()
        rot.setHpr(Vec3(hpr[0], hpr[1], hpr[2]))
        offset_world = rot.xform(offset)
        cx = pos[0] + offset_world[0]
        cy = pos[1] + offset_world[1]
        cz = pos[2] + offset_world[2]
        self.base.camera.setPos(self.base.render, cx, cy, cz)
        self.base.camera.setHpr(self.base.render, hpr[0], hpr[1], hpr[2])

    def get_camera_pose(self) -> Optional[PoseTuple]:
        """Return the camera pose (center, not offset) in world coordinates, if available."""
        if self.base.camera is None:
            return None
        pos_v = self.base.camera.getPos(self.base.render)
        hpr_v = self.base.camera.getHpr(self.base.render)
        offset_x, offset_y, offset_z = AVATAR_CAMERA_OFFSET
        offset = Vec3(offset_x, offset_y, offset_z)
        rot = Quat()
        rot.setHpr(hpr_v)
        offset_world = rot.xform(offset)
        pos = (
            float(pos_v[0] - offset_world[0]),
            float(pos_v[1] - offset_world[1]),
            float(pos_v[2] - offset_world[2]),
        )
        hpr = (float(hpr_v[0]), float(hpr_v[1]), float(hpr_v[2]))
        return pos, hpr

    # ---- background texture ----
    def update_bg_frame(self, rgb: Any) -> None:
        """Upload an RGB image into the background texture."""
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        if self.bg_tex.getXSize() != w or self.bg_tex.getYSize() != h:
            self.bg_tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
        self.bg_tex.setRamImageAs(rgb.tobytes(), "RGB")

    # ---- path markers ----
    def load_path_proto(self) -> Optional[NodePath]:
        """Load prototype model used to render ghost path markers."""
        if self._path_proto_failed:
            return None
        resolved = self._resolve_asset_path(PATH_GHOST_MODEL)
        if resolved is None:
            self._path_proto_failed = True
            return None
        proto_model = self.base.loader.loadModel(str(resolved))
        if proto_model is None or proto_model.isEmpty():
            self._path_proto_failed = True
            return None
        container = NodePath("path_proto_root")
        proto_model.reparentTo(container)
        try:
            bounds = proto_model.getTightBounds()
            if bounds is not None and bounds[0] is not None and bounds[1] is not None:
                mn, mx = bounds
                cx = 0.5 * (mn.x + mx.x)
                cy = 0.5 * (mn.y + mx.y)
                cz = 0.5 * (mn.z + mx.z)
                proto_model.setPos(
                    -cx,
                    -cy,
                    -cz,
                )
        except Exception:
            pass
        return container

    def _clear_path_vis(self, keep_animation: bool = False) -> None:
        """Remove path visuals; optionally keep running animation."""
        if not keep_animation:
            self._stop_path_animation()
        self._clear_path_line()
        for np_ in self._path_markers:
            try:
                np_.removeNode()
            except Exception:
                pass
        self._path_markers.clear()

    def _clear_path_line(self) -> None:
        """Remove only the line overlay, preserving ghost/plane markers."""
        if self._path_line is None:
            return
        try:
            self._path_line.removeNode()
        except Exception:
            pass
        self._path_line = None

    def render_path_markers(self, poses: List[PoseTuple]) -> None:
        """Render path markers from a list of Panda3D (pos, hpr) tuples."""
        if len(poses) < 2 and self.path_mode in (
            "poses",
            "poses_line",
            "animated",
            "planes",
        ):
            self._clear_path_vis()
            return

        if self.path_mode == "animated":
            self._clear_path_vis(keep_animation=True)
            if not poses:
                self._stop_path_animation()
                return
            self._update_animation_path(poses)
            # Keep a thin line for context so the user sees the full intended path.
            if self.anim_line_enabled:
                self._draw_path_line(poses)
            return

        if self.path_mode == "planes":
            self._clear_path_vis()
            if not poses:
                return
            if self._plane_proto is None:
                self._plane_proto = self._make_plane_proto()
            proto_plane = self._plane_proto
            if proto_plane is None:
                return
            marker_poses = self._resample_poses_by_distance(poses)
            last_idx = len(marker_poses) - 1
            skip = max(0, int(PATH_GHOST_SKIP_START))
            for idx, (pos, hpr) in enumerate(marker_poses):
                if idx < skip and idx != last_idx:
                    continue
                plane = proto_plane.copyTo(self.base.render)
                plane.setPos(self.base.render, pos[0], pos[1], pos[2])
                plane.setHpr(self.base.render, hpr[0], hpr[1], hpr[2])
                t = (idx / float(last_idx)) if last_idx > 0 else 0.0
                fade = max(0.08, 0.65 * (1.0 - 0.7 * t))
                plane.setColorScale(1.0, 1.0, 1.0, fade)
                self._path_markers.append(plane)
            return

        self._clear_path_vis()
        if not poses:
            return

        if self._path_proto is None:
            self._path_proto = self.load_path_proto()
        proto = self._path_proto
        if proto is None:
            return

        marker_poses = self._resample_poses_by_distance(poses)
        last_idx = len(marker_poses) - 1
        skip = max(0, int(PATH_GHOST_SKIP_START))
        for idx, (pos, hpr) in enumerate(marker_poses):
            if idx < skip and idx != last_idx:
                continue
            ghost = proto.copyTo(self.base.render)
            ghost.setPos(self.base.render, pos[0], pos[1], pos[2])
            ghost.setHpr(self.base.render, hpr[0], hpr[1], hpr[2])
            ghost.setBin("fixed", 5)
            ghost.setDepthWrite(False)
            # Keep ghosts neutral, but improve readability with progression-based fade/scale.
            t = (idx / float(last_idx)) if last_idx > 0 else 1.0
            alpha = max(0.22, 0.55 - 0.28 * t)
            scale = 0.90 + 0.28 * t
            if idx == last_idx:
                alpha = 0.95
                scale = max(scale, 1.22)
            ghost.setTransparency(TransparencyAttrib.MAlpha)
            ghost.setColorScale(1.0, 1.0, 1.0, alpha)
            ghost.setScale(scale)
            self._path_markers.append(ghost)

        if self.path_mode == "poses_line":
            self._draw_path_line(poses)

    def sync_avatar_to_robot(self, robot_pose: PoseTuple) -> PoseTuple:
        """Align the avatar with the provided robot pose."""
        pos, hpr = robot_pose
        return self.set_avatar_pose(pos, hpr)

    def clear_path_markers(self) -> None:
        """Legacy entrypoint to clear all path visuals."""
        self._clear_path_vis()

    def refresh_path_line(self, poses: List[PoseTuple]) -> None:
        """Rebuild only the path line for quality/style updates, preserving markers."""
        if self.path_mode == "animated":
            if not self.anim_line_enabled:
                self._clear_path_line()
                return
        elif self.path_mode != "poses_line":
            return
        self._clear_path_line()
        if len(poses) >= 2:
            self._draw_path_line(poses)

    # ---- path viz configuration ----
    def set_path_mode(self, mode: str) -> None:
        """Update path visualization mode."""
        if mode not in ("poses", "poses_line", "planes", "animated"):
            return
        self.path_mode = mode

    def set_pose_stride(self, stride: int) -> None:
        """Render pose ghosts at every Nth pose (stride)."""
        self.pose_stride = max(1, int(stride))

    def set_line_stride(self, stride: int) -> None:
        """Render pose+line ghosts at every Nth pose (stride)."""
        self.line_stride = max(1, int(stride))

    def set_marker_spacing(self, spacing_m: float) -> None:
        """Set distance-based spacing for path marker projections."""
        try:
            self.marker_spacing_m = max(0.05, float(spacing_m))
        except Exception:
            pass

    def set_anim_speed(self, speed: float) -> None:
        """Set animation speed for animated path mode."""
        self.anim_speed = max(0.01, float(speed))

    def set_anim_instances(self, count: int) -> None:
        """Set how many animated ghosts to show along the path."""
        self.anim_instances = max(1, int(count))

    def set_anim_line_enabled(self, enabled: bool) -> None:
        """Toggle path line overlay when in animated mode."""
        self.anim_line_enabled = bool(enabled)

    def publish_hold_path(
        self, cmd_pub: Any, pose: PoseTuple, ros_pose_override: Optional[dict] = None
    ) -> None:
        ros_pose = ros_pose_override or panda_pose_to_ros(pose)
        if ros_pose is None:
            return
        poses = [{"pose": ros_pose}, {"pose": ros_pose}]
        msg = {"header": {"frame_id": "map"}, "poses": poses}
        try:
            from config import TOPIC_CMD_PATH

            cmd_pub.publish(TOPIC_CMD_PATH, msg)
        except Exception:
            pass
