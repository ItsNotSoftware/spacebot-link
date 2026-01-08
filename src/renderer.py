"""Rendering utilities: lights, camera lens, background, and avatar setup."""

from __future__ import annotations

from math import pi, sin
from pathlib import Path
from typing import Optional, Tuple, List, Any

from panda3d.core import (
    AmbientLight,
    CardMaker,
    DirectionalLight,
    LineSegs,
    NodePath,
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
    PATH_LINE_STRIDE,
    PATH_LINE_THICKNESS,
    PATH_MODE_DEFAULT,
    PATH_POSE_STRIDE,
    PATH_GHOST_SKIP_START,
    PATH_PLANE_SIZE,
    PATH_PLANE_OUTLINE_COLOR,
    PATH_PLANE_FILL_ALPHA,
    PATH_PLANE_THICKNESS,
    AVATAR_CAMERA_OFFSET,
    FLOOR_SHADOW_SIZE,
    FLOOR_SHADOW_COLOR,
    FLOOR_LINE_COLOR,
    FLOOR_LINE_THICKNESS,
)
from utils import apply_opencv_intrinsics_to_lens, panda_pose_to_ros

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
        self.path_mode: str = PATH_MODE_DEFAULT  # poses | poses_line | planes | animated
        self.pose_stride: int = PATH_POSE_STRIDE
        self.line_stride: int = PATH_LINE_STRIDE
        self.anim_speed: float = PATH_ANIM_SPEED  # units per second

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

        # reasonable default intrinsics (updated once we see cam_info)
        self._init_default_lens()

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
        w, h = 1280, 720
        fx = fy = 900.0
        cx, cy = w / 2, h / 2
        apply_opencv_intrinsics_to_lens(self.base.camLens, w, h, fx, fy, cx, cy)
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

    def _init_floor_indicator(self) -> None:
        """Initialize the floor height shadow and line."""
        self._floor_shadow = self._make_floor_shadow()
        if self._floor_shadow is not None:
            self._floor_shadow.hide()

    def _make_floor_shadow(self) -> Optional[NodePath]:
        """Create a translucent quad used as a floor shadow marker."""
        size = max(0.05, float(FLOOR_SHADOW_SIZE))
        cm = CardMaker("floor_shadow")
        cm.setFrame(-0.5 * size, 0.5 * size, -0.5 * size, 0.5 * size)
        node = cm.generate()
        if node is None:
            return None
        np_shadow = self.base.render.attachNewNode(node)
        np_shadow.setP(-90.0)  # rotate into XZ plane so normal points +Y
        np_shadow.setTransparency(TransparencyAttrib.MAlpha)
        np_shadow.setColor(*FLOOR_SHADOW_COLOR)
        np_shadow.setBin("fixed", 8)
        np_shadow.setDepthWrite(False)
        np_shadow.setDepthTest(False)
        return np_shadow

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
        color: Tuple[float, float, float, float] = PATH_LINE_COLOR,
    ) -> None:
        """Draw a line strip through all poses to preserve path continuity."""
        if len(poses) < 2:
            return
        segs = LineSegs("path_line")
        segs.setThickness(PATH_LINE_THICKNESS)
        segs.setColor(*color)
        first = True
        for pos, _ in poses:
            x, y, z = pos
            if first:
                segs.moveTo(x, y, z)
                first = False
            else:
                segs.drawTo(x, y, z)
        node = segs.create()
        if node is None:
            return
        np_line = self.base.render.attachNewNode(node)
        np_line.setBin("fixed", 4)
        np_line.setDepthWrite(False)
        np_line.setDepthTest(False)
        self._path_line = np_line

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
        self._floor_shadow.lookAt(
            self._floor_shadow.getPos() + Vec3(ax, ay, az)
        )
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
        cx = pos[0] + offset_x
        cy = pos[1] + offset_y
        cz = pos[2] + offset_z
        self.base.camera.setPos(self.base.render, cx, cy, cz)
        self.base.camera.setHpr(self.base.render, hpr[0], hpr[1], hpr[2])

    def get_camera_pose(self) -> Optional[PoseTuple]:
        """Return the camera pose (center, not offset) in world coordinates, if available."""
        if self.base.camera is None:
            return None
        pos_v = self.base.camera.getPos(self.base.render)
        hpr_v = self.base.camera.getHpr(self.base.render)
        offset_x, offset_y, offset_z = AVATAR_CAMERA_OFFSET
        pos = (
            float(pos_v[0] - offset_x),
            float(pos_v[1] - offset_y),
            float(pos_v[2] - offset_z),
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
        if self._path_line is not None:
            try:
                self._path_line.removeNode()
            except Exception:
                pass
            self._path_line = None
        for np_ in self._path_markers:
            try:
                np_.removeNode()
            except Exception:
                pass
        self._path_markers.clear()

    def render_path_markers(self, poses: List[PoseTuple]) -> None:
        """Render path markers from a list of Panda3D (pos, hpr) tuples."""
        if len(poses) < 3 and self.path_mode in ("poses", "poses_line", "animated", "planes"):
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
                self._draw_path_line(poses, color=PATH_LINE_COLOR)
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
            stride = max(1, int(self.pose_stride))
            last_idx = len(poses) - 1
            skip = max(0, int(PATH_GHOST_SKIP_START))
            for idx, (pos, hpr) in enumerate(poses):
                if idx not in (0, last_idx) and (idx + stride - 1) % stride != 0:
                    continue
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

        stride = self.pose_stride if self.path_mode == "poses" else self.line_stride
        stride = max(1, int(stride))
        last_idx = len(poses) - 1
        skip = max(0, int(PATH_GHOST_SKIP_START))
        for idx, (pos, hpr) in enumerate(poses):
            # Always render first and last pose, otherwise stride-filter.
            if idx not in (0, last_idx) and (idx + stride - 1) % stride != 0:
                continue
            if idx < skip and idx != last_idx:
                continue
            ghost = proto.copyTo(self.base.render)
            ghost.setPos(self.base.render, pos[0], pos[1], pos[2])
            ghost.setHpr(self.base.render, hpr[0], hpr[1], hpr[2])
            ghost.setBin("fixed", 5)
            ghost.setDepthWrite(False)
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
