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
    Vec4,
)
from direct.showbase.ShowBase import ShowBase

from avatar import Avatar
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
        self._bg_aspect: float = 0.0
        self._path_line: Optional[NodePath] = None
        self._anim_np: Optional[NodePath] = None
        self._anim_task_name = "PathGhostAnim"
        self._anim_path: List[PoseTuple] = []
        self._anim_idx: int = 0
        self._anim_t: float = 0.0
        self.path_mode: str = "poses"  # poses | poses_line | animated
        self.pose_stride: int = 4
        self.line_stride: int = 8
        self.anim_speed: float = 1.0  # units per second

        self._init_lights()
        self._make_bg_card(initial_aspect=9 / 16)

        model_path = self._resolve_asset_path(gltf_model)
        if model_path is None:
            raise FileNotFoundError(f"Could not resolve GLTF model: {gltf_model}")
        self.avatar = Avatar(self.base.render, self.base.loader, str(model_path))

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
        self, poses: List[PoseTuple], color: Tuple[float, float, float, float] = (0.95, 0.80, 0.20, 0.9)
    ) -> None:
        """Draw a line strip through all poses to preserve path continuity."""
        if len(poses) < 2:
            return
        segs = LineSegs("path_line")
        segs.setThickness(4.0)
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

    # ---- animated ghost ----
    def _start_path_animation(self, poses: List[PoseTuple]) -> None:
        """Animate a single ghost along the provided path."""
        self._stop_path_animation()
        if len(poses) < 2:
            return
        if self._path_proto is None:
            self._path_proto = self.load_path_proto()
        proto = self._path_proto
        if proto is None:
            return
        self._anim_np = proto.copyTo(self.base.render)
        self._anim_np.setBin("fixed", 5)
        self._anim_np.setDepthWrite(False)
        self._anim_path = poses
        self._anim_idx = 0
        self._anim_t = 0.0
        start_pos, start_hpr = poses[0]
        self._anim_np.setPos(self.base.render, *start_pos)
        self._anim_np.setHpr(self.base.render, *start_hpr)
        self.base.taskMgr.add(self._anim_task, self._anim_task_name)

    def _stop_path_animation(self) -> None:
        """Stop path animation and remove animated ghost."""
        try:
            self.base.taskMgr.remove(self._anim_task_name)
        except Exception:
            pass
        if self._anim_np is not None:
            try:
                self._anim_np.removeNode()
            except Exception:
                pass
        self._anim_np = None
        self._anim_path = []
        self._anim_idx = 0
        self._anim_t = 0.0

    def _anim_task(self, task) -> int:
        """Advance animated ghost along the path with constant linear speed."""
        if self._anim_np is None or len(self._anim_path) < 2:
            return task.done
        dt = self.base.taskMgr.globalClock.getDt()
        speed = max(0.01, self.anim_speed)
        # current segment endpoints
        curr = self._anim_path[self._anim_idx]
        nxt = self._anim_path[self._anim_idx + 1]
        (x1, y1, z1), hpr1 = curr
        (x2, y2, z2), hpr2 = nxt
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        seg_len = max(1e-6, (dx * dx + dy * dy + dz * dz) ** 0.5)
        # advance along segment
        self._anim_t += (speed * dt) / seg_len
        if self._anim_t >= 1.0:
            self._anim_idx += 1
            if self._anim_idx >= len(self._anim_path) - 1:
                # Loop back to start for continuous animation.
                self._anim_idx = 0
                self._anim_t = 0.0
                return task.cont
            self._anim_t = 0.0
            return task.cont
        t = self._anim_t
        # linear interp position and simple HPR lerp
        pos = (x1 + dx * t, y1 + dy * t, z1 + dz * t)
        h = hpr1[0] + (hpr2[0] - hpr1[0]) * t
        p = hpr1[1] + (hpr2[1] - hpr1[1]) * t
        r = hpr1[2] + (hpr2[2] - hpr1[2]) * t
        self._anim_np.setPos(self.base.render, *pos)
        self._anim_np.setHpr(self.base.render, h, p, r)
        return task.cont
    # ---- avatar helpers ----
    def get_avatar_pose(self) -> PoseTuple:
        """Return avatar pose (pos, hpr) in world coordinates."""
        return self.avatar.get_pose()

    def set_avatar_pose(self, pos: Tuple[float, float, float], hpr: Tuple[float, float, float]) -> PoseTuple:
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

    # ---- camera helpers ----
    def set_camera_pose(self, pos: Tuple[float, float, float], hpr: Tuple[float, float, float]) -> None:
        """Place the camera at the given world pose."""
        if self.base.camera is None:
            return
        self.base.camera.setPos(self.base.render, pos[0], pos[1], pos[2])
        self.base.camera.setHpr(self.base.render, hpr[0], hpr[1], hpr[2])

    def get_camera_pose(self) -> Optional[PoseTuple]:
        """Return the camera pose in world coordinates, if available."""
        if self.base.camera is None:
            return None
        pos_v = self.base.camera.getPos(self.base.render)
        hpr_v = self.base.camera.getHpr(self.base.render)
        pos = (float(pos_v[0]), float(pos_v[1]), float(pos_v[2]))
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
        resolved = self._resolve_asset_path("../assets/path_ghost.glb")
        if resolved is None:
            self._path_proto_failed = True
            return None
        proto = self.base.loader.loadModel(str(resolved))
        if proto is None or proto.isEmpty():
            self._path_proto_failed = True
            return None
        return proto

    def clear_path_markers(self) -> None:
        """Remove any existing path markers from the scene graph."""
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
        self.clear_path_markers()
        if not poses:
            return
        if self.path_mode == "animated":
            self._start_path_animation(poses)
            return

        if self._path_proto is None:
            self._path_proto = self.load_path_proto()
        proto = self._path_proto
        if proto is None:
            return

        stride = self.pose_stride if self.path_mode == "poses" else self.line_stride
        stride = max(1, int(stride))
        last_idx = len(poses) - 1
        for idx, (pos, hpr) in enumerate(poses):
            # Always render first and last pose, otherwise stride-filter.
            if idx not in (0, last_idx) and idx % stride != 0:
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

    # ---- path viz configuration ----
    def set_path_mode(self, mode: str) -> None:
        """Update path visualization mode."""
        if mode not in ("poses", "poses_line", "animated"):
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
