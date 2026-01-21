from __future__ import annotations
from typing import Tuple, cast
from panda3d.core import (
    TransparencyAttrib,
    CullFaceAttrib,
    DepthOffsetAttrib,
    Point3,
    Quat,
)
from direct.showbase.Loader import Loader
from panda3d.core import NodePath

from config import AVATAR_CAMERA_OFFSET


class Avatar:
    """Two-pass transparent avatar rendering."""

    def __init__(
        self,
        parent: NodePath,
        loader: Loader,
        gltf_path: str,
        scale: float = 1.0,
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        hpr: Tuple[float, float, float] = (0, 0, 0),
    ):
        """Load avatar model and set up dual-pass transparent rendering."""
        self._parent: NodePath = parent
        base_np: NodePath = cast(NodePath, loader.loadModel(gltf_path))
        if base_np.isEmpty():
            raise RuntimeError(
                f"Failed to load model (empty NodePath): {gltf_path} "
                "Make sure 'panda3d-gltf' is installed and the path is correct."
            )

        base_np.setScale(scale)
        self._init_hpr: Tuple[float, float, float] = (
            float(hpr[0]),
            float(hpr[1]),
            float(hpr[2]),
        )

        self._back = parent.attachNewNode("avatar_back")
        self._front = parent.attachNewNode("avatar_front")

        offset_x, offset_y, offset_z = AVATAR_CAMERA_OFFSET
        model_back = base_np.copyTo(self._back)
        model_back.setPos(Point3(0.0, 0.0, 0.0))
        model_front = base_np.copyTo(self._front)
        model_front.setPos(Point3(0.0, 0.0, 0.0))

        self.set_pos(*pos)
        self.set_hpr(*hpr)

        for np_ in (self._back, self._front):
            np_.setTransparency(TransparencyAttrib.MDual)
            np_.setDepthWrite(False)
            np_.setAttrib(DepthOffsetAttrib.make(1))

        self._back.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
        self._back.setBin("fixed", 10)
        self._front.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullClockwise))
        self._front.setBin("fixed", 11)

    def set_pos(self, x: float, y: float, z: float) -> None:
        """Place avatar at given world position."""
        # Explicitly set in parent/world space
        self._back.setPos(self._parent, x, y, z)
        self._front.setPos(self._parent, x, y, z)

    def set_hpr(self, h: float, p: float, r: float) -> None:
        """Set avatar heading/pitch/roll."""
        self._back.setHpr(h, p, r)
        self._front.setHpr(h, p, r)

    def get_hpr(self) -> Tuple[float, float, float]:
        """Return current avatar heading/pitch/roll."""
        h, p, r = self._front.getHpr()
        return float(h), float(p), float(r)

    def get_pos(self) -> Tuple[float, float, float]:
        """Return avatar position in world/parent coordinates."""
        x, y, z = self._front.getPos(self._parent)
        return float(x), float(y), float(z)

    def get_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return avatar (pos, hpr) in world/parent coordinates."""
        px, py, pz = self._front.getPos(self._parent)
        h, p, r = self._front.getHpr(self._parent)
        return (float(px), float(py), float(pz)), (float(h), float(p), float(r))

    def reset_hpr(self) -> None:
        """Restore avatar orientation to initial HPR."""
        self.set_hpr(*self._init_hpr)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the avatar geometry."""
        if visible:
            self._back.show()
            self._front.show()
        else:
            self._back.hide()
            self._front.hide()

    def set_opacity(self, alpha: float) -> None:
        """Set uniform alpha for the avatar's dual-pass geometry."""
        alpha = max(0.0, min(1.0, float(alpha)))
        self._back.setColorScale(1.0, 1.0, 1.0, alpha)
        self._front.setColorScale(1.0, 1.0, 1.0, alpha)

    def set_color(self, r: float, g: float, b: float, a: float) -> None:
        """Set uniform color scale for the avatar's dual-pass geometry."""
        r = max(0.0, min(1.0, float(r)))
        g = max(0.0, min(1.0, float(g)))
        b = max(0.0, min(1.0, float(b)))
        a = max(0.0, min(1.0, float(a)))
        self._back.setColorScale(r, g, b, a)
        self._front.setColorScale(r, g, b, a)

    def set_scale(self, s: float) -> None:
        """Uniformly scale avatar geometry."""
        self._back.setScale(s)
        self._front.setScale(s)

    def move_world(self, dx: float, dy: float, dz: float) -> None:
        """Translate avatar in world space by provided deltas."""
        # Apply deltas in parent/world coordinates, independent of node's local HPR
        bx, by, bz = self._back.getPos(self._parent)
        fx, fy, fz = self._front.getPos(self._parent)
        self._back.setPos(self._parent, bx + dx, by + dy, bz + dz)
        self._front.setPos(self._parent, fx + dx, fy + dy, fz + dz)

    def add_hpr(self, dh: float, dp: float, dr: float) -> None:
        """Increment avatar orientation in its local/body frame."""
        curr_q: Quat = self._front.getQuat()
        dq = Quat()
        dq.setHpr((dh, dp, dr))
        # Pre-multiply so incremental rotations happen in the avatar's local/body frame
        new_q = dq * curr_q
        self._back.setQuat(new_q)
        self._front.setQuat(new_q)
