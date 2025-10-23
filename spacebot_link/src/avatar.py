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


class Avatar:
    """Two-pass transparent avatar rendering."""

    def __init__(
        self,
        parent: NodePath,
        loader: Loader,
        gltf_path: str,
        scale: float = 1.0,
        pos: Tuple[float, float, float] = (0, 1, 0),
        hpr: Tuple[float, float, float] = (0, 90, 270),
    ):
        self._parent: NodePath = parent
        base_np: NodePath = cast(NodePath, loader.loadModel(gltf_path))
        if base_np.isEmpty():
            raise RuntimeError(
                f"Failed to load model (empty NodePath): {gltf_path} "
                "Make sure 'panda3d-gltf' is installed and the path is correct."
            )

        base_np.setScale(scale)
        base_np.setPos(Point3(*pos))
        base_np.setHpr(*hpr)
        self._init_hpr: Tuple[float, float, float] = (
            float(hpr[0]),
            float(hpr[1]),
            float(hpr[2]),
        )

        self._back = base_np.copyTo(parent)
        self._front = base_np.copyTo(parent)
        base_np.hide()

        for np_ in (self._back, self._front):
            np_.setTransparency(TransparencyAttrib.MDual)
            np_.setDepthWrite(False)
            np_.setAttrib(DepthOffsetAttrib.make(1))

        self._back.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
        self._back.setBin("fixed", 10)
        self._front.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullClockwise))
        self._front.setBin("fixed", 11)

    def set_pos(self, x: float, y: float, z: float) -> None:
        # Explicitly set in parent/world space
        self._back.setPos(self._parent, x, y, z)
        self._front.setPos(self._parent, x, y, z)

    def set_hpr(self, h: float, p: float, r: float) -> None:
        self._back.setHpr(h, p, r)
        self._front.setHpr(h, p, r)

    def get_hpr(self) -> Tuple[float, float, float]:
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
        self.set_hpr(*self._init_hpr)

    def set_scale(self, s: float) -> None:
        self._back.setScale(s)
        self._front.setScale(s)

    def move_world(self, dx: float, dy: float, dz: float) -> None:
        # Apply deltas in parent/world coordinates, independent of node's local HPR
        bx, by, bz = self._back.getPos(self._parent)
        fx, fy, fz = self._front.getPos(self._parent)
        self._back.setPos(self._parent, bx + dx, by + dy, bz + dz)
        self._front.setPos(self._parent, fx + dx, fy + dy, fz + dz)

    def add_hpr(self, dh: float, dp: float, dr: float) -> None:
        curr_q: Quat = self._front.getQuat()
        dq = Quat()
        dq.setHpr((dh, dp, dr))
        new_q = curr_q * dq
        self._back.setQuat(new_q)
        self._front.setQuat(new_q)
