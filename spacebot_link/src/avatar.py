# avatar.py
from __future__ import annotations
from typing import Tuple, cast
from panda3d.core import (
    TransparencyAttrib,
    CullFaceAttrib,
    DepthOffsetAttrib,
    Point3,
    Quat,
    Vec3,
)
from direct.showbase.Loader import Loader
from panda3d.core import NodePath


class Avatar:
    """Two-pass transparent avatar rendering.

    Renders the same model twice to mitigate transparency sorting issues:
    first backfaces, then frontfaces, using fixed render bins.

    Args:
        parent: Parent node to attach the avatar instances to.
        loader: Panda3D loader used to load the GLTF model.
        gltf_path: Path to the GLTF avatar model.
        scale: Uniform scale factor applied to the avatar.
        pos: Initial position as ``(x, y, z)``.
        hpr: Initial orientation as ``(h, p, r)`` in degrees.

    Raises:
        RuntimeError: If the model cannot be loaded from ``gltf_path``.
    """

    def __init__(
        self,
        parent: NodePath,
        loader: Loader,
        gltf_path: str,
        scale: float = 1.0,
        pos: Tuple[float, float, float] = (0, 1, 0),
        hpr: Tuple[float, float, float] = (0, 90, 90),
    ):
        # Panda3D loader uses camelCase: loadModel
        base_np: NodePath = cast(NodePath, loader.loadModel(gltf_path))
        if base_np.isEmpty():
            raise RuntimeError(
                f"Failed to load model (empty NodePath): {gltf_path} Make sure 'panda3d-gltf' is installed and the path is correct."
            )

        base_np.setScale(scale)
        base_np.setPos(Point3(*pos))
        base_np.setHpr(*hpr)
        # Remember initial orientation for resets
        self._init_hpr: Tuple[float, float, float] = (float(hpr[0]), float(hpr[1]), float(hpr[2]))

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

    # simple transforms applied to both passes
    def set_pos(self, x: float, y: float, z: float) -> None:
        """Set avatar position for both passes.

        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.
        """
        self._back.setPos(x, y, z)
        self._front.setPos(x, y, z)

    def set_hpr(self, h: float, p: float, r: float) -> None:
        """Set avatar orientation for both passes.

        Args:
            h: Heading (yaw) in degrees.
            p: Pitch in degrees.
            r: Roll in degrees.
        """
        self._back.setHpr(h, p, r)
        self._front.setHpr(h, p, r)

    def get_hpr(self) -> Tuple[float, float, float]:
        """Get current orientation as ``(h, p, r)`` in degrees.

        Returns:
            A tuple of heading, pitch, roll for the avatar.
        """
        h, p, r = self._front.getHpr()
        return float(h), float(p), float(r)

    def reset_hpr(self) -> None:
        """Reset orientation to the original spawn HPR."""
        self.set_hpr(*self._init_hpr)

    def set_scale(self, s: float) -> None:
        """Set uniform scale for both passes.

        Args:
            s: Scale factor.
        """
        self._back.setScale(s)
        self._front.setScale(s)

    def move_world(self, dx: float, dy: float, dz: float) -> None:
        """Translate the avatar in world coordinates.

        Applies the same world-space translation to both render passes.

        Args:
            dx: Delta along world X.
            dy: Delta along world Y.
            dz: Delta along world Z.
        """
        self._back.setPos(
            dx + self._back.getX(), dy + self._back.getY(), dz + self._back.getZ()
        )
        self._front.setPos(
            dx + self._front.getX(), dy + self._front.getY(), dz + self._front.getZ()
        )

    def add_hpr(self, dh: float, dp: float, dr: float) -> None:
        """Incrementally rotate the avatar by the given deltas in degrees.

        Applies the rotation using quaternions around the avatar's local axes
        to avoid gimbal lock when pitch approaches ±90°.

        Args:
            dh: Heading delta (yaw) in degrees, about local +Z.
            dp: Pitch delta in degrees, about local +X.
            dr: Roll delta in degrees, about local +Y.
        """
        # Current orientation (both passes share the same transform)
        curr_q: Quat = self._front.getQuat()

        # Compose delta as Panda HPR-order quaternion to match engine mapping
        dq = Quat()
        dq.setHpr((dh, dp, dr))

        new_q = curr_q * dq  # post-multiply for local-space rotation
        self._back.setQuat(new_q)
        self._front.setQuat(new_q)
