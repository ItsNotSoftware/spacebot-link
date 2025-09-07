# avatar.py
from __future__ import annotations
from typing import Tuple
from panda3d.core import (
    TransparencyAttrib,
    CullFaceAttrib,
    DepthOffsetAttrib,
    Point3,
)
from direct.showbase.Loader import Loader
from panda3d.core import NodePath


class Avatar:
    """
    Two-pass transparent rendering:
      - backfaces (bin 10)
      - frontfaces (bin 11)
    Simple transforms only.
    """

    def __init__(
        self,
        parent: NodePath,
        loader: Loader,
        gltf_path: str,
        scale: float = 20.0,
        pos: Tuple[float, float, float] = (8, 0, 6),
        hpr: Tuple[float, float, float] = (0, 45, 0),
    ):
        base = loader.load_model(gltf_path)
        if base is None:
            raise RuntimeError(f"Failed to load: {gltf_path}")

        base.setScale(scale)  # type: ignore
        base.setPos(Point3(*pos))  # type: ignore
        base.setHpr(*hpr)  # type: ignore

        self._back = base.copy_to(parent)  # type: ignore
        self._front = base.copy_to(parent)  # type: ignore
        base.hide()  # type: ignore

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
        self._back.setPos(x, y, z)
        self._front.setPos(x, y, z)

    def set_hpr(self, h: float, p: float, r: float) -> None:
        self._back.setHpr(h, p, r)
        self._front.setHpr(h, p, r)

    def set_scale(self, s: float) -> None:
        self._back.setScale(s)
        self._front.setScale(s)
