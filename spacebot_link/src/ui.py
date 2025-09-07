from __future__ import annotations
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import PerspectiveLens
from direct.showbase.ShowBase import ShowBase


class UI:
    """Tiny HUD + hotkeys."""

    def __init__(self, base: ShowBase):
        self.base = base
        self._orbit = True

        self._hud = OnscreenText(
            text=self._text(),
            pos=(0.012, 0.95),
            align=0,
            scale=0.045,
            fg=(0.95, 0.95, 0.95, 0.9),
            mayChange=True,
            parent=base.a2dTopLeft,
        )

        base.accept("o", self._toggle_orbit)
        base.accept("f", self._toggle_wire)
        base.accept("1", lambda: self._bump_fov(-2))
        base.accept("2", lambda: self._bump_fov(+2))

    def _text(self) -> str:
        return "O: orbit  F: wireframe  1/2: FOV ±2°\n"

    def update(self, extra: str = "") -> None:
        self._hud.setText(self._text() + extra)

    def _toggle_orbit(self) -> None:
        self._orbit = not self._orbit

    def _toggle_wire(self) -> None:
        if self.base.render.hasRenderMode():
            self.base.render.clearRenderMode()
        else:
            self.base.render.setRenderModeWireframe()

    def _bump_fov(self, delta: float) -> None:
        lens: PerspectiveLens = self.base.camLens
        fx, fy = lens.getFov()
        lens.setFov(max(10.0, fx + delta), max(10.0, fy + delta))

    @property
    def orbit_enabled(self) -> bool:
        return self._orbit
