from __future__ import annotations
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import PerspectiveLens
from direct.showbase.ShowBase import ShowBase


class UI:
    """Tiny HUD + hotkeys."""

    def __init__(self, base: ShowBase):
        self.base = base

        # In a2dTopLeft, the origin is at the top-left corner and
        # positive Y goes up (off-screen). Use a small negative Y
        # offset to move the text down into view.
        self._hud = OnscreenText(
            text=self._text(),
            pos=(0.012, -0.08),
            align=0,
            scale=0.055,
            fg=(0.95, 0.95, 0.95, 0.95),
            mayChange=True,
            parent=base.a2dTopLeft,
        )

        # UI hotkeys
        base.accept("1", lambda: self._bump_fov(-2))
        base.accept("2", lambda: self._bump_fov(+2))

    def _text(self) -> str:
        return "1/2: FOV ±2°\n"

    def update(self, extra: str = "") -> None:
        self._hud.setText(self._text() + extra)

    def _bump_fov(self, delta: float) -> None:
        lens: PerspectiveLens = self.base.camLens
        fx, fy = lens.getFov()
        lens.setFov(max(10.0, fx + delta), max(10.0, fy + delta))
        # Keep background scaled to fill after FOV changes, if available
        update = getattr(self.base, "_update_bg_scale", None)
        if callable(update):
            update()
