from __future__ import annotations
from typing import Callable, Optional
from panda3d.core import PerspectiveLens
from direct.showbase.ShowBase import ShowBase


class UI:
    """Lightweight UI state holder (mode, move target, abort callback)."""

    def __init__(self, base: ShowBase, on_abort: Optional[Callable[[], None]] = None):
        self.base = base
        self.mode: str = "Goal Mode"
        self.move_target: str = "Avatar"  # "Avatar" or "Robot"
        self._last_status: str = ""
        self._on_abort = on_abort

        base.accept("1", lambda: self._bump_fov(-2))
        base.accept("2", lambda: self._bump_fov(+2))

    def update(self, extra: str = "") -> None:
        self._last_status = extra

    def _bump_fov(self, delta: float) -> None:
        lens: PerspectiveLens = self.base.camLens
        fx, fy = lens.getFov()
        lens.setFov(max(10.0, fx + delta), max(10.0, fy + delta))
        update = getattr(self.base, "_update_bg_scale", None)
        if callable(update):
            update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update("")

    def set_move_target(self, target: str) -> None:
        """Update the move target label in the HUD ("Avatar" or "Robot")."""
        self.move_target = target
        self.update("")

    def trigger_abort(self) -> None:
        if callable(self._on_abort):
            try:
                self._on_abort()
            except Exception:
                pass
