from __future__ import annotations
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectFrame, DirectButton
from direct.gui import DirectGuiGlobals as DGG
from panda3d.core import (
    PerspectiveLens,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    TransparencyAttrib,
)
from direct.interval.LerpInterval import LerpPosInterval
from typing import Tuple, List
from direct.showbase.ShowBase import ShowBase


class UI:
    """Tiny HUD + hotkeys."""

    def __init__(self, base: ShowBase):
        self.base = base
        self.mode: str = "Follow Mode"

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

        # Mode selector (top-right)
        self._build_mode_selector()

    def _text(self) -> str:
        return f"1/2: FOV ±2°\nMode: {self.mode}\n"

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

    # -------- Mode selector --------
    def _build_mode_selector(self) -> None:
        """Create a clean, vector-based pill segmented control.

        It draws GPU geometry (no textures) for crisp rounded corners:
        - a dark background pill
        - a selector pill that slides left/right with a rounded outer edge
        Transparent buttons sit on top to handle clicks and display text.
        """
        # Root anchored to top-right; place with some margins
        self._mode_root = DirectFrame(
            parent=self.base.a2dTopRight, frameColor=(0, 0, 0, 0), frameSize=(0, 0, 0, 0), pos=(-0.05, 0, -0.09)
        )

        # Control dimensions in aspect2d units
        self._mode_W = W = 0.62
        self._mode_H = H = 0.115
        self._mode_R = R = H / 2.0

        # Vector rounded-rect geometry (no textures, no artifacts)
        # Radii are in aspect2d units: (tl, tr, br, bl)
        def rounded_rect_node(
            left: float,
            right: float,
            bottom: float,
            top: float,
            radii: Tuple[float, float, float, float],
            color: Tuple[float, float, float, float],
            arc_segs: int = 14,
        ) -> NodePath:
            import math

            tl, tr, br, bl = radii
            width = max(0.0, right - left)
            height = max(0.0, top - bottom)
            # Clamp radii to fit
            tl = min(tl, width / 2.0, height / 2.0)
            tr = min(tr, width / 2.0, height / 2.0)
            br = min(br, width / 2.0, height / 2.0)
            bl = min(bl, width / 2.0, height / 2.0)

            def arc(cx: float, cz: float, r: float, a0: float, a1: float, n: int) -> List[Tuple[float, float]]:
                if r <= 0.0:
                    return [(cx, cz)]
                pts: List[Tuple[float, float]] = []
                for i in range(n + 1):
                    t = i / float(n)
                    a = a0 + (a1 - a0) * t
                    pts.append((cx + math.cos(a) * r, cz + math.sin(a) * r))
                return pts

            outline: List[Tuple[float, float]] = []
            # TL (pi to pi/2), TR (pi/2 to 0), BR (0 to -pi/2), BL (-pi/2 to -pi)
            outline += arc(left + tl, top - tl, tl, math.pi, math.pi / 2.0, arc_segs)
            outline += arc(right - tr, top - tr, tr, math.pi / 2.0, 0.0, arc_segs)
            outline += arc(right - br, bottom + br, br, 0.0, -math.pi / 2.0, arc_segs)
            outline += arc(left + bl, bottom + bl, bl, -math.pi / 2.0, -math.pi, arc_segs)

            fmt = GeomVertexFormat.getV3c4()
            vdata = GeomVertexData("rr", fmt, Geom.UH_static)
            vw = GeomVertexWriter(vdata, "vertex")
            cw = GeomVertexWriter(vdata, "color")

            # center for triangle fan
            cx = (left + right) * 0.5
            cz = (bottom + top) * 0.5
            vw.addData3f(cx, 0.0, cz)
            cw.addData4f(*color)
            for x, z in outline:
                vw.addData3f(x, 0.0, z)
                cw.addData4f(*color)

            tris = GeomTriangles(Geom.UH_static)
            n = len(outline)
            for i in range(1, n):
                tris.addVertices(0, i, i + 1)
            tris.addVertices(0, n, 1)

            geom = Geom(vdata)
            geom.addPrimitive(tris)
            node = GeomNode("rounded")
            node.addGeom(geom)
            np = NodePath(node)
            np.setTransparency(TransparencyAttrib.M_alpha)
            return np

        # Simple axis-aligned rectangle (no rounding)
        def rect_node(left: float, right: float, bottom: float, top: float, color: Tuple[float, float, float, float]) -> NodePath:
            fmt = GeomVertexFormat.getV3c4()
            vdata = GeomVertexData("rect", fmt, Geom.UH_static)
            vw = GeomVertexWriter(vdata, "vertex")
            cw = GeomVertexWriter(vdata, "color")
            pts = [
                (left, 0.0, bottom),
                (right, 0.0, bottom),
                (right, 0.0, top),
                (left, 0.0, top),
            ]
            for x, y, z in pts:
                vw.addData3f(x, y, z)
                cw.addData4f(*color)
            tris = GeomTriangles(Geom.UH_static)
            tris.addVertices(0, 1, 2)
            tris.addVertices(0, 2, 3)
            geom = Geom(vdata)
            geom.addPrimitive(tris)
            node = GeomNode("rect")
            node.addGeom(geom)
            np = NodePath(node)
            np.setTransparency(TransparencyAttrib.M_alpha)
            return np

        # (No texture helpers — we render vector geometry to avoid artifacts.)

        # Background pill (vector)
        self._bg_np = rounded_rect_node(
            -W, 0.0, -H / 2.0, H / 2.0, (R, R, R, R), (0.12, 0.13, 0.15, 0.92)
        )
        self._bg_np.reparentTo(self._mode_root)
        self._bg_np.setBin("fixed", 0)

        # Center divider to accent the two segments
        seam_x = -W + (W / 2.0)
        divider_thick = 0.004
        self._divider = rect_node(
            seam_x - divider_thick,
            seam_x + divider_thick,
            -H / 2.0 + 0.018,
            H / 2.0 - 0.018,
            (1.0, 1.0, 1.0, 0.06),
        )
        self._divider.reparentTo(self._mode_root)
        self._divider.setBin("fixed", 1)

        # Build selector (vector). Two child geometries to swap ends.
        segW = W / 2.0
        self._sel = self._mode_root.attachNewNode("selector")
        self._sel.setBin("fixed", 2)
        sel_color = (0.33, 0.34, 0.38, 0.96)
        self._sel_left_geom = rounded_rect_node(-segW, 0.0, -H / 2.0, H / 2.0, (R, 0.0, 0.0, R), sel_color)
        self._sel_right_geom = rounded_rect_node(-segW, 0.0, -H / 2.0, H / 2.0, (0.0, R, R, 0.0), sel_color)
        self._sel_left_geom.reparentTo(self._sel)
        self._sel_right_geom.reparentTo(self._sel)
        self._sel_right_geom.hide()
        # Position selector to left segment by default
        self._sel.setX(-W + segW)

        # Keep an optional running animation
        self._sel_anim = None

        # Transparent buttons with centered text overlay
        def make_btn(label: str, center_x: float, cb) -> DirectButton:
            btn = DirectButton(
                parent=self._mode_root,
                text=label,
                text_scale=0.048,
                text_shadow=(0, 0, 0, 0.75),
                frameColor=(0, 0, 0, 0),
                relief=None,
                command=cb,
                pressEffect=False,
            )
            btn.setPos(center_x, 0, 0)
            # Expand click area to the segment size
            segW = W / 2.0
            btn["frameSize"] = (-segW / 2.0, segW / 2.0, -H / 2.0, H / 2.0)
            # Hover feedback (slight brighten on unselected)
            def on_hover(_evt, b=btn):
                if (b is self._btn_follow and self.mode != "Follow Mode") or (
                    b is self._btn_goal and self.mode != "Goal Mode"
                ):
                    b["text_fg"] = (0.95, 0.95, 0.95, 1)

            def on_blur(_evt, b=btn):
                if (b is self._btn_follow and self.mode != "Follow Mode") or (
                    b is self._btn_goal and self.mode != "Goal Mode"
                ):
                    b["text_fg"] = (0.82, 0.82, 0.85, 1)

            btn.bind(DGG.WITHIN, on_hover)
            btn.bind(DGG.WITHOUT, on_blur)
            return btn

        left_center = -W + segW / 2.0
        right_center = -W + segW + segW / 2.0
        self._btn_follow = make_btn("Follow Mode", left_center, lambda: self._set_mode("Follow Mode"))
        self._btn_goal = make_btn("Goal Mode", right_center, lambda: self._set_mode("Goal Mode"))

        # Initialize selection visuals
        self._set_mode(self.mode)

    def _set_mode(self, mode: str) -> None:
        """Set the current mode, update visuals, and print it."""
        self.mode = mode
        # Move selector card with a short animation and adjust visuals
        W = self._mode_W
        segW = W / 2.0

        # Stop any running animation
        try:
            if self._sel_anim is not None:
                self._sel_anim.finish()
        except Exception:
            pass

        if mode == "Follow Mode":
            target_x = -W + segW
            # Selector rounded on the left edge
            self._sel_left_geom.show()
            self._sel_right_geom.hide()
            self._btn_follow["text_fg"] = (1, 1, 1, 1)
            self._btn_goal["text_fg"] = (0.82, 0.82, 0.85, 1)
        else:
            target_x = -W + 2 * segW
            # Selector rounded on the right edge
            self._sel_left_geom.hide()
            self._sel_right_geom.show()
            self._btn_follow["text_fg"] = (0.82, 0.82, 0.85, 1)
            self._btn_goal["text_fg"] = (1, 1, 1, 1)

        # Smooth slide
        self._sel_anim = LerpPosInterval(self._sel, 0.12, (target_x, 0, 0), blendType="easeInOut")
        self._sel_anim.start()
        print(f"[UI] Selected: {self.mode}")
        self.update("")
