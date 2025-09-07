# intrinsics.py
from __future__ import annotations
from math import atan, pi
from panda3d.core import PerspectiveLens


def apply_opencv_intrinsics_to_lens(
    lens: PerspectiveLens,
    width_px: int,
    height_px: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    # Convert fx/fy to FOV in degrees
    fov_x = 2.0 * atan(width_px / (2.0 * fx)) * (180.0 / pi)
    fov_y = 2.0 * atan(height_px / (2.0 * fy)) * (180.0 / pi)
    lens.setFov(fov_x, fov_y)
    lens.setFilmSize(width_px, height_px)
    # principal point offset; invert Y to match Panda coordinates
    lens.setFilmOffset(cx - width_px / 2.0, -(cy - height_px / 2.0))
