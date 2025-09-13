"""Camera intrinsics helpers.

Utilities to apply OpenCV-style intrinsics to a Panda3D ``PerspectiveLens``.
"""
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
    """Apply OpenCV intrinsics to a Panda3D lens.

    Converts focal lengths (``fx``, ``fy``) and principal point (``cx``, ``cy``)
    from pixel units into a field-of-view and film offset for the given lens.

    Args:
        lens: Target ``PerspectiveLens`` to mutate.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        fx: Focal length in pixels along X.
        fy: Focal length in pixels along Y.
        cx: Principal point X in pixels.
        cy: Principal point Y in pixels.
    """
    # Convert fx/fy to FOV in degrees
    fov_x = 2.0 * atan(width_px / (2.0 * fx)) * (180.0 / pi)
    fov_y = 2.0 * atan(height_px / (2.0 * fy)) * (180.0 / pi)
    lens.setFov(fov_x, fov_y)
    lens.setFilmSize(width_px, height_px)
    # principal point offset; invert Y to match Panda coordinates
    lens.setFilmOffset(cx - width_px / 2.0, -(cy - height_px / 2.0))
