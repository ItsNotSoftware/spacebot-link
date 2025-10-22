"""Utility helpers shared across the SpaceBotLink viewer.

The module currently folds together intrinsics handling (mapping OpenCV
camera calibration parameters onto Panda3D lenses) and a small ROS-to-Panda3D
orientation converter used for IMU data. Keeping them in one place avoids the
previous duplication between ``intrinsics.py`` and ``imu_utils.py`` and makes
it clear where future bridge helpers should live.
"""

from __future__ import annotations

from math import atan, pi, sqrt
from typing import Any, Dict, Optional, Tuple

from panda3d.core import PerspectiveLens, Quat

__all__ = [
    "apply_opencv_intrinsics_to_lens",
    "ros_orientation_to_panda_hpr",
]


def apply_opencv_intrinsics_to_lens(
    lens: PerspectiveLens,
    width_px: int,
    height_px: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    """Project OpenCV intrinsics onto a Panda3D ``PerspectiveLens``.

    ``fx``/``fy`` are focal lengths in pixel units, ``cx``/``cy`` are the
    principal point offsets, and ``width_px``/``height_px`` describe the image
    resolution those values were calibrated for. The helper computes the
    equivalent horizontal/vertical field of view and film offset so the viewer
    camera mirrors the ROS camera model.
    """

    fov_x = 2.0 * atan(width_px / (2.0 * fx)) * (180.0 / pi)
    fov_y = 2.0 * atan(height_px / (2.0 * fy)) * (180.0 / pi)
    lens.setFov(fov_x, fov_y)
    lens.setFilmSize(width_px, height_px)
    lens.setFilmOffset(cx - width_px / 2.0, -(cy - height_px / 2.0))


_SQRT_HALF = sqrt(0.5)
_ROS_TO_PANDA_ROT = (_SQRT_HALF, 0.0, 0.0, _SQRT_HALF)


def _quat_multiply(
    lhs: Tuple[float, float, float, float],
    rhs: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def ros_orientation_to_panda_hpr(
    orientation: Dict[str, Any]
) -> Optional[Tuple[float, float, float]]:
    """Convert a ROS ``geometry_msgs/Quaternion`` into Panda3D ``(H, P, R)``.

    ROS uses an ENU frame (X forward, Y left, Z up) whereas Panda3D extracts
    heading/pitch/roll assuming a different camera basis. Multiplying the ROS
    quaternion by a fixed 90° rotation around Z bridges the two conventions so
    ``Quat.getHpr`` returns intuitive values for the viewer avatar.
    """

    try:
        qx = float(orientation.get("x"))
        qy = float(orientation.get("y"))
        qz = float(orientation.get("z"))
        qw = float(orientation.get("w"))
    except (TypeError, ValueError):
        return None

    q_ros = (qw, qx, qy, qz)
    q_panda = _quat_multiply(
        _quat_multiply(_ROS_TO_PANDA_ROT, q_ros), _quat_conjugate(_ROS_TO_PANDA_ROT)
    )

    quat = Quat()
    quat.set(*q_panda)
    h, p, r = quat.getHpr()
    return float(h), float(p), float(r)
