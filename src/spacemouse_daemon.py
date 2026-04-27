"""SpaceMouse daemon: read pyspacemouse state and publish over ZMQ.

Publishes a gamepad-compatible schema so the current InputController can
control robot/avatar motion without further changes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import time
from typing import Any, Dict, Iterable, Optional

import zmq

try:
    import pyspacemouse
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "pyspacemouse library not available. Install with: uv add pyspacemouse"
    ) from exc

try:
    import usb.core as _usbcore
    import usb.util as _usbutil
    _PYUSB_AVAILABLE = True
except ImportError:
    _PYUSB_AVAILABLE = False

from config import (
    SPACEMOUSE_AXIS_SCALE,
    SPACEMOUSE_CROSS_DRIFT_MAX,
    SPACEMOUSE_INTENT_MIN,
    SPACEMOUSE_INVERT_PITCH,
    SPACEMOUSE_INVERT_YAW,
    SPACEMOUSE_MIXED_AXIS_CLIP_MIN,
    SPACEMOUSE_MIXED_AXIS_CLIP_RATIO,
    SPACEMOUSE_REMOTE_ENDPOINT,
    SPACEMOUSE_REMOTE_TOPIC,
    SPACEMOUSE_RESPONSE_CURVE,
    SPACEMOUSE_ROTATION_AXIS_CLIP_MIN,
    SPACEMOUSE_ROTATION_AXIS_CLIP_RATIO,
    SPACEMOUSE_ROTATION_DEADZONE,
    SPACEMOUSE_SMOOTHING,
    SPACEMOUSE_TRANSLATION_AXIS_CLIP_MIN,
    SPACEMOUSE_TRANSLATION_AXIS_CLIP_RATIO,
    SPACEMOUSE_TRANSLATION_DEADZONE,
    SPACEMOUSE_VERTICAL_SCALE,
)

AXIS_INDEX = {
    "left_x": 0,
    "left_y": 1,
    "right_x": 2,
    "right_y": 3,
    "left_trigger": 4,
    "right_trigger": 5,
    "dpad_x": 6,
    "dpad_y": 7,
}

BUTTON_INDEX = {
    "face_a": 0,  # switch mode (X)
    "face_b": 1,  # abort
    "left_shoulder": 4,
    "right_shoulder": 5,
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp `value` to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_float(state: Any, name: str) -> float:
    """Return `state.name` as a float, or 0.0 if missing or unconvertible."""
    try:
        return float(getattr(state, name, 0.0))
    except Exception:
        return 0.0


def _axis_value(state: Any, names: Iterable[str]) -> float:
    """Return the first non-zero attribute among `names` on `state`."""
    for name in names:
        value = _safe_float(state, name)
        if value != 0.0:
            return value
    return 0.0


def _apply_deadzone(value: float, deadzone: float) -> float:
    """Suppress small inputs and rescale the rest into [-1, 1]."""
    dz = max(0.0, min(0.95, float(deadzone)))
    av = abs(value)
    if av <= dz:
        return 0.0
    scaled = (av - dz) / (1.0 - dz)
    return scaled if value >= 0.0 else -scaled


def _apply_curve(value: float, curve: float) -> float:
    """Apply a sign-preserving power-law response curve (>=1 sharpens centre)."""
    c = max(1.0, float(curve))
    return (abs(value) ** c) * (1.0 if value >= 0.0 else -1.0)


def _clip_translation_axes(
    x: float, y: float, z: float, clip_min: float, clip_ratio: float
) -> tuple[float, float, float]:
    """Zero out axes weaker than `clip_ratio` × the dominant axis once above `clip_min`."""
    values = [x, y, z]
    abs_values = [abs(v) for v in values]
    dominant = max(abs_values)
    if dominant < float(clip_min):
        return x, y, z
    ratio = float(clip_ratio)
    if ratio <= 0.0:
        return x, y, z
    out = values[:]
    for i, av in enumerate(abs_values):
        if av < dominant * ratio:
            out[i] = 0.0
    return float(out[0]), float(out[1]), float(out[2])


def _clip_rotation_axes(
    roll: float, pitch: float, yaw: float, clip_min: float, clip_ratio: float
) -> tuple[float, float, float]:
    """Same dominant-axis clip as translation, applied to rotation axes."""
    return _clip_translation_axes(roll, pitch, yaw, clip_min, clip_ratio)


def _parse_args() -> argparse.Namespace:
    """Parse CLI / env overrides for the SpaceMouse daemon settings."""
    parser = argparse.ArgumentParser(description="Publish SpaceMouse state over ZMQ.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv(
            "SPACEMOUSE_REMOTE_ENDPOINT",
            os.getenv("GAMEPAD_REMOTE_ENDPOINT", SPACEMOUSE_REMOTE_ENDPOINT),
        ),
        help="ZMQ endpoint for PUB socket.",
    )
    parser.add_argument(
        "--topic",
        default=os.getenv(
            "SPACEMOUSE_REMOTE_TOPIC",
            os.getenv("GAMEPAD_REMOTE_TOPIC", SPACEMOUSE_REMOTE_TOPIC),
        ),
        help="Topic for published remote state.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=float(os.getenv("SPACEMOUSE_REMOTE_HZ", "120")),
        help="Max publish rate.",
    )
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=float(os.getenv("SPACEMOUSE_AXIS_SCALE", str(SPACEMOUSE_AXIS_SCALE))),
        help="Gain applied to raw axes before clamping to [-1, 1].",
    )
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=float(
            os.getenv("SPACEMOUSE_VERTICAL_SCALE", str(SPACEMOUSE_VERTICAL_SCALE))
        ),
        help="Additional gain applied only to vertical (Z) motion.",
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=float(os.getenv("SPACEMOUSE_YAW_BUTTON_THRESHOLD", "0.35")),
        help="Absolute yaw threshold used to emulate L1/R1 yaw buttons.",
    )
    parser.add_argument(
        "--invert-pitch",
        type=int,
        default=int(
            os.getenv(
                "SPACEMOUSE_INVERT_PITCH", "1" if SPACEMOUSE_INVERT_PITCH else "0"
            )
        ),
        help="Invert pitch axis (1 on, 0 off).",
    )
    parser.add_argument(
        "--invert-yaw",
        type=int,
        default=int(
            os.getenv("SPACEMOUSE_INVERT_YAW", "1" if SPACEMOUSE_INVERT_YAW else "0")
        ),
        help="Invert yaw axis (1 on, 0 off).",
    )
    parser.add_argument(
        "--translation-deadzone",
        type=float,
        default=float(
            os.getenv(
                "SPACEMOUSE_TRANSLATION_DEADZONE",
                str(SPACEMOUSE_TRANSLATION_DEADZONE),
            )
        ),
        help="Deadzone for translation axes.",
    )
    parser.add_argument(
        "--rotation-deadzone",
        type=float,
        default=float(
            os.getenv(
                "SPACEMOUSE_ROTATION_DEADZONE",
                str(SPACEMOUSE_ROTATION_DEADZONE),
            )
        ),
        help="Deadzone for rotation axes.",
    )
    parser.add_argument(
        "--response-curve",
        type=float,
        default=float(
            os.getenv("SPACEMOUSE_RESPONSE_CURVE", str(SPACEMOUSE_RESPONSE_CURVE))
        ),
        help="Exponential response curve (>=1).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=float(os.getenv("SPACEMOUSE_SMOOTHING", str(SPACEMOUSE_SMOOTHING))),
        help="Low-pass smoothing [0..1].",
    )
    parser.add_argument(
        "--intent-min",
        type=float,
        default=float(os.getenv("SPACEMOUSE_INTENT_MIN", str(SPACEMOUSE_INTENT_MIN))),
        help="Minimum magnitude to treat translation/rotation as intentional.",
    )
    parser.add_argument(
        "--cross-drift-max",
        type=float,
        default=float(
            os.getenv("SPACEMOUSE_CROSS_DRIFT_MAX", str(SPACEMOUSE_CROSS_DRIFT_MAX))
        ),
        help="Allowed tiny drift in the non-dominant group before suppression.",
    )
    parser.add_argument(
        "--translation-axis-clip-min",
        type=float,
        default=float(
            os.getenv(
                "SPACEMOUSE_TRANSLATION_AXIS_CLIP_MIN",
                str(SPACEMOUSE_TRANSLATION_AXIS_CLIP_MIN),
            )
        ),
        help="Minimum translation magnitude before axis clipping is applied.",
    )
    parser.add_argument(
        "--translation-axis-clip-ratio",
        type=float,
        default=float(
            os.getenv(
                "SPACEMOUSE_TRANSLATION_AXIS_CLIP_RATIO",
                str(SPACEMOUSE_TRANSLATION_AXIS_CLIP_RATIO),
            )
        ),
        help="Suppress weaker translation axes below dominant*ratio.",
    )
    parser.add_argument(
        "--rotation-axis-clip-min",
        type=float,
        default=float(
            os.getenv(
                "SPACEMOUSE_ROTATION_AXIS_CLIP_MIN",
                str(SPACEMOUSE_ROTATION_AXIS_CLIP_MIN),
            )
        ),
        help="Minimum rotation magnitude before axis clipping is applied.",
    )
    parser.add_argument(
        "--rotation-axis-clip-ratio",
        type=float,
        default=float(
            os.getenv(
                "SPACEMOUSE_ROTATION_AXIS_CLIP_RATIO",
                str(SPACEMOUSE_ROTATION_AXIS_CLIP_RATIO),
            )
        ),
        help="Suppress weaker rotation axes below dominant*ratio.",
    )
    parser.add_argument(
        "--mixed-axis-clip-min",
        type=float,
        default=float(
            os.getenv(
                "SPACEMOUSE_MIXED_AXIS_CLIP_MIN",
                str(SPACEMOUSE_MIXED_AXIS_CLIP_MIN),
            )
        ),
        help="Minimum group magnitude before mixed translation/rotation clipping applies.",
    )
    parser.add_argument(
        "--mixed-axis-clip-ratio",
        type=float,
        default=float(
            os.getenv(
                "SPACEMOUSE_MIXED_AXIS_CLIP_RATIO",
                str(SPACEMOUSE_MIXED_AXIS_CLIP_RATIO),
            )
        ),
        help="Suppress weaker translation/rotation group below stronger*ratio.",
    )
    return parser.parse_args()


class _USBState:
    """Minimal state object with the same attribute names pyspacemouse returns."""

    __slots__ = ("x", "y", "z", "roll", "pitch", "yaw", "buttons")

    def __init__(self) -> None:
        """Reset all axes to 0 and buttons to an empty list."""
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.buttons: list[bool] = []


class _USBSpaceMouse:
    """Direct USB reader for SpaceNavigator when hidraw is unavailable.

    Uses pyusb to read raw HID interrupt reports and decode them using the
    same axis/button mapping as pyspacemouse's devices.toml for SpaceNavigator.
    """

    _VID = 0x046D
    _PID = 0xC626
    _IFACE = 0
    _ENDPOINT = 0x81
    _SCALE = 350.0

    def __init__(self, dev: Any, detached: bool) -> None:
        """Wrap an open pyusb device handle and remember whether we detached the kernel driver."""
        self._dev = dev
        self._detached = detached
        self._state = _USBState()

    @classmethod
    def open(cls) -> Optional["_USBSpaceMouse"]:
        """Find and claim a SpaceNavigator over raw USB; return None if unavailable."""
        if not _PYUSB_AVAILABLE:
            return None
        dev = _usbcore.find(idVendor=cls._VID, idProduct=cls._PID)
        if dev is None:
            return None
        try:
            detached = False
            if dev.is_kernel_driver_active(cls._IFACE):
                dev.detach_kernel_driver(cls._IFACE)
                detached = True
            dev.set_configuration()
            _usbutil.claim_interface(dev, cls._IFACE)
            return cls(dev, detached)
        except Exception:
            return None

    @staticmethod
    def _s16(data: Any, offset: int) -> int:
        """Decode a little-endian signed 16-bit integer from `data[offset:offset+2]`."""
        raw = int(data[offset]) | (int(data[offset + 1]) << 8)
        return raw - 65536 if raw >= 32768 else raw

    def read(self) -> Optional[_USBState]:
        """Read one HID report and update the cached state, or return None on timeout."""
        try:
            data = self._dev.read(self._ENDPOINT, 7, timeout=50)
        except _usbcore.USBError as exc:
            if getattr(exc, "errno", None) == 110:  # ETIMEDOUT
                return None
            raise
        except Exception:
            return None
        if not data or len(data) < 2:
            return None
        rid = data[0]
        s = self._s16
        scale = self._SCALE
        if rid == 1 and len(data) >= 7:
            self._state.x = s(data, 1) / scale
            self._state.y = -s(data, 3) / scale
            self._state.z = -s(data, 5) / scale
        elif rid == 2 and len(data) >= 7:
            self._state.pitch = -s(data, 1) / scale
            self._state.roll = -s(data, 3) / scale
            self._state.yaw = s(data, 5) / scale
        elif rid == 3 and len(data) >= 2:
            bits = int(data[1])
            self._state.buttons = [bool(bits & 1), bool(bits & 2)]
        return self._state

    def close(self) -> None:
        """Release the USB interface and reattach the kernel driver if we detached it."""
        try:
            _usbutil.release_interface(self._dev, self._IFACE)
            if self._detached:
                self._dev.attach_kernel_driver(self._IFACE)
        except Exception:
            pass


def _open_device_quiet() -> tuple[bool, Optional[str], Any]:
    """Open the SpaceMouse via pyspacemouse, then pyusb fallback, suppressing log spam."""
    # pyspacemouse.open() prints discovery lines; capture them to avoid spam.
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            handle = pyspacemouse.open()
        if hasattr(handle, "read"):
            return True, None, handle
        if bool(handle):
            if callable(getattr(pyspacemouse, "read", None)):
                return True, None, None
            return (
                False,
                "open succeeded but no readable handle and no module read()",
                None,
            )
        msg = (
            buf_err.getvalue() or buf_out.getvalue()
        ).strip() or "open returned False"
        usb_handle = _USBSpaceMouse.open()
        if usb_handle is not None:
            return True, None, usb_handle
        return False, msg, None
    except Exception as exc:
        usb_handle = _USBSpaceMouse.open()
        if usb_handle is not None:
            return True, None, usb_handle
        return False, str(exc), None


def _read_state(device_handle: Any) -> Any:
    """Read one state sample, preferring an explicit handle over the module-level read."""
    if device_handle is not None and hasattr(device_handle, "read"):
        return device_handle.read()
    if callable(getattr(pyspacemouse, "read", None)):
        return pyspacemouse.read()
    raise RuntimeError(
        "pyspacemouse has no read() and open() did not return a read-capable device"
    )


def main() -> int:
    """Read the SpaceMouse and republish a gamepad-schema payload over ZMQ."""
    args = _parse_args()
    dump = os.getenv("SPACEMOUSE_DUMP") == "1"

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 100)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.bind(args.endpoint)
    except Exception as exc:
        print(f"[spacemouse-daemon] failed to bind {args.endpoint}: {exc}")
        return 1

    axes: Dict[str, float] = {
        "left_x": 0.0,
        "left_y": 0.0,
        "right_x": 0.0,
        "right_y": 0.0,
        "left_trigger": 0.0,
        "right_trigger": 0.0,
        "dpad_x": 0.0,
        "dpad_y": 0.0,
    }
    buttons: Dict[str, bool] = {
        "face_a": False,
        "face_b": False,
        "left_shoulder": False,
        "right_shoulder": False,
    }
    axes_list = [0.0] * 8
    buttons_list = [0] * 20

    min_period = 1.0 / max(1.0, float(args.hz))
    last_pub = 0.0

    opened = False
    device_handle: Any = None
    next_open_at = 0.0
    open_backoff_s = 0.25
    max_backoff_s = 5.0
    last_open_error: Optional[str] = None
    last_error_log = 0.0
    smoothing_alpha = _clamp(float(args.smoothing), 0.0, 1.0)
    filtered = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "rr": 0.0, "rp": 0.0, "ry": 0.0}

    print(f"[spacemouse-daemon] publishing to {args.endpoint} {args.topic}")

    while True:
        if not opened:
            now_mono = time.monotonic()
            if now_mono < next_open_at:
                time.sleep(min(0.05, max(0.0, next_open_at - now_mono)))
                continue
            opened, err, device_handle = _open_device_quiet()
            if not opened:
                if err != last_open_error or (now_mono - last_error_log) >= 2.0:
                    print(f"[spacemouse-daemon] open failed: {err}")
                    last_open_error = err
                    last_error_log = now_mono
                next_open_at = now_mono + open_backoff_s
                open_backoff_s = min(max_backoff_s, open_backoff_s * 1.7)
                continue
            open_backoff_s = 0.25
            last_open_error = None
            continue

        try:
            state = _read_state(device_handle)
        except Exception as exc:
            now_mono = time.monotonic()
            if (now_mono - last_error_log) >= 1.0:
                print(f"[spacemouse-daemon] read failed: {exc}")
                last_error_log = now_mono
            try:
                if device_handle is not None and hasattr(device_handle, "close"):
                    device_handle.close()
                elif callable(getattr(pyspacemouse, "close", None)):
                    pyspacemouse.close()
            except Exception:
                pass
            opened = False
            device_handle = None
            next_open_at = time.monotonic() + open_backoff_s
            open_backoff_s = min(max_backoff_s, open_backoff_s * 1.5)
            continue

        if state is None:
            time.sleep(0.001)
            continue

        now = time.time()
        if (now - last_pub) < min_period:
            continue
        last_pub = now

        gain = max(1e-6, float(args.axis_scale))

        tx = _axis_value(state, ("x", "tx", "trans_x"))
        ty = _axis_value(state, ("y", "ty", "trans_y"))
        tz = _axis_value(state, ("z", "tz", "trans_z"))
        rr = _axis_value(state, ("roll", "rx", "rot_x"))
        rp = _axis_value(state, ("pitch", "ry", "rot_y"))
        ry = _axis_value(state, ("yaw", "rz", "rot_z"))

        tx = _clamp(tx * gain, -1.0, 1.0)
        ty = _clamp(ty * gain, -1.0, 1.0)
        tz = _clamp(tz * gain * float(args.vertical_scale), -1.0, 1.0)
        rr = _clamp(rr * gain, -1.0, 1.0)
        rp = _clamp(rp * gain, -1.0, 1.0)
        ry = _clamp(ry * gain, -1.0, 1.0)
        if bool(int(args.invert_pitch)):
            rp = -rp
        if bool(int(args.invert_yaw)):
            ry = -ry

        tx = _apply_deadzone(tx, args.translation_deadzone)
        ty = _apply_deadzone(ty, args.translation_deadzone)
        tz = _apply_deadzone(tz, args.translation_deadzone)
        rr = _apply_deadzone(rr, args.rotation_deadzone)
        rp = _apply_deadzone(rp, args.rotation_deadzone)
        ry = _apply_deadzone(ry, args.rotation_deadzone)

        tx = _apply_curve(tx, args.response_curve)
        ty = _apply_curve(ty, args.response_curve)
        tz = _apply_curve(tz, args.response_curve)
        rr = _apply_curve(rr, args.response_curve)
        rp = _apply_curve(rp, args.response_curve)
        ry = _apply_curve(ry, args.response_curve)

        # Preserve multi-axis intent; only suppress tiny cross-group drift.
        t_max = max(abs(tx), abs(ty), abs(tz))
        r_max = max(abs(rr), abs(rp), abs(ry))
        intent_min = float(args.intent_min)
        cross_drift_max = float(args.cross_drift_max)
        if t_max >= intent_min and r_max <= cross_drift_max:
            rr = rp = ry = 0.0
        elif r_max >= intent_min and t_max <= cross_drift_max:
            tx = ty = tz = 0.0

        for key, value in (
            ("tx", tx),
            ("ty", ty),
            ("tz", tz),
            ("rr", rr),
            ("rp", rp),
            ("ry", ry),
        ):
            filtered[key] = (1.0 - smoothing_alpha) * filtered[key] + (
                smoothing_alpha * float(value)
            )

        tx = filtered["tx"]
        ty = filtered["ty"]
        tz = filtered["tz"]
        rr = filtered["rr"]
        rp = filtered["rp"]
        ry = filtered["ry"]

        tx, ty, tz = _clip_translation_axes(
            tx,
            ty,
            tz,
            args.translation_axis_clip_min,
            args.translation_axis_clip_ratio,
        )
        rr, rp, ry = _clip_rotation_axes(
            rr,
            rp,
            ry,
            args.rotation_axis_clip_min,
            args.rotation_axis_clip_ratio,
        )

        t_max = max(abs(tx), abs(ty), abs(tz))
        r_max = max(abs(rr), abs(rp), abs(ry))
        mixed_min = float(args.mixed_axis_clip_min)
        mixed_ratio = float(args.mixed_axis_clip_ratio)
        if t_max >= mixed_min and r_max <= t_max * mixed_ratio:
            rr = rp = ry = 0.0
        elif r_max >= mixed_min and t_max <= r_max * mixed_ratio:
            tx = ty = tz = 0.0

        # Gamepad-compatible mapping used by InputController robot mode:
        # lin_x = -left_y, lin_y = left_x, lin_z = right_trigger - left_trigger
        axes["left_x"] = tx
        axes["left_y"] = -ty
        axes["right_x"] = rr
        axes["right_y"] = rp
        axes["left_trigger"] = max(0.0, -tz)
        axes["right_trigger"] = max(0.0, tz)

        yaw_thr = abs(float(args.yaw_threshold))
        buttons["left_shoulder"] = ry >= yaw_thr
        buttons["right_shoulder"] = ry <= -yaw_thr

        raw_buttons = getattr(state, "buttons", [])
        if isinstance(raw_buttons, (list, tuple)):
            hw_buttons = [bool(v) for v in raw_buttons]
        else:
            hw_buttons = []

        # Requested mapping:
        # button 0 -> abort (face_b), button 1 -> switch mode (face_a)
        buttons["face_b"] = bool(hw_buttons[0]) if len(hw_buttons) > 0 else False
        buttons["face_a"] = bool(hw_buttons[1]) if len(hw_buttons) > 1 else False

        for name, idx in AXIS_INDEX.items():
            axes_list[idx] = float(axes.get(name, 0.0))
        for name, idx in BUTTON_INDEX.items():
            buttons_list[idx] = 1 if buttons.get(name, False) else 0

        payload = {
            "name": "spacemouse",
            "axes": axes,
            "buttons": buttons,
            "axes_list": axes_list,
            "buttons_list": buttons_list,
            "ts": now,
        }
        if dump:
            print(
                "[spacemouse-daemon] "
                f"tx={tx:+.3f} ty={ty:+.3f} tz={tz:+.3f} "
                f"roll={rr:+.3f} pitch={rp:+.3f} yaw={ry:+.3f} "
                f"buttons=[{int(buttons['face_b'])},{int(buttons['face_a'])}]"
            )
        try:
            sock.send_json({"topic": args.topic, "data": payload}, zmq.NOBLOCK)
        except zmq.Again:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
