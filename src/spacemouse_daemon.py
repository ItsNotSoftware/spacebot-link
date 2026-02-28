"""SpaceMouse daemon: read pyspacemouse state and publish over ZMQ."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import time
from typing import Any, Optional

import zmq

try:
    import pyspacemouse
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "pyspacemouse library not available. Install with: uv add pyspacemouse"
    ) from exc

from config import SPACEMOUSE_REMOTE_ENDPOINT, SPACEMOUSE_REMOTE_TOPIC


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish SpaceMouse state over ZMQ.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("SPACEMOUSE_REMOTE_ENDPOINT", SPACEMOUSE_REMOTE_ENDPOINT),
        help="ZMQ endpoint for PUB socket.",
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("SPACEMOUSE_REMOTE_TOPIC", SPACEMOUSE_REMOTE_TOPIC),
        help="Topic for published SpaceMouse state.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=float(os.getenv("SPACEMOUSE_REMOTE_HZ", "120")),
        help="Max publish rate.",
    )
    return parser.parse_args()


def _open_device_quiet() -> tuple[bool, Optional[str], Any]:
    # pyspacemouse.open() prints discovery lines (e.g. "SpaceNavigator found").
    # Capture that output so reconnect loops don't flood the terminal.
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
        return False, msg, None
    except Exception as exc:
        return False, str(exc), None


def _safe_float(state: Any, attr: str) -> float:
    try:
        return float(getattr(state, attr, 0.0))
    except Exception:
        return 0.0


def main() -> int:
    args = _parse_args()
    dump_state = os.getenv("SPACEMOUSE_DUMP") == "1"

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 100)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.bind(args.endpoint)
    except Exception as exc:
        print(f"[spacemouse-daemon] failed to bind {args.endpoint}: {exc}")
        return 1

    min_period = 1.0 / max(1.0, float(args.hz))
    last_pub = 0.0
    opened = False
    device_handle: Any = None
    next_open_at = 0.0
    open_backoff_s = 0.25
    max_backoff_s = 5.0
    last_open_error: Optional[str] = None
    last_error_log = 0.0

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
            if device_handle is not None and hasattr(device_handle, "read"):
                state = device_handle.read()
            elif callable(getattr(pyspacemouse, "read", None)):
                state = pyspacemouse.read()
            else:
                raise RuntimeError(
                    "pyspacemouse has no read() and open() did not return a read-capable device"
                )
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

        x = _safe_float(state, "x")
        y = _safe_float(state, "y")
        z = _safe_float(state, "z")
        roll = _safe_float(state, "roll")
        pitch = _safe_float(state, "pitch")
        yaw = _safe_float(state, "yaw")

        buttons_raw = getattr(state, "buttons", [])
        if isinstance(buttons_raw, (list, tuple)):
            buttons_list = [bool(v) for v in buttons_raw]
        else:
            buttons_list = []
        left = bool(buttons_list[0]) if len(buttons_list) > 0 else False
        right = bool(buttons_list[1]) if len(buttons_list) > 1 else False

        payload = {
            "name": "spacemouse",
            "axes": {
                "x": x,
                "y": y,
                "z": z,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            },
            "buttons": {
                "button_0": left,
                "button_1": right,
                "left": left,
                "right": right,
            },
            "axes_list": [x, y, z, roll, pitch, yaw],
            "buttons_list": [left, right],
            "ts": now,
        }
        if dump_state:
            print(
                "[spacemouse-daemon] "
                f"x={x:+.3f} y={y:+.3f} z={z:+.3f} "
                f"roll={roll:+.3f} pitch={pitch:+.3f} yaw={yaw:+.3f} "
                f"buttons=[{int(left)},{int(right)}]"
            )
        try:
            sock.send_json({"topic": args.topic, "data": payload}, zmq.NOBLOCK)
        except zmq.Again:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
