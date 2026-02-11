"""External gamepad reader that publishes normalized state over ZMQ."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import zmq

try:
    from inputs import get_gamepad
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "inputs library not available. Install with: uv add inputs"
    ) from exc

from config import (
    GAMEPAD_REMOTE_ENDPOINT,
    GAMEPAD_REMOTE_TOPIC,
)

STICK_AXES = {
    "ABS_X": ("left_x", 0),
    "ABS_Y": ("left_y", 1),
    "ABS_LX": ("left_x", 0),
    "ABS_LY": ("left_y", 1),
    "ABS_RX": ("right_x", 2),
    "ABS_RY": ("right_y", 3),
}

TRIGGER_AXES = {
    "ABS_Z": ("left_trigger", 4),
    "ABS_RZ": ("right_trigger", 5),
}

BUTTONS = {
    "BTN_TL": ("left_shoulder", 4),
    "BTN_TR": ("right_shoulder", 5),
    "BTN_SOUTH": ("face_a", 0),
    "BTN_NORTH": ("face_y", 3),
    "BTN_SELECT": ("touchpad", 13),
    "BTN_TOUCHPAD": ("touchpad", 13),
    "BTN_MODE": ("touchpad", 13),
}

AXIS_RANGE_MIN = 6
AXIS_CENTER_UPDATE_BAND = 4
_axis_cal: Dict[str, Tuple[float, float]] = {}


def _norm_signed(value: int) -> float:
    if -1 <= value <= 1:
        return float(value)
    if -32768 <= value <= 32767:
        return max(-1.0, min(1.0, value / 32767.0))
    if 0 <= value <= 255:
        return max(-1.0, min(1.0, (value - 128) / 128.0))
    if 0 <= value <= 1023:
        return max(-1.0, min(1.0, (value - 512) / 512.0))
    if 0 <= value <= 65535:
        return max(-1.0, min(1.0, (value - 32768) / 32768.0))
    # Fallback: scale by a large constant to avoid huge values.
    return max(-1.0, min(1.0, float(value) / 32767.0))


def _norm_signed_calibrated(code: str, value: int) -> float:
    if not (0 <= value <= 255):
        return _norm_signed(value)
    center, max_dev = _axis_cal.get(code, (float(value), 0.0))
    if abs(value - center) <= AXIS_CENTER_UPDATE_BAND:
        center = (center * 0.98) + (value * 0.02)
    dev = abs(value - center)
    if dev > max_dev:
        max_dev = dev
    _axis_cal[code] = (center, max_dev)
    denom = max(max_dev, float(AXIS_RANGE_MIN))
    return max(-1.0, min(1.0, (value - center) / denom))


def _norm_trigger(value: int) -> float:
    if 0 <= value <= 255:
        return value / 255.0
    if 0 <= value <= 1023:
        return value / 1023.0
    if -32768 <= value <= 32767:
        return max(0.0, min(1.0, (value + 32768.0) / 65535.0))
    if 0 <= value <= 65535:
        return value / 65535.0
    return max(0.0, min(1.0, float(value) / 255.0))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish gamepad state over ZMQ.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("GAMEPAD_REMOTE_ENDPOINT", GAMEPAD_REMOTE_ENDPOINT),
        help="ZMQ endpoint for PUB socket.",
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("GAMEPAD_REMOTE_TOPIC", GAMEPAD_REMOTE_TOPIC),
        help="Topic for published gamepad state.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=float(os.getenv("GAMEPAD_REMOTE_HZ", "60")),
        help="Max publish rate.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dump_events = os.getenv("GAMEPAD_DUMP") == "1"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 100)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(args.endpoint)

    axes: Dict[str, float] = {}
    buttons: Dict[str, bool] = {}
    axes_list = [0.0] * 8
    buttons_list = [0] * 20
    last_pub = 0.0
    min_period = 1.0 / max(1.0, args.hz)

    print(f"[gamepad-daemon] publishing to {args.endpoint} {args.topic}")

    while True:
        try:
            events = get_gamepad()
        except Exception:
            time.sleep(0.5)
            continue

        changed = False
        for ev in events:
            if dump_events:
                print(f"[gamepad-daemon] {ev.ev_type} {ev.code} {ev.state}")
            if ev.ev_type == "Absolute":
                if ev.code in STICK_AXES:
                    name, idx = STICK_AXES[ev.code]
                    val = _norm_signed_calibrated(ev.code, int(ev.state))
                    axes[name] = val
                    if 0 <= idx < len(axes_list):
                        axes_list[idx] = val
                    changed = True
                elif ev.code in TRIGGER_AXES:
                    name, idx = TRIGGER_AXES[ev.code]
                    val = _norm_trigger(int(ev.state))
                    axes[name] = val
                    if 0 <= idx < len(axes_list):
                        axes_list[idx] = val
                    changed = True
                # Always publish raw ABS_* names as fallback
                if ev.code.startswith("ABS_"):
                    code_name = ev.code.lower()
                    if ev.code in TRIGGER_AXES:
                        raw_val = _norm_trigger(int(ev.state))
                    else:
                        raw_val = _norm_signed(int(ev.state))
                    axes[code_name] = raw_val
                    changed = True
            elif ev.ev_type == "Key":
                if ev.code in BUTTONS:
                    name, idx = BUTTONS[ev.code]
                    val = bool(ev.state)
                    buttons[name] = val
                    if 0 <= idx < len(buttons_list):
                        buttons_list[idx] = 1 if val else 0
                    changed = True

        now = time.time()
        if changed and (now - last_pub) >= min_period:
            last_pub = now
            payload = {
                "name": "inputs",
                "axes": axes,
                "buttons": buttons,
                "axes_list": axes_list,
                "buttons_list": buttons_list,
                "ts": now,
            }
            try:
                sock.send_json({"topic": args.topic, "data": payload}, zmq.NOBLOCK)
            except zmq.Again:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
