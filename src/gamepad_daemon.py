"""Gamepad daemon: read inputs, normalize, publish state over ZMQ."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import zmq

try:
    from inputs import get_gamepad
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit("inputs library not available. Install with: uv add inputs") from exc

from config import GAMEPAD_REMOTE_ENDPOINT, GAMEPAD_REMOTE_TOPIC

STICK_AXES = {
    "ABS_X": "left_x",
    "ABS_Y": "left_y",
    "ABS_LX": "left_x",
    "ABS_LY": "left_y",
    "ABS_RX": "right_x",
    "ABS_RY": "right_y",
}

TRIGGER_AXES = {
    "ABS_Z": "left_trigger",
    "ABS_RZ": "right_trigger",
    "ABS_LT": "left_trigger",
    "ABS_RT": "right_trigger",
    "ABS_L2": "left_trigger",
    "ABS_R2": "right_trigger",
}

DPAD_AXES = {
    "ABS_HAT0X": "dpad_x",
    "ABS_HAT0Y": "dpad_y",
}

BUTTONS = {
    "BTN_TL": "left_shoulder",
    "BTN_TR": "right_shoulder",
    "BTN_SOUTH": "face_a",
    "BTN_EAST": "face_b",
    "BTN_NORTH": "face_y",
    "BTN_WEST": "face_x",
    "BTN_SELECT": "touchpad",
    "BTN_MODE": "touchpad",
    "BTN_TOUCHPAD": "touchpad",
    "BTN_THUMBL": "left_stick",
    "BTN_THUMBR": "right_stick",
}

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
    "face_a": 0,
    "face_b": 1,
    "face_x": 2,
    "face_y": 3,
    "left_shoulder": 4,
    "right_shoulder": 5,
    "touchpad": 13,
    "left_stick": 10,
    "right_stick": 11,
}

AXIS_CENTER_UPDATE_BAND = 4
AXIS_RANGE_MIN = 8
_axis_cal: Dict[str, tuple[float, float]] = {}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp `value` to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def _norm_stick(value: int) -> float:
    """Map a raw stick reading to [-1, 1] using the device's apparent range."""
    if -32768 <= value <= 32767:
        return _clamp(value / 32767.0, -1.0, 1.0)
    if 0 <= value <= 255:
        return _clamp((value - 128.0) / 127.0, -1.0, 1.0)
    if 0 <= value <= 1023:
        return _clamp((value - 512.0) / 512.0, -1.0, 1.0)
    if 0 <= value <= 65535:
        return _clamp((value - 32768.0) / 32767.0, -1.0, 1.0)
    return _clamp(value / 32767.0, -1.0, 1.0)


def _norm_stick_calibrated(code: str, value: int) -> float:
    """Self-calibrating stick normalisation that tracks centre and peak per axis."""
    if not (0 <= value <= 255):
        return _norm_stick(value)
    center, peak = _axis_cal.get(code, (float(value), 0.0))
    if abs(value - center) <= AXIS_CENTER_UPDATE_BAND:
        center = (center * 0.98) + (value * 0.02)
    dev = abs(value - center)
    if dev > peak:
        peak = dev
    _axis_cal[code] = (center, peak)
    denom = max(peak, float(AXIS_RANGE_MIN))
    return _clamp((value - center) / denom, -1.0, 1.0)


def _norm_trigger(value: int) -> float:
    """Map a raw trigger reading to [0, 1] using the device's apparent range."""
    if 0 <= value <= 255:
        return _clamp(value / 255.0, 0.0, 1.0)
    if 0 <= value <= 1023:
        return _clamp(value / 1023.0, 0.0, 1.0)
    if -32768 <= value <= 32767:
        return _clamp((value + 32768.0) / 65535.0, 0.0, 1.0)
    if 0 <= value <= 65535:
        return _clamp(value / 65535.0, 0.0, 1.0)
    return _clamp(value / 255.0, 0.0, 1.0)


def _parse_args() -> argparse.Namespace:
    """Parse CLI / env overrides for the daemon's publish settings."""
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
    """Read gamepad events and republish normalised state on a ZMQ PUB socket."""
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
                    name = STICK_AXES[ev.code]
                    val = _norm_stick_calibrated(ev.code, int(ev.state))
                elif ev.code in TRIGGER_AXES:
                    name = TRIGGER_AXES[ev.code]
                    val = _norm_trigger(int(ev.state))
                elif ev.code in DPAD_AXES:
                    name = DPAD_AXES[ev.code]
                    val = _clamp(float(ev.state), -1.0, 1.0)
                else:
                    name = None
                    val = None
                if name is not None and val is not None:
                    axes[name] = float(val)
                    idx = AXIS_INDEX.get(name)
                    if idx is not None and 0 <= idx < len(axes_list):
                        axes_list[idx] = float(val)
                    changed = True
                if ev.code.startswith("ABS_"):
                    if ev.code in TRIGGER_AXES:
                        raw_val = _norm_trigger(int(ev.state))
                    else:
                        raw_val = _norm_stick(int(ev.state))
                    axes[ev.code.lower()] = raw_val
                    changed = True
            elif ev.ev_type == "Key":
                if ev.code in BUTTONS:
                    name = BUTTONS[ev.code]
                    val = bool(ev.state)
                    buttons[name] = val
                    idx = BUTTON_INDEX.get(name)
                    if idx is not None and 0 <= idx < len(buttons_list):
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
