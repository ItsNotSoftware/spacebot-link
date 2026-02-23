"""Keyboard polling for avatar movement or robot cmd_vel teleop."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

from panda3d.core import ClockObject, Vec3

from config import (
    BACKWARD_BUTTON,
    DOWN_BUTTON,
    DOWN_BUTTON_ALT,
    FORWARD_BUTTON,
    GAMEPAD_AXIS_L2,
    GAMEPAD_AXIS_DPAD_X,
    GAMEPAD_AXIS_DPAD_Y,
    GAMEPAD_AXIS_LEFT_X,
    GAMEPAD_AXIS_LEFT_Y,
    GAMEPAD_AXIS_INDEX_DPAD_X,
    GAMEPAD_AXIS_INDEX_DPAD_Y,
    GAMEPAD_AXIS_INDEX_L2,
    GAMEPAD_AXIS_INDEX_LEFT_X,
    GAMEPAD_AXIS_INDEX_LEFT_Y,
    GAMEPAD_AXIS_INDEX_R2,
    GAMEPAD_AXIS_INDEX_RIGHT_X,
    GAMEPAD_AXIS_INDEX_RIGHT_Y,
    GAMEPAD_AXIS_R2,
    GAMEPAD_AXIS_RIGHT_X,
    GAMEPAD_AXIS_RIGHT_Y,
    GAMEPAD_AXIS_LOCK_RATIO_LEFT,
    GAMEPAD_AXIS_LOCK_RATIO_RIGHT,
    GAMEPAD_BUTTON_L1,
    GAMEPAD_BUTTON_R1,
    GAMEPAD_BUTTON_ABORT,
    GAMEPAD_BUTTON_L3,
    GAMEPAD_BUTTON_R3,
    GAMEPAD_BUTTON_TOUCHPAD,
    GAMEPAD_BUTTON_TRIANGLE,
    GAMEPAD_BUTTON_X,
    GAMEPAD_BUTTON_INDEX_ABORT,
    GAMEPAD_BUTTON_INDEX_L1,
    GAMEPAD_BUTTON_INDEX_R1,
    GAMEPAD_BUTTON_INDEX_L3,
    GAMEPAD_BUTTON_INDEX_R3,
    GAMEPAD_BUTTON_INDEX_TOUCHPAD,
    GAMEPAD_BUTTON_INDEX_TRIANGLE,
    GAMEPAD_BUTTON_INDEX_X,
    GAMEPAD_ENABLED,
    GAMEPAD_INVERT_PITCH,
    GAMEPAD_INVERT_ROLL,
    GAMEPAD_LOOK_CURVE,
    GAMEPAD_LOOK_DEADZONE,
    GAMEPAD_LOOK_SCALE,
    GAMEPAD_LOOK_SMOOTHING,
    GAMEPAD_MOVE_CURVE,
    GAMEPAD_MOVE_DEADZONE,
    GAMEPAD_MOVE_SCALE,
    GAMEPAD_MOVE_SMOOTHING,
    GAMEPAD_AUTOSCALE_DECAY,
    GAMEPAD_AUTOSCALE_MAX_GAIN,
    GAMEPAD_AUTOSCALE_MIN,
    GAMEPAD_DPAD_THRESHOLD,
    GAMEPAD_REMOTE_AUTOSTART,
    GAMEPAD_REMOTE_ENABLED,
    GAMEPAD_REMOTE_ENDPOINT,
    GAMEPAD_REMOTE_TOPIC,
    GAMEPAD_TRIGGER_DEADZONE,
    GAMEPAD_TRIGGER_SMOOTHING,
    LEFT_BUTTON,
    MOVE_SPEED,
    RESET_ORIENT_BUTTON,
    RESET_TO_ROBOT_ORIENT_BUTTON,
    RIGHT_BUTTON,
    ROTATE_SPEED,
    ROLL_LEFT_BUTTON,
    ROLL_RIGHT_BUTTON,
    PITCH_DOWN_BUTTON,
    PITCH_UP_BUTTON,
    UP_BUTTON,
    UP_BUTTON_ALT,
    YAW_LEFT_BUTTON,
    YAW_RIGHT_BUTTON,
    TOPIC_CMD_VEL,
)
from teleop_bus import TeleopBusSub


class _RemoteGamepad:
    def __init__(self) -> None:
        self.name = "remote"
        self.axes: dict[str, float] = {}
        self.buttons: dict[str, bool] = {}
        self.axes_list: list[float] = []
        self.buttons_list: list[bool] = []
        self.last_update: float = 0.0
        self.last_payload_ts: float = 0.0

    def update(self, payload: dict[str, Any], now: float) -> None:
        self.name = str(payload.get("name") or "remote")
        axes = payload.get("axes")
        buttons = payload.get("buttons")
        axes_list = payload.get("axes_list")
        buttons_list = payload.get("buttons_list")
        if isinstance(axes, dict):
            self.axes = {str(k): float(v) for k, v in axes.items()}
        if isinstance(buttons, dict):
            self.buttons = {str(k): bool(v) for k, v in buttons.items()}
        if isinstance(axes_list, list):
            self.axes_list = [float(v) for v in axes_list]
        if isinstance(buttons_list, list):
            self.buttons_list = [bool(v) for v in buttons_list]
        self.last_payload_ts = float(payload.get("ts", 0.0))
        self.last_update = now

    def axis_value(self, axis_names: tuple[str, ...], axis_index: Optional[int]) -> float:
        for name in axis_names:
            if name in self.axes:
                return float(self.axes[name])
        if axis_index is not None and 0 <= axis_index < len(self.axes_list):
            return float(self.axes_list[axis_index])
        return 0.0

    def button_value(
        self, button_names: tuple[str, ...], button_index: Optional[int]
    ) -> bool:
        for name in button_names:
            if name in self.buttons:
                return bool(self.buttons[name])
        if button_index is not None and 0 <= button_index < len(self.buttons_list):
            return bool(self.buttons_list[button_index])
        return False


class _AxisFilter:
    def __init__(
        self,
        deadzone: float,
        curve: float,
        smoothing: float,
        autoscale_min: float,
        autoscale_max_gain: float,
        autoscale_decay: float,
    ) -> None:
        self.deadzone = float(deadzone)
        self.curve = float(curve)
        self.smoothing = float(smoothing)
        self.autoscale_min = float(autoscale_min)
        self.autoscale_max_gain = float(autoscale_max_gain)
        self.autoscale_decay = float(autoscale_decay)
        self.value: float = 0.0
        self._peak: float = 0.0

    def apply(self, raw: float) -> float:
        # Apply deadzone before autoscale so small stick drift is never amplified.
        value = self._apply_deadzone(raw)
        value = self._autoscale(value)
        if self.curve > 1.0:
            value = math.copysign(abs(value) ** self.curve, value)
        if self.smoothing > 0.0:
            alpha = max(0.0, min(1.0, self.smoothing))
            self.value = (1.0 - alpha) * self.value + alpha * value
            return self.value
        self.value = value
        return value

    def _apply_deadzone(self, value: float) -> float:
        deadzone = max(0.0, min(0.999, self.deadzone))
        if abs(value) <= deadzone:
            return 0.0
        scaled = (abs(value) - deadzone) / (1.0 - deadzone)
        return scaled * (1.0 if value >= 0.0 else -1.0)

    def _autoscale(self, value: float) -> float:
        if self.autoscale_max_gain <= 1.0 or self.autoscale_min <= 0.0:
            return value
        peak = max(self._peak * self.autoscale_decay, abs(value))
        self._peak = peak
        if peak <= 1e-6 or peak >= self.autoscale_min:
            return value
        gain = min(self.autoscale_max_gain, self.autoscale_min / peak)
        return max(-1.0, min(1.0, value * gain))


class _TriggerFilter:
    def __init__(self, deadzone: float, smoothing: float) -> None:
        self.deadzone = float(deadzone)
        self.smoothing = float(smoothing)
        self.value: float = 0.0

    def apply(self, raw: float) -> float:
        value = max(0.0, min(1.0, raw))
        if value < self.deadzone:
            value = 0.0
        if self.smoothing > 0.0:
            alpha = max(0.0, min(1.0, self.smoothing))
            self.value = (1.0 - alpha) * self.value + alpha * value
            return self.value
        self.value = value
        return value


class InputController:
    def __init__(
        self,
        base: Any,
        renderer: Any,
        nav: Any,
        cmd_pub: Any,
        on_abort: Optional[callable] = None,
        on_toggle_mode: Optional[callable] = None,
    ) -> None:
        """Handle keyboard/gamepad input for avatar motion or robot cmd_vel teleop."""
        self.base = base
        self.renderer = renderer
        self.nav = nav
        self.cmd_pub = cmd_pub
        self._on_abort = on_abort
        self._on_toggle_mode = on_toggle_mode
        self._move_robot = False
        self._nav_publish_before_robot_mode: Optional[bool] = None
        self._button_prev: dict[str, bool] = {}
        self._dpad_prev: dict[str, float] = {"x": 0.0, "y": 0.0}
        self._gamepad_logged = False
        self._remote_sub: Optional[TeleopBusSub] = None
        self._remote_gamepad: Optional[_RemoteGamepad] = None
        self._remote_enabled = bool(
            GAMEPAD_REMOTE_ENABLED
            or GAMEPAD_REMOTE_AUTOSTART
            or os.getenv("GAMEPAD_REMOTE") == "1"
        )
        if self._remote_enabled:
            self._remote_sub = TeleopBusSub(GAMEPAD_REMOTE_ENDPOINT)
            self._remote_gamepad = _RemoteGamepad()

        self._move_filter_x = _AxisFilter(
            GAMEPAD_MOVE_DEADZONE,
            GAMEPAD_MOVE_CURVE,
            GAMEPAD_MOVE_SMOOTHING,
            GAMEPAD_AUTOSCALE_MIN,
            GAMEPAD_AUTOSCALE_MAX_GAIN,
            GAMEPAD_AUTOSCALE_DECAY,
        )
        self._move_filter_y = _AxisFilter(
            GAMEPAD_MOVE_DEADZONE,
            GAMEPAD_MOVE_CURVE,
            GAMEPAD_MOVE_SMOOTHING,
            GAMEPAD_AUTOSCALE_MIN,
            GAMEPAD_AUTOSCALE_MAX_GAIN,
            GAMEPAD_AUTOSCALE_DECAY,
        )
        self._look_filter_x = _AxisFilter(
            GAMEPAD_LOOK_DEADZONE,
            GAMEPAD_LOOK_CURVE,
            GAMEPAD_LOOK_SMOOTHING,
            GAMEPAD_AUTOSCALE_MIN,
            GAMEPAD_AUTOSCALE_MAX_GAIN,
            GAMEPAD_AUTOSCALE_DECAY,
        )
        self._look_filter_y = _AxisFilter(
            GAMEPAD_LOOK_DEADZONE,
            GAMEPAD_LOOK_CURVE,
            GAMEPAD_LOOK_SMOOTHING,
            GAMEPAD_AUTOSCALE_MIN,
            GAMEPAD_AUTOSCALE_MAX_GAIN,
            GAMEPAD_AUTOSCALE_DECAY,
        )
        self._trigger_filter_l = _TriggerFilter(
            GAMEPAD_TRIGGER_DEADZONE, GAMEPAD_TRIGGER_SMOOTHING
        )
        self._trigger_filter_r = _TriggerFilter(
            GAMEPAD_TRIGGER_DEADZONE, GAMEPAD_TRIGGER_SMOOTHING
        )

    def set_move_mode(self, move_robot: bool) -> None:
        """Toggle control target between avatar and robot."""
        move_robot = bool(move_robot)
        if move_robot == self._move_robot:
            return
        self._move_robot = move_robot
        target = "Robot" if self._move_robot else "Avatar"
        self.nav.set_move_target(target)
        if self._move_robot:
            if self._nav_publish_before_robot_mode is None:
                self._nav_publish_before_robot_mode = bool(
                    self.nav.state.nav_publishing_enabled
                )
            self.nav.state.nav_publishing_enabled = False
            self.renderer.set_avatar_visible(False)
        else:
            if self._nav_publish_before_robot_mode is not None:
                self.nav.state.nav_publishing_enabled = self._nav_publish_before_robot_mode
            self._nav_publish_before_robot_mode = None
            self.renderer.set_avatar_visible(True)
        if not self._move_robot:
            self._publish_cmd_vel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def toggle_move_mode(self) -> None:
        """Invert the current move target."""
        self.set_move_mode(not self._move_robot)

    def is_robot_mode(self) -> bool:
        """Return True when controlling the robot (cmd_vel)."""
        return self._move_robot

    def set_nav_publish_preference(self, enabled: bool) -> None:
        """Apply nav publishing preference, deferring while in robot mode."""
        enabled = bool(enabled)
        if self._move_robot:
            self._nav_publish_before_robot_mode = enabled
            self.nav.state.nav_publishing_enabled = False
        else:
            self.nav.state.nav_publishing_enabled = enabled

    def is_any_input_active(self) -> bool:
        """Return True when any movement/orientation key is pressed."""
        mw = self.base.mouseWatcherNode
        if not mw:
            return self._is_gamepad_active()
        buttons = (
            FORWARD_BUTTON,
            BACKWARD_BUTTON,
            LEFT_BUTTON,
            RIGHT_BUTTON,
            UP_BUTTON,
            UP_BUTTON_ALT,
            DOWN_BUTTON,
    DOWN_BUTTON_ALT,
            YAW_LEFT_BUTTON,
            YAW_RIGHT_BUTTON,
            PITCH_UP_BUTTON,
            PITCH_DOWN_BUTTON,
            ROLL_LEFT_BUTTON,
            ROLL_RIGHT_BUTTON,
            RESET_ORIENT_BUTTON,
            RESET_TO_ROBOT_ORIENT_BUTTON,
        )
        if any(mw.is_button_down(btn) for btn in buttons):
            return True
        return self._is_gamepad_active()

    def poll(self) -> None:
        """Process keyboard/gamepad state each frame."""
        dt = ClockObject.getGlobalClock().getDt()
        mw = self.base.mouseWatcherNode

        def _mw_down(btn: Any) -> bool:
            return bool(mw) and mw.is_button_down(btn)

        def _mw_pressed(btn: Any) -> bool:
            pressed = bool(mw) and mw.is_button_down(btn)
            key = f"key:{btn}"
            prev = self._button_prev.get(key, False)
            self._button_prev[key] = pressed
            return pressed and not prev

        gp = self._get_gamepad()

        if not self._move_robot:
            move = Vec3(0, 0, 0)
            if _mw_down(FORWARD_BUTTON):
                move.y += MOVE_SPEED * dt
            if _mw_down(BACKWARD_BUTTON):
                move.y -= MOVE_SPEED * dt
            if _mw_down(LEFT_BUTTON):
                move.x -= MOVE_SPEED * dt
            if _mw_down(RIGHT_BUTTON):
                move.x += MOVE_SPEED * dt
            if _mw_down(UP_BUTTON) or _mw_down(UP_BUTTON_ALT):
                move.z += MOVE_SPEED * dt
            if _mw_down(DOWN_BUTTON) or _mw_down(DOWN_BUTTON_ALT):
                move.z -= MOVE_SPEED * dt
            if gp is not None:
                lx, ly = self._move_stick(gp)
                lt, rt = self._triggers(gp)
                move.x += lx * MOVE_SPEED * dt
                move.y += (-ly) * MOVE_SPEED * dt
                move.z += (rt - lt) * MOVE_SPEED * dt
            if move.length_squared() > 0:
                frame = self.base.camera if self.base.camera is not None else self.base.render
                q = frame.getQuat(self.base.render)
                delta = q.xform(move)
                self.renderer.move_avatar(delta.x, delta.y, delta.z)

            dh = dp = dr = 0.0
            step = ROTATE_SPEED * 60.0 * dt
            if _mw_down(YAW_LEFT_BUTTON):
                dh += step
            if _mw_down(YAW_RIGHT_BUTTON):
                dh -= step
            if _mw_down(PITCH_UP_BUTTON):
                dp += step
            if _mw_down(PITCH_DOWN_BUTTON):
                dp -= step
            if _mw_down(ROLL_LEFT_BUTTON):
                dr += step
            if _mw_down(ROLL_RIGHT_BUTTON):
                dr -= step
            if gp is not None:
                rx, ry = self._look_stick(gp)
                if self._button_down(gp, GAMEPAD_BUTTON_L1, GAMEPAD_BUTTON_INDEX_L1):
                    dh += step
                if self._button_down(gp, GAMEPAD_BUTTON_R1, GAMEPAD_BUTTON_INDEX_R1):
                    dh -= step
                dp += ry * step
                dr += rx * step
            if dh or dp or dr:
                self.renderer.add_avatar_hpr(dh, dp, dr)
            if (
                _mw_down(RESET_TO_ROBOT_ORIENT_BUTTON)
                and self.nav.state.last_robot_hpr is not None
            ):
                h, p, r = self.nav.state.last_robot_hpr
                self.renderer.avatar.set_hpr(h, p, r)
            if _mw_down(RESET_ORIENT_BUTTON):
                self.renderer.reset_avatar_to_camera_hpr()
            if gp is not None:
                self._handle_dpad_rotation(gp)
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_TRIANGLE, GAMEPAD_BUTTON_INDEX_TRIANGLE
                ):
                    self.renderer.reset_avatar_to_camera_hpr()
                if self._button_pressed(gp, GAMEPAD_BUTTON_L3, GAMEPAD_BUTTON_INDEX_L3):
                    self.renderer.add_avatar_hpr(0.0, 0.0, -90.0)
                if self._button_pressed(gp, GAMEPAD_BUTTON_R3, GAMEPAD_BUTTON_INDEX_R3):
                    self.renderer.add_avatar_hpr(0.0, 0.0, 90.0)
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_ABORT, GAMEPAD_BUTTON_INDEX_ABORT
                ):
                    if callable(self._on_abort):
                        self._on_abort()
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_TOUCHPAD, GAMEPAD_BUTTON_INDEX_TOUCHPAD
                ):
                    if callable(self._on_abort):
                        self._on_abort()
                if self._button_pressed(gp, GAMEPAD_BUTTON_X, GAMEPAD_BUTTON_INDEX_X):
                    if callable(self._on_toggle_mode):
                        self._on_toggle_mode()
        else:
            lin_x = lin_y = lin_z = 0.0
            ang_x = ang_y = ang_z = 0.0

            if _mw_down(FORWARD_BUTTON):
                lin_x = +MOVE_SPEED
            elif _mw_down(BACKWARD_BUTTON):
                lin_x = -MOVE_SPEED
            if _mw_down(LEFT_BUTTON):
                lin_y = +MOVE_SPEED
            elif _mw_down(RIGHT_BUTTON):
                lin_y = -MOVE_SPEED
            if _mw_down(UP_BUTTON) or _mw_down(UP_BUTTON_ALT):
                lin_z = +MOVE_SPEED
            if _mw_down(DOWN_BUTTON) or _mw_down(DOWN_BUTTON_ALT):
                lin_z = -MOVE_SPEED
            if gp is not None:
                lx, ly = self._move_stick(gp)
                lt, rt = self._triggers(gp)
                gx = (-ly) * MOVE_SPEED
                gy = lx * MOVE_SPEED
                gz = (rt - lt) * MOVE_SPEED
                if gx != 0.0:
                    lin_x = gx
                if gy != 0.0:
                    lin_y = gy
                if gz != 0.0:
                    lin_z = gz

            if _mw_down(ROLL_LEFT_BUTTON):
                ang_x = +ROTATE_SPEED
            elif _mw_down(ROLL_RIGHT_BUTTON):
                ang_x = -ROTATE_SPEED
            if _mw_down(PITCH_UP_BUTTON):
                ang_y = +ROTATE_SPEED
            elif _mw_down(PITCH_DOWN_BUTTON):
                ang_y = -ROTATE_SPEED
            if _mw_down(YAW_LEFT_BUTTON):
                ang_z = +ROTATE_SPEED
            elif _mw_down(YAW_RIGHT_BUTTON):
                ang_z = -ROTATE_SPEED
            if gp is not None:
                rx, ry = self._look_stick(gp)
                if rx != 0.0:
                    ang_x = rx * ROTATE_SPEED
                if ry != 0.0:
                    ang_y = ry * ROTATE_SPEED
                if self._button_down(gp, GAMEPAD_BUTTON_L1, GAMEPAD_BUTTON_INDEX_L1):
                    ang_z = +ROTATE_SPEED
                elif self._button_down(gp, GAMEPAD_BUTTON_R1, GAMEPAD_BUTTON_INDEX_R1):
                    ang_z = -ROTATE_SPEED
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_TRIANGLE, GAMEPAD_BUTTON_INDEX_TRIANGLE
                ):
                    self.renderer.reset_avatar_to_camera_hpr()
                if self._button_pressed(gp, GAMEPAD_BUTTON_L3, GAMEPAD_BUTTON_INDEX_L3):
                    self.renderer.add_avatar_hpr(0.0, 0.0, -90.0)
                if self._button_pressed(gp, GAMEPAD_BUTTON_R3, GAMEPAD_BUTTON_INDEX_R3):
                    self.renderer.add_avatar_hpr(0.0, 0.0, 90.0)
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_ABORT, GAMEPAD_BUTTON_INDEX_ABORT
                ):
                    if callable(self._on_abort):
                        self._on_abort()
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_TOUCHPAD, GAMEPAD_BUTTON_INDEX_TOUCHPAD
                ):
                    if callable(self._on_abort):
                        self._on_abort()
                if self._button_pressed(gp, GAMEPAD_BUTTON_X, GAMEPAD_BUTTON_INDEX_X):
                    if callable(self._on_toggle_mode):
                        self._on_toggle_mode()

            self._publish_cmd_vel(lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)

    def _publish_cmd_vel(
        self,
        lin_x: float,
        lin_y: float,
        lin_z: float,
        ang_x: float,
        ang_y: float,
        ang_z: float,
    ) -> None:
        """Publish a geometry_msgs/Twist-style command."""
        data = {
            "linear": {"x": float(lin_x), "y": float(lin_y), "z": float(lin_z)},
            "angular": {"x": float(ang_x), "y": float(ang_y), "z": float(ang_z)},
        }
        try:
            self.cmd_pub.publish(TOPIC_CMD_VEL, data)
        except Exception:
            pass

    def _get_gamepad(self) -> Any:
        if not GAMEPAD_ENABLED or not self._remote_enabled:
            return None
        now = ClockObject.getGlobalClock().getFrameTime()
        if self._remote_sub is None or self._remote_gamepad is None:
            return None
        self._remote_sub.poll(5)
        payload = self._remote_sub.get(GAMEPAD_REMOTE_TOPIC)
        if isinstance(payload, dict):
            self._remote_gamepad.update(payload, now)
            if os.getenv("GAMEPAD_DEBUG") == "1" and not self._gamepad_logged:
                self._log_gamepad(self._remote_gamepad)
                self._gamepad_logged = True
        if self._remote_gamepad.last_update > 0.0:
            return self._remote_gamepad
        return None

    def _move_stick(self, gp: _RemoteGamepad) -> tuple[float, float]:
        lx = self._axis_value(gp, GAMEPAD_AXIS_LEFT_X, GAMEPAD_AXIS_INDEX_LEFT_X)
        ly = self._axis_value(gp, GAMEPAD_AXIS_LEFT_Y, GAMEPAD_AXIS_INDEX_LEFT_Y)
        lx = self._move_filter_x.apply(lx) * GAMEPAD_MOVE_SCALE
        ly = self._move_filter_y.apply(ly) * GAMEPAD_MOVE_SCALE
        lx, ly = self._axis_lock(lx, ly, GAMEPAD_AXIS_LOCK_RATIO_LEFT)
        lx, ly = self._normalize_pair(lx, ly)
        return lx, ly

    def _look_stick(self, gp: _RemoteGamepad) -> tuple[float, float]:
        rx = self._axis_value(gp, GAMEPAD_AXIS_RIGHT_X, GAMEPAD_AXIS_INDEX_RIGHT_X)
        ry = self._axis_value(gp, GAMEPAD_AXIS_RIGHT_Y, GAMEPAD_AXIS_INDEX_RIGHT_Y)
        rx = self._look_filter_x.apply(rx) * GAMEPAD_LOOK_SCALE
        ry = self._look_filter_y.apply(ry) * GAMEPAD_LOOK_SCALE
        rx, ry = self._axis_lock(rx, ry, GAMEPAD_AXIS_LOCK_RATIO_RIGHT)
        rx, ry = self._normalize_pair(rx, ry)
        if GAMEPAD_INVERT_ROLL:
            rx = -rx
        if GAMEPAD_INVERT_PITCH:
            ry = -ry
        return rx, ry

    def _triggers(self, gp: _RemoteGamepad) -> tuple[float, float]:
        lt = self._trigger_value(gp, GAMEPAD_AXIS_L2, GAMEPAD_AXIS_INDEX_L2)
        rt = self._trigger_value(gp, GAMEPAD_AXIS_R2, GAMEPAD_AXIS_INDEX_R2)
        return self._trigger_filter_l.apply(lt), self._trigger_filter_r.apply(rt)

    def _axis_value(
        self, device: _RemoteGamepad, axis_names: tuple[str, ...], axis_index: Optional[int]
    ) -> float:
        return device.axis_value(axis_names, axis_index)

    def _trigger_value(
        self, device: _RemoteGamepad, axis_names: tuple[str, ...], axis_index: Optional[int]
    ) -> float:
        value = device.axis_value(axis_names, axis_index)
        value = max(0.0, min(1.0, value))
        return value

    def _button_down(
        self, device: _RemoteGamepad, button_names: tuple[str, ...], button_index: Optional[int]
    ) -> bool:
        pressed = device.button_value(button_names, button_index)
        if pressed:
            self._log_button_press(device)
        return pressed

    def _button_pressed(
        self, device: _RemoteGamepad, button_names: tuple[str, ...], button_index: Optional[int]
    ) -> bool:
        key = f"edge:{button_index}:{button_names}"
        pressed = device.button_value(button_names, button_index)
        prev = self._button_prev.get(key, False)
        self._button_prev[key] = pressed
        if pressed and not prev:
            self._log_button_press(device)
        return pressed and not prev

    def _axis_lock(self, x: float, y: float, ratio: float) -> tuple[float, float]:
        ratio = float(ratio)
        ax = abs(x)
        ay = abs(y)
        if ax == 0.0 and ay == 0.0:
            return x, y
        if ax >= ay * (1.0 + ratio):
            return x, 0.0
        if ay >= ax * (1.0 + ratio):
            return 0.0, y
        return x, y

    def _normalize_pair(self, x: float, y: float) -> tuple[float, float]:
        mag = math.hypot(x, y)
        if mag > 1.0 and mag > 0.0:
            return x / mag, y / mag
        return x, y

    def _log_gamepad(self, device: _RemoteGamepad) -> None:
        print(f"[gamepad] device: {device.name}")
        if device.axes:
            print("[gamepad] axes:")
            for name in sorted(device.axes.keys()):
                print(f"  - {name}")
        if device.buttons:
            print("[gamepad] buttons:")
            for name in sorted(device.buttons.keys()):
                print(f"  - {name}")

    def _log_button_press(self, device: _RemoteGamepad) -> None:
        if os.getenv("GAMEPAD_DEBUG") != "2":
            return
        print("[gamepad] button pressed")

    def _handle_dpad_rotation(self, gp: _RemoteGamepad) -> None:
        x = self._axis_value(gp, GAMEPAD_AXIS_DPAD_X, GAMEPAD_AXIS_INDEX_DPAD_X)
        y = self._axis_value(gp, GAMEPAD_AXIS_DPAD_Y, GAMEPAD_AXIS_INDEX_DPAD_Y)
        thr = float(GAMEPAD_DPAD_THRESHOLD)
        prev_x = self._dpad_prev["x"]
        prev_y = self._dpad_prev["y"]

        if x <= -thr and prev_x > -thr:
            self.renderer.add_avatar_hpr(90.0, 0.0, 0.0)
        elif x >= thr and prev_x < thr:
            self.renderer.add_avatar_hpr(-90.0, 0.0, 0.0)
        if y <= -thr and prev_y > -thr:
            self.renderer.add_avatar_hpr(0.0, 90.0, 0.0)
        elif y >= thr and prev_y < thr:
            self.renderer.add_avatar_hpr(0.0, -90.0, 0.0)

        self._dpad_prev["x"] = x
        self._dpad_prev["y"] = y

    def _is_gamepad_active(self) -> bool:
        gp = self._get_gamepad()
        if gp is None:
            return False
        lx = gp.axis_value(GAMEPAD_AXIS_LEFT_X, GAMEPAD_AXIS_INDEX_LEFT_X)
        ly = gp.axis_value(GAMEPAD_AXIS_LEFT_Y, GAMEPAD_AXIS_INDEX_LEFT_Y)
        rx = gp.axis_value(GAMEPAD_AXIS_RIGHT_X, GAMEPAD_AXIS_INDEX_RIGHT_X)
        ry = gp.axis_value(GAMEPAD_AXIS_RIGHT_Y, GAMEPAD_AXIS_INDEX_RIGHT_Y)
        if abs(lx) > GAMEPAD_MOVE_DEADZONE or abs(ly) > GAMEPAD_MOVE_DEADZONE:
            return True
        if abs(rx) > GAMEPAD_LOOK_DEADZONE or abs(ry) > GAMEPAD_LOOK_DEADZONE:
            return True
        if gp.axis_value(GAMEPAD_AXIS_L2, GAMEPAD_AXIS_INDEX_L2) > GAMEPAD_TRIGGER_DEADZONE:
            return True
        if gp.axis_value(GAMEPAD_AXIS_R2, GAMEPAD_AXIS_INDEX_R2) > GAMEPAD_TRIGGER_DEADZONE:
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_L1, GAMEPAD_BUTTON_INDEX_L1):
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_R1, GAMEPAD_BUTTON_INDEX_R1):
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_L3, GAMEPAD_BUTTON_INDEX_L3):
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_R3, GAMEPAD_BUTTON_INDEX_R3):
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_ABORT, GAMEPAD_BUTTON_INDEX_ABORT):
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_X, GAMEPAD_BUTTON_INDEX_X):
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_TRIANGLE, GAMEPAD_BUTTON_INDEX_TRIANGLE):
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_TOUCHPAD, GAMEPAD_BUTTON_INDEX_TOUCHPAD):
            return True
        return False
