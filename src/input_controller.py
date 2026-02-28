"""Keyboard polling for avatar movement or robot cmd_vel teleop."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

from panda3d.core import ClockObject, Quat, Vec3

from config import (
    BACKWARD_BUTTON,
    DOWN_BUTTON,
    DOWN_BUTTON_ALT,
    FORWARD_BUTTON,
    GAMEPAD_AUTOSCALE_DECAY,
    GAMEPAD_AUTOSCALE_MAX_GAIN,
    GAMEPAD_AUTOSCALE_MIN,
    GAMEPAD_AXIS_DPAD_X,
    GAMEPAD_AXIS_DPAD_Y,
    GAMEPAD_AXIS_INDEX_DPAD_X,
    GAMEPAD_AXIS_INDEX_DPAD_Y,
    GAMEPAD_AXIS_INDEX_L2,
    GAMEPAD_AXIS_INDEX_LEFT_X,
    GAMEPAD_AXIS_INDEX_LEFT_Y,
    GAMEPAD_AXIS_INDEX_R2,
    GAMEPAD_AXIS_INDEX_RIGHT_X,
    GAMEPAD_AXIS_INDEX_RIGHT_Y,
    GAMEPAD_AXIS_L2,
    GAMEPAD_AXIS_LEFT_X,
    GAMEPAD_AXIS_LEFT_Y,
    GAMEPAD_AXIS_LOCK_RATIO_LEFT,
    GAMEPAD_AXIS_LOCK_RATIO_RIGHT,
    GAMEPAD_AXIS_R2,
    GAMEPAD_AXIS_RIGHT_X,
    GAMEPAD_AXIS_RIGHT_Y,
    GAMEPAD_BUTTON_ABORT,
    GAMEPAD_BUTTON_INDEX_ABORT,
    GAMEPAD_BUTTON_INDEX_L1,
    GAMEPAD_BUTTON_INDEX_L3,
    GAMEPAD_BUTTON_INDEX_R1,
    GAMEPAD_BUTTON_INDEX_R3,
    GAMEPAD_BUTTON_INDEX_TOUCHPAD,
    GAMEPAD_BUTTON_INDEX_TRIANGLE,
    GAMEPAD_BUTTON_INDEX_X,
    GAMEPAD_BUTTON_L1,
    GAMEPAD_BUTTON_L3,
    GAMEPAD_BUTTON_R1,
    GAMEPAD_BUTTON_R3,
    GAMEPAD_BUTTON_TOUCHPAD,
    GAMEPAD_BUTTON_TRIANGLE,
    GAMEPAD_BUTTON_X,
    GAMEPAD_DPAD_THRESHOLD,
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
    GAMEPAD_REMOTE_AUTOSTART,
    GAMEPAD_REMOTE_ENABLED,
    GAMEPAD_REMOTE_ENDPOINT,
    GAMEPAD_REMOTE_TIMEOUT_S,
    GAMEPAD_REMOTE_TOPIC,
    GAMEPAD_TRIGGER_DEADZONE,
    GAMEPAD_TRIGGER_SMOOTHING,
    LEFT_BUTTON,
    MOVE_SPEED,
    PITCH_DOWN_BUTTON,
    PITCH_UP_BUTTON,
    RESET_ORIENT_BUTTON,
    RESET_TO_ROBOT_ORIENT_BUTTON,
    RIGHT_BUTTON,
    ROLL_LEFT_BUTTON,
    ROLL_RIGHT_BUTTON,
    ROTATE_SPEED,
    SPACEMOUSE_AXIS_CLIP_MIN,
    SPACEMOUSE_AXIS_CLIP_RATIO,
    SPACEMOUSE_AXIS_PITCH,
    SPACEMOUSE_AXIS_ROLL,
    SPACEMOUSE_AXIS_X,
    SPACEMOUSE_AXIS_Y,
    SPACEMOUSE_AXIS_YAW,
    SPACEMOUSE_AXIS_Z,
    SPACEMOUSE_BUTTON_ABORT,
    SPACEMOUSE_BUTTON_INDEX_ABORT,
    SPACEMOUSE_BUTTON_INDEX_MODE,
    SPACEMOUSE_BUTTON_MODE,
    SPACEMOUSE_CURVE_ROTATION,
    SPACEMOUSE_CURVE_TRANSLATION,
    SPACEMOUSE_DEADZONE,
    SPACEMOUSE_ENABLED,
    SPACEMOUSE_INVERT_PITCH,
    SPACEMOUSE_INVERT_ROLL,
    SPACEMOUSE_INVERT_Y,
    SPACEMOUSE_INVERT_Z,
    SPACEMOUSE_MIXED_CLIP_MIN,
    SPACEMOUSE_MIXED_CLIP_RATIO,
    SPACEMOUSE_REMOTE_AUTOSTART,
    SPACEMOUSE_REMOTE_ENABLED,
    SPACEMOUSE_REMOTE_ENDPOINT,
    SPACEMOUSE_REMOTE_TIMEOUT_S,
    SPACEMOUSE_REMOTE_TOPIC,
    SPACEMOUSE_ROTATION_SCALE,
    SPACEMOUSE_SMOOTHING,
    SPACEMOUSE_TRANSLATION_SCALE,
    TOPIC_CMD_VEL,
    UP_BUTTON,
    UP_BUTTON_ALT,
    YAW_LEFT_BUTTON,
    YAW_RIGHT_BUTTON,
)
from teleop_bus import TeleopBusSub


class _RemoteInputDevice:
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

    def axis_value(
        self, axis_names: tuple[str, ...], axis_index: Optional[int]
    ) -> float:
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
        self._spacemouse_logged = False

        self._gamepad_sub: Optional[TeleopBusSub] = None
        self._remote_gamepad: Optional[_RemoteInputDevice] = None
        self._gamepad_enabled = bool(
            GAMEPAD_REMOTE_ENABLED
            or GAMEPAD_REMOTE_AUTOSTART
            or os.getenv("GAMEPAD_REMOTE") == "1"
        )
        if self._gamepad_enabled:
            self._gamepad_sub = TeleopBusSub(GAMEPAD_REMOTE_ENDPOINT)
            self._remote_gamepad = _RemoteInputDevice()
        self._gamepad_timeout_s: float = max(0.05, float(GAMEPAD_REMOTE_TIMEOUT_S))

        self._spacemouse_sub: Optional[TeleopBusSub] = None
        self._remote_spacemouse: Optional[_RemoteInputDevice] = None
        self._spacemouse_enabled = bool(
            SPACEMOUSE_ENABLED
            and (
                SPACEMOUSE_REMOTE_ENABLED
                or SPACEMOUSE_REMOTE_AUTOSTART
                or os.getenv("SPACEMOUSE_REMOTE") == "1"
            )
        )
        if self._spacemouse_enabled:
            self._spacemouse_sub = TeleopBusSub(SPACEMOUSE_REMOTE_ENDPOINT)
            self._remote_spacemouse = _RemoteInputDevice()
        self._spacemouse_timeout_s: float = max(
            0.05, float(SPACEMOUSE_REMOTE_TIMEOUT_S)
        )

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
        self._sm_filter_x = _AxisFilter(
            SPACEMOUSE_DEADZONE,
            SPACEMOUSE_CURVE_TRANSLATION,
            SPACEMOUSE_SMOOTHING,
            0.0,
            1.0,
            1.0,
        )
        self._sm_filter_y = _AxisFilter(
            SPACEMOUSE_DEADZONE,
            SPACEMOUSE_CURVE_TRANSLATION,
            SPACEMOUSE_SMOOTHING,
            0.0,
            1.0,
            1.0,
        )
        self._sm_filter_z = _AxisFilter(
            SPACEMOUSE_DEADZONE,
            SPACEMOUSE_CURVE_TRANSLATION,
            SPACEMOUSE_SMOOTHING,
            0.0,
            1.0,
            1.0,
        )
        self._sm_filter_roll = _AxisFilter(
            SPACEMOUSE_DEADZONE,
            SPACEMOUSE_CURVE_ROTATION,
            SPACEMOUSE_SMOOTHING,
            0.0,
            1.0,
            1.0,
        )
        self._sm_filter_pitch = _AxisFilter(
            SPACEMOUSE_DEADZONE,
            SPACEMOUSE_CURVE_ROTATION,
            SPACEMOUSE_SMOOTHING,
            0.0,
            1.0,
            1.0,
        )
        self._sm_filter_yaw = _AxisFilter(
            SPACEMOUSE_DEADZONE,
            SPACEMOUSE_CURVE_ROTATION,
            SPACEMOUSE_SMOOTHING,
            0.0,
            1.0,
            1.0,
        )
        self._spacemouse_cached_frame_s: float = -1.0
        self._spacemouse_cached_device: Optional[_RemoteInputDevice] = None
        self._last_input_6dof: dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }

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
                self.nav.state.nav_publishing_enabled = (
                    self._nav_publish_before_robot_mode
                )
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

    def close(self) -> None:
        if self._gamepad_sub is not None:
            try:
                self._gamepad_sub.close()
            except Exception:
                pass
            self._gamepad_sub = None
        if self._spacemouse_sub is not None:
            try:
                self._spacemouse_sub.close()
            except Exception:
                pass
            self._spacemouse_sub = None

    def get_input_6dof(self) -> dict[str, float]:
        return dict(self._last_input_6dof)

    def is_any_input_active(self) -> bool:
        """Return True when any movement/orientation key is pressed."""
        mw = self.base.mouseWatcherNode
        if not mw:
            return self._is_gamepad_active() or self._is_spacemouse_active()
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
        return self._is_gamepad_active() or self._is_spacemouse_active()

    def poll(self) -> None:
        """Process keyboard/gamepad/SpaceMouse state each frame."""
        dt = ClockObject.getGlobalClock().getDt()
        mw = self.base.mouseWatcherNode

        def _mw_down(btn: Any) -> bool:
            return bool(mw) and mw.is_button_down(btn)

        gp = self._get_gamepad()
        sm = self._get_spacemouse()

        if not self._move_robot:
            sm_axes = self._spacemouse_axes(sm) if sm is not None else None
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
            if sm_axes is not None:
                sx, sy, sz, _roll, _pitch, _yaw = sm_axes
                move.x += sx * MOVE_SPEED * SPACEMOUSE_TRANSLATION_SCALE * dt
                move.y += sy * MOVE_SPEED * SPACEMOUSE_TRANSLATION_SCALE * dt
                move.z += sz * MOVE_SPEED * SPACEMOUSE_TRANSLATION_SCALE * dt
            if move.length_squared() > 0:
                frame = (
                    self.base.camera
                    if self.base.camera is not None
                    else self.base.render
                )
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
            if sm_axes is not None:
                _sx, _sy, _sz, sroll, spitch, syaw = sm_axes
                dh += syaw * step * SPACEMOUSE_ROTATION_SCALE
                dp += spitch * step * SPACEMOUSE_ROTATION_SCALE
                dr += sroll * step * SPACEMOUSE_ROTATION_SCALE
            if dh or dp or dr:
                self._add_avatar_hpr_camera_frame(dh, dp, dr)
            rot_den = max(1e-6, step)
            trans_den = max(1e-6, MOVE_SPEED * max(1e-6, dt))
            self._set_input_6dof(
                move.x / trans_den,
                move.y / trans_den,
                move.z / trans_den,
                dr / rot_den,
                dp / rot_den,
                dh / rot_den,
            )
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
                    self._add_avatar_hpr_camera_frame(0.0, 0.0, -90.0)
                if self._button_pressed(gp, GAMEPAD_BUTTON_R3, GAMEPAD_BUTTON_INDEX_R3):
                    self._add_avatar_hpr_camera_frame(0.0, 0.0, 90.0)
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_ABORT, GAMEPAD_BUTTON_INDEX_ABORT
                ) or self._button_pressed(
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
            if sm is not None:
                sx, sy, sz, sroll, spitch, syaw = self._spacemouse_axes(sm)
                sm_lin_x = sx * MOVE_SPEED * SPACEMOUSE_TRANSLATION_SCALE
                sm_lin_y = sy * MOVE_SPEED * SPACEMOUSE_TRANSLATION_SCALE
                sm_lin_z = sz * MOVE_SPEED * SPACEMOUSE_TRANSLATION_SCALE
                if sm_lin_x != 0.0:
                    lin_x = sm_lin_x
                if sm_lin_y != 0.0:
                    lin_y = sm_lin_y
                if sm_lin_z != 0.0:
                    lin_z = sm_lin_z
                sm_ang_x = sroll * ROTATE_SPEED * SPACEMOUSE_ROTATION_SCALE
                sm_ang_y = spitch * ROTATE_SPEED * SPACEMOUSE_ROTATION_SCALE
                sm_ang_z = syaw * ROTATE_SPEED * SPACEMOUSE_ROTATION_SCALE
                if sm_ang_x != 0.0:
                    ang_x = sm_ang_x
                if sm_ang_y != 0.0:
                    ang_y = sm_ang_y
                if sm_ang_z != 0.0:
                    ang_z = sm_ang_z

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
                ) or self._button_pressed(
                    gp, GAMEPAD_BUTTON_TOUCHPAD, GAMEPAD_BUTTON_INDEX_TOUCHPAD
                ):
                    if callable(self._on_abort):
                        self._on_abort()
                if self._button_pressed(gp, GAMEPAD_BUTTON_X, GAMEPAD_BUTTON_INDEX_X):
                    if callable(self._on_toggle_mode):
                        self._on_toggle_mode()

            self._publish_cmd_vel(lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)
            self._set_input_6dof(
                lin_x / max(1e-6, MOVE_SPEED),
                lin_y / max(1e-6, MOVE_SPEED),
                lin_z / max(1e-6, MOVE_SPEED),
                ang_x / max(1e-6, ROTATE_SPEED),
                ang_y / max(1e-6, ROTATE_SPEED),
                ang_z / max(1e-6, ROTATE_SPEED),
            )

        if sm is not None:
            self._handle_spacemouse_buttons(sm)

    def _set_input_6dof(
        self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float
    ) -> None:
        self._last_input_6dof["x"] = max(-1.0, min(1.0, float(x)))
        self._last_input_6dof["y"] = max(-1.0, min(1.0, float(y)))
        self._last_input_6dof["z"] = max(-1.0, min(1.0, float(z)))
        self._last_input_6dof["roll"] = max(-1.0, min(1.0, float(roll)))
        self._last_input_6dof["pitch"] = max(-1.0, min(1.0, float(pitch)))
        self._last_input_6dof["yaw"] = max(-1.0, min(1.0, float(yaw)))

    def _add_avatar_hpr_camera_frame(self, dh: float, dp: float, dr: float) -> None:
        """Apply avatar incremental rotation in camera frame (not avatar-local frame)."""
        avatar = self.renderer.avatar
        cam_np = self.base.camera if self.base.camera is not None else self.base.render
        cam_q = cam_np.getQuat(self.base.render)
        avatar_q = avatar.get_quat()

        # Control mapping tuned to match in-app ROS control convention:
        # yaw inverted; pitch/roll swapped.
        delta_cam = Quat()
        delta_cam.setHpr((float(-dh), float(-dr), float(-dp)))
        cam_inv = Quat(cam_q)
        cam_inv.invertInPlace()
        world_delta = cam_q * delta_cam * cam_inv
        avatar.set_quat(world_delta * avatar_q)

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
        if not GAMEPAD_ENABLED or not self._gamepad_enabled:
            return None
        now = ClockObject.getGlobalClock().getFrameTime()
        if self._gamepad_sub is None or self._remote_gamepad is None:
            return None
        self._gamepad_sub.poll(5)
        payload = self._gamepad_sub.get(GAMEPAD_REMOTE_TOPIC)
        if isinstance(payload, dict):
            self._remote_gamepad.update(payload, now)
            if os.getenv("GAMEPAD_DEBUG") == "1" and not self._gamepad_logged:
                self._log_device(self._remote_gamepad)
                self._gamepad_logged = True
        if (now - self._remote_gamepad.last_update) <= self._gamepad_timeout_s:
            return self._remote_gamepad
        return None

    def _get_spacemouse(self) -> Optional[_RemoteInputDevice]:
        if not self._spacemouse_enabled:
            return None
        now = ClockObject.getGlobalClock().getFrameTime()
        if now == self._spacemouse_cached_frame_s:
            return self._spacemouse_cached_device
        self._spacemouse_cached_frame_s = now
        self._spacemouse_cached_device = None
        if self._spacemouse_sub is None or self._remote_spacemouse is None:
            return None
        self._spacemouse_sub.poll(5)
        payload = self._spacemouse_sub.get(SPACEMOUSE_REMOTE_TOPIC)
        if isinstance(payload, dict):
            self._remote_spacemouse.update(payload, now)
            if os.getenv("SPACEMOUSE_DEBUG") == "1" and not self._spacemouse_logged:
                self._log_device(self._remote_spacemouse)
                self._spacemouse_logged = True
        if (now - self._remote_spacemouse.last_update) <= self._spacemouse_timeout_s:
            self._spacemouse_cached_device = self._remote_spacemouse
            return self._remote_spacemouse
        return None

    def _move_stick(self, gp: _RemoteInputDevice) -> tuple[float, float]:
        lx = self._axis_value(gp, GAMEPAD_AXIS_LEFT_X, GAMEPAD_AXIS_INDEX_LEFT_X)
        ly = self._axis_value(gp, GAMEPAD_AXIS_LEFT_Y, GAMEPAD_AXIS_INDEX_LEFT_Y)
        lx = self._move_filter_x.apply(lx) * GAMEPAD_MOVE_SCALE
        ly = self._move_filter_y.apply(ly) * GAMEPAD_MOVE_SCALE
        lx, ly = self._axis_lock(lx, ly, GAMEPAD_AXIS_LOCK_RATIO_LEFT)
        lx, ly = self._normalize_pair(lx, ly)
        return lx, ly

    def _look_stick(self, gp: _RemoteInputDevice) -> tuple[float, float]:
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

    def _triggers(self, gp: _RemoteInputDevice) -> tuple[float, float]:
        lt = self._trigger_value(gp, GAMEPAD_AXIS_L2, GAMEPAD_AXIS_INDEX_L2)
        rt = self._trigger_value(gp, GAMEPAD_AXIS_R2, GAMEPAD_AXIS_INDEX_R2)
        return self._trigger_filter_l.apply(lt), self._trigger_filter_r.apply(rt)

    def _spacemouse_axes(
        self, sm: _RemoteInputDevice
    ) -> tuple[float, float, float, float, float, float]:
        x = self._sm_filter_x.apply(self._axis_value(sm, SPACEMOUSE_AXIS_X, None))
        y = self._sm_filter_y.apply(self._axis_value(sm, SPACEMOUSE_AXIS_Y, None))
        z = self._sm_filter_z.apply(self._axis_value(sm, SPACEMOUSE_AXIS_Z, None))
        roll = self._sm_filter_roll.apply(
            self._axis_value(sm, SPACEMOUSE_AXIS_ROLL, None)
        )
        pitch = self._sm_filter_pitch.apply(
            self._axis_value(sm, SPACEMOUSE_AXIS_PITCH, None)
        )
        yaw = self._sm_filter_yaw.apply(self._axis_value(sm, SPACEMOUSE_AXIS_YAW, None))
        if SPACEMOUSE_INVERT_Y:
            y = -y
        if SPACEMOUSE_INVERT_Z:
            z = -z
        if SPACEMOUSE_INVERT_ROLL:
            roll = -roll
        if SPACEMOUSE_INVERT_PITCH:
            pitch = -pitch
        x, y, z = self._clip_dominant_group((x, y, z))
        roll, pitch, yaw = self._clip_dominant_group((roll, pitch, yaw))

        t_max = max(abs(x), abs(y), abs(z))
        r_max = max(abs(roll), abs(pitch), abs(yaw))
        if t_max >= float(SPACEMOUSE_MIXED_CLIP_MIN) and r_max <= t_max * float(
            SPACEMOUSE_MIXED_CLIP_RATIO
        ):
            roll = pitch = yaw = 0.0
        elif r_max >= float(SPACEMOUSE_MIXED_CLIP_MIN) and t_max <= r_max * float(
            SPACEMOUSE_MIXED_CLIP_RATIO
        ):
            x = y = z = 0.0
        return x, y, z, roll, pitch, yaw

    def _clip_dominant_group(
        self, values: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        ax = [abs(v) for v in values]
        dominant = max(ax)
        if dominant < float(SPACEMOUSE_AXIS_CLIP_MIN):
            return values
        out = list(values)
        ratio = float(SPACEMOUSE_AXIS_CLIP_RATIO)
        for i in range(3):
            if ax[i] < dominant * ratio:
                out[i] = 0.0
        return float(out[0]), float(out[1]), float(out[2])

    def _handle_spacemouse_buttons(self, sm: _RemoteInputDevice) -> None:
        if self._button_pressed(
            sm, SPACEMOUSE_BUTTON_ABORT, SPACEMOUSE_BUTTON_INDEX_ABORT
        ):
            if callable(self._on_abort):
                self._on_abort()
        if self._button_pressed(
            sm, SPACEMOUSE_BUTTON_MODE, SPACEMOUSE_BUTTON_INDEX_MODE
        ):
            if callable(self._on_toggle_mode):
                self._on_toggle_mode()

    def _axis_value(
        self,
        device: Any,
        axis_names: tuple[str, ...],
        axis_index: Optional[int],
    ) -> float:
        return device.axis_value(axis_names, axis_index)

    def _trigger_value(
        self,
        device: Any,
        axis_names: tuple[str, ...],
        axis_index: Optional[int],
    ) -> float:
        value = device.axis_value(axis_names, axis_index)
        value = max(0.0, min(1.0, value))
        return value

    def _button_down(
        self,
        device: Any,
        button_names: tuple[str, ...],
        button_index: Optional[int],
    ) -> bool:
        pressed = device.button_value(button_names, button_index)
        if pressed:
            self._log_button_press(device)
        return pressed

    def _button_pressed(
        self,
        device: Any,
        button_names: tuple[str, ...],
        button_index: Optional[int],
    ) -> bool:
        device_name = getattr(device, "name", "unknown")
        key = f"edge:{device_name}:{button_index}:{button_names}"
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

    def _log_device(self, device: _RemoteInputDevice) -> None:
        print(f"[input] device: {device.name}")
        if device.axes:
            print("[input] axes:")
            for name in sorted(device.axes.keys()):
                print(f"  - {name}")
        if device.buttons:
            print("[input] buttons:")
            for name in sorted(device.buttons.keys()):
                print(f"  - {name}")

    def _log_button_press(self, device: Any) -> None:
        if os.getenv("GAMEPAD_DEBUG") != "2":
            return
        print(f"[{getattr(device, 'name', 'input')}] button pressed")

    def _handle_dpad_rotation(self, gp: _RemoteInputDevice) -> None:
        x = self._axis_value(gp, GAMEPAD_AXIS_DPAD_X, GAMEPAD_AXIS_INDEX_DPAD_X)
        y = self._axis_value(gp, GAMEPAD_AXIS_DPAD_Y, GAMEPAD_AXIS_INDEX_DPAD_Y)
        thr = float(GAMEPAD_DPAD_THRESHOLD)
        prev_x = self._dpad_prev["x"]
        prev_y = self._dpad_prev["y"]

        if x <= -thr and prev_x > -thr:
            self._add_avatar_hpr_camera_frame(90.0, 0.0, 0.0)
        elif x >= thr and prev_x < thr:
            self._add_avatar_hpr_camera_frame(-90.0, 0.0, 0.0)
        if y <= -thr and prev_y > -thr:
            self._add_avatar_hpr_camera_frame(0.0, 90.0, 0.0)
        elif y >= thr and prev_y < thr:
            self._add_avatar_hpr_camera_frame(0.0, -90.0, 0.0)

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
        if (
            gp.axis_value(GAMEPAD_AXIS_L2, GAMEPAD_AXIS_INDEX_L2)
            > GAMEPAD_TRIGGER_DEADZONE
        ):
            return True
        if (
            gp.axis_value(GAMEPAD_AXIS_R2, GAMEPAD_AXIS_INDEX_R2)
            > GAMEPAD_TRIGGER_DEADZONE
        ):
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
        if self._button_down(
            gp, GAMEPAD_BUTTON_TRIANGLE, GAMEPAD_BUTTON_INDEX_TRIANGLE
        ):
            return True
        if self._button_down(
            gp, GAMEPAD_BUTTON_TOUCHPAD, GAMEPAD_BUTTON_INDEX_TOUCHPAD
        ):
            return True
        return False

    def _is_spacemouse_active(self) -> bool:
        sm = self._get_spacemouse()
        if sm is None:
            return False
        x = sm.axis_value(SPACEMOUSE_AXIS_X, None)
        y = sm.axis_value(SPACEMOUSE_AXIS_Y, None)
        z = sm.axis_value(SPACEMOUSE_AXIS_Z, None)
        roll = sm.axis_value(SPACEMOUSE_AXIS_ROLL, None)
        pitch = sm.axis_value(SPACEMOUSE_AXIS_PITCH, None)
        yaw = sm.axis_value(SPACEMOUSE_AXIS_YAW, None)
        deadzone = float(SPACEMOUSE_DEADZONE)
        if (
            abs(x) > deadzone
            or abs(y) > deadzone
            or abs(z) > deadzone
            or abs(roll) > deadzone
            or abs(pitch) > deadzone
            or abs(yaw) > deadzone
        ):
            return True
        if self._button_down(
            sm, SPACEMOUSE_BUTTON_ABORT, SPACEMOUSE_BUTTON_INDEX_ABORT
        ):
            return True
        if self._button_down(sm, SPACEMOUSE_BUTTON_MODE, SPACEMOUSE_BUTTON_INDEX_MODE):
            return True
        return False
