"""Keyboard polling for avatar movement or robot cmd_vel teleop."""

from __future__ import annotations

from panda3d.core import ClockObject, InputDevice, Vec3
import os
from typing import Any, Optional

from config import (
    BACKWARD_BUTTON,
    DOWN_BUTTON,
    DOWN_BUTTON_ALT,
    FORWARD_BUTTON,
    GAMEPAD_AXIS_L2,
    GAMEPAD_AXIS_LEFT_X,
    GAMEPAD_AXIS_LEFT_Y,
    GAMEPAD_AXIS_INDEX_L2,
    GAMEPAD_AXIS_INDEX_LEFT_X,
    GAMEPAD_AXIS_INDEX_LEFT_Y,
    GAMEPAD_AXIS_INDEX_R2,
    GAMEPAD_AXIS_INDEX_RIGHT_X,
    GAMEPAD_AXIS_INDEX_RIGHT_Y,
    GAMEPAD_AXIS_R2,
    GAMEPAD_AXIS_RIGHT_X,
    GAMEPAD_AXIS_RIGHT_Y,
    GAMEPAD_BUTTON_L1,
    GAMEPAD_BUTTON_R1,
    GAMEPAD_BUTTON_TOUCHPAD,
    GAMEPAD_BUTTON_TRIANGLE,
    GAMEPAD_BUTTON_X,
    GAMEPAD_BUTTON_INDEX_L1,
    GAMEPAD_BUTTON_INDEX_R1,
    GAMEPAD_BUTTON_INDEX_TOUCHPAD,
    GAMEPAD_BUTTON_INDEX_TRIANGLE,
    GAMEPAD_BUTTON_INDEX_X,
    GAMEPAD_DEADZONE,
    GAMEPAD_ENABLED,
    GAMEPAD_REMOTE_ENABLED,
    GAMEPAD_REMOTE_ENDPOINT,
    GAMEPAD_REMOTE_TIMEOUT_S,
    GAMEPAD_REMOTE_TOPIC,
    GAMEPAD_TRIGGER_DEADZONE,
    GAMEPAD_AXIS_LOCK_RATIO,
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
        self._gamepad = None
        self._gamepad_last_scan: Optional[float] = None
        self._gamepad_axes: dict[str, Any] = {}
        self._gamepad_buttons: dict[str, Any] = {}
        self._button_prev: dict[str, bool] = {}
        self._gamepad_logged = False
        self._axis_prev: dict[int, float] = {}
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
                lx = self._axis_value(gp, GAMEPAD_AXIS_LEFT_X, GAMEPAD_AXIS_INDEX_LEFT_X)
                ly = self._axis_value(gp, GAMEPAD_AXIS_LEFT_Y, GAMEPAD_AXIS_INDEX_LEFT_Y)
                lx, ly = self._axis_lock(lx, ly)
                lt = self._trigger_value(gp, GAMEPAD_AXIS_L2, GAMEPAD_AXIS_INDEX_L2)
                rt = self._trigger_value(gp, GAMEPAD_AXIS_R2, GAMEPAD_AXIS_INDEX_R2)
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
                rx = self._axis_value(gp, GAMEPAD_AXIS_RIGHT_X, GAMEPAD_AXIS_INDEX_RIGHT_X)
                ry = self._axis_value(gp, GAMEPAD_AXIS_RIGHT_Y, GAMEPAD_AXIS_INDEX_RIGHT_Y)
                rx, ry = self._axis_lock(rx, ry)
                if self._button_down(gp, GAMEPAD_BUTTON_L1, GAMEPAD_BUTTON_INDEX_L1):
                    dh += step
                if self._button_down(gp, GAMEPAD_BUTTON_R1, GAMEPAD_BUTTON_INDEX_R1):
                    dh -= step
                dp += (ry) * step
                dr += (rx) * step
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
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_TRIANGLE, GAMEPAD_BUTTON_INDEX_TRIANGLE
                ):
                    self.renderer.reset_avatar_to_camera_hpr()
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
                lx = self._axis_value(gp, GAMEPAD_AXIS_LEFT_X, GAMEPAD_AXIS_INDEX_LEFT_X)
                ly = self._axis_value(gp, GAMEPAD_AXIS_LEFT_Y, GAMEPAD_AXIS_INDEX_LEFT_Y)
                lx, ly = self._axis_lock(lx, ly)
                lt = self._trigger_value(gp, GAMEPAD_AXIS_L2, GAMEPAD_AXIS_INDEX_L2)
                rt = self._trigger_value(gp, GAMEPAD_AXIS_R2, GAMEPAD_AXIS_INDEX_R2)
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
                rx = self._axis_value(gp, GAMEPAD_AXIS_RIGHT_X, GAMEPAD_AXIS_INDEX_RIGHT_X)
                ry = self._axis_value(gp, GAMEPAD_AXIS_RIGHT_Y, GAMEPAD_AXIS_INDEX_RIGHT_Y)
                rx, ry = self._axis_lock(rx, ry)
                if rx != 0.0:
                    ang_x = (rx) * ROTATE_SPEED
                if ry != 0.0:
                    ang_y = (ry) * ROTATE_SPEED
                if self._button_down(gp, GAMEPAD_BUTTON_L1, GAMEPAD_BUTTON_INDEX_L1):
                    ang_z = +ROTATE_SPEED
                elif self._button_down(gp, GAMEPAD_BUTTON_R1, GAMEPAD_BUTTON_INDEX_R1):
                    ang_z = -ROTATE_SPEED
                if self._button_pressed(
                    gp, GAMEPAD_BUTTON_TRIANGLE, GAMEPAD_BUTTON_INDEX_TRIANGLE
                ):
                    self.renderer.reset_avatar_to_camera_hpr()
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
        if not GAMEPAD_ENABLED:
            return None
        now = ClockObject.getGlobalClock().getFrameTime()
        remote = self._poll_remote_gamepad(now)
        if self._remote_enabled:
            return remote
        if self._gamepad is not None:
            for name in ("is_connected", "isConnected"):
                fn = getattr(self._gamepad, name, None)
                if fn is None:
                    continue
                try:
                    if not fn():
                        self._gamepad = None
                    break
                except Exception:
                    self._gamepad = None
                    break
            if self._gamepad is not None:
                return self._gamepad
        if self._gamepad_last_scan is not None and (now - self._gamepad_last_scan) < 1.0:
            return None
        self._gamepad_last_scan = now
        dev_mgr = getattr(self.base, "devices", None)
        if dev_mgr is None:
            return None
        gamepads = self._get_devices(dev_mgr)
        if not gamepads:
            return None
        self._gamepad = gamepads[0]
        self._gamepad_axes = {}
        self._gamepad_buttons = {}
        self._button_prev = {}
        self._attach_gamepad(self._gamepad)
        if os.getenv("GAMEPAD_DEBUG") == "1" and not self._gamepad_logged:
            self._log_gamepad(self._gamepad)
            self._gamepad_logged = True
        return self._gamepad

    def _poll_remote_gamepad(self, now: float) -> Optional[_RemoteGamepad]:
        if not self._remote_enabled or self._remote_sub is None or self._remote_gamepad is None:
            return None
        self._remote_sub.poll(5)
        payload = self._remote_sub.get(GAMEPAD_REMOTE_TOPIC)
        if isinstance(payload, dict):
            self._remote_gamepad.update(payload, now)
            if os.getenv("GAMEPAD_DEBUG") == "1" and not self._gamepad_logged:
                self._log_gamepad(self._remote_gamepad)
                self._gamepad_logged = True
        if (now - self._remote_gamepad.last_update) <= float(GAMEPAD_REMOTE_TIMEOUT_S):
            return self._remote_gamepad
        return None

    def _get_devices(self, dev_mgr: Any) -> list[Any]:
        for name in ("getDevices", "get_devices"):
            fn = getattr(dev_mgr, name, None)
            if fn is None:
                continue
            try:
                return list(fn(InputDevice.DeviceClass.gamepad))
            except TypeError:
                try:
                    return list(fn())
                except Exception:
                    return []
        return []

    def _resolve_axis(
        self,
        device: Any,
        axis_names: tuple[str, ...],
        axis_index: Optional[int],
    ) -> Any:
        for axis_name in axis_names:
            key = f"axis:{axis_name}"
            if key in self._gamepad_axes:
                return self._gamepad_axes[key]
            axis_enum = getattr(InputDevice, "Axis", None)
            if axis_enum is not None:
                axis_id = getattr(axis_enum, axis_name, None)
                if axis_id is not None:
                    axis = self._find_axis(device, axis_id)
                    if axis is not None:
                        self._gamepad_axes[key] = axis
                        return axis
            axis = self._find_axis_by_name(device, axis_name)
            if axis is not None:
                self._gamepad_axes[key] = axis
                return axis
        if axis_index is not None:
            axis = self._find_axis_by_index(device, axis_index)
            if axis is not None:
                self._gamepad_axes[f"axis_index:{axis_index}"] = axis
                return axis
        return None

    def _resolve_button(
        self,
        device: Any,
        button_names: tuple[str, ...],
        button_index: Optional[int],
    ) -> Any:
        for button_name in button_names:
            key = f"btn:{button_name}"
            if key in self._gamepad_buttons:
                return self._gamepad_buttons[key]
            button_enum = getattr(InputDevice, "Button", None)
            if button_enum is not None:
                button_id = getattr(button_enum, button_name, None)
                if button_id is not None:
                    button = self._find_button(device, button_id)
                    if button is not None:
                        self._gamepad_buttons[key] = button
                        return button
            button = self._find_button_by_name(device, button_name)
            if button is not None:
                self._gamepad_buttons[key] = button
                return button
        if button_index is not None:
            button = self._find_button_by_index(device, button_index)
            if button is not None:
                self._gamepad_buttons[f"btn_index:{button_index}"] = button
                return button
        return None

    def _find_axis(self, device: Any, axis_enum: Any) -> Any:
        for name in ("find_axis", "findAxis"):
            fn = getattr(device, name, None)
            if fn is None:
                continue
            try:
                return fn(axis_enum)
            except Exception:
                return None
        return None

    def _find_button(self, device: Any, button_enum: Any) -> Any:
        for name in ("find_button", "findButton"):
            fn = getattr(device, name, None)
            if fn is None:
                continue
            try:
                return fn(button_enum)
            except Exception:
                return None
        return None

    def _find_axis_by_name(self, device: Any, axis_name: str) -> Any:
        axes = getattr(device, "axes", None)
        if axes is None:
            return None
        axis_name_norm = self._normalize_input_name(axis_name)
        for axis in axes:
            name = self._device_item_name(axis)
            if name == axis_name:
                return axis
            if name and self._normalize_input_name(name) == axis_name_norm:
                return axis
        return None

    def _find_button_by_name(self, device: Any, button_name: str) -> Any:
        buttons = getattr(device, "buttons", None)
        if buttons is None:
            return None
        button_name_norm = self._normalize_input_name(button_name)
        for button in buttons:
            name = self._device_item_name(button)
            if name == button_name:
                return button
            if name and self._normalize_input_name(name) == button_name_norm:
                return button
        return None

    def _find_axis_by_index(self, device: Any, index: int) -> Any:
        axes = getattr(device, "axes", None)
        if axes is None:
            return None
        if 0 <= index < len(axes):
            return axes[index]
        return None

    def _find_button_by_index(self, device: Any, index: int) -> Any:
        buttons = getattr(device, "buttons", None)
        if buttons is None:
            return None
        if 0 <= index < len(buttons):
            return buttons[index]
        return None

    def _device_item_name(self, item: Any) -> Optional[str]:
        if hasattr(item, "name"):
            try:
                return str(item.name)
            except Exception:
                return None
        for name in ("get_name", "getName"):
            fn = getattr(item, name, None)
            if fn is None:
                continue
            try:
                return str(fn())
            except Exception:
                return None
        return None

    def _attach_gamepad(self, device: Any) -> None:
        """Ensure Panda3D is actually polling the device."""
        attach = getattr(self.base, "attachInputDevice", None)
        if attach is None:
            return
        try:
            attach(device, "gamepad")
        except Exception:
            return

    def _normalize_input_name(self, name: str) -> str:
        """Normalize device item names for resilient matching."""
        return "".join(ch for ch in name.lower() if ch.isalnum())

    def _log_gamepad(self, device: Any) -> None:
        if isinstance(device, _RemoteGamepad):
            print(f"[gamepad] device: {device.name}")
            if device.axes:
                print("[gamepad] axes:")
                for name in sorted(device.axes.keys()):
                    print(f"  - {name}")
            if device.buttons:
                print("[gamepad] buttons:")
                for name in sorted(device.buttons.keys()):
                    print(f"  - {name}")
            return
        name = getattr(device, "name", None)
        if name is None:
            name = self._device_item_name(device)
        print(f"[gamepad] device: {name}")
        axes = getattr(device, "axes", None) or []
        buttons = getattr(device, "buttons", None) or []
        if axes:
            print("[gamepad] axes:")
            for idx, axis in enumerate(axes):
                print(f"  - {idx}: {self._device_item_name(axis)}")
        if buttons:
            print("[gamepad] buttons:")
            for idx, button in enumerate(buttons):
                print(f"  - {idx}: {self._device_item_name(button)}")

    def _axis_value(
        self, device: Any, axis_names: tuple[str, ...], axis_index: Optional[int]
    ) -> float:
        if isinstance(device, _RemoteGamepad):
            value = device.axis_value(axis_names, axis_index)
            remote_deadzone = float(os.getenv("GAMEPAD_REMOTE_DEADZONE", "0.02"))
            return self._apply_deadzone(value, remote_deadzone)
        axis = self._resolve_axis(device, axis_names, axis_index)
        if axis is None:
            return 0.0
        value = self._axis_raw_value(axis)
        self._log_axis_change(axis, value)
        return self._apply_deadzone(value, GAMEPAD_DEADZONE)

    def _trigger_value(
        self, device: Any, axis_names: tuple[str, ...], axis_index: Optional[int]
    ) -> float:
        if isinstance(device, _RemoteGamepad):
            value = device.axis_value(axis_names, axis_index)
            value = max(0.0, min(1.0, value))
            if value < GAMEPAD_TRIGGER_DEADZONE:
                return 0.0
            return value
        axis = self._resolve_axis(device, axis_names, axis_index)
        if axis is None:
            return 0.0
        value = self._axis_raw_value(axis)
        self._log_axis_change(axis, value)
        if value < -0.5:
            value = (value + 1.0) * 0.5
        value = max(0.0, min(1.0, value))
        if value < GAMEPAD_TRIGGER_DEADZONE:
            return 0.0
        return value

    def _axis_raw_value(self, axis: Any) -> float:
        if hasattr(axis, "value"):
            try:
                return float(axis.value)
            except Exception:
                return 0.0
        for name in ("getValue", "get_value"):
            fn = getattr(axis, name, None)
            if fn is None:
                continue
            try:
                return float(fn())
            except Exception:
                return 0.0
        return 0.0

    def _button_down(
        self, device: Any, button_names: tuple[str, ...], button_index: Optional[int]
    ) -> bool:
        if isinstance(device, _RemoteGamepad):
            pressed = device.button_value(button_names, button_index)
            if pressed:
                self._log_button_press(device)
            return pressed
        button = self._resolve_button(device, button_names, button_index)
        if button is None:
            return False
        pressed = self._button_raw_value(button)
        if pressed:
            self._log_button_press(button)
        return pressed

    def _button_pressed(
        self, device: Any, button_names: tuple[str, ...], button_index: Optional[int]
    ) -> bool:
        if isinstance(device, _RemoteGamepad):
            key = f"edge:{id(device)}:{button_index}:{button_names}"
            pressed = device.button_value(button_names, button_index)
            prev = self._button_prev.get(key, False)
            self._button_prev[key] = pressed
            if pressed and not prev:
                self._log_button_press(device)
            return pressed and not prev
        button = self._resolve_button(device, button_names, button_index)
        if button is None:
            return False
        key = f"edge:{id(button)}"
        pressed = self._button_raw_value(button)
        prev = self._button_prev.get(key, False)
        self._button_prev[key] = pressed
        if pressed and not prev:
            self._log_button_press(button)
        return pressed and not prev

    def _button_raw_value(self, button: Any) -> bool:
        if hasattr(button, "pressed"):
            try:
                return bool(button.pressed)
            except Exception:
                return False
        if hasattr(button, "value"):
            try:
                return bool(button.value)
            except Exception:
                return False
        for name in ("is_pressed", "isPressed"):
            fn = getattr(button, name, None)
            if fn is None:
                continue
            try:
                return bool(fn())
            except Exception:
                return False
        return False

    def _apply_deadzone(self, value: float, deadzone: float) -> float:
        if abs(value) < deadzone:
            return 0.0
        scaled = (abs(value) - deadzone) / (1.0 - deadzone)
        return scaled * (1.0 if value >= 0.0 else -1.0)

    def _axis_lock(self, x: float, y: float) -> tuple[float, float]:
        """Suppress minor cross-axis drift when one axis dominates."""
        ratio = float(GAMEPAD_AXIS_LOCK_RATIO)
        ax = abs(x)
        ay = abs(y)
        if ax == 0.0 and ay == 0.0:
            return x, y
        if ax >= ay * (1.0 + ratio):
            return x, 0.0
        if ay >= ax * (1.0 + ratio):
            return 0.0, y
        return x, y

    def _log_axis_change(self, axis: Any, value: float) -> None:
        if os.getenv("GAMEPAD_DEBUG") != "2":
            return
        key = id(axis)
        prev = self._axis_prev.get(key, 0.0)
        if abs(value - prev) < 0.15:
            return
        self._axis_prev[key] = value
        idx = self._device_item_index(axis, "axes")
        print(f"[gamepad] axis {idx} -> {value:.3f}")

    def _log_button_press(self, button: Any) -> None:
        if os.getenv("GAMEPAD_DEBUG") != "2":
            return
        if isinstance(button, _RemoteGamepad):
            print("[gamepad] button pressed (remote)")
            return
        idx = self._device_item_index(button, "buttons")
        print(f"[gamepad] button {idx} pressed")

    def _device_item_index(self, item: Any, collection_attr: str) -> Optional[int]:
        device = self._gamepad
        if device is None:
            return None
        collection = getattr(device, collection_attr, None)
        if collection is None:
            return None
        for idx, candidate in enumerate(collection):
            if candidate is item:
                return idx
        return None

    def _is_gamepad_active(self) -> bool:
        gp = self._get_gamepad()
        if gp is None:
            return False
        if abs(self._axis_value(gp, GAMEPAD_AXIS_LEFT_X, GAMEPAD_AXIS_INDEX_LEFT_X)) > 0.0:
            return True
        if abs(self._axis_value(gp, GAMEPAD_AXIS_LEFT_Y, GAMEPAD_AXIS_INDEX_LEFT_Y)) > 0.0:
            return True
        if abs(self._axis_value(gp, GAMEPAD_AXIS_RIGHT_X, GAMEPAD_AXIS_INDEX_RIGHT_X)) > 0.0:
            return True
        if abs(self._axis_value(gp, GAMEPAD_AXIS_RIGHT_Y, GAMEPAD_AXIS_INDEX_RIGHT_Y)) > 0.0:
            return True
        if self._trigger_value(gp, GAMEPAD_AXIS_L2, GAMEPAD_AXIS_INDEX_L2) > 0.0:
            return True
        if self._trigger_value(gp, GAMEPAD_AXIS_R2, GAMEPAD_AXIS_INDEX_R2) > 0.0:
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_L1, GAMEPAD_BUTTON_INDEX_L1):
            return True
        if self._button_down(gp, GAMEPAD_BUTTON_R1, GAMEPAD_BUTTON_INDEX_R1):
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
