"""Keyboard polling for avatar movement or robot cmd_vel teleop."""

from __future__ import annotations

from panda3d.core import ClockObject, Vec3
from typing import Any

from config import (
    BACKWARD_BUTTON,
    DOWN_BUTTON,
    DOWN_BUTTON_ALT,
    FORWARD_BUTTON,
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


class InputController:
    def __init__(self, base: Any, renderer: Any, nav: Any, cmd_pub: Any) -> None:
        """Handle keyboard input for avatar motion or robot cmd_vel teleop."""
        self.base = base
        self.renderer = renderer
        self.nav = nav
        self.cmd_pub = cmd_pub
        self._move_robot = False

    def set_move_mode(self, move_robot: bool) -> None:
        """Toggle control target between avatar and robot."""
        move_robot = bool(move_robot)
        if move_robot == self._move_robot:
            return
        self._move_robot = move_robot
        target = "Robot" if self._move_robot else "Avatar"
        self.nav.set_move_target(target)
        if not self._move_robot:
            self._publish_cmd_vel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def toggle_move_mode(self) -> None:
        """Invert the current move target."""
        self.set_move_mode(not self._move_robot)

    def is_robot_mode(self) -> bool:
        """Return True when controlling the robot (cmd_vel)."""
        return self._move_robot

    def poll(self) -> None:
        """Process keyboard state each frame."""
        dt = ClockObject.getGlobalClock().getDt()
        mw = self.base.mouseWatcherNode
        if not mw:
            return

        if not self._move_robot:
            move = Vec3(0, 0, 0)
            if mw.is_button_down(FORWARD_BUTTON):
                move.y += MOVE_SPEED * dt
            if mw.is_button_down(BACKWARD_BUTTON):
                move.y -= MOVE_SPEED * dt
            if mw.is_button_down(LEFT_BUTTON):
                move.x -= MOVE_SPEED * dt
            if mw.is_button_down(RIGHT_BUTTON):
                move.x += MOVE_SPEED * dt
            if mw.is_button_down(UP_BUTTON) or mw.is_button_down(UP_BUTTON_ALT):
                move.z += MOVE_SPEED * dt
            if mw.is_button_down(DOWN_BUTTON) or mw.is_button_down(DOWN_BUTTON_ALT):
                move.z -= MOVE_SPEED * dt
            if move.length_squared() > 0:
                frame = self.base.camera if self.base.camera is not None else self.base.render
                q = frame.getQuat(self.base.render)
                delta = q.xform(move)
                self.renderer.move_avatar(delta.x, delta.y, delta.z)

            dh = dp = dr = 0.0
            step = ROTATE_SPEED * 60.0 * dt
            if mw.is_button_down(YAW_LEFT_BUTTON):
                dh += step
            if mw.is_button_down(YAW_RIGHT_BUTTON):
                dh -= step
            if mw.is_button_down(PITCH_UP_BUTTON):
                dp += step
            if mw.is_button_down(PITCH_DOWN_BUTTON):
                dp -= step
            if mw.is_button_down(ROLL_LEFT_BUTTON):
                dr += step
            if mw.is_button_down(ROLL_RIGHT_BUTTON):
                dr -= step
            if dh or dp or dr:
                self.renderer.add_avatar_hpr(dh, dp, dr)
            if (
                mw.is_button_down(RESET_TO_ROBOT_ORIENT_BUTTON)
                and self.nav.state.last_robot_hpr is not None
            ):
                h, p, r = self.nav.state.last_robot_hpr
                self.renderer.avatar.set_hpr(h, p, r)
            if mw.is_button_down(RESET_ORIENT_BUTTON):
                self.renderer.reset_avatar_to_camera_hpr()
        else:
            lin_x = lin_y = lin_z = 0.0
            ang_x = ang_y = ang_z = 0.0

            if mw.is_button_down(FORWARD_BUTTON):
                lin_x = +MOVE_SPEED
            elif mw.is_button_down(BACKWARD_BUTTON):
                lin_x = -MOVE_SPEED
            if mw.is_button_down(LEFT_BUTTON):
                lin_y = +MOVE_SPEED
            elif mw.is_button_down(RIGHT_BUTTON):
                lin_y = -MOVE_SPEED
            if mw.is_button_down(UP_BUTTON) or mw.is_button_down(UP_BUTTON_ALT):
                lin_z = +MOVE_SPEED
            if mw.is_button_down(DOWN_BUTTON) or mw.is_button_down(DOWN_BUTTON_ALT):
                lin_z = -MOVE_SPEED

            if mw.is_button_down(ROLL_LEFT_BUTTON):
                ang_x = +ROTATE_SPEED
            elif mw.is_button_down(ROLL_RIGHT_BUTTON):
                ang_x = -ROTATE_SPEED
            if mw.is_button_down(PITCH_UP_BUTTON):
                ang_y = +ROTATE_SPEED
            elif mw.is_button_down(PITCH_DOWN_BUTTON):
                ang_y = -ROTATE_SPEED
            if mw.is_button_down(YAW_LEFT_BUTTON):
                ang_z = +ROTATE_SPEED
            elif mw.is_button_down(YAW_RIGHT_BUTTON):
                ang_z = -ROTATE_SPEED

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
