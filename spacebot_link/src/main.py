# app.py
"""SpaceBotLink viewer.

This module defines the main Panda3D application that renders a 3D avatar
over a live video background received via ZMQ. Transparency flicker is
mitigated using a two-pass avatar (backfaces then frontfaces).

The application also listens to a sensor bus for pose and camera intrinsic
updates and reflects those changes in the scene and active camera lens.
"""

from panda3d.core import (
    loadPrcFileData,
    CardMaker,
    Texture,
    DirectionalLight,
    AmbientLight,
    Vec4,
    PythonTask,
    KeyboardButton,
    Vec3,
    ClockObject,
)
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from math import pi, sin, cos
from pathlib import Path

from camera_stream import CameraStream
from sensor_bus import SensorBus
from avatar import Avatar
from ui import UI
from intrinsics import apply_opencv_intrinsics_to_lens

MOVE_SPEED = 0.8
ROTATE_SPEED = 1.5


# --- engine config (before ShowBase) ---
loadPrcFileData("", "window-title SpaceBotLink")
loadPrcFileData("", "framebuffer-srgb true")
loadPrcFileData("", "transparency-sort off")

# Ensure glTF loader is registered (for packaged or strict environments)
try:  # pragma: no cover - optional runtime registration
    from panda3d_gltf import GLTFLoader  # type: ignore

    GLTFLoader.register_loader()
except Exception:
    pass

# --- key mappings ---
forward_button = KeyboardButton.ascii_key("w")
backward_button = KeyboardButton.ascii_key("s")
left_button = KeyboardButton.ascii_key("a")
right_button = KeyboardButton.ascii_key("d")
up_button = KeyboardButton.ascii_key("e")
down_button = KeyboardButton.ascii_key("q")
pitch_up_button = KeyboardButton.ascii_key("i")
pitch_down_button = KeyboardButton.ascii_key("k")
yaw_left_button = KeyboardButton.ascii_key("j")
yaw_right_button = KeyboardButton.ascii_key("l")
roll_left_button = KeyboardButton.ascii_key("u")
roll_right_button = KeyboardButton.ascii_key("o")


class SpacebotLinkApp(ShowBase):
    """Main Panda3D application.

    Args:
        cam_endpoint: ZMQ endpoint for the compressed RGB video stream.
        sensor_endpoint: ZMQ endpoint for sensor messages (pose/intrinsics).
        gltf_model: Path to the GLTF avatar model to load.
    """

    def __init__(
        self,
        cam_endpoint: str = "tcp://localhost:5555",
        sensor_endpoint: str = "tcp://localhost:5556",
        gltf_model: str = "../assets/cobot4.glb",
    ):
        super().__init__()
        self.disableMouse()
        self.render.setShaderAuto()

        # lights
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1, 1, 1, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self.render.setLight(sun_np)

        amb = AmbientLight("ambient")
        amb.setColor(Vec4(0.35, 0.35, 0.35, 1))
        amb_np = self.render.attachNewNode(amb)
        self.render.setLight(amb_np)

        # IO
        self.camera_stream = CameraStream(cam_endpoint)
        self.sensors = SensorBus(sensor_endpoint)

        # background card
        self._make_bg_card(initial_aspect=self.camera_stream.aspect)

        # avatar (resolve model path robustly)
        if self.loader is not None:
            model_path = Path(gltf_model)
            if not model_path.exists():
                # Resolve relative to this file: spacebot_link/src/ -> ../assets/
                model_path = (
                    Path(__file__).resolve().parent.parent
                    / "assets"
                    / Path(gltf_model).name
                )
            self.avatar = Avatar(self.render, self.loader, str(model_path))

        # ui
        self.ui = UI(self)

        w, h = self.camera_stream.size
        fx = fy = 900.0
        cx, cy = w / 2, h / 2
        apply_opencv_intrinsics_to_lens(self.camLens, w, h, fx, fy, cx, cy)
        # Prevent near/far plane clipping of large or close models
        self.camLens.setNear(0.1)
        self.camLens.setFar(5000.0)
        self._update_bg_scale()

        # tasks
        self.taskMgr.add(self._camera_task, "CameraTask")
        self.taskMgr.add(self._sensor_task, "SensorTask")
        self.taskMgr.add(self._pool_keyboard, "PoolKeyboard")
        self.taskMgr.add(self._hud_task, "HUDTask")

        # cleanup
        self.exitFunc = self._cleanup

    # ---- bg helpers ----
    def _make_bg_card(self, initial_aspect: float):
        """Create the textured background card for the video stream.

        Args:
            initial_aspect: Initial height/width ratio used to size the card.
        """
        cm = CardMaker("background")
        cm.setFrame(-1, 1, -initial_aspect, initial_aspect)
        self.bg_card = self.camera.attachNewNode(cm.generate())  # type: ignore
        self.bg_card.setScale(50)
        self.bg_card.setPos(0, 100, 0)
        self.bg_card.setBin("background", 0)
        self.setBackgroundColor(0, 0, 0, 1)
        self.bg_card.setDepthWrite(False)
        self.bg_card.setDepthTest(False)

        self.bg_tex = Texture("background")
        self.bg_tex.setup2dTexture(2, 2, Texture.T_unsigned_byte, Texture.F_rgb)
        self.bg_card.setTexture(self.bg_tex)

        # Remember aspect of the background card and scale to fill the view
        self._bg_aspect = float(initial_aspect)
        self._update_bg_scale()

    def _update_bg_scale(self) -> None:
        """Scale the background card so it fills the current camera view.

        Uses the camera's vertical FOV and the card's distance from the camera
        to compute the required uniform scale so that the card exactly matches
        the frustum extents at that depth (no letterboxing).
        """
        if not hasattr(self, "bg_card"):
            return
        # Distance of the card from the camera (parented to camera)
        d = abs(self.bg_card.getY())  # type: ignore
        # Vertical FOV in degrees -> radians
        fov_x, fov_y = self.camLens.getFov()
        fov_y_rad = fov_y * (pi / 180.0)
        # Required half-height in world units at distance d
        half_h = d * (sin(fov_y_rad / 2.0) / cos(fov_y_rad / 2.0))  # tan
        # Our card has local vertical half-size of `_bg_aspect` before scaling
        # so uniform scale s must satisfy: s * _bg_aspect = half_h
        if getattr(self, "_bg_aspect", 0) > 0:
            s = half_h / self._bg_aspect
            self.bg_card.setScale(s)  # type: ignore

    # ---- tasks ----
    def _camera_task(self, task: PythonTask):
        """Poll the ZMQ camera stream and update the background texture.

        Args:
            task: Panda3D task object.

        Returns:
            `direct.task.Task.cont` to continue scheduling the task.
        """
        if self.camera_stream.poll():
            rgb = self.camera_stream.frame_rgb
            h, w = rgb.shape[:2]  # type: ignore
            # keep it simple: no dynamic card reshape; just update texture
            self.bg_tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
            self.bg_tex.setRamImageAs(rgb.tobytes(), "RGB")  # type: ignore
        return Task.cont

    def _sensor_task(self, task: PythonTask):
        """Handle sensor bus messages (pose and camera intrinsics).

        Args:
            task: Panda3D task object.

        Returns:
            `direct.task.Task.cont` to continue scheduling the task.
        """
        self.sensors.poll()

        pose = self.sensors.get("pose")
        if pose:
            self.avatar.set_pos(pose.get("x", 8), pose.get("y", 0), pose.get("z", 6))
            self.avatar.set_hpr(pose.get("h", 0), pose.get("p", 45), pose.get("r", 0))

        intr = self.sensors.get("intrinsics")
        if intr:
            w = int(intr.get("width", self.camera_stream.size[0]))
            h = int(intr.get("height", self.camera_stream.size[1]))
            fx = float(intr.get("fx"))
            fy = float(intr.get("fy"))
            cx = float(intr.get("cx"))
            cy = float(intr.get("cy"))
            if all(v is not None for v in [fx, fy, cx, cy]):
                apply_opencv_intrinsics_to_lens(self.camLens, w, h, fx, fy, cx, cy)
                # Reapply clip planes after intrinsics change
                self.camLens.setNear(0.1)
                self.camLens.setFar(5000.0)
                self._update_bg_scale()

        return Task.cont

    def _pool_keyboard(self, task: PythonTask):
        """Poll keyboard state and move/rotate the avatar.

        WASD for planar movement, Q/E for down/up. I/K pitch, J/L yaw,
        U/O roll. Movement is in the avatar's local space.

        Args:
            task: Panda3D task object.

        Returns:
            `direct.task.Task.cont` to continue scheduling the task.
        """
        # Time step since last frame
        dt = ClockObject.getGlobalClock().getDt()
        mw = self.mouseWatcherNode  # type: ignore
        if not mw:
            return Task.cont

        # Translation in local avatar space
        move = Vec3(0, 0, 0)
        if mw.is_button_down(forward_button):
            move.y += MOVE_SPEED * dt
        if mw.is_button_down(backward_button):
            move.y -= MOVE_SPEED * dt
        if mw.is_button_down(left_button):
            move.x -= MOVE_SPEED * dt
        if mw.is_button_down(right_button):
            move.x += MOVE_SPEED * dt
        if mw.is_button_down(up_button):
            move.z += MOVE_SPEED * dt
        if mw.is_button_down(down_button):
            move.z -= MOVE_SPEED * dt

        if move.length_squared() > 0:
            self.avatar.move_local(move.x, move.y, move.z)

        # Rotation deltas in degrees
        dh = dp = dr = 0.0
        step = ROTATE_SPEED * 60.0 * dt
        if mw.is_button_down(yaw_left_button):
            dh += step
        if mw.is_button_down(yaw_right_button):
            dh -= step
        if mw.is_button_down(pitch_up_button):
            dp += step
        if mw.is_button_down(pitch_down_button):
            dp -= step
        if mw.is_button_down(roll_left_button):
            dr += step
        if mw.is_button_down(roll_right_button):
            dr -= step

        if dh or dp or dr:
            self.avatar.add_hpr(dh, dp, dr)
        return Task.cont

    def _hud_task(self, task: PythonTask):
        """Update UI/HUD text with current video status.

        Args:
            task: Panda3D task object.

        Returns:
            `direct.task.Task.cont` to continue scheduling the task.
        """
        rgb = self.camera_stream.frame_rgb
        if rgb is not None:
            w, h = rgb.shape[1], rgb.shape[0]
            self.ui.update(f"Video {w}x{h}")
        else:
            self.ui.update("Waiting for video…")
        return Task.cont

    def _cleanup(self):
        """Release resources on exit (camera stream, sensor bus)."""
        try:
            self.camera_stream.close()
        except Exception:
            pass
        try:
            self.sensors.close()
        except Exception:
            pass


if __name__ == "__main__":
    app = SpacebotLinkApp()
    app.run()
