# app.py
"""
SpaceBotLink – Panda3D viewer with ZMQ video background.
Transparency flicker fixed via two-pass avatar (backfaces then frontfaces).
"""

from panda3d.core import (
    loadPrcFileData,
    CardMaker,
    Texture,
    DirectionalLight,
    AmbientLight,
    Vec4,
    PythonTask,
)
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from math import pi, sin, cos

from camera_stream import CameraStream
from sensor_bus import SensorBus
from avatar import Avatar
from ui import UI
from intrinsics import apply_opencv_intrinsics_to_lens


# --- engine config (before ShowBase) ---
loadPrcFileData("", "window-title SpaceBotLink")
loadPrcFileData("", "framebuffer-srgb true")
loadPrcFileData("", "transparency-sort off")


class SpacebotLinkApp(ShowBase):
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

        # avatar
        if self.loader is not None:
            self.avatar = Avatar(self.render, self.loader, gltf_model)

        # ui
        self.ui = UI(self)

        w, h = self.camera_stream.size
        fx = fy = 900.0
        cx, cy = w / 2, h / 2
        apply_opencv_intrinsics_to_lens(self.camLens, w, h, fx, fy, cx, cy)

        # tasks
        self.taskMgr.add(self._camera_task, "CameraTask")
        self.taskMgr.add(self._sensor_task, "SensorTask")
        self.taskMgr.add(self._orbit_task, "OrbitTask")
        self.taskMgr.add(self._hud_task, "HUDTask")

        # cleanup
        self.exitFunc = self._cleanup

    # ---- bg helpers ----
    def _make_bg_card(self, initial_aspect: float):
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

    # ---- tasks ----
    def _camera_task(self, task: PythonTask):
        if self.camera_stream.poll():
            rgb = self.camera_stream.frame_rgb
            h, w = rgb.shape[:2]  # type: ignore
            # keep it simple: no dynamic card reshape; just update texture
            self.bg_tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
            self.bg_tex.setRamImageAs(rgb.tobytes(), "RGB")  # type: ignore
        return Task.cont

    def _sensor_task(self, task: PythonTask):
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

        return Task.cont

    def _orbit_task(self, task: PythonTask):
        if not self.ui.orbit_enabled:
            return Task.cont
        angle_deg = task.time * 25.0
        angle_rad = angle_deg * (pi / 180.0)
        self.camera.setPos(50 * sin(angle_rad), -50 * cos(angle_rad), 3)  # type: ignore
        self.camera.setHpr(angle_deg, 0, 0)  # type: ignore
        return Task.cont

    def _hud_task(self, task: PythonTask):
        rgb = self.camera_stream.frame_rgb
        if rgb is not None:
            w, h = rgb.shape[1], rgb.shape[0]
            self.ui.update(f"Video {w}x{h}")
        else:
            self.ui.update("Waiting for video…")
        return Task.cont

    def _cleanup(self):
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
