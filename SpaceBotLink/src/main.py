"""
SpaceBotLink – Panda3D viewer with ZMQ video background.
Transparency is authored in Blender (glTF). Transparent flicker fixed by
two-pass rendering (backfaces, then frontfaces) with fixed bins.
"""

from math import pi, sin, cos
import cv2
import numpy as np
import zmq

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    loadPrcFileData,
    Point3,
    CardMaker,
    Texture,
    DirectionalLight,
    AmbientLight,
    Vec4,
    TransparencyAttrib,
    CullFaceAttrib,
    DepthOffsetAttrib,
)

# ---- Engine config (set before ShowBase) ----
loadPrcFileData("", "window-title SpaceBotLink")
loadPrcFileData("", "framebuffer-srgb true")  # correct color space for PBR/glTF
# Important: we’ll control draw order ourselves via bins
loadPrcFileData("", "transparency-sort off")
# loadPrcFileData("", "notify-level-gobj debug")
# Optional: MSAA (nice for alpha edges, not required)
# loadPrcFileData("", "framebuffer-multisample 1")
# loadPrcFileData("", "multisamples 8")


class SpaceBotLinkApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        # --- Background video card (no depth interaction) ---
        aspect_ratio = 720 / 1280  # (h / w) of incoming frames
        cm = CardMaker("background")
        cm.setFrame(-1, 1, -aspect_ratio, aspect_ratio)
        self.bg_card = self.camera.attachNewNode(cm.generate())
        self.bg_card.setScale(50)
        self.bg_card.setPos(0, 100, 0)

        self.background_texture = Texture("background")
        self.background_texture.setup2dTexture(
            1, 1, Texture.T_unsigned_byte, Texture.F_rgb
        )
        self.bg_card.setTexture(self.background_texture)
        self.bg_card.setBin("background", 0)
        self.bg_card.setDepthWrite(False)
        self.bg_card.setDepthTest(False)

        # --- PBR shading + simple lights ---
        self.render.setShaderAuto()

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.0, 1.0, 1.0, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self.render.setLight(sun_np)

        amb = AmbientLight("ambient")
        amb.setColor(Vec4(0.35, 0.35, 0.35, 1.0))
        amb_np = self.render.attachNewNode(amb)
        self.render.setLight(amb_np)

        # --- ZMQ subscriber ---
        self._setup_zmq("tcp://localhost:5555")

        # --- Tasks ---
        self.taskMgr.add(self._zmq_task, "ZMQTask")
        self.taskMgr.add(self._spin_camera_task, "SpinCameraTask")

        # --- Load model once (from glTF) ---
        base_model = self.loader.load_model("../assets/cobot4.glb")
        base_model.setScale(20)
        base_model.setPos(Point3(8, 0, 6))
        base_model.setHpr(0, 45, 0)

        # We’ll render two *instances* with opposite culling and fixed order:
        # 1) backfaces first, 2) frontfaces second.
        self.model_back = base_model.copy_to(self.render)
        self.model_front = base_model.copy_to(self.render)

        # (Hide original if you keep it around)
        base_model.hide()

        # Common transparent setup (no depth writes, but DO depth test)
        for np in (self.model_back, self.model_front):
            np.setTransparency(TransparencyAttrib.MDual)  # depth pre-pass helps
            np.setDepthWrite(False)  # blended pixels don't write depth
            # Small depth offset so coplanar bits don’t fight (optional)
            np.setAttrib(DepthOffsetAttrib.make(1))

        # Backface pass (renders what's behind first)
        self.model_back.setAttrib(
            CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise)
        )
        self.model_back.setBin("fixed", 10)

        # Frontface pass (on top of backfaces)
        self.model_front.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullClockwise))
        self.model_front.setBin("fixed", 11)

        # IMPORTANT: Ensure Panda doesn’t collapse the two instances back together.
        # Don’t call flattenStrong() on parent; keep them as separate passes.

    # ------------- helpers -------------

    def _setup_zmq(self, endpoint: str) -> None:
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect(endpoint)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 1)  # ms timeout

    # ------------- tasks -------------

    def _zmq_task(self, task: Task):
        """Poll the ZMQ socket for a JPEG frame and update the background texture."""
        try:
            msg = self.zmq_socket.recv(flags=zmq.NOBLOCK)
            np_arr = np.frombuffer(msg, dtype=np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return Task.cont

            # Flip if needed (keep if your producer is inverted)
            img_bgr = np.flipud(img_bgr)

            # BGR -> RGB for Panda
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            self.background_texture.setup2dTexture(
                w, h, Texture.T_unsigned_byte, Texture.F_rgb
            )
            self.background_texture.setRamImageAs(img_rgb.tobytes(), "RGB")
        except zmq.Again:
            pass
        return Task.cont

    def _spin_camera_task(self, task: Task):
        """Orbit the camera slowly around the origin."""
        angle_deg = task.time * 25.0
        angle_rad = angle_deg * (pi / 180.0)
        self.camera.setPos(50 * sin(angle_rad), -50 * cos(angle_rad), 3)
        self.camera.setHpr(angle_deg, 0, 0)
        return Task.cont


if __name__ == "__main__":
    app = SpaceBotLinkApp()
    app.run()
