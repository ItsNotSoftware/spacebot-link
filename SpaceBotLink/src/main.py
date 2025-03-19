from math import pi, sin, cos
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Point3, CardMaker, Texture, TransparencyAttrib
from panda3d.core import TransparencyAttrib


# Import additional libraries for ZMQ and image processing.
import zmq
import cv2
import numpy as np

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Disable the default mouse-based camera control.
        self.disableMouse()

        # Create a background card.
        cm = CardMaker("background")
        aspect_ratio = 720 / 1280
        cm.setFrame(-1, 1, -aspect_ratio, aspect_ratio)
        self.scene = self.camera.attachNewNode(cm.generate())
        self.scene.setScale(50)
        self.scene.setPos(0, 100, 0)

        # Create an empty texture that we will update from the ZMQ stream.
        self.background_texture = Texture("background")
        # Initialize with a dummy 1x1 texture.
        self.background_texture.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_rgb)
        self.scene.setTexture(self.background_texture)
        self.scene.setBin("background", 0)
        self.scene.setDepthWrite(True)

        # Set up the ZMQ subscriber to receive JPEG images.
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect("tcp://localhost:5555")
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # Optionally, set a receive timeout so the call doesn’t block.
        self.zmq_socket.RCVTIMEO = 1

        # Add a task to poll the ZMQ socket.
        self.taskMgr.add(self.zmqTask, "ZMQTask")
        # Add the camera spin task.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Load and transform the .glb model.
        self.model = self.loader.load_model("../assets/cobot.glb")  # Replace with your model path
        self.model.setScale(12)  # Adjust scale as needed
        self.model.reparentTo(self.render)
        self.model.setPos(Point3(0, 0, 10))  # Adjust position as needed
        self.model.setHpr(Point3(0, 45, 0))  # Adjust rotation as needed

        # Set color to black with some transparency
        self.model.setColor(0.2, 0.1, 0.1, 0.2)  # RGBA (Black with 50% transparency)
        self.model.setTransparency(TransparencyAttrib.M_alpha)
        self.model.setDepthOffset(1)
        self.model.setBin("transparent", 1)
        self.model.setDepthWrite(False)


    def zmqTask(self, task):
        """Poll the ZMQ socket for a new JPEG frame and update the background texture."""
        try:
            # Try to receive a message without blocking.
            msg = self.zmq_socket.recv(flags=zmq.NOBLOCK)
            # Convert the received bytes into a numpy array.
            np_arr = np.frombuffer(msg, dtype=np.uint8)
            # Decode the JPEG image
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                # Rotate the image 90 degrees clockwise.
                img = np.flipud(img)
                print(img.shape)
                # Convert BGR (OpenCV default) to RGB.
                # (Re)initialize the texture size to match the incoming image.
                self.background_texture.setup2dTexture(img.shape[1], img.shape[0], Texture.T_unsigned_byte, Texture.F_rgb)
                # Update the texture with the new image data.
                self.background_texture.setRamImage(img.tobytes())
        except zmq.Again:
            # No message was available this frame.
            pass
        return Task.cont

    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(50 * sin(angleRadians), -50 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont

app = MyApp()
app.run()