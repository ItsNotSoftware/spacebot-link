from math import pi, sin, cos
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3, CardMaker, Texture

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
        self.scene.setDepthWrite(False)

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

        # Load and transform the panda actor.
        self.pandaActor = Actor("models/panda-model", {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)
        self.pandaActor.loop("walk")

        # Create and play the sequence that coordinates the panda’s movement.
        posInterval1 = self.pandaActor.posInterval(13, Point3(0, -10, 0), startPos=Point3(0, 10, 0))
        posInterval2 = self.pandaActor.posInterval(13, Point3(0, 10, 0), startPos=Point3(0, -10, 0))
        hprInterval1 = self.pandaActor.hprInterval(3, Point3(180, 0, 0), startHpr=Point3(0, 0, 0))
        hprInterval2 = self.pandaActor.hprInterval(3, Point3(0, 0, 0), startHpr=Point3(180, 0, 0))

        self.pandaPace = Sequence(posInterval1, hprInterval1, posInterval2, hprInterval2, name="pandaPace")
        self.pandaPace.loop()

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
        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont

app = MyApp()
app.run()
