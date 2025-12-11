"""Shared configuration constants for SpaceBotLink."""

from __future__ import annotations

from panda3d.core import KeyboardButton

# Window / rendering
WINDOW_TITLE = "SpaceBotLink"
FRAMEBUFFER_SRGB_CFG = "framebuffer-srgb true"
TRANSPARENCY_SORT_CFG = "transparency-sort off"

# Default endpoints and assets
default_sensor_endpoint = "tcp://localhost:5556"
default_image_endpoint = "tcp://localhost:5560"
default_cmd_endpoint = "tcp://localhost:5557"
default_gltf_model = "../assets/cobot_ghost.glb"

# ROS topics used by the bridge
TOPIC_IMAGE = "/main_camera/image"
TOPIC_CAMINFO = "/main_camera/camera_info"
TOPIC_IMU = "/imu/data"
TOPIC_POSE = "/space_cobot/pose"
TOPIC_CMD_VEL = "/space_cobot/cmd_vel"
TOPIC_GOAL = "/nav6d/goal"
TOPIC_PATH = "/nav6d/planner/path"
TOPIC_CMD_PATH = "/nav6d/planner/path"

# Speeds and thresholds
MOVE_SPEED = 0.8
ROTATE_SPEED = 1.5
FOLLOW_POS_EPS = 0.02
FOLLOW_HPR_EPS = 1.0
FOLLOW_REACHED_THRESH = 0.2
FOLLOW_SAMPLE_PERIOD = 0.15
# Path visualization defaults
PATH_MODE_DEFAULT = "poses"  # poses | poses_line | animated
PATH_POSE_STRIDE = 4
PATH_LINE_STRIDE = 8
PATH_ANIM_SPEED = 1.0  # m/s
PATH_LINE_COLOR = (0.95, 0.80, 0.20, 0.9)
PATH_LINE_THICKNESS = 4.0

# Avatar offsets (center camera within the mesh)
CAMERA_FORWARD_OFFSET_M = 0.40
CAMERA_UP_OFFSET_M = 0.08
AVATAR_CAMERA_OFFSET = (0.0, -CAMERA_FORWARD_OFFSET_M, -CAMERA_UP_OFFSET_M)

# Key bindings
FORWARD_BUTTON = KeyboardButton.ascii_key("w")
BACKWARD_BUTTON = KeyboardButton.ascii_key("s")
LEFT_BUTTON = KeyboardButton.ascii_key("a")
RIGHT_BUTTON = KeyboardButton.ascii_key("d")
UP_BUTTON = KeyboardButton.ascii_key("e")
DOWN_BUTTON = KeyboardButton.ascii_key("q")
UP_BUTTON_ALT = KeyboardButton.space()
DOWN_BUTTON_ALT = KeyboardButton.lshift()
PITCH_UP_BUTTON = KeyboardButton.ascii_key("i")
PITCH_DOWN_BUTTON = KeyboardButton.ascii_key("k")
YAW_LEFT_BUTTON = KeyboardButton.ascii_key("u")
YAW_RIGHT_BUTTON = KeyboardButton.ascii_key("o")
ROLL_LEFT_BUTTON = KeyboardButton.ascii_key("j")
ROLL_RIGHT_BUTTON = KeyboardButton.ascii_key("l")
RESET_ORIENT_BUTTON = KeyboardButton.ascii_key("r")
RESET_TO_ROBOT_ORIENT_BUTTON = KeyboardButton.backspace()
