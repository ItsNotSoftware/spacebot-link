"""Shared configuration constants for SpaceBotLink."""

from __future__ import annotations

from panda3d.core import KeyboardButton

# Select which remote input daemon to autostart: "gamepad", "spacemouse", "both", "none".
REMOTE_INPUT_DAEMON = "spacemouse"

# Window / rendering
WINDOW_TITLE = "SpaceBotLink"
FRAMEBUFFER_SRGB_CFG = "framebuffer-srgb true"
TRANSPARENCY_SORT_CFG = "transparency-sort off"

# Default endpoints and assets
default_sensor_endpoint = "tcp://localhost:5556"
default_image_endpoint = "tcp://localhost:5560"
default_cmd_endpoint = "tcp://localhost:5557"
default_gltf_model = "../assets/cobot_ghost.glb"
PATH_GHOST_MODEL = "../assets/ghost_006.glb"

# OctoMap raycast service (interface-side)
OCTOMAP_SERVER_BIN = "octomap_raycast_service/build/octomap_raycast_service"
OCTOMAP_SERVER_MAP = "octomap_raycast_service/iss.bt"
OCTOMAP_SERVER_ENDPOINT = "tcp://127.0.0.1:5570"
OCTOMAP_SERVER_MAX_RANGE = 50.0
OCTOMAP_QUERY_PERIOD_S = 0.1
AVATAR_ALPHA_VISIBLE = 0.85
AVATAR_COLOR_VISIBLE = (1.0, 1.0, 1.0, AVATAR_ALPHA_VISIBLE)
# Orange when occluded but reachable, red when inside an obstacle.
AVATAR_COLOR_OCCLUDED = (1.0, 0.55, 0.1, AVATAR_ALPHA_VISIBLE)
AVATAR_COLOR_IN_OBSTACLE = (1.0, 0.0, 0.0, AVATAR_ALPHA_VISIBLE)

# ROS topics used by the bridge
TOPIC_IMAGE = "/main_camera/image"
TOPIC_CAMINFO = "/main_camera/camera_info"
TOPIC_IMU = "/imu/data"
TOPIC_POSE = "/space_cobot/pose"
TOPIC_CMD_VEL = "/space_cobot/cmd_vel"
TOPIC_GOAL = "/nav6d/goal"
TOPIC_PATH = "/nav6d/planner/path"
TOPIC_CMD_PATH = "/nav6d/planner/path"
TOPIC_FLOOR_HEIGHT = "/floor_height"
TOPIC_PATH_QUALITY = "/nav6d/path_quality"
TOPIC_PATH_EXEC_SUMMARY_VEL = "/nav6d/velocity_controller/path_execution_summary"
TOPIC_PATH_EXEC_SUMMARY_FORCE = "/nav6d/force_controller/path_execution_summary"

# Speeds and thresholds
MOVE_SPEED = 2.0
ROTATE_SPEED = 1.2
FOLLOW_POS_EPS = 0.02
FOLLOW_HPR_EPS = 1.0
FOLLOW_REACHED_THRESH = 0.2
FOLLOW_SAMPLE_PERIOD = 0.2
AVATAR_AUTO_RESET_DISTANCE = 0.3
AVATAR_AUTO_RESET_DELAY_S = 2.5
AVATAR_HIDE_DISTANCE = 0.15

# Path visualization defaults
PATH_MODE_DEFAULT = "poses_line"  # poses | poses_line | animated | planes
PATH_MARKER_SPACING_M = 0.50
PATH_LINE_SAMPLE_SPACING_M = 0.15
PATH_ANIM_SPEED = 2.5  # m/s
PATH_ANIM_INSTANCES = 3  # number of animated ghosts shown along the path
PATH_ANIM_LINE_ENABLED = True
PATH_LINE_COLOR = (1.0, 0.55, 0.1, 1.0)
PATH_LINE_THICKNESS = 5.0
PATH_LINE_RIBBON_WIDTH_M = 0.11
PATH_LINE_RIBBON_LIFT_M = 0.0
PATH_LINE_RIBBON_ALPHA = 0.38
PATH_GHOST_START_OFFSET_M = 0.90
PATH_GHOST_END_MARGIN_M = 0.05
PATH_GHOST_FRACTION = 0.25  # draw only this fraction of sampled ghost poses (0,1]
PATH_PLANE_SIZE = (1.1, 0.7)  # (width, height) of pose planes in meters
PATH_PLANE_OUTLINE_COLOR = (0.05, 0.95, 1.0, 1.0)
PATH_PLANE_FILL_ALPHA = 0.035
PATH_PLANE_THICKNESS = 5.5
PATH_QUALITY_LABEL_EXCELLENT = "Excellent"
PATH_QUALITY_LABEL_GOOD = "Good"
PATH_QUALITY_LABEL_RISKY = "Risky"
PATH_QUALITY_LABEL_CRITICAL = "Critical"
PATH_QUALITY_THRESH_EXCELLENT = 0.75
PATH_QUALITY_THRESH_GOOD = 0.45
PATH_QUALITY_THRESH_RISKY = 0.20

# Floor height visualization
FLOOR_SHADOW_BASE_RADIUS = 0.15
FLOOR_SHADOW_INNER_RATIO = 0.6
FLOOR_SHADOW_MIN_SCALE = 0.1
FLOOR_SHADOW_MAX_SCALE = 1.6
FLOOR_SHADOW_NEAR_DIST = 1.0
FLOOR_SHADOW_FAR_DIST = 25.0
FLOOR_SHADOW_COLOR = (0.0, 1.0, 1.0, 1.0)
FLOOR_LINE_COLOR = (0.0, 1.0, 1.0, 1.0)
FLOOR_SHADOW_THICKNESS = 3.0
FLOOR_LINE_THICKNESS = 3.0
FLOOR_PROJECTION_ENABLED = True
# Orientation preview (Panda3D overlay)
ORIENT_PREVIEW_ENABLED = True
# Display region: (left, right, bottom, top) in normalized window coords.
ORIENT_PREVIEW_REGION = (-0.02, 0.20, 0.76, 1.04)
ORIENT_PREVIEW_CROP_TOP = 0.07
ORIENT_PREVIEW_BG = (0.05, 0.05, 0.08, 0.95)
ORIENT_PREVIEW_MODEL = "../assets/space_cobot.glb"
ORIENT_PREVIEW_TARGET_SIZE = 1.0
ORIENT_PREVIEW_CAMERA_DISTANCE = 2.5
ORIENT_PREVIEW_CAMERA_HEIGHT = 0.0
ORIENT_PREVIEW_AVATAR_COLOR = AVATAR_COLOR_VISIBLE
ORIENT_PREVIEW_EXTRA_YAW_DEG = 0.0

# UI font (ImGui). If the file is missing, the default ImGui font is used.
UI_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
UI_FONT_SIZE_PX = 24.0
UI_RESPONSE_DELAY_FILL_S = 2.5

# Avatar offsets (camera center relative to robot body; aligned to SDF link pose)
CAMERA_FORWARD_OFFSET_M = 0.07
CAMERA_UP_OFFSET_M = 0.07
# ROS -> Panda: (x, y, z) = (-Y, X, Z). Camera pose is (0.36, 0, 0.13) in ROS.
AVATAR_CAMERA_OFFSET = (0.0, CAMERA_FORWARD_OFFSET_M, CAMERA_UP_OFFSET_M)

# Default camera intrinsics (matching SDF camera)
CAMERA_WIDTH_PX = 1280
CAMERA_HEIGHT_PX = 720
CAMERA_FX = 1108.7654
CAMERA_FY = 1108.7654
CAMERA_CX = 640.0
CAMERA_CY = 360.0

# Key bindings
FORWARD_BUTTON = KeyboardButton.ascii_key("w")
BACKWARD_BUTTON = KeyboardButton.ascii_key("s")
LEFT_BUTTON = KeyboardButton.ascii_key("a")
RIGHT_BUTTON = KeyboardButton.ascii_key("d")
UP_BUTTON = KeyboardButton.ascii_key("e")
DOWN_BUTTON = KeyboardButton.ascii_key("q")
UP_BUTTON_ALT = DOWN_BUTTON
DOWN_BUTTON_ALT = UP_BUTTON
PITCH_UP_BUTTON = KeyboardButton.ascii_key("i")
PITCH_DOWN_BUTTON = KeyboardButton.ascii_key("k")
YAW_LEFT_BUTTON = KeyboardButton.ascii_key("u")
YAW_RIGHT_BUTTON = KeyboardButton.ascii_key("o")
ROLL_LEFT_BUTTON = KeyboardButton.ascii_key("l")
ROLL_RIGHT_BUTTON = KeyboardButton.ascii_key("j")
RESET_ORIENT_BUTTON = KeyboardButton.ascii_key("r")
RESET_TO_ROBOT_ORIENT_BUTTON = KeyboardButton.backspace()
ARROW_UP_BUTTON = KeyboardButton.up()
ARROW_DOWN_BUTTON = KeyboardButton.down()
ARROW_LEFT_BUTTON = KeyboardButton.left()
ARROW_RIGHT_BUTTON = KeyboardButton.right()

# Gamepad control tuning (daemon input)
GAMEPAD_ENABLED = True
GAMEPAD_MOVE_DEADZONE = 0.30
GAMEPAD_LOOK_DEADZONE = 0.40
GAMEPAD_TRIGGER_DEADZONE = 0.10
GAMEPAD_MOVE_CURVE = 1.6
GAMEPAD_LOOK_CURVE = 1.8
GAMEPAD_MOVE_SMOOTHING = 0.2
GAMEPAD_LOOK_SMOOTHING = 0.25
GAMEPAD_TRIGGER_SMOOTHING = 0.2
GAMEPAD_MOVE_SCALE = 1.0
GAMEPAD_VERTICAL_SCALE = 0.6
GAMEPAD_LOOK_SCALE = 1.0
GAMEPAD_AXIS_LOCK_RATIO_LEFT = 0.98
GAMEPAD_AXIS_LOCK_RATIO_RIGHT = 0.25
GAMEPAD_INVERT_PITCH = False
GAMEPAD_INVERT_ROLL = False
GAMEPAD_DPAD_THRESHOLD = 0.5
GAMEPAD_AUTOSCALE_MIN = 0.35
GAMEPAD_AUTOSCALE_MAX_GAIN = 3.0
GAMEPAD_AUTOSCALE_DECAY = 0.98

# Remote gamepad (external process -> ZMQ)
GAMEPAD_REMOTE_ENABLED = True
GAMEPAD_REMOTE_AUTOSTART = True
GAMEPAD_REMOTE_ENDPOINT = "tcp://127.0.0.1:5580"
GAMEPAD_REMOTE_TOPIC = "/gamepad/state"
GAMEPAD_REMOTE_TIMEOUT_S = 1.0


# Optional SpaceMouse daemon autostart.
# The daemon publishes the same remote schema consumed by InputController.
# Keep disabled by default to avoid binding conflicts when gamepad daemon is also running.
SPACEMOUSE_REMOTE_AUTOSTART = False
SPACEMOUSE_REMOTE_ENDPOINT = GAMEPAD_REMOTE_ENDPOINT
SPACEMOUSE_REMOTE_TOPIC = GAMEPAD_REMOTE_TOPIC
# SpaceMouse feel tuning (professional-style defaults: responsive, low friction).
SPACEMOUSE_AXIS_SCALE = 1.50
SPACEMOUSE_VERTICAL_SCALE = 0.55
SPACEMOUSE_TRANSLATION_DEADZONE = 0.04
SPACEMOUSE_ROTATION_DEADZONE = 0.06
SPACEMOUSE_RESPONSE_CURVE = 1.35
SPACEMOUSE_SMOOTHING = 0.12
SPACEMOUSE_INVERT_PITCH = True
SPACEMOUSE_INVERT_YAW = True
SPACEMOUSE_INTENT_MIN = 0.55
SPACEMOUSE_CROSS_DRIFT_MAX = 0.01
SPACEMOUSE_TRANSLATION_AXIS_CLIP_MIN = 0.35
SPACEMOUSE_TRANSLATION_AXIS_CLIP_RATIO = 0.08
SPACEMOUSE_ROTATION_AXIS_CLIP_MIN = 0.35
SPACEMOUSE_ROTATION_AXIS_CLIP_RATIO = 0.08
SPACEMOUSE_MIXED_AXIS_CLIP_MIN = 0.45
SPACEMOUSE_MIXED_AXIS_CLIP_RATIO = 0.08

# Axis names are resolved against daemon-published axes.
GAMEPAD_AXIS_LEFT_X = ("left_x", "lx", "abs_x")
GAMEPAD_AXIS_LEFT_Y = ("left_y", "ly", "abs_y")
GAMEPAD_AXIS_RIGHT_X = ("right_x", "rx", "abs_rx")
GAMEPAD_AXIS_RIGHT_Y = ("right_y", "ry", "abs_ry")
GAMEPAD_AXIS_L2 = ("left_trigger", "lt", "abs_z", "abs_l2")
GAMEPAD_AXIS_R2 = ("right_trigger", "rt", "abs_rz", "abs_r2")
GAMEPAD_AXIS_DPAD_X = ("dpad_x", "abs_hat0x")
GAMEPAD_AXIS_DPAD_Y = ("dpad_y", "abs_hat0y")

# Optional index fallbacks when axis names are unavailable.
# Set to integers (0-based) or None.
GAMEPAD_AXIS_INDEX_LEFT_X = 0
GAMEPAD_AXIS_INDEX_LEFT_Y = 1
GAMEPAD_AXIS_INDEX_RIGHT_X = 2
GAMEPAD_AXIS_INDEX_RIGHT_Y = 3
GAMEPAD_AXIS_INDEX_L2 = 4
GAMEPAD_AXIS_INDEX_R2 = 5
GAMEPAD_AXIS_INDEX_DPAD_X = 6
GAMEPAD_AXIS_INDEX_DPAD_Y = 7

# Button names are resolved against daemon-published buttons.
GAMEPAD_BUTTON_L1 = ("left_shoulder", "l1")
GAMEPAD_BUTTON_R1 = ("right_shoulder", "r1")
GAMEPAD_BUTTON_X = ("face_a", "south", "cross")
GAMEPAD_BUTTON_TRIANGLE = ("face_y", "north", "triangle")
GAMEPAD_BUTTON_TOUCHPAD = ("touchpad",)
GAMEPAD_BUTTON_ABORT = ("face_b", "east", "circle")
GAMEPAD_BUTTON_INDEX_ABORT = 1
GAMEPAD_BUTTON_L3 = ("left_stick", "l3", "thumb_l")
GAMEPAD_BUTTON_R3 = ("right_stick", "r3", "thumb_r")

# Optional index fallbacks when button names are unavailable.
GAMEPAD_BUTTON_INDEX_L1 = 4
GAMEPAD_BUTTON_INDEX_R1 = 5
GAMEPAD_BUTTON_INDEX_X = 0
GAMEPAD_BUTTON_INDEX_TRIANGLE = 3
GAMEPAD_BUTTON_INDEX_TOUCHPAD = 13
GAMEPAD_BUTTON_INDEX_L3 = 10
GAMEPAD_BUTTON_INDEX_R3 = 11
