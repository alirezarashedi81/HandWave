import cv2
# ============================================
# CAMERA SETTINGS
# ============================================
CAMERA_INDEX = 0
FRAME_WIDTH = 720
FRAME_HEIGHT = 480
FLIP_HORIZONTAL = True  # Mirror the frame

# ============================================
# MEDIAPIPE SETTINGS
# ============================================
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.4
STATIC_IMAGE_MODE = False

# ============================================
# HAND LANDMARK INDICES
# ============================================
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20
WRIST = 0

# ============================================
# GESTURE THRESHOLDS
# ============================================
# Distance thresholds in pixels for gesture detection
PINCH_THRESHOLD = 10  # Below this = pinch detected
RELEASE_THRESHOLD = 15  # Above this = pinch released

# ============================================
# CURSOR CONTROL SETTINGS
# ============================================
# EMA smoothing
EMA_ALPHA = 0.12  # Lower = smoother but more lag

# Dead zone
DEAD_ZONE_THRESHOLD = 3  # Pixels, minimum movement to register

# Non-linear velocity mapping
VELOCITY_GAMMA = 1.0  # >1: fine control for small movements, fast for large
VELOCITY_GAIN = 4.0   # Overall sensitivity multiplier
MAX_CURSOR_STEP = 40.0  # Maximum pixels per frame

# ============================================
# VISUALIZATION SETTINGS
# ============================================
# Colors (BGR format)
RED_COLOR = (57, 57, 243)      # Right hand skeleton
BLUE_COLOR = (243, 131, 57)    # Left hand skeleton
TEXT_COLOR = (255, 255, 0)     # General text
FPS_COLOR = (0, 255, 0)        # FPS counter
DEAD_ZONE_COLOR = (0, 0, 255)  # Dead zone indicator

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX  # Import cv2 in your files
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FPS_FONT_SCALE = 1
FPS_FONT_THICKNESS = 2

# Text positions
FPS_POSITION = (10, 30)
INFO_Y_OFFSET = 30
DEAD_ZONE_POSITION_Y_OFFSET = 7  # Multiplied by INFO_Y_OFFSET

# Custom hand connections to draw
RIGHT_HAND_EXTRA_CONNECTIONS = [(5, 17), (8, 4), (20, 4)]
LEFT_HAND_EXTRA_CONNECTIONS = [(5, 17)]

# Landmark drawing settings
LANDMARK_THICKNESS = 1
LANDMARK_CIRCLE_RADIUS = 1
CONNECTION_THICKNESS = 1

# ============================================
# MOUSE CLICK SETTINGS
# ============================================
CLICK_DELAY = 0.02  # Seconds between mouse down and up
DOUBLE_CLICK_DELAY = 0.06  # Seconds between clicks

# ============================================
# PAIRING SYSTEM SETTINGS
# ============================================
NUM_REGISTRATION_PHOTOS = 5
PAIRING_TIMEOUT = 10  # Seconds to complete pairing
PAIRING_SIMILARITY_THRESHOLD = 0.8  # Minimum match score (0-1)
TEMPLATE_STORAGE_PATH = "user_templates/"  # Where to save user hand data

# ============================================
# TV CONTROL SETTINGS
# ============================================
# These will be used when switching from mouse control to TV control
TV_VOLUME_STEP = 5  # Percentage per volume gesture
TV_CHANNEL_STEP = 1  # Number of channels per channel gesture

# ============================================
# WINDOW SETTINGS
# ============================================
WINDOW_NAME = "Gesture TV Controller"
WINDOW_NORMAL = cv2.WINDOW_NORMAL  # Import cv2 in your files
ENABLE_WINDOW = True  # Set False for headless operation

# ============================================
# SYSTEM SETTINGS
# ============================================
ENABLE_PYGAME = True  # Some environments don't support pygame
ENABLE_DEBUG_PRINTS = True  # Print debug info to console
