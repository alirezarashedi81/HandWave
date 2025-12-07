
import cv2
import mediapipe as mp
import winsound
import time
import pygame
import ctypes
from ctypes import wintypes
import numpy as np

# Constants
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
PINKY_TIP = 20
FONT_SCALE = 0.5
FONT_THICKNESS = 1
TEXT_COLOR = (255, 255, 0)  # Yellow for text
RED_COLOR = (57, 57, 243)  # Red for right hand skeleton
BLUE_COLOR = (243, 131, 57)  # Blue for left hand skeleton
Y_OFFSET = 30
FRAME_WIDTH, FRAME_HEIGHT = 320, 240

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,

)

# Initialize pygame (optional) and Win32 mouse helpers for system cursor control
pygame.init()
try:
    pygame.display.set_mode((1, 1), pygame.HIDDEN)
except Exception:
    # Some environments may not support hidden windows; ignore if not available
    pass

# Win32 API helpers for cursor and mouse events (works on Windows)
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

def get_screen_size():
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def set_cursor_pos(x, y):
    user32.SetCursorPos(int(x), int(y))

def get_cursor_pos():
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

# Mouse event flags
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

def mouse_down(button='left'):
    if button == 'left':
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    else:
        user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

def mouse_up(button='left'):
    if button == 'left':
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    else:
        user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

def click(button='left'):
    mouse_down(button)
    time.sleep(0.02)
    mouse_up(button)

def double_click(button='left'):
    click(button)
    time.sleep(0.06)
    click(button)

class HandTracker:
    def __init__(self, frame_shape):
        self.height, self.width, _ = frame_shape
        self.right_thumb_index_touch = False
        self.right_thumb_pinky_touch = False
        self.left_thumb_index_touch = False
        self.left_thumb_pinky_touch = False
        self.alpha = 0.12  # EMA smoothing factor
        self.prev_middle_x = None
        self.prev_middle_y = None
        self.DEAD_ZONE_THRESHOLD = 3  # Circular threshold in pixels

    def apply_ema(self, current, previous):
        if previous is None or not np.isfinite(previous):
            return float(current)
        return float(self.alpha * current + (1 - self.alpha) * previous)

    def in_dead_zone(self, current_x, current_y, prev_x, prev_y):
        if prev_x is None or prev_y is None:
            return False
        distance = ((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2) ** 0.5
        return distance < self.DEAD_ZONE_THRESHOLD

    def CLICK(self, hand_landmarks, frame, hand_label):
        thumb = hand_landmarks.landmark[THUMB_TIP]
        index = hand_landmarks.landmark[INDEX_TIP]
        middle = hand_landmarks.landmark[MIDDLE_TIP]
        pinky = hand_landmarks.landmark[PINKY_TIP]
        
        thumb_x, thumb_y = int(thumb.x * self.width), int(thumb.y * self.height)
        index_x, index_y = int(index.x * self.width), int(index.y * self.height)
        middle_x, middle_y = int(middle.x * self.width), int(middle.y * self.height)
        pinky_x, pinky_y = int(pinky.x * self.width), int(pinky.y * self.height)
        
        dist_thumb_index = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
        dist_thumb_pinky = ((thumb_x - pinky_x) ** 2 + (thumb_y - pinky_y) ** 2) ** 0.5

        if hand_label == "Right":
            if dist_thumb_index < 10 and not self.right_thumb_index_touch:
                try:
                    # winsound.Beep(1000, 100)
             
                    self.right_thumb_index_touch = True
                except Exception as e:
                    print(f"Right Hand Error: {e}")
            elif dist_thumb_index > 15:
                self.right_thumb_index_touch = False
        
            if dist_thumb_pinky < 10 and not self.right_thumb_pinky_touch:
                try:
                    # winsound.Beep(1500, 100)
                    self.right_thumb_pinky_touch = True
                except Exception as e:
                    print(f"Right Hand Error: {e}")
            elif dist_thumb_pinky > 15:
                self.right_thumb_pinky_touch = False

        if hand_label == "Left":
            # Left thumb + index => immediate double-click
            if dist_thumb_index < 15 and not self.left_thumb_index_touch:
                try:
                    double_click('left')
                    self.left_thumb_index_touch = True
                except Exception as e:
                    print(f"Left Hand Error: {e}")
            elif dist_thumb_index > 20:
                self.left_thumb_index_touch = False

            # Left thumb + pinky => hold mouse down while touching, release when separated
            if dist_thumb_pinky < 10 and not self.left_thumb_pinky_touch:
                try:
                    mouse_down('left')
                    self.left_thumb_pinky_touch = True
                except Exception as e:
                    print(f"Left Hand Error: {e}")
            elif dist_thumb_pinky > 15 and self.left_thumb_pinky_touch:
                try:
                    mouse_up('left')
                except Exception as e:
                    print(f"Left Hand Error: {e}")
                self.left_thumb_pinky_touch = False

        y_offset = Y_OFFSET if hand_label == "Right" else Y_OFFSET * 3
        cv2.putText(frame, f"{hand_label} T-I: {dist_thumb_index:.2f}px", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(frame, f"{hand_label} T-P: {dist_thumb_pinky:.2f}px", (10, y_offset + Y_OFFSET),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        return {'thumb_index': dist_thumb_index, 'thumb_pinky': dist_thumb_pinky, 'middle_x': middle_x, 'middle_y': middle_y}

class FpsCalculator:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
    
    def calculate_fps(self):
        current_time = time.time()
        time_diff = current_time - self.prev_time
        self.fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_time = current_time
        return self.fps
    
    def get_fps(self):
        return self.fps

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("Cannot qpen webcam")
    exit()

screen_width, screen_height = get_screen_size()

# Initialize HandTracker and FpsCalculator
hand_tracker = HandTracker((FRAME_HEIGHT, FRAME_WIDTH, 3))
fps_calculator = FpsCalculator()

# (pyautogui removed) No fail-safe available; be cautious moving cursor to corners.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    fps = fps_calculator.calculate_fps()
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)

    right_hand_landmarks = None
    left_hand_landmarks = None
    right_hand_score = 0
    left_hand_score = 0
    
    if not (results.multi_hand_landmarks and results.multi_handedness):
        hand_tracker.prev_middle_x = None
        hand_tracker.prev_middle_y = None
        print("No hands detected, resetting EMA.")

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, (hand_landmarks, hand_handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_label = hand_handedness.classification[0].label
            hand_score = hand_handedness.classification[0].score

            if hand_label == "Right":
                right_hand_landmarks = hand_landmarks
                right_hand_score = hand_score
            elif hand_label == "Left":
                left_hand_landmarks = hand_landmarks
                left_hand_score = hand_score

            right_custom_connection = mp_hands.HAND_CONNECTIONS.union({(5, 17), (8, 4), (20, 4)})
            right_connection_style = {conn: mp_drawing.DrawingSpec(color=RED_COLOR, thickness=1)
                                     for conn in right_custom_connection}
            right_landmark_style = {i: mp_drawing.DrawingSpec(color=RED_COLOR, thickness=1, circle_radius=1)
                                    for i in range(21)}
            
            left_custom_connection = mp_hands.HAND_CONNECTIONS.union({(5, 17)})
            left_connection_style = {conn: mp_drawing.DrawingSpec(color=BLUE_COLOR, thickness=1)
                                     for conn in left_custom_connection}
            left_landmark_style = {i: mp_drawing.DrawingSpec(color=BLUE_COLOR, thickness=1, circle_radius=1)
                                   for i in range(21)}
            
            if right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    right_hand_landmarks,
                    right_custom_connection,
                    right_landmark_style,
                    right_connection_style
                )
                x = int(hand_landmarks.landmark[0].x * FRAME_WIDTH)
                y = int(hand_landmarks.landmark[0].y * FRAME_HEIGHT)
                distances = hand_tracker.CLICK(hand_landmarks, frame, hand_label)
            
            if left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    left_hand_landmarks,
                    left_custom_connection,
                    left_landmark_style,
                    left_connection_style
                )
                distances = hand_tracker.CLICK(hand_landmarks, frame, hand_label)
                try:
                    screen_width, screen_height = get_screen_size()
                    middle_x = distances['middle_x']
                    middle_y = distances['middle_y']
                    if not (0 <= middle_x <= FRAME_WIDTH and 0 <= middle_y <= FRAME_HEIGHT):
                        print(f"Invalid raw coordinates: middle_x={middle_x}, middle_y={middle_y}")
                        continue
                    if not hand_tracker.in_dead_zone(middle_x, middle_y, hand_tracker.prev_middle_x, hand_tracker.prev_middle_y):
                        # Smooth the middle tip coordinates (frame space)
                        smoothed_middle_x = hand_tracker.apply_ema(middle_x, hand_tracker.prev_middle_x)
                        smoothed_middle_y = hand_tracker.apply_ema(middle_y, hand_tracker.prev_middle_y)

                        if not (np.isfinite(smoothed_middle_x) and np.isfinite(smoothed_middle_y)):
                            print(f"Invalid smoothed coordinates: smoothed_middle_x={smoothed_middle_x}, smoothed_middle_y={smoothed_middle_y}")
                            continue

                        # Compute frame-space deltas using previous smoothed coords
                        prev_x = hand_tracker.prev_middle_x
                        prev_y = hand_tracker.prev_middle_y
                        dx = smoothed_middle_x - (prev_x if prev_x is not None else smoothed_middle_x)
                        dy = smoothed_middle_y - (prev_y if prev_y is not None else smoothed_middle_y)

                        # Non-linear (power) velocity mapping parameters
                        GAMMA = 1.0   # exponent >1: small deltas -> finer control, large deltas -> faster
                        GAIN = 2.0    # overall sensitivity (tune this down if still fast)

                        def nl(delta, gamma=GAMMA, gain=GAIN):
                            if delta == 0:
                                return 0.0
                            sign = 1.0 if delta > 0 else -1.0
                            return sign * (abs(delta) ** gamma) * gain

                        # Map frame deltas -> screen-pixel velocity
                        vx = nl(dx) * (screen_width / FRAME_WIDTH)
                        vy = nl(dy) * (screen_height / FRAME_HEIGHT)

                        # Cap per-frame movement to avoid jumps
                        MAX_STEP = 40.0
                        if abs(vx) > MAX_STEP:
                            vx = np.sign(vx) * MAX_STEP
                        if abs(vy) > MAX_STEP:
                            vy = np.sign(vy) * MAX_STEP

                        try:
                            cur_x, cur_y = get_cursor_pos()
                            target_x = cur_x + vx
                            target_y = cur_y + vy
                            # Clamp to screen
                            target_x = min(max(target_x, 0), screen_width - 1)
                            target_y = min(max(target_y, 0), screen_height - 1)
                            set_cursor_pos(target_x, target_y)
                        except Exception as e:
                            print(f"Mouse Move Error: {e}")

                        # Display info for tuning
                        try:
                            cv2.putText(frame, f"vx:{vx:.1f} vy:{vy:.1f}", (10, Y_OFFSET * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
                            est_x = np.interp(smoothed_middle_x, [0, FRAME_WIDTH], [0, screen_width])
                            est_y = np.interp(smoothed_middle_y, [0, FRAME_HEIGHT], [0, screen_height])
                            cv2.putText(frame, f"Est Cursor: ({int(est_x)}, {int(est_y)})", (10, Y_OFFSET * 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
                        except Exception:
                            pass

                        # Update stored previous smoothed coordinates for next frame
                        hand_tracker.prev_middle_x = smoothed_middle_x
                        hand_tracker.prev_middle_y = smoothed_middle_y
                    else:
                        cv2.putText(frame, "Dead Zone Active", (10, Y_OFFSET * 7), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
                except (TypeError, ValueError) as e:
                    print(f"Mouse Move Error: {e}")
                except Exception as e:
                    print(f"Unexpected Mouse Move Error: {e}")

    cv2.namedWindow("MediaPipe Hands - Real-Time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MediaPipe Hands - Real-Time", FRAME_WIDTH, FRAME_HEIGHT)
    cv2.imshow("MediaPipe Hands - Real-Time", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
