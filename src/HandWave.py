
import cv2
import mediapipe as mp
import winsound
import time
import pyautogui
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

class HandTracker:
    def __init__(self, frame_shape):
        self.height, self.width, _ = frame_shape
        self.right_thumb_index_touch = False
        self.right_thumb_pinky_touch = False
        self.left_thumb_index_touch = False
        self.left_thumb_pinky_touch = False
        self.alpha = 0.3
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
            if dist_thumb_index < 15 and not self.left_thumb_index_touch:
                try:
                    winsound.Beep(1000, 100)
                    self.left_thumb_index_touch = True
                except Exception as e:
                    print(f"Left Hand Error: {e}")
            elif dist_thumb_index > 20:

                self.left_thumb_index_touch = False
            
            if dist_thumb_pinky < 10 and not self.left_thumb_pinky_touch:
                try:
                    # winsound.Beep(1500, 100)
                    self.left_thumb_pinky_touch = True
                except Exception as e:
                    print(f"Left Hand Error: {e}")
            elif dist_thumb_pinky > 15:
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
    print("Cannot open webcam")
    exit()

screen_width, screen_height = pyautogui.size()

# Initialize HandTracker and FpsCalculator
hand_tracker = HandTracker((FRAME_HEIGHT, FRAME_WIDTH, 3))
fps_calculator = FpsCalculator()

# Set PyAutoGUI fail-safe
pyautogui.FAILSAFE = True

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
                    screen_width, screen_height = pyautogui.size()
                    middle_x = distances['middle_x']
                    middle_y = distances['middle_y']
                    if not (0 <= middle_x <= FRAME_WIDTH and 0 <= middle_y <= FRAME_HEIGHT):
                        print(f"Invalid raw coordinates: middle_x={middle_x}, middle_y={middle_y}")
                        continue
                    if not hand_tracker.in_dead_zone(middle_x, middle_y, hand_tracker.prev_middle_x, hand_tracker.prev_middle_y):
                        smoothed_middle_x = hand_tracker.apply_ema(middle_x, hand_tracker.prev_middle_x)
                        smoothed_middle_y = hand_tracker.apply_ema(middle_y, hand_tracker.prev_middle_y)
                        hand_tracker.prev_middle_x = smoothed_middle_x
                        hand_tracker.prev_middle_y = smoothed_middle_y
                        if not (np.isfinite(smoothed_middle_x) and np.isfinite(smoothed_middle_y)):
                            print(f"Invalid smoothed coordinates: smoothed_middle_x={smoothed_middle_x}, smoothed_middle_y={smoothed_middle_y}")
                            continue
                        print(f"Raw: ({middle_x:.2f}, {middle_y:.2f}), Smoothed: ({smoothed_middle_x:.2f}, {smoothed_middle_y:.2f})")
                        distance = ((middle_x - (hand_tracker.prev_middle_x or 0)) ** 2 + (middle_y - (hand_tracker.prev_middle_y or 0)) ** 2) ** 0.5
                        print(f"Distance: {distance:.2f}, Dead Zone: {hand_tracker.in_dead_zone(middle_x, middle_y, hand_tracker.prev_middle_x, hand_tracker.prev_middle_y)}")
                        screen_x = np.interp(smoothed_middle_x, [0, FRAME_WIDTH], [0, screen_width])
                        screen_y = np.interp(smoothed_middle_y, [0, FRAME_HEIGHT], [0, screen_height])
                        if not (np.isfinite(screen_x) and np.isfinite(screen_y)):
                            print(f"Invalid screen coordinates: screen_x={screen_x}, screen_y={screen_y}")
                            continue
                        pyautogui.moveTo(screen_x, screen_y)
                        cv2.putText(frame, f"Left Middle Tip: ({int(smoothed_middle_x)}, {int(smoothed_middle_y)})", 
                                    (10, Y_OFFSET * 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
                        cv2.putText(frame, f"Mouse: ({int(screen_x)}, {int(screen_y)})", 
                                    (10, Y_OFFSET * 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
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
