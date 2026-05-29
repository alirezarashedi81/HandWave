import cv2
import numpy as np
import config

class HandTracker:
    def __init__(self, frame_shape, system_controller):
        self.height, self.width, _ = frame_shape
        self.system = system_controller
        self.right_thumb_index_touch = False
        self.right_thumb_pinky_touch = False
        self.left_thumb_index_touch = False
        self.left_thumb_pinky_touch = False
        self.alpha = config.EMA_ALPHA
        self.prev_middle_x = None
        self.prev_middle_y = None
        self.DEAD_ZONE_THRESHOLD = config.DEAD_ZONE_THRESHOLD

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
        thumb = hand_landmarks.landmark[config.THUMB_TIP]
        index = hand_landmarks.landmark[config.INDEX_FINGER_TIP]
        middle = hand_landmarks.landmark[config.MIDDLE_FINGER_TIP]
        pinky = hand_landmarks.landmark[config.PINKY_TIP]
        
        thumb_x, thumb_y = int(thumb.x * self.width), int(thumb.y * self.height)
        index_x, index_y = int(index.x * self.width), int(index.y * self.height)
        middle_x, middle_y = int(middle.x * self.width), int(middle.y * self.height)
        pinky_x, pinky_y = int(pinky.x * self.width), int(pinky.y * self.height)
        
        dist_thumb_index = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
        dist_thumb_pinky = ((thumb_x - pinky_x) ** 2 + (thumb_y - pinky_y) ** 2) ** 0.5

        if hand_label == "Right":
            if dist_thumb_index < config.PINCH_THRESHOLD and not self.right_thumb_index_touch:
                try:
                    self.system.double_click('left')
                    self.right_thumb_index_touch = True
                except Exception as e:
                    print(f"Right Hand Error: {e}")
            elif dist_thumb_index > config.RELEASE_THRESHOLD:
                self.right_thumb_index_touch = False
        
            if dist_thumb_pinky < config.PINCH_THRESHOLD and not self.right_thumb_pinky_touch:
                try:
                    self.system.mouse_down('left')
                    self.right_thumb_pinky_touch = True
                except Exception as e:
                    print(f"Right Hand Error: {e}")
            elif dist_thumb_pinky > config.RELEASE_THRESHOLD and self.right_thumb_pinky_touch:
                try:
                    self.system.mouse_up('left')
                except Exception as e:
                    print(f"Right Hand Error: {e}")
                self.right_thumb_pinky_touch = False

        # Uncomment if you want left hand gestures later
        # if hand_label == "Left":
        #     if dist_thumb_index < 15 and not self.left_thumb_index_touch:
        #         try:
        #             self.system.double_click('left')
        #             self.left_thumb_index_touch = True
        #         except Exception as e:
        #             print(f"Left Hand Error: {e}")
        #     elif dist_thumb_index > 20:
        #         self.left_thumb_index_touch = False
        #     
        #     if dist_thumb_pinky < 10 and not self.left_thumb_pinky_touch:
        #         try:
        #             self.system.mouse_down('left')
        #             self.left_thumb_pinky_touch = True
        #         except Exception as e:
        #             print(f"Left Hand Error: {e}")
        #     elif dist_thumb_pinky > 15 and self.left_thumb_pinky_touch:
        #         try:
        #             self.system.mouse_up('left')
        #         except Exception as e:
        #             print(f"Left Hand Error: {e}")
        #         self.left_thumb_pinky_touch = False

        y_offset = config.INFO_Y_OFFSET if hand_label == "Right" else config.INFO_Y_OFFSET * 3
        cv2.putText(frame, f"{hand_label} T-I: {dist_thumb_index:.2f}px", (10, y_offset),
                    config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
        cv2.putText(frame, f"{hand_label} T-P: {dist_thumb_pinky:.2f}px", (10, y_offset + config.INFO_Y_OFFSET),
                    config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
        
        return {'thumb_index': dist_thumb_index, 'thumb_pinky': dist_thumb_pinky, 'middle_x': middle_x, 'middle_y': middle_y}
