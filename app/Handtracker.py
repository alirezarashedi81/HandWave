import cv2
import numpy as np
import config
import os
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
click_utils_path = os.path.join(current_dir, "Click_utils")
if click_utils_path not in sys.path:
    sys.path.insert(0, click_utils_path)
import click_utils


class HandTracker:
    def __init__(self, frame_shape, system_controller):
        self.height, self.width, _ = frame_shape
        self.system = system_controller

        # Click state flags — one per hand
        self.right_thumb_index_touch = False
        self.right_thumb_pinky_touch = False
        self.left_thumb_index_touch  = False
        self.left_thumb_pinky_touch  = False

        # One independent ClickProcessor per hand so their smoothed-position
        # state never overwrites each other when both hands are visible.
        self._processors = {
            "Right": click_utils.ClickProcessor(
                alpha=config.EMA_ALPHA,
                dead_zone_threshold=config.DEAD_ZONE_THRESHOLD,
            ),
            "Left": click_utils.ClickProcessor(
                alpha=config.EMA_ALPHA,
                dead_zone_threshold=config.DEAD_ZONE_THRESHOLD,
            ),
        }

        # Per-hand previous-frame smoothed position snapshots
        self._prev_smooth = {
            "Right": (None, None),
            "Left":  (None, None),
        }

        self.alpha = config.EMA_ALPHA
        self.DEAD_ZONE_THRESHOLD = config.DEAD_ZONE_THRESHOLD

    # ------------------------------------------------------------------
    # Pure-Python helpers (kept for unit-test compatibility)
    # ------------------------------------------------------------------

    def apply_ema(self, current, previous):
        if previous is None or not np.isfinite(previous):
            return float(current)
        return float(self.alpha * current + (1 - self.alpha) * previous)

    def in_dead_zone(self, current_x, current_y, prev_x, prev_y):
        if prev_x is None or prev_y is None:
            return False
        distance = ((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2) ** 0.5
        return distance < self.DEAD_ZONE_THRESHOLD

    # Legacy aliases (Gesturetv still references these for the left-hand cursor)
    @property
    def prev_middle_x(self):
        v = self._processors["Left"].smoothed_mx
        return None if v < 0 else v

    @prev_middle_x.setter
    def prev_middle_x(self, v):
        pass  # state lives in ClickProcessor

    @property
    def prev_middle_y(self):
        v = self._processors["Left"].smoothed_my
        return None if v < 0 else v

    @prev_middle_y.setter
    def prev_middle_y(self, v):
        pass

    def reset_smoothing(self):
        """Reset both processors and prev snapshots (call when no hands detected)."""
        for proc in self._processors.values():
            proc.reset()
        self._prev_smooth = {"Right": (None, None), "Left": (None, None)}

    # ------------------------------------------------------------------
    # Core method — one processor per hand, isolated state
    # ------------------------------------------------------------------

    def CLICK(self, hand_landmarks, frame, hand_label):
        """
        Process one hand per frame using its own dedicated ClickProcessor.

        Returns dict:
            thumb_index, thumb_pinky  – pixel distances
            middle_x, middle_y        – raw middle fingertip pixels
            smoothed_mx, smoothed_my  – EMA-smoothed position (this frame)
            prev_mx, prev_my          – EMA-smoothed position (previous frame)
            in_dead_zone              – bool
        """
        proc = self._processors.get(hand_label)
        if proc is None:
            return None

        lm = hand_landmarks.landmark

        thumb  = lm[config.THUMB_TIP]
        index  = lm[config.INDEX_FINGER_TIP]
        middle = lm[config.MIDDLE_FINGER_TIP]
        pinky  = lm[config.PINKY_TIP]

        # Snapshot previous smoothed position BEFORE process() updates it
        prev_mx, prev_my = self._prev_smooth[hand_label]

        # C++ processor: pixel conversion, distances, dead-zone, EMA update
        dist_thumb_index, dist_thumb_pinky, in_dead_zone = proc.process(
            self.width,  self.height,
            thumb.x,  thumb.y,
            index.x,  index.y,
            middle.x, middle.y,
            pinky.x,  pinky.y,
        )

        # Read back freshly updated smoothed position
        smoothed_mx = proc.smoothed_mx
        smoothed_my = proc.smoothed_my

        # Save for next frame
        self._prev_smooth[hand_label] = (smoothed_mx, smoothed_my)

        # Raw pixel coords for bounds check upstream
        middle_x = int(middle.x * self.width)
        middle_y = int(middle.y * self.height)

        # --- gesture events ---
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

        # Overlay text
        y_offset = config.INFO_Y_OFFSET if hand_label == "Right" else config.INFO_Y_OFFSET * 3
        cv2.putText(frame, f"{hand_label} T-I: {dist_thumb_index:.2f}px",
                    (10, y_offset),
                    config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
        cv2.putText(frame, f"{hand_label} T-P: {dist_thumb_pinky:.2f}px",
                    (10, y_offset + config.INFO_Y_OFFSET),
                    config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)

        return {
            'thumb_index':  dist_thumb_index,
            'thumb_pinky':  dist_thumb_pinky,
            'middle_x':     middle_x,
            'middle_y':     middle_y,
            'smoothed_mx':  smoothed_mx,
            'smoothed_my':  smoothed_my,
            'prev_mx':      prev_mx,
            'prev_my':      prev_my,
            'in_dead_zone': bool(in_dead_zone),
        }
