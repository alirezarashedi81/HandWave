# test_gesturetv.py
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch, call
import config
from Fpscalculator import FpsCalculator
from Handtracker import HandTracker
from Handvisualizer import HandVisualizer

# ============================================
# Fixtures
# ============================================
@pytest.fixture
def mock_system():
    """Create a mock SystemController to avoid real mouse calls."""
    system = MagicMock()
    system.double_click = MagicMock()
    system.mouse_down = MagicMock()
    system.mouse_up = MagicMock()
    system.get_cursor_pos.return_value = (500, 500)
    system.get_screen_size.return_value = (1920, 1080)
    return system

@pytest.fixture
def hand_tracker(mock_system):
    """Create a HandTracker with a mock system controller."""
    return HandTracker(frame_shape=(480, 720, 3), system_controller=mock_system)

@pytest.fixture
def blank_frame():
    """Provide a blank black image as a dummy frame."""
    return np.zeros((480, 720, 3), dtype=np.uint8)

@pytest.fixture
def dummy_hand_landmarks():
    """Create a mock hand landmarks object with configurable positions."""
    class Landmark:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z
        def HasField(self, name):       # required by mediapipe drawing utils
            return False

    class HandLandmarks:
        def __init__(self, positions):
            self.landmark = [Landmark(x, y) for x, y in positions]

    # Default positions: all fingers spread wide, no pinch
    default_positions = [
        (0.5, 0.5),  # Wrist (0)
        (0.4, 0.3), (0.3, 0.2), (0.2, 0.15), (0.1, 0.12),  # Thumb (1-4)
        (0.6, 0.3), (0.7, 0.2), (0.8, 0.15), (0.9, 0.12),  # Index (5-8)
        (0.6, 0.5), (0.7, 0.5), (0.8, 0.5), (0.9, 0.5),    # Middle (9-12)
        (0.6, 0.7), (0.7, 0.7), (0.8, 0.7), (0.9, 0.7),    # Ring (13-16)
        (0.6, 0.9), (0.7, 0.9), (0.8, 0.9), (0.9, 0.9),    # Pinky (17-20)
    ]
    return HandLandmarks(default_positions)

# ============================================
# FpsCalculator Tests
# ============================================
class TestFpsCalculator:
    def test_initial_fps_is_zero(self):
        calc = FpsCalculator()
        assert calc.get_fps() == 0

    def test_calculate_fps_returns_value(self):
        calc = FpsCalculator()
        # First call after init may give high FPS; we just check it's a float > 0
        fps = calc.calculate_fps()
        assert isinstance(fps, float)
        assert fps >= 0.0

    def test_multiple_calls_update_fps(self):
        calc = FpsCalculator()
        import time
        calc.calculate_fps()
        time.sleep(0.1)
        fps = calc.calculate_fps()
        # Should be around 10 FPS, but we only check it's not zero
        assert fps > 0

# ============================================
# HandTracker Tests
# ============================================
class TestHandTracker:
    def test_apply_ema_first_value(self, hand_tracker):
        result = hand_tracker.apply_ema(100, None)
        assert result == 100.0

    def test_apply_ema_smoothing(self, hand_tracker):
        prev = 50.0
        current = 60.0
        alpha = hand_tracker.alpha  # 0.12
        expected = alpha * current + (1 - alpha) * prev
        result = hand_tracker.apply_ema(current, prev)
        assert abs(result - expected) < 0.001

    def test_in_dead_zone_no_previous(self, hand_tracker):
        assert hand_tracker.in_dead_zone(10, 10, None, None) is False

    def test_in_dead_zone_within_threshold(self, hand_tracker):
        # threshold is 3, distance ~2.82
        assert hand_tracker.in_dead_zone(0, 0, 2, 2) is True

    def test_in_dead_zone_outside_threshold(self, hand_tracker):
        # distance 5 > 3
        assert hand_tracker.in_dead_zone(0, 0, 3, 4) is False

    def test_CLICK_no_pinch_does_not_trigger(self, hand_tracker, dummy_hand_landmarks, blank_frame):
        """With default spread hand, no click should be called."""
        distances = hand_tracker.CLICK(dummy_hand_landmarks, blank_frame, "Right")
        hand_tracker.system.double_click.assert_not_called()
        hand_tracker.system.mouse_down.assert_not_called()
        hand_tracker.system.mouse_up.assert_not_called()
        assert 'middle_x' in distances

    def test_CLICK_pinch_index_thumb_double_click(self, hand_tracker, mock_system, blank_frame):
        """Bring thumb and index tips very close to simulate a pinch."""
        # Create mock landmarks with thumb tip (4) and index tip (8) at same position
        # We'll manually set the landmark coordinates via a custom mock
        class Landmark:
            def __init__(self, x, y, z=0):
                self.x = x
                self.y = y
                self.z = z
            def HasField(self, name):       # dummy for MediaPipe compatibility
                return False
            
        class MockLandmarks:
            def __init__(self, thumb_x, thumb_y, index_x, index_y):
                self.landmark = [Landmark(0.5,0.5)]*21  # fill all
                # Overwrite specific tips
                self.landmark[config.THUMB_TIP] = Landmark(thumb_x, thumb_y)
                self.landmark[config.INDEX_FINGER_TIP] = Landmark(index_x, index_y)
                self.landmark[config.MIDDLE_FINGER_TIP] = Landmark(0.5,0.5)
                self.landmark[config.PINKY_TIP] = Landmark(0.5,0.5)

        # Pinch: both at same pixel (after scaling to frame size 720x480)
        # Use coordinates that make distance < config.PINCH_THRESHOLD
        # In frame coords: x in [0,720], y in [0,480]
        # Use (360,240) for both
        landmarks = MockLandmarks(0.5, 0.5, 0.5, 0.5)  # both at center
        distances = hand_tracker.CLICK(landmarks, blank_frame, "Right")
        # Should trigger double_click
        mock_system.double_click.assert_called_once_with('left')
        # State flag should be set
        assert hand_tracker.right_thumb_index_touch is True

        # Release by moving apart
        landmarks = MockLandmarks(0.2, 0.5, 0.8, 0.5)  # far apart horizontally
        hand_tracker.CLICK(landmarks, blank_frame, "Right")
        assert hand_tracker.right_thumb_index_touch is False

    def test_CLICK_pinky_thumb_drag(self, hand_tracker, mock_system, blank_frame):
        """Thumb and pinky pinch should trigger mouse_down and then mouse_up on release."""
        class Landmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def HasField(self, name):       # added for safety
                return False
        class MockLandmarks:
            def __init__(self, thumb_x, thumb_y, pinky_x, pinky_y):
                self.landmark = [Landmark(0.5,0.5)]*21
                self.landmark[config.THUMB_TIP] = Landmark(thumb_x, thumb_y)
                self.landmark[config.INDEX_FINGER_TIP] = Landmark(0.5,0.5)
                self.landmark[config.MIDDLE_FINGER_TIP] = Landmark(0.5,0.5)
                self.landmark[config.PINKY_TIP] = Landmark(pinky_x, pinky_y)

        # Pinch: both at same point
        landmarks = MockLandmarks(0.5, 0.5, 0.5, 0.5)
        hand_tracker.CLICK(landmarks, blank_frame, "Right")
        mock_system.mouse_down.assert_called_once_with('left')
        assert hand_tracker.right_thumb_pinky_touch is True

        # Release
        landmarks = MockLandmarks(0.2, 0.5, 0.8, 0.5)  # far
        hand_tracker.CLICK(landmarks, blank_frame, "Right")
        mock_system.mouse_up.assert_called_once_with('left')
        assert hand_tracker.right_thumb_pinky_touch is False

    def test_CLICK_left_hand_ignored(self, hand_tracker, dummy_hand_landmarks, blank_frame):
        """Left hand gestures are currently commented out; ensure no mouse calls."""
        hand_tracker.CLICK(dummy_hand_landmarks, blank_frame, "Left")
        hand_tracker.system.double_click.assert_not_called()
        hand_tracker.system.mouse_down.assert_not_called()

    def test_reset_smoothing(self, hand_tracker):
        hand_tracker.prev_middle_x = 100
        hand_tracker.prev_middle_y = 200
        hand_tracker.reset_smoothing()
        assert hand_tracker.prev_middle_x is None
        assert hand_tracker.prev_middle_y is None

# ============================================
# HandVisualizer Tests
# ============================================
class TestHandVisualizer:
    @pytest.fixture
    def visualizer(self):
        return HandVisualizer()

    def test_draw_fps_no_error(self, visualizer, blank_frame):
        """Just ensure drawing FPS doesn't crash."""
        visualizer.draw_fps(blank_frame, 30)
        # Check that something was drawn (we can check pixel change)
        assert np.any(blank_frame != 0)

    def test_draw_hand_no_error(self, visualizer, blank_frame, dummy_hand_landmarks):
        """Draw left and right hands without crash."""
        visualizer.draw_hand(blank_frame, dummy_hand_landmarks, "Right")
        visualizer.draw_hand(blank_frame, dummy_hand_landmarks, "Left")
        # Not crashing is success

    def test_draw_gesture_info_no_error(self, visualizer, blank_frame):
        distances = {'thumb_index': 10.5, 'thumb_pinky': 20.3,
                     'middle_x': 360, 'middle_y': 240}
        visualizer.draw_gesture_info(blank_frame, distances, "Right")
        visualizer.draw_gesture_info(blank_frame, distances, "Left")
        # Should not crash

    def test_draw_cursor_debug_no_error(self, visualizer, blank_frame):
        visualizer.draw_cursor_debug(blank_frame, 5.0, -3.0,
                                     360, 240, 720, 480, 1920, 1080)
        assert np.any(blank_frame != 0)

    def test_draw_dead_zone_no_error(self, visualizer, blank_frame):
        visualizer.draw_dead_zone(blank_frame)
        assert np.any(blank_frame != 0)

# ============================================
# Config Tests (sanity check)
# ============================================
class TestConfig:
    def test_config_values_exist(self):
        assert hasattr(config, 'PINCH_THRESHOLD')
        assert hasattr(config, 'RELEASE_THRESHOLD')
        assert config.PINCH_THRESHOLD == 10
        assert config.RELEASE_THRESHOLD == 15
        assert config.EMA_ALPHA == 0.12
        assert config.DEAD_ZONE_THRESHOLD == 3

# ============================================
# Optional: Integration test for GestureTVController
# (Skipped because it requires webcam and windows)
# ============================================
# @pytest.mark.skip(reason="Requires webcam and Windows API")
# def test_controller_init():
#     from Gesturetv import GestureTVController
#     # ...
