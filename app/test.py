# test_gesturetv.py
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
import config
from Fpscalculator import FpsCalculator
from Handtracker import HandTracker
from Handvisualizer import HandVisualizer


# ============================================
# Shared landmark helpers
# ============================================

class Landmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
    def HasField(self, name):
        return False


class HandLandmarks:
    def __init__(self, positions):
        self.landmark = [Landmark(x, y) for x, y in positions]


DEFAULT_POSITIONS = [
    (0.5, 0.5),                                          # Wrist (0)
    (0.4, 0.3), (0.3, 0.2), (0.2, 0.15), (0.1, 0.12),  # Thumb  (1-4)
    (0.6, 0.3), (0.7, 0.2), (0.8, 0.15), (0.9, 0.12),  # Index  (5-8)
    (0.6, 0.5), (0.7, 0.5), (0.8, 0.5),  (0.9, 0.5),   # Middle (9-12)
    (0.6, 0.7), (0.7, 0.7), (0.8, 0.7),  (0.9, 0.7),   # Ring   (13-16)
    (0.6, 0.9), (0.7, 0.9), (0.8, 0.9),  (0.9, 0.9),   # Pinky  (17-20)
]


def make_landmarks(thumb_x, thumb_y,
                   index_x=0.5, index_y=0.5,
                   middle_x=0.5, middle_y=0.5,
                   pinky_x=0.5, pinky_y=0.5):
    """Build a 21-landmark mock with specific tip positions."""
    positions = list(DEFAULT_POSITIONS)
    positions[config.THUMB_TIP]         = (thumb_x,  thumb_y)
    positions[config.INDEX_FINGER_TIP]  = (index_x,  index_y)
    positions[config.MIDDLE_FINGER_TIP] = (middle_x, middle_y)
    positions[config.PINKY_TIP]         = (pinky_x,  pinky_y)
    return HandLandmarks(positions)


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def mock_system():
    system = MagicMock()
    system.double_click = MagicMock()
    system.mouse_down   = MagicMock()
    system.mouse_up     = MagicMock()
    system.get_cursor_pos.return_value  = (500, 500)
    system.get_screen_size.return_value = (1920, 1080)
    return system


@pytest.fixture
def hand_tracker(mock_system):
    return HandTracker(frame_shape=(480, 720, 3), system_controller=mock_system)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 720, 3), dtype=np.uint8)


@pytest.fixture
def spread_landmarks():
    """Default spread hand — no pinch."""
    return HandLandmarks(DEFAULT_POSITIONS)


# ============================================
# FpsCalculator
# ============================================

class TestFpsCalculator:
    def test_initial_fps_is_zero(self):
        calc = FpsCalculator()
        assert calc.get_fps() == 0

    def test_calculate_fps_returns_non_negative(self):
        calc = FpsCalculator()
        fps = calc.calculate_fps()
        assert isinstance(fps, float)
        assert fps >= 0.0

    def test_multiple_calls_update_fps(self):
        import time
        calc = FpsCalculator()
        calc.calculate_fps()
        time.sleep(0.1)
        fps = calc.calculate_fps()
        assert fps > 0


# ============================================
# HandTracker — pure-Python helpers
# ============================================

class TestHandTrackerHelpers:
    def test_apply_ema_first_value(self, hand_tracker):
        assert hand_tracker.apply_ema(100, None) == 100.0

    def test_apply_ema_smoothing(self, hand_tracker):
        prev, current = 50.0, 60.0
        alpha    = hand_tracker.alpha
        expected = alpha * current + (1 - alpha) * prev
        assert abs(hand_tracker.apply_ema(current, prev) - expected) < 0.001

    def test_in_dead_zone_no_previous(self, hand_tracker):
        assert hand_tracker.in_dead_zone(10, 10, None, None) is False

    def test_in_dead_zone_within_threshold(self, hand_tracker):
        # distance ~2.83 < threshold 3
        assert hand_tracker.in_dead_zone(0, 0, 2, 2) is True

    def test_in_dead_zone_outside_threshold(self, hand_tracker):
        # distance 5 > threshold 3
        assert hand_tracker.in_dead_zone(0, 0, 3, 4) is False


# ============================================
# HandTracker — dead zone source
# ============================================

class TestDeadZoneSource:
    """
    Verify that the dead-zone flag in CLICK() comes from the C++ ClickProcessor,
    not from the Python in_dead_zone() helper.

    The ClickProcessor tracks the raw middle-finger pixel position between calls.
    On the very first call prev_raw is uninitialised (-1), so dead==False.
    On the second call with the same position, distance==0 < threshold, so dead==True.
    """

    def test_first_call_not_in_dead_zone(self, hand_tracker, blank_frame):
        lm = make_landmarks(0.1, 0.1, middle_x=0.5, middle_y=0.5)
        result = hand_tracker.CLICK(lm, blank_frame, "Left")
        # First call: C++ has no previous raw position yet → not dead
        assert result['in_dead_zone'] is False

    def test_second_call_same_position_is_dead_zone(self, hand_tracker, blank_frame):
        lm = make_landmarks(0.1, 0.1, middle_x=0.5, middle_y=0.5)
        hand_tracker.CLICK(lm, blank_frame, "Left")
        result = hand_tracker.CLICK(lm, blank_frame, "Left")
        # Second call: same raw position → distance 0 < threshold → dead
        assert result['in_dead_zone'] is True

    def test_large_movement_exits_dead_zone(self, hand_tracker, blank_frame):
        lm_still = make_landmarks(0.1, 0.1, middle_x=0.5, middle_y=0.5)
        lm_moved = make_landmarks(0.1, 0.1, middle_x=0.9, middle_y=0.9)
        hand_tracker.CLICK(lm_still, blank_frame, "Left")
        hand_tracker.CLICK(lm_still, blank_frame, "Left")   # now in dead zone
        result = hand_tracker.CLICK(lm_moved, blank_frame, "Left")
        # Large movement → distance >> threshold → not dead
        assert result['in_dead_zone'] is False

    def test_reset_clears_dead_zone_state(self, hand_tracker, blank_frame):
        lm = make_landmarks(0.1, 0.1, middle_x=0.5, middle_y=0.5)
        hand_tracker.CLICK(lm, blank_frame, "Left")
        hand_tracker.reset_smoothing()
        result = hand_tracker.CLICK(lm, blank_frame, "Left")
        # After reset, C++ prev_raw is -1 again → not dead on first call
        assert result['in_dead_zone'] is False


# ============================================
# HandTracker — per-hand isolated processors
# ============================================

class TestPerHandIsolation:
    """
    Each hand has its own ClickProcessor.
    Calling CLICK() for one hand must not affect the smoothed state of the other.
    """

    def test_right_and_left_processors_are_independent(self, hand_tracker):
        assert hand_tracker._processors["Right"] is not hand_tracker._processors["Left"]

    def test_left_hand_state_unaffected_by_right_calls(self, hand_tracker, blank_frame):
        lm_right = make_landmarks(0.1, 0.1, middle_x=0.2, middle_y=0.2)
        lm_left  = make_landmarks(0.9, 0.9, middle_x=0.8, middle_y=0.8)

        hand_tracker.CLICK(lm_right, blank_frame, "Right")
        hand_tracker.CLICK(lm_right, blank_frame, "Right")

        # Left processor has never been called; its smoothed_mx should still be -1
        assert hand_tracker._processors["Left"].smoothed_mx < 0

        # Now call left — first call, so not in dead zone
        result = hand_tracker.CLICK(lm_left, blank_frame, "Left")
        assert result['in_dead_zone'] is False

    def test_prev_smooth_tracked_per_hand(self, hand_tracker, blank_frame):
        lm_r = make_landmarks(0.1, 0.1, middle_x=0.3, middle_y=0.3)
        lm_l = make_landmarks(0.9, 0.9, middle_x=0.7, middle_y=0.7)

        hand_tracker.CLICK(lm_r, blank_frame, "Right")
        hand_tracker.CLICK(lm_l, blank_frame, "Left")

        prev_r = hand_tracker._prev_smooth["Right"]
        prev_l = hand_tracker._prev_smooth["Left"]
        # The two hands updated different entries
        assert prev_r != prev_l


# ============================================
# HandTracker — CLICK return dict
# ============================================

class TestCLICKReturnDict:
    def test_return_keys_present(self, hand_tracker, blank_frame, spread_landmarks):
        result = hand_tracker.CLICK(spread_landmarks, blank_frame, "Right")
        for key in ('thumb_index', 'thumb_pinky', 'middle_x', 'middle_y',
                    'smoothed_mx', 'smoothed_my', 'prev_mx', 'prev_my', 'in_dead_zone'):
            assert key in result, f"Missing key: {key}"

    def test_prev_mx_none_on_first_call(self, hand_tracker, blank_frame, spread_landmarks):
        result = hand_tracker.CLICK(spread_landmarks, blank_frame, "Right")
        assert result['prev_mx'] is None
        assert result['prev_my'] is None

    def test_prev_mx_populated_on_second_call(self, hand_tracker, blank_frame, spread_landmarks):
        hand_tracker.CLICK(spread_landmarks, blank_frame, "Right")
        result = hand_tracker.CLICK(spread_landmarks, blank_frame, "Right")
        assert result['prev_mx'] is not None
        assert result['prev_my'] is not None

    def test_unknown_hand_label_returns_none(self, hand_tracker, blank_frame, spread_landmarks):
        result = hand_tracker.CLICK(spread_landmarks, blank_frame, "Both")
        assert result is None


# ============================================
# HandTracker — click / drag gestures
# ============================================

class TestCLICKGestures:
    def test_no_pinch_no_click(self, hand_tracker, mock_system, blank_frame, spread_landmarks):
        hand_tracker.CLICK(spread_landmarks, blank_frame, "Right")
        mock_system.double_click.assert_not_called()
        mock_system.mouse_down.assert_not_called()
        mock_system.mouse_up.assert_not_called()

    def test_thumb_index_pinch_triggers_double_click(self, hand_tracker, mock_system, blank_frame):
        # Both at same normalised position → distance 0 < PINCH_THRESHOLD
        lm = make_landmarks(thumb_x=0.5, thumb_y=0.5, index_x=0.5, index_y=0.5)
        hand_tracker.CLICK(lm, blank_frame, "Right")
        mock_system.double_click.assert_called_once_with('left')
        assert hand_tracker.right_thumb_index_touch is True

    def test_thumb_index_release_clears_flag(self, hand_tracker, mock_system, blank_frame):
        pinch   = make_landmarks(0.5, 0.5, index_x=0.5, index_y=0.5)
        release = make_landmarks(0.1, 0.1, index_x=0.9, index_y=0.9)
        hand_tracker.CLICK(pinch,   blank_frame, "Right")
        hand_tracker.CLICK(release, blank_frame, "Right")
        assert hand_tracker.right_thumb_index_touch is False

    def test_thumb_pinky_pinch_triggers_mouse_down(self, hand_tracker, mock_system, blank_frame):
        lm = make_landmarks(thumb_x=0.5, thumb_y=0.5, pinky_x=0.5, pinky_y=0.5)
        hand_tracker.CLICK(lm, blank_frame, "Right")
        mock_system.mouse_down.assert_called_once_with('left')
        assert hand_tracker.right_thumb_pinky_touch is True

    def test_thumb_pinky_release_triggers_mouse_up(self, hand_tracker, mock_system, blank_frame):
        pinch   = make_landmarks(0.5, 0.5, pinky_x=0.5, pinky_y=0.5)
        release = make_landmarks(0.1, 0.1, pinky_x=0.9, pinky_y=0.9)
        hand_tracker.CLICK(pinch,   blank_frame, "Right")
        hand_tracker.CLICK(release, blank_frame, "Right")
        mock_system.mouse_up.assert_called_once_with('left')
        assert hand_tracker.right_thumb_pinky_touch is False

    def test_left_hand_no_mouse_events(self, hand_tracker, mock_system, blank_frame):
        # Left hand gestures are currently disabled — no mouse calls expected
        lm = make_landmarks(0.5, 0.5, index_x=0.5, index_y=0.5,
                            pinky_x=0.5, pinky_y=0.5)
        hand_tracker.CLICK(lm, blank_frame, "Left")
        mock_system.double_click.assert_not_called()
        mock_system.mouse_down.assert_not_called()


# ============================================
# HandTracker — reset_smoothing
# ============================================

class TestResetSmoothing:
    def test_reset_clears_prev_smooth(self, hand_tracker, blank_frame, spread_landmarks):
        hand_tracker.CLICK(spread_landmarks, blank_frame, "Right")
        hand_tracker.CLICK(spread_landmarks, blank_frame, "Left")
        hand_tracker.reset_smoothing()
        assert hand_tracker._prev_smooth["Right"] == (None, None)
        assert hand_tracker._prev_smooth["Left"]  == (None, None)

    def test_reset_clears_processor_state(self, hand_tracker, blank_frame, spread_landmarks):
        hand_tracker.CLICK(spread_landmarks, blank_frame, "Right")
        hand_tracker.reset_smoothing()
        assert hand_tracker._processors["Right"].smoothed_mx < 0
        assert hand_tracker._processors["Left"].smoothed_mx  < 0


# ============================================
# HandVisualizer
# ============================================

class TestHandVisualizer:
    @pytest.fixture
    def visualizer(self):
        return HandVisualizer()

    @pytest.fixture
    def blank(self):
        return np.zeros((480, 720, 3), dtype=np.uint8)

    @pytest.fixture
    def dummy_landmarks(self):
        return HandLandmarks(DEFAULT_POSITIONS)

    def test_draw_fps_draws_on_frame(self, visualizer, blank):
        visualizer.draw_fps(blank, 30)
        assert np.any(blank != 0)

    def test_draw_hand_right_no_error(self, visualizer, blank, dummy_landmarks):
        visualizer.draw_hand(blank, dummy_landmarks, "Right")

    def test_draw_hand_left_no_error(self, visualizer, blank, dummy_landmarks):
        visualizer.draw_hand(blank, dummy_landmarks, "Left")

    def test_draw_gesture_info_no_error(self, visualizer, blank):
        distances = {'thumb_index': 10.5, 'thumb_pinky': 20.3,
                     'middle_x': 360, 'middle_y': 240}
        visualizer.draw_gesture_info(blank, distances, "Right")
        visualizer.draw_gesture_info(blank, distances, "Left")

    def test_draw_cursor_debug_draws_on_frame(self, visualizer, blank):
        visualizer.draw_cursor_debug(blank, 5.0, -3.0, 360, 240, 720, 480, 1920, 1080)
        assert np.any(blank != 0)

    def test_draw_dead_zone_draws_on_frame(self, visualizer, blank):
        visualizer.draw_dead_zone(blank)
        assert np.any(blank != 0)


# ============================================
# Config sanity checks
# ============================================

class TestConfig:
    def test_required_attributes_exist(self):
        for attr in ('PINCH_THRESHOLD', 'RELEASE_THRESHOLD', 'EMA_ALPHA',
                     'DEAD_ZONE_THRESHOLD', 'THUMB_TIP', 'INDEX_FINGER_TIP',
                     'MIDDLE_FINGER_TIP', 'PINKY_TIP'):
            assert hasattr(config, attr), f"config missing: {attr}"

    def test_threshold_values(self):
        assert config.PINCH_THRESHOLD   == 10
        assert config.RELEASE_THRESHOLD == 15
        assert config.EMA_ALPHA         == 0.12
        assert config.DEAD_ZONE_THRESHOLD == 3
