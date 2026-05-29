from Handvisualizer import HandVisualizer
import cv2
import mediapipe as mp
import numpy as np
import config  # Add this
from Handtracker import HandTracker
from Fpscalculator import FpsCalculator
from Systemcontrol import SystemController
import time

class GestureTVController:
    def __init__(self, frame_width=None, frame_height=None, camera_index=None):
        # Use provided values or fall back to config
        self.FRAME_WIDTH = frame_width if frame_width is not None else config.FRAME_WIDTH
        self.FRAME_HEIGHT = frame_height if frame_height is not None else config.FRAME_HEIGHT
        camera_idx = camera_index if camera_index is not None else config.CAMERA_INDEX
        
        # 1. Initialize SystemController first
        self.system = SystemController()
        self.screen_width, self.screen_height = self.system.get_screen_size()
        
        # 2. Camera
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # 4. Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=config.STATIC_IMAGE_MODE,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # 5. Initialize trackers
        self.hand_tracker = HandTracker(
            (self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), 
            self.system
        )
        self.fps_calculator = FpsCalculator()
        self.visualizer = HandVisualizer()

        
        # Pairing state
        self.paired_user = None
        self.registered_users = {}
        
        # Initialize drawing styles
        self._init_drawing_styles()
    
    def _init_drawing_styles(self):
        """Initialize hand skeleton drawing styles"""
        self.right_custom_connection = self.mp_hands.HAND_CONNECTIONS.union(
            set(config.RIGHT_HAND_EXTRA_CONNECTIONS)
        )
        self.right_connection_style = {
            conn: self.mp_drawing.DrawingSpec(
                color=config.RED_COLOR, 
                thickness=config.CONNECTION_THICKNESS
            ) 
            for conn in self.right_custom_connection
        }
        self.right_landmark_style = {
            i: self.mp_drawing.DrawingSpec(
                color=config.RED_COLOR, 
                thickness=config.LANDMARK_THICKNESS, 
                circle_radius=config.LANDMARK_CIRCLE_RADIUS
            ) 
            for i in range(21)
        }
        
        self.left_custom_connection = self.mp_hands.HAND_CONNECTIONS.union(
            set(config.LEFT_HAND_EXTRA_CONNECTIONS)
        )
        self.left_connection_style = {
            conn: self.mp_drawing.DrawingSpec(
                color=config.BLUE_COLOR, 
                thickness=config.CONNECTION_THICKNESS
            ) 
            for conn in self.left_custom_connection
        }
        self.left_landmark_style = {
            i: self.mp_drawing.DrawingSpec(
                color=config.BLUE_COLOR, 
                thickness=config.LANDMARK_THICKNESS, 
                circle_radius=config.LANDMARK_CIRCLE_RADIUS
            ) 
            for i in range(21)
        }
    
    def cleanup(self):
        """Release resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'hands'):
            self.hands.close()
        if config.ENABLE_DEBUG_PRINTS:
            print("Controller shut down.")

    def read_frame(self):
        """Read and preprocess a frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        return frame
    
    def draw_fps(self, frame):
        """Draw FPS counter on frame"""
        fps = self.fps_calculator.calculate_fps()
        cv2.putText(
            frame, f"FPS: {int(fps)}", 
            config.FPS_POSITION,
            config.FONT, 
            config.FPS_FONT_SCALE, 
            config.FPS_COLOR, 
            config.FPS_FONT_THICKNESS
        )
        return frame
    
    def handle_cursor_control(self, distances, frame):
        """Handle cursor movement using left hand middle finger"""
        try:
            self.screen_width, self.screen_height = self.system.get_screen_size()
            middle_x = distances['middle_x']
            middle_y = distances['middle_y']
            
            if not (0 <= middle_x <= self.FRAME_WIDTH and 0 <= middle_y <= self.FRAME_HEIGHT):
                return frame
            
            if not self.hand_tracker.in_dead_zone(middle_x, middle_y, 
                                                    self.hand_tracker.prev_middle_x, 
                                                    self.hand_tracker.prev_middle_y):
                smoothed_middle_x = self.hand_tracker.apply_ema(middle_x, self.hand_tracker.prev_middle_x)
                smoothed_middle_y = self.hand_tracker.apply_ema(middle_y, self.hand_tracker.prev_middle_y)
                
                if not (np.isfinite(smoothed_middle_x) and np.isfinite(smoothed_middle_y)):
                    return frame
                
                vx, vy = self._calculate_velocity(smoothed_middle_x, smoothed_middle_y)
                self._move_cursor(vx, vy)
                self._draw_cursor_debug(frame, vx, vy, smoothed_middle_x, smoothed_middle_y)
                
                self.hand_tracker.prev_middle_x = smoothed_middle_x
                self.hand_tracker.prev_middle_y = smoothed_middle_y
            else:
                cv2.putText(frame, "Dead Zone Active", 
                           (10, config.INFO_Y_OFFSET * config.DEAD_ZONE_POSITION_Y_OFFSET),
                           config.FONT, config.FONT_SCALE, config.DEAD_ZONE_COLOR, config.FONT_THICKNESS)
        
        except Exception as e:
            print(f"Cursor control error: {e}")
        
        return frame
    
    def _calculate_velocity(self, smoothed_x, smoothed_y):
        """Calculate velocity with non-linear mapping"""
        prev_x = self.hand_tracker.prev_middle_x
        prev_y = self.hand_tracker.prev_middle_y
        
        dx = smoothed_x - (prev_x if prev_x is not None else smoothed_x)
        dy = smoothed_y - (prev_y if prev_y is not None else smoothed_y)
        
        def nl(delta, gamma=config.VELOCITY_GAMMA, gain=config.VELOCITY_GAIN):
            if delta == 0:
                return 0.0
            sign = 1.0 if delta > 0 else -1.0
            return sign * (abs(delta) ** gamma) * gain
        
        vx = nl(dx) * (self.screen_width / self.FRAME_WIDTH)
        vy = nl(dy) * (self.screen_height / self.FRAME_HEIGHT)
        
        vx = np.clip(vx, -config.MAX_CURSOR_STEP, config.MAX_CURSOR_STEP)
        vy = np.clip(vy, -config.MAX_CURSOR_STEP, config.MAX_CURSOR_STEP)
        
        return vx, vy
    
    def _move_cursor(self, vx, vy):
        """Move system cursor by velocity amounts"""
        cur_x, cur_y = self.system.get_cursor_pos()
        target_x = np.clip(cur_x + vx, 0, self.screen_width - 1)
        target_y = np.clip(cur_y + vy, 0, self.screen_height - 1)
        self.system.set_cursor_pos(target_x, target_y)
    
    def _draw_cursor_debug(self, frame, vx, vy, smoothed_x, smoothed_y):
        """Draw cursor debugging information"""
        cv2.putText(frame, f"vx:{vx:.1f} vy:{vy:.1f}", 
                   (10, config.INFO_Y_OFFSET * 5), 
                   config.FONT, 0.5, config.TEXT_COLOR, 1)
        
        est_x = np.interp(smoothed_x, [0, self.FRAME_WIDTH], [0, self.screen_width])
        est_y = np.interp(smoothed_y, [0, self.FRAME_HEIGHT], [0, self.screen_height])
        cv2.putText(frame, f"Est Cursor: ({int(est_x)}, {int(est_y)})", 
                   (10, config.INFO_Y_OFFSET * 6), 
                   config.FONT, 0.5, config.TEXT_COLOR, 1)
    
    def process_frame(self, frame):
        """Main frame processing pipeline"""
        # Safety check
        if frame is None:
            return None
        
        # Draw FPS first
        fps = self.fps_calculator.calculate_fps()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Draw landmarks if hands detected
        if results.multi_hand_landmarks and results.multi_handedness:
            # Process each hand
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_handedness.classification[0].label
                
                # Draw landmarks with bright colors
                if hand_label == "Right":
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        self.right_custom_connection,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 0, 255), thickness=4, circle_radius=5),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2)
                    )
                else:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        self.left_custom_connection,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(255, 0, 0), thickness=4, circle_radius=5),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=2)
                    )
                
                # Handle gestures
                distances = self.hand_tracker.CLICK(hand_landmarks, frame, hand_label)
                
                # Handle cursor control for left hand
                if hand_label == "Left" and distances:
                    frame = self.handle_cursor_control(distances, frame)
        else:
            # Reset smoothing when no hands
            self.hand_tracker.prev_middle_x = None
            self.hand_tracker.prev_middle_y = None
        
        return frame

    def run(self):
        """Main loop"""
        if config.ENABLE_WINDOW:
            cv2.namedWindow(config.WINDOW_NAME, config.WINDOW_NORMAL)
            cv2.resizeWindow(config.WINDOW_NAME, self.FRAME_WIDTH, self.FRAME_HEIGHT)
        
        if config.ENABLE_DEBUG_PRINTS:
            print(f"Gesture TV Controller started")
            print(f"Screen: {self.screen_width}x{self.screen_height}")
            print(f"Frame: {self.FRAME_WIDTH}x{self.FRAME_HEIGHT}")
        
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    print("Failed to grab frame")
                    time.sleep(0.01)
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Only show if we have a valid frame
                if processed_frame is not None and processed_frame.size > 0:
                    if config.ENABLE_WINDOW:
                        cv2.imshow(config.WINDOW_NAME, processed_frame)
                else:
                    # Show original frame if processing fails
                    if config.ENABLE_WINDOW:
                        cv2.imshow(config.WINDOW_NAME, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
