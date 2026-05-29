import cv2
import mediapipe as mp
import numpy as np
import config

class HandVisualizer:
    """Handles all drawing and visualization"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize drawing styles
        self._init_styles()
    
    def _init_styles(self):
        """Initialize drawing styles for both hands"""
        # Right hand (Red)
        self.right_connections = self.mp_hands.HAND_CONNECTIONS.union(
            set(config.RIGHT_HAND_EXTRA_CONNECTIONS)
        )
        self.right_connection_style = {
            conn: self.mp_drawing.DrawingSpec(
                color=config.RED_COLOR, 
                thickness=config.CONNECTION_THICKNESS
            )
            for conn in self.right_connections
        }
        self.right_landmark_style = {
            i: self.mp_drawing.DrawingSpec(
                color=config.RED_COLOR, 
                thickness=config.LANDMARK_THICKNESS, 
                circle_radius=config.LANDMARK_CIRCLE_RADIUS
            )
            for i in range(21)
        }
        
        # Left hand (Blue)
        self.left_connections = self.mp_hands.HAND_CONNECTIONS.union(
            set(config.LEFT_HAND_EXTRA_CONNECTIONS)
        )
        self.left_connection_style = {
            conn: self.mp_drawing.DrawingSpec(
                color=config.BLUE_COLOR, 
                thickness=config.CONNECTION_THICKNESS
            )
            for conn in self.left_connections
        }
        self.left_landmark_style = {
            i: self.mp_drawing.DrawingSpec(
                color=config.BLUE_COLOR, 
                thickness=config.LANDMARK_THICKNESS, 
                circle_radius=config.LANDMARK_CIRCLE_RADIUS
            )
            for i in range(21)
        }
    
    def draw_fps(self, frame, fps):
        """Draw FPS on frame"""
        cv2.putText(
            frame, 
            f"FPS: {int(fps)}", 
            config.FPS_POSITION,
            config.FONT, 
            config.FPS_FONT_SCALE, 
            config.FPS_COLOR, 
            config.FPS_FONT_THICKNESS
        )
    
    def draw_hand(self, frame, hand_landmarks, hand_label):
        """Draw hand skeleton based on label"""
        if hand_label == "Right":
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                self.right_connections,
                self.right_landmark_style,
                self.right_connection_style
            )
        elif hand_label == "Left":
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                self.left_connections,
                self.left_landmark_style,
                self.left_connection_style
            )
    
    def draw_gesture_info(self, frame, distances, hand_label):
        """Draw gesture distance information"""
        base_y = config.INFO_Y_OFFSET if hand_label == "Right" else config.INFO_Y_OFFSET * 3
        
        cv2.putText(
            frame, 
            f"{hand_label} T-I: {distances['thumb_index']:.2f}px", 
            (10, base_y), 
            config.FONT, 
            config.FONT_SCALE, 
            config.TEXT_COLOR, 
            config.FONT_THICKNESS
        )
        cv2.putText(
            frame, 
            f"{hand_label} T-P: {distances['thumb_pinky']:.2f}px", 
            (10, base_y + config.INFO_Y_OFFSET), 
            config.FONT, 
            config.FONT_SCALE, 
            config.TEXT_COLOR, 
            config.FONT_THICKNESS
        )
    
    def draw_cursor_debug(self, frame, vx, vy, smoothed_x, smoothed_y, 
                         frame_width, frame_height, screen_width, screen_height):
        """Draw cursor debugging info"""
        cv2.putText(
            frame, 
            f"vx:{vx:.1f} vy:{vy:.1f}", 
            (10, config.INFO_Y_OFFSET * 5), 
            config.FONT, 
            0.5,  # still a small literal, could be added to config later
            config.TEXT_COLOR, 
            1
        )
        
        est_x = np.interp(smoothed_x, [0, frame_width], [0, screen_width])
        est_y = np.interp(smoothed_y, [0, frame_height], [0, screen_height])
        cv2.putText(
            frame, 
            f"Est Cursor: ({int(est_x)}, {int(est_y)})", 
            (10, config.INFO_Y_OFFSET * 6), 
            config.FONT, 
            0.5, 
            config.TEXT_COLOR, 
            1
        )
    
    def draw_dead_zone(self, frame):
        """Draw dead zone indicator"""
        cv2.putText(
            frame, 
            "Dead Zone Active", 
            (10, config.INFO_Y_OFFSET * config.DEAD_ZONE_POSITION_Y_OFFSET),
            config.FONT, 
            config.FONT_SCALE, 
            config.DEAD_ZONE_COLOR, 
            config.FONT_THICKNESS
        )
