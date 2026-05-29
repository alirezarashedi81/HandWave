import cv2
import mediapipe as mp
import winsound
import time
import pygame
import ctypes
from ctypes import wintypes
import numpy as np

class FpsCalculator:
    """Simple FPS calculator"""
    
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        time_diff = current_time - self.prev_time
        self.fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_time = current_time
        return self.fps
    
    def get_fps(self):
        """Get last calculated FPS"""
        return self.fps
