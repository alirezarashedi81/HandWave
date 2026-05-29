import ctypes
from ctypes import wintypes
import time
import pygame
import config

class SystemController:
    def __init__(self):
        if config.ENABLE_PYGAME:
            pygame.init()
            try:
                pygame.display.set_mode((1, 1), pygame.HIDDEN)
            except Exception:
                pass
        
        self.user32 = ctypes.windll.user32
        self.user32.SetProcessDPIAware()
        
        self.MOUSEEVENTF_LEFTDOWN = 0x0002
        self.MOUSEEVENTF_LEFTUP = 0x0004
        self.MOUSEEVENTF_RIGHTDOWN = 0x0008
        self.MOUSEEVENTF_RIGHTUP = 0x0010
    
    # ... (other methods stay the same) ...
    
    def click(self, button='left'):
        self.mouse_down(button)
        time.sleep(config.CLICK_DELAY)  # Use config
        self.mouse_up(button)
    
    def double_click(self, button='left'):
        self.click(button)
        time.sleep(config.DOUBLE_CLICK_DELAY)  # Use config
        self.click(button)
