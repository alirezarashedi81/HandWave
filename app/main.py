# main.py
from Gesturetv import GestureTVController

def main():
    """Entry point for the Gesture TV Controller"""
    try:
        controller = GestureTVController(
            frame_width=720,
            frame_height=480,
            camera_index=0
        )
        controller.run()
    except Exception as e:
        print(f"Error starting controller: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
