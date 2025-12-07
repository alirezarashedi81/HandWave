# Hand Gesture Mouse Control

A real-time hand gesture recognition system that allows you to control your computer mouse using hand movements and gestures detected through your webcam.

## Features

- **Hand Tracking**: Uses MediaPipe to detect and track hand landmarks in real-time
- **Cursor Control**: Move your mouse cursor by moving your left hand's middle finger
- **Gesture-Based Clicks**:
  - Left hand thumb + index finger: Double-click
  - Left hand thumb + pinky: Click and drag (hold)
  - Right hand gestures: Reserved for future functionality
- **Smooth Movement**: Implements exponential moving average (EMA) smoothing for stable cursor control
- **Dead Zone**: Prevents jittery movements with a configurable threshold
- **Visual Feedback**: Real-time display of hand skeleton and gesture distances
- **Performance Monitoring**: FPS counter to monitor system performance

## Requirements

```
opencv-python
mediapipe
pygame
numpy
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/alirezarashedi81/hand-gesture-mouse-control.git
cd hand-gesture-mouse-control
```

2. Install the required dependencies:
```bash
Python 3.10
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python hand_mouse_control.py
```

2. Position your left hand in front of the webcam

3. Control the mouse:
   - **Move cursor**: Move your left hand to control the cursor position
   - **Double-click**: Touch your left thumb to your left index finger
   - **Click and drag**: Touch your left thumb to your left pinky finger (hold to drag, release to drop)

4. Press `q` to quit the application

## Configuration

You can adjust various parameters in the code:

- `FRAME_WIDTH`, `FRAME_HEIGHT`: Webcam resolution (default: 320x240)
- `alpha`: EMA smoothing factor (default: 0.12, lower = smoother but slower response)
- `DEAD_ZONE_THRESHOLD`: Minimum movement to register (default: 3 pixels)
- `GAMMA`: Non-linear velocity mapping exponent (default: 1.0)
- `GAIN`: Overall sensitivity multiplier (default: 2.0)
- `MAX_STEP`: Maximum cursor movement per frame (default: 40 pixels)

## How It Works

1. **Hand Detection**: MediaPipe Hands detects hand landmarks in each frame
2. **Gesture Recognition**: Calculates distances between finger tips to detect gestures
3. **Cursor Movement**: Tracks the left hand's middle finger tip position and translates it to cursor coordinates
4. **Smoothing**: Applies exponential moving average to reduce jitter
5. **Mouse Events**: Uses Windows API (ctypes) to control system cursor and generate mouse clicks

## Limitations

- **Windows Only**: Currently uses Windows-specific APIs for mouse control
- **Single Hand**: Optimized for tracking one hand at a time
- **Lighting Conditions**: Performance depends on good lighting for accurate hand detection
- **CPU Intensive**: Real-time video processing may impact system performance

## Future Enhancements

- Cross-platform support (macOS, Linux)
- Right-hand gesture functionality
- Customizable gesture mappings
- Scrolling gestures
- Configuration file for easy parameter adjustment

## Troubleshooting

**Low FPS**: Reduce webcam resolution or close other applications

**Jittery cursor**: Decrease `alpha` value for more smoothing or increase `DEAD_ZONE_THRESHOLD`

**Cursor too fast/slow**: Adjust `GAIN` and `GAMMA` parameters

**Hand not detected**: Ensure good lighting and hand is within webcam view

## License

MIT License - feel free to use and modify as needed

## Acknowledgments

- Built with [MediaPipe](https://google.github.io/mediapipe/) by Google
- Uses [OpenCV](https://opencv.org/) for video processing
