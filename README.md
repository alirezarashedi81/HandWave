🖐️ Camera Mouse Control (Hand Gesture Interface)
This project uses MediaPipe Hands and OpenCV to create a virtual mouse controller. It allows users to manipulate the system cursor and perform mouse clicks using hand movements and specific finger gestures captured by a webcam.

🌟 Features
Cursor Movement: Controls the system cursor position based on the movement of the Middle Finger Tip of the Left Hand.

Uses Exponential Moving Average (EMA) smoothing to reduce jitter.

Implements a Dead Zone to prevent unwanted micro-movements when the hand is still.

Applies non-linear velocity mapping (power function) for finer control at slow speeds.

Left Double-Click: Performed by touching the Thumb Tip to the Index Finger Tip on the Left Hand.

Left Click & Drag (Hold): Performed by touching the Thumb Tip to the Pinky Finger Tip on the Left Hand. The mouse button remains pressed until the fingers are separated.

Real-time Visualization: Displays the detected hand landmarks, FPS, and gesture status.

🛠️ Prerequisites
Operating System: Windows (due to reliance on Win32 API for cursor control).

Webcam: A functioning webcam is required.

Python: Version 3.x.

📦 Installation
Clone the repository:

Bash

git clone https://github.com/your-username/camera-mouse-control.git
cd camera-mouse-control
Install the required Python packages:

Bash

pip install opencv-python mediapipe pygame numpy
🚀 Usage
Run the main script:

Bash

python your_script_name.py # Replace with the actual file name
A window titled "MediaPipe Hands - Real-Time" will open, displaying the live video feed from your webcam.

Gesture Guide
The system primarily uses the Left Hand for cursor control and clicking.

Action	Hand	Gesture
Cursor Movement	Left	Move the hand freely. Cursor tracks the Middle Finger Tip (Landmark 12).
Left Double-Click	Left	Touch Thumb Tip (4) to Index Finger Tip (8).
Left Click & Drag	Left	Touch Thumb Tip (4) to Pinky Finger Tip (20) to press down. Separate them to release.

The MediaPipe library provides 21 landmarks per hand. The finger tips used for interaction correspond to the following indices: .

⚙️ Configuration & Tuning
You can adjust the responsiveness and click sensitivity by modifying constants in the script.

Variable	Location	Description
FRAME_WIDTH, FRAME_HEIGHT	Constants	Resolution of the webcam feed. Lower values improve processing speed.
alpha	HandTracker.__init__	EMA Smoothing Factor. Controls cursor jitter vs. lag. Lower values (≈0.1) mean more smoothing/lag.
DEAD_ZONE_THRESHOLD	HandTracker.__init__	Radius (in frame pixels) where movement is ignored to prevent cursor drift when the hand is still.
GAIN	Main Loop (nl func)	Overall Sensitivity. A multiplier for cursor speed. Lower values slow down the cursor.
GAMMA	Main Loop (nl func)	Velocity Power Exponent. Higher values create greater cursor acceleration: small hand movements give precise control, large movements are fast. (Default: 1.0 - linear)
Distance Thresholds	HandTracker.CLICK()	The pixel distance thresholds (e.g., dist_thumb_index < 15) that trigger a click. Adjust based on camera resolution and hand size.
