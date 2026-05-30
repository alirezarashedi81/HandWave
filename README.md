# Gesture TV Controller

Control your mouse cursor and trigger clicks using hand gestures detected through a webcam. Built with MediaPipe, OpenCV, and a C++ pybind11 extension for performance-critical processing.

---

## Project Structure

```
G:/
│   config.py           — all tunable parameters (thresholds, camera, colours)
│   Fpscalculator.py    — FPS tracking
│   Gesturetv.py        — main controller class (camera loop, cursor control)
│   Handtracker.py      — gesture detection, wraps the C++ ClickProcessor
│   Handvisualizer.py   — all OpenCV drawing helpers
│   Systemcontrol.py    — Windows mouse/cursor API via ctypes
│   main.py             — entry point, handles dependency install + C++ build
│   test.py             — pytest suite
│
└───Click_utils/
        click_utils.cpp                  — C++ ClickProcessor (EMA, dead-zone, distances)
        setup.py                         — pybind11 build script
        click_utils.cp310-win_amd64.pyd  — compiled extension (auto-built by main.py)
```

---

## Requirements

- Windows 10/11
- Python 3.10
- A webcam
- A C++ compiler — **Visual Studio Build Tools** (MSVC) recommended on Windows
  - Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - During install select **"Desktop development with C++"**

---

## Quick Start

Just run:

```bash
python main.py
```

`main.py` will automatically:
1. Install any missing Python packages via pip
2. Compile `Click_utils/click_utils.cpp` if the `.pyd` is missing or out of date
3. Launch the controller

No manual build step needed.

---

## Gestures

| Hand  | Gesture                        | Action         |
|-------|--------------------------------|----------------|
| Left  | Move middle finger             | Move cursor    |
| Right | Pinch thumb + index finger     | Double click   |
| Right | Pinch thumb + pinky finger     | Click and drag |

---

## Configuration

All parameters are in `config.py`. Key ones:

| Parameter             | Default | Description                                      |
|-----------------------|---------|--------------------------------------------------|
| `PINCH_THRESHOLD`     | 10 px   | Distance below which a pinch is detected         |
| `RELEASE_THRESHOLD`   | 15 px   | Distance above which a pinch is released         |
| `EMA_ALPHA`           | 0.12    | Smoothing factor — lower is smoother, more lag   |
| `DEAD_ZONE_THRESHOLD` | 3 px    | Minimum movement before cursor updates           |
| `VELOCITY_GAIN`       | 4.0     | Overall cursor speed multiplier                  |
| `MAX_CURSOR_STEP`     | 40 px   | Maximum cursor movement per frame                |
| `FLIP_HORIZONTAL`     | True    | Mirror the camera feed                           |
| `CAMERA_INDEX`        | 0       | Webcam index                                     |

---

## How It Works

### C++ Extension — `ClickProcessor`

`click_utils.cpp` is compiled into a Python extension via pybind11. Each hand gets its own independent `ClickProcessor` instance so their state never interferes. On every frame it:

1. Converts normalised MediaPipe landmark coordinates to pixel coordinates
2. Computes Euclidean distances between thumb-index and thumb-pinky tips
3. Checks the dead zone by comparing the raw middle-finger position to the previous frame
4. Applies EMA smoothing to the middle-finger position for stable cursor movement

The two tuneable parameters are exposed as Python properties so they can be read or changed at runtime:

```python
proc.alpha               = 0.12
proc.dead_zone_threshold = 3.0
```

### Hand Label Correction

MediaPipe labels hands from the subject's perspective. Because the frame is horizontally flipped (`FLIP_HORIZONTAL = True`), the labels are swapped in `process_frame()` so Left/Right always match what appears on screen.

### Cursor Control

The left hand controls the cursor. The smoothed middle-finger position is computed by the C++ processor each frame. Velocity is derived from the delta between the previous and current smoothed positions and passed through a non-linear mapping (`VELOCITY_GAMMA`, `VELOCITY_GAIN`) before being applied to the system cursor.

---

## Running Tests

```bash
pytest test.py -v
```

The test suite covers:
- `FpsCalculator` — initial state and FPS calculation
- `HandTracker` helpers — EMA, dead zone (Python path)
- Dead zone source — verifies the flag comes from the C++ processor's raw-position tracking
- Per-hand isolation — confirms the two `ClickProcessor` instances are fully independent
- `CLICK()` return dict — all keys, first-call `None` for `prev_mx/my`, unknown label
- Click/drag gestures — pinch detection and release for both thumb-index and thumb-pinky
- `reset_smoothing()` — clears both processor state and prev snapshots
- `HandVisualizer` — all drawing methods run without error
- `config` — required attributes and correct default values

---

## Manual Build (if needed)

If you want to compile the extension yourself without running `main.py`:

```bash
cd Click_utils
python setup.py build_ext --inplace
```

The resulting `.pyd` file stays inside `Click_utils/` and is picked up automatically.
