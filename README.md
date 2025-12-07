# HandWave — Real-Time Hand Tracking Mouse Controller (MediaPipe + OpenCV)

HandWave is a real-time hand-tracking mouse controller built using **MediaPipe**, **OpenCV**, and **Windows system cursor APIs**.  
It allows you to control your mouse using natural hand gestures:

- 🖱️ **Left hand** → cursor control  
- 👆 Thumb + Index → **Double-click**  
- 🤙 Thumb + Pinky → **Hold & Drag**  
- ✋ Right hand → **Gesture distance tracking** (debug / future features)

This project uses smoothing, dead-zone filtering, velocity scaling, and stable MediaPipe landmark tracking to achieve **fast and accurate cursor movement**.

---

## ✨ Features

### 🎯 **Cursor Control (Left Hand)**
- Move your hand → moves the mouse cursor  
- Natural motion with:
  - EMA smoothing  
  - Dead-zone to remove jitter  
  - Velocity mapping for high-resolution screens  

### 👆 **Gestures**
| Gesture | Hand | Action |
|--------|------|--------|
| Thumb + Index | Left | Double-click |
| Thumb + Pinky | Left | Hold left mouse button (drag) |
| Thumb + Index | Right | Distance display (debug) |
| Thumb + Pinky | Right | Distance display (debug) |

---

## 📦 Installation

### 1. **Install Python 3.10**
MediaPipe **only supports Python ≤ 3.10**.

Download Python 3.10 from:  
https://www.python.org/downloads/release/python-3100/

---

## 2. **Create Virtual Environment**

