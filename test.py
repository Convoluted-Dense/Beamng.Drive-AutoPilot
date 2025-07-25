import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import mss
import time
import pyvjoy
from collections import deque
from pynput import keyboard  # <== For pause toggle
from tensorflow.keras import layers





# Load RNN model
model = keras.models.load_model(
    "models/v6.6.h5",
    compile=False
)

# Initialize vJoy device (ID 1 by default)
vjoy = pyvjoy.VJoyDevice(1)

# Screen capture setup
mss_instance = mss.mss()
region = {
    "top": 525,
    "left": 2560,
    "width": 1920,
    "height": 225
}

# Constants
SEQ_LENGTH = 5
IMG_WIDTH, IMG_HEIGHT = 200, 66

# Frame buffer
frame_buffer = deque(maxlen=SEQ_LENGTH)

# Pause flag
paused = False

# Key listener function
def on_press(key):
    global paused
    try:
        if key.char == 'l':  # Press 'P' to toggle pause
            paused = not paused
            if paused:
                print("[PAUSED] Controls disabled.")
                vjoy.set_axis(pyvjoy.HID_USAGE_X, 16384)
                vjoy.set_axis(pyvjoy.HID_USAGE_Y, 16384)
            else:
                print("[RESUMED] Controls enabled.")
    except AttributeError:
        pass  # Handle special keys like shift, etc.

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

def preprocess(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return img

print("Press Ctrl+C to stop. Press 'P' to pause/resume.")

try:
    while True:
        if paused:
            time.sleep(0.1)
            continue

        screenshot = mss_instance.grab(region)
        img = np.array(screenshot)
        img = preprocess(img)
        frame_buffer.append(img)

        if len(frame_buffer) == SEQ_LENGTH:
            sequence = np.array(frame_buffer).reshape((1, SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3))
            pred = float(model.predict(sequence, verbose=0)[0][0])

            pred = max(-1.0, min(1.0, pred))
            x_val = int((pred + 1) * 16384)
            x_val = max(1, min(32768, x_val))
            vjoy.set_axis(pyvjoy.HID_USAGE_X, x_val)
            vjoy.set_axis(pyvjoy.HID_USAGE_Y, 16384)

            print(f"Predicted steering: {pred:.3f} | X axis: {x_val}")

        time.sleep(0.05)

except KeyboardInterrupt:
    vjoy.set_axis(pyvjoy.HID_USAGE_X, 16384)
    vjoy.set_axis(pyvjoy.HID_USAGE_Y, 16384)
    print("Stopped.")
