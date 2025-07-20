import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import mss
import time
import pyvjoy

# Load model without compiling, then recompile
model = keras.models.load_model("models/v3.1.h5", compile=False)
model.compile(optimizer="adam", loss="mse")

# Initialize vJoy device (ID 1 by default)
vjoy = pyvjoy.VJoyDevice(1)

# Screen capture setup (same region as Capture.py)
mss_instance = mss.mss()
region = {
    "top": 500,
    "left": 2560,
    "width": 1920,
    "height": 300
}

def preprocess(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.resize(img, (200, 66))
    img = img.astype(np.float32) / 255.0
    return img

print("Press Ctrl+C to stop.")
try:
    while True:
        screenshot = mss_instance.grab(region)
        img = np.array(screenshot)
        img = preprocess(img)
        img = np.expand_dims(img, axis=0)
        pred = float(model.predict(img)[0][0])

        # Map prediction (-1 to 1) to X axis value (1 to 32768, center=16384)
        # vJoy expects values in [1, 32768]
        pred = max(-1.0, min(1.0, pred))*1.5
        x_val = int((pred + 1) * 16384)
        x_val = max(1, min(32768, x_val))
        vjoy.set_axis(pyvjoy.HID_USAGE_X, x_val)
        # Optionally keep Y axis centered
        vjoy.set_axis(pyvjoy.HID_USAGE_Y, 16384)

        print(f"Predicted steering: {pred:.3f} | X axis: {x_val}")
        time.sleep(0.2)
except KeyboardInterrupt:
    # Center the stick on exit
    vjoy.set_axis(pyvjoy.HID_USAGE_X, 16384)
    vjoy.set_axis(pyvjoy.HID_USAGE_Y, 16384)
    print("Stopped.")
