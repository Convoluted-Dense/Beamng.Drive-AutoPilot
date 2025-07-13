import mss
import numpy as np
import os
import cv2
import time
from pynput import keyboard

os.makedirs("Frames", exist_ok=True)
mss_instance = mss.mss()


def capture_screen(region=None):
    """
    Capture the screen or a specific region from the secondary monitor.
    If region is None, captures the full secondary monitor.
    Returns:
        np.ndarray: Captured image.
    """
    monitor = region if region else mss_instance.monitors[2]
    screenshot = mss_instance.grab(monitor)
    img = np.array(screenshot)
    return img

def save_screenshot(filename):
    """
    Save a screenshot of the AOI from the secondary monitor.
    """
    monitor = mss_instance.monitors[2]
    region = {
        "top": monitor["top"] + 500,  # AOI top offset
        "left": monitor["left"],
        "width": monitor["width"],
        "height": monitor["height"] - 780  # AOI height
    }
    img = capture_screen(region)
    cv2.imwrite(filename, img)

def steering_wheel_capture():
    """
    Analyze the steering wheel UI area and return signed pixel count:
    - Positive: steering right
    - Negative: steering left
    - Zero: no input
    """
    region = {"top": 1050, "left": 3405, "width": 230, "height": 25} # Adjusted region for steering wheel capture
    img = capture_screen(region)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    lower_orange = np.array([0, 100, 200], dtype=np.uint8)
    upper_orange = np.array([80, 180, 255], dtype=np.uint8)
    mask = cv2.inRange(img, lower_orange, upper_orange)
    result = np.zeros_like(mask)
    result[mask > 0] = 255
    mid = result.shape[1] // 2
    left_val = np.sum(result[:, :mid] == 255)
    right_val = np.sum(result[:, mid:] == 255)
    if left_val > right_val:
        steering_angle = -left_val
    elif right_val > left_val:
        steering_angle = right_val
    else:
        steering_angle = 0
    return steering_angle

if __name__ == "__main__":
    stop_flag = False
    def on_press(key):
        global stop_flag
        if key == keyboard.Key.esc:
            stop_flag = True
            print("Screenshot capture stopped by ESC key.")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    count = 0
    print("Press ESC to stop capturing.")
    try:
        while not stop_flag:
            filename = os.path.join('Frames', f'screenshot_{count}.png')
            save_screenshot(filename)
            steering_angle = steering_wheel_capture()
            print(f" Screenshot saved: {filename} | val: {steering_angle}")
            count += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Screenshot capture stopped by Ctrl+C.")
    listener.stop()

