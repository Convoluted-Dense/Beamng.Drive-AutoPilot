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
    Capture the screen or a specific region.

    :param region: A dict {'top': ..., 'left': ..., 'width': ..., 'height': ...}
        If None, captures the full screen (primary monitor).
    :return: A numpy array of the captured image.
    """
    monitor = region if region else mss_instance.monitors[2] # use [1] for primary monitor
    screenshot = mss_instance.grab(monitor)
    img = np.array(screenshot)
    return img

def save_screenshot(filename, region=None):
    # Get secondary monitor inf
    monitor = mss_instance.monitors[2]
    top = monitor["top"] + 500 # AOI
    left = monitor["left"]
    width = monitor["width"]
    height = monitor["height"] - 740  # AOI
    region = {"top": top, "left": left, "width": width, "height": height}
    img = capture_screen(region)
    cv2.imwrite(filename, img)

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
            print(f" Screenshot saved: {filename}")
            count += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Screenshot capture stopped by Ctrl+C.")
    listener.stop()

