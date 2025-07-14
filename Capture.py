import mss
import numpy as np
import os
import cv2
import time
import pyautogui
from pynput import keyboard

# === Setup ===
output_dir = os.path.abspath("Frames")
os.makedirs(output_dir, exist_ok=True)
mss_instance = mss.mss()
stop_flag = False

def capture_screen(region=None):
    """
    Capture the screen or a specific region from the secondary monitor.
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
    region = {
        "top": 500,        
        "left": 2560,         
        "width": 1920,         
        "height": 300   
    }
    img = capture_screen(region)
    cv2.imwrite(filename, img)

def steering_wheel_capture():
    """
    Analyze the steering wheel UI area and return signed pixel count.
    """
    region = {"top": 1050, "left": 3405, "width": 230, "height": 25}
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
        return -left_val
    elif right_val > left_val:
        return right_val
    return 0

def on_press(key):
    global stop_flag
    if key == keyboard.Key.esc:
        stop_flag = True
        print("Screenshot capture stopped by ESC key.")

# === Main Loop ===
if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    count = 0
    print("Press ESC to stop capturing.")

    try:
        while not stop_flag:
            # === LEFT KEY ===
            pyautogui.keyDown('left')
            time.sleep(0.2)  # Let input settle
            for _ in range(3):
                filename = os.path.join(output_dir, f'screenshot_{count}.png')
                save_screenshot(filename)
                angle = steering_wheel_capture()
                print(f"[LEFT] Screenshot saved: {filename} | val: {angle}")
                count += 1
                time.sleep(1)
            pyautogui.keyUp('left')

            # === NUM5 ACTION ===
            pyautogui.press('num5')
            time.sleep(0.2)
            for _ in range(3):
                filename = os.path.join(output_dir, f'screenshot_{count}.png')
                save_screenshot(filename)
                angle = steering_wheel_capture()
                print(f"[NUM5] Screenshot saved: {filename} | val: {angle}")
                count += 1
                time.sleep(1)

            # === RIGHT KEY ===
            pyautogui.keyDown('right')
            time.sleep(0.2)
            for _ in range(5):
                filename = os.path.join(output_dir, f'screenshot_{count}.png')
                save_screenshot(filename)
                angle = steering_wheel_capture()
                print(f"[RIGHT] Screenshot saved: {filename} | val: {angle}")
                count += 1
                time.sleep(1)
            pyautogui.keyUp('right')

            # === NUM5 AGAIN ===
            pyautogui.press('num5')
            time.sleep(0.2)
            for _ in range(3):
                filename = os.path.join(output_dir, f'screenshot_{count}.png')
                save_screenshot(filename)
                angle = steering_wheel_capture()
                print(f"[NUM5] Screenshot saved: {filename} | val: {angle}")
                count += 1
                time.sleep(1)

    except KeyboardInterrupt:
        print("Screenshot capture stopped by Ctrl+C.")

    listener.stop()
