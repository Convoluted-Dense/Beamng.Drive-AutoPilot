import mss
import numpy as np
import os
import cv2
import time
import pyautogui
import pyvjoy
from pynput import keyboard
import random

# === Setup ===555
output_dir = os.path.abspath("val_Frames")
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

def save_screenshot(filename,max_shift, shift=0):
    """
    Save a screenshot of the AOI from the secondary monitor, with optional lateral shift.
    Always crop for the maximum possible shift to ensure consistent region.
    """
    region = {
        "top": 500,
        "left": 2560,
        "width": 1920,
        "height": 300
    }
    img = capture_screen(region)
    height, width = img.shape[:2]
      # Make sure this matches your main loop's max_shift

    # Apply random lateral shift if specified
    if shift != 0:
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        dst_points = np.float32([
            [shift, 0],
            [width + shift, 0],
            [width - shift, height],
            [-shift, height]
        ])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img = cv2.warpPerspective(img, matrix, (width, height))

    # Always crop for the maximum possible shift to avoid black pixels
    crop_margin = int(abs(max_shift))
    img = img[:, crop_margin:width-crop_margin]
    img = cv2.resize(img, (1920, 300))  # Resize back to original region

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.resize(img, (200, 66))
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
        return -left_val/2464
    elif right_val > left_val:
        return right_val/2464
    return 0

def on_press(key):
    global stop_flag
    if key == keyboard.Key.f6:
        stop_flag = True
        print("Screenshot capture stopped by ESC key.")

# === Main Loop ===
if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    count = 0
    print("Press ESC to stop capturing.")

    print("Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    log_path = "val_steering_log.txt"
    log_file = open(log_path, "a")

    vjoy = pyvjoy.VJoyDevice(1)
    max_shift = 0

    try:
        while not stop_flag:
            # Generate a random shift between -max_shift and +max_shift (e.g., -60 to 60)
            shift = random.randint(-max_shift, max_shift)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f'screenshot_{count}_{timestamp}.png')
            save_screenshot(filename, shift=shift,max_shift=max_shift)
            angle = steering_wheel_capture()
            # Compensate steering angle for shift: if shift is negative, add positive compensation, and vice versa
            if max_shift == 0:
              compensation = 0
            else:
              compensation = round(-shift / (max_shift * 10), 3)
            adjusted_angle = round(angle + compensation, 3)
            print(f"[CAPTURE] Screenshot saved: {filename} | val: {adjusted_angle} | shift: {shift}")
            log_file.write(f"{os.path.basename(filename)},{adjusted_angle}\n")
            count += 1
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("Screenshot capture stopped by Ctrl+C.")
    finally:
        log_file.close()
        listener.stop()