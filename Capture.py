import mss
import numpy as np
import os
import cv2

os.makedirs('Frames', exist_ok=True)
mss_instance = mss.mss()

def capture_screen(region=None):
    """
    Capture the screen or a specific region.
    
    :param region: A tuple (left, top, width, height) defining the region to capture.
                If None, captures the full screen.
    :return: A numpy array of the captured image.
    """
    screenshot = mss_instance.grab(region) if region else mss_instance.grab(mss_instance.monitors[1])
    img = np.array(screenshot)
    return img
def save_screenshot(filename, region=None):
    img = capture_screen(region)
    cv2.imwrite(filename, img)

save_screenshot(os.path.join('Frames', 'screenshot.png'))
print("Screenshot saved.")
    
        
    

