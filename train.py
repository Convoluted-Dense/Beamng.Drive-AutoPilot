import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths
data_dir = os.path.abspath("Frames")
log_path = "steering_log.txt"

# Load image paths and angles
img_paths = []
angles = []
with open(log_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            img_paths.append(os.path.join(data_dir, parts[0]))
            angles.append(float(parts[1]))

# Split into training and validation
train_img_paths, val_img_paths, train_angles, val_angles = train_test_split(
    img_paths, angles, test_size=0.2, random_state=42
)

# Augmentation function
def augment_image(img, angle):
    h, w = img.shape[:2]

    # === 1. Random rotation (±15 degrees) ===
    if np.random.rand() < 0.5:
        rot_angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rot_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        angle += rot_angle / 90.0

    # === 2. Brightness adjustment ===
    bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    brightness_scale = 0.5 + np.random.uniform()
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_scale, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # === 3. Contrast adjustment ===
    if np.random.rand() < 0.5:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        contrast = np.random.uniform(0.7, 1.3)
        l = np.clip(l * contrast, 0, 255).astype(np.uint8)
        lab = cv2.merge((l, a, b))
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # === 4. Convert back to YUV for consistency ===
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

    # === 5. Shadow bands (on Y channel) ===
    num_bands = np.random.randint(1, 4)
    for _ in range(num_bands):
        band_width = np.random.randint(w // 8, w // 3)
        x_start = np.random.randint(0, w - band_width)
        y_start = np.random.randint(0, h - 10)
        y_end = np.random.randint(y_start + 10, min(h, y_start + h // 2))
        shadow_mask = np.zeros_like(img[:, :, 0])
        cv2.rectangle(shadow_mask, (x_start, y_start), (x_start + band_width, y_end), 255, -1)
        rand_alpha = np.random.uniform(0.4, 0.75)
        img[shadow_mask == 255] = (img[shadow_mask == 255] * rand_alpha).astype(np.uint8)

    # === 6. Grain (Gaussian noise) ===
    noise = np.random.normal(0, 0.02, img.shape) * 255
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # === 7. Normalize ===
    img = img.astype(np.float32) / 255.0

    return img, angle

# Batch generator (with augmentation)
def batch_generator(img_paths, angles, batch_size):
    num_samples = len(img_paths)
    while True:
        indices = np.random.permutation(num_samples)
        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset+batch_size]
            batch_imgs = []
            batch_angles = []
            for idx in batch_indices:
                img = cv2.imread(img_paths[idx])
                angle = angles[idx]
                if img is not None:
                    img, angle = augment_image(img, angle)
                    batch_imgs.append(img)
                    batch_angles.append(angle)
            yield np.array(batch_imgs), np.array(batch_angles)

# Validation generator (no augmentation)
def val_batch_generator(img_paths, angles, batch_size):
    num_samples = len(img_paths)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_paths = img_paths[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]
            batch_imgs = []
            for path in batch_paths:
                img = cv2.imread(path)
                if img is not None:
                    img = img.astype(np.float32) / 255.0
                    batch_imgs.append(img)
            yield np.array(batch_imgs), np.array(batch_angles)

# NVIDIA PilotNet model
model = keras.Sequential([
    keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)),
    keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
    keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
print(model.summary())

# Training
batch_size = 32
steps_per_epoch = len(train_img_paths) // batch_size
validation_steps = max(1, len(val_img_paths) // batch_size)

history = model.fit(
    batch_generator(train_img_paths, train_angles, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=val_batch_generator(val_img_paths, val_angles, batch_size),
    validation_steps=validation_steps
)

# Save model and logs
os.makedirs("models", exist_ok=True)
os.makedirs("loss_log", exist_ok=True)
model.save("models/v5.1.h5")

# Plot loss graph
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history.get('val_loss', []), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_log/v5.1.png')
