import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Paths
data_dir = os.path.abspath("Frames")
log_path = "steering_log_balanced.txt"

# Load data
img_paths = []
angles = []
with open(log_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            img_paths.append(os.path.join(data_dir, parts[0]))
            angles.append(float(parts[1]))


# Augmentation functions
def augment_image(img):
    # Random brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ratio = 0.5 + np.random.uniform()
    hsv[:,:,2] = np.clip(hsv[:,:,2] * ratio, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Random shadow bands
    h, w = img.shape[:2]
    num_bands = np.random.randint(1, 4)  # 1 to 3 bands
    for _ in range(num_bands):
        band_width = np.random.randint(w//8, w//3)
        x_start = np.random.randint(0, w - band_width)
        y_start = np.random.randint(0, h - 10)
        y_end = np.random.randint(y_start + 10, min(h, y_start + h//2))
        shadow_mask = np.zeros_like(img[:,:,0])
        cv2.rectangle(shadow_mask, (x_start, y_start), (x_start + band_width, y_end), 255, -1)
        rand_alpha = np.random.uniform(0.5, 0.85)
        img[shadow_mask==255] = (img[shadow_mask==255] * rand_alpha).astype(np.uint8)

    # Random noise
    noise = np.random.normal(0, 0.03, img.shape) * 255
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    img = img.astype(np.float32) / 255.0
    return img

# Batch generator
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
                if img is not None:
                    img = augment_image(img)
                    batch_imgs.append(img)
                    batch_angles.append(angles[idx])
            yield np.array(batch_imgs), np.array(batch_angles)

# NVIDIA PilotNet model
model = keras.Sequential([
    keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)),
    keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
    keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

print(model.summary())


# Train with generator
batch_size = 32
steps_per_epoch = len(img_paths) // batch_size
history = model.fit(
    batch_generator(img_paths, angles, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=25,
    validation_data=batch_generator(img_paths, angles, batch_size),
    validation_steps=max(1, steps_per_epoch//10)
)

# Save model
model.save("models/v3.1.h5")

# Plot and save loss vs epoch graph

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history.get('val_loss', []), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_log/v3.1.png')
