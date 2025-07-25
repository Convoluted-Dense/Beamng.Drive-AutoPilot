import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.layers import TimeDistributed, Conv2D, Dropout, Flatten, Dense, LSTM, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Parameters ---
SEQ_LENGTH = 5
IMG_HEIGHT, IMG_WIDTH = 66, 200

# --- Paths ---
data_dir = os.path.abspath("Non_map_dataset/Frames")
log_path = "Non_map_dataset/steering_log.txt"

# --- Load image paths and angles ---
img_paths = []
angles = []
with open(log_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            img_paths.append(os.path.join(data_dir, parts[0]))
            angles.append(float(parts[1]))

# --- Group frames into sequences ---
sequences = []
seq_angles = []
for i in range(len(img_paths) - SEQ_LENGTH + 1):
    seq = img_paths[i:i + SEQ_LENGTH]
    label = angles[i + SEQ_LENGTH - 1]  # Predict angle of last frame
    sequences.append(seq)
    seq_angles.append(label)

train_seqs, val_seqs, train_angles, val_angles = train_test_split(
    sequences, seq_angles, test_size=0.2, random_state=42
)

# --- Data generator ---
def sequence_generator(seqs, angles, batch_size):
    while True:
        indices = np.random.permutation(len(seqs))
        for offset in range(0, len(seqs), batch_size):
            batch_indices = indices[offset:offset+batch_size]
            batch_imgs = []
            batch_labels = []
            for idx in batch_indices:
                frames = []
                for img_path in seqs[idx]:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                        img = img.astype(np.float32) / 255.0
                        frames.append(img)
                if len(frames) == SEQ_LENGTH:
                    batch_imgs.append(frames)
                    batch_labels.append(angles[idx])
            yield np.array(batch_imgs), np.array(batch_labels)

# --- Model ---
model = keras.Sequential([
    Input(shape=(SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)),
    TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), activation='relu')),
    TimeDistributed(Conv2D(36, (5, 5), strides=(2, 2), activation='relu')),
    TimeDistributed(Conv2D(48, (5, 5), strides=(2, 2), activation='relu')),
    
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(Flatten()),
    TimeDistributed(Dropout(0.7)),
    LSTM(100, return_sequences=False),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
model.summary()

# --- Training ---
batch_size = 16
steps_per_epoch = len(train_seqs) // batch_size
validation_steps = len(val_seqs) // batch_size

history = model.fit(
    sequence_generator(train_seqs, train_angles, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=sequence_generator(val_seqs, val_angles, batch_size),
    validation_steps=validation_steps
)

# --- Save model and plot ---
os.makedirs("models", exist_ok=True)
model.save("models/v6.2.h5")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RNN Steering Prediction Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_log/v6.2.png")
plt.close()