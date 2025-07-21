import os
import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# === Load and parse the log file ===
log_path = "steering_log.txt"
entries = []

with open(log_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            try:
                angle = round(float(parts[1]), 3)
                entries.append((parts[0], angle))
            except ValueError:
                continue

# === Split into categories ===
zero_entries = [e for e in entries if e[1] == 0]
pos_entries = [e for e in entries if e[1] > 0]
neg_entries = [e for e in entries if e[1] < 0]

# === Calculate target counts (with clamping to available data) ===
total = len(entries)
target_zero = min(int(total * 0.6), len(zero_entries))
target_pos = min(int(total * 0.2), len(pos_entries))
target_neg = min(int(total * 0.2), len(neg_entries))

# === Randomly sample entries ===
random.shuffle(zero_entries)
random.shuffle(pos_entries)
random.shuffle(neg_entries)

balanced = zero_entries[:target_zero] + pos_entries[:target_pos] + neg_entries[:target_neg]
random.shuffle(balanced)

# === Save balanced entries ===
with open("steering_log_balanced.txt", "w") as f:
    for img, angle in balanced:
        f.write(f"{img},{angle}\n")

# === Histogram plotting setup ===
def plot_distribution(title, angles, color, subplot_position):
    bins = np.arange(min(angles), max(angles) + 0.1, 0.1)
    hist, bin_edges = np.histogram(angles, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.subplot(1, 2, subplot_position)
    plt.bar(bin_centers, hist, width=0.09, color=color, align='center')
    plt.title(title)
    plt.xlabel("Steering Angle")
    plt.ylabel("Image Count")
    plt.xticks(np.round(bin_centers, 2), rotation=90)
    plt.grid(True)

# === Plot Original vs Balanced ===
original_angles = [angle for _, angle in entries]
balanced_angles = [angle for _, angle in balanced]

plt.figure(figsize=(14, 6))
plot_distribution("Original Distribution", original_angles, "orange", 1)
plot_distribution("Balanced Distribution", balanced_angles, "green", 2)
plt.tight_layout()
plt.show()
