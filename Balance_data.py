import matplotlib.pyplot as plt
import os
import random

log_path = "steering_log.txt"
entries = []

with open(log_path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            try:
                angle = round(float(parts[1]), 1)
                entries.append((parts[0], angle))
            except ValueError:
                continue

# Split into groups
zero_entries = [e for e in entries if e[1] == 0]
pos_entries = [e for e in entries if e[1] > 0]
neg_entries = [e for e in entries if e[1] < 0]

total = len(entries)
target_zero = int(total * 0.6)
target_pos = int(total * 0.2)
target_neg = int(total * 0.2)

# Randomly sample from each group
random.shuffle(zero_entries)
random.shuffle(pos_entries)
random.shuffle(neg_entries)

balanced = zero_entries[:target_zero] + pos_entries[:target_pos] + neg_entries[:target_neg]
random.shuffle(balanced)

# Save to new file
with open("steering_log_balanced.txt", "w") as f:
    for img, angle in balanced:
        f.write(f"{img},{angle}\n")

# Plot the balanced distribution
from collections import Counter
angles = [angle for _, angle in balanced]
angle_counts = Counter(angles)

plt.figure(figsize=(10,5))
plt.bar(angle_counts.keys(), angle_counts.values(), color='g')
plt.xlabel('Steering Angle')
plt.ylabel('Number of Images')
plt.title('Balanced: Number of Images per Steering Angle')
plt.grid(True)
plt.tight_layout()
plt.show()
