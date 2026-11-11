# Beamng.Drive-AutoPilot

End-to-end deep learning framework for autonomous vehicle steering guidance within the BeamNG.drive physics simulation engine.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Overview

Beamng.Drive-AutoPilot implements an end-to-end vision-based steering controller trained on real-time screen captures and telemetry from BeamNG.drive. Using a hybrid Convolutional and Recurrent Neural Network (CNN-LSTM) architecture inspired by NVIDIA's DAVE-2 self-driving pipeline, the model learns temporal spatial dynamics across consecutive video frames to predict continuous steering angles with high fidelity.

---

## Architectural Highlights

- Temporal Sequence Modeling: Integrates TimeDistributed convolutional feature extractors with LSTM layers to capture steering momentum, yaw changes, and road curvature over sliding temporal windows (SEQ_LENGTH = 5).
- Data Balancing and Normalization: Features a dataset curation pipeline (Balance_data.py) that bins steering angles, trims excessive zero-angle highway frames, and mitigates bias toward straight driving.
- Real-Time Telemetry Pipeline: Captures game frames (66x200 normalized YUV/RGB inputs) and maps analog steering wheel inputs using OpenCV and Windows API hooks.
- Closed-Loop Inference Engine: Evaluates the trained model in real time (test.py), reading the live game viewport and injecting virtual steering inputs to drive autonomously.

---

## Repository Structure

```
Beamng.Drive-AutoPilot/
├── Balance_data.py      # Telemetry log parsing, steering histogram balancing, and downsampling
├── Capture.py           # Real-time screen capture, telemetry synchronization, and frame logging
├── Capture_raw          # Low-level capture scripts and raw viewport utilities
├── test.py              # Live closed-loop autonomous driving inference engine
├── train.py             # CNN-LSTM model definition, data generator, and training loop
└── README.md            # Project documentation
```

---

## Machine Learning Pipeline

### 1. Model Architecture

The neural network consumes a sliding window of sequential front-facing camera frames:
- Input Dimension: (Batch, 5, 66, 200, 3)
- Feature Extractor: 5 TimeDistributed 2D Convolutional layers with Strided Convolutions and ReLU activations
- Sequence Processing: LSTM(64) layer capturing vehicle momentum and curvature transitions
- Dense Regressor: Fully connected layers (100 -> 50 -> 10 -> 1) with Dropout regularization (0.2) outputting a continuous steering prediction in [-1.0, 1.0].

### 2. Dataset Balancing (Balance_data.py)

Driving datasets frequently suffer from zero-bias (prolonged straight segments). The balancing script:
1. Parses steering_log.txt timestamps and corresponding frame filenames.
2. Constructs a histogram of steering values across 31 discrete bins.
3. Subsamples dominant zero-angle bins according to a target threshold, ensuring balanced exposure to sharp turns, recovery maneuvers, and apex transitions.

---

## Prerequisites and Installation

### Hardware & Environment Requirements
- Windows 10/11 64-bit
- NVIDIA GPU with CUDA and cuDNN support
- Python 3.9+
- BeamNG.drive simulation installed

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Convoluted-Dense/Beamng.Drive-AutoPilot.git
cd Beamng.Drive-AutoPilot
```

2. Create and activate a Python virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib mss pywin32
```

---

## Usage Workflow

### Step 1: Telemetry & Frame Collection
Launch BeamNG.drive in windowed mode (e.g., 1280x720 or 800x600). Start the capture script to record your driving demonstrations:
```bash
python Capture.py
```
Outputs are written to Non_map_dataset/Frames/ and Non_map_dataset/steering_log.txt.

### Step 2: Balance Dataset
Run the balancing utility to eliminate straight-line bias:
```bash
python Balance_data.py
```

### Step 3: Train the CNN-LSTM Network
Train the temporal guidance network:
```bash
python train.py
```
The script splits data 80/20 for training and validation, tracks MSE loss, and exports the trained model weights.

### Step 4: Closed-Loop Autonomous Test
Position your vehicle in BeamNG.drive and launch the autonomous inference engine:
```bash
python test.py
```
The agent reads screen inputs in real time, computes target steering commands, and guides the vehicle along the course.

---

## License

This project is licensed under the MIT License.
