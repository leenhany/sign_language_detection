# Hand Gesture Detection for Deaf Communication
---
## Features:
  - Real-time hand gesture recognition using webcam
  - Customizable gesture vocabulary via label files
  - Simple interface for data collection and model training
  - Visual feedback of detected gestures
  - Text output of recognized signs

## Demo

    ![Untitled video - Made with Clipchamp (1)](https://github.com/user-attachments/assets/ab6292b1-e133-4e63-b3c7-560c2b17b8df)


## Installation:
  ### Prerequisites:
   <li/>Python 3.11
    <li/>pip install opencv-python
    <li/>pip install mediapipe
    <li/>pip install numpy
    <li/>pip install tensorflow

  ### Dependencies:
    ```bash![Untitled video - Made with Clipchamp](https://github.com/user-attachments/assets/365fe298-84b3-4dad-97cf-5135045b3072)

    pip install opencv-python mediapipe numpy tensorflow
    ```

## Project Structure:

```
├── Collector/
│   ├── collect.py          # Data collection script
│   ├── data.csv           # Collected landmark data
│   └── label.csv           # Gesture labels
│
├── Reader/
│   ├── model/
│   │   ├── classifier.py   # Model inference class
│   │   ├── label.csv       # Gesture labels
│   │   └── model.tflite    # Trained model
│   └── app.py              # Real-time detection script
│
└── Trainer/
    ├── model/
    │   ├── data/          # Training data
    |   |    ├── data.csv
    |   |    └──  label.csv
    │   └── model.tflite    # Trained model
    └── train_model.ipynb   # Model training notebook
```

## Usage:
### 1. Data Collection:
  ```bash
  python Collector/collect.py
  ```
  - Controls:
    - ENTER: Enter recording mode
    - Arrow keys: Select gesture label
    - SPACE: Start/stop auto-recording
    - BACKSPACE: Exit recording mode

### 2. Model Training:
  ```
  bash
  jupyter notebook Trainer/train_model.ipynb
  ```

### 3. Real-time Detection:
  ```
  bash
  python Reader/app.py
  ```
  - Controls:
    - 'c': Clear text
    - BACKSPACE: Delete last word
    - ESC: Exit

## Technical Details:
### Data Representation:
  - landmark_list: 21 (x,y) coordinates
  - landmark_list_3D: 21 (x,y,z) coordinates
  - Coordinates normalized relative to wrist

### Model Output:
  - N_hand_sign_id: Top N predicted indices
  - N_hand_sign_probs: Confidence scores
  - Minimum confidence: 10% (adjustable)

## References:
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
- TensorFlow Lite for on-device inference

## License:
Apache License 2.0

## Acknowledgments:
Developed to assist the deaf community through technology
