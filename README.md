# Hand Gesture Detection for Deaf Communication
---
## Features:
  - Real-time hand gesture recognition using webcam
  - Customizable gesture vocabulary via label files
  - Simple interface for data collection and model training
  - detect (A->Z) and some common words like (Hi,how are,nice,meet...)
  - Text output of recognized signs

## Demo
  ### Final real_time output
  ![Untitled video - Made with Clipchamp (2)](https://github.com/user-attachments/assets/80bc7164-0d44-40e3-84f4-b3c573915940)


## Installation:
  ### Prerequisites:
   <li/>Python 3.11
    <li/>pip install opencv-python
    <li/>pip install mediapipe
    <li/>pip install numpy
    <li/>pip install tensorflow

  ### Dependencies:
    ```bash
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
  - Coordinates normalized relative to wrist
  - 42 input 

### Model Output:
  - N_hand_sign_id: Top N predicted indices
  - N_hand_sign_probs: Confidence scores
  - Minimum confidence: 10% (adjustable)
### Model conversion
  - convert model to tensorflow lite to be easy and fast in real_time
  - to give it to the Flutter team to connect it to the applications

## References:
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
- TensorFlow Lite for on-device inference

