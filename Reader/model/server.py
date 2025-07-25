import sys
import os
import csv
import copy
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import itertools
import time


import cv2 as cv
import mediapipe as mp # <--- Keep this import at the top
from fastapi.responses import HTMLResponse
import uvicorn
# --- Path Adjustments ---
# Get the directory where THIS server.py file is located.
# This will be 'C:\Users\leenh\ML\Questure_Tracker\Reader\model'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to sys.path so 'classifier.py' can be imported directly
sys.path.append(CURRENT_DIR)

# Import your Classifier class directly from the sibling file
from classifier import Classifier

# --- Constants ---
# All models and labels are in the CURRENT_DIR (where server.py is).
MODEL_DIR = CURRENT_DIR
LABEL_PATH = os.path.join(MODEL_DIR, 'label.csv')
STATIC_MODEL_PATH = os.path.join(MODEL_DIR, 'model.tflite')

MAX_NUM_HANDS = 1
DETECTION_THRESHOLD = 0.7
OUTPUT_COUNT = 1

# --- FastAPI App Initialization ---
from contextlib import asynccontextmanager

# Global variables that will be assigned within the lifespan context
hands_detector = None
sign_classifier = None
classifier_labels = []

@asynccontextmanager
async def lifespan_events(app: FastAPI):
    # We only need to declare globals that are ASSIGNED within this function.
    # mp is already imported globally, so no need to declare it global here.
    global hands_detector, sign_classifier, classifier_labels

    print("Server starting up... Loading ML models and labels.")

    # Define mp_hands *inside* the lifespan context
    # This ensures Pylance understands its scope, even though mp is global.
    mp_hands_local = mp.solutions.hands # <--- CHANGE THIS LINE
    hands_detector = mp_hands_local.Hands( # <--- CHANGE THIS LINE to use mp_hands_local
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    print("MediaPipe Hands loaded.")

    # Load Classifier
    if not os.path.exists(STATIC_MODEL_PATH):
        print(f"Error: Static model not found at {STATIC_MODEL_PATH}. Please ensure it exists.")
        raise RuntimeError(f"Model file not found: {STATIC_MODEL_PATH}")

    sign_classifier = Classifier(model_path=STATIC_MODEL_PATH)
    print(f"Classifier loaded from {STATIC_MODEL_PATH}")

    # Load labels
    if not os.path.exists(LABEL_PATH):
        print(f"Error: Label file not found at {LABEL_PATH}. Please ensure it exists.")
        raise RuntimeError(f"Label file not found: {LABEL_PATH}")
    try:
        with open(LABEL_PATH, encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            classifier_labels = [row[0] for row in reader if row]
        print(f"Loaded {len(classifier_labels)} labels from {LABEL_PATH}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        raise RuntimeError(f"Error loading labels: {e}")

    yield # This yields control to the application, serving requests

    # This block runs on shutdown
    print("Server shutting down... Releasing resources.")
    if hands_detector:
        hands_detector.close()

# Pass the lifespan context manager to the FastAPI app
app = FastAPI(lifespan=lifespan_events)

# --- Helper Functions (No changes needed here) ---
def calc_landmark_list(landmarks):
    landmark_point = []
    for lm in landmarks.landmark:
        landmark_point.append([lm.x, lm.y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    if not temp_landmark_list:
        return np.zeros(42, dtype=np.float32)

    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index in range(len(temp_landmark_list)):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = np.max(np.abs(temp_landmark_list))
    if max_value == 0:
        return np.zeros(len(temp_landmark_list), dtype=np.float32)

    temp_landmark_list = np.array(temp_landmark_list, dtype=np.float32) / max_value
    return temp_landmark_list.tolist()

# --- WebSocket Endpoint for Video Stream (No changes needed here) ---
@app.websocket("/ws/video_stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"Client {websocket.client} connected.")
    try:
        while True:
            jpeg_bytes = await websocket.receive_bytes()
            start_time = time.time()

            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            image_bgr = cv.imdecode(np_arr, cv.IMREAD_COLOR)

            if image_bgr is None:
                print("Failed to decode image from bytes.")
                continue

            image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands_detector.process(image_rgb) # Uses the global hands_detector
            image_rgb.flags.writeable = True

            gesture_label = "No Hand Detected"
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list = calc_landmark_list(hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                if pre_processed_landmark_list:
                    N_hand_sign_id, N_hand_sign_probs = sign_classifier(pre_processed_landmark_list, OUTPUT_COUNT)

                    if N_hand_sign_id and N_hand_sign_probs:
                        predicted_id = N_hand_sign_id[0]
                        predicted_prob = N_hand_sign_probs[0]

                        if predicted_prob > DETECTION_THRESHOLD:
                            gesture_label = classifier_labels[predicted_id]
                            confidence = predicted_prob
                        else:
                            gesture_label = "Uncertain"
                            confidence = predicted_prob
                else:
                    gesture_label = "Error processing landmarks"

            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000

            response_data = {
                "gesture": gesture_label,
                "confidence": float(confidence),
                "processing_time_ms": round(processing_time_ms, 2)
            }
            await websocket.send_json(response_data)

    except WebSocketDisconnect:
        print(f"Client {websocket.client} disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            await websocket.send_json({"error": str(e), "gesture": "Error"})
        except RuntimeError:
            pass
    finally:
        print(f"Client {websocket.client} handler finished.")

# --- Root endpoint (optional) ---
@app.get("/")
async def get_root():
    return {"message": "Hand Gesture Detection Server is running!"}