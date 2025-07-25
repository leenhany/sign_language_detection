import csv
import copy
import shutil
import argparse
import itertools
from collections import deque
import time # Import time for adding a small delay

import numpy as np
import cv2 as cv
import mediapipe as mp

# Attributes Constants
MAX_NUM_HANDS = 1
SEQUENCE_FRAME_NUM = 21 # Still used for deque length, though movement sequences are no longer processed
LANDMARK_MOVEMENT_SEQUENCE = deque(maxlen = SEQUENCE_FRAME_NUM) # Retained for potential future use or display
HAND_MOVEMENT_SEQUENCE = deque(maxlen = SEQUENCE_FRAME_NUM)     # Retained for potential future use or display
ZERO_LIST = list(np.zeros((21, 2)))
ZERO_LIST_3D = list(np.zeros((21, 3)))

# Recording Constants
TARGET_SAMPLES_PER_LABEL = 2000 # Set the target number of samples for each label
RECORDING_DELAY_MS = 0 # Add a small delay between automatic recordings to prevent overwhelming

# Path Constants
DATA_CSV_PATH = 'Collector/data2.csv'       # For static gestures
LABEL_PATH = 'Collector/label.csv'
MOVEMENT_CSV_PATH = 'Collector/m_data.csv' # No longer actively written to for this version

TRAINER_DATA_DIR = 'Trainer/model/data2/'
READER_MODEL_DIR = 'Reader/model/'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type = int, default = 0)
    parser.add_argument("--width", help = 'cap width', type = int, default = 960)
    parser.add_argument("--height", help = 'cap height', type = int, default = 540)
    
    parser.add_argument('--use_static_image_mode', action = 'store_true')
    parser.add_argument("--min_detection_confidence", help = 'min_detection_confidence', type = float, default = 0.7)
    parser.add_argument("--min_tracking_confidence", help = 'min_tracking_confidence', type = int, default = 0.5)
    
    args = parser.parse_args()

    return args

# select_mode function now also returns a flag for spacebar press
def select_mode(key, mode, auto_record_active):
    
    number = 0
    space_pressed = False # Flag for spacebar press
    
    if key == 8:        # Backspace: 8
        mode = 0
        auto_record_active = False # Stop auto-record when exiting mode
    if key == 13:       # Enter:     13
        mode = 1
    
    if key == 2555904:  # Right Key: 2555904
        number = 1
    if key == 2424832:  # Left Key : 2424832
        number = -1
    
    if key == 32:       # Space bar: 32 - Toggles auto_record_active
        space_pressed = True
    
    return number, mode, space_pressed, auto_record_active


def calc_landmark_list(image, landmarks): 
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point 

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0: 
        return list(np.zeros_like(temp_landmark_list))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

 
# Data Recording (to csv file)
def record_data(path, label_index, pp_list):    
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label_index, *pp_list])
    return

# Static signs recording
def record_static(image, csv_path, index_in_label, landmark_list, sample_counts):
    # Only record if we haven't reached the target samples for this label
    if sample_counts[index_in_label] < TARGET_SAMPLES_PER_LABEL:
        # Write to csv
        record_data(csv_path, index_in_label, pre_process_landmark(landmark_list))
        sample_counts[index_in_label] += 1
        print(f"Recorded label {index_in_label} ('{words[index_in_label]}'): {sample_counts[index_in_label]}/{TARGET_SAMPLES_PER_LABEL}")
        cv.putText(image, "RECORDED!!", (20,220), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
        time.sleep(RECORDING_DELAY_MS / 1000) # Small delay to prevent too many samples in a short burst
    else:
        print(f"Label '{words[index_in_label]}' has already reached {TARGET_SAMPLES_PER_LABEL} samples. Cannot record more for this label.")
        cv.putText(image, "MAX SAMPLES REACHED!", (20,220), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
    return 

# Movement signs recording (removed as all gestures are static)
def record_movement(image, hand_movement_seq_list, csv_path, index_in_label, landmark_movement_seq_list, sample_counts):
    print("Movement recording is disabled in this version. All gestures are treated as static.")
    return


### Visualization Functions
# No longer relevant for purely static gestures as we don't track paths
def pre_process_landmark_movement(image, m_sequence):
    w, h = image.shape[1], image.shape[0]
    movement_sequence = copy.deepcopy(m_sequence)

    base_x, base_y = 0, 0
    for index, movement_point in enumerate(movement_sequence):
        if index == 0:
            base_x, base_y = movement_point[0], movement_point[1]

        movement_sequence[index][0] = (movement_sequence[index][0] - base_x) / w
        movement_sequence[index][1] = (movement_sequence[index][1] - base_y) / h

    return movement_sequence

def visualization_pre_process_hand_movement(image, m_sequence):
    
    copy_of_m_sequence = copy.deepcopy(m_sequence) 

    landmark_sequences = [] 

    for landmark in range(0, 21):
        landmark_seq_cord = []
        for frame in range(0, SEQUENCE_FRAME_NUM):
            landmark_seq_cord.append(copy_of_m_sequence[frame][landmark])
        landmark_seq_cord = list(itertools.chain.from_iterable(
            pre_process_landmark_movement(image, landmark_seq_cord)))

        landmark_sequences.append(landmark_seq_cord)

    landmark_sequences = list(itertools.chain.from_iterable(landmark_sequences))

    return landmark_sequences

# Hint Display
def screen_drawing(image, mode, current_word, current_sample_count, auto_record_active):
    
    if mode == 0:
        cv.putText(image, "Press ENTER to enter recording mode.", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)

    elif mode == 1:
        cv.putText(image, "Press ARROW KEYS to change label.", (20,100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)
        cv.putText(image, "Press SPACE to START/PAUSE auto-record.", (20,160), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)
        cv.putText(image, "Press BACKSPACE to exit recording mode.", (20,280), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)
        cv.putText(image, f"Current Label: {current_word}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv.LINE_AA)
        cv.putText(image, f"Samples: {current_sample_count}/{TARGET_SAMPLES_PER_LABEL}", (20, 220), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv.LINE_AA)
        
        status_text = "ACTIVE" if auto_record_active else "PAUSED"
        status_color = (0, 255, 0) if auto_record_active else (0, 165, 255) # Green for active, Orange for paused
        cv.putText(image, f"Auto-Record: {status_text}", (20, 340), cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 3, cv.LINE_AA)

    return image

# Movement indicator reference display (simplified as we only show wrist for static)
def landmark_path_drawing(image, m_sequence):
    # For static gestures, we can just highlight the wrist or a central point
    # For example, draw the wrist (landmark 0)
    if len(m_sequence) > 0 and m_sequence[0][0] != 0 and m_sequence[0][1] != 0:
        cv.circle(image, (m_sequence[0][0], m_sequence[0][1]), 5, (0, 255, 255), -1) # Yellow circle on wrist
    
    return image

# Global variable to store words from label.csv for access in record_static
words = []

def copy_data_files_to_other_modules():
    shutil.copy2(DATA_CSV_PATH, TRAINER_DATA_DIR)
    shutil.copy2(LABEL_PATH, TRAINER_DATA_DIR)
    # shutil.copy2(MOVEMENT_CSV_PATH, TRAINER_DATA_DIR) # Not copying movement data anymore
    shutil.copy2(LABEL_PATH, READER_MODEL_DIR)

def main():

    global words # Declare words as global to be accessible in record_static

    args = get_args()

    # Setup camera 
    cap = cv.VideoCapture(args.device, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    # Attributes setup
    mode = 0
    current_index = 0
    track_landmark_movement_index = 0 # Still present, but not actively used for movement tracking
    auto_record_active = False # New flag to control automatic recording

    # Load mediapipe model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode = args.use_static_image_mode, max_num_hands = MAX_NUM_HANDS,
                            min_detection_confidence = args.min_detection_confidence, 
                            min_tracking_confidence = args.min_tracking_confidence)
    
    # Load mediapipe drawing tool
    mp_drawing = mp.solutions.drawing_utils

    # Read CSV labels
    try:
        with open(LABEL_PATH, 'r', encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            for word_entry in reader:
                words.append(word_entry[0])
    except FileNotFoundError:
        print(f"Error: {LABEL_PATH} not found. Please ensure your label.csv file exists.")
        return # Exit if labels file is not found

    # Initialize sample counts for each label
    # This dictionary will store the number of samples recorded for each label index
    sample_counts = {i: 0 for i in range(len(words))}

    # CV Loop
    while True:

        # End process
        key = cv.waitKeyEx(10)
        if key == 27:   # ESC
            try:
                copy_data_files_to_other_modules()
                print("Data Copied!")
            except Exception as e:
                print(f"Data not copied to other modules. Error: {e}. Copy data files if needed.")
                pass
            break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            # If camera fails to read, set results to None to avoid NameError
            results = None 
            cv.putText(debug_image, "Camera Error!", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
            cv.imshow('Hand Gesture Recorder', debug_image) # Show debug image with error
            # You might want to break here or attempt to reinitialize camera
            continue # Continue to next loop iteration, hoping camera recovers


        # Mirror display
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        # Resize debug_image to args.width, args.height for consistent display and landmark mapping
        debug_image = cv.resize(debug_image, (args.width, args.height)) 

        # Mediapipe detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image) # results is always defined here
        image.flags.writeable = True

        # Mode selection
        arrow, mode, space_pressed, auto_record_active_returned = select_mode(key, mode, auto_record_active)
        
        # If backspace was pressed, auto_record_active_returned will be False.
        # Otherwise, if space was pressed, toggle auto_record_active.
        if space_pressed and mode == 1: # Only toggle if in recording mode
            auto_record_active = not auto_record_active
            print(f"Auto-recording {'ACTIVATED' if auto_record_active else 'PAUSED'}")
        elif key == 8: # If backspace is pressed, ensure auto_record_active is off
            auto_record_active = False

        # Word selection 
        if mode == 1:
            if arrow != 0: # If arrow key is pressed, change label
                current_index += arrow
                # Ensure current_index stays within bounds
                if current_index < 0:
                    current_index = 0
                elif current_index >= len(words):
                    current_index = len(words) - 1
                print(f"Switched to label: {words[current_index]}")
                # When switching labels, pause auto-recording to give user control
                auto_record_active = False 
        elif mode == 0:
            current_index = 0 # Reset current index when exiting recording mode
            auto_record_active = False # Ensure auto-record is off when not in recording mode


        # Record landmarks (always static now)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks: # Removed handedness as it's not needed
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks) # Adjusted call

                # Screen drawing for hand detection
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if mode == 1 and auto_record_active: # Only record if in recording mode AND auto-record is active
                    record_static(debug_image, DATA_CSV_PATH, current_index, landmark_list, sample_counts)
                
                # Update landmark movement sequence for visualization (even if not for recording)
                # This ensures the dot follows your hand's wrist.
                if landmark_list: # Check if landmark_list is not empty
                    LANDMARK_MOVEMENT_SEQUENCE.append(landmark_list[track_landmark_movement_index]) # Uses track_landmark_movement_index (wrist by default)
                else:
                    LANDMARK_MOVEMENT_SEQUENCE.append([0,0]) # Append zero if no hand detected

        else:
            cv.putText(debug_image, "Show me your hands!", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
            LANDMARK_MOVEMENT_SEQUENCE.append([0, 0]) # Append zero if no hand detected
            HAND_MOVEMENT_SEQUENCE.append(ZERO_LIST) # Still append zero list to maintain deque length

        # Landmark movement sequence drawing (simplified to just show wrist for static gestures)
        debug_image = landmark_path_drawing(debug_image, LANDMARK_MOVEMENT_SEQUENCE)
        
        # Screen drawing for hints and current state
        screen_drawing(debug_image, mode, words[current_index] if words else "N/A", sample_counts[current_index], auto_record_active)
        
        # Screen Reflection
        cv.imshow('Hand Gesture Recorder', debug_image)
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()