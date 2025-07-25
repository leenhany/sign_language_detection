import csv
import copy
import shutil
import argparse
import itertools
import numpy as np
import time
import cv2 as cv
import mediapipe as mp


# Import your Classifier class
from model.classifier import Classifier

# --- Constants ---
MAX_NUM_HANDS = 1 
LABEL_PATH = 'Reader/model/label.csv'
READER_MODEL_DIR = 'Reader/model'
STATIC_MODEL_PATH = 'Reader/model/model.tflite'
TRAINER_STATIC_MODEL_PATH = 'Trainer/model/model.tflite'

DETECTION_THRESHOLD = 0.1
OUTPUT_COUNT = 1

# --- Global variables for sign holding and display bar ---
last_detected_sign = ""
time_last_sign_changed = 0.0
confirmed_sign_text = ""
display_bar_text = ""
CONFIRMATION_HOLD_TIME = 1.1

# --- Constants for Bottom Text Bar ---
TEXT_BAR_MIN_HEIGHT = 50
TEXT_BAR_X_PADDING = 15
TEXT_BAR_Y_PADDING_TOP = 35
TEXT_BAR_LINE_HEIGHT = 40
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2
BAR_BACKGROUND_COLOR = (0, 0, 0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)  # Increased width
    parser.add_argument("--height", type=int, default=720)   # Increased height
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()



def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 255), 1)
    return image

def calc_landmark_list(image, landmarks):
    return [[int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])] for landmark in landmarks.landmark]

def pre_process_landmark(landmark_list):
    if not landmark_list:
        return [0.0] * 42
    base_x, base_y = landmark_list[0]
    temp_landmark_list = [(x - base_x, y - base_y) for x, y in landmark_list]
    flattened_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, flattened_list)) or 1
    return [n / max_value for n in flattened_list]

def get_wrapped_text_height(text, max_width, font, font_scale, thickness, line_height):
    words = text.split(' ')
    if not words or text.strip() == "":
        return 0
    current_line_width = 0
    num_lines = 1
    for word in words:
        word_width, _ = cv.getTextSize(word + " ", font, font_scale, thickness)[0]
        if current_line_width + word_width > max_width and current_line_width != 0:
            num_lines += 1
            current_line_width = word_width
        else:
            current_line_width += word_width
    return num_lines * line_height

def draw_text_with_wrap(image, text, x_start, y_start, max_width, font, font_scale, color, thickness, line_height):
    words = text.split(' ')
    if not words or text.strip() == "":
        return
    current_line_text = ""
    current_y = y_start
    for word in words:
        test_line_text = current_line_text + word + " "
        test_line_width, _ = cv.getTextSize(test_line_text, font, font_scale, thickness)[0]
        if test_line_width > max_width and current_line_text.strip() != "":
            cv.putText(image, current_line_text.strip(), (x_start, current_y), font, font_scale, color, thickness, cv.LINE_AA)
            current_y += line_height
            current_line_text = word + " "
        else:
            current_line_text = test_line_text
    if current_line_text.strip() != "":
        cv.putText(image, current_line_text.strip(), (x_start, current_y), font, font_scale, color, thickness, cv.LINE_AA)

def main():
    global last_detected_sign, time_last_sign_changed, confirmed_sign_text, display_bar_text

    args = get_args()
    cap = cv.VideoCapture(args.device, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=args.use_static_image_mode,
                           max_num_hands=MAX_NUM_HANDS,
                           min_detection_confidence=args.min_detection_confidence,
                           min_tracking_confidence=args.min_tracking_confidence)
    mp_drawing = mp.solutions.drawing_utils

    try:
        shutil.copy2(TRAINER_STATIC_MODEL_PATH, STATIC_MODEL_PATH)
        print(f"Copied static model from {TRAINER_STATIC_MODEL_PATH} to {STATIC_MODEL_PATH}")
    except FileNotFoundError:
        print(f"Warning: Static model not found at {TRAINER_STATIC_MODEL_PATH}.")
    except Exception as e:
        print(f"An error occurred while copying static model: {e}")

    sign_classifier = Classifier(model_path=STATIC_MODEL_PATH)

    try:
        with open(LABEL_PATH, encoding='utf-8-sig') as f:
            classifier_labels = [row[0] for row in csv.reader(f)]
        print(f"Loaded {len(classifier_labels)} labels from {LABEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Label file not found at {LABEL_PATH}.")
        return

    time_last_sign_changed = time.time()

    while True:
        key = cv.waitKey(10)
        if key == 27:
            break
        elif key == ord('c'):
            display_bar_text = ""
            last_detected_sign = ""
            time_last_sign_changed = time.time()
            confirmed_sign_text = ""
        elif key == ord('\b'):
            if display_bar_text:
                display_bar_text = ' '.join(display_bar_text.split()[:-1]) + " "
                last_detected_sign = ""
                time_last_sign_changed = time.time()
                confirmed_sign_text = ""

        ret, image = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        current_time = time.time()

        if results.multi_hand_landmarks is not None:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0]

            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            play_hand = handedness.classification[0].label

            processed_landmark_list_for_classifier = landmark_list
            if play_hand == 'Left':
                processed_landmark_list_for_classifier = [[image.shape[1] - x, y] for x, y in landmark_list]

            pre_processed_landmark_list = pre_process_landmark(processed_landmark_list_for_classifier)
            N_hand_sign_id, N_hand_sign_probs = sign_classifier(pre_processed_landmark_list, OUTPUT_COUNT)

            mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            debug_image = draw_bounding_rect(debug_image, calc_bounding_rect(debug_image, hand_landmarks))

            current_detected_sign_label = ""
            current_detected_sign_confidence = 0.0

            for N_detected_elements in zip(N_hand_sign_id, N_hand_sign_probs):
                display_prob = "{:.2f}%".format(N_detected_elements[1] * 100)
                if N_detected_elements[1] > DETECTION_THRESHOLD:
                    current_detected_sign_label = classifier_labels[N_detected_elements[0]]
                    current_detected_sign_confidence = N_detected_elements[1]
                    cv.putText(debug_image, f"{current_detected_sign_label} : {display_prob}",
                               (20, 80), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv.LINE_AA)

            if current_detected_sign_label and current_detected_sign_confidence > DETECTION_THRESHOLD:
                if current_detected_sign_label != last_detected_sign:
                    last_detected_sign = current_detected_sign_label
                    time_last_sign_changed = current_time
                    confirmed_sign_text = ""
                else:
                    time_held = current_time - time_last_sign_changed
                    if time_held >= CONFIRMATION_HOLD_TIME:
                        if confirmed_sign_text != last_detected_sign:
                            confirmed_sign_text = last_detected_sign
                            display_bar_text += confirmed_sign_text + " "
                            last_detected_sign = ""
                            time_last_sign_changed = current_time

            else:
                last_detected_sign = ""
                time_last_sign_changed = current_time
                confirmed_sign_text = ""

        else:
            cv.putText(debug_image, "No Hand Detected", (20, 40), FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS, cv.LINE_AA)
            last_detected_sign = ""
            time_last_sign_changed = current_time
            confirmed_sign_text = ""

        wrapped_text_display_width = args.width - 2 * TEXT_BAR_X_PADDING
        total_text_height_needed = get_wrapped_text_height(display_bar_text, wrapped_text_display_width, FONT, FONT_SCALE, FONT_THICKNESS, TEXT_BAR_LINE_HEIGHT)
        actual_bar_height = max(TEXT_BAR_MIN_HEIGHT, total_text_height_needed + TEXT_BAR_Y_PADDING_TOP * 2)
        bar_y_start = args.height - actual_bar_height

        cv.rectangle(debug_image, (0, bar_y_start), (args.width, args.height), BAR_BACKGROUND_COLOR, -1)
        draw_text_with_wrap(debug_image, display_bar_text, TEXT_BAR_X_PADDING, bar_y_start + TEXT_BAR_Y_PADDING_TOP, wrapped_text_display_width, FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, TEXT_BAR_LINE_HEIGHT)

        cv.imshow('Hand Gesture Reader', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()