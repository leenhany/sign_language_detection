#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import shutil
import argparse
import itertools
import numpy as np
from collections import deque
from collections import Counter

import cv2 as cv
import mediapipe as mp

from model.classifier import Classifier

MAX_NUM_HANDS = 2
LABEL_PATH = 'Reader/model/label.csv'
READER_MODEL_DIR = 'Reader/model'
TRAINED_STATIC_MODEL_PATH = 'Trainer/model/model.tflite'
TRAINED_MOVEMENT_MODEL_PATH = 'Trainer/model/m_model.tflite'
STATIC_MODEL_PATH = shutil.copy2(TRAINED_STATIC_MODEL_PATH, READER_MODEL_DIR)
MOVEMENT_MODEL_PATH = shutil.copy2(TRAINED_MOVEMENT_MODEL_PATH, READER_MODEL_DIR)

DETECTION_THRESHOLD = 0.8
MOVEMENT_THRESHOLD = 0.95
OUTPUT_COUNT = 5

SEQUENCE_FRAME_NUM = 21
MOVEMENT_HISTORY = deque([[[0] * 2] * 21] * 21, maxlen=SEQUENCE_FRAME_NUM)
MOVEMENT_DICT = dict([(0, -1), (1, 9), (2, 25)])

# TODO: MOVEMENT SIGN EXPANSION
# MOVEMENT_LABEL_PATH = 'Reader/model/m_label.csv'

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--device", type=int, default=0)
	parser.add_argument("--width", help='cap width', type=int, default=960)
	parser.add_argument("--height", help='cap height', type=int, default=540)

	parser.add_argument('--use_static_image_mode', action='store_true')
	parser.add_argument("--min_detection_confidence",
						help='min_detection_confidence',
						type=float,
						default=0.7)
	parser.add_argument("--min_tracking_confidence",
						help='min_tracking_confidence',
						type=int,
						default=0.5)

	args = parser.parse_args()

	return args

def main():

	# Argument parsing 
	args = get_args()

	cap_device = args.device
	cap_width = args.width
	cap_height = args.height

	use_static_image_mode = args.use_static_image_mode
	min_detection_confidence = args.min_detection_confidence
	min_tracking_confidence = args.min_tracking_confidence

	# OpenCV Camera preparation 
	cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
	cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
	cap.set(cv.CAP_PROP_FPS, 30)
	cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

	# Mediapipe Model load 
	mp_hands = mp.solutions.hands
	hands = mp_hands.Hands(
		static_image_mode=use_static_image_mode,
		max_num_hands=MAX_NUM_HANDS,
		min_detection_confidence=min_detection_confidence,
		min_tracking_confidence=min_tracking_confidence,
	)

	# Mediapipe landmarks drawing tool load
	mp_drawing = mp.solutions.drawing_utils

	# Init Classifiers
	sign_classifier = Classifier(model_path=STATIC_MODEL_PATH)
	movement_sign_classifier = Classifier(model_path=MOVEMENT_MODEL_PATH)

	# Read labels 
	with open(LABEL_PATH,encoding='utf-8-sig') as f:
		classifier_labels = csv.reader(f)
		classifier_labels = [
			row[0] for row in classifier_labels
		]
	
	# TODO: Movement Labels
	# with open(MOVEMENT_LABEL_PATH,encoding='utf-8-sig') as f:
	# 	movement_classifier_labels = csv.reader(f)
	# 	movement_classifier_labels = [
	# 		row[0] for row in movement_classifier_labels
	# ]
		
	while True:

		# Process Key (ESC: end) 
		key = cv.waitKey(10)
		if key == 27:  # ESC
			break

		# Camera capture
		ret, image = cap.read()
		if not ret:
			break
		image = cv.flip(image, 1)  # Mirror display

		# detection corrupts the image 
		debug_image = copy.deepcopy(image)
		debug_image = cv.resize(debug_image, (cap_width * 2, cap_height * 2))

		# Detection implementation 
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

		results = hands.process(image)
		
		if results.multi_hand_landmarks is not None:
			for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
				# Landmark calculation
				landmark_list, landmark_list_3D = calc_landmark_list(debug_image, hand_landmarks)
				MOVEMENT_HISTORY.append(landmark_list)

				# Handedness Detection
				play_hand = handedness.classification[0].label[0:]

				# Conversion to relative coordinates / normalized coordinates
				pre_processed_landmark_list = pre_process_landmark(landmark_list)
				pre_processed_movement_list = pre_process_hand_movement(debug_image, MOVEMENT_HISTORY)
				# print(np.shape(pre_processed_movement_list)) (SEQ*(21*2),)

				# Hand sign classification
				N_hand_sign_id, N_hand_sign_probs = sign_classifier(pre_processed_landmark_list, OUTPUT_COUNT)
				N_move_sign_id, N_move_sign_probs = movement_sign_classifier(pre_processed_movement_list, OUTPUT_COUNT)
				# Key-value replacement
				N_move_sign_id = list(MOVEMENT_DICT[move_sign_id] for move_sign_id in N_move_sign_id)


				# Landmarks visulaization
				mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
				brect = calc_bounding_rect(debug_image, hand_landmarks)
				debug_image = draw_bounding_rect(debug_image, brect)


				# Output results
				additional_display_height = 0
				additional_display_width = 0
				
				if(play_hand == 'Right'):
					additional_display_width = cap_width*2 - 300
				
				static_tuple_list = list(zip(N_hand_sign_id, N_hand_sign_probs))
				movement_tuple_list = list(zip(N_move_sign_id, N_move_sign_probs))


				# Default static list
				display_list = static_tuple_list

				if(movement_tuple_list[0][0] != -1):
					display_list = movement_tuple_list

				for N_detected_elements in display_list:
					
					additional_display_height += 40

					# Prob > threshold*100% display
					threshold = DETECTION_THRESHOLD if display_list == static_tuple_list else MOVEMENT_THRESHOLD
					if N_detected_elements[1] > threshold:
						display_prob = "{:.2f}%".format(N_detected_elements[1] * 100)
						cv.putText(debug_image, str(classifier_labels[N_detected_elements[0]]) + " : " + str(display_prob), 
							   (20 + additional_display_width, 40 + additional_display_height), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

				#print(play_hand)	 

		else:
			#print("NO HAND")
			MOVEMENT_HISTORY.append([[0] * 2] * 21)


		# Display
		cv.imshow('Hand Gesture Reader', debug_image)

	cap.release()
	cv.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
	image_width, image_height = image.shape[1], image.shape[0]

	landmark_array = np.empty((0, 2), int)

	for _, landmark in enumerate(landmarks.landmark):
		landmark_x = min(int(landmark.x * image_width), image_width - 1)
		landmark_y = min(int(landmark.y * image_height), image_height - 1)

		landmark_point = [np.array((landmark_x, landmark_y))]

		landmark_array = np.append(landmark_array, landmark_point, axis=0)

	x, y, w, h = cv.boundingRect(landmark_array)

	return [x, y, x + w, y + h]

def draw_bounding_rect(image, brect):
	
	# Outer rectangle
	cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
					(0, 0, 255), 1)

	return image

def calc_landmark_list(image, landmarks):
	image_width, image_height = image.shape[1], image.shape[0]

	landmark_point = []
	landmark_point_3D = []

	# Keypoint
	for _, landmark in enumerate(landmarks.landmark):
		landmark_x = min(int(landmark.x * image_width), image_width - 1)
		landmark_y = min(int(landmark.y * image_height), image_height - 1)
		landmark_z = landmark.z

		landmark_point.append([landmark_x, landmark_y])
		landmark_point_3D.append([landmark_x, landmark_y, landmark_z])

	return landmark_point, landmark_point_3D

def pre_process_landmark(landmark_list):
	temp_landmark_list = copy.deepcopy(landmark_list)

	# Convert to relative coordinates
	base_x, base_y = 0, 0
	for index, landmark_point in enumerate(temp_landmark_list):
		if index == 0:
			base_x, base_y = landmark_point[0], landmark_point[1]

		temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
		temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

	# Convert to a one-dimensional list
	temp_landmark_list = list(
		itertools.chain.from_iterable(temp_landmark_list))

	# Normalization
	max_value = max(list(map(abs, temp_landmark_list)))

	def normalize_(n):
		return n / max_value

	temp_landmark_list = list(map(normalize_, temp_landmark_list))

	return temp_landmark_list

def pre_process_landmark_movement(image, m_sequence):
	w, h = image.shape[1], image.shape[0]
	movement_sequence = copy.deepcopy(m_sequence)

	# Convert to relative coordinates
	base_x, base_y = 0, 0
	for index, movement_point in enumerate(movement_sequence):
		if index == 0:
			base_x, base_y = movement_point[0], movement_point[1]

		movement_sequence[index][0] = (movement_sequence[index][0] - base_x) / w
		movement_sequence[index][1] = (movement_sequence[index][1] - base_y) / h

	return movement_sequence

def pre_process_hand_movement(image, m_sequence):
	width, height = image.shape[1], image.shape[0]
	copy_of_m_sequence = copy.deepcopy(m_sequence) #(SEQ, 21, 2)
	
	landmarks_seq = [] #(SEQ, 21*2)
	
	# Save first corrdinates for replacing start sequence landmarks position
	first_seq_landmarks = copy.deepcopy(copy_of_m_sequence[0])
	# print(first_seq_landmarks)

	for seq in range(0, SEQUENCE_FRAME_NUM):
		landmarks_seq.append(list(np.array(copy_of_m_sequence[seq]).flatten()))

	# Convert to relative coordinates
	first_seq_landmarks_flatten = copy.deepcopy(landmarks_seq[0])
	
	for seq in range(0, SEQUENCE_FRAME_NUM):
		for landmark in range(0, (21*2)):
			if ((landmark % 2) == 0):
				landmarks_seq[seq][landmark] = (landmarks_seq[seq][landmark] - first_seq_landmarks_flatten[landmark]) / width
			else:
				landmarks_seq[seq][landmark] = (landmarks_seq[seq][landmark] - first_seq_landmarks_flatten[landmark]) / height

	
	if(first_seq_landmarks[0][0] != 0 and first_seq_landmarks[0][1] != 0):
		landmarks_seq[0] = pre_process_landmark(first_seq_landmarks)
	
	# Convert to a one-dimensional list #(SEQ*21*2,)
	landmarks_seq = list(itertools.chain.from_iterable(landmarks_seq))

	return landmarks_seq


if __name__ == '__main__':
	main()
