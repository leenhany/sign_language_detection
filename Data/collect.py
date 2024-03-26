# TODO: Description

import csv
import copy
import shutil
import argparse
import itertools
from collections import deque

import numpy as np
import cv2 as cv
import mediapipe as mp

# Attributes Constants
MAX_NUM_HANDS = 1
SEQUENCE_FRAME_NUM = 21
LANDMARK_MOVEMENT_SEQUENCE = deque(maxlen = SEQUENCE_FRAME_NUM)
# TODO: Display landmark movement distionary
HAND_MOVEMENT_SEQUENCE = deque(maxlen = SEQUENCE_FRAME_NUM)
ZERO_LIST = list(np.zeros((21, 2)))
ZERO_LIST_3D = list(np.zeros((21, 3)))

# Path Constants
# TODO: setup remote path
DATA_CSV_PATH = 'Data/data.csv'
LABEL_PATH = 'Data/label.csv'
MOVEMENT_CSV_PATH = 'Data/m_data.csv'
TRAINER_DATA_DIR = 'Trainer/data'
READER_MODEL_DIR = 'Reader/model'

# Module System Functions
def copy_data_files_to_training_module():
    shutil.copy2(DATA_CSV_PATH, TRAINER_DATA_DIR)
    shutil.copy2(LABEL_PATH, TRAINER_DATA_DIR)
    shutil.copy2(LABEL_PATH, READER_MODEL_DIR)
    shutil.copy2(MOVEMENT_CSV_PATH, TRAINER_DATA_DIR)

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

def select_mode(key, mode):
	
	number = 0
	trigger = 0
	
	if key == 8:		# Backspace: 8
		mode = 0
	if key == 13:		# Enter:	 13
		mode = 1
	
	if key == 2555904:	# Right Key: 2555904
		number = 1
	if key == 2424832:	# Left Key : 2424832
		number = -1
	
	if key == 32:		# Space bar: 32
		trigger = 1
	
	return number, mode, trigger


# Landmarks Processing Functions
def calc_landmark_list(image, landmarks, handedness):
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


	return landmark_point, landmark_point_3D, handedness

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

def pre_process_hand_movement(image, m_sequence):
	width, height = image.shape[1], image.shape[0]
	copy_of_m_sequence = copy.deepcopy(m_sequence) #(SEQ, 21, 2)
	
	landmarks_seq = [] #(SEQ, 21*2)
	

	for seq in range(0, SEQUENCE_FRAME_NUM):
		landmarks_seq.append(list(np.array(copy_of_m_sequence[seq]).flatten()))

	# Convert to relative coordinates
	first_seq_landmarks = copy.deepcopy(landmarks_seq[0])

	for seq in range(0, SEQUENCE_FRAME_NUM):
		for landmark in range(0, (21*2)):
			if ((landmark % 2) == 0):
				landmarks_seq[seq][landmark] = (landmarks_seq[seq][landmark] - first_seq_landmarks[landmark]) / width
			else:
				landmarks_seq[seq][landmark] = (landmarks_seq[seq][landmark] - first_seq_landmarks[landmark]) / height

	print(np.shape(landmarks_seq))
	# Convert to a one-dimensional list #(SEQ*21*2,)
	landmarks_seq = list(itertools.chain.from_iterable(landmarks_seq))

	return landmarks_seq


# Data Recording Functions
def record_data(path, label_index, pp_list):	
	with open(path, 'a', newline="") as f:
		writer = csv.writer(f)
		writer.writerow([label_index, *pp_list])
		#writer.writerow([_handedness])
	return

def record_static(image, csv_path, index_in_label, landmark_list):
	
	# Write to csv
	record_data(csv_path, index_in_label, pre_process_landmark(landmark_list))
	print("Recorded")
	cv.putText(image, "RECORDED!!", (20,220), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

	return 

def record_movement(image, hand_movement_seq_list, csv_path, index_in_label, landmark_movement_seq_list):

	record_label = -1

	match index_in_label:
		case 9:
			record_label = 1
		case 25:
			record_label = 2
		case 26:
			record_label = 0
		case 27:
			record_label = 3

	# Write to csv
	record_data(csv_path, record_label, pre_process_hand_movement(image, hand_movement_seq_list))
	
	# Kill Visualization
	last_seq = landmark_movement_seq_list[-1]
	landmark_movement_seq_list.clear()
	landmark_movement_seq_list.append(last_seq)
	
	
	print("Recorded")
	cv.putText(image, "RECORDED!!", (20,220), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

	return


### TODO: Visualization Functions
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

def visualization_pre_process_hand_movement(image, m_sequence):
	
	copy_of_m_sequence = copy.deepcopy(m_sequence) # shape(SEQ, 21, 2)

	landmark_sequences = [] # goal shape(21, SEQ, 2)

	for landmark in range(0, 21):
		landmark_seq_cord = []
		for frame in range(0, SEQUENCE_FRAME_NUM):
			landmark_seq_cord.append(copy_of_m_sequence[frame][landmark])
		landmark_seq_cord = list(itertools.chain.from_iterable(
			pre_process_landmark_movement(image, landmark_seq_cord)))

		landmark_sequences.append(landmark_seq_cord)

	landmark_sequences = list(itertools.chain.from_iterable(landmark_sequences))

	return landmark_sequences

def screen_drawing(image, mode):
	
	if mode == 0:
		cv.putText(image, "Press ENTER to enter recording mode.", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)

	elif mode == 1:
		cv.putText(image, "Press ARROW KEYS to choose letter.", (20,100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)
		cv.putText(image, "Press SPACE to record.", (20,160), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)
		cv.putText(image, "Press BACKSPACE to exit recording mode.", (20,280), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)

	return image

def landmark_path_drawing(image, m_sequence):
	for _, seq in enumerate(m_sequence):
		if(seq[0] != 0 and seq[1] != 0):
			cv.circle(image, (seq[0], seq[1]), 2, (0, 255, 0), 20)
	
	return image

# Main
def main():

	args = get_args()

	# Setup camera 
	cap = cv.VideoCapture(args.device, cv.CAP_DSHOW)
	cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
	cap.set(cv.CAP_PROP_FPS, 30)
	cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

	# Attributes setup
	# Set mode
	mode = 0
	# Set index
	current_index = 0
	# TODO: Set movement visualization index (group?)
	track_landmark_movement_index = 0


	# Load mediapipe model
	mp_hands = mp.solutions.hands
	hands = mp_hands.Hands(static_image_mode = args.use_static_image_mode, max_num_hands = MAX_NUM_HANDS,
						min_detection_confidence = args.min_detection_confidence, 
						min_tracking_confidence = args.min_tracking_confidence)
	# Load mediapipe drawing tool
	mp_drawing = mp.solutions.drawing_utils



	# Read CSV Labels
	with open(LABEL_PATH, 'r', encoding="utf-8-sig`") as file:
		reader = csv.reader(file)
		words = []
		for word in reader:
			words.append(word[0])
	
	# TODO: Read movemnt CSV labels


	# CV Loop
	while True:

		# End process
		key = cv.waitKeyEx(10)
		if key == 27:  # ESC

			# Copy csv files to Trainer Module
			try:
				copy_data_files_to_training_module()
			except:
				print("Data not copied for training module. Copy data files if needed.")
				pass
			break

		# Camera capture
		ret, image = cap.read()
		if not ret:
			break

		# Mirror Display
		image = cv.flip(image, 1)
		debug_image = copy.deepcopy(image)
		debug_image = cv.resize(debug_image, (args.width*2, args.height*2))

		# Mediapipe detection implementation
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

		image.flags.writeable = False
		results = hands.process(image)
		image.flags.writeable = True

		# Mode selection
		arrow, mode, trigger = select_mode(key, mode)

		# Word Selection 
		if (arrow != 0 and mode == 1):
			current_index += arrow
			if current_index < 0:
				current_index = 0
			elif current_index >= len(words):
				current_index = len(words) - 1
		elif mode == 0:
			current_index = 0
			
		# TODO: Display landmark movement selection

		if results.multi_hand_landmarks is not None:
			for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

				# Landmark calculation
				landmark_list, landmark_list_3D, _handedness = calc_landmark_list(debug_image, hand_landmarks, handedness.classification[0].label[0:])

				# Screen Drawing for hand detection
				mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

				# Conversion to relative coordinates / normalized coordinates
				# pre_processed_landmark_list = pre_process_landmark(landmark_list)
				#pre_processed_movement_list = pre_process_movement(debug_image, MOVEMENT_SEQUENCE)

				# Record
				if(mode == 1):
					
					# Not J and Z
					if(current_index != 9 and current_index != 25 and current_index < 26):

						# Clear movement lists
						LANDMARK_MOVEMENT_SEQUENCE.clear()
						HAND_MOVEMENT_SEQUENCE.clear()
						
						if(trigger == 1):
							record_static(image, DATA_CSV_PATH, current_index, landmark_list)
					# J with m_label_index 1
					elif(current_index == 9):

						# Track pinky finger tip
						LANDMARK_MOVEMENT_SEQUENCE.append(landmark_list[20])

						# Track every landmark
						HAND_MOVEMENT_SEQUENCE.append(landmark_list)
						
						
						if(trigger == 1):
							record_movement(debug_image, HAND_MOVEMENT_SEQUENCE, MOVEMENT_CSV_PATH, current_index, LANDMARK_MOVEMENT_SEQUENCE)

					# Z with m_label_index 2
					elif(current_index == 25):
						
						# Track index finger tip
						LANDMARK_MOVEMENT_SEQUENCE.append(landmark_list[8])

						# Track every landmark
						HAND_MOVEMENT_SEQUENCE.append(landmark_list)

						if(trigger == 1):
							record_movement(debug_image, HAND_MOVEMENT_SEQUENCE, MOVEMENT_CSV_PATH, current_index, LANDMARK_MOVEMENT_SEQUENCE)
					
					# NONE with m_label_index 0
					elif(current_index >= 26):
						
						# Track custom index landmark
						LANDMARK_MOVEMENT_SEQUENCE.append(landmark_list[track_landmark_movement_index])

						# Track every landmark
						HAND_MOVEMENT_SEQUENCE.append(landmark_list)

						if(trigger == 1):
							record_movement(debug_image, HAND_MOVEMENT_SEQUENCE, MOVEMENT_CSV_PATH, current_index, LANDMARK_MOVEMENT_SEQUENCE)
					else:
						#TODO: sign expansions (add another m_label file),
						#	   m_label index 0 should be 'None'
						print("THIS WILL NEVER PRINT")


					# Hint Display
					cv.putText(debug_image, "You are now recording the letter: " + words[current_index], (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 155), 3, cv.LINE_AA)
				
				# Screen Drawing for hand recording
				screen_drawing(debug_image, mode)
		
		else:
			cv.putText(debug_image, "Show me your hands!", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
			LANDMARK_MOVEMENT_SEQUENCE.append([0, 0])
			HAND_MOVEMENT_SEQUENCE.append(ZERO_LIST)


		# Landmark movement sequence drawing
		debug_image = landmark_path_drawing(debug_image, LANDMARK_MOVEMENT_SEQUENCE)
		
		# Screen Reflection
		cv.imshow('Hand Gesture Recorder', debug_image)
		

	#TODO: Write to database
	
	cap.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	main()