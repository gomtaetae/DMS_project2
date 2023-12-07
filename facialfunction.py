import cv2
import time
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import facial_tracking.conf as conf
from facial_tracking.eye import Eye
from facial_tracking.faceMesh import FaceMesh
from facial_tracking.iris import Iris
import pygame


class FacialProcessor:
	def __init__(self):
		self.face_mesh = FaceMesh()
		self.detected = False
		self.eyes_status = "Eyes not detected"
		# **아래부분 추가
		self.face_mesh_enabled = True  # FaceMesh 처리를 기본적으로 활성화
	
	# face mesh toggle 키 만들기
	def toggle_face_mesh(self):
		self.face_mesh_enabled = not self.face_mesh_enabled
	
# ======================================================================================================================
	
	def process_frame(self, frame):
		self.face_mesh.process_frame(frame)
		
		# Add your logic to detect eyes and update self.detected and self.eyes_status
		# For demonstration purposes, let's assume eyes are always detected.
		self.detected = True
		self.eyes_status = "Eyes detected"
		
		if self.face_mesh.mesh_result.multi_face_landmarks:
			for face_landmarks in self.face_mesh.mesh_result.multi_face_landmarks:
				left_iris = Iris(frame, face_landmarks, conf.LEFT_EYE)
				right_iris = Iris(frame, face_landmarks, conf.RIGHT_EYE)
				left_iris.draw_iris(True)
				right_iris.draw_iris(True)
				
				# Add eye-tracking logic here
				left_eye = Eye(frame, face_landmarks, conf.LEFT_EYE)
				right_eye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
				left_eye.iris.draw_iris()
				right_eye.iris.draw_iris()
				
				if left_eye.eye_closed() or right_eye.eye_closed():
					self.eyes_status = 'Eye closed'
				else:
					if left_eye.gaze_right() and right_eye.gaze_right():
						self.eyes_status = 'Gazing right'
					elif left_eye.gaze_left() and right_eye.gaze_left():
						self.eyes_status = 'Gazing left'
					elif left_eye.gaze_center() and right_eye.gaze_center():
						self.eyes_status = 'Gazing center'
		
		# Draw the face mesh after drawing the iris
		self.face_mesh.draw_mesh()


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


thresh = 0.2
frame_check = 15
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

detect = dlib.get_frontal_face_detector()
path = '/Users/bagsangbeom/PycharmProjects/DMS_project/haar/'
predict = dlib.shape_predictor(path + 'shape_predictor_68_face_landmarks.dat')
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("/Users/bagsangbeom/PycharmProjects/DMS_project/sound/alarm.wav")

# Adjust the capture device index (0 or 1) based on your camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width of the frames
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1240)  # Set the height of the frames
cap.set(cv2.CAP_PROP_FPS, 30)  # Set the frames per second

flag = 0
facial_processor = FacialProcessor()
ptime = 0
ctime = 0
leftEAR = 0.0
rightEAR = 0.0

while True:
	ret, frame = cap.read()
	
	# Check if frame is not None before processing
	if frame is not None:
		frame = imutils.resize(frame, width=800)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Use the face detector
		subjects = detect(gray)
		
		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)
			
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			
			if ear < thresh:
				flag += 1
				print(flag)
				
				if flag == frame_check:
					cv2.putText(frame, "", (10, 30),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					alarm_sound.play()
					print("Drowsiness Detected")
				elif flag > frame_check:
					cv2.putText(frame, "", (10, 30),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				flag = 0
		
		facial_processor.process_frame(frame)
		
		ctime = time.time()
		fps = 1 / (ctime - ptime)
		ptime = ctime
		
		frame = cv2.flip(frame, 1)
		cv2.putText(frame, f'FPS: {int(fps)}', (30, 30), 0, 0.6,
		            conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)
		
		cv2.putText(frame, f'{facial_processor.eyes_status}', (30, 70), 0, 0.8,
		            conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
		
		# Flip text rendering for the left and right ears
		cv2.putText(frame, "Left EAR {:.2f}".format(leftEAR), (30, 550),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
		cv2.putText(frame, "Right EAR {:.2f}".format(rightEAR), (30, 570),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
		
		cv2.imshow('Facial tracking', frame)
	
	# Delay to control the frames per second
	key = cv2.waitKey(1)
	if key == ord('q'):
		break

# Release the capture and close all windows
cv2.destroyAllWindows()
cap.release()
