'''
Program for creating dataset of faceparts using webcam
'''

from collections import OrderedDict
import time

import numpy as np
import dlib
import cv2

# Custom libs
import imutils
from imutils import face_utils

def show_part(image, name, shape_array):
    landmarks = np.append(*shape_array, axis=0)
    (x, y, w, h) = cv2.boundingRect(np.array([landmarks]))
    roi = image[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

    # Show the particular face part
    if name == 'right_eye':
        roi = cv2.flip(roi, 1)
    cv2.imshow("ROI", roi)

    # Save image
    # timestamp = int(time.time())
    # folder = 'train_dataset/mouth/open'
    # cv2.imwrite(f'{folder}/{timestamp}.png', roi) 
    cv2.waitKey(0)

def get_faceparts(face_rect, predictor, gray, image):
    # Determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, face_rect)
    landmarks = face_utils.shape_to_np(shape)

    landmarks_dict = {
        # 'left_eye': [landmarks[22:27], landmarks[42:48]],
        # 'right_eye': [landmarks[17:22], landmarks[36:42]],
        'mouth': [landmarks[48:68], [landmarks[6], landmarks[10], landmarks[33]]]
    }
    for name, facepart_lm in landmarks_dict.items():
        show_part(image, name, facepart_lm)

def main():
    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        _, image = cap.read()

        # Load the input image, resize it, and convert it to grayscale
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        face_rects = detector(gray, 1)
        if face_rects:
            # Choose only first face
            face_rect = face_rects[0]
            get_faceparts(face_rect, predictor, gray, image)

        counter += 1
        if counter == 10:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()