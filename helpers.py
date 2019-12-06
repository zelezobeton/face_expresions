from collections import OrderedDict
import time

import numpy as np
import dlib
import cv2

# Custom libs
import imutils
from imutils import face_utils

def create_dataset(rects, predictor, gray, image):
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        EMOTICON_LANDMARKS_IDXS = [
            ("mouth", (48, 68), (None, None)),
            # ("right_eye", (17, 22), (36, 42)),
            # ("left_eye", (22, 27), (42, 48)),
        ]

        # loop over the face parts individually
        for (name, (i, j), (k,l)) in EMOTICON_LANDMARKS_IDXS:
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
            
            if k is None and l is None:
                landmarks = np.append(shape[i:j], [shape[6], shape[10], shape[33]], axis=0)
            else:
                landmarks = np.concatenate((shape[i:j], shape[k:l]), axis=0)
                
            for (x, y) in landmarks:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            
            (x, y, w, h) = cv2.boundingRect(np.array([landmarks]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            if name == 'right_eye':
                roi = cv2.flip(roi, 1)
            cv2.imshow("ROI", roi)
            # timestamp = int(time.time())
            # folder = 'train_dataset/mouth/open'
            # cv2.imwrite(f'{folder}/{timestamp}.png', roi) 
            cv2.waitKey(0)

def main():
    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        _, image = cap.read()

        # Load the input image, resize it, and convert it to grayscale
        # image = cv2.imread(args["image"])
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        rects = detector(gray, 1)

        create_dataset(rects, predictor, gray, image)
        counter += 1
        if counter == 10:
            break

        # Show the output image with the face detections + facial landmarks
        # cv2.imshow("Output", image)
        # if cv2.waitKey(1) == 27:
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()