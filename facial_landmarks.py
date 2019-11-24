'''
Program for detecting eye and mouth expressions.

USE WITH:
    python facial_landmarks.py -p shape_predictor_68_face_landmarks.dat

Using these tuts:
- https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
- https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

Pretrained facial landmarks detector can be downloaded here:
- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

TODO:
- Find out how to get rectangle around eye+eyebrow
'''
import argparse
from collections import OrderedDict
import time

import numpy as np
import dlib
import cv2

# Custom libs
import imutils
from imutils import face_utils

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    # ap.add_argument("-i", "--image", required=True,
    #     help="path to input image")
    args = vars(ap.parse_args())

    return args

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()

	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220), (0, 0, 255)]

	# loop over the facial landmark regions individually
	for (i, name) in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = face_utils.FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]

		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)

		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

	# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	# return the output image
	return output

def draw_features(rects, predictor, gray, image):
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            cv2.imshow("ROI", roi)
            cv2.imshow("Image", clone)
            cv2.waitKey(0)

        # visualize all facial landmarks with a transparent overlay
        output = visualize_facial_landmarks(image, shape)
        cv2.imshow("Image", output)
        cv2.waitKey(0)

def show_face_pointcloud(rects, predictor, gray, image):
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

def detect_emoticon(rects, predictor, gray, image):
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        EMOTICON_LANDMARKS_IDXS = [
            ("mouth", (48, 68), (None, None)),
            ("right_eye", (17, 22), (36, 42)),
            ("left_eye", (22, 27), (42, 48)),
        ]

        # loop over the face parts individually
        for (name, (i, j), (k,l)) in EMOTICON_LANDMARKS_IDXS:
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
            
            if k is None and l is None:
                landmarks = shape[i:j]
            else:
                landmarks = np.concatenate((shape[i:j], shape[k:l]), axis=0)
                
            for (x, y) in landmarks:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            
            (x, y, w, h) = cv2.boundingRect(np.array([landmarks]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            cv2.imshow("ROI", roi)
            # timestamp = int(time.time())
            # folder = 'dataset/eyes/normal'
            # ret = cv2.imwrite(f'{folder}/{name}-{timestamp}.png', roi) 
            cv2.imshow("Image", clone)
            cv2.waitKey(0)

def main():
    args = parse_args()

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()

        # Load the input image, resize it, and convert it to grayscale
        # image = cv2.imread(args["image"])
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        rects = detector(gray, 1)

        # show_face_pointcloud(rects, predictor, gray, image)
        # draw_features(rects, predictor, gray, image); break
        detect_emoticon(rects, predictor, gray, image); break

        # Show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()