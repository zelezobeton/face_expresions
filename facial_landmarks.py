'''
Program for detecting eye and mouth expressions.

USE WITH:
    python facial_landmarks.py

Using these tuts:
- https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
- https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

Pretrained facial landmarks detector can be downloaded here:
- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

TODO:
- Make more various dataset, at least different lightning, because the first one
overfits

DONE:
- Find out how to get rectangle around eye+eyebrow
- Alter program/network so they work with my dataset
'''
import argparse
from collections import OrderedDict
import time

import numpy as np
import dlib
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

# Custom libs
import imutils
from imutils import face_utils
from helpers import draw_features, show_face_pointcloud, create_dataset

from face_training import Net

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    # ap.add_argument("-i", "--image", required=True,
    #     help="path to input image")
    args = vars(ap.parse_args())

    return args

def predict_expression(rects, predictor, gray, image, net):
    l_eye = None
    r_eye = None

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        EMOTICON_LANDMARKS_IDXS = [
            # ("mouth", (48, 68), (None, None)),
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
            if name == 'right_eye':
                roi = cv2.flip(roi, 1)
                r_eye = find_out_eye_expr(roi, net)
            else:
                l_eye = find_out_eye_expr(roi, net)
            # cv2.imshow("ROI", roi)
    return l_eye, r_eye

def find_out_eye_expr(image, net):
    '''
    Take in cv2 image of eye, put it through neural net 
    and print predicted expression
    '''
    cv2.imshow("img", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    classes = ('closed', 'frowned', 'normal', 'wide_open')

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = transform(image)
    image = image.unsqueeze(0)
    output = net(image)
    _, predicted = torch.max(output, 1)
    return classes[predicted]

def main():
    # Init neural network
    net = Net()
    net.load_state_dict(torch.load("./parameters.pt"))

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)
    # counter = 0
    while True:
        _, image = cap.read()

        # Load the input image, resize it, and convert it to grayscale
        # image = cv2.imread(args["image"])
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        rects = detector(gray, 1)

        l_eye, r_eye = predict_expression(rects, predictor, gray, image, net)
        if l_eye == 'closed' and r_eye == 'normal':
            print('^_°')
        elif r_eye == 'closed' and l_eye == 'normal':
            print('°_^')
        elif r_eye == 'closed' and l_eye == 'closed':
            print('^_^')
        elif r_eye == 'wide_open' and l_eye == 'wide_open':
            print('O_O')
        elif r_eye == 'frowned' and l_eye == 'frowned':
            print('ಠ_ಠ')
        else:
            print('°_°')
        # create_dataset(rects, predictor, gray, image)
        # counter += 1
        # if counter == 5:
        #     break

        # Show the output image with the face detections + facial landmarks
        # cv2.imshow("Output", image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()