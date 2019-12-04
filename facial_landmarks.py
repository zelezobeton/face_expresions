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

DONE:
- Create new dataset for mouths, 22 for expression (10 dark, 10 light, 2 test) 
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

def predict_expression(rects, predictor, gray, image, eyes_net, mouth_net):
    l_eye = None
    r_eye = None
    mouth = None

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

        # clone = image.copy()
        # loop over the face parts individually
        for (name, (i, j), (k,l)) in EMOTICON_LANDMARKS_IDXS:
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            # cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7, (0, 0, 255), 2)
            
            if k is None and l is None:
                landmarks = np.append(shape[i:j], [shape[6], shape[10], shape[33]], axis=0)
            else:
                landmarks = np.concatenate((shape[i:j], shape[k:l]), axis=0)
            
            (x, y, w, h) = cv2.boundingRect(np.array([landmarks]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            if name == 'mouth':
                mouth = find_out_eye_expr(roi, mouth_net, 'mouth')
            elif name == 'right_eye':
                roi = cv2.flip(roi, 1)
                r_eye = find_out_eye_expr(roi, eyes_net, 'eyes')
            elif name == 'left_eye':
                l_eye = find_out_eye_expr(roi, eyes_net, 'eyes')
            # cv2.imshow("ROI", roi)
    return l_eye, r_eye, mouth

def find_out_eye_expr(image, net, facepart):
    '''
    Take in cv2 image of eye, put it through neural net 
    and print predicted expression
    '''
    # cv2.imshow("img", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    if facepart == 'eyes':
        classes = ('closed', 'frowned', 'normal', 'wide_open')
    elif facepart == 'mouth':
        classes = ('normal', 'open', 'smile', 'wide_smile')

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
    # Init neural networks
    eyes_net = Net()
    eyes_net.load_state_dict(torch.load("./eyes.pt"))
    mouth_net = Net()
    mouth_net.load_state_dict(torch.load("./mouth.pt"))

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    l_eye_dict = {
        'frowned': '>',
        'closed': '^',
        'wide_open': 'O',
        'normal': '*' 
    }
    r_eye_dict = {
        'frowned': '<',
        'closed': '^',
        'wide_open': 'O',
        'normal': '*' 
    }
    mouth_dict = {
        'smile': 'u',
        'open': 'o',
        'wide_smile': 'U',
        'normal': '_'
    }

    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()

        # Load the input image, resize it, and convert it to grayscale
        # image = cv2.imread(args["image"])
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        rects = detector(gray, 1)

        l_eye, r_eye, mouth = predict_expression(
            rects, predictor, gray, image, eyes_net, mouth_net)
        
        if l_eye and r_eye and mouth:
            expr = l_eye_dict[l_eye] + mouth_dict[mouth] + r_eye_dict[r_eye] 
            cv2.putText(image, expr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 2)

        # Show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()