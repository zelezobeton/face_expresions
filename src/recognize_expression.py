'''
Program for detecting eye and mouth expressions.

USE WITH:
    python recognize_expression.py

Pretrained facial landmarks detector can be downloaded here:
- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
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
from faceparts_training import Net, EYE_CLASSES, MOUTH_CLASSES

# Define different expressions
L_EYE_DICT = {
    'frowned': '>',
    'closed': '^',
    'wide_open': 'O',
    'normal': '*' 
}
R_EYE_DICT = {
    'frowned': '<',
    'closed': '^',
    'wide_open': 'O',
    'normal': '*' 
}
MOUTH_DICT = {
    'smile': 'u',
    'open': 'o',
    'wide_smile': 'U',
    'normal': '_'
}

def cut_out_facepart(image, landmarks):
    (x, y, w, h) = cv2.boundingRect(np.array([landmarks]))
    roi = image[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    return roi

def use_neural_net(image, net, facepart):
    '''
    Take in cv2 image, put it through neural net 
    and return predicted expression
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    if facepart == 'eyes':
        classes = EYE_CLASSES
    elif facepart == 'mouth':
        classes = MOUTH_CLASSES

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = transform(image)
    image = image.unsqueeze(0)
    output = net(image)
    _, predicted = torch.max(output, 1)
    return classes[predicted]

def get_expression(image, name, shape_array, net):
    landmarks = np.append(*shape_array, axis=0)
    roi = cut_out_facepart(image, landmarks)

    if name == 'right_eye':
        roi = cv2.flip(roi, 1)

    facepart = ('mouth' if name == 'mouth' else 'eyes')    
    return use_neural_net(roi, net, facepart)

def predict_expression(face_rect, predictor, gray, image, eyes_net, mouth_net):
    # Determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, face_rect)
    landmarks = face_utils.shape_to_np(shape)

    # Specify landmarks for individual faceparts
    l_eye_landmarks = [landmarks[22:27], landmarks[42:48]]
    r_eye_landmarks = [landmarks[17:22], landmarks[36:42]]
    mouth_landmarks = [landmarks[48:68], 
        [landmarks[6], landmarks[10], landmarks[33]]]

    # Get expression for every wanted facepart
    l_eye = get_expression(image, 'left_eye', l_eye_landmarks, eyes_net)
    r_eye = get_expression(image, 'right_eye', r_eye_landmarks, eyes_net)
    mouth = get_expression(image, 'mouth', mouth_landmarks, mouth_net)

    return l_eye, r_eye, mouth

def draw_expression(image, detector, predictor, eyes_net, mouth_net):
    # Resize image and convert it to grayscale
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    face_rects = detector(gray, 1)
    if face_rects:
        # Choose only first face
        face_rect = face_rects[0]
        l_eye, r_eye, mouth = predict_expression(
            face_rect, predictor, gray, image, eyes_net, mouth_net)
        
        if l_eye and r_eye and mouth:
            # Draw expression
            expr = L_EYE_DICT[l_eye] + MOUTH_DICT[mouth] + R_EYE_DICT[r_eye] 
            cv2.putText(image, expr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 2)
    return image

def main():
    # Init neural networks
    eyes_net = Net()
    eyes_net.load_state_dict(torch.load("./eyes.pt"))
    mouth_net = Net()
    mouth_net.load_state_dict(torch.load("./mouth.pt"))

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()

        # Figure out expression and draw it on image
        image = draw_expression(image, detector, predictor, eyes_net, mouth_net)

        # Show image with drawn expression
        cv2.imshow("Output", image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()