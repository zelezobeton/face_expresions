#!/usr/bin/env python3
'''
Program for training/testing neural network on dataset of facepart images with 
different expressions

USE WITH:
    python faceparts_training.py --MODE FACEPART
    MODE ... train or test
    FACEPART ... eyes or mouth

Useful links:
- https://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.ImageFolder
- https://pytorch.org/docs/stable/torchvision/transforms.html
'''
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define expression classes
EYE_CLASSES = ('closed', 'frowned', 'normal', 'wide_open')
MOUTH_CLASSES = ('normal', 'open', 'smile', 'wide_smile')

class Net(nn.Module):
    '''
    Define a Convolutional Neural Network
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_dataset(name):
    '''
    Load dataset from folders and define transformations done on images
    '''
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.ImageFolder(f'train_dataset/{name}', transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(f'test_dataset/{name}', transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    if name == 'eyes':
        classes = EYE_CLASSES
    elif name == 'mouth':
        classes = MOUTH_CLASSES

    return trainset, trainloader, testset, testloader, classes

def imshow(img):
    '''
    Unnormalize and show image
    '''
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_random_images(trainloader, classes):
    '''
    Show some random images from dataset
    '''
    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # Show images
    imshow(torchvision.utils.make_grid(images))

def test_network(testloader, classes, net):
    '''
    Test network with test dataset
    ''' 
    # Show 4 tested images and their values
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Test network with those images
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # Test network with whole dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 8 test images: %d %%' % (100 * correct / total))

def train_network(trainloader, optimizer, net, criterion):
    '''
    Train the network
    '''
    # Loop over the dataset multiple times
    for epoch in range(40):  

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2 == 1:    # print every 2 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

    print('Finished Training')

def main():
    # Parse args
    if len(sys.argv) == 3:
        mode = sys.argv[1]
        name = sys.argv[2]
    else:
        print('Not enough arguments!')

    # Loading and normalizing image dataset
    trainset, trainloader, testset, testloader, classes = load_dataset(name)
    
    # Show some random images
    # show_random_images(trainloader, classes)
    # exit()

    # Create a Convolutional Neural Network
    net = Net()

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if mode == '--train':
        # Train the network
        train_network(trainloader, optimizer, net, criterion)
        # Save model parameters
        torch.save(net.state_dict(), f'./{name}.pt')
    elif mode == '--test':
        # Load model parameters
        net.load_state_dict(torch.load(f'./{name}.pt'))

    # Test network with test dataset
    test_network(testloader, classes, net)

if __name__ == "__main__":
    main()
