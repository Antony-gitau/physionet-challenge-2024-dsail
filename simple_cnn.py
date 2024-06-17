"""Simple convolutional neural network"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, list_of_classes):
        super(SimpleCNN, self).__init__()
        self.list_of_classes = list_of_classes
        self.num_classes = len(self.list_of_classes)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*53*68, 512)          # Adjusted  to input size (425, 550)
        self.fc2 = nn.Linear(512, self.num_classes)  # 11 output classes for multilabel classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, x.shape[1] * x.shape[2]* x.shape[3])
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for multilabel classification
        return x
    
    
