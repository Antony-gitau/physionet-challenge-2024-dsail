import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class InceptionV3(nn.Module):
    def __init__(self, list_of_classes, weights_path=None):
        super(InceptionV3, self).__init__()
        self.list_of_classes = list_of_classes
        self.num_classes = len(self.list_of_classes)
        #self.weights_path = weights_path
        self.base_model = self.load_inception(weights_path)
        #self.fc1 = nn.Linear(self.base_model.fc.out_features, 512)
        #self.fc2 = nn.Linear(512, self.num_classes)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, self.num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        #x = x.view(-1, x.shape[1] * x.shape[2]* x.shape[3])
        #x = torch.relu(self.fc1(x))
        #x = torch.sigmoid(self.fc2(x))
        if isinstance(x, models.inception.InceptionOutputs):
            x = x.logits
        x = torch.sigmoid(x)
        return x
    
    def load_inception(self, weights_path):
        model = models.inception_v3(weights=None)
        if weights_path:
            print('Loading Weights')
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)
        return model
