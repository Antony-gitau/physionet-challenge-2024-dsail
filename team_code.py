#!/usr/bin/env python

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################
# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image...
        current_features = extract_features(record)
        features.append(current_features)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # This overly simple model uses the mean of these overly simple features as a seed for a random number generator.
    model = np.mean(features)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_digitization_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Define the FCNN architecture for classifying the images
class ECGFCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ECGFCNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x

# Custom Dataset class for ECG data
class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()
    dxs = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = load_dx(record)
        if dx:
            current_features = extract_features(record)
            features.append(current_features)
            dxs.append(dx)

    if not dxs:
        raise Exception('There are no labels for the data.')

    features = np.array(features)
    classes = sorted(set.union(*map(set, dxs)))
    dxs = compute_one_hot_encoding(dxs, classes)

    # Ensure features are in the correct shape for the FCNN
    features = torch.tensor(features, dtype=torch.float32)
    dxs = torch.tensor(dxs, dtype=torch.float32)

    # Create DataLoader
    dataset = ECGDataset(features, dxs)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the FCNN model.
    if verbose:
        print('Training the model on the data...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGFCNN(input_dim=features.shape[1], num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    torch.save(model.state_dict(), os.path.join(model_folder, 'dx_model.pth'))
    np.save(os.path.join(model_folder, 'dx_classes.npy'), classes)

    if verbose:
        print('Done.')
        print()

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    if verbose:
        print('Loading the dx classification model...')

    filename = os.path.join(model_folder, 'dx_model.pth')
    classes = np.load(os.path.join(model_folder, 'dx_classes.npy'))

    # Define the model architecture
    model = ECGFCNN(input_dim=2, num_classes=len(classes))  # Adjust input_dim as necessary
    model.load_state_dict(torch.load(filename))
    model.eval()
    
    return model, classes

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    model, classes = dx_model

    # Extract features.
    features = extract_features(record)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Get model probabilities.
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1).numpy()

    # Choose the class(es) with the highest probability as the label(s).
    max_probability = np.max(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities[0]) if probability == max_probability]

    return labels

# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

    # Save your trained digitization model.
def save_digitization_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = load_image(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Note: Other helper functions such as load_image, load_dx, find_records, compute_one_hot_encoding 
# should be defined in the `helper_code` module as referenced in the original script.
