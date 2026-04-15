# Import required libraries
# --------------------------------------------
# Data loading
import random
import numpy as np
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Train model
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# Evaluate model
from torchmetrics import Accuracy, F1Score

# --------------------------------------------

# Set random seeds for reproducibility
torch.manual_seed(101010)
np.random.seed(101010)
random.seed(101010)

import os
import zipfile

# Unzip the data folder
if not os.path.exists('data/chestxrays'):
    with zipfile.ZipFile('data/chestxrays.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

# Define the transformations to apply to the images for use with ResNet-18.
# The images need to be normalized to the same domain as the original training data of ResNet-18 network.
# We normalize the X-rays using transforms.Normalize function that takes as input the means and
# standard deviations of the three color channels, (R,G,B), from the original ResNet-18 training dataset.
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=transform_mean, std=transform_std)])

# Apply the image transforms
train_dataset = ImageFolder('data/chestxrays/train', transform=transform)
test_dataset = ImageFolder('data/chestxrays/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset) // 2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# --------------------------
# Q1: Instantiate the model
# --------------------------

# Load the pre-trained ResNet-18 model
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# --------------------------
# Q2: Modify the model
# --------------------------

# Freeze the parameters of the model
for param in resnet18.parameters():
    param.requires_grad = False

# Modify the final layer for binary classification
resnet18.fc = nn.Linear(resnet18.fc.in_features, 1)


# ------------------------------
# Q3a: Define the training loop
# ------------------------------

# Model training/fine-tuning loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_accuracy = 0

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Ensure labels have the same dimensions as outputs
            labels = labels.float().unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5  # Binary classification
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(preds == labels.data)

        # Calculate the train loss and accuracy for the current epoch
        train_loss = running_loss / len(train_dataset)
        train_acc = running_accuracy.double() / len(train_dataset)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss, train_acc))


# -------------------------
# Q3b: Fine-tune the model
# -------------------------

# Set the model to ResNet-18
model = resnet18

# Fine-tune the ResNet-18 model for 3 epochs using the train_loader
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
train(model, train_loader, criterion, optimizer, num_epochs=3)

# -----------------------
# Evaluation code
# -----------------------

# Set model to evaluation mode
model = resnet18
model.eval()

# Initialize metrics for accuracy and F1 score
accuracy_metric = Accuracy(task="binary")
f1_metric = F1Score(task="binary")

# Create lists store all predictions and labels
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation for evaluation
    for inputs, labels in test_loader:
        # Forward pass
        outputs = model(inputs)
        preds = torch.sigmoid(outputs).round()  # Round to 0 or 1

        # Extend the lists with predictions and labels
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.unsqueeze(1).tolist())

        # Convert lists back to tensors
        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)

        # Calculate accuracy and F1 score
        test_accuracy = accuracy_metric(all_preds, all_labels).item()
        test_f1_score = f1_metric(all_preds, all_labels).item()

print(f"\nTest accuracy: {test_accuracy:.3f}\nTest F1-score: {test_f1_score:.3f}")

# -----------------------------------------------------------------------------------
# Below is a sample code for the bonus task.
# This code divides the training set into training and validation subsets.
# You will have 150 examples per class for training and 50 for validation.
# Don't forget to create new `val_dataset` and `val_loader` after running this code.
# -----------------------------------------------------------------------------------
'''
import os, random, shutil

# Function to move 50 random files from class folder in training to validation folder
def move_files(src_class_dir, dest_class_dir, n=50):
    if not os.path.exists(dest_class_dir):
        os.makedirs(dest_class_dir)
    files = os.listdir(src_class_dir)
    random_files = random.sample(files, n)
    for f in random_files:
        shutil.move(os.path.join(src_class_dir, f), os.path.join(dest_class_dir, f))

# Move 50 images from each class to validation folder
move_files('data/chestxrays/train/NORMAL', 'data/chestxrays/val/NORMAL')
move_files('data/chestxrays/train/PNEUMONIA', 'data/chestxrays/val/PNEUMONIA')
'''
