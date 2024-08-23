#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 19:59:45 2024

@author: dunnchadnstrnad
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# Synthetic data
np.random.seed(10)
X = np.linspace(-10, 10, 1000)
y = np.sin(X) + np.random.normal(0, 0.3, X.shape)
X = X.reshape(-1, 1).astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Define a function to create a model with a given number of hidden neurons
class NonlinRegModel(nn.Module):
    def __init__(self, hidden_size):
        super(NonlinRegModel, self).__init__()
        self.layer1 = nn.Linear(1, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x

# Define a function to train and evaluate the model
def train_and_evaluate(hidden_size):
    model = NonlinRegModel(hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        predicted = model(X_tensor).detach().numpy()
    
    mse = mean_squared_error(y, predicted)
    return mse

# Define the parameter grid
param_grid = {'hidden_size': [8, 16, 32, 64, 128]}

# Perform grid search
best_mse = float('inf')
best_hidden_size = None
for hidden_size in param_grid['hidden_size']:
    mse = train_and_evaluate(hidden_size)
    if mse < best_mse:
        best_mse = mse
        best_hidden_size = hidden_size

print(f'Best hidden layer size: {best_hidden_size}, MSE: {best_mse:.4f}')