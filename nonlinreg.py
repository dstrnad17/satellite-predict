#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Synthetic data
np.random.seed(10)
X = np.linspace(-10, 10, 1000)
y = np.sin(X)**3 - np.cos(3 * X) + np.random.normal(0, 0.3, X.shape)
X = X.reshape(-1, 1).astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

#Define model, 4 layers
class NonlinRegModel(nn.Module):
    def __init__(self):
        super(NonlinRegModel, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return x

model = NonlinRegModel()
crit = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=0.001)    #Using Adam as optimizer

# Weights and biases of each layer
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'Param name: {name}')
        print(f'Param shape: {param.shape}')
        print(f'Param values: {param.data}')
        print()

#Train model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_tensor)
    loss = crit(outputs, y_tensor)
    
    # Backward pass and optimization
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
model.eval()
with torch.no_grad():
    predicted = model(X_tensor).detach().numpy()
    
# Error measurements
mse = mean_squared_error(y, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, predicted)
r2 = r2_score(y, predicted)

with open('NLR_error_metrics_sin3cos3.txt', 'w') as f:
    f.write(f'MSE: {mse:.4f}\n')
    f.write(f'RMSE: {rmse:.4f}\n')
    f.write(f'MAE: {mae:.4f}\n')
    f.write(f'R^2: {r2:.4f}\n')

# Plot the results
plt.plot(X, y, 'ro', label='Original data')
plt.plot(X, predicted, 'b-', label='Fitted line')
plt.title('Neural Net Regression for sin^3(x) - cos(3x) Data')
plt.legend()
plt.savefig('NLR_sin3cos3_plot.pdf')
plt.savefig('NLR_sin3cos3_plot.png')