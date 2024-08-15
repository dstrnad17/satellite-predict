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
y = np.sin(X) + np.random.normal(0, 0.3, X.shape)
X = X.reshape(-1, 1).astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

#Define model, 4 layers
class NonlinRegModel(nn.Module):
    def __init__(self):
        super(NonlinRegModel, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
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

losses = []
mses = []
rmses = []
maes = []
r2s = []

#Train model
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_tensor)
    loss = crit(outputs, y_tensor)
    
    # Backward pass and optimization
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            predicted = model(X_tensor).detach().numpy()
        
        mse = mean_squared_error(y, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predicted)
        r2 = r2_score(y, predicted)
        
        losses.append(loss.item())
        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        
        model.train()

with open('NLR_test.txt', 'w') as f:
    f.write(f'MSE: {mse:.4f}\n')
    f.write(f'RMSE: {rmse:.4f}\n')
    f.write(f'MAE: {mae:.4f}\n')
    f.write(f'R^2: {r2:.4f}\n')

# Plot the results
plt.plot(X, y, 'ro', label='Original data')
plt.plot(X, predicted, 'b-', label='Fitted line')
plt.title('Neural Net Regression')
plt.legend()
plt.savefig('NLR_test.pdf')
plt.savefig('NLR_test.png')

# Plot the error metrics
epochs = range(10, num_epochs + 1, 10)
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(epochs, losses, 'r', label='Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, mses, 'g', label='MSE')
plt.title('Mean Squared Error')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, rmses, 'b', label='RMSE')
plt.title('Root Mean Squared Error')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, maes, 'm', label='MAE')
plt.title('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.savefig('NLR_error.png')