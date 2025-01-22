import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate some noisy data
np.random.seed(42)
m = 2
b = 5
X = np.linspace(-1, 1, 200)  # Features
#y = m * X + b + np.random.normal(0, .5, 200)

y = np.tanh(2*X) + np.random.normal(0, .2, 200)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Train the linear regression model
lin_model = LinearRegressionModel()
lin_criterion = nn.MSELoss()
lin_optimizer = optim.SGD(lin_model.parameters(), lr=0.01)

# Training loop for linear regression
for epoch in range(1000):
    lin_optimizer.zero_grad()
    y_pred_lin = lin_model(X_train)
    loss = lin_criterion(y_pred_lin, y_train)
    loss.backward()
    lin_optimizer.step()

# Neural network for residual error (delta)
class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Activation in the hidden layer
        x = self.fc2(x)
        return x

# Train the residual network
residual_net = ResidualNet()
residual_criterion = nn.MSELoss()
residual_optimizer = optim.Adam(residual_net.parameters(), lr=0.01)

# Calculate the residuals based on the linear regression predictions
y_pred_lin_train = lin_model(X_train)
residuals = y_train - y_pred_lin_train

# Training loop for residual network
for epoch in range(1000):
    residual_optimizer.zero_grad()
    
    # Forward pass for the residual network
    residual_pred = residual_net(X_train)
    
    # Compute residual loss
    residual_loss = residual_criterion(residual_pred, residuals)
    
    # Perform backward pass
    residual_loss.backward(retain_graph=True)  # Retain graph for future backward passes
    residual_optimizer.step()

# Combine the linear predictions and residual predictions
with torch.no_grad():
    y_pred_lin_test = lin_model(X_test)
    residual_pred_test = residual_net(X_test)
    y_pred_final = y_pred_lin_test + residual_pred_test

#real_line = m * X_test.numpy() + b
#real_line = np.tanh(2*X_test.numpy())

# Plot results
plt.scatter(X_test.numpy(), y_test.numpy(), label='True Data', color='blue')
plt.scatter(X_test.numpy(), y_pred_lin_test.detach().numpy(), label='Linear Prediction', color='red')
plt.scatter(X_test.numpy(), y_pred_final.detach().numpy(), label='Final Prediction (w/ Residuals)', color='green')
#plt.plot(X_test.numpy(), real_line, label='Real Line', color='orange', linestyle='--')
plt.legend()
plt.show()

# Calculate the MSE for the final prediction
init_mse = nn.MSELoss()(y_pred_lin_test, y_test)
final_mse = nn.MSELoss()(y_pred_final, y_test)
print(f'Initial MSE: {init_mse.item()}')
print(f'Final MSE: {final_mse.item()}')

def relative_variance(predictions, actual):
    mean_actual = actual.mean().item()
    variance_pred = predictions.var().item()
    return variance_pred / (mean_actual ** 2)

init_rel_variance = relative_variance(y_pred_lin_test, y_test)
final_rel_variance = relative_variance(y_pred_final, y_test)
print(f'Initial ARV: {init_rel_variance}')
print(f'Final ARV: {final_rel_variance}')