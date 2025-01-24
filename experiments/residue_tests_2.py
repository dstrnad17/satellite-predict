import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def arv(targets, predictions):
    residuals = targets - predictions
    residual_variance = torch.var(residuals)
    target_variance = torch.var(targets)
    return (residual_variance / target_variance).item()

# Generate noisy multi-input data
np.random.seed(42)
m1, m2 = 2, -1
#b = 5
X1 = np.linspace(-1, 1, 200)
X2 = X1 + np.random.normal(0, 0.2, 200)

#y = m1 * X1 + m2 * X2 + b + np.random.normal(0, 0.5, 200)
X = np.column_stack((X1, X2))
y = np.tanh(2 * (m1 * X1 + m2 * X2)) + np.random.normal(0, 0.2, 200)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)
    
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
        self.fc1 = nn.Linear(2, 10)
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

# Calculate the MSE for the final prediction
init_mse = nn.MSELoss()(y_pred_lin_test, y_test)
final_mse = nn.MSELoss()(y_pred_final, y_test)
print(f'Linear MSE: {init_mse.item()}')
print(f'ResidualNet MSE: {final_mse.item()}')

init_arv = arv(y_pred_lin_test, y_test)
final_arv = arv(y_pred_final, y_test)
print(f'Linear ARV: {init_arv}')
print(f'ResidualNet ARV: {final_arv}')

# Plot results with ARV annotations
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.scatter(X_test[:, 0].numpy(), y_test.numpy(), label='True Data', color='blue', alpha=0.6)
plt.scatter(X_test[:, 0].numpy(), y_pred_lin_test.detach().numpy(), label=f'Linear Prediction, ARV: {init_arv:.4f}', color='red', alpha=0.6)
plt.scatter(X_test[:, 0].numpy(), y_pred_final.detach().numpy(), label=f'ResidualNet Prediction, ARV: {final_arv:.4f}', color='green', alpha=0.6)

# Add labels and legend
plt.xlabel('X1')
plt.ylabel('y')
plt.title('Model Predictions')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 1, 2)
plt.scatter(X_test[:, 1].numpy(), y_test.numpy(), label='True Data', color='blue', alpha=0.6)
plt.scatter(X_test[:, 1].numpy(), y_pred_lin_test.detach().numpy(), label='Linear Prediction', color='red', alpha=0.6)
plt.scatter(X_test[:, 1].numpy(), y_pred_final.detach().numpy(), label='ResidualNet Prediction', color='green', alpha=0.6)

# Add labels and legend
plt.xlabel('X2')
plt.ylabel('y')
plt.grid(alpha=0.3)

plt.show()
