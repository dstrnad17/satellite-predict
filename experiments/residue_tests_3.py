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

# Generalized data generation
np.random.seed(42)
num_features = 3  # Number of inputs (slopes)
slopes = np.random.uniform(-2, 2, num_features)  # Random slopes
X1 = np.linspace(-1, 1, 200) # No noise
X = np.column_stack([X1] + [np.linspace(-1, 1, 200) + np.random.normal(0, 0.2, 200) for _ in range(1, num_features)])
y = np.tanh(X @ slopes) + np.random.normal(0, 0.2, 200)  # Non-linear target with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Generalized Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Train the linear regression model
lin_model = LinearRegressionModel(num_features)
lin_criterion = nn.MSELoss()
lin_optimizer = optim.SGD(lin_model.parameters(), lr=0.01)

for epoch in range(1000):
    lin_optimizer.zero_grad()
    y_pred_lin = lin_model(X_train)
    loss = lin_criterion(y_pred_lin, y_train)
    loss.backward()
    lin_optimizer.step()

# Generalized Residual Network
class ResidualNet(nn.Module):
    def __init__(self, input_dim):
        super(ResidualNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the residual network
residual_net = ResidualNet(num_features)
residual_criterion = nn.MSELoss()
residual_optimizer = optim.Adam(residual_net.parameters(), lr=0.01)

# Calculate residuals
y_pred_lin_train = lin_model(X_train)
residuals = y_train - y_pred_lin_train.detach()

for epoch in range(1000):
    residual_optimizer.zero_grad()
    residual_pred = residual_net(X_train)
    residual_loss = residual_criterion(residual_pred, residuals)
    residual_loss.backward()
    residual_optimizer.step()

# Combine predictions
with torch.no_grad():
    y_pred_lin_test = lin_model(X_test)
    residual_pred_test = residual_net(X_test)
    y_pred_final = y_pred_lin_test + residual_pred_test

# Metrics
init_mse = lin_criterion(y_pred_lin_test, y_test).item()
final_mse = lin_criterion(y_pred_final, y_test).item()
init_arv = arv(y_test, y_pred_lin_test)
final_arv = arv(y_test, y_pred_final)

# Display metrics
print(f"Linear MSE: {init_mse:.4f}, ResidualNet MSE: {final_mse:.4f}")
print(f"Linear ARV: {init_arv:.4f}, ResidualNet ARV: {final_arv:.4f}")

# Plot results
fig, axs = plt.subplots(num_features, 1, figsize=(10, 4 * num_features))
for i in range(num_features):
    axs[i].scatter(X_test[:, i].numpy(), y_test.numpy(), label='True Data', color = 'blue', alpha=0.6)
    axs[i].scatter(X_test[:, i].numpy(), y_pred_lin_test.numpy(), label=f'Linear Prediction (ARV: {init_arv:.4f})', color = 'red', alpha=0.6)
    axs[i].scatter(X_test[:, i].numpy(), y_pred_final.numpy(), label=f'ResidualNet Prediction (ARV: {final_arv:.4f})', color = 'green', alpha=0.6)
    axs[i].set_xlabel(f'X{i + 1}')
    axs[i].set_ylabel('y')
    axs[i].grid(alpha=0.3)
    
    if i == 0:
        axs[i].set_title(f'Model Predictions')
        axs[i].legend()

plt.tight_layout()
plt.show()