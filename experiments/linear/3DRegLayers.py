
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate example 3D data
np.random.seed(0)
X = np.random.rand(500, 3)  # 1000 samples, 3 features
y = X @ np.array([1, -2, 4]) + 0.5  # A linear combination plus some noise

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

### Neural Network Model ###

# Define the neural network model
class Reg3D(nn.Module):
    def __init__(self):
        super(Reg3D, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer
        self.fc2 = nn.Linear(64, 32) # Hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the model
model = Reg3D()

# Define loss function and optimizer
crit = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=0.01)
loss_values = []

# Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    optim.zero_grad()
    outputs = model(X_train)
    loss = crit(outputs, y_train)
    loss.backward()
    optim.step()
    
    loss_values.append(loss.item())

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    NN_pred = model(X_train).numpy()

# Simple evaluation metric (Mean Squared Error)
mseNN = np.mean((NN_pred - y_train.numpy())**2)
r2NN = r2_score(y, NN_pred)
print(f'NN Mean Squared Error: {mseNN:.4f}')
print(f'NN R2: {r2NN:.4f}')


### OLS Linear Regression Model ###

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict using the model
OLS_pred = model.predict(X)
mseOLS = mean_squared_error(y, OLS_pred)
r2OLS = r2_score(y, OLS_pred)
print(f'OLSR Mean Squared Error: {mseOLS:.4f}')
print(f'OLSR R2: {r2OLS:.4f}')

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

y_diff = OLS_pred - NN_pred.flatten()

### Results ###

# Plot the results
plt.figure(figsize=(10, 10))
plt.suptitle('Predictions vs. True Data, Linear 3D Combo')

plt.subplot(2, 2, 1)
plt.scatter(y_train.numpy(), NN_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'NN True vs Predicted Values, 2 Hidden Layers, MSE: {mseNN:.4f}')

plt.subplot(2, 2, 2)
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NN Training Loss Over Epochs')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(y, OLS_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'OLSR True vs Predicted Values, MSE: {mseOLS:.4f}')

plt.subplot(2, 2, 4)
plt.scatter(y, y_diff, alpha = 0.5)
plt.xlabel('True')
plt.ylabel('Difference')
plt.title('Difference Between Predictions (OLS - NN)')

plt.savefig('3DRegLinear.png')
