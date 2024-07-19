
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate some example 3D data
np.random.seed(0)
X = np.random.rand(1000, 3)  # 1000 samples, 3 features
y = X @ np.array([3.5, -2.1, 5.7]) + 0.5  # A linear combination plus some noise

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Step 2: Define the neural network model
class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer
        self.fc2 = nn.Linear(64, 32) # Hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the model
model = SimpleRegressor()

# Step 3: Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 4: Train the model
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Evaluate the model
model.eval()
with torch.no_grad():
    predicted = model(X_train).numpy()

# Simple evaluation metric (Mean Squared Error)
mse = np.mean((predicted - y_train.numpy())**2)
print(f'Mean Squared Error: {mse:.4f}')

# Step 6: Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(y_train.numpy(), predicted, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()
