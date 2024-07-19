import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#example data
np.random.seed(0)
X = np.random.rand(100, 3)  # 1000 samples, 3 features

#target data
weights = np.array([[1, 2, 3], [-1, -2, -3], [4, 5, -6]])
biases = np.array([0.1, -0.2, 0.3])
y = X @ weights.T + biases  # A linear combination plus some bias

#numpy to torch
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

#define
class threeout(nn.Module):
    def __init__(self):
        super(threeout, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer
        self.fc2 = nn.Linear(64, 32) # Hidden layer
        self.fc3 = nn.Linear(32, 3)  # Output layer (3 features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the model
model = threeout()

#criteria and optimizer
crit = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=0.01)

#train
epochs = 500
for epoch in range(epochs):
    model.train()
    optim.zero_grad()
    outputs = model(X_train)
    loss = crit(outputs, y_train)
    loss.backward()
    optim.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#evaluate
model.eval()
with torch.no_grad():
    predicted = model(X_train).numpy()

#error
mse = np.mean((predicted - y_train.numpy())**2)
print(f'Mean Squared Error: {mse:.4f}')

#plot

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_train[:, 0], predicted[:, 0], alpha=0.5, color='blue')
plt.scatter(X_train[:, 0], y_train[:, 0], alpha=0.5, color='red')
plt.title('Dimension 1')

plt.subplot(1, 3, 2)
plt.scatter(X_train[:, 1], predicted[:, 1], alpha=0.5, color='blue')
plt.scatter(X_train[:, 1], y_train[:, 1], alpha=0.5, color='red')
plt.title('Dimension 2')

plt.subplot(1, 3, 3)
plt.scatter(X_train[:, 2], predicted[:, 2], alpha=0.5, color='blue')
plt.scatter(X_train[:, 2], y_train[:, 2], alpha=0.5, color='red')
plt.title('Dimension 3')

plt.tight_layout()
plt.show()

# 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_train[:, 0], y_train[:, 1], y_train[:,2], color='red')
ax.scatter(predicted[:, 0], predicted[:,1], predicted[:,2], color='blue')
plt.show()