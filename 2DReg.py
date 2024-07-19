
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate 2D data
np.random.seed(0)
x = np.random.rand(500, 2)
y = np.sin(x[:, 0] * np.pi) + np.cos(x[:, 1] * np.pi) + 0.1 * np.random.randn(500)

# numpy to torch
x_train = torch.tensor(x, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x

# Initialize
model = SimpleNN()
crit = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=0.01)
loss_values = []

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optim.zero_grad()
    outputs = model(x_train)
    loss = crit(outputs, y_train)
    loss.backward()
    optim.step()
    
    loss_values.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    predictions = model(x_train).numpy()

# Plot results
plt.figure(figsize=(10, 10))
plt.suptitle('Predictions vs. True Data, sin(x1) + cos(x2), 1 Hidden Layer')

plt.subplot(3, 1, 1)
plt.scatter(x[:, 0], y, color='blue', label='True data')
plt.scatter(x[:, 0], predictions, color='red', label='Predictions')
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('Y vs. X1')
plt.legend()

plt.subplot(3, 1, 2)
plt.scatter(x[:, 1], y, color='blue', label='True data')
plt.scatter(x[:, 1], predictions, color='red', label='Predictions')
plt.xlabel('X2')
plt.ylabel('Y')
plt.title('Y vs. X2')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.savefig('2DReg.png')

# 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, color='blue')
ax.scatter(x[:, 0], x[:, 1], predictions, color='red')
plt.show()