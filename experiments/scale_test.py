import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression

# Step 1: Create sample data (for demonstration purposes)
X, y = make_regression(n_samples=100, n_features=3, noise=0.1)

# Initialize MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Normalize the features (X) and target (y)
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Step 2: Example PyTorch Neural Network
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

# Step 3: Use the trained model to make predictions
model.eval()
predictions = model(X_tensor).detach().numpy()

# Step 4: Descale the predicted values
predictions_descale = scaler_y.inverse_transform(predictions)

# Print the first 5 descaled predictions
print(predictions_descale[:5])
