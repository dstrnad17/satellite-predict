import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.onnx

# Load the DataFrame from the pickle file
main_dataframe = pd.read_pickle('./Satellite_data/sat_dataframe.pkl')

x = main_dataframe.iloc[:, 10]
y = main_dataframe.iloc[:, 11]
z = main_dataframe.iloc[:, 12]
bx = main_dataframe.iloc[:, 13].values

# Neural Network
class DataReg(nn.Module):
    def __init__(self):
        super(DataReg, self).__init__()
        self.fc1 = nn.Linear(3, 6)  
        self.fc2 = nn.Linear(6, 3)
        self.fc3 = nn.Linear(3, 1)    

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DataReg().to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_values = []

# Early Stopping parameters
patience = 10
best_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0

# Prepare data
X = np.vstack((x, y, z)).T  # Stack X, Y, Z horizontally and transpose
X = torch.tensor(X, dtype=torch.float32)
Bx = torch.tensor(bx, dtype=torch.float32).view(-1, 1)

# Dataloader for batch processing
batch_size = 1024
dataset = TensorDataset(X, Bx)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Check the shapes
print("X shape:", X.shape)
print("Bx shape:", Bx.shape)

# Train the model in batches
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_X, batch_Bx in dataloader:
        batch_X, batch_Bx = batch_X.to(device), batch_Bx.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Bx)
        loss.backward()
        optimizer.step()
    
    loss_values.append(loss.item())
    
    # Early Stopping check
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4g}')
        
print(f'Best epoch: {best_epoch+1}, Best loss: {best_loss:.4g}')

# Evaluate the model
model.eval()
with torch.no_grad():
    NNpred = model(X.to(device)).cpu().numpy()

# Simple evaluation metric (Mean Squared Error)
mseNN = np.mean((NNpred - Bx.numpy())**2)
r2NN = r2_score(Bx.numpy(), NNpred)
print(f'NN Mean Squared Error: {mseNN:.4g}')
print(f'NN R2: {r2NN:.4g}')

### OLS Linear Regression Model ###

# Fit the linear regression model
model = LinearRegression()
model.fit(X, Bx)

# Predict using the model
OLSpred = model.predict(X)
mseOLS = mean_squared_error(Bx, OLSpred)
r2OLS = r2_score(Bx, OLSpred)
print(f'OLSR Mean Squared Error: {mseOLS:.4g}')
print(f'OLSR R2: {r2OLS:.4g}')

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

diff = OLSpred - NNpred.flatten()

### Results ###

plt.plot(loss_values, label='Training Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NN Training Loss Over Epochs')
plt.legend()
plt.savefid('DataRegLoss.png')

# Plot the results
plt.figure(figsize=(10, 10))
plt.suptitle('Predictions vs. True Data, Bx Data')

plt.subplot(2, 2, 1)
plt.scatter(Bx, NNpred, alpha=0.5)
plt.plot([Bx.min(), Bx.max()], [Bx.min(), Bx.max()], 'r--', lw=2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'NN True vs Predicted Values, 2 Hidden Layers, MSE: {mseNN:.4g}')

plt.subplot(2, 2, 2)
plt.plot(loss_values, label='Training Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NN Training Loss Over Epochs')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(Bx, OLSpred, alpha=0.5)
plt.plot([Bx.min(), Bx.max()], [Bx.min(), Bx.max()], 'r--', lw=2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'OLSR True vs Predicted Values, MSE: {mseOLS:.4f}')

plt.subplot(2, 2, 4)
plt.scatter(Bx, diff, alpha = 0.5)
plt.xlabel('True')
plt.ylabel('Difference')
plt.title('Difference Between Predictions (OLS - NN)')

plt.savefig('DataReg.png')
