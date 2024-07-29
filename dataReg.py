import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def load_and_extract_columns(filepath, columns):
    try:
        data = pd.read_csv(filepath, delim_whitespace=True, header=None)
    except pd.errors.ParserError:
        print("Error reading the file with delim_whitespace=True. Trying manual split...")
        with open(filepath, 'r') as file:
            lines = file.readlines()
        data = pd.DataFrame([line.split() for line in lines])
    
    print("DataFrame shape:", data.shape)
    print("First few rows:\n", data.head())
    
    if any(i >= data.shape[1] for i in columns):
        raise IndexError("One or more column indices are out of bounds.")
    
    extracted_columns = []
    for i in columns:
        column = pd.to_numeric(data.iloc[:, i], errors='coerce')
        extracted_columns.append(column)
    
    return extracted_columns

# Example usage
file_path = "/Users/dunnchadnstrnad/Documents/GitHub/2024phys798/downloaded_files/cluster1_2001_avg_300_omni.dat"
columns_to_extract = [10, 11, 12, 13]
array_names = ['X', 'Y', 'Z', 'Bz']

extracted_columns = load_and_extract_columns(file_path, columns_to_extract)

# Assign arrays to variables dynamically
for name, column in zip(array_names, extracted_columns):
    globals()[name] = np.array(column.dropna(), dtype=float)
    print(f"Variable {name} created with data:")
    print(globals()[name])

# Neural Network
class DataReg(nn.Module):
    def __init__(self):
        super(DataReg, self).__init__()
        self.fc1 = nn.Linear(3, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)    

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DataReg()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_values = []

# Early Stopping parameters
patience = 25
best_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0

# Prepare data
X = np.vstack((X, Y, Z)).T  # Stack X, Y, Z horizontally and transpose
X = torch.tensor(X, dtype=torch.float32)
Bz = torch.tensor(Bz, dtype=torch.float32).view(-1, 1)  # Reshape Bz to match output dimensions

# Check the shapes
print("X shape:", X.shape)
print("Bz shape:", Bz.shape)

# Train the model
epochs = 5000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Bz)
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

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
print(f'Best epoch: {best_epoch+1}, Best loss: {best_loss:.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    pred = model(X).numpy()

# Simple evaluation metric (Mean Squared Error)
mse = np.mean((pred - Bz.numpy())**2)
r2 = r2_score(Bz.numpy(), pred)
print(f'NN Mean Squared Error: {mse:.4f}')
print(f'NN R2: {r2:.4f}')
