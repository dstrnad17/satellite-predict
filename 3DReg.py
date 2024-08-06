
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary



# Generate example 3D data
np.random.seed(0)
X = np.random.rand(500, 3)  # 500 samples, 3 features
y = X @ np.array([1, -2, 4]) + 0.5  # A linear combination plus some noise

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

### Neural Network Models ###

#Define the neural network model (1 Layer)
class Reg3D1L(nn.Module):
    def __init__(self):
        super(Reg3D1L, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer
        self.fc2 = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model
model1L = Reg3D1L()

# Define loss function and optimizer
crit = nn.MSELoss()
optim1L = optim.Adam(model1L.parameters(), lr=0.003)
loss_values1L = []

# Early Stopping parameters
patience = 50
best_loss1L = float('inf')
best_epoch1L = 0
epochs_no_improve1L = 0

#Tensorboard
writer = SummaryWriter('runs/3D_regression_1L')
boardx = torch.randn(10, 3)
writer.add_graph(model1L, boardx)

#Torchsummary
summary(model1L, input_size=(3,))

#Parameters
def save_model_parameters(model1L, filename):
    with open(filename, 'w') as f:
        for name, param in model1L.named_parameters():
            f.write(f'{name}:\n')
            f.write(f'{param.data}\n')
            f.write(f'{"-"*40}\n')  #Separator
            
save_model_parameters(model1L, 'model1L_parameters.txt')

# Train the model
epochs = 5000
for epoch in range(epochs):
    model1L.train()
    optim1L.zero_grad()
    outputs = model1L(X_train)
    loss = crit(outputs, y_train)
    loss.backward()
    optim1L.step()
    
    loss_values1L.append(loss.item())
    
    writer.add_scalar('Loss/1L', loss.item(), epoch)
    
    # Early Stopping check
    if loss.item() < best_loss1L:
        best_loss1L = loss.item()
        best_epoch1L = epoch
        epochs_no_improve1L = 0
    else:
        epochs_no_improve1L += 1

    if epochs_no_improve1L == patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4g}')
        
print(f'Best epoch: {best_epoch1L+1}, Best loss: {best_loss1L:.4g}')

# Evaluate the model
model1L.eval()
with torch.no_grad():
    NN1L_pred = model1L(X_train).numpy()

# Simple evaluation metric (Mean Squared Error)
mseNN1L = np.mean((NN1L_pred - y_train.numpy())**2)
r2NN1L = r2_score(y, NN1L_pred)
print(f'NN Mean Squared Error: {mseNN1L:.4f}')
print(f'NN R2: {r2NN1L:.4f}')

# Define the neural network model (2 Layers)
class Reg3D2L(nn.Module):
    def __init__(self):
        super(Reg3D2L, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer
        self.fc2 = nn.Linear(64, 32) # Hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the model
model2L = Reg3D2L()

# Define loss function and optimizer
crit = nn.MSELoss()
optim2L = optim.Adam(model2L.parameters(), lr=0.003)
loss_values2L = []

best_loss2L = float('inf')
best_epoch2L = 0
epochs_no_improve2L = 0

writer2L = SummaryWriter('runs/3D_regression_2L')
writer2L.add_graph(model2L, boardx)

summary(model2L, input_size=(3,))

def save_model2L_parameters(model2L, filename):
    with open(filename, 'w') as f:
        for name, param in model2L.named_parameters():
            f.write(f'{name}:\n')
            f.write(f'{param.data}\n')
            f.write(f'{"-"*40}\n')  #Separator
            
save_model2L_parameters(model2L, 'model2L_parameters.txt')

# Train the model
for epoch in range(epochs):
    model2L.train()
    optim2L.zero_grad()
    outputs = model2L(X_train)
    loss = crit(outputs, y_train)
    loss.backward()
    optim2L.step()
    
    loss_values2L.append(loss.item())
    
    writer2L.add_scalar('Loss/2L', loss.item(), epoch)
    
    # Early Stopping check
    if loss.item() < best_loss2L:
        best_loss2L = loss.item()
        best_epoch2L = epoch
        epochs_no_improve2L = 0
    else:
        epochs_no_improve2L += 1

    if epochs_no_improve2L == patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
print(f'Best epoch: {best_epoch2L+1}, Best loss: {best_loss2L:.4f}')

# Evaluate the model
model2L.eval()
with torch.no_grad():
    NN2L_pred = model2L(X_train).numpy()

# Simple evaluation metric (Mean Squared Error)
mseNN2L = np.mean((NN2L_pred - y_train.numpy())**2)
r2NN2L = r2_score(y, NN2L_pred)
print(f'NN Mean Squared Error: {mseNN2L:.4f}')
print(f'NN R2: {r2NN2L:.4f}')


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

y_diff = OLS_pred - NN2L_pred.flatten()

### Results ###

# Plot the results
plt.figure(figsize=(10, 10))
plt.suptitle('Predictions vs. True Data, Linear 3D Combo')

plt.subplot(2, 2, 1)
plt.scatter(y_train.numpy(), NN2L_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'NN True vs Predicted Values, 2 Hidden Layers, MSE: {mseNN2L:.4f}')

plt.subplot(2, 2, 2)
plt.plot(loss_values2L, label='Training Loss')
plt.yscale('log')
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

plt.figure(figsize=(10, 10))
plt.suptitle('Neural Network Layer Comparisons')

plt.subplot(2, 2, 1)
plt.scatter(y_train.numpy(), NN1L_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'True vs Predicted Values, 1 Hidden Layer, MSE: {mseNN2L:.4f}')

plt.subplot(2, 2, 2)
plt.plot(loss_values1L, label='Training Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs, 1 Hidden Layer')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(y_train.numpy(), NN2L_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'True vs Predicted Values, 2 Hidden Layers, MSE: {mseNN2L:.4f}')

plt.subplot(2, 2, 4)
plt.plot(loss_values2L, label='Training Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs, 2 Hidden Layers')
plt.legend()

plt.savefig('3DRegLayers.png')

writer.close()
writer2L.close()