import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate noisy data
torch.manual_seed(42)
x = torch.linspace(-1, 1, 100).unsqueeze(1)  # Shape (100, 1)
true_m = 2.0
true_b = 5.0

#y = true_m * x + true_b + torch.randn_like(x) * 0.2  # Add noise
y = torch.tanh(true_m*x) + true_b + torch.randn_like(x) * 0.2

# Split data into training and testing sets (80% train, 20% test)
train_ratio = 0.8
num_train = int(len(x) * train_ratio)
indices = torch.randperm(len(x))  # Random permutation of indices

train_indices = indices[:num_train]
test_indices = indices[num_train:]

x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indices], y[test_indices]

# Linear Regression Model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        return self.m * x + self.b

# Train the linear model
linear_model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    y_pred_lin = linear_model(x_train)
    loss = criterion(y_pred_lin, y_train)
    loss.backward()
    optimizer.step()
    print(f"lin_loss = {loss}")

# Evaluate the linear model on the test set
y_test_pred_lin = linear_model(x_test)
linear_test_loss = criterion(y_test_pred_lin, y_test).item()

# Neural Network Model
class NonlinearModel(nn.Module):
    def __init__(self, m, b):
        super().__init__()
        self.w = nn.Parameter(m / 10)
        self.W = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(b)
    
    def forward(self, x):
        return self.W * torch.tanh(self.w * x) + self.b

trained_m = linear_model.m.detach()
trained_b = linear_model.b.detach()
nonlinear_model = NonlinearModel(trained_m, trained_b)
y_pred_nl = nonlinear_model(x_train)
y_pred_lin = linear_model(x_train)

plt.figure
plt.scatter(x_train, y_pred_lin.detach().numpy(), color="blue", label="True Data")
plt.scatter(x_train, y_pred_nl.detach().numpy(), color="green", label="Predicted (Nonlinear)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Train the nonlinear model
optimizer = optim.SGD(nonlinear_model.parameters(), lr=0.001)
for epoch in range(10000):
    optimizer.zero_grad()
    y_pred_nl = nonlinear_model(x_train)
    loss = criterion(y_pred_nl, y_train)
    loss.backward()
    optimizer.step()
    print(f"nonlin_loss = {loss}")

# Evaluate the nonlinear model on the test set
y_test_pred_nl = nonlinear_model(x_test)
nonlinear_test_loss = criterion(y_test_pred_nl, y_test).item()

# Calculate ARV (Average Relative Variance)
def compute_arv(predictions, targets):
    residuals = targets - predictions
    residual_variance = torch.var(residuals)
    target_variance = torch.var(targets)
    return (residual_variance / target_variance).item()

linear_arv = compute_arv(y_test, y_test_pred_lin)
nonlinear_arv = compute_arv(y_test, y_test_pred_nl)

# Plot predictions versus actual data
plt.figure(figsize=(12, 6))

# Linear Model Plot
plt.subplot(1, 2, 1)
plt.scatter(x_test, y_test, color="blue", label="True Data")
plt.scatter(x_test, y_test_pred_lin.detach(), color="red", label="Predicted (Linear)")
plt.title(f"Linear Model\nMSE: {linear_test_loss:.4f}, ARV: {linear_arv:.4f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Nonlinear Model Plot
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_test, color="blue", label="True Data")
plt.scatter(x_test, y_test_pred_nl.detach(), color="green", label="Predicted (Nonlinear)")
plt.title(f"Nonlinear Model\nMSE: {nonlinear_test_loss:.4f}, ARV: {nonlinear_arv:.4f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()