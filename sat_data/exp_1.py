import torch
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to convert Cartesian to spherical coordinates
def appendSpherical_np(xyz):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r = np.sqrt(xy + xyz[:, 2]**2)
    theta = np.arctan2(xyz[:, 2], np.sqrt(xy)) * 180 / np.pi
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi) * 180 / np.pi
    return np.column_stack((r, theta, phi))

# Define input and output column names
inputs = ["r", "theta", "phi", "vsw", "ey", "imfbz", "nsw"]
field = ["bx[nT]", "by[nT]", "bz[nT]"]
position_cart = ["x[km]", "y[km]", "z[km]"]
position_sph = ["r[km]", "theta[deg]", "phi[deg]"]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SatNet(nn.Module):
    def __init__(self, num_inputs, hidden_size=50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
    def forward(self, x):
        return self.network(x)

# Training and testing functions
def train_and_test(num_inputs, removed_input=None):
    current_inputs = [inp for inp in inputs if inp != removed_input] if removed_input else inputs
    train_df = combined_df.sample(frac=0.8, random_state=42)
    test_df = combined_df.drop(train_df.index)

    # Delete data that has NA values
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    
    # Define inputs
    train_inputs = torch.tensor(train_df[current_inputs].values, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_df[field].values, dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test_df[current_inputs].values, dtype=torch.float32).to(device)
    test_targets = torch.tensor(test_df[field].values, dtype=torch.float32).to(device)

    train_ds = TensorDataset(train_inputs, train_targets)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    # Define model
    model = SatNet(num_inputs=len(current_inputs)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    num_epochs = 20  # Number of epochs
    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in train_dl:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = nn.MSELoss()(model(data), target)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_dl)}")

    model.eval()
    with torch.no_grad():
        predicted = model(test_inputs)
        test_loss = nn.MSELoss()(predicted, test_targets).item()
        # Extracting the datetime values for the test set
        test_time = test_df['datetime'].values
        # Returning predicted, real values, and time for plotting
        return test_loss, predicted.cpu().numpy(), test_targets.cpu().numpy(), test_time

# Linear regression functions
def lin_reg(current_inputs):
    train_df = combined_df.sample(frac=0.8, random_state=42)
    test_df = combined_df.drop(train_df.index)
    
    # Delete data that has NA values
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    
    # Define inputs and targets
    X_train = train_df[current_inputs]
    y_train = train_df[field]
    X_test = test_df[current_inputs]
    y_test = test_df[field]
    
    # Initialize and train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Compute mean squared error
    test_loss = mean_squared_error(y_test, y_pred)
    
    # Extracting the corresponding datetime values for the test set
    test_time = test_df['datetime'].values
    
    # Returning test loss, predicted values, actual values, and time for plotting
    return test_loss, y_pred, y_test.values, test_time

# Load and preprocess data
directory = "./data/"
files = glob.glob(os.path.join(directory, "cluster1_*.pkl"))
dataframe_combo = [pd.read_pickle(f) for f in files]

for df in dataframe_combo:
    cartesian = df[position_cart].to_numpy()
    df[position_sph] = appendSpherical_np(cartesian)

combined_df = pd.concat(dataframe_combo, ignore_index=True)
combined_df.rename(columns={
    "r[km]": "r", 
    "theta[deg]": "theta", 
    "phi[deg]": "phi", 
    "vsw[km/s]": "vsw", 
    "ey[mV/m]": "ey", 
    "imfbz[nT]": "imfbz", 
    "nsw[1/cm^3]": "nsw"
}, inplace=True)

combined_df['datetime'] = pd.to_datetime(
    combined_df[['year', 'month', 'day', 'hour', 'minute', 'second']]
)

# Define the directory to save plots
plot_directory = './exp_1_plots/'

# Create the directory if it does not exist
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# Define error plotting
def plot_error(preds, real, time, title, input_to_remove=None):
    sorted_indices = np.argsort(time)
    time = time[sorted_indices]
    preds = preds[sorted_indices]
    real = real[sorted_indices]

    plt.style.use('default')
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(field):
        plt.subplot(3, 1, i + 1)
        error = preds[:, i] - real[:, i]
        plt.plot(time, error, label=f'{col} Error')
        plt.ylabel('Error')
        plt.title(f'{col} Error')
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    filename = f'{plot_directory}exp_1_no_{input_to_remove}.png' if input_to_remove else f'{plot_directory}exp_1_all_params.png'
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()

def plot_reg_error(preds, real, time, title):
    sorted_indices = np.argsort(time)
    time = time[sorted_indices]
    preds = preds[sorted_indices]
    real = real[sorted_indices]

    plt.style.use('default')
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(field):
        plt.subplot(3, 1, i + 1)
        error = preds[:, i] - real[:, i]
        plt.plot(time, error, label=f'{col} Error')
        plt.ylabel('Error')
        plt.title(f'{col} Error')
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(f'{plot_directory}exp_1_reg_error.png')
    print(f"Regression error plot saved as {plot_directory}exp_1_reg_error.png")

    plt.close()

# Training, evaluation, and plotting
results, losses = [], []
print(f"Training model with no parameters removed")
loss_7, preds_7, real_7, time_7 = train_and_test(num_inputs=7)
plot_error(preds_7, real_7, time_7, 'Error - All Parameters')
losses.append(loss_7)
results.append({
    "Removed Input": "None",
    **dict(zip(field, preds_7.mean(axis=0)))
})

for input_to_remove in inputs:
    print(f"Training model with {input_to_remove} removed")
    test_loss, preds, real, test_time = train_and_test(num_inputs=len(inputs) - 1, removed_input=input_to_remove)
    plot_error(preds, real, test_time, f'Error - {input_to_remove} Removed', input_to_remove)
    losses.append(test_loss)
    results.append({
        "Removed Input": input_to_remove,
        **dict(zip(field, preds.mean(axis=0)))
    })

real_mean = combined_df[field].mean()
results.append({
    "Removed Input": "Real Mean",
    **real_mean.to_dict()
})

# Run linear regression
lr_loss, lr_preds, lr_real, lr_time = lin_reg(inputs)
results.append({
    "Removed Input": "Lin Reg",
    **dict(zip(field, lr_preds.mean(axis=0)))
})

# Results
results_df = pd.DataFrame(results)
markdown_table = results_df.to_markdown(index=False)
print(markdown_table)

with open('exp_1.md', 'w') as file:
    file.write(markdown_table)
    print("Model results saved to 'exp_1.md'")

plot_reg_error(lr_preds, lr_real, lr_time, 'Regression Error')

# Plotting loss over iterations
plt.style.use('default')
plt.figure(figsize=(12, 6))
plt.plot(range(len(losses)), losses, marker='o', label='Model Loss')
plt.xlabel('Iteration')
plt.ylabel('Test Loss')
plt.title('Test Loss for Each Input Removal')
plt.xticks(range(len(losses)), ['All Parameters'] + [f'Removed {inp}' for inp in inputs], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{plot_directory}exp_1_loss.png')