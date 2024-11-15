import os
import glob

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

num_epochs = 2  # Number of epochs
num_boot_reps = 2  # Number of bootstrap repetitions
ny = 2 # Number of years to use

all = False
if all:
    file_pattern = ["cluster1_*.pkl", "goes8_*.pkl", "themise_*.pkl"]  # Desired satellites
else:
    file_pattern = ["cluster1_*.pkl"]

# Define input and output column names
inputs = ["r", "theta", "phi", "vsw", "ey", "imfbz", "nsw"]
field = ["bx[nT]", "by[nT]", "bz[nT]"]
position_cart = ["x[km]", "y[km]", "z[km]"]
position_sph = ["r[km]", "theta[deg]", "phi[deg]"]

data_directory = "./data/"


# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS on Mac
elif torch.cuda.is_available():
    device = torch.device("cuda")   # Use CUDA on Windows/Linux
else:
    device = torch.device("cpu")    # CPU Fallback
print(f"Using device: {device}")

# Define directories to save plots and results
results_directory = './main_results/'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Define neural network
class SatNet(nn.Module):
    def __init__(self, num_inputs, hidden_size=32, single_output=False, output_index=None):
        super().__init__()
        self.single_output = single_output
        self.output_index = output_index  # Index to select bx, by, bz
        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1 if single_output else 3)  # Single or three outputs
        )

    def forward(self, x):
        output = self.network(x)
        if self.single_output:
            return output.squeeze(-1)  # Return a single output if single_output=True
        return output

# Function to convert Cartesian to spherical coordinates
def appendSpherical_np(xyz):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r = np.sqrt(xy + xyz[:, 2]**2)
    theta = np.arctan2(xyz[:, 2], np.sqrt(xy)) * 180 / np.pi
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi) * 180 / np.pi
    return np.column_stack((r, theta, phi))

# Function to load and preprocess data
def data_load(data_directory, file_pattern, position_cart, position_sph):
    files = glob.glob(os.path.join(data_directory, file_pattern))
    
    dataframes = []
    for f in files:
        df = pd.read_pickle(f)  # Load the DataFrame from pickle

        cartesian = df[position_cart].to_numpy()  
        spherical = appendSpherical_np(cartesian) 
        
        # Add spherical coordinates to the DataFrame
        for i, col in enumerate(position_sph):
            df[col] = spherical[:, i]
        
        df.rename(columns={
            "r[km]": "r",
            "theta[deg]": "theta",
            "phi[deg]": "phi",
            "vsw[km/s]": "vsw",
            "ey[mV/m]": "ey",
            "imfbz[nT]": "imfbz",
            "nsw[1/cm^3]": "nsw"
        }, inplace=True)

        df['datetime'] = pd.to_datetime(
            df[['year', 'month', 'day', 'hour', 'minute', 'second']]
        )

        dataframes.append(df)

    return dataframes

def train_and_test(combined_dfs, file_pattern, **kwargs):
    num_boot_reps = kwargs.get('num_boot_reps', 1)
    removed_input = kwargs.get('removed_input', None)
    results_directory = kwargs.get('results_directory', './main_results/')

    is_loo = len(combined_dfs) > 1  # Check if leave-one-out is needed
    datasets = combined_dfs if is_loo else [combined_dfs[0]]

    for i, test_data in enumerate(datasets):
        if is_loo:
            # For leave-one-out, concatenate training datasets excluding the current test dataset
            print(f"  Leave-one-out: Using segment {i + 1}/{len(datasets)} as the test set")
            train_data = pd.concat([df for j, df in enumerate(datasets) if j != i], ignore_index=True)
            method_label = f"LOO_{i + 1}"
        else:
            # For full-data, use the full dataset
            print(f"  Full data repetition {i + 1}/{num_boot_reps}")
            train_data = datasets[0]
            method_label = "full_data"

        # Loop for bootstrapping and processing repetitions
        results = {}
        for rep in range(num_boot_reps):
            print(f"    Repetition {rep + 1}/{num_boot_reps}")
            train_boot = train_data.sample(frac=0.8, random_state=rep)
            test_data = datasets[i] if is_loo else datasets[0]  # Test data comes from the current fold
            rep_results = process_single_rep(train_boot, test_data, removed_input)
            results[f'rep_{rep + 1}'] = rep_results

        # Save the results after all repetitions are completed
        save_results(results, file_pattern, removed_input, results_directory, method=method_label)

# Helper function to process a single training/testing repetition
def process_single_rep(train_df, test_df, removed_input=None):

    indent = "      "

    # Determine the current input features
    current_inputs = [inp for inp in inputs if inp != removed_input] if removed_input else inputs
    num_inputs = len(current_inputs)  # Calculate number of inputs based on current inputs

    # Fill missing values
    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

    # Convert data to tensors
    train_inputs = torch.tensor(train_df[current_inputs].values, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_df[field].values, dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test_df[current_inputs].values, dtype=torch.float32).to(device)
    test_targets = torch.tensor(test_df[field].values, dtype=torch.float32).to(device)

    # Neural network training with multi-output
    model_multi = SatNet(num_inputs=num_inputs).to(device)
    opt_multi = torch.optim.Adam(model_multi.parameters(), lr=0.0006)

    print(f"{indent}Training multi-output neural network")
    for epoch in range(num_epochs):
        print(f"{indent}  Epoch {epoch + 1}/{num_epochs}", end='')
        total_loss = 0
        for data, target in DataLoader(TensorDataset(train_inputs, train_targets), batch_size=256, shuffle=True):
            opt_multi.zero_grad()
            loss = nn.MSELoss()(model_multi(data), target)
            loss.backward()
            opt_multi.step()
            total_loss += loss.item()
        print(f" loss = {loss}")

    model_multi.eval()
    with torch.no_grad():
        nn_preds = model_multi(test_inputs).cpu().numpy()  # Multi-output NN predictions

    # Single-output neural networks for bx, by, and bz
    print(f"{indent}Training single-output neural networks")
    nn_single_outputs = {}
    for i, component in enumerate(['bx', 'by', 'bz']):
        model_single = SatNet(num_inputs=len(current_inputs), single_output=True).to(device)
        opt_single = torch.optim.Adam(model_single.parameters(), lr=0.0006)
        print(f"{indent}  Training {component} neural network")

        for epoch in range(num_epochs):
            print(f"{indent}    Epoch {epoch + 1}/{num_epochs}", end='')
            for data, target in DataLoader(TensorDataset(train_inputs, train_targets[:, i:i+1]), batch_size=256, shuffle=True):
                opt_single.zero_grad()
                loss = nn.MSELoss()(model_single(data), target.squeeze(-1))
                loss.backward()
                opt_single.step()
            print(f" loss = {loss}")

        model_single.eval()
        with torch.no_grad():
            nn_single_outputs[component] = model_single(test_inputs).cpu().numpy()

    # Linear Regression as a baseline
    print(f"{indent}Performing linear regresssion")
    lr_model = LinearRegression()
    lr_model.fit(train_df[current_inputs], train_df[field])
    lr_preds = lr_model.predict(test_df[current_inputs])

    # Collect results in a DataFrame
    rep_results = pd.DataFrame({
        'timestamp': test_df['datetime'].values,
        'bx_actual': test_targets[:, 0].cpu().numpy().round(3),
        'by_actual': test_targets[:, 1].cpu().numpy().round(3),
        'bz_actual': test_targets[:, 2].cpu().numpy().round(3),
        'bx_nn3': nn_preds[:, 0].round(3),
        'by_nn3': nn_preds[:, 1].round(3),
        'bz_nn3': nn_preds[:, 2].round(3),
        'bx_nn1': nn_single_outputs['bx'].round(3),
        'by_nn1': nn_single_outputs['by'].round(3),
        'bz_nn1': nn_single_outputs['bz'].round(3),
        'bx_lr': lr_preds[:, 0].round(3),
        'by_lr': lr_preds[:, 1].round(3),
        'bz_lr': lr_preds[:, 2].round(3),
    })

    return rep_results

# Save results to file
def save_results(results_dict, file_pattern, removed_input, results_directory, method="leave"):
    # Create a subdirectory for each removed input parameter
    removed_label = "none" if removed_input is None else removed_input
    results_subdirectory = os.path.join(results_directory, file_pattern.split('_')[0], removed_label)
    if not os.path.exists(results_subdirectory):
        os.makedirs(results_subdirectory)

    # Define the filenames based on method and removed input
    method_label = "full_data" if method == "full" else method
    pkl_filename = f"{file_pattern.split('_')[0]}_{method_label}.pkl"
    pkl_filepath = os.path.join(results_subdirectory, pkl_filename)  # Save .pkl in respective subdirectory

    dat_filename = f"{file_pattern.split('_')[0]}_{method_label}.dat"
    dat_filepath = os.path.join(results_subdirectory, dat_filename)

    # Save to pickle file
    pd.to_pickle(results_dict, pkl_filepath)
    print(f"  Saved '{pkl_filepath}'")

    # Optionally, save to a text file for additional analysis
    with open(dat_filepath, 'w') as f:
        for rep, df in results_dict.items():
            f.write(f"# {rep}\n")
            df.to_string(f)
            f.write("\n\n")
    
    print(f"  Saved '{dat_filepath}'")

###### Main Execution Begins ######

kwargs = {
    'num_boot_reps': num_boot_reps,
}

# Iterate over each file pattern
for pattern in file_pattern:
    print("-" * 50)
    print(f"Processing file pattern: {pattern}")

    # Load data using data_load function
    combined_dfs = data_load(data_directory, pattern, position_cart, position_sph)

    ny = min(ny, len(combined_dfs))  # Limit to ny for testing
    if len(combined_dfs) > 1:
        combined_dfs = combined_dfs[0:ny]

    print(f"Loaded {len(combined_dfs)} datasets from pattern: {pattern}")

    if not combined_dfs:
        print(f"No datasets found for pattern: {pattern}. Skipping.")
        continue

    # Determine whether to use leave-one-out or full data
    use_leave_one_out = len(combined_dfs) > 1
    print(f"Leave-one-out mode: {use_leave_one_out}")

    # Train models with all parameters and each parameter removed
    for input_to_remove in [None] + inputs:
        kwargs['removed_input'] = input_to_remove

        if input_to_remove is None:
            print("Training model with no parameters removed")
        else:
            print(f"Training model with '{input_to_remove}' removed")

        train_and_test(combined_dfs=combined_dfs, file_pattern=pattern, **kwargs)