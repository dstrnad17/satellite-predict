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

num_epochs = 5  # Number of epochs
num_boot_reps = 2 # Number of bootstrap repetitions

file_pattern = ["goes8_*.pkl", "cluster1_*.pkl", "themise_*.pkl"]  # Desired satellites

# Define input and output column names
inputs = ["r", "theta", "phi", "vsw", "ey", "imfbz", "nsw"]
field = ["bx[nT]", "by[nT]", "bz[nT]"]
position_cart = ["x[km]", "y[km]", "z[km]"]
position_sph = ["r[km]", "theta[deg]", "phi[deg]"]

data_directory = "./data/"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define directories to save plots and results
results_directory = './main_results/'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
pkl_results_directory = './main_results/results_pkl'
if not os.path.exists(pkl_results_directory):
    os.makedirs(pkl_results_directory)

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

# Load and preprocess data
def data_load(data_directory, file_pattern, position_cart, position_sph):
    files = glob.glob(os.path.join(data_directory, file_pattern))
    dataframe_combo = [pd.read_pickle(f) for f in files]

    # Load and preprocess data
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

    return combined_df

# Training and testing functions
def train_and_test(num_inputs, num_boot_reps, removed_input=None):
    results_dict = {}  # Dictionary to hold results for all repetitions

    for boot_rep in range(num_boot_reps):
        print(f"Processing repetition {boot_rep + 1}")
        current_inputs = [inp for inp in inputs if inp != removed_input] if removed_input else inputs
        train_df = combined_df.sample(frac=0.8, random_state=boot_rep)
        test_df = combined_df.drop(train_df.index)

        # Fill missing values
        train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
        test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

        train_inputs = torch.tensor(train_df[current_inputs].values, dtype=torch.float32).to(device)
        train_targets = torch.tensor(train_df[field].values, dtype=torch.float32).to(device)
        test_inputs = torch.tensor(test_df[current_inputs].values, dtype=torch.float32).to(device)
        test_targets = torch.tensor(test_df[field].values, dtype=torch.float32).to(device)

        # Train multi-output neural network
        model_multi = SatNet(num_inputs=len(current_inputs)).to(device)
        opt_multi = torch.optim.Adam(model_multi.parameters(), lr=0.0006)

        for epoch in range(num_epochs):
            total_loss = 0
            for data, target in DataLoader(TensorDataset(train_inputs, train_targets), batch_size=256, shuffle=True):
                opt_multi.zero_grad()
                loss = nn.MSELoss()(model_multi(data), target)
                loss.backward()
                opt_multi.step()
                total_loss += loss.item()

        model_multi.eval()
        with torch.no_grad():
            nn_preds = model_multi(test_inputs).cpu().numpy()  # Multi-output NN predictions
        
        # Train and test single-output neural networks for bx, by, and bz
        nn_single_outputs = {}
        for i, component in enumerate(['bx', 'by', 'bz']):
            model_single = SatNet(num_inputs=len(current_inputs), single_output=True).to(device)
            opt_single = torch.optim.Adam(model_single.parameters(), lr=0.0006)

            for epoch in range(num_epochs):
                total_loss = 0
                for data, target in DataLoader(TensorDataset(train_inputs, train_targets[:, i:i+1]), batch_size=256, shuffle=True):
                    opt_single.zero_grad()
                    loss = nn.MSELoss()(model_single(data), target.squeeze(-1))  # Reshape target to [batch_size]
                    loss.backward()
                    opt_single.step()
                    total_loss += loss.item()

            model_single.eval()
            with torch.no_grad():
                nn_single_outputs[component] = model_single(test_inputs).cpu().numpy()

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(train_df[current_inputs], train_df[field])
        lr_preds = lr_model.predict(test_df[current_inputs])

        # Collect results
        boot_rep_results = pd.DataFrame({
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

        pd.set_option('display.float_format', '{:.3f}'.format)
        results_dict[f'rep_{boot_rep + 1}'] = boot_rep_results

    return results_dict

def save_results(results_dict, file_pattern, removed_input, results_directory):
    # Create a results subdirectory based on the file pattern
    results_subdirectory = os.path.join(results_directory, file_pattern.split('_')[0])
    if not os.path.exists(results_subdirectory):
        os.makedirs(results_subdirectory)

    # Define the filenames based on removed input
    removed_label = "all" if removed_input is None else removed_input
    pkl_filename = f"{file_pattern.split('_')[0]}_{removed_label}.pkl"
    pkl_filepath = os.path.join(results_subdirectory, pkl_filename)  # Save .pkl in the same subdirectory

    dat_filename = f"{file_pattern.split('_')[0]}_{removed_label}.dat"
    dat_filepath = os.path.join(results_subdirectory, dat_filename)

    # Save the dictionary of DataFrames to a .pkl file
    with open(pkl_filepath, 'wb') as f:
        pd.to_pickle(results_dict, f)

    print(f"All repetitions saved to '{pkl_filepath}'")
    
    # Save the dictionary to a .dat file
    with open(dat_filepath, 'w') as f:
        # Iterate over each DataFrame in the results_dict
        for key, df in results_dict.items():
            f.write(f"# {key} DataFrame\n")  # Add a header for each DataFrame
            df.to_csv(f, sep='\t', index=False)  # Use tab as separator
            f.write('\n')  # Add a blank line between DataFrames

    print(f"All repetitions saved to '{dat_filepath}'")

###### Main Execution Begins ######

# Create dataframe for each file pattern
dataframes = {pattern: data_load(data_directory, pattern, position_cart, position_sph) for pattern in file_pattern}

# Iterate over each dataframe
for file_pattern, combined_df in dataframes.items():
    print(f"Processing file pattern: {file_pattern}")

    # Loop to train with all parameters and each parameter removed
    for input_to_remove in [None] + inputs:  # None indicates all parameters
        if input_to_remove is None:
            print("Training model with no parameters removed")
        else:
            print(f"Training model with {input_to_remove} removed")

        # Train and evaluate the neural network model
        results_dict = train_and_test(num_inputs=len(inputs), num_boot_reps=num_boot_reps, removed_input=None)
        removed_input = "None" if input_to_remove is None else input_to_remove

        # Store full results
        save_results(results_dict, file_pattern, removed_input, results_directory)