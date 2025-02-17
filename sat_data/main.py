import os
import glob
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

import multiprocessing

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

data_directory = "./data/"
start = time.time()

ny = 2              # Number of years to use; Use 'All' for all years
num_epochs = 3      # Number of epochs
num_boot_reps = 1   # Number of bootstrap repetitions

parallel = False     # Parallel processing
batch_size = 256
lr = 0.006           # Learning rate
print(f"Batch Size = {batch_size}, Learning Rate = {lr}")

all_patterns = False
if all_patterns:
    file_patterns = ["cluster1_*.pkl", "goes8_*.pkl", "themise_*.pkl"]  # Desired satellites
else:
    file_patterns = ["cluster1_*.pkl"]

# Define inputs and outputs
all_input_model = True
leave_outs = ["r[km]", "theta[deg]"]   # ["r[km]", "theta[deg]", etc.] or None
inputs = ["r[km]", "theta[deg]", "phi[deg]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]
outputs = ["bx[nT]", "by[nT]", "bz[nT]"]
output_bases = [output.split('[')[0] for output in outputs]
# Code assumes columns with these unit structures (i.e. bx[nT]), modification may be needed depending on column names

# Labels for derived columns used to convert from Cartesian to spherical
position_sph = ["r[km]", "theta[deg]", "phi[deg]"]
position_cart = ["x[km]", "y[km]", "z[km]"]

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")    # Use MPS on Mac
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
    def __init__(self, num_inputs, num_outputs, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),  # Single or multiple outputs
        )

    def forward(self, x):
        output = self.network(x)
        return output

# Function to convert Cartesian to spherical coordinates
def appendSpherical_np(xyz):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r = np.sqrt(xy + xyz[:, 2]**2)
    theta = np.arctan2(xyz[:, 2], np.sqrt(xy)) * 180 / np.pi
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi) * 180 / np.pi
    return np.column_stack((r, theta, phi))

# Function to compute average relative variance (ARV), this function appears in each program
def compute_arv(A, P):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(P, pd.DataFrame):
        P = P.to_numpy()

    if A.ndim == 1:
        A = A[:, np.newaxis]
    if P.ndim == 1:
        P = P[:, np.newaxis]

    arvs = []
    for i in range(A.shape[1]):
        var_A = np.var(A[:, i])
        arv = np.var(A[:, i] - P[:, i]) / var_A if var_A > 0 else np.nan
        arvs.append(arv)
    
    return arvs

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

        df['datetime'] = pd.to_datetime(
            df[['year', 'month', 'day', 'hour', 'minute', 'second']]
        )

        dataframes.append(df)

    return dataframes

def train_and_test(combined_dfs, file_pattern, **kwargs):
    num_boot_reps = kwargs.get('num_boot_reps', 1)
    inputs = kwargs.get('inputs', None)
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
            print(f"    Repetition {rep + 1}/{num_boot_reps} started at {time.time()-start}")
            train_boot = train_data.sample(frac=0.8, random_state=rep)
            test_data = datasets[i] if is_loo else datasets[0]  # Test data comes from the current fold
            rep_results = process_single_rep(train_boot, test_data, inputs, outputs, removed_input=removed_input)
            results[f'rep_{rep + 1}'] = rep_results

        # Save the results after all repetitions are completed
        save_results(results, file_pattern, removed_input, results_directory, method=method_label)

# Generic function to train the model (multi-output or single-output)
def train_model(model, train_inputs, train_targets, opt, outputs):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    data_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True)

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        
        opt.zero_grad()
        predictions = model(data)

        # Compute the loss for multi-output
        loss = nn.MSELoss()(predictions, target)
        loss.backward()
        opt.step()
        total_loss += loss.item()

        all_predictions.append(predictions.detach().cpu().numpy())
        all_targets.append(target.cpu().numpy())

    # Concatenate predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute and print ARV/loss
    arvs = compute_arv(all_targets, all_predictions)
    for base, arv in zip(outputs, arvs):
        print(f" | {base} ARV = {arv:.3f}", end='')
    print(f" loss = {total_loss:.4f}")

    return total_loss, all_predictions, all_targets, arvs

def process_single_rep(train_df, test_df, inputs, outputs, removed_input=None):
    indent = "      "

    current_inputs = [inp for inp in inputs if inp != removed_input] if removed_input else inputs

    # Fill missing values and normalize
    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

    scaler_inputs = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    train_inputs_scaled = scaler_inputs.fit_transform(train_df[current_inputs])
    train_targets_scaled = scaler_targets.fit_transform(train_df[outputs])
    test_inputs_scaled = scaler_inputs.transform(test_df[current_inputs])
    test_targets = test_df[outputs].values

    # Convert to tensors
    train_inputs = torch.tensor(train_inputs_scaled, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_targets_scaled, dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test_inputs_scaled, dtype=torch.float32).to(device)

    # Train multi-output neural network
    print(f"{indent}Training multi-output neural network")
    model_multi = SatNet(num_inputs=len(current_inputs), num_outputs=len(outputs)).to(device)
    opt_multi = torch.optim.Adam(model_multi.parameters(), lr)
    for epoch in range(num_epochs):
        print(f"{indent}  Epoch {epoch + 1}/{num_epochs}")
        _, _, _, _ = train_model(model_multi, train_inputs, train_targets, opt_multi, outputs)

    # Test multi-output neural network
    model_multi.eval()
    with torch.no_grad():
        nn3_preds = model_multi(test_inputs).cpu().numpy()
        nn3_preds = scaler_targets.inverse_transform(nn3_preds)

    # Train single-output neural networks
    nn1_preds = {}
    for i, output in enumerate(outputs):
        print(f"{indent}Training single-output neural network for {output}")
        model_single = SatNet(num_inputs=len(current_inputs), num_outputs=1).to(device)
        opt_single = torch.optim.Adam(model_single.parameters(), lr)

        for epoch in range(num_epochs):
            print(f"{indent}  Epoch {epoch + 1}/{num_epochs}")
            _, _, _, _ = train_model(
                model_single,
                train_inputs,
                train_targets[:, i:i + 1],
                opt_single,
                output
            )

        model_single.eval()
        with torch.no_grad():
            nn1_preds[output] = model_single(test_inputs).cpu().numpy()

    # Combine single-output predictions
    nn1_preds_combined = np.column_stack([nn1_preds[output] for output in outputs])
    nn1_preds_descale = scaler_targets.inverse_transform(nn1_preds_combined)

    # Linear Regression baseline
    print(f"{indent}Performing linear regression")
    lr_model = LinearRegression()
    lr_model.fit(train_df[current_inputs], train_df[outputs])
    lr_preds = lr_model.predict(test_df[current_inputs])

    # Compile results
    results = {
        'timestamp': test_df['datetime'].values,
    }
    for i, output in enumerate(outputs):
        results[f'{output}_actual'] = test_targets[:, i]
        results[f'{output}_nn3'] = nn3_preds[:, i]
        results[f'{output}_nn1'] = nn1_preds_descale[:, i]
        results[f'{output}_lr'] = lr_preds[:, i]

    results_df = pd.DataFrame(results)
    print(results_df)

    return results_df

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

def job(**kwargs):
    file_pattern = kwargs['file_pattern']
    print("-" * 50)
    print(f"Processing file pattern: {file_pattern} in {data_directory}")

    # Load data using data_load function
    combined_dfs = data_load(data_directory, file_pattern, position_cart, position_sph)

    if ny == 'All':
        ny_min = len(combined_dfs)  # Use all available data if ny is 'All'
    else:
        ny_min = min(ny, len(combined_dfs))  # Limit to ny if it's a number
        if len(combined_dfs) > 1:
            combined_dfs = combined_dfs[:ny_min]

    print(f"Loaded {len(combined_dfs)} datasets from pattern: {file_pattern}")

    if not combined_dfs:
        print(f"No datasets found for pattern: {file_pattern}. Skipping.")
        return

    if kwargs['removed_input'] is None:
        print("Training model with no parameters removed")
    else:
        print(f"Training model with '{kwargs['removed_input']}' removed")

    train_and_test(combined_dfs=combined_dfs, **kwargs)

if leave_outs is None:
    leave_outs = []
for leave_out in leave_outs:
    if leave_out not in inputs:
        raise ValueError(f"leave_out variable = '{leave_out}' not in inputs = {inputs}")

job_inputs = []
for file_pattern in file_patterns:
    kwargs = {
        "file_pattern": file_pattern,
        "inputs": inputs,
        "num_boot_reps": num_boot_reps,
    }
    if all_input_model:
        kwargs['removed_input'] = None
        job_inputs.append(kwargs.copy())
    for leave_out in leave_outs:
        leave_out_kwargs = kwargs.copy()
        leave_out_kwargs['removed_input'] = leave_out
        job_inputs.append(leave_out_kwargs)

def job_wrapper(args):
    return job(**args)

if __name__ == '__main__':
    if parallel:
        ncpu = multiprocessing.cpu_count()
        print(f"# cpus in parallel processing: {ncpu}")
        with multiprocessing.Pool(ncpu - 1 if ncpu > 1 else 1) as p:
            p.map(job_wrapper, job_inputs)  # Use job_wrapper instead of lambda
    else:
        for job_input in job_inputs:
            job(**job_input)