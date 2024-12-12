import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from .arv import arv

def _device(device_name):

  if device_name is None:
    return None

  if device_name == 'mps':
      if torch.backends.mps.is_available():
        device = torch.device("mps")    # Use MPS on Mac

  if device_name == 'cuda':
    if torch.cuda.is_available():
      device = torch.device("cuda")   # Use CUDA on Windows/Linux

  if device_name == 'cpu':
    if torch.cuda.is_available():
      device = torch.device("cpu")   # Use CUDA on Windows/Linux

  return device

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

def train_and_test(combined_dfs, tag, **kwargs):
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
            print(f"    Repetition {rep + 1}/{num_boot_reps} started")
            train_boot = train_data.sample(frac=0.8, random_state=rep)
            test_data = datasets[i] if is_loo else datasets[0]  # Test data comes from the current fold
            rep_results = process_single_rep(train_boot, test_data, removed_input=removed_input, **kwargs)
            results[f'rep_{rep + 1}'] = rep_results

        # Save the results after all repetitions are completed
        save_results(results, tag, removed_input, results_directory, method=method_label)

# Helper function to process a single training/testing repetition
def process_single_rep(train_df, test_df, removed_input=None, **kwargs):
    indent = "      "

    lr = kwargs.get('lr', 0.0006)
    inputs = kwargs.get('inputs', None)
    outputs = kwargs.get('outputs', None)
    num_epochs = kwargs.get('num_epochs', 2)
    batch_size = kwargs.get('batch_size', 256)
    device_name = kwargs.get('device')
    device = _device(device_name)

    if device is None:
      print(f"{indent}Device '{device_name}' is not available. Using 'cpu' instead.")
      device = torch.device("cpu")
    else:
      print(f"{indent}Using device: {device}")

    # Determine the current input features
    current_inputs = [inp for inp in inputs if inp != removed_input] if removed_input else inputs
    num_inputs = len(current_inputs)

    # Fill missing values
    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

    # Normalize the data
    scaler_inputs = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    train_inputs_scaled = pd.DataFrame(scaler_inputs.fit_transform(train_df[current_inputs]), 
                                    columns=current_inputs, index=train_df.index)
    train_targets_scaled = pd.DataFrame(scaler_targets.fit_transform(train_df[outputs]), 
                                    columns=outputs, index=train_df.index)
    test_inputs_scaled = pd.DataFrame(scaler_inputs.transform(test_df[current_inputs]), 
                                    columns=current_inputs, index=test_df.index)

    # Convert data to tensors
    train_inputs = torch.tensor(train_inputs_scaled.values, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_targets_scaled.values, dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test_inputs_scaled.values, dtype=torch.float32).to(device)
    test_targets = torch.tensor(test_df[outputs].values, dtype=torch.float32).to(device)

    # Neural network training with multi-output
    model_multi = SatNet(num_inputs=num_inputs).to(device)
    opt_multi = torch.optim.Adam(model_multi.parameters(), lr)

    print(f"{indent}Training multi-output neural network")
    for epoch in range(num_epochs):
        print(f"{indent}  Epoch {epoch + 1}/{num_epochs}", end='')
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for data, target in DataLoader(TensorDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True):
            opt_multi.zero_grad()
            predictions = model_multi(data)
            loss = nn.MSELoss()(predictions, target)
            loss.backward()
            opt_multi.step()
            total_loss += loss.item()

            # Collect all predictions and targets for ARV calculation
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(target.cpu().numpy())

        # Compute ARV for each output
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        arvs = arv(all_predictions, all_targets)
        for output, _arv in zip(outputs, arvs):
            print(f" | {output} ARV = {_arv:.3f}", end='')
        print(f" loss = {total_loss:.4f}")

    model_multi.eval()
    with torch.no_grad():
        nn3_preds = model_multi(test_inputs).cpu().numpy()  # Multi-output NN predictions
        # Denormalize predictions
        nn3_preds = scaler_targets.inverse_transform(nn3_preds)

    # Single-output neural networks
    print(f"{indent}Training single-output neural networks")
    nn1_preds = {}
    for output in outputs:
        model_single = SatNet(num_inputs=len(current_inputs), single_output=True).to(device)
        opt_single = torch.optim.Adam(model_single.parameters(), lr)
        print(f"{indent}  Training {output} neural network")

        for epoch in range(num_epochs):
            print(f"{indent}    Epoch {epoch + 1}/{num_epochs}", end='')
            target_index = outputs.index(output)
            for data, target in DataLoader(TensorDataset(train_inputs, train_targets[:, target_index:target_index+1]), batch_size=batch_size, shuffle=True):
                opt_single.zero_grad()
                loss = nn.MSELoss()(model_single(data), target.squeeze(-1))
                A = model_single(data).detach().cpu().squeeze(-1).numpy()
                P = target.detach().cpu().squeeze(-1).numpy()
                _arv = arv(A,P)
                loss.backward()
                opt_single.step()
            print(f" loss = {loss:.4f}; ARV = {_arv:.3f}")

        model_single.eval()
        with torch.no_grad():
            nn1_preds[output] = model_single(test_inputs).cpu().numpy()
        # Denormalize predictions
    nn1_preds_combine = np.column_stack([nn1_preds[output] for output in outputs])
    nn1_preds_descale = scaler_targets.inverse_transform(nn1_preds_combine)
    for idx, output in enumerate(outputs):
        nn1_preds[output] = nn1_preds_descale[:, idx]

    # Linear Regression as a baseline
    print(f"{indent}Performing linear regression")
    lr_model = LinearRegression()
    lr_model.fit(train_df[current_inputs], train_df[outputs])
    lr_preds = lr_model.predict(test_df[current_inputs])
    arvs = arv(test_df[outputs].values, lr_preds)
    for output, _arv in zip(outputs, arvs):
        print(f"| {output} ARV = {_arv:.3f}", end='')

  #  import pdb; pdb.set_trace()
    model_types = ['nn3', 'nn1', 'lr']
    results_dict = {
        'timestamp': test_df['datetime'].values,
    }

    for output in outputs:
        results_dict[f'{output}_actual'] = test_targets[:, outputs.index(output)].cpu().numpy()

    for model in model_types:
        for output in outputs:
            if model == 'nn3':
                results_dict[f'{output}_{model}'] = nn3_preds[:, outputs.index(output)].flatten()
            elif model == 'nn1':
                results_dict[f'{output}_{model}'] = nn1_preds[output].flatten()
            elif model == 'lr':
                results_dict[f'{output}_{model}'] = lr_preds[:, outputs.index(output)].flatten()

    rep_results = pd.DataFrame(results_dict)
    #print(rep_results)

    return rep_results

# Save results to file
def save_results(results_dict, tag, removed_input, results_directory, method="leave"):
    import os
    # Create a subdirectory for each removed input parameter
    removed_label = "none" if removed_input is None else removed_input
    results_subdirectory = os.path.join(results_directory, tag.split('_')[0], removed_label)
    if not os.path.exists(results_subdirectory):
        os.makedirs(results_subdirectory)

    # Define the filenames based on method and removed input
    method_label = "full_data" if method == "full" else method
    pkl_filename = f"{tag.split('_')[0]}_{method_label}.pkl"
    pkl_filepath = os.path.join(results_subdirectory, pkl_filename)  # Save .pkl in respective subdirectory

    dat_filename = f"{tag.split('_')[0]}_{method_label}.dat"
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
