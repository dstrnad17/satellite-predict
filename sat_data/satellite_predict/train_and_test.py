import os

print("Importing torch. ", end="")
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
print("Done")

print("Importing sklearn. ", end="")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
print("Done")

print("Importing numpy and pandas. ", end="")
import numpy as np
import pandas as pd
print("Done")

from .arv import arv
from .summary import table

def train_and_test(combined_dfs, **kwargs):

  tag = kwargs.get('tag', 'tag1')
  num_boot_reps = kwargs.get('num_boot_reps', 1)
  removed_inputs = kwargs.get('removed_inputs', [None])
  results_directory = kwargs.get('results_directory', './results/')

  print(f"{tag} started")
  is_loo = len(combined_dfs) > 1  # Check if leave-one-out is needed
  datasets = combined_dfs if is_loo else [combined_dfs[0]]

  for removed_input in removed_inputs:
    print(f"  Removed input: {removed_input}")
    for i, test_data in enumerate(datasets):
      if is_loo:
        # For leave-one-out, concatenate training datasets excluding the current test dataset
        print(f"    Training with DataFrame {i + 1}/{len(datasets)} excluded")
        train_data_subset = [df for j, df in enumerate(datasets) if j != i]
        train_data = pd.concat(train_data_subset, ignore_index=True)
        method_label = f"loo_{i + 1}"
      else:
        print(f"    Training using all {len(datasets)} DataFrame(s)")
        train_data = datasets[0]
        method_label = "all"

      # Loop for bootstrap repetitions
      results = []
      for rep in range(num_boot_reps):
        print(f"        Bootstrap repetition {rep + 1}/{num_boot_reps}")
        train_boot = train_data.sample(frac=0.8, random_state=rep)
        test_data = datasets[i] if is_loo else datasets[0]  # Test data comes from the current fold
        result = process_single_rep(train_boot, test_data, removed_input=removed_input, **kwargs)
        results.append(result)

      save(results, tag, removed_input, results_directory, method=method_label)

  print("  Creating tables")
  table(**kwargs)

  print(f"{tag} finished\n")


def _device(device_name):

  if device_name is None:
    return None

  if device_name == 'mps':
    if torch.backends.mps.is_available():
      device = torch.device("mps") # Use MPS on Mac

  if device_name == 'cuda':
    if torch.cuda.is_available():
      device = torch.device("cuda") # Use CUDA on Windows/Linux

  if device_name == 'cpu':
    if torch.cuda.is_available():
      device = torch.device("cpu")

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

# Helper function to process a single training/testing repetition
def process_single_rep(train_df, test_df, removed_input=None, **kwargs):

    indent = "          "

    lr = kwargs.get('lr', 0.0006)
    inputs = kwargs.get('inputs', None)
    outputs = kwargs.get('outputs', None)
    models = kwargs.get('models', ['ols', 'nn1'])
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

    results = {'actual': {'timestamp': test_df['datetime'].values}}
    for output in outputs:
      results['actual'][output] = test_targets[:, outputs.index(output)].cpu().numpy()
    results['actual'] = pd.DataFrame(results['actual'])

    for model in kwargs['models']:
      results[model] = {}
      if model != 'ols':
        results[model]['epochs'] = []
      results[model]['predicted'] = pd.DataFrame({'timestamp': test_df['datetime'].values})

    if 'ols' in models:
      # Linear regression
      print(f"{indent}Performing linear regression")
      lr_model = LinearRegression()
      lr_model.fit(train_df[current_inputs], train_df[outputs])
      lr_preds = lr_model.predict(test_df[current_inputs])
      arvs = arv(test_df[outputs].values, lr_preds)
      print(indent, end='')
      for output, _arv in zip(outputs, arvs):
          print(f" | {output} ARV = {_arv:6.3f}", end='')
      results['ols']['predicted'][outputs] = lr_preds

    if 'nn3' in models:
      # TODO: This need to be generalized to be named mimo
      # Multi-output neural network
      model_multi = SatNet(num_inputs=num_inputs).to(device)
      opt_multi = torch.optim.Adam(model_multi.parameters(), lr)

      print(f"\n{indent}Training multi-output neural network")
      for epoch in range(num_epochs):
        print(f"{indent}  Epoch {epoch + 1}/{num_epochs}", end='')
        total_loss = 0
        all_predictions = []
        all_targets = []

        tds = TensorDataset(train_inputs, train_targets)
        for data, target in DataLoader(tds, batch_size=batch_size, shuffle=True):
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
        results[model]['epochs'].append(arvs)
        print(f" loss = {total_loss:7.4f}", end='')
        for output, _arv in zip(outputs, arvs):
          print(f" | {output} ARV = {_arv:6.3f}", end='')
        print()

      model_multi.eval()

      with torch.no_grad():
        nn3_preds = model_multi(test_inputs).cpu().numpy()  # Multi-output NN predictions
        # Denormalize predictions
        nn3_preds = scaler_targets.inverse_transform(nn3_preds)
        results['nn3']['predicted'][outputs] = nn3_preds

    if 'nn1' in models:
      # TODO: Rename to miso
      # Single-output neural networks
      arv_epochs['nn1'] = []
      print(f"\n{indent}Training single-output neural networks w/ input '{removed_input}' removed")
      nn1_preds = {}
      for output in outputs:
        model_single = SatNet(num_inputs=len(current_inputs), single_output=True).to(device)
        opt_single = torch.optim.Adam(model_single.parameters(), lr)
        print(f"{indent}  Training {output} neural network")
        for epoch in range(num_epochs):
          print(f"{indent}    Epoch {epoch + 1}/{num_epochs}", end='')
          target_index = outputs.index(output)
          tds = TensorDataset(train_inputs, train_targets[:, target_index:target_index+1])
          for data, target in DataLoader(tds, batch_size=batch_size, shuffle=True):
            opt_single.zero_grad()
            loss = nn.MSELoss()(model_single(data), target.squeeze(-1))
            A = model_single(data).detach().cpu().squeeze(-1).numpy()
            P = target.detach().cpu().squeeze(-1).numpy()
            _arv = arv(A,P)
            loss.backward()
            opt_single.step()
          print(f" {output} loss = {loss:7.4f} | ARV = {_arv:6.3f}")

        model_single.eval()
        with torch.no_grad():
          nn1_preds[output] = model_single(test_inputs).cpu().numpy()

      # Denormalize predictions
      nn1_preds_combine = np.column_stack([nn1_preds[output] for output in outputs])
      nn1_preds = scaler_targets.inverse_transform(nn1_preds_combine)
      results['nn1'][outputs] = nn1_preds

    return results

def save(results_dict, tag, removed_input, results_directory, method):

    # If method = 'loo'
    # removed_input/
    #   loo/
    #     loo_1.pkl: [boot1, boot2, ...]
    #     loo_2.pkl: [boot1, boot2, ...]
    # where
    #   bootN = {actual: df, nn1: df, nn3: df, lr: df}
    # is a bootstrap repetition and
    #   df is a DataFrame with columns of timestamp and outputs

    if removed_input is None:
      removed_input = 'none'

    subdir = os.path.join(results_directory, tag, removed_input)
    if method.startswith('loo'):
      subdir = os.path.join(subdir, 'loo')

    if not os.path.exists(subdir):
      os.makedirs(subdir)

    pkl_filepath = os.path.join(subdir, f"{method}.pkl")
    pd.to_pickle(results_dict, pkl_filepath)
    print(f"  Saved '{pkl_filepath}'")
