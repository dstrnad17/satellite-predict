import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specify the base directory containing the results
base_directory = "./main_results"
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

all = False

if all:
    satellite_list = ["cluster1", "goes8", "themise"]
else:
    satellite_list = ["cluster1"]

# List of columns to calculate errors
preds = {
    'nn3': ['bx_nn3', 'by_nn3', 'bz_nn3'],
    'nn1': ['bx_nn1', 'by_nn1', 'bz_nn1'],
    'lr': ['bx_lr', 'by_lr', 'bz_lr']
}
actual = ['bx_actual', 'by_actual', 'bz_actual']

# Function to compute average relative variance (ARV), this function appears in each program
def compute_arv(A, P):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(P, pd.DataFrame):
        P = P.to_numpy()

    if A.ndim == 1:
        A, P = np.expand_dims(A, axis=1), np.expand_dims(P, axis=1)

    arvs = []
    for i in range(A.shape[1]):
        var_A = np.var(A[:, i])
        arv = np.var(A[:, i] - P[:, i]) / var_A if var_A > 0 else 0
        arvs.append(arv)
    
    return arvs if len(arvs) > 1 else arvs[0]

def plot_time_series_errors(total_data, model_type, removed_input, arv, plots_dir):
    # Generalized plotting function for time-series errors
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for i, (component, axis) in enumerate(zip(['bx', 'by', 'bz'], axes)):
        error_column = f'error_{component}_{model_type}'
        axis.plot(total_data['timestamp'], total_data[error_column], label=f'{model_type} {component}')
        axis.set_ylabel(component.capitalize())
        axis.set_xticks([] if i < 2 else axis.get_xticks())
        if i == 0:
            axis.set_title(f'{model_type} Error for {removed_input} Removed')
        axis.text(0.01, 0.95, f'arv: {arv[i]:.4f}', transform=axis.transAxes, fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

    axes[2].set_xlabel('Time')
    plt.savefig(os.path.join(plots_dir, f"{removed_input}_{model_type}_time_error.png"), transparent=False)
    plt.close()

def plot_time_series_predictions(total_data, model_type, removed_input, arv, plots_dir):
    # Generalized plotting function for time-series predictions vs actuals
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for i, (component, axis) in enumerate(zip(['bx', 'by', 'bz'], axes)):
        actual_column = f'{component}_actual'
        predicted_column = f'{component}_{model_type}'
        axis.plot(total_data['timestamp'], total_data[actual_column], label='Actual', color='black', linestyle='--')
        axis.plot(total_data['timestamp'], total_data[predicted_column], label=f'Predicted', color='red', alpha=0.7)
        axis.set_ylabel(component.capitalize())
        axis.set_xticks([] if i < 2 else axis.get_xticks())
        if i == 0:
            axis.set_title(f'{model_type} for Removed {removed_input}')
            axis.legend(loc='upper right')
        axis.text(0.01, 0.95, f'arv: {arv[i]:.4f}', transform=axis.transAxes, fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

    axes[2].set_xlabel('Time')
    plt.savefig(os.path.join(plots_dir, f"{removed_input}_{model_type}_time_predictions.png"), transparent=False)
    plt.close()

# Iterate over each pattern (e.g., goes8, cluster1)
for directory_name in satellite_list:
    directory = os.path.join(base_directory, directory_name)
    data_input = {}

    # Iterate through subdirectories under each pattern (e.g., 'none', 'bx', 'by', etc.)
    for removed_input_dir in os.listdir(directory):
        removed_input_dir_path = os.path.join(directory, removed_input_dir)
        if os.path.isdir(removed_input_dir_path):
            print(f"Processing removed input: {removed_input_dir}")

            data_input[removed_input_dir] = []

            plots_dir = os.path.join(removed_input_dir_path, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)

            # Iterate through each .pkl file in the removed_input subdirectory
            for file_name in os.listdir(removed_input_dir_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(removed_input_dir_path, file_name)

                    removed_input = removed_input_dir

                    print(f"Loading file: {file_path}")

                    try:
                        df_dict = pd.read_pickle(file_path)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue

                    rep_dataframes = []
                    for key, df in df_dict.items():
                        rep_dataframes.append(df)

                    concat_df_rep = pd.concat(rep_dataframes, ignore_index=True)
                    data_input[removed_input].append(concat_df_rep)

            if removed_input in data_input:
                total_data = pd.concat(data_input[removed_input], ignore_index=True)

                if 'timestamp' in total_data.columns:
                    total_data['timestamp'] = pd.to_datetime(total_data['timestamp'])
                    total_data = total_data.sort_values(by='timestamp')
                else:
                    print(f"Warning: 'timestamp' column not found in {removed_input} data.")

                # Calculate errors (preds - actual) for each model type
                for model_type, model_preds in preds.items():
                    for i, component in enumerate(['bx', 'by', 'bz']):
                        total_data[f'error_{component}_{model_type}'] = total_data[model_preds[i]] - total_data[f'{component}_actual']

                    arv = compute_arv(total_data[actual], total_data[model_preds])

                    # Plot time-series errors and predictions
                    plot_time_series_errors(total_data, model_type, removed_input, arv, plots_dir)
                    plot_time_series_predictions(total_data, model_type, removed_input, arv, plots_dir)

    print(f"Finished processing for {directory_name}")