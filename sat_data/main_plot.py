import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specify the base directory containing the results
base_directory = "./main_results"
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# List desired satellites or patterns
satellite_list = ["goes8", "cluster1", "themise"]  # Replace with actual sub-directory names

# List of columns to calculate errors
preds_nn3 = ['bx_nn3', 'by_nn3', 'bz_nn3']
preds_lr = ['bx_lr', 'by_lr', 'bz_lr']
preds_nn1 = ['bx_nn1', 'by_nn1', 'bz_nn1']
actual = ['bx_actual', 'by_actual', 'bz_actual']

# Iterate over each pattern (e.g., goes8, cluster1)
for directory_name in satellite_list:
    directory = os.path.join(base_directory, directory_name)
    data_input = {}  # Store data for each removed_input

    # Iterate through subdirectories under each pattern (e.g., 'none', 'bx', 'by', etc.)
    for removed_input_dir in os.listdir(directory):
        removed_input_dir_path = os.path.join(directory, removed_input_dir)
        if os.path.isdir(removed_input_dir_path):
            print(f"Processing removed input: {removed_input_dir}")  # Debug print

            # Initialize data for this removed input
            data_input[removed_input_dir] = []

            plots_dir = os.path.join(removed_input_dir_path, "error_plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                print(f"Created new directory for plots: {plots_dir}")

            # Iterate through each .pkl file in the removed_input subdirectory
            for file_name in os.listdir(removed_input_dir_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(removed_input_dir_path, file_name)

                    # Extract the removed input from the directory name
                    removed_input = removed_input_dir

                    print(f"Loading file: {file_path}")  # Debug print

                    # Load the .pkl file
                    try:
                        df_dict = pd.read_pickle(file_path)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue

                    rep_dataframes = []

                    # Iterate through each DataFrame in the .pkl file
                    for key, df in df_dict.items():
                        rep_dataframes.append(df)

                    # Concatenate each rep df
                    concat_df_rep = pd.concat(rep_dataframes, ignore_index=True)

                    # Append concatenated dataframe to data_input
                    data_input[removed_input].append(concat_df_rep)

            # After all files for this removed_input are processed, concatenate the data
            if removed_input in data_input:
                total_data = pd.concat(data_input[removed_input], ignore_index=True)

                # Ensure 'timestamp' column is in timestamp format, assuming the column name is 'timestamp'
                if 'timestamp' in total_data.columns:
                    total_data['timestamp'] = pd.to_datetime(total_data['timestamp'])
                    total_data = total_data.sort_values(by='timestamp')  # Sort data by timestamp
                else:
                    print(f"Warning: 'timestamp' column not found in {removed_input} data.")

                # Calculate errors (preds - actual) for each model type
                total_data['error_bx_nn3'] = total_data['bx_nn3'] - total_data['bx_actual']
                total_data['error_by_nn3'] = total_data['by_nn3'] - total_data['by_actual']
                total_data['error_bz_nn3'] = total_data['bz_nn3'] - total_data['bz_actual']

                total_data['error_bx_nn1'] = total_data['bx_nn1'] - total_data['bx_actual']
                total_data['error_by_nn1'] = total_data['by_nn1'] - total_data['by_actual']
                total_data['error_bz_nn1'] = total_data['bz_nn1'] - total_data['bz_actual']

                total_data['error_bx_lr'] = total_data['bx_lr'] - total_data['bx_actual']
                total_data['error_by_lr'] = total_data['by_lr'] - total_data['by_actual']
                total_data['error_bz_lr'] = total_data['bz_lr'] - total_data['bz_actual']

                # Plot time-series errors for each model type
                for model_type in ['nn3', 'nn1', 'lr']:
                    plt.figure(figsize=(12, 12))

                    # Plot errors over time for bx, by, bz
                    plt.subplot(3, 1, 1)
                    plt.plot(total_data['timestamp'], total_data[f'error_bx_{model_type}'], label=f'{model_type} bx')
                    plt.title(f'Time-Series Error for {model_type}')
                    plt.ylabel('Bx')
                    plt.xticks([])

                    plt.subplot(3, 1, 2)
                    plt.plot(total_data['timestamp'], total_data[f'error_by_{model_type}'], label=f'{model_type} by')
                    plt.ylabel('By')
                    plt.xticks([])

                    plt.subplot(3, 1, 3)
                    plt.plot(total_data['timestamp'], total_data[f'error_bz_{model_type}'], label=f'{model_type} bz')
                    plt.xlabel('Time')
                    plt.ylabel('Bz')

                    # Calculate MSE for the model type
                    mse_bx = np.mean((total_data[f'bx_actual'] - total_data[f'bx_{model_type}']) ** 2)
                    mse_by = np.mean((total_data[f'by_actual'] - total_data[f'by_{model_type}']) ** 2)
                    mse_bz = np.mean((total_data[f'bz_actual'] - total_data[f'bz_{model_type}']) ** 2)

                    # Add the MSE to the plot as a text box
                    mse_text = f'MSE bx: {mse_bx:.4f}\nMSE by: {mse_by:.4f}\nMSE bz: {mse_bz:.4f}'
                    plt.gcf().text(0.85, 0.95, mse_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.5'))
                    plt.tight_layout()
                    
                    # Save the figure with a white background in the new subdirectory
                    plt.savefig(os.path.join(plots_dir, f"{directory_name}_{removed_input}_{model_type}_time_error.png"), transparent=False)
                    plt.close()
                    print(f"Saved time-series error plot for {model_type} in {removed_input} to {directory}")

    print(f"Finished processing for {directory_name}")