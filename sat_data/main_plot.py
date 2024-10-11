import os
import pandas as pd
import matplotlib.pyplot as plt

# Define desired satellite file patterns
directory_list = ["goes8", "cluster1", "themise"]
# Directories for .pkl files and plots
base_directory = "./main_results/"
base_plot_directory = './main_plots/'
if not os.path.exists(base_plot_directory):
    os.makedirs(base_plot_directory)

# Prediction and actual columns
preds_nn3 = ['bx_nn3', 'by_nn3', 'bz_nn3']
preds_nn1 = ['bx_nn1', 'by_nn1', 'bz_nn1']
preds_lr = ['bx_lr', 'by_lr', 'bz_lr']
actual = ['bx_actual', 'by_actual', 'bz_actual']

def plot_error(rep_name, timestamp, model_errors, model_name, plot_directory):

    plt.style.use('default')
    plt.figure(figsize=(12, 8))

    # Plot for Bx
    plt.subplot(3, 1, 1)
    plt.plot(timestamp, model_errors['bx'], label=f'{model_name} Bx Error')
    plt.title(f"{model_name.upper()} Error for {rep_name}")
    plt.ylabel('Bx (nT)')
    plt.xticks([])

    # Plot for By
    plt.subplot(3, 1, 2)
    plt.plot(timestamp, model_errors['by'], label=f'{model_name} By Error')
    plt.ylabel('By (nT)')
    plt.xticks([])

    # Plot for Bz
    plt.subplot(3, 1, 3)
    plt.plot(timestamp, model_errors['bz'], label=f'{model_name} Bz Error')
    plt.xlabel('Date')
    plt.ylabel('Bz (nT)')

    plt.tight_layout()
    plot_filename = f"{rep_name}_{model_name}.png"
    plt.savefig(os.path.join(plot_directory, plot_filename))
    plt.close()
    print(f"{model_name.upper()} error plot saved to {plot_filename}")

def calc_errors(preds, actual, model_name):
    errors = actual.values - preds.values
    error = pd.DataFrame(errors, columns=[f'bx_{model_name}', f'by_{model_name}', f'bz_{model_name}'], index=actual_values.index)
    return error

# Loop over all .pkl files in the directory
for directory_name in directory_list:
    directory = os.path.join(base_directory, directory_name)
    # Create a subdirectory for each file pattern
    plot_directory = os.path.join(base_plot_directory, directory_name)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            print(f"Loading {filename}")
            # Load the .pkl file (which contains multiple DataFrames)
            file_path = os.path.join(directory, filename)
            data_dict = pd.read_pickle(file_path)  # Load the .pkl file

            # Loop through each key (repetition) in the pickle dictionary
            for key, df in data_dict.items():
                rep_name = f"{filename.split('.')[0]}_{key}"

                # Ensure the DataFrame is sorted by timestamp
                data_sorted = df.sort_values(by='timestamp')

                # Extract the timestamp, actual, and predicted values from the DataFrame
                timestamp = data_sorted['timestamp']
                actual_values = data_sorted[actual]

                # Calculate and plot for each model (nn1, nn3, lr)
                models = {
                    'nn3': preds_nn3,
                    'nn1': preds_nn1,
                    'lr': preds_lr
                }

                for model_name, model_preds in models.items():
                    # Extract the predicted values for the current model
                    preds_values = data_sorted[model_preds]

                    # Calculate errors for the current model
                    model_error = calc_errors(preds_values, actual_values, model_name)

                    # Prepare errors in a dictionary for plotting
                    model_errors = {
                        'bx': model_error['bx_' + model_name],
                        'by': model_error['by_' + model_name],
                        'bz': model_error['bz_' + model_name]
                    }

                    # Plot the errors for the current model
                    plot_error(rep_name, timestamp, model_errors, model_name, plot_directory)
