import os
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch

# Define desired satellite file patterns
file_patterns = ["goes8_*.pkl", "cluster1_*.pkl", "themise_*.pkl"]

# Directories for .pkl files and plots
pkl_directory = "./main_results/results_pkl/"
base_plot_directory = './main_plots/'

# Create the base plot directory if it doesn't exist
if not os.path.exists(base_plot_directory):
    os.makedirs(base_plot_directory)

def plot_error(rep_name, timestamp, nn_error_bx, nn_error_by, nn_error_bz, lr_error_bx, lr_error_by, lr_error_bz, plot_directory):
    # Plot NN Error
    plt.style.use('default')
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(timestamp, nn_error_bx)
    plt.title(f"NN Error for {rep_name}")
    plt.ylabel('Bx (nT)')
    plt.xticks([])

    plt.subplot(3, 1, 2)
    plt.plot(timestamp, nn_error_by)
    plt.ylabel('By (nT)')
    plt.xticks([])

    plt.subplot(3, 1, 3)
    plt.plot(timestamp, nn_error_bz)
    plt.xlabel('Date')
    plt.ylabel('Bz (nT)')

    plt.tight_layout()
    plot_filename = f"{rep_name}_nn.png"
    plt.savefig(os.path.join(plot_directory, plot_filename))
    plt.close()
    print(f"NN error plot saved to {plot_filename}")

    # Plot LR Error
    plt.style.use('default')
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(timestamp, lr_error_bx)
    plt.title(f'LR Error for {rep_name}')
    plt.ylabel('Bx (nT)')
    plt.xticks([])

    plt.subplot(3, 1, 2)
    plt.plot(timestamp, lr_error_by)
    plt.ylabel('By (nT)')
    plt.xticks([])

    plt.subplot(3, 1, 3)
    plt.plot(timestamp, lr_error_bz)
    plt.xlabel('Date')
    plt.ylabel('Bz (nT)')

    plt.tight_layout()
    plot_filename = f"{rep_name}_lr.png"
    plt.savefig(os.path.join(plot_directory, plot_filename))
    plt.close()
    print(f"LR error plot saved to {plot_filename}")

# Loop over all .pkl files in the directory
for pattern in file_patterns:
    # Create a subdirectory for each file pattern
    pattern_name = pattern.split('_')[0]
    plot_directory = os.path.join(base_plot_directory, pattern_name)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    for filename in os.listdir(pkl_directory):
        if fnmatch.fnmatch(filename, pattern):
            print(f"Loading {filename}")
            # Load the .pkl file (which contains multiple DataFrames)
            file_path = os.path.join(pkl_directory, filename)
            with pd.HDFStore(file_path, 'r') as store:
                # Get the list of DataFrame keys (repetitions) stored in the .pkl file
                data_keys = store.keys()

                # Loop through each DataFrame (repetition) in the file
                for key in data_keys:
                    rep_name = f"{filename.split('.')[0]}_{key.strip('/')}"
                    df = store[key]

                    # Ensure the DataFrame is sorted by timestamp
                    data_sorted = df.sort_values(by='timestamp')

                    # Extract timestamp and errors
                    timestamp = data_sorted['timestamp']
                    nn_error_bx = data_sorted['nn_error_bx']
                    nn_error_by = data_sorted['nn_error_by']
                    nn_error_bz = data_sorted['nn_error_bz']
                    lr_error_bx = data_sorted['lr_error_bx']
                    lr_error_by = data_sorted['lr_error_by']
                    lr_error_bz = data_sorted['lr_error_bz']

                    # Call the plot function with relevant data
                    plot_error(rep_name, timestamp, nn_error_bx, nn_error_by, nn_error_bz, lr_error_bx, lr_error_by, lr_error_bz, plot_directory)