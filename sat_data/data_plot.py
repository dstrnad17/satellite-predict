import os
import glob
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# column_info function to extract name, symbol, and unit from column names
def column_info(column_name):
    name = column_name.split('[')[0]
    unit = column_name.split('[')[1].split(']')[1]
    symbols = {'x': 'x', 'y': 'y', 'z': 'z', 'bx': 'B_x', 'by': 'B_y', 'bz': 'B_z'}
    symbol = symbols.get(name, name)  # Default to name if not in the symbol dictionary
    return name, symbol, unit

# Define plot function
def plot_data(dataframe, columns, labels, save_path, title, suptitle):
    plt.figure(figsize=(12, 12), facecolor='white')
    plt.suptitle(suptitle, fontsize=16)
    for i, (col, label) in enumerate(zip(columns, labels)):
        plt.subplot(3, 1, i + 1)
        plt.plot(dataframe['date'], dataframe[col], linestyle='-')
        plt.ylabel(label)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xlabel('Date')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {title} plot to {save_path}")

# Directory paths
data_directory = './data'
save_directory = './data_plot'
os.makedirs(save_directory, exist_ok=True)

# Process each pickle file in the directory
for file_path in glob.glob(os.path.join(data_directory, '*.pkl')):
    dataframe = pd.read_pickle(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing {file_name}")

    # Convert to spherical coordinates
    dataframe['r[km]'] = np.sqrt(dataframe['x[km]']**2 + dataframe['y[km]']**2 + dataframe['z[km]']**2)
    dataframe['theta[rad]'] = np.arctan2(dataframe['y[km]'], dataframe['x[km]'])
    dataframe['phi[rad]'] = np.arccos(dataframe['z[km]'] / dataframe['r[km]'])
    dataframe['b[nT]'] = np.sqrt(dataframe['bx[nT]']**2 + dataframe['by[nT]']**2 + dataframe['bz[nT]']**2)
    dataframe['theta_b[rad]'] = np.arctan2(dataframe['by[nT]'], dataframe['bx[nT]'])
    dataframe['phi_b[rad]'] = np.arccos(dataframe['bz[nT]'] / dataframe['b[nT]'])
    print(f"Added polar coordinates for {file_name}")

    # Add datetime column to dataframe
    dataframe['date'] = pd.to_datetime(dataframe[['year', 'month', 'day', 'hour', 'minute', 'second']])

    # Define plot groups
    plot_groups = [
        (['x[km]', 'y[km]', 'z[km]'], ['x (km)', 'y (km)', 'z (km)'], f"{file_name}_position.png", "Position", f"{file_name} - Position"),
        (['r[km]', 'theta[rad]', 'phi[rad]'], ['r (km)', 'φ (rad)', 'θ (rad)'], f"{file_name}_position_polar.png", "Polar Position", f"{file_name} - Position, Polar"),
        (['bx[nT]', 'by[nT]', 'bz[nT]'], ['Bx (nT)', 'By (nT)', 'Bz (nT)'], f"{file_name}_bvalues.png", "Magnetic Field", f"{file_name} - Magnetic Field"),
        (['b[nT]', 'theta_b[rad]', 'phi_b[rad]'], ['B (nT)', 'φ (rad)', 'θ (rad)'], f"{file_name}_bvalues_polar.png", "Polar Magnetic Field", f"{file_name} - Magnetic Field, Polar")
    ]

    # Generate and save plots
    for columns, labels, save_filename, title, suptitle in plot_groups:
        save_path = os.path.join(save_directory, save_filename)
        plot_data(dataframe, columns, labels, save_path, title, suptitle)

print('All plots created from directory')