import pandas as pd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import datetime
import matplotlib.dates as mdates


#Define plot function
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
save_directory = './plot'
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
    dataframe['date'] = pd.to_datetime(dataframe[['year', 'month', 'day']])

    # Plot position data
    plot_data(
        dataframe, 
        ['x[km]', 'y[km]', 'z[km]'], 
        ['x (km)', 'y (km)', 'z (km)'], 
        os.path.join(save_directory, f"{file_name}_position.png"), 
        "position",
        suptitle = f"{file_name} - Position"
    )

    # Plot polar position data
    plot_data(
        dataframe, 
        ['r[km]', 'theta[rad]', 'phi[rad]'], 
        ['r (km)', 'φ (rad)', 'θ (rad)'], 
        os.path.join(save_directory, f"{file_name}_position_polar.png"), 
        "polar position",
        suptitle = f"{file_name} - Position, Polar"
    )

    # Plot magnetic field data
    plot_data(
        dataframe, 
        ['bx[nT]', 'by[nT]', 'bz[nT]'], 
        ['Bx (nT)', 'By (nT)', 'Bz (nT)'], 
        os.path.join(save_directory, f"{file_name}_bvalues.png"), 
        "magnetic field",
        suptitle = f"{file_name} - Magnetic Field"
    )

    # Plot polar magnetic field data
    plot_data(
        dataframe, 
        ['b[nT]', 'theta_b[rad]', 'phi_b[rad]'], 
        ['B (nT)', 'φ (rad)', 'θ (rad)'], 
        os.path.join(save_directory, f"{file_name}_bvalues_polar.png"), 
        "polar magnetic field",
        suptitle = f"{file_name} - Magnetic Field, Polar"
    )

print('All plots created from directory')