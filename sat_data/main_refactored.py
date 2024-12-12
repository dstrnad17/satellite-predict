from satellite_predict.train_and_test import train_and_test, table

# This will be the main interface to the program. User provides kwargs
# and a program to return combined_dfs.
kwargs = {
    "ny": 2,
    "num_epochs": 2,
    "num_boot_reps": 1,
    "batch_size": 256,
    "lr": 0.0006,
    "leave_outs": ["r[km]", "theta[deg]"],
    "models": ["nn1", "nn3", "lr"],
    "inputs": ["r[km]", "theta[deg]", "phi[deg]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"],
    "outputs": ["bx[nT]", "by[nT]", "bz[nT]"],
    "results_dir": None,

    "device": None,
    "parallel": False
}

# Function to load and preprocess data
def data_load(data_directory, file_pattern):
    import os
    import glob
    import numpy as np
    import pandas as pd
    # Function to convert Cartesian to spherical coordinates
    def appendSpherical_np(xyz):
        xy = xyz[:, 0]**2 + xyz[:, 1]**2
        r = np.sqrt(xy + xyz[:, 2]**2)
        theta = np.arctan2(xyz[:, 2], np.sqrt(xy)) * 180 / np.pi
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi = np.where(phi < 0, phi + 2 * np.pi, phi) * 180 / np.pi
        return np.column_stack((r, theta, phi))

    # Labels for derived columns used to convert from Cartesian to spherical
    position_sph = ["r[km]", "theta[deg]", "phi[deg]"]
    position_cart = ["x[km]", "y[km]", "z[km]"]

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
        break
    return dataframes


tag = "cluster1"
combined_dfs = data_load("./data/", f"{tag}*.pkl")
print(len(combined_dfs))
print(combined_dfs[0].head())
train_and_test(combined_dfs, tag, **kwargs)

#table(tag, **kwargs)