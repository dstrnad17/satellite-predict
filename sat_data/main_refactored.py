from satellite_predict.train_and_test import train_and_test

# This will be the main interface to the program. User provides kwargs
# and a program to return combined_dfs.
kwargs = {
    "config": {
      "n_df": 2, # Num. of DataFrames for data_load to return. None => all DataFrames.
      "data_directory": "./data",
      "file_pattern": "cluster1*.pkl"
    },
    "tag": "cluster1",
    "num_epochs": 1,
    "num_boot_reps": 2,
    "batch_size": 256,
    "lr": 0.0006,
    "removed_inputs": [None, "r[km]"],
    "models": ["ols", "nn1", "nn3"],
    "inputs": ["r[km]", "theta[deg]", "phi[deg]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"],
    "outputs": ["bx[nT]", "by[nT]", "bz[nT]"],
    "results_dir": None,

    "device": None,
    "parallel": False
}

#from satellite_predict.table import table
#table(**kwargs)
#exit()

def data_load(**config):
    import os
    import glob
    import numpy as np
    import pandas as pd

    def appendSpherical_np(xyz):
      xy = xyz[:, 0]**2 + xyz[:, 1]**2
      r = np.sqrt(xy + xyz[:, 2]**2)
      theta = np.arctan2(xyz[:, 2], np.sqrt(xy)) * 180 / np.pi
      phi = np.arctan2(xyz[:, 1], xyz[:, 0])
      phi = np.where(phi < 0, phi + 2 * np.pi, phi) * 180 / np.pi
      return np.column_stack((r, theta, phi))

    # Labels for Cartesian position columns
    position_cart = ["x[km]", "y[km]", "z[km]"]

    # Labels for derived columns used to convert from Cartesian to spherical
    position_sph = ["r[km]", "theta[deg]", "phi[deg]"]

    fglob = os.path.join(config['data_directory'], config['file_pattern'])
    files = glob.glob(fglob)

    dataframes = []
    n_r = 0 # Number of DataFrames read
    for f in files:
      df = pd.read_pickle(f)  # Load the DataFrame from pickle
      n_r = n_r + 1
      cartesian = df[position_cart].to_numpy()
      spherical = appendSpherical_np(cartesian)

      # Add spherical coordinates to the DataFrame
      for i, col in enumerate(position_sph):
        df[col] = spherical[:, i]

      ymdhms = df[['year', 'month', 'day', 'hour', 'minute', 'second']]
      df['datetime'] = pd.to_datetime(ymdhms)

      dataframes.append(df)
      if config['n_df'] is not None and n_r == config['n_df']:
        # Break if n_df (number of DataFrames to return) is specified
        break
    return dataframes

combined_dfs = data_load(**kwargs['config'])
train_and_test(combined_dfs, **kwargs)
