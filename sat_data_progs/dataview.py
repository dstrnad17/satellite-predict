#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:30:21 2024

@author: dunnchadnstrnad
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Load the DataFrame from the pickle file
data_directory = './sat_data_files'
save_directory = './satplots'
os.makedirs(save_directory, exist_ok=True)

#Makes plots for each file in the directory
for file_path in glob.glob(os.path.join(data_directory, '*.pkl')):

    dataframe = pd.read_pickle(file_path)
    print(f"Processing {file_path}")

    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.scatter(dataframe['epoch'], dataframe['x[km]'], marker='o', s=9)
    plt.title('x[km] vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('x[km]')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.scatter(dataframe['epoch'], dataframe['y[km]'], marker='o', s=9)
    plt.title('y[km] vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('y[km]')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.scatter(dataframe['epoch'], dataframe['z[km]'], marker='o', s=9)
    plt.title('z[km] vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('z[km]')
    plt.grid(True)

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    plt.savefig(os.path.join(save_directory, f"{file_name}_position.png"))

    plt.close

    print(f"Saved position plot of {file_name}")

    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.scatter(dataframe['epoch'], dataframe['bx[nT]'], marker='o', s=9)
    plt.title('bx[nT] vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('bx[nT]')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.scatter(dataframe['epoch'], dataframe['by[nT]'], marker='o', s=9)
    plt.title('by[nT] vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('by[nT]')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.scatter(dataframe['epoch'], dataframe['bz[nT]'], marker='o', s=9)
    plt.title('bz[nT] vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('bz[nT]')
    plt.grid(True)

    plt.savefig(os.path.join(save_directory, f"{file_name}_bvalues.png"))

    plt.close

    print(f"Saved magnetic field plot of {file_name}")

print('All plots created from directory')