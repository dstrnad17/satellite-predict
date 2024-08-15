#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:30:21 2024

@author: dunnchadnstrnad
"""
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Load the DataFrame from the pickle file
main_dataframe = pd.read_pickle('./Satellite_data/sat_dataframe.pkl')

print(main_dataframe)

if False:
    # Assuming the columns represent X, Y, Z, and a fourth dimension (W)
    x = main_dataframe.iloc[:, 10]
    y = main_dataframe.iloc[:, 11]
    z = main_dataframe.iloc[:, 12]
    bx = main_dataframe.iloc[:, 13]


    fig = px.scatter_3d(main_dataframe, x='x[km]', y='y[km]', z='z[km]', color='bx[nT]', color_continuous_scale='Viridis')

    fig.show()