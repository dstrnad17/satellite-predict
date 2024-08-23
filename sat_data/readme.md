sat_data basic programs

data.py downloads data files from the NASA repository, creates dataframes, and saves the dataframe as .pkl files
A new folder is created called "data" containing all of these files

plot.py creates plots from the .pkl files created by data.py
Dataframes are read and plotted for both magnetic field and position data in cartesian and polar coordinates vs. time