import pandas as pd
import logging, os
from os import listdir
from os.path import isfile, join, dirname, abspath

logging.basicConfig(filename='sat_pickle.log', encoding='utf-8', level=logging.DEBUG)

# Create empty log file
logfile = os.path.realpath(__file__)[0:-2] + "log"
with open(logfile, "w") as f:
    pass

def xprint(msg):
    print(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")

# Use the script's directory
script_dir = dirname(abspath(__file__))
mypath = join(script_dir, "sat_data_files/")

# Ensure the directory exists
if not os.path.exists(mypath):
    os.makedirs(mypath)

# List all files in the directory
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Extract column names from the first file
names = pd.DataFrame(pd.read_csv(mypath + files[0], nrows=0, delimiter=',')).columns

# Process each file in the directory
for i, file in enumerate(files):
    xprint(f"Processing file {i+1}/{len(files)}: {file}")
    
    # Read the data skipping the first row and using whitespace as the delimiter
    data = pd.read_csv(mypath + file, skiprows=1, header=None, delimiter='\s+')
    
    # Convert to DataFrame and assign column names
    df = pd.DataFrame(data)
    df.columns = names
    
    # Save the DataFrame as a pickle file
    pkl_filename = file.replace(".dat", ".pkl")
    df.to_pickle(join(mypath, pkl_filename))
    
    xprint(f"Saved DataFrame to {pkl_filename}")

xprint("All files pickled successfully")