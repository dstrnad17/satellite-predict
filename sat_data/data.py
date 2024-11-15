import os
import logging
import requests

from os import listdir
from os.path import isfile, join, dirname, abspath

import pandas as pd
from bs4 import BeautifulSoup

## Download data ##

all = False

# URL of the directory
url = "https://spdf.gsfc.nasa.gov/pub/data/aaa_special-purpose-datasets/empirical-magnetic-field-modeling-database-with-TS07D-coefficients/database/ascii/"

# Function to download a file
def download_file(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    print(f"Getting {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses
    file_name = os.path.join(dest_folder, url.split("/")[-1])
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Saved: {file_name}")

# Get list of files
response = requests.get(url)
response.raise_for_status()  # Ensure we notice bad responses
soup = BeautifulSoup(response.content, "html.parser")
links = soup.find_all("a")

# Filter out links for the files
files = [link.get("href") for link in links if link.get("href").endswith((".txt", ".csv", ".dat"))]

# Download each file
for file in files:
    if not all and not file.startswith("cluster1"):
        # Only d/l cluster1 files
        continue
    file_url = url + file
    download_file(file_url, "data")
    if not all:
        print("File download complete, all=False")
        break

print("Downloading complete")

## Create pickle files ##

logging.basicConfig(filename='data_pickle.log', encoding='utf-8', level=logging.DEBUG)

# Create empty log file
logfile = os.path.realpath(__file__)[0:-2] + "log"
with open(logfile, "w") as f:
    pass

def xprint(msg):
    print(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")

# Use script's directory
script_dir = dirname(abspath(__file__))
mypath = join(script_dir, "data/")

# Ensure directory exists
if not os.path.exists(mypath):
    os.makedirs(mypath)

# List all files in directory
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".dat")]

xprint(f"Saving DataFrames for {len(files)} file(s)")
# Extract column names from first file
names = pd.DataFrame(pd.read_csv(mypath + files[0], nrows=0, delimiter=',')).columns

# Process each file in directory
for i, file in enumerate(files):
    file = mypath + file
    xprint(f"  Processing file {i+1}/{len(files)}: {file}")
    
    # Read the data, skip first row, whitespace as delimiter
    data = pd.read_csv(file, skiprows=1, header=None, delimiter=r'\s+')
    
    # Convert to DataFrame, assign column names
    df = pd.DataFrame(data)
    df.columns = names
    
    # Save DataFrame as pickle file
    pkl_filename = file.replace(".dat", ".pkl")
    df.to_pickle(join(mypath, pkl_filename))
    
    xprint(f"  Saved DataFrame to {pkl_filename}")

xprint("All files pickled and saved")