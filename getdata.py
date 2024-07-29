import os
import requests
from bs4 import BeautifulSoup
import numpy as np  # or import pandas as pd

# URL of the directory
url = "https://spdf.gsfc.nasa.gov/pub/data/aaa_special-purpose-datasets/empirical-magnetic-field-modeling-database-with-TS07D-coefficients/database/ascii/"

# Function to download a file
def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses
    file_name = os.path.join(dest_folder, url.split("/")[-1])
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {file_name}")
    return file_name

# Function to load data from a .dat file
def load_dat_file(filepath):
    data = np.loadtxt(filepath)  # or use pandas: pd.read_csv(filepath, delim_whitespace=True, header=None)
    return data

# Get list of files
response = requests.get(url)
response.raise_for_status()  # Ensure we notice bad responses
soup = BeautifulSoup(response.content, "html.parser")
links = soup.find_all("a")

# Filter out links for the files
files = [link.get("href") for link in links if link.get("href").endswith((".txt", ".csv", ".dat"))]

# Download and load each file
for file in files:
    file_url = url + file
    downloaded_file_path = download_file(file_url, "downloaded_files")
    if downloaded_file_path.endswith(".dat"):
        data_array = load_dat_file(downloaded_file_path)
        print(f"Data from {downloaded_file_path}:")
        print(data_array)

print("All files downloaded and data loaded")