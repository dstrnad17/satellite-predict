import os
import requests
from bs4 import BeautifulSoup

all = True

# URL of the directory
url = "https://spdf.gsfc.nasa.gov/pub/data/aaa_special-purpose-datasets/empirical-magnetic-field-modeling-database-with-TS07D-coefficients/database/ascii/"

# Function to download a file
def download_file(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses
    file_name = os.path.join(dest_folder, url.split("/")[-1])
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {file_name}")

# Get list of files
response = requests.get(url)
response.raise_for_status()  # Ensure we notice bad responses
soup = BeautifulSoup(response.content, "html.parser")
links = soup.find_all("a")

# Filter out links for the files
files = [link.get("href") for link in links if link.get("href").endswith((".txt", ".csv", ".dat"))]

# Download each file
for file in files:
    file_url = url + file
    download_file(file_url, "sat_data_files")
    if all == False: 
        print("File download complete, all=False")
        break

print("Downloading complete")
