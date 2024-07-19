
import os
import requests

# URL of the ZIP file
api_url = 'http://spdf.gsfc.nasa.gov/pub/data/aaa_special-purpose-datasets/empirical-magnetic-field-modeling-database-with-TS07D-coefficients/database/ascii/'

base_url = 'http://spdf.gsfc.nasa.gov/pub/data/aaa_special-purpose-datasets/empirical-magnetic-field-modeling-database-with-TS07D-coefficients/database/ascii/'

save_dir = '/Users/dunnchadnstrnad/Documents/GitHub/2024phys798'


os.makedirs(save_dir, exist_ok=True)

# Step 1: Get the list of files from the API
response = requests.get(api_url)
if response.status_code == 200:
    file_list = response.json()  # Assuming the API returns a JSON array of filenames

    for file_name in file_list:
        file_url = base_url + file_name
        # Step 2: Download each file
        file_response = requests.get(file_url)
        if file_response.status_code == 200:
            save_path = os.path.join(save_dir, file_name)
            # Step 3: Save each file
            with open(save_path, 'wb') as file:
                file.write(file_response.content)
            print(f"Downloaded {save_path}")
        else:
            print(f"Failed to download {file_url}. Status code: {file_response.status_code}")
else:
    print(f"Failed to retrieve the directory listing from the API. Status code: {response.status_code}")