import numpy as np
import pandas as pd

def load_and_extract_columns(filepath, columns):
    # Load the data from the .dat file
    try:
        # Attempt to read the file with whitespace as delimiter
        data = pd.read_csv(filepath, delim_whitespace=True, header=None)
    except pd.errors.ParserError:
        print("Error reading the file with delim_whitespace=True. Trying manual split...")
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # Split the lines manually
        data = pd.DataFrame([line.split() for line in lines])
    
    # Debug
    print("DataFrame shape:", data.shape)
    print("First few rows:\n", data.head())
    
    # Ensure we have the right number of columns
    if any(i >= data.shape[1] for i in columns):
        raise IndexError("One or more column indices are out of bounds.")
    
    # Extract columns and convert to numeric
    extracted_columns = []
    for i in columns:
        # Convert column to numeric, coerce errors to NaN
        column = pd.to_numeric(data.iloc[:, i], errors='coerce')
        extracted_columns.append(column)
    
    return extracted_columns

# Example usage
file_path = "/Users/dunnchadnstrnad/Documents/GitHub/2024phys798/downloaded_files/cluster1_2001_avg_300_omni.dat"
columns_to_extract = [10, 11, 12, 13]  # Replace with the actual column indices you want to extract
array_names = ['x', 'y', 'z', 'bx']  # Desired variable names

# Load and extract columns
extracted_columns = load_and_extract_columns(file_path, columns_to_extract)

# Assign arrays to variables dynamically
for name, column in zip(array_names, extracted_columns):
    globals()[name] = np.array(column.dropna(), dtype=float)
    print(f"Variable {name} created with data:")
    print(globals()[name])


