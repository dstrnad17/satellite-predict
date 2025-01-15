import os
import pandas as pd
import numpy as np

base_directory = "./main_results/"
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

all = False

# List desired satellites or patterns
if all:
    satellite_list = ["cluster1", "goes8", "themise"] 
else:
    satellite_list = ["cluster1"] 

# List of columns to calculate mean, std dev, and RMSE
preds = {
    'nn3': ['bx[nT]_nn3', 'by[nT]_nn3', 'bz[nT]_nn3'],
    'nn1': ['bx[nT]_nn1', 'by[nT]_nn1', 'bz[nT]_nn1'],
    'lr': ['bx[nT]_lr', 'by[nT]_lr', 'bz[nT]_lr']
}
actual = ['bx[nT]_actual', 'by[nT]_actual', 'bz[nT]_actual']

# Function to compute average relative variance (ARV), this function appears in each program
def compute_arv(A, P):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(P, pd.DataFrame):
        P = P.to_numpy()

    if A.ndim == 1:
        A, P = np.expand_dims(A, axis=1), np.expand_dims(P, axis=1)

    arvs = []
    for i in range(A.shape[1]):
        var_A = np.var(A[:, i])
        arv = np.var(A[:, i] - P[:, i]) / var_A if var_A > 0 else 0
        arvs.append(arv)
    
    return arvs if len(arvs) > 1 else arvs[0]

# Iterate over each pattern (e.g., goes8, cluster1)
for directory_name in satellite_list:
    directory = os.path.join(base_directory, directory_name)
    summary_data = []
    data_input = {}

    # Iterate through subdirectories under each pattern
    for removed_input_dir in os.listdir(directory):
        removed_input_dir_path = os.path.join(directory, removed_input_dir)
        if os.path.isdir(removed_input_dir_path):
            print(f"Processing removed input: {removed_input_dir}")  # Debug print

            data_input[removed_input_dir] = []

            # Iterate through each .pkl file in the removed_input subdirectory
            for file_name in os.listdir(removed_input_dir_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(removed_input_dir_path, file_name)

                    # Extract the removed input from the directory name
                    removed_input = removed_input_dir

                    print(f"Loading file: {file_path}")

                    # Load the .pkl file
                    try:
                        df_dict = pd.read_pickle(file_path)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue

                    rep_dataframes = []

                    # Iterate through each DataFrame in the .pkl file
                    for key, df in df_dict.items():
                        rep_dataframes.append(df)

                    # Concatenate each rep df
                    concat_df_rep = pd.concat(rep_dataframes, ignore_index=True)
                    data_input[removed_input].append(concat_df_rep)

            # After all files for this removed_input are processed, concatenate the data
            if removed_input in data_input:
                total_data = pd.concat(data_input[removed_input], ignore_index=True)
                print("Columns in total_data:", total_data.columns)

                # Calculate the mean and std deviation after concatenation (averaging over reps)
                avg_df = total_data.mean(axis=0)
                std_df = total_data.std(axis=0)

                for model_type, model_preds in preds.items():
                    means = avg_df[model_preds]
                    stds = std_df[model_preds]
                    arv = compute_arv(total_data[actual], total_data[model_preds])

                    result = {'Removed Input': removed_input, 'Model': model_type}
                    
                    components = set(col.split('_')[0] for col in model_preds)  # Extract 'bx', 'by', 'bz'

                    # Add mean, std, and ARV for each component
                    for component in components:
                        result[f'Mean_{component}'] = means[means.index.str.startswith(component)].values[0]
                        result[f'Std_{component}'] = stds[stds.index.str.startswith(component)].values[0]
                        result[f'ARV_{component}'] = arv[list(components).index(component)]

                    summary_data.append(result)

    # Combine all the summary data into one table
    summary = pd.DataFrame(summary_data)
    summary = summary.drop_duplicates()

    # Save summary as markdown file
    summary.to_markdown(os.path.join(directory, f"{directory_name}_result.md"), index=False)
    print(f"Summary Results saved to {directory}/{directory_name}_result.md")