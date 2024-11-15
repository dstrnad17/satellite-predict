import os
import pandas as pd
import numpy as np

# Specify the base directory containing the results
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
preds_nn3 = ['bx_nn3', 'by_nn3', 'bz_nn3']
preds_lr = ['bx_lr', 'by_lr', 'bz_lr']
preds_nn1 = ['bx_nn1', 'by_nn1', 'bz_nn1']
actual = ['bx_actual', 'by_actual', 'bz_actual']

# Iterate over each pattern (e.g., goes8, cluster1)
for directory_name in satellite_list:
    directory = os.path.join(base_directory, directory_name)
    summary_data = []
    data_input = {}

    # Iterate through subdirectories under each pattern (e.g., 'none', 'bx', 'by', etc.)
    for removed_input_dir in os.listdir(directory):
        removed_input_dir_path = os.path.join(directory, removed_input_dir)
        if os.path.isdir(removed_input_dir_path):
            print(f"Processing removed input: {removed_input_dir}")  # Debug print

            # Initialize data for this removed input
            data_input[removed_input_dir] = []

            # Iterate through each .pkl file in the removed_input subdirectory
            for file_name in os.listdir(removed_input_dir_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(removed_input_dir_path, file_name)

                    # Extract the removed input from the directory name
                    removed_input = removed_input_dir

                    print(f"Loading file: {file_path}")  # Debug print

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

                    # Append concatenated dataframe to data_input
                    data_input[removed_input].append(concat_df_rep)

            # After all files for this removed_input are processed, concatenate the data
            if removed_input in data_input:
                total_data = pd.concat(data_input[removed_input], ignore_index=True)

                # Calculate the mean and std deviation after concatenation (averaging over reps)
                avg_df = total_data.mean(axis=0)
                std_df = total_data.std(axis=0)

                # Calculate mean, std, rmse after averaging the reps
                means_nn3 = avg_df[preds_nn3]
                stds_nn3 = std_df[preds_nn3]

                means_nn1 = avg_df[preds_nn1]
                stds_nn1 = std_df[preds_nn1]

                means_lr = avg_df[preds_lr]
                stds_lr = std_df[preds_lr]

                # Calculate RMSE for nn3
                rmse_nn3 = [
                    np.sqrt(np.mean(((total_data['bx_actual'] - total_data['bx_nn3']) ** 2))),
                    np.sqrt(np.mean(((total_data['by_actual'] - total_data['by_nn3']) ** 2))),
                    np.sqrt(np.mean(((total_data['bz_actual'] - total_data['bz_nn3']) ** 2)))
                ]

                # Calculate RMSE for nn1
                rmse_nn1 = [
                    np.sqrt(np.mean(((total_data['bx_actual'] - total_data['bx_nn1']) ** 2))),
                    np.sqrt(np.mean(((total_data['by_actual'] - total_data['by_nn1']) ** 2))),
                    np.sqrt(np.mean(((total_data['bz_actual'] - total_data['bz_nn1']) ** 2)))
                ]

                # Calculate RMSE for lr
                rmse_lr = [
                    np.sqrt(np.mean(((total_data['bx_actual'] - total_data['bx_lr']) ** 2))),
                    np.sqrt(np.mean(((total_data['by_actual'] - total_data['by_lr']) ** 2))),
                    np.sqrt(np.mean(((total_data['bz_actual'] - total_data['bz_lr']) ** 2)))
                ]

                # Create a combined result for each model type
                for model_type, means, stds, rmse in zip(
                    ['nn3', 'nn1', 'lr'],
                    [means_nn3, means_nn1, means_lr],
                    [stds_nn3, stds_nn1, stds_lr],
                    [rmse_nn3, rmse_nn1, rmse_lr]
                ):
                    result = {
                        'Removed Input': removed_input,
                        'Model': model_type,
                        'Mean_bx': means['bx_' + model_type],
                        'Std_bx': stds['bx_' + model_type],
                        'RMSE_bx': rmse[0],
                        'Mean_by': means['by_' + model_type],
                        'Std_by': stds['by_' + model_type],
                        'RMSE_by': rmse[1],
                        'Mean_bz': means['bz_' + model_type],
                        'Std_bz': stds['bz_' + model_type],
                        'RMSE_bz': rmse[2],
                    }

                    # Append the result to the summary_data list
                    summary_data.append(result)

    # Combine all the summary data into one table
    summary = pd.DataFrame(summary_data)
    
    # Remove duplicates if any
    summary = summary.drop_duplicates()

    # Save summary as markdown file
    summary.to_markdown(os.path.join(directory, f"{directory_name}_result.md"), index=False)
    print(f"Summary Results saved to {directory}/{directory_name}_result.md")