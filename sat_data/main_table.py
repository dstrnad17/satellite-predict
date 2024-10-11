import os
import pandas as pd
import numpy as np

# Specify directory containing the .pkl files
directory_list = ["goes8", "cluster1", "themise"] # List desired satellites
base_directory = "./main_results/"
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# List of columns to calculate mean, std dev, and MSE
preds_nn3 = ['bx_nn3', 'by_nn3', 'bz_nn3']
preds_lr = ['bx_lr', 'by_lr', 'bz_lr']
preds_nn1 = ['bx_nn1', 'by_nn1', 'bz_nn1']
actual = ['bx_actual', 'by_actual', 'bz_actual']

# Iterate over each directory in the directory_list
for directory_name in directory_list:
    directory = os.path.join(base_directory, directory_name)
    summary_data = []
    combined_data_by_input = {}

    # Iterate through all .pkl files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(directory, file_name)
            
            # Extract the removed input from the file name
            removed_input = file_name.split('_')[1].split('.')[0]
            
            # Load the .pkl file
            df_dict = pd.read_pickle(file_path)
            
            # Combine data for each removed input
            if removed_input not in combined_data_by_input:
                combined_data_by_input[removed_input] = []
            
            # Iterate through each DataFrame in the .pkl file
            for key, df in df_dict.items():

                combined_data_by_input[removed_input].append(df)
                
                # Calculate mean, std, mse
                means_nn = df[preds_nn3].mean()
                stds_nn = df[preds_nn3].std()
                mse_nn = np.mean(((df[actual].to_numpy() - df[preds_nn3].to_numpy()) ** 2), axis=0)

                means_lr = df[preds_lr].mean()
                stds_lr = df[preds_lr].std()
                mse_lr = np.mean(((df[actual].to_numpy() - df[preds_lr].to_numpy()) ** 2), axis=0)

                means_nn_single = df[preds_nn1].mean()
                stds_nn_single = df[preds_nn1].std()
                mse_nn_single = np.mean(((df[actual].to_numpy() - df[preds_nn1].to_numpy()) ** 2), axis=0)

                # Create a combined result for each model
                for model_type, means, stds, mse in zip(
                    ['nn3', 'nn1', 'lr'], 
                    [means_nn, means_nn_single, means_lr], 
                    [stds_nn, stds_nn_single, stds_lr], 
                    [mse_nn, mse_nn_single, mse_lr]
                ):
                    result = {
                        'Removed Input': removed_input,
                        'Model': model_type,
                        'Mean_bx': means['bx_' + model_type],
                        'Std_bx': stds['bx_' + model_type],
                        'MSE_bx': mse[0],
                        'Mean_by': means['by_' + model_type],
                        'Std_by': stds['by_' + model_type],
                        'MSE_by': mse[1],
                        'Mean_bz': means['bz_' + model_type],
                        'Std_bz': stds['bz_' + model_type],
                        'MSE_bz': mse[2],
                        'Key': key
                    }

                    # Append the result to the summary_data list
                    summary_data.append(result)

    table_directory = f"./{directory}/{directory_name}_results"
    if not os.path.exists(table_directory):
        os.makedirs(table_directory)

    # Combine all the summary data into one table
    combined_summary = pd.DataFrame(summary_data)

    key_list = combined_summary['Key'].unique()

    # Create and save a table for each key
    for key in key_list:
        table_rep = combined_summary[combined_summary['Key'] == key]
        table_rep.to_markdown(os.path.join(table_directory, f"{directory_name}_result_{key}.md"), index=False)

    combined_result = []

    for removed_input, df_list in combined_data_by_input.items():
        combined_df = pd.concat(df_list, ignore_index=True)  # Combine all repetitions for this input
        
        # Calculate the overall mean, std dev, and mse across all repetitions
        means_combined_nn = combined_df[preds_nn3].mean()
        stds_combined_nn = combined_df[preds_nn3].std()
        mse_combined_nn = np.mean(((combined_df[actual].to_numpy() - combined_df[preds_nn3].to_numpy()) ** 2), axis=0)

        means_combined_lr = combined_df[preds_lr].mean()
        stds_combined_lr = combined_df[preds_lr].std()
        mse_combined_lr = np.mean(((combined_df[actual].to_numpy() - combined_df[preds_lr].to_numpy()) ** 2), axis=0)

        means_combined_nn_single = combined_df[preds_nn1].mean()
        stds_combined_nn_single = combined_df[preds_nn1].std()
        mse_combined_nn_single = np.mean(((combined_df[actual].to_numpy() - combined_df[preds_nn1].to_numpy()) ** 2), axis=0)

        # Create the summary for the combined DataFrame for each removed input
        for model_type, means, stds, mse in zip(
            ['nn3', 'nn1', 'lr'], 
            [means_combined_nn, means_combined_nn_single, means_combined_lr], 
            [stds_combined_nn, stds_combined_nn_single, stds_combined_lr], 
            [mse_combined_nn, mse_combined_nn_single, mse_combined_lr]
        ):
            combined_result.append({
                'Removed Input': removed_input,
                'Model': model_type,
                'Mean_bx': means['bx_' + model_type],
                'Std_bx': stds['bx_' + model_type],
                'MSE_bx': mse[0],
                'Mean_by': means['by_' + model_type],
                'Std_by': stds['by_' + model_type],
                'MSE_by': mse[1],
                'Mean_bz': means['bz_' + model_type],
                'Std_bz': stds['bz_' + model_type],
                'MSE_bz': mse[2]
            })

    # Save the combined result as a markdown file
    combined_result_df = pd.DataFrame(combined_result)
    combined_result_df.to_markdown(os.path.join(directory, f"{directory_name}_result.md"), index=False)