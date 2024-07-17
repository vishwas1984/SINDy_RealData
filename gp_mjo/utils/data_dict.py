import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle

# Function to read and process all txt files in a given folder
def process_folder(subfolder_path):
    # Columns to ignore
    ignore_columns = []#["RMM1(0)", "RMM2(0)", "RMM1_0", "RMM2_0"]


    # List to hold dataframes for each txt file
    dataframes = []
    
    # Traverse the subfolder for txt files
    for root, _, files in os.walk(subfolder_path):
        for file in files:
            if file.endswith('.txt') and not file.startswith("Results"):
                # Extract lead_time from the file name
                lead_time = int(os.path.splitext(file)[0].replace('ECMWF', ''))
                file_path = os.path.join(root, file)
                
                # Read the txt file into a dataframe with comma as separator
                df = pd.read_csv(file_path, sep=',', engine='python')
                
                # Drop the columns to be ignored
                df.drop(columns=[col for col in ignore_columns if col in df.columns], inplace=True)
                
                # Add lead_time column
                df['lead_time'] = lead_time

                if "RMM1_0" in df.columns:
                    df.rename(columns={"RMM1_0": "RMM1(0)"}, inplace=True)
                if "RMM2_0" in df.columns:
                    df.rename(columns={"RMM2_0": "RMM2(0)"}, inplace=True)
                
                # Append the dataframe to the list
                dataframes.append(df)

                
    
    # Concatenate all dataframes into one for the subfolder
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Convert 'Date_day0' to datetime and add 1 day to create 'S'
        combined_df['Date_day0'] = pd.to_datetime(combined_df['Date_day0'], format='%Y-%m-%d')
        combined_df['S'] = (combined_df['Date_day0'] + timedelta(days=1) - datetime(1979, 1, 1)).dt.days

        # Add Amplitude(0)
        if "RMM1(0)" in combined_df.columns and "RMM2(0)" in combined_df.columns:
            combined_df['Amplitude(0)'] = np.sqrt(combined_df["RMM1(0)"]**2 + combined_df["RMM2(0)"]**2)
        elif "RMM1_0" in combined_df.columns and "RMM2_0" in combined_df.columns:
            combined_df['Amplitude(0)'] = np.sqrt(combined_df["RMM1_0"]**2 + combined_df["RMM2_0"]**2)

        # Sort by Date_day0 and lead_time
        combined_df.sort_values(by=['Date_day0', 'lead_time'], inplace=True)
        
        return combined_df
    else:
        print(f"No .txt files found in subfolder: {subfolder_path}")
        return None

def creat_dict(folder_path):
    # Dictionary to store dataframes
    dataframes_dict = {}

    # Traverse the main folder to find subfolders
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            df = process_folder(subfolder_path)
            if df is not None:
                if subfolder == "EnsMean":
                    subfolder_key = "ensemble_mean"
                    # Combine rows to a numpy array
                    num_dates = df['Date_day0'].nunique()
                    num_lead_times = df['lead_time'].nunique()
                    combined_arrays = {}
                    for col in df.columns:
                        if col not in ['Date_day0', 'lead_time']:
                            combined_arrays[col] = df.pivot(index='Date_day0', columns='lead_time', values=col).to_numpy()
                    dataframes_dict[subfolder_key] = combined_arrays
                elif subfolder == "EnsMembers":
                    subfolder_key = "ensembles"
                    # Create combined arrays for RMM1(forecast) and RMM2(forecast)
                    num_dates = df['Date_day0'].nunique()
                    num_lead_times = df['lead_time'].nunique()
                    num_pf = len([col for col in df.columns if col.startswith('RMM1_pf')])
                    combined_arrays = {
                        'RMM1(forecast)': np.zeros((num_dates, num_lead_times, num_pf)),
                        'RMM2(forecast)': np.zeros((num_dates, num_lead_times, num_pf)),
                    }
                    for i in range(num_pf):
                        pf_index = i + 1
                        combined_arrays['RMM1(forecast)'][:, :, i] = df.pivot(index='Date_day0', columns='lead_time', values=f'RMM1_pf{pf_index}').to_numpy()
                        combined_arrays['RMM2(forecast)'][:, :, i] = df.pivot(index='Date_day0', columns='lead_time', values=f'RMM2_pf{pf_index}').to_numpy()
                    combined_arrays['S'] = df.pivot(index='Date_day0', columns='lead_time', values='S').to_numpy()
                    combined_arrays['RMM1(obs)'] = df.pivot(index='Date_day0', columns='lead_time', values='RMM1_obs').to_numpy()
                    combined_arrays['RMM2(obs)'] = df.pivot(index='Date_day0', columns='lead_time', values='RMM2_obs').to_numpy()
                    dataframes_dict[subfolder_key] = combined_arrays
                else:
                    dataframes_dict[subfolder] = df
            
            # Process Results files
            for file in os.listdir(subfolder_path):
                if file.startswith("Results"):
                    file_path = os.path.join(subfolder_path, file)
                    df_results = pd.read_csv(file_path, sep=',', engine='python')
                    if "Bivariate Correlation" in df_results.columns:
                        df_results.rename(columns={"Bivariate Correlation": "COR"}, inplace=True)
                    for col in df_results:
                        combined_arrays[col] = df_results[col].to_numpy()
                    dataframes_dict[subfolder] = combined_arrays
            
            print(f"Processed dataframe for subfolder: {subfolder}")

    print("Processing completed!")

    return dataframes_dict


def save(dataframes_dict, folder_path):
    # Compare 'S' values and keep only the rows with the same value
    ensemble_mean_S = dataframes_dict['ensemble_mean']['S'][:, 0]
    ensembles_S = dataframes_dict['ensembles']['S'][:, 0]

    common_S_values = np.intersect1d(ensemble_mean_S, ensembles_S)

    # Filter the dataframes to keep only rows with common 'S' values
    def filter_by_common_S(data, common_S):
        mask = np.isin(data['S'][:, 0], common_S)
        ignore_keys = ['COR', 'RMSE', 'Err Amp', 'Err Phase', 'CRPS']
        filtered_data = {k: v[mask] for k, v in data.items() if k not in ignore_keys}
        return filtered_data

    filtered_ensemble_mean = filter_by_common_S(dataframes_dict['ensemble_mean'], common_S_values)
    filtered_ensembles = filter_by_common_S(dataframes_dict['ensembles'], common_S_values)

    # Save the filtered dataframes_dict to a file
    filtered_save_path = os.path.join(folder_path, 'filtered_ecmwf_dict.pkl')
    filtered_dataframes_dict = {
        'ensemble_mean': filtered_ensemble_mean,
        'ensembles': filtered_ensembles
    }

    # Save the dataframes_dict to a file
    save_path = os.path.join(folder_path, 'ecmwf_dict.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(filtered_dataframes_dict, f)

    print(f"Filtered dataframes dictionary saved to {save_path}")

    # Example usage to access the dataframes:
    for subfolder, df in dataframes_dict.items():
        print(f"Data for {subfolder}:")
        if isinstance(df, dict):
            for key, value in df.items():
                print(f"{key}:\n{value}\n")
        else:
            print(df.head())

# Example code to load the saved dataframes dictionary
def load(folder_path):
    load_path = os.path.join(folder_path, 'ecmwf_dict.pkl')
    with open(load_path, 'rb') as f:
        loaded_dataframes_dict = pickle.load(f)
        print(loaded_dataframes_dict.keys())


# Set the directory paths
data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(data_dir, "data", 'ecmwf_txt')



def main():
    dataframes_dict = creat_dict(folder_path)
    save(dataframes_dict, folder_path)
    load(folder_path)



if __name__ == "__main__":
    main()