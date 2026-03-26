## Reading traces from database

# load_database.py

import os
import numpy as np
import pandas as pd

def read_binary_file(file_path, dtype=np.float32):
    """
    Reads a binary file into a single-column DataFrame.
    """
    try:
        data = np.fromfile(file_path, dtype=dtype)
        df = pd.DataFrame(data, columns=[os.path.basename(file_path)])
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def load_database_data(base_path, years, site_names, levels, variables_to_read=None):
    """
    Reads selected binary files across multiple years, levels, and sites, 
    and stores each site's data in a separate DataFrame.
    Returns a dictionary with site names as keys and corresponding DataFrames as values.
    """
    site_data = {}  # Dictionary to store site-specific DataFrames
    
    for site_name in site_names:
        all_data = []
        
        for year in years:
            clean_tv_path = os.path.join(base_path, str(year), site_name, "Clean/SecondStage/clean_tv")
            if not os.path.exists(clean_tv_path):
                print(f"Skipping year {year} for site {site_name}: clean_tv not found")
                continue
            
            try:
                dt = np.fromfile(clean_tv_path, dtype=np.float64)
                timestamp_end = pd.to_datetime(dt - 719529, unit='D').round('s')
                timestamp_start = timestamp_end - pd.Timedelta(minutes=30)
                timestamp_data = pd.DataFrame({'timestamp_start': timestamp_start})
            except Exception as e:
                print(f"Error processing clean_tv for {site_name} in {year}: {e}")
                continue
            
            for level in levels:
                folder_path = os.path.join(base_path, str(year), site_name, level)
                
                if not os.path.exists(folder_path):
                    print(f"Skipping: {folder_path} (does not exist)")
                    continue
                
                year_level_data = {}
                
                for file in sorted(os.listdir(folder_path)):
                    if variables_to_read and file not in variables_to_read:
                        continue
                    
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        df = read_binary_file(file_path, dtype=np.float32)
                        if not df.empty:
                            year_level_data[os.path.basename(file_path)] = df.iloc[:, 0]
                
                if year_level_data:
                    df = pd.DataFrame(year_level_data)
                    
                    if not timestamp_data.empty:
                        df = pd.concat([timestamp_data, df], axis=1)
                        df.set_index('timestamp_start', inplace=True)
                        df.index.name = 'Datetime'
                        df['Year'] = df.index.year
                        df['Month'] = df.index.month
                        df['Hour'] = df.index.hour
                        df = df[['Year', 'Month', 'Hour'] + [col for col in df.columns if col not in ['Year', 'Month', 'Hour']]]
                    
                    all_data.append(df)
        
        if all_data:
            site_data[site_name] = pd.concat(all_data, ignore_index=False)
            print(f"Data loaded for site: {site_name}")
        else:
            print(f"No valid data found for site: {site_name}")
            site_data[site_name] = pd.DataFrame()
    
    return site_data