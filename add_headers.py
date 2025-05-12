import pandas as pd
import os
def add_lidar_headers(input_directory):

    """
    Add headers to lidar CSV files that don't have them
    """
    subdirs = [f.path for f in os.scandir(input_directory) if f.is_dir()]
    
    for data_dir in subdirs:
        lidar_path = f"{data_dir}/input_data/lidar_data.csv"
        
        if os.path.exists(lidar_path):
            # Read the file without headers
            lidar_data = pd.read_csv(lidar_path, header=None)
            
            # Create header names based on number of columns
            num_cols = lidar_data.shape[1]
            headers = [f'lidar_{i}' for i in range(num_cols)]
            
            # Set the column names
            lidar_data.columns = headers
            
            # Save back with headers
            lidar_data.to_csv(lidar_path, index=False)
            print(f"Added headers to {lidar_path}")
        else:
            print(f"Lidar file not found: {lidar_path}")


add_lidar_headers("04_combined/rosbag2_2025_04_29-15_46_09")
