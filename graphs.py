import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import ros_csv

parser = argparse.ArgumentParser(description="Robot Model Training and Inference Graphs utils")
parser.add_argument("input_bag", type=str, help="Path to input data bag")
parser.add_argument("odom_csv_file", type=str, help="name of output file")
args = parser.parse_args()

if os.path.exists(args.input_bag):
    print("input bag exists")
else:
    raise ValueError("No bag")

# Construct file paths
csv_file_input = os.path.join(args.input_bag, "input_data", "odom_data.csv")
output_dkr = os.path.join(args.input_bag, "output_bag")
csv_file_output = os.path.join(args.input_bag, "output_bag", args.odom_csv_file)

# Ensure the output_data directory exists
output_data_dir = os.path.join(args.input_bag, "output_data")
os.makedirs(output_data_dir, exist_ok=True)

# If the CSV file doesn't exist, create it using ros_csv.save_to_csv
if not os.path.exists(csv_file_output):
    ros_csv.save_to_csv(output_dkr, csv_file_output)
else:
    print("CSV file already exists at", csv_file_output)

# Read the CSV files
input_df = pd.read_csv(csv_file_input)
output_df = pd.read_csv(csv_file_output)

# Extract odometry data
odom_x_input = input_df['odom_x'].tolist()
odom_y_input = input_df['odom_y'].tolist()
odom_x_output = output_df['odom_x'].tolist()
odom_y_output = output_df['odom_y'].tolist()

# Plot the odometry paths
plt.figure(figsize=(8, 6))
plt.plot(odom_x_input, odom_y_input, marker='o', linestyle='-', markersize=3, color='blue', label="Odometry Path Input")
plt.plot(odom_x_output, odom_y_output, marker='o', linestyle='-', markersize=3, color='red', label="Odometry Path Output")
plt.legend()
plt.show()

plt.savefig(f"{args.input_bag}/output_data/{args.odom_csv_file}.png")

print("png has been saved")
