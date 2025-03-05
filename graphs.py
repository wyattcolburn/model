import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import ros_csv
parser = argparse.ArgumentParser(description="Robot Model Training and Inference Graphs utils")
parser.add_argument("input_bag", type=str, help="Path to input data bag")


args = parser.parse_args()
if os.path.exists(args.input_bag):
    print("input bag exists")
else:
    raise ValueError("No bag")



csv_file_input = os.path.join(args.input_bag,"input_data/odom_data.csv")
output_dkr = os.path.join(args.input_bag, "output_bag")

if os.path.exists(output_dkr)
    continue
else:
    ros_csv.save_to_csv(output_dkr, os.path.join(args.input_bag, "output_data/odom_data.csv")

csv_file_output = os.path.join(args.input_bag, "output_data/odom_data.csv")
input = pd.read_csv(csv_file_input)
output = pd.read_csv(csv_file_output)

odom_x_input = input['odom_x'].tolist()
odom_y_input = input['odom_y'].tolist()
odom_x_output= output['odom_x'].tolist()
odom_y_output = output['odom_y'].tolist()

plt.figure(figsize=(8, 6))

    ### Plot odometry path (without yaw)
plt.plot(odom_x_input, odom_y_input, marker='o', linestyle='-', markersize=3, color='blue', label="Odometry Path Input")

plt.plot(odom_x_output, odom_y_output, marker='o', linestyle='-', markersize=3, color='red', label="Odometry Path output")

plt.show()
