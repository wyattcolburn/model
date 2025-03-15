import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
radius = .1
frame_dkr_input = "test1/output_bag"

csv_file_input = os.path.join(frame_dkr_input,"odom_output.csv")

#output, and i want overlay the hallucinated objects
csv_file_output = "test1/input_data/obactles.csv"

input = pd.read_csv(csv_file_input)
output = pd.read_csv(csv_file_output)

odom_x_input = input['odom_x'].tolist()
odom_y_input = input['odom_y'].tolist()
center_x = output['cx'].tolist()
center_y = output['cy'].tolist()

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot odometry path (without yaw)
ax.plot(odom_x_input, odom_y_input, marker='o', linestyle='-', markersize=3, color='blue', label="Odometry Path Input")

# Plot hallucinated objects as red circles
for x, y in zip(center_x, center_y):
    circle = patches.Circle((x, y), radius, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circle)

# Set axis limits
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Odometry Path with Hallucinated Objects")
ax.legend()
ax.set_aspect('equal')  # Keep aspect ratio square

# Show and save the plot
plt.savefig(f"{csv_file_output}.png")
plt.show()

