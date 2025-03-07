import argparse
from training_complete import generate_frames_obst
parser = argparse.ArgumentParser(description="Robot Model Training and Inference Graphs utils")
parser.add_argument("input_bag", type=str, help="Path to input data bag")
args = parser.parse_args()


input_file = os.path.join(args.input_bag, "input_data")


