import onnx
import tf2onnx
import tensorflow as tf
import os
import shutil
from tensorflow.keras.models import load_model

# Load the Keras model
keras_model = load_model('may15_min_max.keras')

# Print model details for debugging
print(f"Model type: {type(keras_model)}")
print(f"Input shape: {keras_model.input_shape}")
print(f"Output shape: {keras_model.output_shape}")

# Save as SavedModel first
save_path = "./saved_model"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
    
tf.saved_model.save(keras_model, save_path)
print("Model saved as SavedModel format")

# Use the command-line converter instead of the Python API
import subprocess
result = subprocess.run([
    "python3", "-m", "tf2onnx.convert",
    "--saved-model", save_path,
    "--output", "may15_min_max.onnx",
    "--opset", "13"
], capture_output=True, text=True)

print("Command output:")
print(result.stdout)

if result.returncode != 0:
    print("Error:")
    print(result.stderr)
else:
    print("Conversion successful!")
