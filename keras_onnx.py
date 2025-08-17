# import onnx
# import tf2onnx
# import tensorflow as tf
# import os
# import shutil
# from tensorflow.keras.models import load_model
#
# # Load the Keras model
# keras_model = load_model('combine_08_14.keras')
#
# # Print model details for debugging
# print(f"Model type: {type(keras_model)}")
# print(f"Input shape: {keras_model.input_shape}")
# print(f"Output shape: {keras_model.output_shape}")
#
# # Save as SavedModel first
# save_path = "./saved_model"
# if os.path.exists(save_path):
#     shutil.rmtree(save_path)
#     
# tf.saved_model.save(keras_model, save_path)
# print("Model saved as SavedModel format")
#
# # Use the command-line converter instead of the Python API
# import subprocess
# result = subprocess.run([
#     "python3", "-m", "tf2onnx.convert",
#     "--saved-model", save_path,
#     "--output", "combine_08_14.onnx",
#     "--opset", "13"
# ], capture_output=True, text=True)
#
# print("Command output:")
# print(result.stdout)
#
# if result.returncode != 0:
#     print("Error:")
#     print(result.stderr)
# else:
#     print("Conversion successful!")


import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model

keras_model = load_model('combine_08_14.keras')

spec = (tf.TensorSpec([None, 1085], tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(
    keras_model, input_signature=spec, opset=17, output_path="combine_08_14.onnx"
)
print("Exported combine_08_14.onnx")
