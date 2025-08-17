
# convert_keras_onnx.py
import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model

m = load_model("combine_08_14.keras")

# Patch for tf2onnx<=1.16 expecting Keras-2 API
flat_out = tf.nest.flatten(m.outputs)
m.output_names = [t.name.split(":")[0] for t in flat_out]

INPUT_DIM = m.input_shape[-1]
spec = (tf.TensorSpec([None, INPUT_DIM], tf.float32, name="input"),)

onnx_model, _ = tf2onnx.convert.from_keras(
    m, input_signature=spec, opset=17, output_path="combine_08_14.onnx"
)
print("Exported model.onnx")
