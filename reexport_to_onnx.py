
#!/usr/bin/env python3
# Re-export a saved Keras model to ONNX using CPU+fp32 so no CuDNN ops leak in.

import os, argparse, yaml, numpy as np

# MUST set this before importing tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # hide all GPUs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"         # quiet logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import tf2onnx

mixed_precision.set_global_policy("float32")     # disable fp16 for export

def main(keras_path, meta_yaml, out_path):
    with open(meta_yaml, "r") as f:
        meta = yaml.safe_load(f)
    T = int(meta["model"]["seq_len"])
    # Infer L,S from onnx_inputs we saved
    L = int(meta["model"]["onnx_inputs"]["lidar_seq"][2])
    S = int(meta["model"]["onnx_inputs"]["state_seq"][2])

    m = keras.models.load_model(keras_path, compile=False)

    # Ensure all variables are float32 (in case you trained with mixed precision)
    for v in m.variables:
        if v.dtype != tf.float32:
            v.assign(tf.cast(v, tf.float32))

    # Warmup on CPU so the graph is definitely non-CuDNN
    dummy_lidar = np.zeros((1, T, L, 1), np.float32)
    dummy_state = np.zeros((1, T, S),    np.float32)
    _ = m.predict([dummy_lidar, dummy_state], verbose=0)

    sig = (
        tf.TensorSpec([None, T, L, 1], tf.float32, name="lidar_seq"),
        tf.TensorSpec([None, T, S],    tf.float32, name="state_seq"),
    )
    tf2onnx.convert.from_keras(
        m, input_signature=sig, opset=13, output_path=out_path
    )
    print(f"[ONNX] Wrote: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", required=True)
    ap.add_argument("--meta",  required=True)  # path to metadata.yaml the trainer wrote
    ap.add_argument("--out",   required=True)
    args = ap.parse_args()
    main(args.keras, args.meta, args.out)
