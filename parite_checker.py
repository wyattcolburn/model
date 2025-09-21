
import os, glob, numpy as np, onnxruntime as ort
from tensorflow import keras

# --- pick your run dir (set explicitly or take newest) ---
run_dir = "data_set/2025_09_19_22_51"
if not os.path.isdir(run_dir):
    candidates = sorted(glob.glob("data_set/*"), reverse=True)
    run_dir = candidates[0]
print("[Using]", run_dir)

# load sample batch we saved earlier
lidar = np.load(os.path.join(run_dir, "val_lidar.npy")).astype(np.float32)
state = np.load(os.path.join(run_dir, "val_state.npy")).astype(np.float32)
y     = np.load(os.path.join(run_dir, "val_y.npy")).astype(np.float32)

# load models
keras_path = max(glob.glob(os.path.join(run_dir, "*.keras")))
onnx_path  = max(glob.glob(os.path.join(run_dir, "*.onnx")))
print("[Keras]", keras_path)
print("[ONNX ]", onnx_path)

k = keras.models.load_model(keras_path, compile=False)
k_pred = k.predict([lidar, state], verbose=0)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
in0, in1 = sess.get_inputs()[0].name, sess.get_inputs()[1].name
out0     = sess.get_outputs()[0].name
onnx_pred = sess.run([out0], {in0: lidar, in1: state})[0]

print("Keras MAE      :", np.abs(y - k_pred).mean(axis=0))
print("ONNX vs Keras :", np.abs(onnx_pred - k_pred).mean(axis=0))
