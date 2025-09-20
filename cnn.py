
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tf2onnx


# =========================
# Model + Utilities
# =========================

def create_cnn_model_with_norm(lidar_len: int, scalar_len: int,
                               conv_channels=(32, 64, 128),
                               dense_units=(128, 64), dropout=0.1):
    """
    Two-input CNN with built-in Normalization layers.
    Inputs:
      - lidar:   [B, lidar_len, 1]  (1D LiDAR beams)
      - state:   [B, scalar_len]    (odom v,w + goal features, etc.)
    Output:
      - [v_hat, w_hat]
    """
    # Normalizers: statistics will be ADAPTED before training and baked into the graph.
    norm_lidar  = layers.Normalization(axis=-1, name="norm_lidar")   # expects [B, lidar_len]
    norm_scalar = layers.Normalization(axis=-1, name="norm_scalar")  # expects [B, scalar_len]

    # --- LiDAR branch ---
    lidar_in = keras.Input(shape=(lidar_len, 1), name="lidar")
    x = layers.Reshape((lidar_len,), name="lidar_flat")(lidar_in)
    x = norm_lidar(x)                                  # z-score per beam
    x = layers.Reshape((lidar_len, 1), name="lidar_unflat")(x)
    for i, ch in enumerate(conv_channels):
        x = layers.Conv1D(ch, kernel_size=5,
                          strides=2 if i < len(conv_channels)-1 else 1,
                          padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalMaxPooling1D()(x)                 # [B, C]

    # --- Scalar branch ---
    scal_in = keras.Input(shape=(scalar_len,), name="state")
    s = norm_scalar(scal_in)

    # --- Fusion + head ---
    z = layers.Concatenate()([x, s])
    for u in dense_units:
        z = layers.Dense(u, activation="relu")(z)
        if dropout and dropout > 0:
            z = layers.Dropout(dropout)(z)
    out = layers.Dense(2, name="cmd_out")(z)           # -> [v_hat, w_hat]

    model = keras.Model([lidar_in, scal_in], out, name="LidarCNNPolicyNorm")

    lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 10000, 0.9)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def split_lidar_scalar(features_df):
    """
    Split columns into LiDAR vs. scalar using 'lidar' prefix.
    Returns numpy arrays ready for the model:
      - X_lidar:  [N, L, 1]
      - X_scalar: [N, S]
    """
    lidar_cols = [c for c in features_df.columns if c.startswith("lidar")]
    scalar_cols = [c for c in features_df.columns if not c.startswith("lidar")]

    X_lidar = features_df[lidar_cols].to_numpy(dtype=np.float32).reshape((-1, len(lidar_cols), 1))
    X_scalar = features_df[scalar_cols].to_numpy(dtype=np.float32)
    return X_lidar, X_scalar, lidar_cols, scalar_cols


def convert_keras_onnx(keras_model_path, output_model_path, lidar_len: int, scalar_len: int):
    """
    Export Keras model (with built-in Normalization) to ONNX with two inputs.
    """
    m = keras.models.load_model(keras_model_path)
    sig = (
        tf.TensorSpec([None, lidar_len, 1], tf.float32, name="lidar"),
        tf.TensorSpec([None, scalar_len],   tf.float32, name="state"),
    )
    _onnx_model, _ = tf2onnx.convert.from_keras(
        m, input_signature=sig, opset=17, output_path=output_model_path
    )
    print(f"[ONNX] Wrote: {output_model_path}")


def graphs(history, out_dir):
    """
    Save training curves to out_dir/graphs.png
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    out_path = os.path.join(out_dir, "graphs.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Graphs] Wrote: {out_path}")


# =========================
# Dataset Builder & Trainer
# =========================

def large_dataset(input_directory, single_dkr_flag, adaptive_flag):
    """
    Creates a combined dataset from one or more data directories and trains the CNN.
    Each data_dir can already contain combined CSVs OR per-segment (seg_*) folders.
    """
    # Resolve subdirectories to process
    if single_dkr_flag:
        if isinstance(input_directory, str):
            if os.path.isdir(input_directory):
                subdirs = [f.path for f in os.scandir(input_directory) if f.is_dir()]
            else:
                raise ValueError(f"Directory not found: {input_directory}")
        elif isinstance(input_directory, list) and len(input_directory) == 1:
            base = input_directory[0]
            subdirs = [os.path.join(base, d) for d in os.listdir(base)
                       if os.path.isdir(os.path.join(base, d))]
        else:
            raise ValueError("input_directory must be a string path or a single-item list")
    else:
        if isinstance(input_directory, list):
            subdirs = input_directory
        elif isinstance(input_directory, str):
            subdirs = [input_directory]
        else:
            raise ValueError("input_directory must be a string path or list of paths")
        print(f"Processing {len(subdirs)} directories")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    new_dir = os.path.join("data_set", timestamp)
    os.makedirs(new_dir, exist_ok=True)

    yaml_data = {
        "run": {"timestamp": timestamp, "output_dir": os.path.abspath(new_dir)},
        "datasets": [],
        "combined": {},
    }

    combined_features = None
    combined_labels = None
    total_rows = 0

    # --- Collect / build combined_features & combined_labels ---
    for data_dir in subdirs:
        feats_p = os.path.join(data_dir, "combined_features.csv")
        labs_p  = os.path.join(data_dir, "combined_labels.csv")

        if os.path.exists(feats_p) and os.path.exists(labs_p):
            print(f"[{data_dir}] Using existing combined CSVs")
            df_feats = pd.read_csv(feats_p, header=0)
            df_labs  = pd.read_csv(labs_p,  header=0)
        else:
            print(f"[{data_dir}] Building combined CSVs from seg_*")
            try:
                seg_dirs_all = [d for d in os.listdir(data_dir)
                                if os.path.isdir(os.path.join(data_dir, d))]
                seg_dirs = [d for d in seg_dirs_all if d.startswith("seg_")]
                if not seg_dirs:
                    print(f"  No seg_* directories in {data_dir}, skipping")
                    continue
            except OSError as e:
                print(f"  Error accessing {data_dir}: {e}")
                continue

            local_features = None
            local_labels   = None

            for seg in seg_dirs:
                seg_path = os.path.join(data_dir, seg)
                try:
                    training_lidar = pd.read_csv(os.path.join(seg_path, "input_data/lidar_data.csv"), header=0)
                    training_odom  = pd.read_csv(os.path.join(seg_path, "input_data/odom_data.csv"),  header=0)
                    if adaptive_flag:
                        training_local_goals = pd.read_csv(os.path.join(seg_path, "input_data/adaptive_local_goals.csv"), header=0)
                    else:
                        training_local_goals = pd.read_csv(os.path.join(seg_path, "input_data/local_goals.csv"), header=0)
                    training_labels = pd.read_csv(os.path.join(seg_path, "input_data/cmd_vel_output.csv"), header=0)

                    # Basic preprocessing consistent with your previous script
                    training_lidar = training_lidar.iloc[:-1, :]
                    training_odom  = training_odom.iloc[:, [5, 6]]
                    if adaptive_flag:
                        training_local_goals = training_local_goals[:-1]
                    else:
                        training_local_goals = training_local_goals.iloc[:, [1, 2, 3]]
                    training_labels = training_labels.iloc[:, [2, 3]]

                    # Rename for clarity / consistency
                    training_odom.columns  = [f'odom_{c}' for c in training_odom.columns]
                    training_lidar.columns = [f'lidar_{i}' for i in range(training_lidar.shape[1])]
                    training_local_goals.columns = [f'goal_{c}' for c in training_local_goals.columns]

                    if training_lidar.shape[0] <= 200:
                        print(f"  {seg_path} too small ({training_lidar.shape[0]} rows), skipping")
                        continue

                    feats = pd.concat([training_odom, training_local_goals, training_lidar], axis=1)

                    if local_features is None:
                        local_features = feats
                        local_labels   = training_labels
                    else:
                        if feats.shape[1] != local_features.shape[1]:
                            print(f"  Feature dim mismatch in {seg}: {feats.shape[1]} vs {local_features.shape[1]}, skipping")
                            continue
                        local_features = pd.concat([local_features, feats], axis=0, ignore_index=True)
                        local_labels   = pd.concat([local_labels,   training_labels], axis=0, ignore_index=True)

                except Exception as e:
                    print(f"  Error processing {seg_path}: {e}")
                    continue

            if local_features is None or local_labels is None:
                print(f"  No valid data in {data_dir}, skipping")
                continue

            # Persist per-directory combined CSVs for reuse
            local_features.to_csv(feats_p, index=False)
            local_labels.to_csv(labs_p, index=False)
            df_feats, df_labs = local_features, local_labels
            print(f"  Wrote {feats_p} and {labs_p}")

        # Append into global combined
        if combined_features is None:
            combined_features = df_feats
            combined_labels   = df_labs
        else:
            if df_feats.shape[1] != combined_features.shape[1]:
                print(f"  Skipping {data_dir} due to feature dim mismatch: {df_feats.shape[1]} vs {combined_features.shape[1]}")
                continue
            combined_features = pd.concat([combined_features, df_feats], axis=0, ignore_index=True)
            combined_labels   = pd.concat([combined_labels,   df_labs], axis=0, ignore_index=True)

        total_rows += len(df_feats)

        yaml_data["datasets"].append({
            "name": os.path.basename(os.path.normpath(data_dir)),
            "path": os.path.abspath(data_dir),
            "features_csv": os.path.abspath(feats_p),
            "labels_csv":   os.path.abspath(labs_p),
            "features_shape": {"rows": int(df_feats.shape[0]), "cols": int(df_feats.shape[1])},
            "labels_shape":   {"rows": int(df_labs.shape[0]),  "cols": int(df_labs.shape[1])},
        })

    if combined_features is None:
        print("No valid data found across all directories.")
        return

    yaml_data["combined"] = {
        "features_shape": {"rows": int(combined_features.shape[0]), "cols": int(combined_features.shape[1])},
        "labels_shape":   {"rows": int(combined_labels.shape[0]),   "cols": int(combined_labels.shape[1])},
        "num_datasets": len(yaml_data["datasets"]),
        "total_rows": int(total_rows),
    }

    # Write run metadata YAML
    meta_path = os.path.join(new_dir, "metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)
    print(f"[Meta] Wrote: {meta_path}")

    # =========================
    # Train / Validate Split
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        combined_features, combined_labels, test_size=0.2, random_state=42
    )

    # Split into LiDAR vs scalar (NO external scaling)
    Xtr_lidar, Xtr_scalar, lidar_cols, scalar_cols = split_lidar_scalar(X_train)
    Xva_lidar, Xva_scalar, _, _ = split_lidar_scalar(X_val)

    L = Xtr_lidar.shape[1]
    S = Xtr_scalar.shape[1]
    print(f"[Shapes] LiDAR beams (L) = {L}, scalar feats (S) = {S}")
    print(f"[Combined] features {combined_features.shape}, labels {combined_labels.shape}")

    # =========================
    # Build, Adapt, Train
    # =========================
    epochsVal = 500
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=5, min_delta=0.001, restore_best_weights=True
    )

    model = create_cnn_model_with_norm(lidar_len=L, scalar_len=S)

    # IMPORTANT: adapt normalization layers on RAW training data
    norm_lidar  = model.get_layer("norm_lidar")
    norm_scalar = model.get_layer("norm_scalar")
    norm_lidar.adapt(Xtr_lidar.reshape(-1, L))  # [N, L]
    norm_scalar.adapt(Xtr_scalar)               # [N, S]

    history = model.fit(
        [Xtr_lidar, Xtr_scalar], y_train.values,
        validation_data=([Xva_lidar, Xva_scalar], y_val.values),
        epochs=epochsVal, batch_size=256, callbacks=[], verbose=1
    )

    # =========================
    # Save Keras + ONNX + Graphs
    # =========================
    model_path = os.path.join(new_dir, f"{timestamp}.keras")
    model.save(model_path)
    print(f"[Keras] Wrote: {model_path}")

    onnx_path = os.path.join(new_dir, f"{timestamp}.onnx")
    convert_keras_onnx(model_path, onnx_path, L, S)

    graphs(history, new_dir)


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Robot CNN (LiDAR+Scalars) Training with Built-in Normalization")
    parser.add_argument("input_bag", type=str, nargs='+',
                        help="Path(s) to data directories (or a parent directory with seg_* subfolders)")
    parser.add_argument("--large", action="store_true",
                        help="Multiple data directories mode (or treat single path as parent dir list)")
    parser.add_argument("--single_dkr", action='store_true',
                        help="All training data within one directory containing subdirectories")
    parser.add_argument("--adaptive", action='store_true',
                        help="Use adaptive_local_goals.csv instead of local_goals.csv")
    args = parser.parse_args()

    # Use the same behavior you showed: only the 'large' pipeline here
    if args.large:
        large_dataset(args.input_bag, args.single_dkr, args.adaptive)
    else:
        # If not using --large, still call large_dataset with a single path list
        large_dataset(args.input_bag, args.single_dkr, args.adaptive)


if __name__ == "__main__":
    main()
