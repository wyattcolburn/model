
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from datetime import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GroupShuffleSplit
import tf2onnx


# =========================
# Model + Utilities
# =========================
# with these two lines:
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="custom")
class ScaleCmdOut(keras.layers.Layer):
    """Scales tanh outputs to physical limits for (v, w)."""
    def __init__(self, v_max=0.6, w_max=2.5, **kwargs):
        super().__init__(**kwargs)
        self.v_max = float(v_max)
        self.w_max = float(w_max)

    def call(self, inputs):
        v = inputs[:, 0] * self.v_max
        w = inputs[:, 1] * self.w_max
        return tf.stack([v, w], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"v_max": self.v_max, "w_max": self.w_max})
        return cfg


def create_cnn_model_with_norm(
    lidar_len: int,
    scalar_len: int,
    conv_channels=(32, 64, 128),
    dense_units=(128, 64),
    dropout=0.1,
    limit_outputs=True,
    v_max=0.6,
    w_max=2.5,
    kernel_l2=1e-5,          # mild regularization for generalization
):
    """
    Two-input CNN with built-in Normalization layers.
    Inputs:
      - lidar:  [B, lidar_len, 1]  (1D LiDAR beams)
      - state:  [B, scalar_len]    (odom v,w + goal features, etc.)
    Output:
      - [v_hat, w_hat] (optionally range-limited by tanh * [v_max, w_max])
    """
    # Normalizers (adapted on training data)
    norm_lidar  = layers.Normalization(axis=-1, name="norm_lidar")
    norm_scalar = layers.Normalization(axis=-1, name="norm_scalar")

    # --- LiDAR branch ---
    lidar_in = keras.Input(shape=(lidar_len, 1), name="lidar")
    x = layers.Reshape((lidar_len,), name="lidar_flat")(lidar_in)
    x = norm_lidar(x)
    x = layers.Reshape((lidar_len, 1), name="lidar_unflat")(x)

    for i, ch in enumerate(conv_channels):
        stride = 2 if i < len(conv_channels) - 1 else 1
        dilation = 1 if stride > 1 else 2  # never combine stride>1 with dilation>1
        x = layers.Conv1D(
            ch, kernel_size=5, strides=stride, padding="same",
            dilation_rate=dilation,
            kernel_regularizer=keras.regularizers.l2(kernel_l2)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)

    # --- Scalar branch ---
    scal_in = keras.Input(shape=(scalar_len,), name="state")
    s = norm_scalar(scal_in)

    # --- Fusion + head ---
    z = layers.Concatenate()([x, s])
    for u in dense_units:
        z = layers.Dense(u, activation="relu")(z)
        if dropout and dropout > 0:
            z = layers.Dropout(dropout)(z)

    if limit_outputs:
        raw = layers.Dense(2, activation="tanh", name="cmd_out_raw")(z)
        out = ScaleCmdOut(v_max=v_max, w_max=w_max, name="cmd_out")(raw)
    else:
        out = layers.Dense(2, name="cmd_out")(z)

    model = keras.Model([lidar_in, scal_in], out, name="LidarCNNPolicyNorm")

    # Compile (Huber is robust to label outliers)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.Huber(delta=0.5),
        metrics=["mae"],
        jit_compile=True,
    )
    return model


def split_lidar_scalar(features_df: pd.DataFrame):
    """
    Split columns into LiDAR vs. scalar using 'lidar_' prefix.
    Returns numpy arrays ready for the model:
      - X_lidar:  [N, L, 1]
      - X_scalar: [N, S]
    """
    lidar_cols = [c for c in features_df.columns if c.startswith("lidar_")]
    scalar_cols = [c for c in features_df.columns if not c.startswith("lidar_")]

    X_lidar = features_df[lidar_cols].to_numpy(dtype=np.float32).reshape((-1, len(lidar_cols), 1))
    X_scalar = features_df[scalar_cols].to_numpy(dtype=np.float32)
    return X_lidar, X_scalar, lidar_cols, scalar_cols


def convert_keras_onnx(keras_model_path_or_obj, output_model_path, lidar_len: int, scalar_len: int):
    """
    Export Keras model (with built-in Normalization) to ONNX with two inputs.
    Accepts a path or a model instance.
    """
    if isinstance(keras_model_path_or_obj, str):
        # Thanks to the registered custom layer, this works without custom_objects
        m = keras.models.load_model(keras_model_path_or_obj)
    else:
        m = keras_model_path_or_obj

    sig = (
        tf.TensorSpec([None, lidar_len, 1], tf.float32, name="lidar"),
        tf.TensorSpec([None, scalar_len],   tf.float32, name="state"),
    )
    tf2onnx.convert.from_keras(
        m, input_signature=sig, opset=17, output_path=output_model_path
    )
    print(f"[ONNX] Wrote: {output_model_path}")


def graphs(history, out_dir):
    """Save training curves to out_dir/graphs.png"""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Huber Loss')
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

            # Ensure a 'group' column exists even for older CSVs
            if 'group' not in df_feats.columns:
                inferred_gid = os.path.basename(data_dir)
                df_feats['group'] = inferred_gid

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
                group_id = f"{os.path.basename(data_dir)}::{seg}"
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
                    feats = feats.copy()
                    feats['group'] = group_id

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

            # Persist per-directory combined CSVs for reuse (includes 'group')
            local_features.to_csv(feats_p, index=False)
            local_labels.to_csv(labs_p, index=False)
            df_feats, df_labs = local_features, local_labels
            print(f"  Wrote {feats_p} and {labs_p}")

        # Append into global combined
        if combined_features is None:
            combined_features = df_feats
            combined_labels   = df_labs
        else:
            # Align columns (in case some files initially lacked 'group')
            missing_cols = set(combined_features.columns) - set(df_feats.columns)
            for c in missing_cols:
                df_feats[c] = np.nan
            missing_cols_global = set(df_feats.columns) - set(combined_features.columns)
            for c in missing_cols_global:
                combined_features[c] = np.nan
            df_feats = df_feats[combined_features.columns]

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

    # Combined meta
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
    # Train / Validate Split (group-aware)
    # =========================
    if 'group' not in combined_features.columns:
        raise RuntimeError("Expected 'group' column in combined_features before splitting.")

    groups = combined_features['group']
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(combined_features, groups=groups))

    # Sanity: no group overlap
    tr_groups = set(groups.iloc[train_idx])
    va_groups = set(groups.iloc[val_idx])
    print(f"[Groups] train={len(tr_groups)} val={len(va_groups)} disjoint={tr_groups.isdisjoint(va_groups)}")

    X_train = combined_features.iloc[train_idx].drop(columns=['group'])
    X_val   = combined_features.iloc[val_idx].drop(columns=['group'])
    y_train = combined_labels.iloc[train_idx]
    y_val   = combined_labels.iloc[val_idx]

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
    model = create_cnn_model_with_norm(lidar_len=L, scalar_len=S)

    # Adapt normalization layers on RAW training data
    model.get_layer("norm_lidar").adapt(Xtr_lidar.reshape(-1, L))
    model.get_layer("norm_scalar").adapt(Xtr_scalar)

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=12, min_delta=1e-4, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )
    best_path = os.path.join(new_dir, "best.keras")
    ckpt = keras.callbacks.ModelCheckpoint(
        best_path, monitor='val_loss', save_best_only=True
    )

    history = model.fit(
        [Xtr_lidar, Xtr_scalar], y_train.values,
        validation_data=([Xva_lidar, Xva_scalar], y_val.values),
        epochs=500, batch_size=256, callbacks=[early_stopping, reduce_lr, ckpt], verbose=1
    )

    # =========================
    # Save Keras + ONNX + Graphs
    # =========================
    final_path = os.path.join(new_dir, f"{timestamp}.keras")
    model.save(final_path)
    print(f"[Keras] Wrote final: {final_path}")

    # Prefer exporting ONNX from the best checkpoint (val_loss-min); fall back to final
    export_from = best_path if os.path.exists(best_path) else final_path
    onnx_path = os.path.join(new_dir, f"{timestamp}.onnx")
    convert_keras_onnx(export_from, onnx_path, L, S)

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
