
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a CNN+LSTM policy on LiDAR + scalar features using fixed-length windows,
then export to ONNX for stateless (windowed) inference in ROS.

Data layout (per segment folder):
  input_data/
    lidar_data.csv         # LiDAR beams, columns = beams
    odom_data.csv          # uses cols [5, 6] (v, w) as features
    local_goals.csv        # uses cols [1, 2, 3]  OR adaptive_local_goals.csv if --adaptive
    cmd_vel_output.csv     # uses cols [2, 3] as regression targets (v_hat, w_hat)

Author: you :)
"""

import os
import glob
import argparse
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tf2onnx


# =========================
# Utilities: Plotting
# =========================

def graphs(history, out_dir):
    """
    Save training curves to out_dir/graphs.png
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Training and Validation Huber Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Huber Loss')
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation')
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
# Data Builders
# =========================

def split_lidar_scalar(features_df):
    """
    Split columns into LiDAR vs. scalar using 'lidar_' prefix.
    Returns:
      X_lidar:  numpy [N, L, 1]
      X_scalar: numpy [N, S]
      lidar_cols, scalar_cols (lists of names in order)
    """
    lidar_cols = [c for c in features_df.columns if c.startswith("lidar_")]
    scalar_cols = [c for c in features_df.columns if c not in lidar_cols + ['group']]

    X_lidar = features_df[lidar_cols].to_numpy(dtype=np.float32).reshape((-1, len(lidar_cols), 1))
    X_scalar = features_df[scalar_cols].to_numpy(dtype=np.float32)
    return X_lidar, X_scalar, lidar_cols, scalar_cols


def build_sequence_dataset(features_df, labels_df, seq_len=10, stride=1):
    """
    Create sliding windows per 'group' (episode/segment) so sequences don't cross boundaries.

    Returns:
      X_lidar_seq:  [N, T, L, 1]
      X_scalar_seq: [N, T, S]
      Y:            [N, 2]  (last-step cmd_vel)
      lidar_cols, scalar_cols
    """
    assert 'group' in features_df.columns, "features_df must contain a 'group' column"

    lidar_cols  = [c for c in features_df.columns if c.startswith("lidar_")]
    scalar_cols = [c for c in features_df.columns if c not in lidar_cols + ['group']]

    Xl_list, Xs_list, Y_list = [], [], []

    # groupby but preserve the original order within each group
    for gid, idx in features_df.groupby('group').groups.items():
        Fg = features_df.loc[idx].reset_index(drop=True)
        Lg = labels_df.loc[idx].reset_index(drop=True)

        Xl = Fg[lidar_cols].to_numpy(dtype=np.float32)   # [G, L]
        Xs = Fg[scalar_cols].to_numpy(dtype=np.float32)  # [G, S]
        Y  = Lg.to_numpy(dtype=np.float32)               # [G, 2]

        G = Xl.shape[0]
        if G < seq_len:
            continue

        for start in range(0, G - seq_len + 1, stride):
            end = start + seq_len
            Xl_win = Xl[start:end]                       # [T, L]
            Xs_win = Xs[start:end]                       # [T, S]
            y_win  = Y[end - 1]                          # [2] (last step label)

            Xl_list.append(Xl_win[:, :, None])           # [T, L, 1]
            Xs_list.append(Xs_win)                       # [T, S]
            Y_list.append(y_win)

    if not Xl_list:
        raise ValueError("No sequences built. Check seq_len/stride and data lengths.")

    X_lidar_seq  = np.stack(Xl_list, axis=0)             # [N, T, L, 1]
    X_scalar_seq = np.stack(Xs_list, axis=0)             # [N, T, S]
    Y            = np.stack(Y_list, axis=0)              # [N, 2]

    return X_lidar_seq, X_scalar_seq, Y, lidar_cols, scalar_cols


# =========================
# Model
# =========================

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")  # optional speed-up

def create_cnn_lstm_model(lidar_len, scalar_len, seq_len,
                          conv_channels=(16, 32, 64),   # lighter & faster than before
                          lstm_units=128,               # this becomes GRU units
                          dense_units=(64,),
                          dropout=0.1,
                          pool_factor=4):               # 1080 -> 270 beams for big speedup

    # Normalizers (returned so you can .adapt() them)
    norm_lidar  = layers.Normalization(axis=-1, name="norm_lidar")
    norm_scalar = layers.Normalization(axis=-1, name="norm_scalar")

    lidar_seq = keras.Input(shape=(seq_len, lidar_len, 1), name="lidar_seq")
    state_seq = keras.Input(shape=(seq_len, scalar_len),   name="state_seq")

    # Per-timestep LiDAR backbone
    x = layers.TimeDistributed(layers.Reshape((lidar_len,)))(lidar_seq)
    x = layers.TimeDistributed(norm_lidar,  name="td_norm_lidar")(x)
    x = layers.TimeDistributed(layers.Reshape((lidar_len, 1)))(x)

    # Downsample beams early to cut compute
    if pool_factor and pool_factor > 1:
        x = layers.TimeDistributed(
            layers.AveragePooling1D(pool_size=pool_factor, strides=pool_factor, padding="same"),
            name="td_lidar_pool")(x)

    for i, ch in enumerate(conv_channels):
        x = layers.TimeDistributed(layers.Conv1D(
            ch, kernel_size=5,
            strides=2 if i < len(conv_channels)-1 else 1,
            padding="same"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.GlobalMaxPooling1D())(x)   # [B,T,C]

    s = layers.TimeDistributed(norm_scalar, name="td_norm_scalar")(state_seq)  # [B,T,S]

    # === GRU instead of LSTM ===
    z = layers.Concatenate()([x, s])               # [B,T,C+S]
    z = layers.GRU(lstm_units, return_sequences=False, name="gru")(z)

    for u in dense_units:
        z = layers.Dense(u, activation="relu")(z)
        if dropout and dropout > 0:
            z = layers.Dropout(dropout)(z)

    # Keep final dtype float32 when using mixed precision
    out = layers.Dense(2, dtype="float32", name="cmd_out")(z)

    model = keras.Model([lidar_seq, state_seq], out, name="LidarCNNLSTMPolicy")
    lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 10000, 0.9)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.Huber(delta=.5),
                  metrics=["mae"])
    return model, norm_lidar, norm_scalar
def large_dataset(input_directory, single_dkr_flag, adaptive_flag,
                  seq_len=10, stride=1, epochs=500, batch_size=64,
                  lstm_units=256, dropout=0.1):
    """
    Build a combined dataset from one or more data directories and train the CNN+LSTM.
    Each data_dir can already contain combined CSVs OR per-segment (seg_*) folders.
    """

    def _has_seg_subdirs(path):
        try:
            return any(os.path.isdir(os.path.join(path, d)) and d.startswith("seg_")
                       for d in os.listdir(path))
        except Exception:
            return False

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
        "model": {}
    }

    combined_features = None
    combined_labels = None
    total_rows = 0

    # --- Collect / build combined_features & combined_labels ---
    for data_dir in subdirs:

        feats_p = os.path.join(data_dir, "combined_features.csv")
        labs_p  = os.path.join(data_dir, "combined_labels.csv")

        use_existing = os.path.exists(feats_p) and os.path.exists(labs_p)
        if use_existing:
            print(f"[{data_dir}] Using existing combined CSVs")
            df_feats = pd.read_csv(feats_p, header=0)
            df_labs  = pd.read_csv(labs_p,  header=0)

            if 'group' not in df_feats.columns:
                print(f"[{data_dir}] WARNING: 'group' column missing in existing combined_features.csv")
                if _has_seg_subdirs(data_dir):
                    print(f"[{data_dir}] Rebuilding from seg_* to restore per-segment grouping.")
                    use_existing = False  # fall through to builder below
                else:
                    # Fallback: assign a single group per directory
                    df_feats = df_feats.copy()
                    df_feats['group'] = f"{os.path.basename(os.path.normpath(data_dir))}::all"
                    print(f"[{data_dir}] Assigned single group '{df_feats['group'].iat[0]}'")
        if not use_existing:
            print(f"[{data_dir}] Building combined CSVs from seg_*")
            try:
                seg_dirs_all = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
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
                group_id = f"{os.path.basename(os.path.normpath(data_dir))}::{seg}"

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
            combined_labels   = pd.concat([combined_labels,   df_labs],  axis=0, ignore_index=True)
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

    # Final safety: ensure 'group' exists
    if 'group' not in combined_features.columns:
        combined_features = combined_features.copy()
        combined_features['group'] = "ALL::fallback"
        print("[WARN] Global 'group' column was missing; assigned single fallback group.")

    yaml_data["combined"] = {
        "features_shape": {"rows": int(combined_features.shape[0]), "cols": int(combined_features.shape[1])},
        "labels_shape":   {"rows": int(combined_labels.shape[0]),   "cols": int(combined_labels.shape[1])},
        "num_datasets": len(yaml_data["datasets"]),
        "total_rows": int(total_rows),
        "num_groups": int(combined_features['group'].nunique()),
    }

    # Write initial run metadata YAML (will update later with model stats)
    meta_path = os.path.join(new_dir, "metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)
    print(f"[Meta] Wrote: {meta_path}")
    print(f"[Debug] num groups: {combined_features['group'].nunique()}")

    # =========================
    # Train / Validate Split (by group, with fallback)
    # =========================
    groups = combined_features['group']
    if groups.nunique() >= 2:
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, val_idx = next(gss.split(combined_features, groups=groups))
        print("[Split] Group-wise GroupShuffleSplit used.")
    else:
        print("[Split] Only one unique group detected; falling back to random instance split (no group).")
        n = len(combined_features)
        rng = np.random.default_rng(42)
        val_size = max(1, int(round(0.2 * n)))
        val_idx = np.sort(rng.choice(n, size=val_size, replace=False))
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = np.nonzero(mask)[0]

    train_feats = combined_features.iloc[train_idx].copy()
    val_feats   = combined_features.iloc[val_idx].copy()
    train_labs  = combined_labels.iloc[train_idx].copy()
    val_labs    = combined_labels.iloc[val_idx].copy()

    # =========================
    # Build sequences (windows don't cross 'group' boundaries)
    # =========================
    Xtr_lidar, Xtr_scalar, y_train, lidar_cols, scalar_cols = build_sequence_dataset(
        train_feats, train_labs, seq_len=seq_len, stride=stride
    )
    Xva_lidar, Xva_scalar, y_val, _, _ = build_sequence_dataset(
        val_feats,   val_labs,   seq_len=seq_len, stride=stride
    )

    Btr, T, L, _ = Xtr_lidar.shape
    S = Xtr_scalar.shape[-1]
    print(f"[Seq Shapes] Train lidar {Xtr_lidar.shape}, state {Xtr_scalar.shape}, y {y_train.shape}")
    print(f"[Seq Shapes]   Val lidar {Xva_lidar.shape}, state {Xva_scalar.shape}, y {y_val.shape}")
    print(f"[Dims] T={T}, L={L}, S={S}")

    # =========================
    # Build, Adapt, Train
    # =========================
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=10, min_delta=1e-3, restore_best_weights=True
    )
    model, norm_lidar, norm_scalar = create_cnn_lstm_model(
        lidar_len=L, scalar_len=S, seq_len=T,
        lstm_units=lstm_units, dropout=dropout
    )

    # Adapt normalization layers on flattened time
    norm_lidar.adapt(Xtr_lidar.reshape(-1, L))   # [N*T, L]
    norm_scalar.adapt(Xtr_scalar.reshape(-1, S)) # [N*T, S]

    history = model.fit(
        [Xtr_lidar, Xtr_scalar], y_train,
        validation_data=([Xva_lidar, Xva_scalar], y_val),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1
    )

    # =========================
    # Save Keras + ONNX + Graphs + Metadata
    # =========================
    model_path = os.path.join(new_dir, f"{timestamp}.keras")
    model.save(model_path)
    print(f"[Keras] Wrote: {model_path}")

    onnx_path = os.path.join(new_dir, f"{timestamp}.onnx")
    convert_keras_onnx(model_path, onnx_path, L, S, T)

    graphs(history, new_dir)

    # Update YAML with model/column info + normalization stats
    yaml_data["model"] = {
        "seq_len": int(T),
        "lidar_cols": lidar_cols,
        "scalar_cols": scalar_cols,
        "normalization": {
            "lidar_mean": norm_lidar.mean.numpy().tolist(),
            "lidar_var":  norm_lidar.variance.numpy().tolist(),
            "scalar_mean": norm_scalar.mean.numpy().tolist(),
            "scalar_var":  norm_scalar.variance.numpy().tolist(),
        },
        "architecture": {
            "conv_channels": [32, 64, 128],
            "lstm_units": int(lstm_units),
            "dense_units": [64],
            "dropout": float(dropout),
        },
        "loss": "Huber(delta=0.5)",
        "metrics": ["mae"],
        "onnx_inputs": {
            "lidar_seq": [None, int(T), int(L), 1],
            "state_seq": [None, int(T), int(S)]
        },
        "output": [None, 2]
    }
    meta_path = os.path.join(new_dir, "metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)
    print(f"[Meta] Updated: {meta_path}")

    print("\n=== DONE ===")
    print(f"Keras: {model_path}")
    print(f"ONNX : {onnx_path}")
    print(f"Graphs: {os.path.join(new_dir, 'graphs.png')}")
    print(f"Metadata: {meta_path}")
# =========================
# ONNX Export
# =========================

def convert_keras_onnx(keras_model_path, output_model_path,
                       lidar_len: int, scalar_len: int, seq_len: int):
    """
    Export a Keras CNN+LSTM with two inputs to ONNX.
    """
    m = keras.models.load_model(keras_model_path)
    sig = (
        tf.TensorSpec([None, seq_len, lidar_len, 1], tf.float32, name="lidar_seq"),
        tf.TensorSpec([None, seq_len, scalar_len],   tf.float32, name="state_seq"),
    )
    _onnx_model, _ = tf2onnx.convert.from_keras(
        m, input_signature=sig, opset=17, output_path=output_model_path
    )
    print(f"[ONNX] Wrote: {output_model_path}")


# =========================
# Pipeline
# =========================


def main():
    parser = argparse.ArgumentParser(description="CNN+LSTM (LiDAR+Scalars) Training with Fixed Sequences -> ONNX")
    parser.add_argument("input_bag", type=str, nargs='+',
                        help="Path(s) to data directories (or a parent directory with seg_* subfolders)")
    parser.add_argument("--large", action="store_true",
                        help="Multiple data directories mode (or treat single path as parent dir list)")
    parser.add_argument("--single_dkr", action='store_true',
                        help="All training data within one directory containing subdirectories")
    parser.add_argument("--adaptive", action='store_true',
                        help="Use adaptive_local_goals.csv instead of local_goals.csv")

    # Sequence/training hyperparams
    parser.add_argument("--seq_len", type=int, default=10, help="Window length T")
    parser.add_argument("--stride", type=int, default=1, help="Stride between windows")
    parser.add_argument("--epochs", type=int, default=500, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lstm_units", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout after dense layers")

    args = parser.parse_args()

    # Route to unified pipeline (large mode also covers single path lists)
    large_dataset(
        input_directory=args.input_bag,
        single_dkr_flag=args.single_dkr,
        adaptive_flag=args.adaptive,
        seq_len=args.seq_len,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        dropout=args.dropout
    )


if __name__ == "__main__":
    main()
