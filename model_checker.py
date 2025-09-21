
# make_sample_batch.py
import os, yaml, numpy as np, pandas as pd
from pathlib import Path

run_dir = Path("data_set/2025_09_19_22_51")  # <- your run
with open(run_dir / "metadata.yaml") as f:
    meta = yaml.safe_load(f)

T = int(meta["model"]["seq_len"])
L = int(meta["model"]["onnx_inputs"]["lidar_seq"][2])
S = int(meta["model"]["onnx_inputs"]["state_seq"][2])

# Pick any dataset dir that has combined CSVs
ds = meta["datasets"][0]
feats = pd.read_csv(ds["features_csv"])
labs  = pd.read_csv(ds["labels_csv"])

# Ensure there's a 'group' column; if not, make one
if "group" not in feats.columns:
    feats["group"] = "ALL::fallback"

lidar_cols  = [c for c in feats.columns if c.startswith("lidar_")]
scalar_cols = [c for c in feats.columns if c not in lidar_cols + ["group"]]

Xl_list, Xs_list, Y_list = [], [], []
for gid, idx in feats.groupby("group").groups.items():
    Fg = feats.loc[idx].reset_index(drop=True)
    Lg = labs.loc[idx].reset_index(drop=True)
    Xl = Fg[lidar_cols].to_numpy(np.float32)
    Xs = Fg[scalar_cols].to_numpy(np.float32)
    Y  = Lg.to_numpy(np.float32)

    G = len(Fg)
    if G < T: continue
    for start in range(0, min(G - T + 1, 1000), T):  # take a few windows
        end = start + T
        Xl_list.append(Xl[start:end][:, :, None])
        Xs_list.append(Xs[start:end])
        Y_list.append(Y[end-1])

lidar = np.stack(Xl_list).astype(np.float32)   # [N,T,L,1]
state = np.stack(Xs_list).astype(np.float32)   # [N,T,S]
y     = np.stack(Y_list).astype(np.float32)    # [N,2]

np.save(run_dir / "val_lidar.npy", lidar)
np.save(run_dir / "val_state.npy", state)
np.save(run_dir / "val_y.npy", y)
print("Wrote sample npys to:", run_dir)
