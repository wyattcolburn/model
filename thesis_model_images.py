
#!/usr/bin/env python3
import os, sys
from tensorflow import keras
from tensorflow.keras.layers import GlobalMaxPooling1D, Normalization, Concatenate

# ---------- helpers ----------
def src_inputs(tensor):
    return keras.utils.get_source_inputs(tensor)

def find_lidar_tip(m):
    pools = [l for l in m.layers if isinstance(l, GlobalMaxPooling1D)]
    for layer in reversed(pools):
        if any("lidar" in t.name for t in src_inputs(layer.output)):
            return layer
    return pools[-1] if pools else None

def find_state_tip(m):
    norms = [l for l in m.layers if isinstance(l, Normalization)]
    for l in norms:
        if l.name == "norm_scalar":
            return l
    for l in norms:
        if any("state" in t.name for t in src_inputs(l.output)):
            return l
    return norms[-1] if norms else None

def find_concat(m):
    concats = [l for l in m.layers if isinstance(l, Concatenate)]
    for l in concats:
        names = ",".join(t.name for t in src_inputs(l.output))
        if ("lidar" in names) and ("state" in names):
            return l
    return concats[-1] if concats else None

def find_output_tensor(m):
    try:
        return m.get_layer("cmd_out").output
    except Exception:
        return m.outputs[0]

def save_svg(model, path):
    keras.utils.plot_model(model, to_file=path, show_shapes=True, rankdir="LR", dpi=220)

# ---------- main ----------
def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.keras"
    model = keras.models.load_model(model_path, compile=False)

    lidar_tip  = find_lidar_tip(model)
    state_tip  = find_state_tip(model)
    concat     = find_concat(model)
    out_tensor = find_output_tensor(model)

    if not (lidar_tip and state_tip and concat):
        raise RuntimeError("Could not locate lidar tip, state tip, or concatenate layer.")

    # Branch submodels (Option B): infer correct Input tensors from outputs
    lidar_branch = keras.Model(inputs=src_inputs(lidar_tip.output),
                               outputs=lidar_tip.output,
                               name="lidar_branch")

    state_branch = keras.Model(inputs=src_inputs(state_tip.output),
                               outputs=state_tip.output,
                               name="state_branch")

    # Head (fusion → commands)
    head_model = keras.Model(inputs=model.inputs, outputs=out_tensor, name="policy_head")

    outdir = os.path.dirname(os.path.abspath(model_path)) or "."
    save_svg(lidar_branch, os.path.join(outdir, "lidar_branch.png"))
    save_svg(state_branch, os.path.join(outdir, "state_branch.png"))
    save_svg(head_model,  os.path.join(outdir, "policy_head.png"))

    print("Wrote:")
    print(" ", os.path.join(outdir, "lidar_branch.svg"))
    print(" ", os.path.join(outdir, "state_branch.svg"))
    print(" ", os.path.join(outdir, "policy_head.svg"))

if __name__ == "__main__":
    main()
