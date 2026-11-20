import json, os
import numpy as np
import torch
from model import GPT, GPTConfig


def export_weights(state, cfg_dict, metadata, out_dir):
    # Writes weights.bin (float32 little-endian, concatenated) + model_manifest.json
    os.makedirs(out_dir, exist_ok=True)
    tensors, chunks, offset = [], [], 0
    for name, t in state.items():
        a = t.detach().cpu().float().contiguous().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset, "length": int(a.size)})
        chunks.append(a.tobytes())
        offset += int(a.size)
    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        f.write(b"".join(chunks))
    manifest = {"format": "float32-le", "layout": "row-major; Linear.weight is [out, in]",
                "config": cfg_dict, "metadata": metadata, "tensors": tensors}
    with open(os.path.join(out_dir, "model_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def load_exported(out_dir):
    # Rebuilds a PyTorch model purely from the exported files (used for verification).
    with open(os.path.join(out_dir, "model_manifest.json")) as f:
        man = json.load(f)
    raw = np.fromfile(os.path.join(out_dir, "weights.bin"), dtype="<f4")
    state = {}
    for t in man["tensors"]:
        arr = raw[t["offset"]:t["offset"] + t["length"]].copy()
        state[t["name"]] = torch.from_numpy(arr).reshape(t["shape"])
    model = GPT(GPTConfig(**man["config"]))
    model.load_state_dict(state)
    model.eval()
    return model, man
