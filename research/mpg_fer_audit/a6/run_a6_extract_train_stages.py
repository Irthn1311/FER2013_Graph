"""Extract Train stages R1..R6 for v2.1 and v2.2 sequentially to maintain peak GPU efficiency.

Saves:
- a6_train_v21_stages.npz
- a6_train_v22_stages.npz
- a6_train_features.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6"
A4_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"

V21_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"
V22_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src"

sys.path.insert(0, str(V21_SRC))
sys.path.insert(0, str(V22_SRC))

from mpg_fer_v2_1.config import MPGConfig as MPGConfigV21
from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path
from mpg_fer_v2_1.model import MPGFER as MPGFERV21

from mpg_fer_v2_2.config import MPGConfig as MPGConfigV22
from mpg_fer_v2_2.model import MPGFER as MPGFERV22

V21_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "best_val_acc.pt"
V22_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "best_val_acc.pt"


def extract_model_train(model_key: str, device: str):
    t0 = time.time()
    out_file = AUDIT_DIR / f"a6_train_{model_key}_stages.npz"
    if out_file.exists():
        print(f"File {out_file} already exists; skipping extraction.", flush=True)
        return

    print(f"\n--- Extracting Train stages for {model_key} on {device} ---", flush=True)
    if model_key == "v21":
        model = MPGFERV21(MPGConfigV21()).to(device).eval()
        ckpt = torch.load(V21_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict(ckpt, strict=True)
    else:
        model = MPGFERV22(MPGConfigV22()).to(device).eval()
        ckpt = torch.load(V22_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict(ckpt, strict=True)

    captured = {}
    def make_hook(key):
        return lambda m, inp, out: captured.update({key: (out[0] if isinstance(out, tuple) else out).detach()})

    model.motif_composer.register_forward_hook(make_hook("R1"))
    for l_i in range(5):
        model.motif_gnn[l_i].register_forward_hook(make_hook(f"R{l_i+2}"))

    train_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "train.csv", "train"), split="train", augment=False)
    loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    reps = {f"R{l}": [] for l in range(1, 7)}

    with torch.no_grad():
        for b_i, (x, y) in enumerate(loader):
            if (b_i + 1) % 50 == 0 or (b_i + 1) == len(loader):
                print(f"  {model_key} Batch {b_i + 1}/{len(loader)} in {time.time() - t0:.1f}s", flush=True)
            x = x.to(device)
            captured.clear()
            _ = model(x)
            for l in range(1, 7):
                reps[f"R{l}"].append(captured[f"R{l}"].mean(dim=1).cpu().numpy().astype(np.float32))

    save_dict = {f"R{l}": np.concatenate(reps[f"R{l}"], axis=0) for l in range(1, 7)}
    save_dict["targets"] = train_ds.labels.astype(np.int64)
    np.savez_compressed(out_file, **save_dict)
    print(f"Saved {out_file} in {time.time() - t0:.1f}s.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "v21", "v22"], default="all")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model in ["all", "v21"]:
        extract_model_train("v21", device)
        torch.cuda.empty_cache()

    if args.model in ["all", "v22"]:
        extract_model_train("v22", device)
        torch.cuda.empty_cache()

    # Merge into a6_train_features.npz if both exist
    v21_f = AUDIT_DIR / "a6_train_v21_stages.npz"
    v22_f = AUDIT_DIR / "a6_train_v22_stages.npz"

    if v21_f.exists() and v22_f.exists():
        print("\nMerging into a6_train_features.npz...", flush=True)
        t21 = np.load(v21_f)
        t22 = np.load(v22_f)
        train_a4 = np.load(A4_DIR / "train_pooled_features.npz")

        merged = {
            "targets": t21["targets"],
            "v21_R0": train_a4["v21_orig_r0"],
            "v21_R7": train_a4["v21_orig_r7"],
            "v21_R8": train_a4["v21_orig_r8"],
            "v21_R9": train_a4["v21_orig_r9"],
            "v21_R10": train_a4["v21_orig_r10"],
            "v22_R0": train_a4["v22_orig_r0"],
            "v22_R7": train_a4["v22_orig_r7"],
            "v22_R8": train_a4["v22_orig_r8"],
            "v22_R9": train_a4["v22_orig_r9"],
            "v22_R10": train_a4["v22_orig_r10"],
        }
        for l in range(1, 7):
            merged[f"v21_R{l}"] = t21[f"R{l}"]
            merged[f"v22_R{l}"] = t22[f"R{l}"]

        out_merged = AUDIT_DIR / "a6_train_features.npz"
        np.savez_compressed(out_merged, **merged)
        print(f"Saved {out_merged} successfully!", flush=True)


if __name__ == "__main__":
    main()
