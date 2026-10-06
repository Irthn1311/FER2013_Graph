"""A6.1 Stage-Wise Representation Extraction (R0-R10).

Extracts matched representations for v2.1 and v2.2 under canonical raw single-view FP32 inference:
R0  : Pixel readout [128]
R1  : PRE-Motif-Graph node states mean-pooled [192]
R2  : Motif layer 1 output mean-pooled [192]
R3  : Motif layer 2 output mean-pooled [192]
R4  : Motif layer 3 output mean-pooled [192]
R5  : Motif layer 4 output mean-pooled [192]
R6  : Motif layer 5 output mean-pooled [192]
R7  : Motif readout [384]
R8  : Fusion [512]
R9  : Classifier hidden [256]
R10 : Final logits [7]

Retains full [N, 49, 192] node tensors for model-resolvable samples.
Verifies hooked vs unhooked logits max absolute difference = 0.0.

Saves:
- a6_train_features.npz
- a6_public_features.npz
- a6_private_features.npz
- a6_resolvable_node_features.npz
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running A6.1 Representation Extraction on {device}...", flush=True)

    # 1. Load models
    m21 = MPGFERV21(MPGConfigV21()).to(device).eval()
    m22 = MPGFERV22(MPGConfigV22()).to(device).eval()

    c21 = torch.load(V21_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    c22 = torch.load(V22_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]

    m21.load_state_dict(c21, strict=True)
    m22.load_state_dict(c22, strict=True)

    # 2. Hook registration & verification
    test_x = torch.randn(2, 1, 48, 48, device=device)
    unhooked_21, _ = m21(test_x)
    unhooked_22, _ = m22(test_x)

    cap21 = {}
    cap22 = {}

    def make_hook(target_dict, key):
        return lambda m, inp, out: target_dict.update({key: (out[0] if isinstance(out, tuple) else out).detach()})
    def make_pre_hook(target_dict, key):
        return lambda m, inp: target_dict.update({key: inp[0].detach()})

    for model, cap_dict in [(m21, cap21), (m22, cap22)]:
        model.pixel_readout_proj.register_forward_hook(make_hook(cap_dict, "R0"))
        model.motif_composer.register_forward_hook(make_hook(cap_dict, "R1"))
        for l_i in range(5):
            model.motif_gnn[l_i].register_forward_hook(make_hook(cap_dict, f"R{l_i+2}"))
        model.motif_readout_proj.register_forward_hook(make_hook(cap_dict, "R7"))
        model.classifier.register_forward_pre_hook(make_pre_hook(cap_dict, "R8"))
        model.classifier[2].register_forward_hook(make_hook(cap_dict, "R9"))
        model.classifier.register_forward_hook(make_hook(cap_dict, "R10"))

    hooked_21, _ = m21(test_x)
    hooked_22, _ = m22(test_x)
    diff21 = (hooked_21 - unhooked_21).abs().max().item()
    diff22 = (hooked_22 - unhooked_22).abs().max().item()
    print(f"Hook Verification: v2.1 max logit diff = {diff21:.2e}, v2.2 max logit diff = {diff22:.2e}")
    if diff21 != 0.0 or diff22 != 0.0:
        print("STOP: A6_BLOCKED_HOOK_VALIDATION")
        sys.exit(2)

    # 3. Load sample groups to identify model-resolvable sample indices
    groups_df = pd.read_csv(AUDIT_DIR / "a6_sample_groups.csv")
    pub_res_idx = set(groups_df[(groups_df["split"] == "public") & (groups_df["is_model_resolvable"])]["row_index"])
    priv_res_idx = set(groups_df[(groups_df["split"] == "private") & (groups_df["is_model_resolvable"])]["row_index"])
    print(f"Model-resolvable indices to retain full node tensors: {len(pub_res_idx)} Public, {len(priv_res_idx)} Private.")

    # 4. Extract Public and Private splits (skip if already generated)
    pub_npz_path = AUDIT_DIR / "a6_public_features.npz"
    priv_npz_path = AUDIT_DIR / "a6_private_features.npz"
    res_npz_path = AUDIT_DIR / "a6_resolvable_node_features.npz"

    if pub_npz_path.exists() and priv_npz_path.exists() and res_npz_path.exists():
        print("Public, Private, and Resolvable node NPZs already exist; skipping test extraction.", flush=True)
    else:
        val_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val"), split="val", augment=False)
        test_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test"), split="test", augment=False)

        resolvable_nodes = {
            "public": {f"v21_R{l}": [] for l in range(1, 7)} | {f"v22_R{l}": [] for l in range(1, 7)},
            "private": {f"v21_R{l}": [] for l in range(1, 7)} | {f"v22_R{l}": [] for l in range(1, 7)},
        }
        resolvable_meta = {"public": [], "private": []}

        for split_name, ds, res_set in [("public", val_ds, pub_res_idx), ("private", test_ds, priv_res_idx)]:
            split_t0 = time.time()
            print(f"\nExtracting representations for {split_name.upper()} ({len(ds)} samples)...", flush=True)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

            v21_reps = {f"R{i}": [] for i in range(11)}
            v22_reps = {f"R{i}": [] for i in range(11)}

            global_idx = 0
            with torch.no_grad():
                for b_i, (x, y) in enumerate(loader):
                    b_size = len(y)
                    x = x.to(device)

                    # v2.1
                    cap21.clear()
                    _ = m21(x)
                    # v2.2
                    cap22.clear()
                    _ = m22(x)

                    # Store R0, R7, R8, R9, R10
                    for r_k in ["R0", "R7", "R8", "R9", "R10"]:
                        v21_reps[r_k].append(cap21[r_k].cpu().numpy().astype(np.float32))
                        v22_reps[r_k].append(cap22[r_k].cpu().numpy().astype(np.float32))

                    # For R1..R6: mean pool over 49 nodes: [B, 192]
                    for l in range(1, 7):
                        t21_node = cap21[f"R{l}"]  # [B, 49, 192]
                        t22_node = cap22[f"R{l}"]  # [B, 49, 192]

                        v21_reps[f"R{l}"].append(t21_node.mean(dim=1).cpu().numpy().astype(np.float32))
                        v22_reps[f"R{l}"].append(t22_node.mean(dim=1).cpu().numpy().astype(np.float32))

                        # Retain full node tensors for model-resolvable subset
                        for i_sub in range(b_size):
                            cur_row = global_idx + i_sub
                            if cur_row in res_set:
                                resolvable_nodes[split_name][f"v21_R{l}"].append(t21_node[i_sub].cpu().numpy().astype(np.float32))
                                resolvable_nodes[split_name][f"v22_R{l}"].append(t22_node[i_sub].cpu().numpy().astype(np.float32))
                                if l == 1:
                                    resolvable_meta[split_name].append({
                                        "split": split_name,
                                        "row_index": cur_row,
                                        "true_label": int(y[i_sub]),
                                    })

                    global_idx += b_size

            # Concatenate and save split NPZ
            split_save_dict = {"targets": ds.labels.astype(np.int64)}
            for r_k in [f"R{i}" for i in range(11)]:
                split_save_dict[f"v21_{r_k}"] = np.concatenate(v21_reps[r_k], axis=0)
                split_save_dict[f"v22_{r_k}"] = np.concatenate(v22_reps[r_k], axis=0)

            out_split_npz = AUDIT_DIR / f"a6_{split_name}_features.npz"
            np.savez_compressed(out_split_npz, **split_save_dict)
            print(f"Saved {out_split_npz} in {time.time() - split_t0:.1f}s.")

        # Save resolvable node features NPZ
        res_node_save_dict = {}
        for s_name in ["public", "private"]:
            for k, v in resolvable_nodes[s_name].items():
                res_node_save_dict[f"{s_name}_{k}"] = np.stack(v, axis=0)
            res_node_save_dict[f"{s_name}_meta_row_indices"] = np.array([m["row_index"] for m in resolvable_meta[s_name]], dtype=np.int64)
            res_node_save_dict[f"{s_name}_meta_targets"] = np.array([m["true_label"] for m in resolvable_meta[s_name]], dtype=np.int64)

        out_res_npz = AUDIT_DIR / "a6_resolvable_node_features.npz"
        np.savez_compressed(out_res_npz, **res_node_save_dict)
        print(f"Saved {out_res_npz} (Full [N_res, 49, 192] node tensors).")

    # 5. Extract Train representations (R1..R6 mean-pooled; reuse R0, R7, R8, R9 from A4 cache)
    train_t0 = time.time()
    print("\nExtracting Train split representations (28,709 samples)...", flush=True)
    train_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "train.csv", "train"), split="train", augment=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    tr_v21_node_pooled = {f"R{l}": [] for l in range(1, 7)}
    tr_v22_node_pooled = {f"R{l}": [] for l in range(1, 7)}

    with torch.no_grad():
        for b_i, (x, y) in enumerate(train_loader):
            if (b_i + 1) % 50 == 0 or (b_i + 1) == len(train_loader):
                print(f"  Train Batch {b_i + 1}/{len(train_loader)} in {time.time() - train_t0:.1f}s", flush=True)
            x = x.to(device)

            cap21.clear()
            _ = m21(x)
            cap22.clear()
            _ = m22(x)

            for l in range(1, 7):
                tr_v21_node_pooled[f"R{l}"].append(cap21[f"R{l}"].mean(dim=1).cpu().numpy().astype(np.float32))
                tr_v22_node_pooled[f"R{l}"].append(cap22[f"R{l}"].mean(dim=1).cpu().numpy().astype(np.float32))

    # Load pooled features from A4
    train_a4 = np.load(A4_DIR / "train_pooled_features.npz")

    train_save_dict = {
        "targets": train_ds.labels.astype(np.int64),
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
        train_save_dict[f"v21_R{l}"] = np.concatenate(tr_v21_node_pooled[f"R{l}"], axis=0)
        train_save_dict[f"v22_R{l}"] = np.concatenate(tr_v22_node_pooled[f"R{l}"], axis=0)

    out_train_npz = AUDIT_DIR / "a6_train_features.npz"
    np.savez_compressed(out_train_npz, **train_save_dict)
    print(f"Saved {out_train_npz} in {time.time() - train_t0:.1f}s.")

    print(f"\nA6.1 finished successfully in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
