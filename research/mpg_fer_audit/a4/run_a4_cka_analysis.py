"""A4.1 Representation Similarity: Linear CKA Analysis.

Computes:
1. Pooled Linear CKA across Train, Public, Private:
   - Pixel readout [128]
   - Motif readout [384]
   - Fusion [512]
   - classifier hidden [256]
   - logits [7]
2. Node-level Linear CKA across layers PRE, L1, L2, L3, L4, L5 for 1024 Public & Private subsets.
   Reshaped as observations: [N_samples * 49, 192].
3. 6x6 Cross-layer CKA matrices for Public and Private.
4. Saves:
   - a4_representation_cka.json
   - a4_cross_layer_cka.csv
   - a4_cka_pooled.png
   - a4_cka_cross_layer_public.png
   - a4_cka_cross_layer_private.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"


import torch

def linear_cka(X: np.ndarray | torch.Tensor, Y: np.ndarray | torch.Tensor) -> float:
    """Compute linear CKA using PyTorch with GPU acceleration if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    else:
        X = X.to(device=device, dtype=torch.float32)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
    else:
        Y = Y.to(device=device, dtype=torch.float32)

    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)

    XtY = X_c.t() @ Y_c
    hsic_xy = (XtY ** 2).sum()

    XtX = X_c.t() @ X_c
    hsic_xx = (XtX ** 2).sum()

    YtY = Y_c.t() @ Y_c
    hsic_yy = (YtY ** 2).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom.item() <= 1e-12:
        return 0.0
    return float((hsic_xy / denom).item())


def main():
    print("Computing A4.1 Linear CKA Representation Similarity...", flush=True)

    train_pooled = np.load(AUDIT_DIR / "train_pooled_features.npz")
    pub_pooled = np.load(AUDIT_DIR / "public_pooled_features.npz")
    priv_pooled = np.load(AUDIT_DIR / "private_pooled_features.npz")

    pub_node = np.load(AUDIT_DIR / "public_1024_node_features.npz")
    priv_node = np.load(AUDIT_DIR / "private_1024_node_features.npz")

    # 1. Pooled CKA
    pooled_keys = [
        ("pixel_readout", "v21_orig_r0", "v22_orig_r0"),
        ("motif_readout", "v21_orig_r7", "v22_orig_r7"),
        ("fusion", "v21_orig_r8", "v22_orig_r8"),
        ("classifier_hidden", "v21_orig_r9", "v22_orig_r9"),
        ("logits", "v21_orig_r10", "v22_orig_r10"),
    ]

    pooled_cka = {}
    for name, k21, k22 in pooled_keys:
        cka_train = linear_cka(train_pooled[k21], train_pooled[k22])
        cka_pub = linear_cka(pub_pooled[k21], pub_pooled[k22])
        cka_priv = linear_cka(priv_pooled[k21], priv_pooled[k22])
        pooled_cka[name] = {
            "train": cka_train,
            "public": cka_pub,
            "private": cka_priv,
        }
        print(f"Pooled CKA - {name:<18}: Train={cka_train:.4f}, Public={cka_pub:.4f}, Private={cka_priv:.4f}", flush=True)

    # 2. Node-level within-layer CKA
    node_layer_names = ["PRE", "L1", "L2", "L3", "L4", "L5"]
    node_layer_keys = [f"R{i}" for i in range(1, 7)]

    node_within_cka = {}
    for name, r_k in zip(node_layer_names, node_layer_keys):
        # Public
        # Shape: [1024, 49, 192] -> reshape [1024*49, 192]
        x_pub = pub_node[f"v21_{r_k}"].reshape(-1, 192)
        y_pub = pub_node[f"v22_{r_k}"].reshape(-1, 192)
        cka_pub = linear_cka(x_pub, y_pub)

        # Private
        x_priv = priv_node[f"v21_{r_k}"].reshape(-1, 192)
        y_priv = priv_node[f"v22_{r_k}"].reshape(-1, 192)
        cka_priv = linear_cka(x_priv, y_priv)

        node_within_cka[name] = {
            "public": cka_pub,
            "private": cka_priv,
        }
        print(f"Node-level CKA - {name:<4}: Public={cka_pub:.4f}, Private={cka_priv:.4f}", flush=True)

    # 3. 6x6 Cross-Layer CKA Matrices
    print("Computing cross-layer matrices...", flush=True)
    def compute_cross_layer_matrix(node_data):
        mat = np.zeros((6, 6), dtype=np.float64)
        for i, k21 in enumerate(node_layer_keys):
            x = node_data[f"v21_{k21}"].reshape(-1, 192)
            for j, k22 in enumerate(node_layer_keys):
                y = node_data[f"v22_{k22}"].reshape(-1, 192)
                mat[i, j] = linear_cka(x, y)
        return mat

    pub_matrix = compute_cross_layer_matrix(pub_node)
    priv_matrix = compute_cross_layer_matrix(priv_node)

    # Save CSV
    csv_path = AUDIT_DIR / "a4_cross_layer_cka.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "v21_layer", "v22_layer", "linear_cka"])
        for i, l21 in enumerate(node_layer_names):
            for j, l22 in enumerate(node_layer_names):
                writer.writerow(["public", l21, l22, f"{pub_matrix[i, j]:.6f}"])
                writer.writerow(["private", l21, l22, f"{priv_matrix[i, j]:.6f}"])
    print(f"Saved {csv_path}")

    # Save JSON summary
    out_json = {
        "pooled_cka": pooled_cka,
        "node_level_within_layer_cka": node_within_cka,
        "cross_layer_matrix_public": {
            l21: {l22: float(pub_matrix[i, j]) for j, l22 in enumerate(node_layer_names)}
            for i, l21 in enumerate(node_layer_names)
        },
        "cross_layer_matrix_private": {
            l21: {l22: float(priv_matrix[i, j]) for j, l22 in enumerate(node_layer_names)}
            for i, l21 in enumerate(node_layer_names)
        },
    }
    json_path = AUDIT_DIR / "a4_representation_cka.json"
    json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")

    # Generate Plots
    # 1. Pooled CKA bar plot
    plt.figure(figsize=(9, 5))
    x_pos = np.arange(len(pooled_keys))
    width = 0.25

    names = [k[0] for k in pooled_keys]
    train_vals = [pooled_cka[k]["train"] for k in names]
    pub_vals = [pooled_cka[k]["public"] for k in names]
    priv_vals = [pooled_cka[k]["private"] for k in names]

    plt.bar(x_pos - width, train_vals, width, label="Train", color="#2b5c8f")
    plt.bar(x_pos, pub_vals, width, label="Public", color="#4682b4")
    plt.bar(x_pos + width, priv_vals, width, label="Private", color="#87ceeb")

    plt.xlabel("Representation", fontsize=11, fontweight="bold")
    plt.ylabel("Linear CKA (v2.1 vs v2.2)", fontsize=11, fontweight="bold")
    plt.title("A4.1: Pooled Representation Similarity (Linear CKA)", fontsize=12, fontweight="bold")
    plt.xticks(x_pos, ["Pixel", "Motif", "Fusion", "Classifier H", "Logits"], fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(AUDIT_DIR / "a4_cka_pooled.png", dpi=150)
    plt.close()
    print("Saved a4_cka_pooled.png")

    # 2. Cross-layer heatmaps
    for split_label, mat, fname in [
        ("Public", pub_matrix, "a4_cka_cross_layer_public.png"),
        ("Private", priv_matrix, "a4_cka_cross_layer_private.png"),
    ]:
        plt.figure(figsize=(7, 6))
        plt.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Linear CKA")
        plt.xticks(range(6), node_layer_names, fontsize=10)
        plt.yticks(range(6), node_layer_names, fontsize=10)
        plt.xlabel("v2.2 Sparse Layer", fontsize=11, fontweight="bold")
        plt.ylabel("v2.1 Dense Layer", fontsize=11, fontweight="bold")
        plt.title(f"Cross-Layer CKA Matrix ({split_label} 1024 Subset)", fontsize=12, fontweight="bold")

        for r in range(6):
            for c in range(6):
                val = mat[r, c]
                color = "white" if val < 0.6 else "black"
                plt.text(c, r, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(AUDIT_DIR / fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()
