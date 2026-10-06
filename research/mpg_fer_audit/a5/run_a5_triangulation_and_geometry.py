"""A5.4 Independent Neighborhood Triangulation, Confusion-Pair Audit, and Class Geometry Revisit.

Compares:
1. Learned Fusion space (v2.1 and v2.2) vs. Non-learned Raw Pixel and Gradient PCA spaces.
2. 5-NN true-label purity and entropy across consensus categories.
3. Confusion-pair analysis on primary hard pairs (symmetric check).
4. Class separability revisit: learned vs. raw vs. gradient centroid margins for Angry, Fear, Sad, Neutral.

Produces:
- a5_neighborhood_triangulation.json
- a5_confusion_pair_analysis.json
- a5_nonlearned_class_geometry.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a5"
A4_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def compute_5nn_stats(query_feats: torch.Tensor, train_feats: torch.Tensor, query_y: torch.Tensor, train_y: torch.Tensor):
    """Compute 5-NN label purity, modal label, and entropy against Train reference."""
    k = 5
    # L2-normalize
    q_norm = F.normalize(query_feats, dim=-1)
    tr_norm = F.normalize(train_feats, dim=-1)

    purities = []
    modal_labels = []
    entropies = []

    for start_i in range(0, len(q_norm), 512):
        end_i = min(start_i + 512, len(q_norm))
        sim = q_norm[start_i:end_i] @ tr_norm.t()  # [B_sub, N_tr]
        topk = torch.topk(sim, k=k, dim=-1)
        nn_labels = train_y[topk.indices]  # [B_sub, 5]
        targets_batch = query_y[start_i:end_i].unsqueeze(1)

        batch_purity = (nn_labels == targets_batch).float().mean(dim=-1).cpu().numpy()
        purities.extend(batch_purity.tolist())

        for row_lbls in nn_labels.cpu().numpy():
            counts = np.bincount(row_lbls, minlength=7)
            modal = int(np.argmax(counts))
            modal_labels.append(modal)
            probs = counts[counts > 0] / float(k)
            ent = float(-np.sum(probs * np.log(probs)))
            entropies.append(ent)

    return np.array(purities), np.array(modal_labels), np.array(entropies)


def compute_centroid_margins(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray):
    """Standardize features with Train, compute Train centroids, and return test sample margins."""
    mean = train_X.mean(axis=0, keepdims=True)
    std = np.clip(train_X.std(axis=0, keepdims=True), 1e-6, None)

    tr_std = (train_X - mean) / std
    te_std = (test_X - mean) / std

    centroids = np.zeros((7, train_X.shape[1]), dtype=np.float64)
    for c in range(7):
        centroids[c] = tr_std[train_y == c].mean(axis=0)

    # Test sample margins: min_{c != y} ||x - mu_c|| - ||x - mu_y||
    margins = np.zeros(len(test_y), dtype=np.float64)
    dists = np.zeros((len(test_y), 7), dtype=np.float64)

    for c in range(7):
        diff = te_std - centroids[c]
        dists[:, c] = np.linalg.norm(diff, axis=-1)

    for i in range(len(test_y)):
        y = test_y[i]
        d_true = dists[i, y]
        d_other = np.min([dists[i, c] for c in range(7) if c != y])
        margins[i] = d_other - d_true

    classwise = {}
    for c in range(7):
        c_name = CLASS_NAMES[c]
        c_mask = (test_y == c)
        classwise[c_name] = float(np.mean(margins[c_mask]))

    return float(np.mean(margins)), classwise, dists


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running A5.4 Neighborhood Triangulation and Geometry Revisit on {device}...", flush=True)

    # 1. Load data
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"))
    from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path

    train_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "train.csv", "train"), split="train")
    val_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val"), split="val")
    test_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test"), split="test")

    # Load multi-model features from A5.2
    multi_data = np.load(AUDIT_DIR / "a5_multi_model_predictions.npz")
    # Load train pooled features from A4
    train_pooled = np.load(A4_DIR / "train_pooled_features.npz")
    train_y = torch.from_numpy(train_ds.labels).to(device)

    # Prep raw pixel arrays (centered)
    tr_raw = train_ds.images.reshape(len(train_ds), -1).astype(np.float32) / 255.0
    pub_raw = val_ds.images.reshape(len(val_ds), -1).astype(np.float32) / 255.0
    priv_raw = test_ds.images.reshape(len(test_ds), -1).astype(np.float32) / 255.0

    # Prep gradient descriptors + PCA
    from scipy.ndimage import convolve
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 4.0
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 4.0
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    def get_grad(imgs):
        n = len(imgs)
        feats = np.zeros((n, 4 * 48 * 48), dtype=np.float32)
        for i in range(n):
            img_f = imgs[i].astype(np.float32) / 255.0
            gx = convolve(img_f, sobel_x, mode="reflect")
            gy = convolve(img_f, sobel_y, mode="reflect")
            gmag = np.sqrt(gx**2 + gy**2)
            lap = convolve(img_f, laplacian, mode="reflect")
            feats[i] = np.concatenate([gx.flatten(), gy.flatten(), gmag.flatten(), lap.flatten()])
        return feats

    print("Computing gradient descriptors...", flush=True)
    tr_g = get_grad(train_ds.images)
    pub_g = get_grad(val_ds.images)
    priv_g = get_grad(test_ds.images)

    # PCA 128
    torch.manual_seed(42)
    tr_g_t = torch.from_numpy(tr_g).to(device)
    U, S, V = torch.pca_lowrank(tr_g_t, q=128, center=True)
    g_mean = tr_g_t.mean(dim=0, keepdim=True)

    tr_grad_pca = ((tr_g_t - g_mean) @ V).cpu().numpy()
    pub_grad_pca = ((torch.from_numpy(pub_g).to(device) - g_mean) @ V).cpu().numpy()
    priv_grad_pca = ((torch.from_numpy(priv_g).to(device) - g_mean) @ V).cpu().numpy()

    # Load consensus CSV rows
    consensus_csv = AUDIT_DIR / "a5_model_consensus.csv"
    import csv
    with consensus_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_consensus_rows = list(reader)

    pub_consensus = [r for r in all_consensus_rows if r["split"] == "public"]
    priv_consensus = [r for r in all_consensus_rows if r["split"] == "private"]

    # =========================================================================
    # PART 1: 5-NN NEIGHBORHOOD TRIANGULATION ACROSS 4 SPACES
    # =========================================================================
    print("\n--- Part 1: 5-NN Neighborhood Triangulation Across 4 Spaces ---", flush=True)

    # 4 spaces:
    # 1. v2.1 Fusion (512d)
    # 2. v2.2 Fusion (512d)
    # 3. Non-learned Raw Pixel (2304d)
    # 4. Non-learned Gradient PCA (128d)

    spaces = {
        "v2_1_fusion": (
            torch.from_numpy(train_pooled["v21_orig_r8"]).to(device),
            lambda split: torch.from_numpy(multi_data[f"{split}_v2_1_fusion"]).to(device)
        ),
        "v2_2_fusion": (
            torch.from_numpy(train_pooled["v22_orig_r8"]).to(device),
            lambda split: torch.from_numpy(multi_data[f"{split}_v2_2_fusion"]).to(device)
        ),
        "raw_pixel": (
            torch.from_numpy(tr_raw).to(device),
            lambda split: torch.from_numpy(pub_raw if split == "public" else priv_raw).to(device)
        ),
        "gradient_pca": (
            torch.from_numpy(tr_grad_pca).to(device),
            lambda split: torch.from_numpy(pub_grad_pca if split == "public" else priv_grad_pca).to(device)
        ),
    }

    triangulation_summary = {}

    for split_name, ds, c_rows in [("public", val_ds, pub_consensus), ("private", test_ds, priv_consensus)]:
        triangulation_summary[split_name] = {}
        query_y = torch.from_numpy(ds.labels).to(device)
        cats = [r["category"] for r in c_rows]

        for s_name, (tr_t, q_func) in spaces.items():
            q_t = q_func(split_name)
            purity, modal_lbls, entropy = compute_5nn_stats(q_t, tr_t, query_y, train_y)

            # Break down by consensus categories
            group_breakdown = {}
            for cat_k in [
                "ALL_CORRECT", "MOSTLY_CORRECT", "SPLIT_DECISION",
                "ALL_WRONG_SAME_LABEL", "ALL_WRONG_MIXED_LABEL",
                "HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL"
            ]:
                mask = np.array([c == cat_k for c in cats])
                cnt = int(np.sum(mask))
                if cnt > 0:
                    group_breakdown[cat_k] = {
                        "count": cnt,
                        "mean_purity": float(np.mean(purity[mask])),
                        "mean_entropy": float(np.mean(entropy[mask])),
                        "modal_matches_true_fraction": float(np.mean(modal_lbls[mask] == ds.labels[mask])),
                    }
                else:
                    group_breakdown[cat_k] = {"count": 0, "mean_purity": 0.0, "mean_entropy": 0.0, "modal_matches_true_fraction": 0.0}

            triangulation_summary[split_name][s_name] = group_breakdown
            print(f"  {split_name.upper()} | {s_name:<16} | ALL_CORRECT Purity={group_breakdown['ALL_CORRECT']['mean_purity']:.3f} | HIGH_CONF_WRONG Purity={group_breakdown['HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL']['mean_purity']:.3f}")

    (AUDIT_DIR / "a5_neighborhood_triangulation.json").write_text(json.dumps(triangulation_summary, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a5_neighborhood_triangulation.json'}")

    # =========================================================================
    # PART 2: CONFUSION-PAIR ANALYSIS (PRIMARY HARD EMOTIONS)
    # =========================================================================
    print("\n--- Part 2: Confusion-Pair Analysis ---", flush=True)

    target_pairs = [
        ("Fear", "Sad"), ("Sad", "Fear"),
        ("Fear", "Neutral"), ("Neutral", "Fear"),
        ("Fear", "Angry"), ("Angry", "Fear"),
        ("Sad", "Neutral"), ("Neutral", "Sad"),
        ("Sad", "Angry"), ("Angry", "Sad"),
    ]

    confusion_summary = {}

    for split_name, ds, c_rows in [("public", val_ds, pub_consensus), ("private", test_ds, priv_consensus)]:
        confusion_summary[split_name] = {}
        targets = ds.labels

        # Model predictions
        p21 = np.array([int(r["v21_pred"]) for r in c_rows])
        p22 = np.array([int(r["v22_pred"]) for r in c_rows])
        p_v1 = np.array([int(r["v1_pred"]) for r in c_rows])
        p_v2 = np.array([int(r["v2_pred"]) for r in c_rows])

        mean_confs = np.array([float(r["mean_confidence"]) for r in c_rows])

        # Get purities in 3 spaces (v2.2 fusion, raw, gradient)
        q_fusion = spaces["v2_2_fusion"][1](split_name)
        pur_fusion, _, _ = compute_5nn_stats(q_fusion, spaces["v2_2_fusion"][0], torch.from_numpy(targets).to(device), train_y)

        q_raw = spaces["raw_pixel"][1](split_name)
        pur_raw, _, _ = compute_5nn_stats(q_raw, spaces["raw_pixel"][0], torch.from_numpy(targets).to(device), train_y)

        q_grad = spaces["gradient_pca"][1](split_name)
        pur_grad, _, _ = compute_5nn_stats(q_grad, spaces["gradient_pca"][0], torch.from_numpy(targets).to(device), train_y)

        for true_name, pred_name in target_pairs:
            t_idx = CLASS_NAMES.index(true_name)
            p_idx = CLASS_NAMES.index(pred_name)

            # Samples where v2.1 or v2.2 or both predict pred_name for true_name
            cond_true = (targets == t_idx)
            err_21 = cond_true & (p21 == p_idx)
            err_22 = cond_true & (p22 == p_idx)
            err_both = err_21 & err_22
            err_any = err_21 | err_22

            # 4-model consensus: all 4 models predict p_idx
            err_quad = cond_true & (p_v1 == p_idx) & (p_v2 == p_idx) & (p21 == p_idx) & (p22 == p_idx)
            # Majority of 4 models predict p_idx
            quad_preds = np.stack([p_v1, p_v2, p21, p22], axis=-1)
            quad_majority = cond_true & (np.sum(quad_preds == p_idx, axis=-1) >= 3)

            n_any = int(np.sum(err_any))
            n_both = int(np.sum(err_both))
            n_quad = int(np.sum(err_quad))
            n_maj = int(np.sum(quad_majority))

            pair_key = f"{true_name} -> {pred_name}"

            if n_both > 0:
                both_indices = np.where(err_both)[0]
                m_conf = float(np.mean(mean_confs[both_indices]))
                m_pur_f = float(np.mean(pur_fusion[both_indices]))
                m_pur_raw = float(np.mean(pur_raw[both_indices]))
                m_pur_grad = float(np.mean(pur_grad[both_indices]))
            else:
                m_conf, m_pur_f, m_pur_raw, m_pur_grad = 0.0, 0.0, 0.0, 0.0

            confusion_summary[split_name][pair_key] = {
                "true_class": true_name,
                "predicted_class": pred_name,
                "v21_errors": int(np.sum(err_21)),
                "v22_errors": int(np.sum(err_22)),
                "v21_v22_shared_errors": n_both,
                "union_v21_v22_errors": n_any,
                "v21_v22_shared_fraction": float(n_both / n_any) if n_any > 0 else 0.0,
                "four_model_majority_predicting_error": n_maj,
                "four_model_unanimous_error": n_quad,
                "mean_confidence_on_shared": m_conf,
                "learned_fusion_true_purity": m_pur_f,
                "raw_pixel_true_purity": m_pur_raw,
                "gradient_pca_true_purity": m_pur_grad,
            }
            print(f"  {split_name:<7} | {pair_key:<20} | v21/v22 shared: {n_both:>3}/{n_any:<3} ({n_both/max(n_any,1)*100:.1f}%) | 4-Model Unanimous: {n_quad:>2} | Fusion Pur: {m_pur_f:.3f} | Raw Pur: {m_pur_raw:.3f}")

    (AUDIT_DIR / "a5_confusion_pair_analysis.json").write_text(json.dumps(confusion_summary, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a5_confusion_pair_analysis.json'}")

    # =========================================================================
    # PART 3: CLASS SEPARABILITY REVISIT (LEARNED VS. NON-LEARNED)
    # =========================================================================
    print("\n--- Part 3: Class Separability Revisit across Representations ---", flush=True)

    geom_spaces = {
        "v2_1_fusion": (train_pooled["v21_orig_r8"], multi_data["public_v2_1_fusion"], multi_data["private_v2_1_fusion"]),
        "v2_2_fusion": (train_pooled["v22_orig_r8"], multi_data["public_v2_2_fusion"], multi_data["private_v2_2_fusion"]),
        "raw_pixel": (tr_raw, pub_raw, priv_raw),
        "gradient_pca": (tr_grad_pca, pub_grad_pca, priv_grad_pca),
    }

    class_geom_summary = {}

    for s_name, (tr_arr, pub_arr, priv_arr) in geom_spaces.items():
        class_geom_summary[s_name] = {}
        for split_label, test_arr, test_y in [("public", pub_arr, val_ds.labels), ("private", priv_arr, test_ds.labels)]:
            mean_m, cls_m, dists = compute_centroid_margins(tr_arr, train_ds.labels, test_arr, test_y)

            # Nearest centroid accuracy
            preds = np.argmin(dists, axis=-1)
            nc_acc = float(np.mean(preds == test_y))

            class_geom_summary[s_name][split_label] = {
                "nearest_centroid_accuracy": nc_acc,
                "mean_centroid_margin": mean_m,
                "classwise_margins": cls_m,
            }
            print(f"  {s_name:<16} ({split_label:<7}) | NC Acc: {nc_acc:.4f} | Mean Margin: {mean_m:.3f} | Fear Margin: {cls_m['Fear']:.3f} | Sad Margin: {cls_m['Sad']:.3f} | Happy Margin: {cls_m['Happy']:.3f}")

    (AUDIT_DIR / "a5_nonlearned_class_geometry.json").write_text(json.dumps(class_geom_summary, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a5_nonlearned_class_geometry.json'}")

    print(f"\nA5.4 finished in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
