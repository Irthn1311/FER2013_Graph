"""A5.3 Exact Duplicate and Near-Duplicate Audit on FER2013.

Performs:
1. Exact Duplicate Audit:
   - SHA-256 on raw 48x48 uint8 pixel bytes across Train (28,709), Public (3,589), Private (3,589).
   - Identifies within-split and cross-split duplicates.
   - Flags EXACT_DUPLICATE_CONSISTENT_LABEL vs. EXACT_DUPLICATE_CONFLICTING_LABEL.
   - Saves a5_exact_duplicates.csv and a5_exact_duplicate_summary.json.

2. Near-Duplicate Audit:
   - Non-learned raw-pixel cosine similarity.
   - Non-learned deterministic gradient descriptor (Sobel X, Sobel Y, Grad Mag, Laplacian) + PCA (d=128, random_state=42, fit Train only).
   - Retrieves nearest Train neighbors for all Public and Private samples.
   - Reports similarity quantiles (99%, 99.5%, 99.9%, 99.95%, 99.99%).
   - Surfaces candidate near-duplicates at extreme similarity quantiles.
   - Saves a5_near_duplicate_candidates.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import convolve
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a5"

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def main():
    start_time = time.time()
    print("Running A5.3 Exact and Near-Duplicate Audits...", flush=True)

    # 1. Load raw datasets
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"))
    from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path

    train_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "train.csv", "train"), split="train")
    val_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val"), split="val")
    test_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test"), split="test")

    splits_data = [
        ("train", train_ds.images, train_ds.labels),
        ("public", val_ds.images, val_ds.labels),
        ("private", test_ds.images, test_ds.labels),
    ]

    total_images = sum(len(imgs) for _, imgs, _ in splits_data)
    print(f"Loaded {total_images} total images (Train: {len(train_ds)}, Public: {len(val_ds)}, Private: {len(test_ds)}).", flush=True)

    # =========================================================================
    # PART 1: EXACT DUPLICATE AUDIT
    # =========================================================================
    print("\n--- Part 1: Cryptographic Exact Duplicate Audit ---", flush=True)
    hash_to_members = {}

    for split_name, imgs, lbls in splits_data:
        for idx in range(len(imgs)):
            raw_bytes = imgs[idx].tobytes()
            h = hashlib.sha256(raw_bytes).hexdigest()
            hash_to_members.setdefault(h, []).append({
                "split": split_name,
                "row_index": idx,
                "label": int(lbls[idx]),
                "class_name": CLASS_NAMES[int(lbls[idx])],
            })

    duplicate_groups = {h: members for h, members in hash_to_members.items() if len(members) > 1}
    print(f"Found {len(duplicate_groups)} exact duplicate image groups containing {sum(len(m) for m in duplicate_groups.values())} total images.", flush=True)

    exact_dup_rows = []
    consistent_count = 0
    conflicting_count = 0
    cross_split_count = 0
    within_split_count = 0

    group_id = 1
    for h, members in duplicate_groups.items():
        unique_labels = set(m["label"] for m in members)
        unique_splits = set(m["split"] for m in members)
        is_conflicting = (len(unique_labels) > 1)
        is_cross_split = (len(unique_splits) > 1)

        status_flag = "EXACT_DUPLICATE_CONFLICTING_LABEL" if is_conflicting else "EXACT_DUPLICATE_CONSISTENT_LABEL"
        if is_conflicting:
            conflicting_count += 1
        else:
            consistent_count += 1

        if is_cross_split:
            cross_split_count += 1
        else:
            within_split_count += 1

        for m in members:
            exact_dup_rows.append({
                "group_id": group_id,
                "sha256": h,
                "split": m["split"],
                "row_index": m["row_index"],
                "label": m["label"],
                "class_name": m["class_name"],
                "group_size": len(members),
                "is_cross_split": is_cross_split,
                "splits_in_group": "/".join(sorted(unique_splits)),
                "labels_in_group": "/".join(sorted([CLASS_NAMES[l] for l in unique_labels])),
                "status_flag": status_flag,
            })
        group_id += 1

    # Save exact duplicates CSV
    exact_csv = AUDIT_DIR / "a5_exact_duplicates.csv"
    with exact_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "group_id", "sha256", "split", "row_index", "label", "class_name",
            "group_size", "is_cross_split", "splits_in_group", "labels_in_group", "status_flag"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exact_dup_rows)
    print(f"Saved {exact_csv} ({len(exact_dup_rows)} duplicate occurrences across {len(duplicate_groups)} groups).")

    # Detailed duplicate breakdown by split pair
    split_pair_counts = {
        "within_train": 0,
        "within_public": 0,
        "within_private": 0,
        "train_public": 0,
        "train_private": 0,
        "public_private": 0,
        "all_three_splits": 0,
    }
    split_pair_conflicts = {k: 0 for k in split_pair_counts}

    for h, members in duplicate_groups.items():
        s_set = set(m["split"] for m in members)
        l_set = set(m["label"] for m in members)
        is_conf = (len(l_set) > 1)

        if s_set == {"train"}:
            k = "within_train"
        elif s_set == {"public"}:
            k = "within_public"
        elif s_set == {"private"}:
            k = "within_private"
        elif s_set == {"train", "public"}:
            k = "train_public"
        elif s_set == {"train", "private"}:
            k = "train_private"
        elif s_set == {"public", "private"}:
            k = "public_private"
        else:
            k = "all_three_splits"

        split_pair_counts[k] += 1
        if is_conf:
            split_pair_conflicts[k] += 1

    exact_summary = {
        "total_exact_duplicate_groups": len(duplicate_groups),
        "total_images_in_duplicate_groups": sum(len(m) for m in duplicate_groups.values()),
        "consistent_label_groups": consistent_count,
        "conflicting_label_groups": conflicting_count,
        "conflicting_fraction": float(conflicting_count / len(duplicate_groups)),
        "cross_split_groups": cross_split_count,
        "within_split_groups": within_split_count,
        "breakdown_by_split_pair": {
            k: {
                "groups": split_pair_counts[k],
                "conflicting_groups": split_pair_conflicts[k],
                "conflicting_fraction": float(split_pair_conflicts[k] / split_pair_counts[k]) if split_pair_counts[k] > 0 else 0.0,
            }
            for k in split_pair_counts
        }
    }
    (AUDIT_DIR / "a5_exact_duplicate_summary.json").write_text(json.dumps(exact_summary, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a5_exact_duplicate_summary.json'}")

    # =========================================================================
    # PART 2: NEAR-DUPLICATE AUDIT
    # =========================================================================
    print("\n--- Part 2: Deterministic Non-Learned Near-Duplicate Audit ---", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A. Raw pixel vectors (flattened 2304 float32, centered, L2-normalized)
    def prep_raw_pixels(imgs):
        raw = imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0
        # Centering per sample
        raw_c = raw - raw.mean(axis=-1, keepdims=True)
        norm = np.linalg.norm(raw_c, axis=-1, keepdims=True)
        return raw_c / np.clip(norm, 1e-12, None)

    # B. Deterministic gradient descriptor
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 4.0
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 4.0
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    def extract_gradient_descriptors(imgs):
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

    print("Extracting gradient descriptors for Train, Public, Private...", flush=True)
    t_desc0 = time.time()
    train_raw = prep_raw_pixels(train_ds.images)
    pub_raw = prep_raw_pixels(val_ds.images)
    priv_raw = prep_raw_pixels(test_ds.images)

    train_grad = extract_gradient_descriptors(train_ds.images)
    pub_grad = extract_gradient_descriptors(val_ds.images)
    priv_grad = extract_gradient_descriptors(test_ds.images)
    print(f"Extracted gradient descriptors in {time.time() - t_desc0:.1f}s.", flush=True)

    # GPU-accelerated deterministic PCA (d=128, random_state=42) fit on Train only
    print("Fitting GPU-accelerated PCA (d=128, random_state=42) on Train gradient descriptors...", flush=True)
    t_pca0 = time.time()
    torch.manual_seed(42)
    tr_grad_tensor = torch.from_numpy(train_grad).to(device)
    U, S, V = torch.pca_lowrank(tr_grad_tensor, q=128, center=True)
    grad_mean = tr_grad_tensor.mean(dim=0, keepdim=True)

    def project_pca(mat_np):
        t = torch.from_numpy(mat_np).to(device)
        proj = (t - grad_mean) @ V  # [N, 128]
        proj_norm = F.normalize(proj, dim=-1)
        return proj_norm.cpu().numpy()

    train_grad_pca = project_pca(train_grad)
    pub_grad_pca = project_pca(pub_grad)
    priv_grad_pca = project_pca(priv_grad)
    print(f"PCA fit and projection completed in {time.time() - t_pca0:.2f}s.", flush=True)

    # Transfer to PyTorch GPU for rapid top-1 retrieval
    tr_raw_t = torch.from_numpy(train_raw).to(device)
    tr_grad_t = torch.from_numpy(train_grad_pca).to(device)
    tr_labels = train_ds.labels

    near_dup_records = []
    quantiles_summary = {}

    for split_name, test_raw, test_grad, test_labels in [
        ("public", pub_raw, pub_grad_pca, val_ds.labels),
        ("private", priv_raw, priv_grad_pca, test_ds.labels),
    ]:
        te_raw_t = torch.from_numpy(test_raw).to(device)
        te_grad_t = torch.from_numpy(test_grad).to(device)

        # Batch-compute top-1 Train neighbor under raw cosine and gradient cosine
        raw_top1_sims = []
        raw_top1_idx = []
        grad_top1_sims = []
        grad_top1_idx = []

        for start_i in range(0, len(test_raw), 512):
            end_i = min(start_i + 512, len(test_raw))
            sim_raw = te_raw_t[start_i:end_i] @ tr_raw_t.t()
            best_raw = torch.topk(sim_raw, k=1, dim=-1)
            raw_top1_sims.extend(best_raw.values.squeeze(-1).cpu().numpy().tolist())
            raw_top1_idx.extend(best_raw.indices.squeeze(-1).cpu().numpy().tolist())

            sim_grad = te_grad_t[start_i:end_i] @ tr_grad_t.t()
            best_grad = torch.topk(sim_grad, k=1, dim=-1)
            grad_top1_sims.extend(best_grad.values.squeeze(-1).cpu().numpy().tolist())
            grad_top1_idx.extend(best_grad.indices.squeeze(-1).cpu().numpy().tolist())

        raw_sims = np.array(raw_top1_sims)
        grad_sims = np.array(grad_top1_sims)

        # Compute quantiles: 99%, 99.5%, 99.9%, 99.95%, 99.99%
        q_levels = [99.0, 99.5, 99.9, 99.95, 99.99]
        raw_q = {f"q_{q}": float(np.percentile(raw_sims, q)) for q in q_levels}
        grad_q = {f"q_{q}": float(np.percentile(grad_sims, q)) for q in q_levels}

        quantiles_summary[split_name] = {
            "raw_pixel_cosine_quantiles": raw_q,
            "gradient_descriptor_cosine_quantiles": grad_q,
        }
        print(f"\n{split_name.upper()} Similarity Quantiles:")
        print(f"  Raw Pixel Cosine    : 99%={raw_q['q_99.0']:.4f}, 99.5%={raw_q['q_99.5']:.4f}, 99.9%={raw_q['q_99.9']:.4f}, 99.99%={raw_q['q_99.99']:.4f}")
        print(f"  Gradient PCA Cosine : 99%={grad_q['q_99.0']:.4f}, 99.5%={grad_q['q_99.5']:.4f}, 99.9%={grad_q['q_99.9']:.4f}, 99.99%={grad_q['q_99.99']:.4f}")

        # Threshold at 99.5% quantile to surface extreme near duplicates
        thresh_raw = raw_q["q_99.5"]
        thresh_grad = grad_q["q_99.5"]

        extreme_mask = (raw_sims >= thresh_raw) | (grad_sims >= thresh_grad)
        candidate_indices = np.where(extreme_mask)[0]

        for c_idx in candidate_indices:
            r_sim = float(raw_sims[c_idx])
            g_sim = float(grad_sims[c_idx])
            tr_r_idx = int(raw_top1_idx[c_idx])
            tr_g_idx = int(grad_top1_idx[c_idx])

            y_test = int(test_labels[c_idx])
            y_tr_raw = int(tr_labels[tr_r_idx])
            y_tr_grad = int(tr_labels[tr_g_idx])

            # Label conflict flags
            raw_conflict = (y_test != y_tr_raw)
            grad_conflict = (y_test != y_tr_grad)

            near_dup_records.append({
                "test_split": split_name,
                "test_row_index": int(c_idx),
                "test_label": y_test,
                "test_class": CLASS_NAMES[y_test],
                "raw_cosine_similarity": r_sim,
                "raw_nn_train_index": tr_r_idx,
                "raw_nn_train_label": y_tr_raw,
                "raw_nn_train_class": CLASS_NAMES[y_tr_raw],
                "raw_label_conflict": raw_conflict,
                "gradient_cosine_similarity": g_sim,
                "gradient_nn_train_index": tr_g_idx,
                "gradient_nn_train_label": y_tr_grad,
                "gradient_nn_train_class": CLASS_NAMES[y_tr_grad],
                "gradient_label_conflict": grad_conflict,
                "extreme_near_duplicate": bool((r_sim >= raw_q["q_99.9"]) or (g_sim >= grad_q["q_99.9"])),
            })

    # Sort candidates by descending max(raw_sim, grad_sim)
    near_dup_records.sort(key=lambda x: -max(x["raw_cosine_similarity"], x["gradient_cosine_similarity"]))

    # Save candidates CSV
    near_csv = AUDIT_DIR / "a5_near_duplicate_candidates.csv"
    with near_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(near_dup_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(near_dup_records)
    print(f"Saved {near_csv} ({len(near_dup_records)} candidate rows).")

    # Update summary with near-duplicate counts
    quantiles_summary["candidate_count"] = len(near_dup_records)
    quantiles_summary["candidate_conflicting_raw_count"] = sum(1 for r in near_dup_records if r["raw_label_conflict"])
    quantiles_summary["candidate_conflicting_grad_count"] = sum(1 for r in near_dup_records if r["gradient_label_conflict"])
    (AUDIT_DIR / "a5_near_duplicate_summary.json").write_text(json.dumps(quantiles_summary, indent=2), encoding="utf-8")

    print(f"\nA5.3 finished in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
