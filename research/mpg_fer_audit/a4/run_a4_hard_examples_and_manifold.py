"""A4.9 Shared Hard Examples, A4.10 Error Manifold Agreement, and 5-NN Purity.

Implements:
1. Sample-level audit tables (Categories A, B, C, D, E).
2. SHARED_HIGH_CONFIDENCE_ERROR_CANDIDATES:
   - both wrong, same wrong prediction, both confidence >= 0.80
   - saves a4_shared_high_confidence_errors.csv
   - generates contact sheets grouped by true class -> shared predicted class (max 20 per pair)
3. MODEL-DISAGREEMENT REVIEW SET:
   - v2.1 correct / v2.2 wrong & v2.2 correct / v2.1 wrong
   - prioritized by highest confidence difference (max 50 per split)
   - saves a4_model_disagreements.csv
   - generates contact sheets
4. ERROR MANIFOLD AGREEMENT (A4.10):
   - confusion pair analysis (Fear->Sad, Fear->Neutral, Sad->Neutral, Angry->Sad, Neutral->Sad, etc.)
   - errors v2.1, errors v2.2, shared errors, shared fraction relative to union, mean confidence, mean fusion cosine
   - saves a4_error_overlap.csv
   - plots a4_error_overlap_public.png, a4_error_overlap_private.png
5. OPTIONAL 5-NN LOCAL LABEL PURITY (A4.11 / Section 23):
   - cosine 5-nearest Train neighbors using L2-normalized Train Fusion features
   - neighbor label purity and entropy per group
   - saves a4_knn_purity.json
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"
CONTACT_DIR = AUDIT_DIR / "contact_sheets"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def make_contact_sheet(
    images: np.ndarray,
    items: list[dict],
    title: str,
    save_path: Path,
    max_images: int = 20,
):
    if len(items) == 0:
        return
    items = items[:max_images]
    n = len(items)
    cols = min(5, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i < n:
            item = items[i]
            idx = item["row_index"]
            img = images[idx]
            ax.imshow(img, cmap="gray")
            lbl_text = (
                f"Idx: {idx}\n"
                f"T: {CLASS_NAMES[item['true_label']]}\n"
                f"2.1: {CLASS_NAMES[item['v2_1_pred']]} ({item['v2_1_confidence']:.2f})\n"
                f"2.2: {CLASS_NAMES[item['v2_2_pred']]} ({item['v2_2_confidence']:.2f})"
            )
            ax.set_title(lbl_text, fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    print("Running A4.9 Shared Hard Examples and A4.10 Error Manifold Audit...", flush=True)

    # Load pooled features
    train_data = np.load(AUDIT_DIR / "train_pooled_features.npz")
    pub_data = np.load(AUDIT_DIR / "public_pooled_features.npz")
    priv_data = np.load(AUDIT_DIR / "private_pooled_features.npz")

    # Load original raw images from CSV
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"))
    from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path

    val_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val"), split="val")
    test_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test"), split="test")

    pub_images = val_ds.images
    priv_images = test_ds.images

    pub_rows = json.loads((AUDIT_DIR / "public_sample_audit_table.json").read_text(encoding="utf-8"))
    priv_rows = json.loads((AUDIT_DIR / "private_sample_audit_table.json").read_text(encoding="utf-8"))

    all_shared_high_conf = []
    all_disagreements = []
    manifold_rows = []

    # Process each split
    for split_name, rows, images, pooled_feat in [
        ("public", pub_rows, pub_images, pub_data),
        ("private", priv_rows, priv_images, priv_data),
    ]:
        print(f"\nAnalyzing {split_name.upper()} split...", flush=True)
        fusion_21 = pooled_feat["v21_orig_r8"]
        fusion_22 = pooled_feat["v22_orig_r8"]

        # 1. SHARED_HIGH_CONFIDENCE_ERROR_CANDIDATES
        # both wrong, same wrong prediction, both conf >= 0.80
        shared_hc = [
            r for r in rows
            if r["category"] == "D_both_wrong_same_label"
            and r["v2_1_confidence"] >= 0.80
            and r["v2_2_confidence"] >= 0.80
        ]
        # Sort by descending mean confidence, then row index
        shared_hc.sort(key=lambda x: (-(x["v2_1_confidence"] + x["v2_2_confidence"]) / 2.0, x["row_index"]))
        print(f"  Found {len(shared_hc)} SHARED_HIGH_CONFIDENCE_ERROR_CANDIDATES in {split_name}.")

        for item in shared_hc:
            item_copy = dict(item)
            item_copy["mean_confidence"] = float(0.5 * (item["v2_1_confidence"] + item["v2_2_confidence"]))
            all_shared_high_conf.append(item_copy)

        # Generate contact sheets grouped by true class -> shared predicted class
        # Maximum 20 per pair
        pairs_dict = {}
        for item in shared_hc:
            pair_key = (item["true_label"], item["v2_1_pred"])
            pairs_dict.setdefault(pair_key, []).append(item)

        for (true_c, pred_c), pair_items in pairs_dict.items():
            pair_name = f"{CLASS_NAMES[true_c]}_to_{CLASS_NAMES[pred_c]}"
            cs_title = f"{split_name.upper()}: Shared High-Conf Error ({pair_name}) [N={len(pair_items)}]"
            cs_file = CONTACT_DIR / f"{split_name}_shared_hc_{pair_name}.png"
            make_contact_sheet(images, pair_items, cs_title, cs_file, max_images=20)

        # 2. MODEL-DISAGREEMENT REVIEW SET
        # B: v2.1 correct / v2.2 wrong
        v21_only = [r for r in rows if r["category"] == "B_v21_only_correct"]
        # C: v2.2 correct / v2.1 wrong
        v22_only = [r for r in rows if r["category"] == "C_v22_only_correct"]

        # Prioritize by highest confidence difference |conf21 - conf22|
        v21_only.sort(key=lambda x: (-abs(x["v2_1_confidence"] - x["v2_2_confidence"]), x["row_index"]))
        v22_only.sort(key=lambda x: (-abs(x["v2_2_confidence"] - x["v2_1_confidence"]), x["row_index"]))

        top_v21_only = v21_only[:50]
        top_v22_only = v22_only[:50]

        for item in top_v21_only:
            d_item = dict(item)
            d_item["type"] = "v21_correct_v22_wrong"
            d_item["conf_diff"] = float(item["v2_1_confidence"] - item["v2_2_confidence"])
            all_disagreements.append(d_item)

        for item in top_v22_only:
            d_item = dict(item)
            d_item["type"] = "v22_correct_v21_wrong"
            d_item["conf_diff"] = float(item["v2_2_confidence"] - item["v2_1_confidence"])
            all_disagreements.append(d_item)

        # Contact sheets for model disagreements
        make_contact_sheet(
            images, top_v21_only,
            f"{split_name.upper()}: v2.1 Correct / v2.2 Wrong (Top 50 Conf Delta)",
            CONTACT_DIR / f"{split_name}_disagree_v21_correct.png",
            max_images=50,
        )
        make_contact_sheet(
            images, top_v22_only,
            f"{split_name.upper()}: v2.2 Correct / v2.1 Wrong (Top 50 Conf Delta)",
            CONTACT_DIR / f"{split_name}_disagree_v22_correct.png",
            max_images=50,
        )

        # 3. ERROR MANIFOLD AGREEMENT (A4.10)
        # Compute confusion pairs
        targets = np.array([r["true_label"] for r in rows])
        pred21 = np.array([r["v2_1_pred"] for r in rows])
        pred22 = np.array([r["v2_2_pred"] for r in rows])
        conf21 = np.array([r["v2_1_confidence"] for r in rows])
        conf22 = np.array([r["v2_2_confidence"] for r in rows])

        # Cosine similarity between fusion representations
        f21_norm = fusion_21 / np.linalg.norm(fusion_21, axis=-1, keepdims=True)
        f22_norm = fusion_22 / np.linalg.norm(fusion_22, axis=-1, keepdims=True)
        cos_sims = np.sum(f21_norm * f22_norm, axis=-1)

        both_corr_mask = (pred21 == targets) & (pred22 == targets)
        mean_cos_both_corr = float(np.mean(cos_sims[both_corr_mask]))

        # Key class pairs and all pairs
        for true_c in range(7):
            for pred_c in range(7):
                if true_c == pred_c:
                    continue
                # Errors where model predicts pred_c for true_c
                err21_idx = np.where((targets == true_c) & (pred21 == pred_c))[0]
                err22_idx = np.where((targets == true_c) & (pred22 == pred_c))[0]

                s21 = set(err21_idx)
                s22 = set(err22_idx)
                shared = s21 & s22
                union = s21 | s22

                if len(union) == 0:
                    continue

                shared_frac = float(len(shared) / len(union))
                shared_idx = list(shared)
                mean_conf_pair = float(0.5 * (np.mean(conf21[shared_idx]) + np.mean(conf22[shared_idx]))) if shared else 0.0
                mean_cos_pair = float(np.mean(cos_sims[shared_idx])) if shared else 0.0

                manifold_rows.append({
                    "split": split_name,
                    "true_class": CLASS_NAMES[true_c],
                    "predicted_class": CLASS_NAMES[pred_c],
                    "pair_label": f"{CLASS_NAMES[true_c]} -> {CLASS_NAMES[pred_c]}",
                    "n_errors_v21": len(s21),
                    "n_errors_v22": len(s22),
                    "shared_error_count": len(shared),
                    "union_error_count": len(union),
                    "shared_error_fraction_of_union": shared_frac,
                    "mean_confidence_on_shared": mean_conf_pair,
                    "mean_fusion_cosine_on_shared": mean_cos_pair,
                    "mean_fusion_cosine_both_correct_baseline": mean_cos_both_corr,
                })

    # Save CSVs
    # a4_shared_high_confidence_errors.csv
    sh_csv = AUDIT_DIR / "a4_shared_high_confidence_errors.csv"
    with sh_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "split", "row_index", "true_label", "true_class",
            "v2_1_pred", "v2_2_pred", "shared_predicted_class",
            "v2_1_confidence", "v2_2_confidence", "mean_confidence",
            "v2_1_margin", "v2_2_margin", "v2_1_entropy", "v2_2_entropy"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_shared_high_conf:
            writer.writerow({
                "split": r["split"],
                "row_index": r["row_index"],
                "true_label": r["true_label"],
                "true_class": CLASS_NAMES[r["true_label"]],
                "v2_1_pred": r["v2_1_pred"],
                "v2_2_pred": r["v2_2_pred"],
                "shared_predicted_class": CLASS_NAMES[r["v2_1_pred"]],
                "v2_1_confidence": f"{r['v2_1_confidence']:.4f}",
                "v2_2_confidence": f"{r['v2_2_confidence']:.4f}",
                "mean_confidence": f"{r['mean_confidence']:.4f}",
                "v2_1_margin": f"{r['v2_1_margin']:.4f}",
                "v2_2_margin": f"{r['v2_2_margin']:.4f}",
                "v2_1_entropy": f"{r['v2_1_entropy']:.4f}",
                "v2_2_entropy": f"{r['v2_2_entropy']:.4f}",
            })
    print(f"Saved {sh_csv} ({len(all_shared_high_conf)} rows).")

    # a4_model_disagreements.csv
    dis_csv = AUDIT_DIR / "a4_model_disagreements.csv"
    with dis_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "split", "type", "row_index", "true_label", "true_class",
            "v2_1_pred", "v2_1_pred_class", "v2_2_pred", "v2_2_pred_class",
            "v2_1_confidence", "v2_2_confidence", "conf_diff"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_disagreements:
            writer.writerow({
                "split": r["split"],
                "type": r["type"],
                "row_index": r["row_index"],
                "true_label": r["true_label"],
                "true_class": CLASS_NAMES[r["true_label"]],
                "v2_1_pred": r["v2_1_pred"],
                "v2_1_pred_class": CLASS_NAMES[r["v2_1_pred"]],
                "v2_2_pred": r["v2_2_pred"],
                "v2_2_pred_class": CLASS_NAMES[r["v2_2_pred"]],
                "v2_1_confidence": f"{r['v2_1_confidence']:.4f}",
                "v2_2_confidence": f"{r['v2_2_confidence']:.4f}",
                "conf_diff": f"{r['conf_diff']:.4f}",
            })
    print(f"Saved {dis_csv} ({len(all_disagreements)} rows).")

    # a4_error_overlap.csv
    err_csv = AUDIT_DIR / "a4_error_overlap.csv"
    with err_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "split", "true_class", "predicted_class", "pair_label",
            "n_errors_v21", "n_errors_v22", "shared_error_count",
            "union_error_count", "shared_error_fraction_of_union",
            "mean_confidence_on_shared", "mean_fusion_cosine_on_shared",
            "mean_fusion_cosine_both_correct_baseline"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in manifold_rows:
            writer.writerow({
                "split": r["split"],
                "true_class": r["true_class"],
                "predicted_class": r["predicted_class"],
                "pair_label": r["pair_label"],
                "n_errors_v21": r["n_errors_v21"],
                "n_errors_v22": r["n_errors_v22"],
                "shared_error_count": r["shared_error_count"],
                "union_error_count": r["union_error_count"],
                "shared_error_fraction_of_union": f"{r['shared_error_fraction_of_union']:.4f}",
                "mean_confidence_on_shared": f"{r['mean_confidence_on_shared']:.4f}",
                "mean_fusion_cosine_on_shared": f"{r['mean_fusion_cosine_on_shared']:.4f}",
                "mean_fusion_cosine_both_correct_baseline": f"{r['mean_fusion_cosine_both_correct_baseline']:.4f}",
            })
    print(f"Saved {err_csv} ({len(manifold_rows)} rows).")

    # Plots for error manifold
    for split_name in ["public", "private"]:
        sub_rows = [r for r in manifold_rows if r["split"] == split_name]
        # Sort by shared error count descending
        sub_rows.sort(key=lambda x: -x["shared_error_count"])
        top_pairs = sub_rows[:12]

        labels = [p["pair_label"] for p in top_pairs]
        shared_counts = [p["shared_error_count"] for p in top_pairs]
        union_counts = [p["union_error_count"] for p in top_pairs]
        fracs = [p["shared_error_fraction_of_union"] for p in top_pairs]

        plt.figure(figsize=(10, 6))
        x_idx = np.arange(len(top_pairs))
        plt.bar(x_idx - 0.2, union_counts, width=0.4, label="Union Errors (v2.1 ∪ v2.2)", color="#a0c4df")
        plt.bar(x_idx + 0.2, shared_counts, width=0.4, label="Shared Errors (v2.1 ∩ v2.2)", color="#2b5c8f")

        for i in range(len(top_pairs)):
            plt.text(x_idx[i] + 0.2, shared_counts[i] + 1, f"{fracs[i]*100:.0f}%", ha="center", fontsize=8, fontweight="bold")

        plt.xlabel("Confusion Pair", fontsize=11, fontweight="bold")
        plt.ylabel("Number of Errors", fontsize=11, fontweight="bold")
        plt.title(f"A4.10: Shared Error Overlap by Top Confusion Pairs ({split_name.upper()})", fontsize=12, fontweight="bold")
        plt.xticks(x_idx, labels, rotation=35, ha="right", fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(AUDIT_DIR / f"a4_error_overlap_{split_name}.png", dpi=150)
        plt.close()
        print(f"Saved a4_error_overlap_{split_name}.png")

    # --- 4. OPTIONAL 5-NN LOCAL LABEL PURITY (Section 23) ---
    print("\nComputing 5-NN Local Label Purity from Train Fusion features...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    knn_summary = {}
    for model_key, model_label in [("v21", "v2.1"), ("v22", "v2.2")]:
        tr_f = torch.from_numpy(train_data[f"{model_key}_orig_r8"]).to(device)
        tr_f_norm = F.normalize(tr_f, dim=-1) # [28709, 512]
        tr_y = torch.from_numpy(train_data["targets"]).to(device)

        knn_summary[model_label] = {}

        for split_name, pooled_feat, s_rows in [("public", pub_data, pub_rows), ("private", priv_data, priv_rows)]:
            te_f = torch.from_numpy(pooled_feat[f"{model_key}_orig_r8"]).to(device)
            te_f_norm = F.normalize(te_f, dim=-1) # [3589, 512]
            te_y = torch.from_numpy(pooled_feat["targets"]).to(device)

            # Cosine similarity matrix: [3589, 28709]
            # Batching to avoid peak memory
            purities = []
            entropies = []

            for start_i in range(0, len(te_f_norm), 512):
                end_i = min(start_i + 512, len(te_f_norm))
                sim_batch = te_f_norm[start_i:end_i] @ tr_f_norm.t() # [B_sub, 28709]
                top5_res = torch.topk(sim_batch, k=5, dim=-1)
                top5_labels = tr_y[top5_res.indices] # [B_sub, 5]
                targets_batch = te_y[start_i:end_i].unsqueeze(1) # [B_sub, 1]

                # Label purity = fraction of 5 neighbors that match target
                batch_purity = (top5_labels == targets_batch).float().mean(dim=-1).cpu().numpy()
                purities.extend(batch_purity.tolist())

                # Label entropy
                for row_labels in top5_labels.cpu().numpy():
                    counts = np.bincount(row_labels, minlength=7)
                    probs = counts[counts > 0] / 5.0
                    ent = -np.sum(probs * np.log(probs))
                    entropies.append(float(ent))

            # Group by categories A, B, C, D, E
            cats = [r["category"] for r in s_rows]
            group_results = {}
            for cat_name in [
                "A_both_correct", "B_v21_only_correct", "C_v22_only_correct",
                "D_both_wrong_same_label", "E_both_wrong_different_labels"
            ]:
                cat_mask = np.array([c == cat_name for c in cats])
                if np.sum(cat_mask) > 0:
                    group_results[cat_name] = {
                        "count": int(np.sum(cat_mask)),
                        "mean_5nn_purity": float(np.mean(np.array(purities)[cat_mask])),
                        "mean_5nn_entropy": float(np.mean(np.array(entropies)[cat_mask])),
                    }
                else:
                    group_results[cat_name] = {"count": 0, "mean_5nn_purity": 0.0, "mean_5nn_entropy": 0.0}

            knn_summary[model_label][split_name] = group_results

    (AUDIT_DIR / "a4_knn_purity.json").write_text(json.dumps(knn_summary, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a4_knn_purity.json'}")


if __name__ == "__main__":
    main()
