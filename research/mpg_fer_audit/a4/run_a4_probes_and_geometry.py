"""A4.6 Frozen Linear Probes and A4.7 Class Separability Geometry.

Fits frozen linear probes independently for v2.1 and v2.2 on:
- Motif readout [384]
- Fusion [512]
Using exact A1 protocol:
- StandardScaler fit Train only
- Multinomial LogisticRegression (C=1.0, L2, lbfgs, max_iter=5000, tol=1e-6)
Evaluates on Train, Public, Private for:
- raw feature representation
- TTA feature representation: 0.5 * (rep(orig) + rep(flip))
Computes paired bootstrap (B=2000) against official classifier outputs.

Computes Class Separability Geometry (A4.7):
- Train class centroids on standardized features
- Nearest-centroid accuracy
- True-class centroid margin (mean and class-wise)
- Within-class scatter
- Between-class centroid separation
- Fisher-style ratio (between / within)

Saves:
- a4_probe_metrics.json
- a4_probe_bootstrap.json
- a4_class_geometry.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"


def paired_bootstrap_probe_vs_official(
    y_true: np.ndarray,
    probe_pred: np.ndarray,
    official_pred: np.ndarray,
    B: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    N = len(y_true)
    classes = np.unique(y_true)
    class_indices = {c: np.where(y_true == c)[0] for c in classes}

    acc_deltas = []
    f1_deltas = []

    pe_acc = float(np.mean(probe_pred == y_true) - np.mean(official_pred == y_true))
    pe_f1 = float(f1_score(y_true, probe_pred, average="macro", zero_division=0) - f1_score(y_true, official_pred, average="macro", zero_division=0))

    for _ in range(B):
        sample_idx_list = []
        for c in classes:
            c_idx = class_indices[c]
            resamp = rng.choice(c_idx, size=len(c_idx), replace=True)
            sample_idx_list.append(resamp)
        boot_idx = np.concatenate(sample_idx_list)

        y_b = y_true[boot_idx]
        p_probe_b = probe_pred[boot_idx]
        p_off_b = official_pred[boot_idx]

        acc_deltas.append(float(np.mean(p_probe_b == y_b) - np.mean(p_off_b == y_b)))
        f1_p = f1_score(y_b, p_probe_b, average="macro", zero_division=0)
        f1_o = f1_score(y_b, p_off_b, average="macro", zero_division=0)
        f1_deltas.append(float(f1_p - f1_o))

    low_acc, high_acc = np.percentile(acc_deltas, [2.5, 97.5])
    low_f1, high_f1 = np.percentile(f1_deltas, [2.5, 97.5])

    return {
        "accuracy_delta_probe_minus_official": {
            "estimate": pe_acc,
            "ci_95_lower": float(low_acc),
            "ci_95_upper": float(high_acc),
        },
        "macro_f1_delta_probe_minus_official": {
            "estimate": pe_f1,
            "ci_95_lower": float(low_f1),
            "ci_95_upper": float(high_f1),
        },
    }


def compute_geometry(train_X_std: np.ndarray, train_y: np.ndarray, test_X_std: np.ndarray, test_y: np.ndarray) -> dict:
    classes = np.arange(7)
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    centroids = np.zeros((7, train_X_std.shape[1]), dtype=np.float64)

    for c in classes:
        mask = (train_y == c)
        centroids[c] = train_X_std[mask].mean(axis=0)

    # Between-class separation: average pairwise distance between class centroids
    dists_b = []
    for c1 in range(7):
        for c2 in range(c1 + 1, 7):
            dists_b.append(np.linalg.norm(centroids[c1] - centroids[c2]))
    between_sep = float(np.mean(dists_b))

    # Test evaluations
    # Distances to all centroids: [N, 7]
    # ||x - mu_c||_2
    test_dists = np.zeros((len(test_y), 7), dtype=np.float64)
    for c in classes:
        diff = test_X_std - centroids[c]
        test_dists[:, c] = np.linalg.norm(diff, axis=-1)

    nearest_pred = np.argmin(test_dists, axis=-1)
    nearest_acc = float(np.mean(nearest_pred == test_y))

    # True class centroid margin: min_{c != y} dist_c - dist_y
    margins = []
    classwise_margins = {c_name: [] for c_name in class_names}
    for i, y in enumerate(test_y):
        d_true = test_dists[i, y]
        d_other = np.min([test_dists[i, c] for c in classes if c != y])
        m = d_other - d_true
        margins.append(m)
        classwise_margins[class_names[y]].append(m)

    mean_margin = float(np.mean(margins))
    classwise_mean_margins = {k: float(np.mean(v)) for k, v in classwise_margins.items()}

    # Within-class scatter: average squared distance to true class centroid
    within_scatter = float(np.mean([test_dists[i, y] ** 2 for i, y in enumerate(test_y)]))
    fisher_ratio = float((between_sep ** 2) / (within_scatter + 1e-12))

    return {
        "nearest_centroid_accuracy": nearest_acc,
        "mean_true_class_centroid_margin": mean_margin,
        "classwise_true_class_margin": classwise_mean_margins,
        "within_class_scatter": within_scatter,
        "between_class_centroid_separation": between_sep,
        "fisher_style_ratio_between_sq_over_within": fisher_ratio,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "v21", "v22"], default="all")
    parser.add_argument("--rep", choices=["all", "motif_readout", "fusion"], default="all")
    args = parser.parse_args()

    start_time = time.time()
    print(f"Running A4.6 Frozen Linear Probes and A4.7 Class Separability Geometry (model: {args.model}, rep: {args.rep})...", flush=True)

    train_data = np.load(AUDIT_DIR / "train_pooled_features.npz")
    pub_data = np.load(AUDIT_DIR / "public_pooled_features.npz")
    priv_data = np.load(AUDIT_DIR / "private_pooled_features.npz")

    train_y = train_data["targets"]
    pub_y = pub_data["targets"]
    priv_y = priv_data["targets"]

    probe_metrics = {}
    probe_bootstrap = {}
    class_geometry = {}

    targets_dict = {"train": train_y, "public": pub_y, "private": priv_y}

    all_models = [("v21", "v2.1"), ("v22", "v2.2")]
    if args.model == "v21":
        model_list = [("v21", "v2.1")]
    elif args.model == "v22":
        model_list = [("v22", "v2.2")]
    else:
        model_list = all_models

    if args.rep == "motif_readout":
        rep_targets = ["motif_readout"]
    elif args.rep == "fusion":
        rep_targets = ["fusion"]
    else:
        rep_targets = ["motif_readout", "fusion"]

    for model_key, model_label in model_list:
        probe_metrics.setdefault(model_label, {})
        probe_bootstrap.setdefault(model_label, {})
        class_geometry.setdefault(model_label, {})

        # Load representations: raw and TTA
        # TTA representation = 0.5 * (orig + flip)
        reps = {
            "motif_readout": {
                "train_raw": train_data[f"{model_key}_orig_r7"],
                "train_tta": 0.5 * (train_data[f"{model_key}_orig_r7"] + train_data[f"{model_key}_flip_r7"]),
                "public_raw": pub_data[f"{model_key}_orig_r7"],
                "public_tta": 0.5 * (pub_data[f"{model_key}_orig_r7"] + pub_data[f"{model_key}_flip_r7"]),
                "private_raw": priv_data[f"{model_key}_orig_r7"],
                "private_tta": 0.5 * (priv_data[f"{model_key}_orig_r7"] + priv_data[f"{model_key}_flip_r7"]),
            },
            "fusion": {
                "train_raw": train_data[f"{model_key}_orig_r8"],
                "train_tta": 0.5 * (train_data[f"{model_key}_orig_r8"] + train_data[f"{model_key}_flip_r8"]),
                "public_raw": pub_data[f"{model_key}_orig_r8"],
                "public_tta": 0.5 * (pub_data[f"{model_key}_orig_r8"] + pub_data[f"{model_key}_flip_r8"]),
                "private_raw": priv_data[f"{model_key}_orig_r8"],
                "private_tta": 0.5 * (priv_data[f"{model_key}_orig_r8"] + priv_data[f"{model_key}_flip_r8"]),
            },
        }

        # Official classifier predictions (TTA)
        pub_off_pred = np.argmax(0.5 * (pub_data[f"{model_key}_orig_r10"] + pub_data[f"{model_key}_flip_r10"]), axis=-1)
        priv_off_pred = np.argmax(0.5 * (priv_data[f"{model_key}_orig_r10"] + priv_data[f"{model_key}_flip_r10"]), axis=-1)

        for rep_name in rep_targets:
            int_file = AUDIT_DIR / f"intermediate_probe_{model_key}_{rep_name}.json"
            if int_file.exists():
                print(f"Loading cached {model_label} {rep_name} from {int_file}...", flush=True)
                cached = json.loads(int_file.read_text(encoding="utf-8"))
                probe_metrics[model_label][rep_name] = cached["metrics"]
                probe_bootstrap[model_label][rep_name] = cached["bootstrap"]
                class_geometry[model_label][rep_name] = cached["geometry"]
                continue

            fit_t0 = time.time()
            dim = 384 if rep_name == "motif_readout" else 512
            print(f"\n--- Fitting probe for {model_label} on {rep_name} [{dim}d] ---", flush=True)

            X_tr_raw = reps[rep_name]["train_raw"]
            X_tr_tta = reps[rep_name]["train_tta"]

            # We fit on raw train features, and evaluate on both raw and TTA
            scaler = StandardScaler()
            X_tr_raw_std = scaler.fit_transform(X_tr_raw)
            X_tr_tta_std = scaler.transform(X_tr_tta)

            # Fit multinomial LogisticRegression
            clf = LogisticRegression(
                C=1.0,
                max_iter=5000,
                tol=1e-6,
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(X_tr_raw_std, train_y)
            fit_dur = time.time() - fit_t0
            converged = bool(clf.n_iter_[0] < clf.max_iter)
            print(f"  Fit finished in {fit_dur:.1f}s | Iterations: {clf.n_iter_[0]} | Converged: {converged}", flush=True)

            # Evaluate on all splits
            split_evals = {}
            for split_k in ["train", "public", "private"]:
                y_true = targets_dict[split_k]

                # Raw
                X_raw_std = scaler.transform(reps[rep_name][f"{split_k}_raw"])
                pred_raw = clf.predict(X_raw_std)
                acc_raw = float(np.mean(pred_raw == y_true))
                f1_raw = float(f1_score(y_true, pred_raw, average="macro", zero_division=0))

                # TTA
                X_tta_std = scaler.transform(reps[rep_name][f"{split_k}_tta"])
                pred_tta = clf.predict(X_tta_std)
                acc_tta = float(np.mean(pred_tta == y_true))
                f1_tta = float(f1_score(y_true, pred_tta, average="macro", zero_division=0))

                split_evals[split_k] = {
                    "raw": {"accuracy": acc_raw, "macro_f1": f1_raw},
                    "tta": {"accuracy": acc_tta, "macro_f1": f1_tta},
                }
                print(f"  {split_k.upper():<7} | Raw Acc: {acc_raw:.4f}, F1: {f1_raw:.4f} | TTA Acc: {acc_tta:.4f}, F1: {f1_tta:.4f}", flush=True)

            tr_acc = split_evals["train"]["tta"]["accuracy"]
            pub_acc = split_evals["public"]["tta"]["accuracy"]
            priv_acc = split_evals["private"]["tta"]["accuracy"]

            probe_metrics[model_label][rep_name] = {
                "dimensions": dim,
                "fit_duration_sec": fit_dur,
                "iterations": int(clf.n_iter_[0]),
                "converged": converged,
                "splits": split_evals,
                "train_public_gap": float(tr_acc - pub_acc),
                "train_private_gap": float(tr_acc - priv_acc),
            }

            # Paired bootstrap against official classifier
            # On Public
            pred_pub_tta = clf.predict(scaler.transform(reps[rep_name]["public_tta"]))
            boot_pub = paired_bootstrap_probe_vs_official(pub_y, pred_pub_tta, pub_off_pred, B=2000, seed=42)

            # On Private
            pred_priv_tta = clf.predict(scaler.transform(reps[rep_name]["private_tta"]))
            boot_priv = paired_bootstrap_probe_vs_official(priv_y, pred_priv_tta, priv_off_pred, B=2000, seed=42)

            probe_bootstrap[model_label][rep_name] = {
                "public": boot_pub,
                "private": boot_priv,
            }

            # --- A4.7 Class Separability Geometry ---
            # Using standardized Train features
            geom_pub = compute_geometry(
                X_tr_raw_std, train_y,
                scaler.transform(reps[rep_name]["public_raw"]), pub_y
            )
            geom_priv = compute_geometry(
                X_tr_raw_std, train_y,
                scaler.transform(reps[rep_name]["private_raw"]), priv_y
            )
            class_geometry[model_label][rep_name] = {
                "public": geom_pub,
                "private": geom_priv,
            }

            int_file = AUDIT_DIR / f"intermediate_probe_{model_key}_{rep_name}.json"
            int_file.write_text(json.dumps({
                "metrics": probe_metrics[model_label][rep_name],
                "bootstrap": probe_bootstrap[model_label][rep_name],
                "geometry": class_geometry[model_label][rep_name],
            }, indent=2), encoding="utf-8")
            print(f"Saved {int_file}", flush=True)

    # Load all 4 intermediate files if they exist
    for m_k, m_lbl in [("v21", "v2.1"), ("v22", "v2.2")]:
        for r_name in ["motif_readout", "fusion"]:
            f_int = AUDIT_DIR / f"intermediate_probe_{m_k}_{r_name}.json"
            if f_int.exists():
                c_data = json.loads(f_int.read_text(encoding="utf-8"))
                probe_metrics.setdefault(m_lbl, {})[r_name] = c_data["metrics"]
                probe_bootstrap.setdefault(m_lbl, {})[r_name] = c_data["bootstrap"]
                class_geometry.setdefault(m_lbl, {})[r_name] = c_data["geometry"]

    # Save output JSONs if both models present
    has_v21 = "v2.1" in probe_metrics and len(probe_metrics["v2.1"]) == 2
    has_v22 = "v2.2" in probe_metrics and len(probe_metrics["v2.2"]) == 2
    if has_v21 and has_v22:
        out_metrics_path = AUDIT_DIR / "a4_probe_metrics.json"
        out_metrics_path.write_text(json.dumps(probe_metrics, indent=2), encoding="utf-8")
        print(f"\nSaved {out_metrics_path}", flush=True)

        out_boot_path = AUDIT_DIR / "a4_probe_bootstrap.json"
        out_boot_path.write_text(json.dumps(probe_bootstrap, indent=2), encoding="utf-8")
        print(f"Saved {out_boot_path}", flush=True)

        out_geom_path = AUDIT_DIR / "a4_class_geometry.json"
        out_geom_path.write_text(json.dumps(class_geometry, indent=2), encoding="utf-8")
        print(f"Saved {out_geom_path}", flush=True)

    print(f"\nCompleted in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
