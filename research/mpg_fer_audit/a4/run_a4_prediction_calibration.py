"""A4.4 & A4.5: Prediction Agreement, Paired Bootstrap Statistics, and Calibration Audit."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"


def compute_nll(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-12) -> float:
    p_true = probs[np.arange(len(targets)), targets]
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))


def compute_brier(probs: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)
    corrects = (preds == targets).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs > low) & (confs <= high)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(corrects[mask]))
            bin_conf = float(np.mean(confs[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / len(targets)) * gap
        else:
            bin_acc = 0.0
            bin_conf = float(0.5 * (low + high))
            gap = 0.0

        bin_details.append({
            "bin": b,
            "range": [float(low), float(high)],
            "count": count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "gap": float(gap),
        })

    return float(ece), bin_details


def mcnemar_test(y_true, pred1, pred2) -> dict:
    corr1 = (pred1 == y_true)
    corr2 = (pred2 == y_true)

    b = int(np.sum(corr1 & (~corr2)))  # v21 correct, v22 wrong
    c = int(np.sum((~corr1) & corr2))  # v21 wrong, v22 correct
    n = b + c

    if n > 0:
        res = scipy.stats.binomtest(min(b, c), n, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)
    else:
        p_val = 1.0

    return {
        "v21_correct_v22_wrong": b,
        "v21_wrong_v22_correct": c,
        "discordant_total": n,
        "mcnemar_exact_p_value": p_val,
        "net_accuracy_delta": float((c - b) / len(y_true)),
    }


def paired_stratified_bootstrap(
    y_true: np.ndarray,
    probs21: np.ndarray,
    probs22: np.ndarray,
    B: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    N = len(y_true)
    classes = np.unique(y_true)
    class_indices = {c: np.where(y_true == c)[0] for c in classes}

    acc_deltas = []
    f1_deltas = []
    nll_deltas = []
    brier_deltas = []

    pred21 = np.argmax(probs21, axis=-1)
    pred22 = np.argmax(probs22, axis=-1)

    # Point estimates
    pe_acc = float(np.mean(pred22 == y_true) - np.mean(pred21 == y_true))
    pe_f1 = float(f1_score(y_true, pred22, average="macro", zero_division=0) - f1_score(y_true, pred21, average="macro", zero_division=0))
    pe_nll = float(compute_nll(probs22, y_true) - compute_nll(probs21, y_true))
    pe_brier = float(compute_brier(probs22, y_true) - compute_brier(probs21, y_true))

    for _ in range(B):
        sample_idx_list = []
        for c in classes:
            c_idx = class_indices[c]
            resamp = rng.choice(c_idx, size=len(c_idx), replace=True)
            sample_idx_list.append(resamp)
        boot_idx = np.concatenate(sample_idx_list)

        y_b = y_true[boot_idx]
        p21_b = probs21[boot_idx]
        p22_b = probs22[boot_idx]
        pr21_b = pred21[boot_idx]
        pr22_b = pred22[boot_idx]

        # Acc delta
        acc_deltas.append(float(np.mean(pr22_b == y_b) - np.mean(pr21_b == y_b)))
        # F1 delta
        f1_21 = f1_score(y_b, pr21_b, average="macro", zero_division=0)
        f1_22 = f1_score(y_b, pr22_b, average="macro", zero_division=0)
        f1_deltas.append(float(f1_22 - f1_21))
        # NLL delta
        nll_deltas.append(float(compute_nll(p22_b, y_b) - compute_nll(p21_b, y_b)))
        # Brier delta
        brier_deltas.append(float(compute_brier(p22_b, y_b) - compute_brier(p21_b, y_b)))

    def get_ci(arr, pe):
        low, high = np.percentile(arr, [2.5, 97.5])
        return {"estimate": float(pe), "ci_95_lower": float(low), "ci_95_upper": float(high)}

    return {
        "bootstrap_samples": B,
        "seed": seed,
        "accuracy_delta_v22_minus_v21": get_ci(acc_deltas, pe_acc),
        "macro_f1_delta_v22_minus_v21": get_ci(f1_deltas, pe_f1),
        "nll_delta_v22_minus_v21": get_ci(nll_deltas, pe_nll),
        "brier_delta_v22_minus_v21": get_ci(brier_deltas, pe_brier),
    }


def main():
    print("Computing A4.4 Prediction Agreement and A4.5 Calibration Audit...", flush=True)

    pub_data = np.load(AUDIT_DIR / "public_pooled_features.npz")
    priv_data = np.load(AUDIT_DIR / "private_pooled_features.npz")

    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    agreement_record = {}
    paired_record = {}
    calibration_record = {}

    for split, data in [("public", pub_data), ("private", priv_data)]:
        targets = data["targets"]
        N = len(targets)

        # Raw logits and TTA logits
        raw_l21 = data["v21_orig_r10"]
        raw_l22 = data["v22_orig_r10"]
        flip_l21 = data["v21_flip_r10"]
        flip_l22 = data["v22_flip_r10"]

        tta_l21 = 0.5 * (raw_l21 + flip_l21)
        tta_l22 = 0.5 * (raw_l22 + flip_l22)

        # Softmax
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)

        raw_p21 = softmax(raw_l21)
        raw_p22 = softmax(raw_l22)
        tta_p21 = softmax(tta_l21)
        tta_p22 = softmax(tta_l22)

        raw_pred21 = np.argmax(raw_p21, axis=-1)
        raw_pred22 = np.argmax(raw_p22, axis=-1)
        tta_pred21 = np.argmax(tta_p21, axis=-1)
        tta_pred22 = np.argmax(tta_p22, axis=-1)

        # --- A4.4 Prediction Agreement & Overlap ---
        def analyze_agreement(p1, p2, name):
            corr1 = (p1 == targets)
            corr2 = (p2 == targets)

            both_c = int(np.sum(corr1 & corr2))
            v21_only = int(np.sum(corr1 & (~corr2)))
            v22_only = int(np.sum((~corr1) & corr2))
            both_w = int(np.sum((~corr1) & (~corr2)))

            # Among both-wrong
            bw_mask = (~corr1) & (~corr2)
            same_w = int(np.sum(bw_mask & (p1 == p2)))
            diff_w = int(np.sum(bw_mask & (p1 != p2)))

            agree = float(np.mean(p1 == p2))
            kappa = float(cohen_kappa_score(p1, p2))

            # Error set Jaccard: union of errors vs intersection of errors
            err1 = set(np.where(~corr1)[0])
            err2 = set(np.where(~corr2)[0])
            err_inter = len(err1 & err2)
            err_union = len(err1 | err2)
            err_jaccard = float(err_inter / err_union) if err_union > 0 else 1.0

            # 7x7 Agreement matrix
            cm_agree = confusion_matrix(p1, p2, labels=range(7)).tolist()

            # Per class breakdown
            per_class = {}
            for c_i, c_n in enumerate(class_names):
                c_mask = (targets == c_i)
                c_n_samples = int(np.sum(c_mask))
                c_p1 = p1[c_mask]
                c_p2 = p2[c_mask]
                c_corr1 = (c_p1 == c_i)
                c_corr2 = (c_p2 == c_i)
                per_class[c_n] = {
                    "count": c_n_samples,
                    "both_correct": int(np.sum(c_corr1 & c_corr2)),
                    "v21_only_correct": int(np.sum(c_corr1 & (~c_corr2))),
                    "v22_only_correct": int(np.sum((~c_corr1) & c_corr2)),
                    "both_wrong": int(np.sum((~c_corr1) & (~c_corr2))),
                    "both_wrong_same_pred": int(np.sum((~c_corr1) & (~c_corr2) & (c_p1 == c_p2))),
                    "agreement_rate": float(np.mean(c_p1 == c_p2)),
                }

            return {
                "view": name,
                "both_correct_count": both_c,
                "both_correct_fraction": float(both_c / N),
                "v21_only_correct_count": v21_only,
                "v21_only_correct_fraction": float(v21_only / N),
                "v22_only_correct_count": v22_only,
                "v22_only_correct_fraction": float(v22_only / N),
                "both_wrong_count": both_w,
                "both_wrong_fraction": float(both_w / N),
                "both_wrong_same_pred_count": same_w,
                "both_wrong_same_pred_fraction_of_both_wrong": float(same_w / both_w) if both_w > 0 else 0.0,
                "both_wrong_diff_pred_count": diff_w,
                "both_wrong_diff_pred_fraction_of_both_wrong": float(diff_w / both_w) if both_w > 0 else 0.0,
                "prediction_agreement": agree,
                "cohen_kappa": kappa,
                "error_set_jaccard": err_jaccard,
                "agreement_matrix_v21_rows_v22_cols": cm_agree,
                "per_class_breakdown": per_class,
            }

        agreement_record[split] = {
            "raw": analyze_agreement(raw_pred21, raw_pred22, "raw"),
            "tta": analyze_agreement(tta_pred21, tta_pred22, "tta"),
        }

        # --- A4.4 Paired Performance Statistics ---
        mcnemar_res = mcnemar_test(targets, tta_pred21, tta_pred22)
        boot_res = paired_stratified_bootstrap(targets, tta_p21, tta_p22, B=2000, seed=42)

        paired_record[split] = {
            "mcnemar_test": mcnemar_res,
            "paired_bootstrap_tta": boot_res,
        }

        # --- A4.5 Calibration & Confidence Audit ---
        def get_calib(probs, preds, name):
            nll = compute_nll(probs, targets)
            brier = compute_brier(probs, targets)
            ece, bins = compute_ece(probs, targets, n_bins=15)

            confs = np.max(probs, axis=-1)
            sorted_p = np.sort(probs, axis=-1)[:, ::-1]
            margins = sorted_p[:, 0] - sorted_p[:, 1]
            entropies = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)

            corr_mask = (preds == targets)
            incorr_mask = ~corr_mask

            return {
                "view": name,
                "accuracy": float(np.mean(corr_mask)),
                "nll": nll,
                "brier": brier,
                "ece_15_bins": ece,
                "mean_confidence_correct": float(np.mean(confs[corr_mask])),
                "mean_confidence_incorrect": float(np.mean(confs[incorr_mask])),
                "mean_margin_correct": float(np.mean(margins[corr_mask])),
                "mean_margin_incorrect": float(np.mean(margins[incorr_mask])),
                "mean_entropy_correct": float(np.mean(entropies[corr_mask])),
                "mean_entropy_incorrect": float(np.mean(entropies[incorr_mask])),
                "bins": bins,
            }

        calib_raw_21 = get_calib(raw_p21, raw_pred21, "v21_raw")
        calib_raw_22 = get_calib(raw_p22, raw_pred22, "v22_raw")
        calib_tta_21 = get_calib(tta_p21, tta_pred21, "v21_tta")
        calib_tta_22 = get_calib(tta_p22, tta_pred22, "v22_tta")

        def compute_tta_gain(raw_c, tta_c):
            return {
                "accuracy_gain": float(tta_c["accuracy"] - raw_c["accuracy"]),
                "nll_delta": float(tta_c["nll"] - raw_c["nll"]),
                "brier_delta": float(tta_c["brier"] - raw_c["brier"]),
                "ece_delta": float(tta_c["ece_15_bins"] - raw_c["ece_15_bins"]),
            }

        calibration_record[split] = {
            "v2_1": {
                "raw": calib_raw_21,
                "tta": calib_tta_21,
                "tta_gain": compute_tta_gain(calib_raw_21, calib_tta_21),
            },
            "v2_2": {
                "raw": calib_raw_22,
                "tta": calib_tta_22,
                "tta_gain": compute_tta_gain(calib_raw_22, calib_tta_22),
            },
            "comparison_v22_minus_v21_tta": {
                "accuracy_delta": float(calib_tta_22["accuracy"] - calib_tta_21["accuracy"]),
                "nll_delta": float(calib_tta_22["nll"] - calib_tta_21["nll"]),
                "brier_delta": float(calib_tta_22["brier"] - calib_tta_21["brier"]),
                "ece_delta": float(calib_tta_22["ece_15_bins"] - calib_tta_21["ece_15_bins"]),
            }
        }

        # Plot calibration curve for split
        plt.figure(figsize=(7, 6))
        bins_21 = calib_tta_21["bins"]
        bins_22 = calib_tta_22["bins"]

        confs_21 = [b["confidence"] for b in bins_21 if b["count"] > 0]
        accs_21 = [b["accuracy"] for b in bins_21 if b["count"] > 0]
        confs_22 = [b["confidence"] for b in bins_22 if b["count"] > 0]
        accs_22 = [b["accuracy"] for b in bins_22 if b["count"] > 0]

        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        plt.plot(confs_21, accs_21, "o-", color="#2b5c8f", label=f"v2.1 Dense (ECE={calib_tta_21['ece_15_bins']:.4f})")
        plt.plot(confs_22, accs_22, "s-", color="#e24a33", label=f"v2.2 Sparse (ECE={calib_tta_22['ece_15_bins']:.4f})")

        plt.xlabel("Mean Predicted Confidence (15 bins)", fontsize=11, fontweight="bold")
        plt.ylabel("Observed Empirical Accuracy", fontsize=11, fontweight="bold")
        plt.title(f"A4.5: Reliability Diagram ({split.upper()} TTA)", fontsize=12, fontweight="bold")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(AUDIT_DIR / f"a4_calibration_{split}.png", dpi=150)
        plt.close()
        print(f"Saved a4_calibration_{split}.png")

    (AUDIT_DIR / "a4_prediction_agreement.json").write_text(json.dumps(agreement_record, indent=2), encoding="utf-8")
    (AUDIT_DIR / "a4_paired_statistics.json").write_text(json.dumps(paired_record, indent=2), encoding="utf-8")
    (AUDIT_DIR / "a4_calibration.json").write_text(json.dumps(calibration_record, indent=2), encoding="utf-8")
    print("Saved a4_prediction_agreement.json, a4_paired_statistics.json, a4_calibration.json")

    # Routing overlap plot
    routing_data = json.load(open(AUDIT_DIR / "a4_routing_overlap.json"))
    layers = [1, 2, 3, 4, 5]
    jacc_pub = [routing_data["public"][f"layer_{l}"]["support_jaccard_mean"] for l in layers]
    jacc_priv = [routing_data["private"][f"layer_{l}"]["support_jaccard_mean"] for l in layers]
    overlap_pub = [routing_data["public"][f"layer_{l}"]["overlap_k_mean"] for l in layers]
    overlap_priv = [routing_data["private"][f"layer_{l}"]["overlap_k_mean"] for l in layers]

    plt.figure(figsize=(8, 5))
    x_idx = np.arange(1, 6)
    plt.plot(x_idx, jacc_pub, "o-", color="#2b5c8f", label="Support Jaccard (Public)")
    plt.plot(x_idx, jacc_priv, "o--", color="#4682b4", label="Support Jaccard (Private)")
    plt.plot(x_idx, overlap_pub, "s-", color="#e24a33", label="Overlap / K (Public)")
    plt.plot(x_idx, overlap_priv, "s--", color="#ff7f50", label="Overlap / K (Private)")

    for i, l in enumerate(layers):
        k = routing_data["public"][f"layer_{l}"]["K"]
        plt.annotate(f"K={k}", (x_idx[i], jacc_pub[i] - 0.03), ha="center", fontsize=9, fontweight="bold")

    plt.xlabel("Motif GNN Layer", fontsize=11, fontweight="bold")
    plt.ylabel("Routing Overlap Metric", fontsize=11, fontweight="bold")
    plt.title("A4.3: Dense vs. Sparse Routing Support Overlap", fontsize=12, fontweight="bold")
    plt.xticks(x_idx, [f"Layer {l}" for l in layers])
    plt.ylim(0.15, 0.65)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(AUDIT_DIR / "a4_routing_overlap_by_layer.png", dpi=150)
    plt.close()
    print("Saved a4_routing_overlap_by_layer.png")

    # Flip Support Jaccard plot
    flip_data = json.load(open(AUDIT_DIR / "a4_flip_equivariance.json"))
    v21_flip_pub = [flip_data["public"]["routing_support_flip_jaccard"][f"layer_{l}"]["v2_1"] for l in layers]
    v22_flip_pub = [flip_data["public"]["routing_support_flip_jaccard"][f"layer_{l}"]["v2_2"] for l in layers]
    v21_flip_priv = [flip_data["private"]["routing_support_flip_jaccard"][f"layer_{l}"]["v2_1"] for l in layers]
    v22_flip_priv = [flip_data["private"]["routing_support_flip_jaccard"][f"layer_{l}"]["v2_2"] for l in layers]

    plt.figure(figsize=(8, 5))
    plt.plot(x_idx, v21_flip_pub, "o-", color="#2b5c8f", label="v2.1 Dense (Public)")
    plt.plot(x_idx, v22_flip_pub, "s-", color="#e24a33", label="v2.2 Sparse (Public)")
    plt.plot(x_idx, v21_flip_priv, "o--", color="#87ceeb", label="v2.1 Dense (Private)")
    plt.plot(x_idx, v22_flip_priv, "s--", color="#ff7f50", label="v2.2 Sparse (Private)")

    plt.xlabel("Motif GNN Layer", fontsize=11, fontweight="bold")
    plt.ylabel("Mirror-Mapped Flip Support Jaccard", fontsize=11, fontweight="bold")
    plt.title("A4.8: Flip Equivariance of Relational Support", fontsize=12, fontweight="bold")
    plt.xticks(x_idx, [f"Layer {l}" for l in layers])
    plt.ylim(0.15, 0.55)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(AUDIT_DIR / "a4_flip_support_jaccard.png", dpi=150)
    plt.close()
    print("Saved a4_flip_support_jaccard.png")


if __name__ == "__main__":
    main()
