"""A6.2 & A6.3: Layerwise Probe Metrics, True-Class Margins, and First-Rescue/Collapse Analysis.

Loads cached probes from research/mpg_fer_audit/a6/probe_cache/ and produces:
- a6_stage_probe_metrics.json
- a6_stage_margin_samples.csv
- a6_stage_margin_summary.json
- a6_first_rescue_collapse.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6"
CACHE_DIR = AUDIT_DIR / "probe_cache"

STAGES = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
ALL_STAGES_WITH_LOGITS = STAGES + ["R10"]
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def compute_true_margin(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """margin_true = logit_true - max(logit_other)."""
    N = len(targets)
    margins = np.zeros(N, dtype=np.float32)
    for i in range(N):
        y = targets[i]
        z = logits[i]
        z_true = z[y]
        other_mask = np.ones(7, dtype=bool)
        other_mask[y] = False
        z_other_max = np.max(z[other_mask])
        margins[i] = z_true - z_other_max
    return margins


def paired_bootstrap_delta(deltas: np.ndarray, B: int = 2000, seed: int = 42) -> dict:
    """Percentile bootstrap 95% CI for paired deltas."""
    rng = np.random.RandomState(seed)
    N = len(deltas)
    if N == 0:
        return {"mean": 0.0, "median": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0}
    means = [float(np.mean(rng.choice(deltas, size=N, replace=True))) for _ in range(B)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(np.mean(deltas)),
        "median": float(np.median(deltas)),
        "std": float(np.std(deltas)),
        "ci_95_lower": float(low),
        "ci_95_upper": float(high),
    }


def find_first_rescue_stage(margins: list[float], stages: list[str]) -> str:
    """Earliest stage where margin > 0 and remains > 0 for at least the next stage."""
    K = len(margins)
    if margins[0] > 0:
        # Check if it was already correct from R0
        return "ALREADY_CORRECT_AT_R0"

    for i in range(1, K):
        if margins[i] > 0:
            # Check persistence for next stage if available
            if i + 1 < K:
                if margins[i + 1] > 0:
                    return f"FIRST_RESCUE_{stages[i]}"
            else:
                return f"LATE_RESCUE_{stages[i]}"
    return "NEVER_RESCUED"


def find_first_collapse_stage(margins: list[float], stages: list[str]) -> str:
    """Earliest stage where margin changes from > 0 to < 0 and remains < 0 for at least next stage."""
    K = len(margins)
    if margins[0] <= 0:
        return "ALREADY_WRONG_AT_R0"

    for i in range(1, K):
        if margins[i] <= 0:
            if i + 1 < K:
                if margins[i + 1] <= 0:
                    return f"FIRST_COLLAPSE_{stages[i]}"
            else:
                return f"LATE_COLLAPSE_{stages[i]}"
    return "NEVER_COLLAPSED"


def main():
    start_time = time.time()
    print("Running A6.2 & A6.3 Margin & Rescue Analysis...", flush=True)

    # 1. Load sample groups
    groups_df = pd.read_csv(AUDIT_DIR / "a6_sample_groups.csv")

    # Load representations
    pub_data = np.load(AUDIT_DIR / "a6_public_features.npz")
    priv_data = np.load(AUDIT_DIR / "a6_private_features.npz")
    train_data = np.load(AUDIT_DIR / "a6_train_features.npz")

    # Load probe parameters
    probes = {}
    probe_metrics_doc = {"v21": {}, "v22": {}}

    for m in ["v21", "v22"]:
        for s in STAGES:
            ckpt = np.load(CACHE_DIR / f"probe_{m}_{s}.npz")
            meta = json.load(open(CACHE_DIR / f"meta_{m}_{s}.json"))
            probes[(m, s)] = {
                "coef": ckpt["coef"],
                "intercept": ckpt["intercept"],
                "mean": ckpt["mean"],
                "scale": ckpt["scale"],
            }
            probe_metrics_doc[m][s] = meta

    (AUDIT_DIR / "a6_stage_probe_metrics.json").write_text(json.dumps(probe_metrics_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6_stage_probe_metrics.json'}")

    # 2. Compute probe logits and margins for Public and Private
    margins_by_stage = {
        "public": {m: {s: None for s in ALL_STAGES_WITH_LOGITS} for m in ["v21", "v22"]},
        "private": {m: {s: None for s in ALL_STAGES_WITH_LOGITS} for m in ["v21", "v22"]},
    }

    for split, data in [("public", pub_data), ("private", priv_data)]:
        targets = data["targets"]
        for m in ["v21", "v22"]:
            for s in STAGES:
                X = data[f"{m}_{s}"]
                p = probes[(m, s)]
                Xs = (X - p["mean"]) / p["scale"]
                logits = Xs @ p["coef"].T + p["intercept"]
                margins = compute_true_margin(logits, targets)
                margins_by_stage[split][m][s] = margins

            # Final logits stage R10
            l_final = data[f"{m}_R10"]
            margins_by_stage[split][m]["R10"] = compute_true_margin(l_final, targets)

    # 3. Build sample-level margin table
    sample_margin_rows = []
    first_events_records = []

    for _, row in groups_df.iterrows():
        split = row["split"]
        idx = int(row["row_index"])
        cat = row["sample_category"]
        y = int(row["true_label"])
        is_clean = bool(row["is_clean"])
        is_strict_clean = bool(row["is_strict_clean"])

        m21_stages = [float(margins_by_stage[split]["v21"][s][idx]) for s in ALL_STAGES_WITH_LOGITS]
        m22_stages = [float(margins_by_stage[split]["v22"][s][idx]) for s in ALL_STAGES_WITH_LOGITS]

        # Delta margins
        if cat == "V21_CORRECT_V22_WRONG":
            deltas = [m21_stages[i] - m22_stages[i] for i in range(len(ALL_STAGES_WITH_LOGITS))]
            first_rescue = find_first_rescue_stage(m21_stages, ALL_STAGES_WITH_LOGITS)
            first_collapse = find_first_collapse_stage(m22_stages, ALL_STAGES_WITH_LOGITS)
        elif cat == "V22_CORRECT_V21_WRONG":
            deltas = [m22_stages[i] - m21_stages[i] for i in range(len(ALL_STAGES_WITH_LOGITS))]
            first_rescue = find_first_rescue_stage(m22_stages, ALL_STAGES_WITH_LOGITS)
            first_collapse = find_first_collapse_stage(m21_stages, ALL_STAGES_WITH_LOGITS)
        else:
            deltas = [0.0] * len(ALL_STAGES_WITH_LOGITS)
            first_rescue = "N/A"
            first_collapse = "N/A"

        row_dict = {
            "split": split,
            "row_index": idx,
            "true_label": y,
            "true_class": CLASS_NAMES[y],
            "sample_category": cat,
            "is_clean": is_clean,
            "is_strict_clean": is_strict_clean,
            "first_rescue_event": first_rescue,
            "first_collapse_event": first_collapse,
        }
        for i, s in enumerate(ALL_STAGES_WITH_LOGITS):
            row_dict[f"margin_v21_{s}"] = m21_stages[i]
            row_dict[f"margin_v22_{s}"] = m22_stages[i]
            row_dict[f"delta_margin_{s}"] = deltas[i]

        sample_margin_rows.append(row_dict)

        if cat in ["V21_CORRECT_V22_WRONG", "V22_CORRECT_V21_WRONG"]:
            first_events_records.append({
                "split": split,
                "row_index": idx,
                "true_class": CLASS_NAMES[y],
                "category": cat,
                "is_clean": is_clean,
                "first_rescue": first_rescue,
                "first_collapse": first_collapse,
            })

    # Save sample-level margins CSV
    margin_csv_path = AUDIT_DIR / "a6_stage_margin_samples.csv"
    pd.DataFrame(sample_margin_rows).to_csv(margin_csv_path, index=False)
    print(f"Saved {margin_csv_path} ({len(sample_margin_rows)} rows)")

    # 4. Aggregate Margins and Paired Bootstrap CIs
    margin_df = pd.DataFrame(sample_margin_rows)
    margin_summary_doc = {}

    target_groups = ["V21_CORRECT_V22_WRONG", "V22_CORRECT_V21_WRONG", "BOTH_CORRECT", "BOTH_WRONG"]

    for split in ["public", "private", "combined"]:
        margin_summary_doc[split] = {}
        sub_df = margin_df if split == "combined" else margin_df[margin_df["split"] == split]

        for grp in target_groups:
            grp_df = sub_df[sub_df["sample_category"] == grp]
            clean_grp_df = grp_df[grp_df["is_clean"]]

            def summarize_group(df_in):
                stage_res = {}
                for s in ALL_STAGES_WITH_LOGITS:
                    v21_m = df_in[f"margin_v21_{s}"].values
                    v22_m = df_in[f"margin_v22_{s}"].values
                    delta_m = df_in[f"delta_margin_{s}"].values

                    stage_res[s] = {
                        "v21_margin_mean": float(np.mean(v21_m)),
                        "v21_margin_median": float(np.median(v21_m)),
                        "v22_margin_mean": float(np.mean(v22_m)),
                        "v22_margin_median": float(np.median(v22_m)),
                        "paired_delta": paired_bootstrap_delta(delta_m, B=2000, seed=42),
                    }
                return {"sample_count": len(df_in), "stages": stage_res}

            margin_summary_doc[split][grp] = {
                "all": summarize_group(grp_df),
                "clean": summarize_group(clean_grp_df),
            }

    (AUDIT_DIR / "a6_stage_margin_summary.json").write_text(json.dumps(margin_summary_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6_stage_margin_summary.json'}")

    # 5. First Rescue & Collapse Distributions
    events_df = pd.DataFrame(first_events_records)
    events_summary = {}

    for split in ["public", "private", "combined"]:
        events_summary[split] = {}
        sub_e = events_df if split == "combined" else events_df[events_df["split"] == split]

        for grp in ["V21_CORRECT_V22_WRONG", "V22_CORRECT_V21_WRONG"]:
            grp_e = sub_e[sub_e["category"] == grp]
            clean_e = grp_e[grp_e["is_clean"]]

            def dist_events(df_in):
                N = len(df_in)
                res_counts = df_in["first_rescue"].value_counts().to_dict()
                col_counts = df_in["first_collapse"].value_counts().to_dict()
                return {
                    "count": N,
                    "first_rescue_distribution": {k: int(v) for k, v in res_counts.items()},
                    "first_rescue_proportions": {k: float(v / N) for k, v in res_counts.items()} if N > 0 else {},
                    "first_collapse_distribution": {k: int(v) for k, v in col_counts.items()},
                    "first_collapse_proportions": {k: float(v / N) for k, v in col_counts.items()} if N > 0 else {},
                }

            events_summary[split][grp] = {
                "all": dist_events(grp_e),
                "clean": dist_events(clean_e),
            }

    (AUDIT_DIR / "a6_first_rescue_collapse.json").write_text(json.dumps(events_summary, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6_first_rescue_collapse.json'}")

    print(f"\nA6.2 & A6.3 completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
