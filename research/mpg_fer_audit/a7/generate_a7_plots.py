"""Generate all required plots for MPG-FER A7 Audit.

Required plots:
1. a7_single_vs_cross_depth_public.png
2. a7_dimension_control_public.png
3. a7_cross_depth_delta_by_stage.png
4. a7_probe_error_overlap.png
5. a7_classwise_l2_l5_complementarity.png
6. a7_cka_depth_matrix.png
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
A7_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a7"

with open(A7_DIR / "a7_master_results.json", "r", encoding="utf-8") as f:
    master = json.load(f)

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

# -------------------------------------------------------------
# PLOT 1: a7_single_vs_cross_depth_public.png
# -------------------------------------------------------------
def plot_single_vs_cross_depth():
    fig, ax = plt.subplots(figsize=(12, 6))

    stages_single = ["PRE", "L1", "L2", "L5", "Motif Readout", "Fusion"]
    pairs_cross = ["PRE+L5", "L1+L5", "L2+L5", "L1+L2+L5", "L2+Motif Readout", "L2+Fusion"]

    labels = stages_single + ["|"] + pairs_cross
    v23_vals = []
    v22_vals = []

    for s in stages_single:
        v23_vals.append(master["single_stage"]["v23"][s]["public_acc"] * 100)
        v22_vals.append(master["single_stage"]["v22"][s]["public_acc"] * 100)

    v23_vals.append(np.nan)
    v22_vals.append(np.nan)

    for p in pairs_cross:
        v23_vals.append(master["raw_concat"]["v23"][p]["public_acc"] * 100)
        v22_vals.append(master["raw_concat"]["v22"][p]["public_acc"] * 100)

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, v23_vals, width, label="v2.3 (Residual Scale [0.5,0.5,1,1,1])", color="#2b5c8f")
    ax.bar(x + width/2, v22_vals, width, label="v2.2 Baseline (Scale [1,1,1,1,1])", color="#e27c38")

    ax.axvline(x=len(stages_single), color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("Public Accuracy (%)")
    ax.set_title("A7: Single-Stage Probes vs. Cross-Depth Raw Concatenation (PublicTest)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(52, 70)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(A7_DIR / "a7_single_vs_cross_depth_public.png")
    plt.close(fig)
    print("Saved a7_single_vs_cross_depth_public.png")


# -------------------------------------------------------------
# PLOT 2: a7_dimension_control_public.png
# -------------------------------------------------------------
def plot_dimension_control():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    pairs = ["PRE+L5", "L1+L5", "L2+L5", "L1+L2+L5", "L2+Motif Readout", "L2+Fusion"]

    for ax, model, model_name in [(ax1, "v23", "MPG-FER v2.3"), (ax2, "v22", "MPG-FER v2.2")]:
        raw_deltas = [master["raw_concat"][model][p]["delta_acc_vs_late"] * 100 for p in pairs]
        pca_deltas = [master["pca_controls"][model][p]["delta_acc_vs_late"] * 100 for p in pairs]

        # Duplicate deltas
        dup_deltas = []
        for p in pairs:
            if p in ["PRE+L5", "L1+L5", "L2+L5"]:
                dup_deltas.append(master["duplicate_controls"][model]["L5+L5"]["delta_acc_vs_late"] * 100)
            elif p == "L1+L2+L5":
                dup_deltas.append(master["duplicate_controls"][model]["L5+L5+L5"]["delta_acc_vs_late"] * 100)
            elif p == "L2+Motif Readout":
                dup_deltas.append(master["duplicate_controls"][model]["Motif Readout+Motif Readout"]["delta_acc_vs_late"] * 100)
            elif p == "L2+Fusion":
                dup_deltas.append(master["duplicate_controls"][model]["Fusion+Fusion"]["delta_acc_vs_late"] * 100)

        x = np.arange(len(pairs))
        width = 0.25

        ax.bar(x - width, raw_deltas, width, label="Raw Concat (Increased Dim)", color="#4a7bb0")
        ax.bar(x, dup_deltas, width, label="Duplicate-Late Control", color="#9aa0a6")
        ax.bar(x + width, pca_deltas, width, label="Train-Only PCA (Matched Dim)", color="#d95f02")

        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.axhline(0.75, color="green", linestyle=":", linewidth=1.0, label="Actionable Threshold (+0.75pp)")
        ax.set_title(f"{model_name}: Accuracy Delta vs Late Baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=35, ha="right")
        ax.set_xlabel("Cross-Depth Pair")
        if ax == ax1:
            ax.set_ylabel("Accuracy Delta (percentage points)")
        ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    fig.savefig(A7_DIR / "a7_dimension_control_public.png")
    plt.close(fig)
    print("Saved a7_dimension_control_public.png")


# -------------------------------------------------------------
# PLOT 3: a7_cross_depth_delta_by_stage.png
# -------------------------------------------------------------
def plot_cross_depth_delta_by_stage():
    fig, ax = plt.subplots(figsize=(10, 5))

    pairs = ["PRE+L5", "L1+L5", "L2+L5", "L1+L2+L5", "L2+Motif Readout", "L2+Fusion"]

    v23_raw = [master["raw_concat"]["v23"][p]["delta_acc_vs_late"] * 100 for p in pairs]
    v23_pca = [master["pca_controls"]["v23"][p]["delta_acc_vs_late"] * 100 for p in pairs]
    v22_raw = [master["raw_concat"]["v22"][p]["delta_acc_vs_late"] * 100 for p in pairs]
    v22_pca = [master["pca_controls"]["v22"][p]["delta_acc_vs_late"] * 100 for p in pairs]

    x = np.arange(len(pairs))
    width = 0.2

    ax.bar(x - 1.5*width, v23_raw, width, label="v2.3 Raw Concat", color="#1f77b4")
    ax.bar(x - 0.5*width, v23_pca, width, label="v2.3 PCA Control", color="#aec7e8")
    ax.bar(x + 0.5*width, v22_raw, width, label="v2.2 Raw Concat", color="#ff7f0e")
    ax.bar(x + 1.5*width, v22_pca, width, label="v2.2 PCA Control", color="#ffbb78")

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.axhline(0.75, color="green", linestyle=":", linewidth=1.0, label="Actionable Threshold (+0.75pp)")
    ax.set_ylabel("Public Accuracy Delta vs Late Baseline (pp)")
    ax.set_title("A7: Cross-Depth Prediction Delta by Pair and Model")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=25, ha="right")
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(A7_DIR / "a7_cross_depth_delta_by_stage.png")
    plt.close(fig)
    print("Saved a7_cross_depth_delta_by_stage.png")


# -------------------------------------------------------------
# PLOT 4: a7_probe_error_overlap.png
# -------------------------------------------------------------
def plot_probe_error_overlap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    comparisons = ["PRE_vs_L5", "L1_vs_L5", "L2_vs_L5"]
    comp_labels = ["PRE vs L5", "L1 vs L5", "L2 vs L5"]

    for ax, model, model_name in [(ax1, "v23", "MPG-FER v2.3"), (ax2, "v22", "MPG-FER v2.2")]:
        both_c = [master["probe_complementarity_sets"][model][c]["both_correct"] for c in comparisons]
        early_c = [master["probe_complementarity_sets"][model][c]["early_correct_late_wrong"] for c in comparisons]
        late_c = [master["probe_complementarity_sets"][model][c]["early_wrong_late_correct"] for c in comparisons]
        both_w = [master["probe_complementarity_sets"][model][c]["both_wrong"] for c in comparisons]

        x = np.arange(len(comparisons))
        width = 0.5

        ax.bar(x, both_c, width, label="Both Correct", color="#2ca02c")
        ax.bar(x, late_c, width, bottom=both_c, label="Late Correct, Early Wrong", color="#1f77b4")
        ax.bar(x, early_c, width, bottom=np.array(both_c) + np.array(late_c), label="Early Correct, Late Wrong (Complementary Pool)", color="#ff7f0e")
        ax.bar(x, both_w, width, bottom=np.array(both_c) + np.array(late_c) + np.array(early_c), label="Both Wrong", color="#d62728")

        ax.set_title(f"{model_name}: Sample Breakdown (Total N=3589)")
        ax.set_xticks(x)
        ax.set_xticklabels(comp_labels)
        ax.set_ylabel("Number of PublicTest Samples")
        ax.set_ylim(0, 3700)
        if ax == ax1:
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(A7_DIR / "a7_probe_error_overlap.png")
    plt.close(fig)
    print("Saved a7_probe_error_overlap.png")


# -------------------------------------------------------------
# PLOT 5: a7_classwise_l2_l5_complementarity.png
# -------------------------------------------------------------
def plot_classwise_l2_l5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    classes = CLASS_NAMES

    for ax, model, model_name in [(ax1, "v23", "MPG-FER v2.3"), (ax2, "v22", "MPG-FER v2.2")]:
        l5_accs = [master["classwise_summary"][model][c]["L5_probe_acc"] * 100 for c in classes]
        l2_accs = [master["classwise_summary"][model][c]["L2_probe_acc"] * 100 for c in classes]
        raw_accs = [master["classwise_summary"][model][c]["L2_L5_raw_acc"] * 100 for c in classes]
        pca_accs = [master["classwise_summary"][model][c]["L2_L5_pca_acc"] * 100 for c in classes]

        x = np.arange(len(classes))
        width = 0.2

        ax.bar(x - 1.5*width, l2_accs, width, label="L2 Probe", color="#17becf")
        ax.bar(x - 0.5*width, l5_accs, width, label="L5 Probe (Late Baseline)", color="#bcbd22")
        ax.bar(x + 0.5*width, raw_accs, width, label="L2+L5 Raw Concat", color="#1f77b4")
        ax.bar(x + 1.5*width, pca_accs, width, label="PCA(L2+L5)->192", color="#ff7f0e")

        ax.set_title(f"{model_name}: Classwise Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=25)
        if ax == ax1:
            ax.set_ylabel("Class Accuracy (%)")
            ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(A7_DIR / "a7_classwise_l2_l5_complementarity.png")
    plt.close(fig)
    print("Saved a7_classwise_l2_l5_complementarity.png")


# -------------------------------------------------------------
# PLOT 6: a7_cka_depth_matrix.png
# -------------------------------------------------------------
def plot_cka():
    fig, ax = plt.subplots(figsize=(8, 5))

    pairs = ["PRE vs L5", "L1 vs L5", "L2 vs L5", "L5 vs Readout", "L5 vs Fusion"]

    v23_tr = [
        master["cka"]["v23"]["train"]["PRE_vs_L5"],
        master["cka"]["v23"]["train"]["L1_vs_L5"],
        master["cka"]["v23"]["train"]["L2_vs_L5"],
        master["cka"]["v23"]["train"]["L5_vs_MotifReadout"],
        master["cka"]["v23"]["train"]["L5_vs_Fusion"],
    ]
    v23_pub = [
        master["cka"]["v23"]["public"]["PRE_vs_L5"],
        master["cka"]["v23"]["public"]["L1_vs_L5"],
        master["cka"]["v23"]["public"]["L2_vs_L5"],
        master["cka"]["v23"]["public"]["L5_vs_MotifReadout"],
        master["cka"]["v23"]["public"]["L5_vs_Fusion"],
    ]
    v22_pub = [
        master["cka"]["v22"]["public"]["PRE_vs_L5"],
        master["cka"]["v22"]["public"]["L1_vs_L5"],
        master["cka"]["v22"]["public"]["L2_vs_L5"],
        master["cka"]["v22"]["public"]["L5_vs_MotifReadout"],
        master["cka"]["v22"]["public"]["L5_vs_Fusion"],
    ]

    x = np.arange(len(pairs))
    width = 0.25

    ax.bar(x - width, v23_tr, width, label="v2.3 Train CKA", color="#2b5c8f")
    ax.bar(x, v23_pub, width, label="v2.3 Public CKA", color="#4a7bb0")
    ax.bar(x + width, v22_pub, width, label="v2.2 Public CKA", color="#e27c38")

    ax.set_ylabel("Linear CKA Similarity")
    ax.set_title("A7: Representation Redundancy (Linear CKA)")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(A7_DIR / "a7_cka_depth_matrix.png")
    plt.close(fig)
    print("Saved a7_cka_depth_matrix.png")


def main():
    print("Generating A7 plots...")
    plot_single_vs_cross_depth()
    plot_dimension_control()
    plot_cross_depth_delta_by_stage()
    plot_probe_error_overlap()
    plot_classwise_l2_l5()
    plot_cka()
    print("All A7 plots generated successfully.")


if __name__ == "__main__":
    main()
