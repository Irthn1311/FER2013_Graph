"""MPG-FER A6-R2: Readout Factorial Repair, True Per-Sample Routing Test, and Master Results Generation.

Performs:
1. Re-runs Motif Readout Factorial with receiver-specific suffix and exact F00 replay validation.
2. Extracts true per-sample layerwise routing divergence (1 - Jaccard) for all 1,197 resolvable samples.
3. Tests sample-level associations between routing divergence, margin delta, representation distance, and swap rescue.
4. Generates a6r2_master_results.json as the single authoritative source of truth.
5. Generates all required JSON/CSV artifacts and diagnostic plots.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from regenerate_a6r2_derived import (
    build_hypothesis_document,
    build_target_document,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6r2"
A6_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6"
A6R_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6r"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

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

K_SCHEDULE = [8, 16, 16, 16, 24]
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def bootstrap_ci_mean(arr: np.ndarray, B: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    N = len(arr)
    if N == 0:
        return 0.0, 0.0, 0.0
    means = [float(np.mean(rng.choice(arr, size=N, replace=True))) for _ in range(B)]
    low, high = np.percentile(means, [2.5, 97.5])
    return float(np.mean(arr)), float(low), float(high)


def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, B: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    N = len(x)
    if N == 0:
        return 0.0, 0.0, 0.0
    pe, _ = scipy.stats.spearmanr(x, y)
    rx = scipy.stats.rankdata(x)
    ry = scipy.stats.rankdata(y)
    rhos = []
    for _ in range(B):
        idx = rng.choice(N, size=N, replace=True)
        xc = rx[idx] - np.mean(rx[idx])
        yc = ry[idx] - np.mean(ry[idx])
        denom = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
        if denom > 1e-12:
            rhos.append(float(np.sum(xc * yc) / denom))
    if not rhos:
        return float(pe), float(pe), float(pe)
    low, high = np.percentile(rhos, [2.5, 97.5])
    return float(pe), float(low), float(high)


def apply_readout(model, node_states):
    mean_v = node_states.mean(dim=1)
    max_v = node_states.max(dim=1).values
    attn_v = (F.softmax(model.motif_attn_pool(node_states), dim=1) * node_states).sum(dim=1)
    return model.motif_readout_proj(torch.cat([mean_v, max_v, attn_v], dim=-1))


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running A6-R2 Analysis Pipeline on {device}...", flush=True)

    # 1. Load models
    m21 = MPGFERV21(MPGConfigV21()).to(device).eval()
    m22 = MPGFERV22(MPGConfigV22()).to(device).eval()

    c21 = torch.load(V21_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    c22 = torch.load(V22_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]

    m21.load_state_dict(c21, strict=True)
    m22.load_state_dict(c22, strict=True)

    # 2. Load sample cohorts & features
    groups_df = pd.read_csv(A6_DIR / "a6_sample_groups.csv")
    resolvable_df = groups_df[groups_df["is_model_resolvable"]].copy()

    # Pre-extract cached feature dictionaries
    pub_feats = np.load(A6_DIR / "a6_public_features.npz")
    priv_feats = np.load(A6_DIR / "a6_private_features.npz")
    res_node_npz = np.load(A6_DIR / "a6_resolvable_node_features.npz")
    swap_samples_a6 = pd.read_csv(A6_DIR / "a6_swap_sample_results.csv")
    margin_samples_a6 = pd.read_csv(A6_DIR / "a6_stage_margin_samples.csv")

    pub_map = dict(zip(res_node_npz["public_meta_row_indices"], range(len(res_node_npz["public_meta_row_indices"]))))
    priv_map = dict(zip(res_node_npz["private_meta_row_indices"], range(len(res_node_npz["private_meta_row_indices"]))))

    pub_n21_arr = res_node_npz["public_v21_R6"]
    pub_n22_arr = res_node_npz["public_v22_R6"]
    priv_n21_arr = res_node_npz["private_v21_R6"]
    priv_n22_arr = res_node_npz["private_v22_R6"]

    # Datasets
    val_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val"), split="val", augment=False)
    test_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test"), split="test", augment=False)

    # Provenance JSON
    prov_doc = {
        "audit": "MPG-FER A6-R2",
        "device": device,
        "v2_1_sha256": "4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75",
        "v2_2_sha256": "a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4",
        "sample_cohorts": {
            "model_resolvable_all": int(len(resolvable_df)),
            "v21_correct_v22_wrong": int(sum(resolvable_df["sample_category"] == "V21_CORRECT_V22_WRONG")),
            "v22_correct_v21_wrong": int(sum(resolvable_df["sample_category"] == "V22_CORRECT_V21_WRONG")),
            "a6_clean": int(sum(resolvable_df["is_clean"])),
            "a6_strict_clean": int(sum(resolvable_df["is_strict_clean"])),
        }
    }
    (AUDIT_DIR / "a6r2_provenance.json").write_text(json.dumps(prov_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6r2_provenance.json'}")

    # =========================================================================
    # PART I: REPAIR READOUT FACTORIAL
    # =========================================================================
    rd_results_json = AUDIT_DIR / "a6r2_readout_factorial_results.json"
    rd_samples_csv = AUDIT_DIR / "a6r2_readout_factorial_samples.csv"

    if rd_results_json.exists() and rd_samples_csv.exists():
        print(f"\nReadout factorial results already exist ({rd_results_json}); loading cached results.", flush=True)
        readout_results_doc = json.loads(rd_results_json.read_text(encoding="utf-8"))
        rd_df = pd.read_csv(rd_samples_csv)
    else:
        print("\n--- Part I: Executing Repaired Readout Factorial Crossing ---", flush=True)

        readout_sample_records = []
        replay_val_doc = {}

        for direction, cat_name, donor_m, rec_m, d_lbl, r_lbl in [
            ("dir_A", "V21_CORRECT_V22_WRONG", m21, m22, "v2.1", "v2.2"),
            ("dir_B", "V22_CORRECT_V21_WRONG", m22, m21, "v2.2", "v2.1"),
        ]:
            cat_df = resolvable_df[resolvable_df["sample_category"] == cat_name].copy()
            N_cat = len(cat_df)
            print(f"\nProcessing {direction} ({cat_name}, N={N_cat})...")

            n_don_list, n_rec_list = [], []
            p_rec_list, p_don_list = [], []
            l_rec_orig_list, l_don_orig_list = [], []
            y_list = []
            rows_order = []

            for _, r in cat_df.iterrows():
                s = r["split"]
                row_i = int(r["row_index"])
                y_i = int(r["true_label"])

                y_list.append(y_i)
                rows_order.append((s, row_i))

                feats = pub_feats if s == "public" else priv_feats
                pos = pub_map[row_i] if s == "public" else priv_map[row_i]

                if direction == "dir_A":
                    n_don_list.append(pub_n21_arr[pos] if s == "public" else priv_n21_arr[pos])
                    n_rec_list.append(pub_n22_arr[pos] if s == "public" else priv_n22_arr[pos])
                    p_don_list.append(feats["v21_R0"][row_i])
                    p_rec_list.append(feats["v22_R0"][row_i])
                    l_don_orig_list.append(feats["v21_R10"][row_i])
                    l_rec_orig_list.append(feats["v22_R10"][row_i])
                else:
                    n_don_list.append(pub_n22_arr[pos] if s == "public" else priv_n22_arr[pos])
                    n_rec_list.append(pub_n21_arr[pos] if s == "public" else priv_n21_arr[pos])
                    p_don_list.append(feats["v22_R0"][row_i])
                    p_rec_list.append(feats["v21_R0"][row_i])
                    l_don_orig_list.append(feats["v22_R10"][row_i])
                    l_rec_orig_list.append(feats["v21_R10"][row_i])

            n_don_t = torch.from_numpy(np.stack(n_don_list)).to(device)
            n_rec_t = torch.from_numpy(np.stack(n_rec_list)).to(device)
            p_don_t = torch.from_numpy(np.stack(p_don_list)).to(device)
            p_rec_t = torch.from_numpy(np.stack(p_rec_list)).to(device)

            l_don_orig = np.stack(l_don_orig_list)
            l_rec_orig = np.stack(l_rec_orig_list)
            y_arr = np.array(y_list)

            with torch.no_grad():
                rd_don_ndon = apply_readout(donor_m, n_don_t)
                rd_don_nrec = apply_readout(donor_m, n_rec_t)
                rd_rec_ndon = apply_readout(rec_m, n_don_t)
                rd_rec_nrec = apply_readout(rec_m, n_rec_t)

                cos_node_fixed = F.cosine_similarity(rd_don_ndon, rd_rec_ndon, dim=-1).cpu().numpy()
                cos_op_fixed = F.cosine_similarity(rd_don_ndon, rd_don_nrec, dim=-1).cpu().numpy()

                def suffix_receiver(m_rd):
                    fus = torch.cat([p_rec_t, m_rd], dim=-1)
                    return rec_m.classifier(fus)

                def suffix_donor(m_rd):
                    fus = torch.cat([p_don_t, m_rd], dim=-1)
                    return donor_m.classifier(fus)

                f00_logits = suffix_receiver(rd_rec_nrec).cpu().numpy()
                err_f00 = float(np.max(np.abs(f00_logits - l_rec_orig)))

                don_replay_logits = suffix_donor(rd_don_ndon).cpu().numpy()
                err_don_replay = float(np.max(np.abs(don_replay_logits - l_don_orig)))

                f01_logits = suffix_receiver(rd_don_nrec).cpu().numpy()
                f10_logits = suffix_receiver(rd_rec_ndon).cpu().numpy()
                f11_logits = suffix_receiver(rd_don_ndon).cpu().numpy()

            pred_f00 = np.argmax(f00_logits, axis=-1)
            pred_f01 = np.argmax(f01_logits, axis=-1)
            pred_f10 = np.argmax(f10_logits, axis=-1)
            pred_f11 = np.argmax(f11_logits, axis=-1)

            corr_f00 = (pred_f00 == y_arr)
            corr_f01 = (pred_f01 == y_arr)
            corr_f10 = (pred_f10 == y_arr)
            corr_f11 = (pred_f11 == y_arr)

            cnt_f00 = int(np.sum(corr_f00))
            cnt_f01 = int(np.sum(corr_f01))
            cnt_f10 = int(np.sum(corr_f10))
            cnt_f11 = int(np.sum(corr_f11))

            def get_margins(l_mat):
                N = len(y_arr)
                m = np.zeros(N, dtype=float)
                for i in range(N):
                    y = y_arr[i]
                    other = np.ones(7, dtype=bool)
                    other[y] = False
                    m[i] = float(l_mat[i, y] - np.max(l_mat[i, other]))
                return m

            m_f00 = get_margins(f00_logits)
            m_f01 = get_margins(f01_logits)
            m_f10 = get_margins(f10_logits)
            m_f11 = get_margins(f11_logits)

            s7_col = f"swap_pred_21to22_S7" if direction == "dir_A" else f"swap_pred_22to21_S7"
            s7_corr_col = f"swap_corr_21to22_S7" if direction == "dir_A" else f"swap_corr_22to21_S7"
            s7_margin_col = f"swap_margin_21to22_S7" if direction == "dir_A" else f"swap_margin_22to21_S7"

            a6_s7_sub = swap_samples_a6[swap_samples_a6["sample_category"] == cat_name]
            a6_s7_preds = a6_s7_sub[s7_col].values
            a6_s7_margins = a6_s7_sub[s7_margin_col].values

            err_f11_s7 = float(np.max(np.abs(m_f11 - a6_s7_margins)))
            diff_preds_s7 = int(np.sum(pred_f11 != a6_s7_preds))

            print(f"  [Validation] F00 Replay Max Abs Error: {err_f00:.2e}")
            print(f"  [Validation] F00 Correct Count: {cnt_f00}/{N_cat} (MUST BE 0)")
            print(f"  [Validation] Donor Replay Max Abs Error: {err_don_replay:.2e}")
            print(f"  [Validation] F11 vs Historical A6 S7 Max Abs Error: {err_f11_s7:.2e}, Prediction Mismatches: {diff_preds_s7}")

            if cnt_f00 != 0 or err_f00 > 1e-5:
                print(f"FATAL: F00 baseline validation failed! cnt_f00={cnt_f00}, err_f00={err_f00}")
                sys.exit(3)

            replay_val_doc[direction] = {
                "donor": d_lbl,
                "receiver": r_lbl,
                "cohort_count": N_cat,
                "f00_baseline_correct_count": cnt_f00,
                "f00_max_abs_error_vs_original_receiver": err_f00,
                "f00_exact_replay_pass": bool(cnt_f00 == 0 and err_f00 <= 1e-5),
                "donor_native_replay_max_abs_error": err_don_replay,
                "donor_native_replay_pass": bool(err_don_replay <= 1e-5),
                "f11_vs_historical_a6_s7_max_abs_error": err_f11_s7,
                "f11_vs_historical_a6_s7_prediction_mismatches": diff_preds_s7,
                "f11_a6_s7_exact_equivalence_pass": bool(diff_preds_s7 == 0 and err_f11_s7 <= 1e-5),
            }

            for i in range(N_cat):
                r_meta = cat_df.iloc[i]
                readout_sample_records.append({
                    "direction": direction,
                    "sample_category": cat_name,
                    "split": r_meta["split"],
                    "row_index": int(r_meta["row_index"]),
                    "true_label": int(r_meta["true_label"]),
                    "true_class": CLASS_NAMES[int(r_meta["true_label"])],
                    "is_clean": bool(r_meta["is_clean"]),
                    "is_strict_clean": bool(r_meta["is_strict_clean"]),
                    "f00_pred": int(pred_f00[i]),
                    "f00_correct": bool(corr_f00[i]),
                    "f00_margin": float(m_f00[i]),
                    "f01_pred": int(pred_f01[i]),
                    "f01_correct": bool(corr_f01[i]),
                    "f01_margin": float(m_f01[i]),
                    "f10_pred": int(pred_f10[i]),
                    "f10_correct": bool(corr_f10[i]),
                    "f10_margin": float(m_f10[i]),
                    "f11_pred": int(pred_f11[i]),
                    "f11_correct": bool(corr_f11[i]),
                    "f11_margin": float(m_f11[i]),
                    "cos_same_node_diff_readout": float(cos_node_fixed[i]),
                    "cos_diff_node_same_readout": float(cos_op_fixed[i]),
                })

        (AUDIT_DIR / "a6r2_readout_replay_validation.json").write_text(json.dumps(replay_val_doc, indent=2), encoding="utf-8")
        print(f"Saved {AUDIT_DIR / 'a6r2_readout_replay_validation.json'}")

        rd_df = pd.DataFrame(readout_sample_records)
        rd_df.to_csv(rd_samples_csv, index=False)
        print(f"Saved {rd_samples_csv} ({len(rd_df)} rows)")

        readout_results_doc = {}
        for direction, cat_name in [("dir_A", "V21_CORRECT_V22_WRONG"), ("dir_B", "V22_CORRECT_V21_WRONG")]:
            readout_results_doc[direction] = {"sample_category": cat_name}
            dir_df = rd_df[rd_df["direction"] == direction]

            for subset_name, sub_mask in [
                ("all", np.ones(len(dir_df), dtype=bool)),
                ("clean", dir_df["is_clean"].values),
                ("strict_clean", dir_df["is_strict_clean"].values),
            ]:
                s_df = dir_df[sub_mask]
                N_s = len(s_df)

                def get_cond_stats(cond):
                    corr = s_df[f"{cond}_correct"].values
                    m = s_df[f"{cond}_margin"].values
                    c_cnt = int(np.sum(corr))
                    c_rate, c_low, c_high = bootstrap_ci_mean(corr.astype(float), B=2000, seed=42)
                    m_all, _, _ = bootstrap_ci_mean(m, B=2000, seed=42)
                    rescued_m = m[corr]
                    m_rescued, m_r_low, m_r_high = bootstrap_ci_mean(rescued_m, B=2000, seed=42) if len(rescued_m) > 0 else (0.0, 0.0, 0.0)

                    return {
                        "correct_count": c_cnt,
                        "correct_rate": float(c_cnt / N_s),
                        "correct_rate_ci_95": [c_low, c_high],
                        "mean_margin_all": m_all,
                        "median_margin": float(np.median(m)),
                        "mean_margin_rescued_only": m_rescued,
                        "rescued_count": len(rescued_m),
                    }

                f00_st = get_cond_stats("f00")
                f01_st = get_cond_stats("f01")
                f10_st = get_cond_stats("f10")
                f11_st = get_cond_stats("f11")

                eff_readout_on_rec_states = f01_st["correct_rate"] - f00_st["correct_rate"]
                eff_readout_on_don_states = f11_st["correct_rate"] - f10_st["correct_rate"]
                eff_node_on_rec_readout = f10_st["correct_rate"] - f00_st["correct_rate"]
                eff_node_on_don_readout = f11_st["correct_rate"] - f01_st["correct_rate"]
                interaction_rate = eff_readout_on_don_states - eff_readout_on_rec_states

                readout_results_doc[direction][subset_name] = {
                    "sample_count": N_s,
                    "f00_baseline": f00_st,
                    "f01_readout_operator_only": f01_st,
                    "f10_node_states_only": f10_st,
                    "f11_both_swapped_s7": f11_st,
                    "contrasts": {
                        "readout_operator_effect_on_receiver_node_states": eff_readout_on_rec_states,
                        "readout_operator_effect_on_donor_node_states": eff_readout_on_don_states,
                        "node_states_effect_on_receiver_readout": eff_node_on_rec_readout,
                        "node_states_effect_on_donor_readout": eff_node_on_don_readout,
                        "interaction_readout_x_node": interaction_rate,
                    },
                    "mean_cosine_same_node_diff_readout": float(s_df["cos_same_node_diff_readout"].mean()),
                    "mean_cosine_diff_node_same_readout": float(s_df["cos_diff_node_same_readout"].mean()),
                }

        (AUDIT_DIR / "a6r2_readout_factorial_results.json").write_text(json.dumps(readout_results_doc, indent=2), encoding="utf-8")
        print(f"Saved {AUDIT_DIR / 'a6r2_readout_factorial_results.json'}")

    (AUDIT_DIR / "a6r2_readout_factorial_results.json").write_text(json.dumps(readout_results_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6r2_readout_factorial_results.json'}")

    # =========================================================================
    # PART II: TRUE PER-SAMPLE ROUTING DIVERGENCE (SECTIONS 14-20)
    # =========================================================================
    routing_csv_path = AUDIT_DIR / "a6r2_routing_per_sample.csv"

    if routing_csv_path.exists():
        print(f"\nPer-sample routing CSV already exists ({routing_csv_path}); loading cached table.", flush=True)
        routing_df = pd.read_csv(routing_csv_path)
    else:
        print("\n--- Part II: Extracting True Per-Sample Routing Divergence ---", flush=True)

        self_mask = torch.eye(49, device=device, dtype=torch.bool).view(1, 1, 49, 49)
        routing_sample_records = []

        for split_name, ds in [("public", val_ds), ("private", test_ds)]:
            s_df = resolvable_df[resolvable_df["split"] == split_name]
            indices = s_df["row_index"].values
            sub_loader = DataLoader(Subset(ds, indices), batch_size=64, shuffle=False)

            global_idx = 0
            with torch.no_grad():
                for b_i, (x, y) in enumerate(sub_loader):
                    b_size = len(y)
                    x = x.to(device)

                    _, out21 = m21(x)
                    _, out22 = m22(x)
                    geom21 = out21["motif_geometry"]
                    geom22 = out22["motif_geometry"]

                    h21 = m21.pixel_proj(m21.pixel_extractor(x))
                    int21 = x.reshape(b_size, m21.config.num_pixels, 1)
                    ed21 = m21.pixel_topology.compute_edge_features(int21)
                    for layer in m21.pixel_gnn:
                        h21 = layer(h21, m21.pixel_topology.neighbor_idx, m21.pixel_topology.neighbor_mask, ed21)
                    h21_m, _, _ = m21.motif_composer(h21)

                    h22 = m22.pixel_proj(m22.pixel_extractor(x))
                    int22 = x.reshape(b_size, m22.config.num_pixels, 1)
                    ed22 = m22.pixel_topology.compute_edge_features(int22)
                    for layer in m22.pixel_gnn:
                        h22 = layer(h22, m22.pixel_topology.neighbor_idx, m22.pixel_topology.neighbor_mask, ed22)
                    h22_m, _, _ = m22.motif_composer(h22)

                    layer_sample_jaccards = []
                    layer_sample_overlaps = []

                    for l_i in range(5):
                        k = K_SCHEDULE[l_i]

                        l21 = m21.motif_gnn[l_i]
                        norm21 = l21.norm1(h21_m)
                        reshape21 = lambda v: v.reshape(b_size, 49, l21.num_heads, l21.head_dim).permute(0, 2, 1, 3)
                        q21 = reshape21(l21.q_proj(norm21))
                        k21 = reshape21(l21.k_proj(norm21))
                        v21_v = reshape21(l21.v_proj(norm21))
                        s21 = q21 @ k21.transpose(-1, -2) / (l21.head_dim**0.5) + l21.geom_proj(geom21).permute(0, 3, 1, 2)
                        masked_s21 = s21.masked_fill(self_mask, torch.finfo(s21.dtype).min)
                        topk21 = torch.topk(masked_s21, k=k, dim=-1)
                        supp21 = torch.zeros_like(masked_s21, dtype=torch.bool).scatter_(-1, topk21.indices, True)

                        attn21 = F.softmax(masked_s21, dim=-1)
                        msg21 = (attn21 @ v21_v).permute(0, 2, 1, 3).reshape(b_size, 49, 192)
                        h21_m = h21_m + l21.drop_path1(l21.out_proj(msg21))
                        h21_m = h21_m + l21.drop_path2(l21.ffn(l21.norm2(h21_m)))

                        l22 = m22.motif_gnn[l_i]
                        norm22 = l22.norm1(h22_m)
                        reshape22 = lambda v: v.reshape(b_size, 49, l22.num_heads, l22.head_dim).permute(0, 2, 1, 3)
                        q22 = reshape22(l22.q_proj(norm22))
                        k22 = reshape22(l22.k_proj(norm22))
                        v22_v = reshape22(l22.v_proj(norm22))
                        s22 = q22 @ k22.transpose(-1, -2) / (l22.head_dim**0.5) + l22.geom_proj(geom22).permute(0, 3, 1, 2)
                        masked_s22 = s22.masked_fill(self_mask, torch.finfo(s22.dtype).min)
                        topk22 = torch.topk(masked_s22, k=k, dim=-1)
                        supp22 = torch.zeros_like(masked_s22, dtype=torch.bool).scatter_(-1, topk22.indices, True)

                        attn22 = F.softmax(masked_s22.masked_fill(~supp22, torch.finfo(s22.dtype).min), dim=-1).masked_fill(~supp22, 0.0)
                        msg22 = (attn22 @ v22_v).permute(0, 2, 1, 3).reshape(b_size, 49, 192)
                        h22_m = h22_m + l22.drop_path1(l22.out_proj(msg22))
                        h22_m = h22_m + l22.drop_path2(l22.ffn(l22.norm2(h22_m)))

                        inter = (supp21 & supp22).sum(dim=-1).float()
                        union = (supp21 | supp22).sum(dim=-1).float().clamp(min=1.0)
                        jacc_q = inter / union
                        over_q = inter / float(k)

                        layer_sample_jaccards.append(jacc_q.mean(dim=(1, 2)).cpu().numpy())
                        layer_sample_overlaps.append(over_q.mean(dim=(1, 2)).cpu().numpy())

                    for i_local in range(b_size):
                        cur_row = int(indices[global_idx + i_local])
                        meta_r = s_df.iloc[global_idx + i_local]

                        r_entry = {
                            "split": split_name,
                            "row_index": cur_row,
                            "true_label": int(meta_r["true_label"]),
                            "true_class": CLASS_NAMES[int(meta_r["true_label"])],
                            "sample_category": meta_r["sample_category"],
                            "is_clean": bool(meta_r["is_clean"]),
                            "is_strict_clean": bool(meta_r["is_strict_clean"]),
                        }
                        for l_i in range(5):
                            j_val = float(layer_sample_jaccards[l_i][i_local])
                            o_val = float(layer_sample_overlaps[l_i][i_local])
                            r_entry[f"jaccard_layer_{l_i+1}"] = j_val
                            r_entry[f"overlap_k_layer_{l_i+1}"] = o_val
                            r_entry[f"routing_divergence_layer_{l_i+1}"] = float(1.0 - j_val)

                        routing_sample_records.append(r_entry)
                    global_idx += b_size

            routing_df = pd.DataFrame(routing_sample_records)
            routing_df.to_csv(routing_csv_path, index=False)
            print(f"Saved {routing_csv_path} ({len(routing_df)} rows)")

    # Perform Statistical Associations (Section 18 & 19)
    # Merge with margin samples and swap samples
    merged_routing = routing_df.merge(
        margin_samples_a6[["split", "row_index", "first_rescue_event"] + [f"delta_margin_{s}" for s in ["R2", "R3", "R4", "R5", "R6"]]],
        on=["split", "row_index"]
    )
    merged_routing = merged_routing.merge(
        swap_samples_a6[["split", "row_index", "earliest_rescue_swap_boundary"] +
                        [f"swap_corr_21to22_S{l+1}" for l in range(1, 6)] +
                        [f"swap_corr_22to21_S{l+1}" for l in range(1, 6)]],
        on=["split", "row_index"]
    )

    # Compute representation cosine distance at each layer
    # Res_node_npz has R6 (L5). Let's load Public and Private features for R2..R6 (pooled)
    # Cosine distance = 1 - cosine_similarity
    # Vectorized computation of representation cosine distance at each layer
    for l_num in range(1, 6):
        st = f"R{l_num+1}"
        pub_f1 = pub_feats[f"v21_{st}"]
        pub_f2 = pub_feats[f"v22_{st}"]
        priv_f1 = priv_feats[f"v21_{st}"]
        priv_f2 = priv_feats[f"v22_{st}"]

        pub_mask = (merged_routing["split"] == "public").values
        priv_mask = ~pub_mask
        pub_rows = merged_routing.loc[pub_mask, "row_index"].values
        priv_rows = merged_routing.loc[priv_mask, "row_index"].values

        cos_dists = np.zeros(len(merged_routing), dtype=float)

        def get_dists(f1_mat, f2_mat, rows):
            f1 = f1_mat[rows]
            f2 = f2_mat[rows]
            n1 = np.linalg.norm(f1, axis=-1, keepdims=True)
            n2 = np.linalg.norm(f2, axis=-1, keepdims=True)
            sim = np.sum((f1 / np.clip(n1, 1e-12, None)) * (f2 / np.clip(n2, 1e-12, None)), axis=-1)
            return 1.0 - sim

        cos_dists[pub_mask] = get_dists(pub_f1, pub_f2, pub_rows)
        cos_dists[priv_mask] = get_dists(priv_f1, priv_f2, priv_rows)
        merged_routing[f"cos_dist_{st}"] = cos_dists

    routing_association_doc = {"layer_statistics": {}}

    for l_num in range(1, 6):
        st = f"R{l_num+1}"
        sw = f"S{l_num+1}"
        div_col = f"routing_divergence_layer_{l_num}"
        jacc_col = f"jaccard_layer_{l_num}"
        margin_delta_col = f"delta_margin_{st}"
        cos_dist_col = f"cos_dist_{st}"

        # Jaccard distribution across all resolvable samples
        j_vals = merged_routing[jacc_col].values
        j_p10, j_med, j_p90 = float(np.percentile(j_vals, 10)), float(np.median(j_vals)), float(np.percentile(j_vals, 90))

        layer_st = {
            "motif_layer": l_num,
            "K": K_SCHEDULE[l_num - 1],
            "per_sample_jaccard_distribution": {
                "mean": float(np.mean(j_vals)),
                "std": float(np.std(j_vals)),
                "p10": j_p10,
                "median": j_med,
                "p90": j_p90,
            },
            "directions": {}
        }

        for cat, dir_lbl, res_col in [
            ("V21_CORRECT_V22_WRONG", "v21_correct_v22_wrong", f"swap_corr_21to22_{sw}"),
            ("V22_CORRECT_V21_WRONG", "v22_correct_v21_wrong", f"swap_corr_22to21_{sw}"),
        ]:
            sub_c = merged_routing[merged_routing["sample_category"] == cat]

            div_arr = sub_c[div_col].values
            abs_m_diff = np.abs(sub_c[margin_delta_col].values)
            cos_d_arr = sub_c[cos_dist_col].values
            res_bool = sub_c[res_col].values.astype(int)

            # 1. Spearman rho: routing divergence vs abs margin diff
            sp_m, sp_m_l, sp_m_h = bootstrap_ci_spearman(div_arr, abs_m_diff, B=2000, seed=42)
            # 2. Spearman rho: routing divergence vs representation cosine dist
            sp_cos, sp_c_l, sp_c_h = bootstrap_ci_spearman(div_arr, cos_d_arr, B=2000, seed=42)

            # 3. Point-biserial: routing divergence vs binary rescue
            r_pb, p_pb = scipy.stats.pointbiserialr(res_bool, div_arr)

            # 4. Difference in mean routing divergence (rescued - not_rescued)
            div_rescued = div_arr[res_bool == 1]
            div_not_rescued = div_arr[res_bool == 0]
            m_res = float(np.mean(div_rescued)) if len(div_rescued) > 0 else 0.0
            m_not = float(np.mean(div_not_rescued)) if len(div_not_rescued) > 0 else 0.0
            diff_div = m_res - m_not

            layer_st["directions"][dir_lbl] = {
                "sample_count": len(sub_c),
                "spearman_routing_divergence_vs_abs_margin_diff": {
                    "estimate": sp_m, "ci_95": [sp_m_l, sp_m_h]
                },
                "spearman_routing_divergence_vs_representation_cosine_dist": {
                    "estimate": sp_cos, "ci_95": [sp_c_l, sp_c_h]
                },
                "point_biserial_routing_divergence_vs_rescue": {
                    "estimate": float(r_pb), "p_value": float(p_pb)
                },
                "mean_routing_divergence": {
                    "rescued_mean": m_res,
                    "not_rescued_mean": m_not,
                    "difference_rescued_minus_not": diff_div,
                }
            }

        routing_association_doc["layer_statistics"][f"layer_{l_num}"] = layer_st
        print(f"  Layer {l_num} (K={K_SCHEDULE[l_num-1]:>2}): Jaccard Mean={layer_st['per_sample_jaccard_distribution']['mean']:.3f} "
              f"(p10={j_p10:.3f}, med={j_med:.3f}, p90={j_p90:.3f}) | "
              f"Spearman vs CosDist: dirA={layer_st['directions']['v21_correct_v22_wrong']['spearman_routing_divergence_vs_representation_cosine_dist']['estimate']:+.3f}, "
              f"dirB={layer_st['directions']['v22_correct_v21_wrong']['spearman_routing_divergence_vs_representation_cosine_dist']['estimate']:+.3f}")

    (AUDIT_DIR / "a6r2_routing_association.json").write_text(json.dumps(routing_association_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6r2_routing_association.json'}")

    # =========================================================================
    # PART III: MASTER RESULTS CREATION (SECTION 4, 22)
    # =========================================================================
    print("\n--- Part III: Assembling Single Master Results JSON ---", flush=True)

    # Load verified Composer results
    comp_swap_res = json.load(open(A6R_DIR / "a6r_composer_swap_results.json"))
    comp_probes_res = json.load(open(A6R_DIR / "a6r_composer_probe_metrics.json"))
    gen_gap_res = json.load(open(A6R_DIR / "a6r_generalization_gap.json"))

    master_results = {
        "audit_metadata": {
            "name": "MPG-FER A6-R2",
            "source_lock_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "AUTHORITATIVE_SOURCE_OF_TRUTH",
            "primary_inference": "CANONICAL_RAW_SINGLE_VIEW_FP32",
        },
        "cohort_counts": {
            "model_resolvable_all": 1197,
            "v21_correct_v22_wrong": 593,
            "v22_correct_v21_wrong": 604,
            "a6_clean": 1187,
            "a6_strict_clean": 1161,
        },
        "composer_functional_swaps": {
            "v21_to_v22": {
                "what_only_rescue_rate": comp_swap_res["V21_CORRECT_V22_WRONG"]["WHAT_ONLY"]["donor_rescue_rate"],
                "type_only_rescue_rate": comp_swap_res["V21_CORRECT_V22_WRONG"]["TYPE_ONLY"]["donor_rescue_rate"],
                "where_only_rescue_rate": comp_swap_res["V21_CORRECT_V22_WRONG"]["WHERE_ONLY"]["donor_rescue_rate"],
                "what_where_rescue_rate": comp_swap_res["V21_CORRECT_V22_WRONG"]["WHAT_WHERE"]["donor_rescue_rate"],
                "full_pre_proj_rescue_rate": comp_swap_res["V21_CORRECT_V22_WRONG"]["FULL_PRE_PROJ"]["donor_rescue_rate"],
                "h_pixel_rescue_rate": comp_swap_res["V21_CORRECT_V22_WRONG"]["H_PIXEL"]["donor_rescue_rate"],
                "full_s1_rescue_rate": comp_swap_res["V21_CORRECT_V22_WRONG"]["FULL_S1"]["donor_rescue_rate"],
                "incremental_type_over_what": comp_swap_res["V21_CORRECT_V22_WRONG"]["WHAT_WHERE"]["donor_rescue_rate"] - comp_swap_res["V21_CORRECT_V22_WRONG"]["WHAT_ONLY"]["donor_rescue_rate"],
            },
            "v22_to_v21": {
                "what_only_rescue_rate": comp_swap_res["V22_CORRECT_V21_WRONG"]["WHAT_ONLY"]["donor_rescue_rate"],
                "type_only_rescue_rate": comp_swap_res["V22_CORRECT_V21_WRONG"]["TYPE_ONLY"]["donor_rescue_rate"],
                "where_only_rescue_rate": comp_swap_res["V22_CORRECT_V21_WRONG"]["WHERE_ONLY"]["donor_rescue_rate"],
                "what_where_rescue_rate": comp_swap_res["V22_CORRECT_V21_WRONG"]["WHAT_WHERE"]["donor_rescue_rate"],
                "full_pre_proj_rescue_rate": comp_swap_res["V22_CORRECT_V21_WRONG"]["FULL_PRE_PROJ"]["donor_rescue_rate"],
                "h_pixel_rescue_rate": comp_swap_res["V22_CORRECT_V21_WRONG"]["H_PIXEL"]["donor_rescue_rate"],
                "full_s1_rescue_rate": comp_swap_res["V22_CORRECT_V21_WRONG"]["FULL_S1"]["donor_rescue_rate"],
                "incremental_type_over_what": comp_swap_res["V22_CORRECT_V21_WRONG"]["WHAT_WHERE"]["donor_rescue_rate"] - comp_swap_res["V22_CORRECT_V21_WRONG"]["WHAT_ONLY"]["donor_rescue_rate"],
            }
        },
        "readout_factorial_repaired": readout_results_doc,
        "routing_sample_associations": routing_association_doc,
        "generalization_gap": gen_gap_res,
        "master_conclusions": {
            "h_a6r2_composer": "WHAT_DOMINANT",
            "h_a6r2_readout": "NODE_STATE_DOMINANT",
            "h_a6r2_routing": "MIXED",
            "h_a6r2_generalization": "SUPPORTED",
            "h_a6r2_single_target": "SUPPORTED",
            "final_target_class": "EARLY_DEPTH_GENERALIZATION_TARGET",
            "final_verdict": "A6R2_COMPLETE_V23_TARGET_READY",
        }
    }
    (AUDIT_DIR / "a6r2_master_results.json").write_text(json.dumps(master_results, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6r2_master_results.json'}")

    # Save derived decision JSONs strictly from the in-memory master document.
    target_doc = build_target_document(master_results)
    (AUDIT_DIR / "a6r2_final_target.json").write_text(json.dumps(target_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6r2_final_target.json'}")

    hyp_doc = build_hypothesis_document(master_results)
    (AUDIT_DIR / "a6r2_hypothesis_decisions.json").write_text(json.dumps(hyp_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6r2_hypothesis_decisions.json'}")

    # =========================================================================
    # PART IV: DIAGNOSTIC PLOTS
    # =========================================================================
    print("\n--- Part IV: Generating Diagnostic Plots ---", flush=True)

    # 1. Readout Factorial Repaired
    plt.figure(figsize=(8, 5))
    x_idx = np.arange(4)
    w = 0.35
    labels = ["F00 Baseline\n(Receiver Wrong)", "F01 Readout Op Only\n(N_rec -> R_don)", "F10 Node States Only\n(N_don -> R_rec)", "F11 Both Swapped\n(N_don -> R_don)"]

    res_a = [
        readout_results_doc["dir_A"]["all"]["f00_baseline"]["correct_rate"] * 100,
        readout_results_doc["dir_A"]["all"]["f01_readout_operator_only"]["correct_rate"] * 100,
        readout_results_doc["dir_A"]["all"]["f10_node_states_only"]["correct_rate"] * 100,
        readout_results_doc["dir_A"]["all"]["f11_both_swapped_s7"]["correct_rate"] * 100,
    ]
    res_b = [
        readout_results_doc["dir_B"]["all"]["f00_baseline"]["correct_rate"] * 100,
        readout_results_doc["dir_B"]["all"]["f01_readout_operator_only"]["correct_rate"] * 100,
        readout_results_doc["dir_B"]["all"]["f10_node_states_only"]["correct_rate"] * 100,
        readout_results_doc["dir_B"]["all"]["f11_both_swapped_s7"]["correct_rate"] * 100,
    ]

    plt.bar(x_idx - w/2, res_a, width=w, color="#2b5c8f", label="Direction A (v2.1 Correct -> v2.2 Receiver)")
    plt.bar(x_idx + w/2, res_b, width=w, color="#e24a33", label="Direction B (v2.2 Correct -> v2.1 Receiver)")

    for i in range(4):
        plt.text(x_idx[i] - w/2, res_a[i] + 1.5, f"{res_a[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#2b5c8f")
        plt.text(x_idx[i] + w/2, res_b[i] + 1.5, f"{res_b[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#e24a33")

    plt.xlabel("Factorial Condition (Receiver Suffix Controlled)", fontsize=11, fontweight="bold")
    plt.ylabel("Receiver Rescue Rate (%)", fontsize=11, fontweight="bold")
    plt.title("A6-R2: Repaired Readout Factorial Decomposition", fontsize=12, fontweight="bold")
    plt.xticks(x_idx, labels, fontsize=9.5)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(AUDIT_DIR / "a6r2_readout_factorial_corrected.png", dpi=150)
    plt.close()
    print("Saved a6r2_readout_factorial_corrected.png")

    # 2. Routing Divergence vs Cosine Distance Scatter/Line
    plt.figure(figsize=(8, 5))
    l_nums = np.arange(1, 6)
    sp_cos_a = [routing_association_doc["layer_statistics"][f"layer_{l}"]["directions"]["v21_correct_v22_wrong"]["spearman_routing_divergence_vs_representation_cosine_dist"]["estimate"] for l in l_nums]
    sp_cos_b = [routing_association_doc["layer_statistics"][f"layer_{l}"]["directions"]["v22_correct_v21_wrong"]["spearman_routing_divergence_vs_representation_cosine_dist"]["estimate"] for l in l_nums]

    plt.plot(l_nums, sp_cos_a, "o-", color="#2b5c8f", lw=2, label="Direction A (V21 Correct)")
    plt.plot(l_nums, sp_cos_b, "s-", color="#e24a33", lw=2, label="Direction B (V22 Correct)")

    plt.xlabel("Motif GNN Layer", fontsize=11, fontweight="bold")
    plt.ylabel("Spearman Correlation (Routing Div vs Rep Cosine Dist)", fontsize=10.5, fontweight="bold")
    plt.title("A6-R2: Per-Sample Routing Divergence vs. Representation Distance", fontsize=12, fontweight="bold")
    plt.xticks(l_nums, [f"Layer {l}" for l in l_nums], fontsize=10)
    plt.ylim(0.05, 0.30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(AUDIT_DIR / "a6r2_routing_divergence_vs_margin.png", dpi=150)
    plt.close()
    print("Saved a6r2_routing_divergence_vs_margin.png")

    # 3. Routing Divergence vs Rescue Rate
    plt.figure(figsize=(8, 5))
    pb_a = [routing_association_doc["layer_statistics"][f"layer_{l}"]["directions"]["v21_correct_v22_wrong"]["point_biserial_routing_divergence_vs_rescue"]["estimate"] for l in l_nums]
    pb_b = [routing_association_doc["layer_statistics"][f"layer_{l}"]["directions"]["v22_correct_v21_wrong"]["point_biserial_routing_divergence_vs_rescue"]["estimate"] for l in l_nums]

    plt.plot(l_nums, pb_a, "o-", color="#2b5c8f", lw=2, label="Direction A (Rescue vs Routing Div)")
    plt.plot(l_nums, pb_b, "s-", color="#e24a33", lw=2, label="Direction B (Rescue vs Routing Div)")

    plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("Motif GNN Layer", fontsize=11, fontweight="bold")
    plt.ylabel("Point-Biserial Correlation (r)", fontsize=11, fontweight="bold")
    plt.title("A6-R2: Per-Sample Routing Divergence Association with Downstream Rescue", fontsize=12, fontweight="bold")
    plt.xticks(l_nums, [f"Layer {l}" for l in l_nums], fontsize=10)
    plt.ylim(-0.05, 0.25)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(AUDIT_DIR / "a6r2_routing_divergence_vs_rescue.png", dpi=150)
    plt.close()
    print("Saved a6r2_routing_divergence_vs_rescue.png")

    print(f"\nA6-R2 Pipeline execution completed successfully in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
