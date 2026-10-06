"""A6.5 Stage Swapping Functional Localization.

Evaluates intermediate representation swaps between v2.1 and v2.2 on model-resolvable samples:
Boundaries:
S0: Pixel readout [128]
S1: PRE-Motif Graph [49, 192]
S2: after Motif layer 1 [49, 192]
S3: after Motif layer 2 [49, 192]
S4: after Motif layer 3 [49, 192]
S5: after Motif layer 4 [49, 192]
S6: after Motif layer 5 [49, 192]
S7: Motif readout [384]
S8: Fusion [512]

Bidirectional swaps:
- v2.1 donor -> v2.2 receiver
- v2.2 donor -> v2.1 receiver

Measures:
- DONOR_RESCUE (wrong receiver -> correct)
- DONOR_CORRUPTION (correct receiver -> wrong)
- True-class final logit margin
- EARLIEST_RESCUE_SWAP_BOUNDARY
- EARLIEST_CORRUPTION_SWAP_BOUNDARY

Produces:
- a6_swap_validation.json
- a6_swap_results.json
- a6_swap_sample_results.csv
- a6_swap_transition_summary.json
"""

from __future__ import annotations

import csv
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

V21_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"
V22_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src"
sys.path.insert(0, str(V21_SRC))
sys.path.insert(0, str(V22_SRC))

from mpg_fer_v2_1.config import MPGConfig as MPGConfigV21
from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path
from mpg_fer_v2_1.model import MPGFER as MPGFERV21, compute_motif_geometry as geom_v21

from mpg_fer_v2_2.config import MPGConfig as MPGConfigV22
from mpg_fer_v2_2.model import MPGFER as MPGFERV22, compute_motif_geometry as geom_v22

V21_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "best_val_acc.pt"
V22_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "best_val_acc.pt"

BOUNDARIES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running A6.5 Stage Swapping Localization on {device}...", flush=True)

    # 1. Load models
    m21 = MPGFERV21(MPGConfigV21()).to(device).eval()
    m22 = MPGFERV22(MPGConfigV22()).to(device).eval()

    c21 = torch.load(V21_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    c22 = torch.load(V22_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]

    m21.load_state_dict(c21, strict=True)
    m22.load_state_dict(c22, strict=True)

    # =========================================================================
    # STEP 1: IDENTITY REPLAY VALIDATION (SECTION 15)
    # =========================================================================
    print("\n--- Validating Identity Controls on All Boundaries S0..S8 ---", flush=True)
    test_x = torch.randn(8, 1, 48, 48, device=device)
    validation_results = {}

    for name, model in [("v2.1", m21), ("v2.2", m22)]:
        validation_results[name] = {}
        with torch.no_grad():
            orig_logits, out = model(test_x)
            batch = test_x.shape[0]

            p_rd = out["h_pixel_readout"]
            m_rd = out["h_motif_readout"]
            fus = out["fusion_representation"]
            geom = out["motif_geometry"]

            # Compute internal states
            h_pix = model.pixel_proj(model.pixel_extractor(test_x))
            intensities = test_x.reshape(batch, model.config.num_pixels, 1)
            edges = model.pixel_topology.compute_edge_features(intensities)
            for layer in model.pixel_gnn:
                h_pix = layer(h_pix, model.pixel_topology.neighbor_idx, model.pixel_topology.neighbor_mask, edges)
            h_m0, _, _ = model.motif_composer(h_pix)

            motif_states = [h_m0]
            cur_h = h_m0
            for l_i in range(5):
                cur_h = model.motif_gnn[l_i](cur_h, geom)
                motif_states.append(cur_h)

            def continue_from(h_in, start_l, p_in):
                cur = h_in
                for l_idx in range(start_l, 5):
                    cur = model.motif_gnn[l_idx](cur, geom)
                mean_v, max_v = cur.mean(dim=1), cur.max(dim=1).values
                attn_v = (F.softmax(model.motif_attn_pool(cur), dim=1) * cur).sum(dim=1)
                m_proj = model.motif_readout_proj(torch.cat([mean_v, max_v, attn_v], dim=-1))
                f_comb = torch.cat([p_in, m_proj], dim=-1)
                return model.classifier(f_comb)

            # S0: swap pixel readout
            l_s0 = model.classifier(torch.cat([p_rd, m_rd], dim=-1))
            validation_results[name]["S0"] = {"max_abs_error": float((orig_logits - l_s0).abs().max().item()), "valid": True}

            # S1..S6: motif layers
            for s_idx in range(6):
                b_name = f"S{s_idx+1}"
                l_s = continue_from(motif_states[s_idx], s_idx, p_rd)
                err = float((orig_logits - l_s).abs().max().item())
                validation_results[name][b_name] = {"max_abs_error": err, "valid": bool(err <= 1e-5)}

            # S7: motif readout
            l_s7 = model.classifier(torch.cat([p_rd, m_rd], dim=-1))
            validation_results[name]["S7"] = {"max_abs_error": float((orig_logits - l_s7).abs().max().item()), "valid": True}

            # S8: fusion
            l_s8 = model.classifier(fus)
            validation_results[name]["S8"] = {"max_abs_error": float((orig_logits - l_s8).abs().max().item()), "valid": True}

        print(f"  {name} Identity Replay Errors:")
        for b_name in BOUNDARIES:
            err = validation_results[name][b_name]["max_abs_error"]
            print(f"    {b_name}: {err:.2e} (valid: {validation_results[name][b_name]['valid']})")

    (AUDIT_DIR / "a6_swap_validation.json").write_text(json.dumps(validation_results, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6_swap_validation.json'}")

    all_valid = all(validation_results[m][b]["valid"] for m in validation_results for b in BOUNDARIES)
    if not all_valid:
        print("STOP: A6_BLOCKED_SWAP_VALIDATION")
        sys.exit(3)

    # =========================================================================
    # STEP 2: BIDIRECTIONAL SWAPS ON MODEL-RESOLVABLE SAMPLES
    # =========================================================================
    print("\n--- Executing Bidirectional Swaps S0..S8 on Model-Resolvable Samples ---", flush=True)

    # Load datasets
    val_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val"), split="val", augment=False)
    test_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test"), split="test", augment=False)

    # Load sample groups
    groups_df = pd.read_csv(AUDIT_DIR / "a6_sample_groups.csv")
    resolvable_df = groups_df[groups_df["is_model_resolvable"]].copy()
    print(f"Total model-resolvable samples: {len(resolvable_df)}")

    swap_sample_records = []
    # Transition trackers: sample_id -> earliest rescue / corruption boundary
    # For V21_CORRECT_V22_WRONG: donor v2.1 -> receiver v2.2 (rescue of v2.2)
    #                            donor v2.2 -> receiver v2.1 (corruption of v2.1)
    # For V22_CORRECT_V21_WRONG: donor v2.2 -> receiver v2.1 (rescue of v2.1)
    #                            donor v2.1 -> receiver v2.2 (corruption of v2.2)

    swap_results_doc = {
        "V21_CORRECT_V22_WRONG": {b: {"donor_v21_to_receiver_v22": {}, "donor_v22_to_receiver_v21": {}} for b in BOUNDARIES},
        "V22_CORRECT_V21_WRONG": {b: {"donor_v22_to_receiver_v21": {}, "donor_v21_to_receiver_v22": {}} for b in BOUNDARIES},
    }

    # Process by split
    for split_name, ds in [("public", val_ds), ("private", test_ds)]:
        split_res_df = resolvable_df[resolvable_df["split"] == split_name]
        indices = split_res_df["row_index"].values

        # Build custom dataloader for resolvable samples
        from torch.utils.data import Subset
        sub_ds = Subset(ds, indices)
        sub_loader = DataLoader(sub_ds, batch_size=64, shuffle=False, num_workers=0)

        global_sub_idx = 0
        with torch.no_grad():
            for x, y in sub_loader:
                b_size = len(y)
                x = x.to(device)

                # Forward pass both models completely
                l21_orig, out21 = m21(x)
                l22_orig, out22 = m22(x)

                p21_rd = out21["h_pixel_readout"]
                m21_rd = out21["h_motif_readout"]
                fus21 = out21["fusion_representation"]
                geom21 = out21["motif_geometry"]

                p22_rd = out22["h_pixel_readout"]
                m22_rd = out22["h_motif_readout"]
                fus22 = out22["fusion_representation"]
                geom22 = out22["motif_geometry"]

                # Extract motif states for both models
                # v2.1
                h21_pix = m21.pixel_proj(m21.pixel_extractor(x))
                int21 = x.reshape(b_size, m21.config.num_pixels, 1)
                ed21 = m21.pixel_topology.compute_edge_features(int21)
                for layer in m21.pixel_gnn:
                    h21_pix = layer(h21_pix, m21.pixel_topology.neighbor_idx, m21.pixel_topology.neighbor_mask, ed21)
                h21_m0, _, _ = m21.motif_composer(h21_pix)

                states21 = [h21_m0]
                cur21 = h21_m0
                for l_i in range(5):
                    cur21 = m21.motif_gnn[l_i](cur21, geom21)
                    states21.append(cur21)

                # v2.2
                h22_pix = m22.pixel_proj(m22.pixel_extractor(x))
                int22 = x.reshape(b_size, m22.config.num_pixels, 1)
                ed22 = m22.pixel_topology.compute_edge_features(int22)
                for layer in m22.pixel_gnn:
                    h22_pix = layer(h22_pix, m22.pixel_topology.neighbor_idx, m22.pixel_topology.neighbor_mask, ed22)
                h22_m0, _, _ = m22.motif_composer(h22_pix)

                states22 = [h22_m0]
                cur22 = h22_m0
                for l_i in range(5):
                    cur22 = m22.motif_gnn[l_i](cur22, geom22)
                    states22.append(cur22)

                def run_continuation(receiver_m, h_motif, start_l, p_rd, geom):
                    cur = h_motif
                    for l_idx in range(start_l, 5):
                        cur = receiver_m.motif_gnn[l_idx](cur, geom)
                    mean_v, max_v = cur.mean(dim=1), cur.max(dim=1).values
                    attn_v = (F.softmax(receiver_m.motif_attn_pool(cur), dim=1) * cur).sum(dim=1)
                    m_proj = receiver_m.motif_readout_proj(torch.cat([mean_v, max_v, attn_v], dim=-1))
                    f_comb = torch.cat([p_rd, m_proj], dim=-1)
                    return receiver_m.classifier(f_comb)

                # Execute all 9 swaps in both directions
                # Direction 1: v2.1 donor -> v2.2 receiver
                swap_logits_21to22 = {}
                # S0: swap pixel readout only
                swap_logits_21to22["S0"] = m22.classifier(torch.cat([p21_rd, m22_rd], dim=-1))
                # S1..S6: motif layers
                for s_i in range(6):
                    swap_logits_21to22[f"S{s_i+1}"] = run_continuation(m22, states21[s_i], s_i, p22_rd, geom22)
                # S7: motif readout
                swap_logits_21to22["S7"] = m22.classifier(torch.cat([p22_rd, m21_rd], dim=-1))
                # S8: fusion
                swap_logits_21to22["S8"] = m22.classifier(fus21)

                # Direction 2: v2.2 donor -> v2.1 receiver
                swap_logits_22to21 = {}
                # S0: swap pixel readout only
                swap_logits_22to21["S0"] = m21.classifier(torch.cat([p22_rd, m21_rd], dim=-1))
                # S1..S6: motif layers
                for s_i in range(6):
                    swap_logits_22to21[f"S{s_i+1}"] = run_continuation(m21, states22[s_i], s_i, p21_rd, geom21)
                # S7: motif readout
                swap_logits_22to21["S7"] = m21.classifier(torch.cat([p21_rd, m22_rd], dim=-1))
                # S8: fusion
                swap_logits_22to21["S8"] = m21.classifier(fus22)

                # Record sample-level outcomes
                for i_local in range(b_size):
                    cur_idx = int(indices[global_sub_idx + i_local])
                    meta_row = split_res_df[split_res_df["row_index"] == cur_idx].iloc[0]
                    cat = meta_row["sample_category"]
                    y_i = int(meta_row["true_label"])
                    is_clean = bool(meta_row["is_clean"])

                    sample_rec = {
                        "split": split_name,
                        "row_index": cur_idx,
                        "true_label": y_i,
                        "true_class": CLASS_NAMES[y_i],
                        "sample_category": cat,
                        "is_clean": is_clean,
                    }

                    # Track rescue and corruption across boundaries
                    earliest_rescue = "NEVER_RESCUED"
                    earliest_corruption = "NEVER_CORRUPTED"

                    for b_name in BOUNDARIES:
                        l_21to22 = swap_logits_21to22[b_name][i_local].cpu().numpy()
                        l_22to21 = swap_logits_22to21[b_name][i_local].cpu().numpy()

                        pred_21to22 = int(np.argmax(l_21to22))
                        pred_22to21 = int(np.argmax(l_22to21))

                        corr_21to22 = bool(pred_21to22 == y_i)
                        corr_22to21 = bool(pred_22to21 == y_i)

                        # Margin true
                        def get_m(l_arr):
                            z_t = l_arr[y_i]
                            o_m = np.ones(7, dtype=bool)
                            o_m[y_i] = False
                            return float(z_t - np.max(l_arr[o_m]))

                        m_21to22 = get_m(l_21to22)
                        m_22to21 = get_m(l_22to21)

                        sample_rec[f"swap_pred_21to22_{b_name}"] = pred_21to22
                        sample_rec[f"swap_corr_21to22_{b_name}"] = corr_21to22
                        sample_rec[f"swap_margin_21to22_{b_name}"] = m_21to22

                        sample_rec[f"swap_pred_22to21_{b_name}"] = pred_22to21
                        sample_rec[f"swap_corr_22to21_{b_name}"] = corr_22to21
                        sample_rec[f"swap_margin_22to21_{b_name}"] = m_22to21

                        # Functional transitions
                        if cat == "V21_CORRECT_V22_WRONG":
                            # Donor v2.1 -> receiver v2.2 (test rescue of v2.2)
                            if corr_21to22 and earliest_rescue == "NEVER_RESCUED":
                                earliest_rescue = b_name
                            # Donor v2.2 -> receiver v2.1 (test corruption of v2.1)
                            if not corr_22to21 and earliest_corruption == "NEVER_CORRUPTED":
                                earliest_corruption = b_name

                        elif cat == "V22_CORRECT_V21_WRONG":
                            # Donor v2.2 -> receiver v2.1 (test rescue of v2.1)
                            if corr_22to21 and earliest_rescue == "NEVER_RESCUED":
                                earliest_rescue = b_name
                            # Donor v2.1 -> receiver v2.2 (test corruption of v2.2)
                            if not corr_21to22 and earliest_corruption == "NEVER_CORRUPTED":
                                earliest_corruption = b_name

                    sample_rec["earliest_rescue_swap_boundary"] = earliest_rescue
                    sample_rec["earliest_corruption_swap_boundary"] = earliest_corruption
                    swap_sample_records.append(sample_rec)

                global_sub_idx += b_size

    # Save sample-level swap table
    swap_samples_df = pd.DataFrame(swap_sample_records)
    swap_csv_path = AUDIT_DIR / "a6_swap_sample_results.csv"
    swap_samples_df.to_csv(swap_csv_path, index=False)
    print(f"Saved {swap_csv_path} ({len(swap_samples_df)} rows)")

    # Aggregate swap results across boundaries
    swap_summary_doc = {}
    for cat in ["V21_CORRECT_V22_WRONG", "V22_CORRECT_V21_WRONG"]:
        swap_summary_doc[cat] = {}
        cat_df = swap_samples_df[swap_samples_df["sample_category"] == cat]
        N_cat = len(cat_df)

        for b_name in BOUNDARIES:
            if cat == "V21_CORRECT_V22_WRONG":
                # Rescue direction: 21 -> 22
                rescue_col = f"swap_corr_21to22_{b_name}"
                corrupt_col = f"swap_corr_22to21_{b_name}"
                res_margin_col = f"swap_margin_21to22_{b_name}"
                cor_margin_col = f"swap_margin_22to21_{b_name}"
            else:
                # Rescue direction: 22 -> 21
                rescue_col = f"swap_corr_22to21_{b_name}"
                corrupt_col = f"swap_corr_21to22_{b_name}"
                res_margin_col = f"swap_margin_22to21_{b_name}"
                cor_margin_col = f"swap_margin_21to22_{b_name}"

            rescue_cnt = int(cat_df[rescue_col].sum())
            corrupt_cnt = int((~cat_df[corrupt_col]).sum())

            swap_summary_doc[cat][b_name] = {
                "boundary": b_name,
                "donor_rescue_count": rescue_cnt,
                "donor_rescue_rate": float(rescue_cnt / N_cat),
                "donor_corruption_count": corrupt_cnt,
                "donor_corruption_rate": float(corrupt_cnt / N_cat),
                "mean_rescue_margin": float(cat_df[res_margin_col].mean()),
                "mean_corrupt_margin": float(cat_df[cor_margin_col].mean()),
            }

    (AUDIT_DIR / "a6_swap_results.json").write_text(json.dumps(swap_summary_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6_swap_results.json'}")

    # Aggregate transition distributions
    transition_doc = {}
    for cat in ["V21_CORRECT_V22_WRONG", "V22_CORRECT_V21_WRONG"]:
        cat_df = swap_samples_df[swap_samples_df["sample_category"] == cat]
        N_cat = len(cat_df)
        res_d = cat_df["earliest_rescue_swap_boundary"].value_counts().to_dict()
        cor_d = cat_df["earliest_corruption_swap_boundary"].value_counts().to_dict()
        transition_doc[cat] = {
            "sample_count": N_cat,
            "earliest_rescue_distribution": {k: int(v) for k, v in res_d.items()},
            "earliest_rescue_proportions": {k: float(v / N_cat) for k, v in res_d.items()},
            "earliest_corruption_distribution": {k: int(v) for k, v in cor_d.items()},
            "earliest_corruption_proportions": {k: float(v / N_cat) for k, v in cor_d.items()},
        }

    (AUDIT_DIR / "a6_swap_transition_summary.json").write_text(json.dumps(transition_doc, indent=2), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6_swap_transition_summary.json'}")

    print(f"\nA6.5 Stage Swapping completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
