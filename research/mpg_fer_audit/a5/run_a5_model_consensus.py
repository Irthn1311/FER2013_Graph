"""A5.2 Multi-Model FP32 Evaluation, Four-Model Consensus Table, and Oracle Complementarity.

Evaluates frozen official checkpoints under canonical FP32 inference:
- v1 (Diffuse TYPE)
- v2 (Sharp TYPE)
- v2.1 (Dense Complete)
- v2.2 (Dynamic Sparse)
Produces:
- research/mpg_fer_audit/a5/a5_model_consensus.csv
- research/mpg_fer_audit/a5/a5_oracle_complementarity.json
- research/mpg_fer_audit/a5/a5_multi_model_predictions.npz
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a5"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    {
        "key": "v1",
        "label": "v1",
        "src": PROJECT_ROOT / "research" / "mpg_fer_v1" / "src",
        "pkg": "mpg_fer_v1",
        "ckpt": PROJECT_ROOT / "research" / "mpg_fer_v1" / "outputs" / "kaggle_t4_final" / "mpg_fer_v1_run" / "best_val_acc.pt",
        "expected_sha": "548325add48fc87a1f5011875c1f005123bbb82d8dfce5e7ba23285ad3f64a52",
    },
    {
        "key": "v2",
        "label": "v2",
        "src": PROJECT_ROOT / "research" / "mpg_fer_v2" / "src",
        "pkg": "mpg_fer_v2",
        "ckpt": PROJECT_ROOT / "research" / "mpg_fer_v2" / "outputs" / "kaggle_v2_final" / "final_run" / "best_val_acc.pt",
        "expected_sha": "f3cda72fc4d791e7017e2e0374f83ef22e9f22f03e8172b7389e53f8cbc6dc1c",
    },
    {
        "key": "v2_1",
        "label": "v2.1",
        "src": PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src",
        "pkg": "mpg_fer_v2_1",
        "ckpt": PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "best_val_acc.pt",
        "expected_sha": "4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75",
    },
    {
        "key": "v2_2",
        "label": "v2.2",
        "src": PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src",
        "pkg": "mpg_fer_v2_2",
        "ckpt": PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "best_val_acc.pt",
        "expected_sha": "a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4",
    },
]

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def load_model(cfg: dict, device: str) -> tuple[torch.nn.Module, dict]:
    src_str = str(cfg["src"].resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    mod_model = __import__(f"{cfg['pkg']}.model", fromlist=["MPGFER"])
    mod_config = __import__(f"{cfg['pkg']}.config", fromlist=["MPGConfig"])

    model = mod_model.MPGFER(mod_config.MPGConfig()).to(device).eval()
    ckpt = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model, ckpt


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running A5.2 Multi-Model FP32 Evaluation on {device}...", flush=True)

    # Load all models
    models = {}
    for cfg in MODEL_CONFIGS:
        t0 = time.time()
        m, ckpt = load_model(cfg, device)
        models[cfg["key"]] = m
        print(f"Loaded {cfg['label']} ({cfg['key']}) strictly in {time.time() - t0:.2f}s", flush=True)

    # Datasets
    from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path
    val_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val"), split="val", augment=False)
    test_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test"), split="test", augment=False)

    consensus_rows = []
    npz_save_dict = {}
    oracle_summary = {}

    for split_name, ds in [("public", val_ds), ("private", test_ds)]:
        split_t0 = time.time()
        print(f"\nEvaluating {split_name.upper()} split ({len(ds)} samples) with FP32...", flush=True)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

        # Collect model predictions: [N, 7]
        model_tta_logits = {cfg["key"]: [] for cfg in MODEL_CONFIGS}
        model_fusion = {cfg["key"]: [] for cfg in MODEL_CONFIGS}

        with torch.no_grad():
            for b_i, (x, y) in enumerate(loader):
                if (b_i + 1) % 20 == 0 or (b_i + 1) == len(loader):
                    print(f"  Batch {b_i + 1}/{len(loader)} in {time.time() - split_t0:.1f}s", flush=True)
                x = x.to(device)
                x_flip = TF.hflip(x)

                for cfg in MODEL_CONFIGS:
                    m = models[cfg["key"]]
                    # FP32 inference without autocast
                    raw_l, out_orig = m(x)
                    flip_l, _ = m(x_flip)
                    tta_l = 0.5 * (raw_l + flip_l)

                    if "fusion_representation" in out_orig:
                        fusion = out_orig["fusion_representation"]
                    else:
                        fusion = torch.cat([out_orig["h_pixel_readout"], out_orig["h_motif_readout"]], dim=-1)

                    model_tta_logits[cfg["key"]].append(tta_l.cpu().numpy().astype(np.float32))
                    model_fusion[cfg["key"]].append(fusion.cpu().numpy().astype(np.float32))

        # Concatenate
        for cfg in MODEL_CONFIGS:
            model_tta_logits[cfg["key"]] = np.concatenate(model_tta_logits[cfg["key"]], axis=0)
            model_fusion[cfg["key"]] = np.concatenate(model_fusion[cfg["key"]], axis=0)
            npz_save_dict[f"{split_name}_{cfg['key']}_tta_logits"] = model_tta_logits[cfg["key"]]
            npz_save_dict[f"{split_name}_{cfg['key']}_fusion"] = model_fusion[cfg["key"]]

        npz_save_dict[f"{split_name}_targets"] = ds.labels.astype(np.int64)

        # Compute per-model predictions, probabilities, metrics
        N = len(ds)
        targets = ds.labels

        model_probs = {}
        model_preds = {}
        model_confs = {}
        model_margins = {}
        model_entropies = {}
        model_correct = {}

        for cfg in MODEL_CONFIGS:
            k = cfg["key"]
            logits = model_tta_logits[k]
            # Softmax in float64 for stability
            e_x = np.exp(logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True))
            probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
            preds = np.argmax(probs, axis=-1)
            confs = np.max(probs, axis=-1)
            
            sorted_p = np.sort(probs, axis=-1)[:, ::-1]
            margins = sorted_p[:, 0] - sorted_p[:, 1]
            entropies = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
            correct = (preds == targets)

            model_probs[k] = probs
            model_preds[k] = preds
            model_confs[k] = confs
            model_margins[k] = margins
            model_entropies[k] = entropies
            model_correct[k] = correct

            acc = float(np.mean(correct))
            print(f"  {cfg['label']:<6} FP32 TTA Accuracy on {split_name.upper()}: {acc:.6f} ({int(np.sum(correct))}/{N})", flush=True)

        # Build 4-model consensus rows
        split_consensus_categories = {}

        for i in range(N):
            y_i = int(targets[i])

            p_v1 = int(model_preds["v1"][i])
            p_v2 = int(model_preds["v2"][i])
            p_v21 = int(model_preds["v2_1"][i])
            p_v22 = int(model_preds["v2_2"][i])

            preds_all = [p_v1, p_v2, p_v21, p_v22]
            confs_all = [
                float(model_confs["v1"][i]),
                float(model_confs["v2"][i]),
                float(model_confs["v2_1"][i]),
                float(model_confs["v2_2"][i]),
            ]
            correct_all = [
                bool(model_correct["v1"][i]),
                bool(model_correct["v2"][i]),
                bool(model_correct["v2_1"][i]),
                bool(model_correct["v2_2"][i]),
            ]

            num_correct = int(sum(correct_all))

            # Vote analysis
            vote_counts = np.bincount(preds_all, minlength=7)
            modal_pred = int(np.argmax(vote_counts))
            modal_count = int(vote_counts[modal_pred])

            vote_probs = vote_counts[vote_counts > 0] / 4.0
            vote_entropy = float(-np.sum(vote_probs * np.log(vote_probs)))

            mean_conf = float(np.mean(confs_all))

            # Categorization
            if num_correct == 4:
                cat = "ALL_CORRECT"
            elif num_correct == 3:
                cat = "MOSTLY_CORRECT"
            elif 1 <= num_correct <= 2:
                cat = "SPLIT_DECISION"
            else:
                # num_correct == 0 (all 4 models wrong)
                if modal_count == 4:
                    if mean_conf >= 0.80:
                        cat = "HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL"
                    else:
                        cat = "ALL_WRONG_SAME_LABEL"
                else:
                    cat = "ALL_WRONG_MIXED_LABEL"

            split_consensus_categories[cat] = split_consensus_categories.get(cat, 0) + 1

            consensus_rows.append({
                "split": split_name,
                "row_index": i,
                "true_label": y_i,
                "true_class": CLASS_NAMES[y_i],
                # v1
                "v1_pred": p_v1,
                "v1_conf": float(model_confs["v1"][i]),
                "v1_entropy": float(model_entropies["v1"][i]),
                "v1_margin": float(model_margins["v1"][i]),
                "v1_correct": bool(model_correct["v1"][i]),
                # v2
                "v2_pred": p_v2,
                "v2_conf": float(model_confs["v2"][i]),
                "v2_entropy": float(model_entropies["v2"][i]),
                "v2_margin": float(model_margins["v2"][i]),
                "v2_correct": bool(model_correct["v2"][i]),
                # v2.1
                "v21_pred": p_v21,
                "v21_conf": float(model_confs["v2_1"][i]),
                "v21_entropy": float(model_entropies["v2_1"][i]),
                "v21_margin": float(model_margins["v2_1"][i]),
                "v21_correct": bool(model_correct["v2_1"][i]),
                # v2.2
                "v22_pred": p_v22,
                "v22_conf": float(model_confs["v2_2"][i]),
                "v22_entropy": float(model_entropies["v2_2"][i]),
                "v22_margin": float(model_margins["v2_2"][i]),
                "v22_correct": bool(model_correct["v2_2"][i]),
                # Consensus
                "num_models_correct": num_correct,
                "num_models_predicting_true_label": num_correct,
                "modal_predicted_label": modal_pred,
                "modal_predicted_class": CLASS_NAMES[modal_pred],
                "modal_vote_count": modal_count,
                "prediction_vote_entropy": vote_entropy,
                "mean_confidence": mean_conf,
                "category": cat,
            })

        print(f"\n  Consensus Breakdown ({split_name.upper()}):")
        for cat_k, cnt in sorted(split_consensus_categories.items()):
            print(f"    {cat_k:<36}: {cnt:>5} ({cnt/N*100:.2f}%)")

        # --- A5.2 Oracle Complementarity Analysis ---
        # 1. Pair: v2.1 + v2.2
        corr_21 = model_correct["v2_1"]
        corr_22 = model_correct["v2_2"]
        pair_any = corr_21 | corr_22
        pair_both_c = corr_21 & corr_22
        pair_both_w = (~corr_21) & (~corr_22)
        pair_only_one = corr_21 ^ corr_22
        pair_disagree = (model_preds["v2_1"] != model_preds["v2_2"])

        # 2. Quad: v1 + v2 + v2.1 + v2.2
        c_v1 = model_correct["v1"]
        c_v2 = model_correct["v2"]
        quad_any = c_v1 | c_v2 | corr_21 | corr_22
        quad_all_c = c_v1 & c_v2 & corr_21 & corr_22
        quad_all_w = (~c_v1) & (~c_v2) & (~corr_21) & (~corr_22)
        num_c_arr = c_v1.astype(int) + c_v2.astype(int) + corr_21.astype(int) + corr_22.astype(int)
        quad_only_one = (num_c_arr == 1)
        quad_disagree = ~((model_preds["v1"] == model_preds["v2"]) & (model_preds["v2"] == model_preds["v2_1"]) & (model_preds["v2_1"] == model_preds["v2_2"]))

        oracle_summary[split_name] = {
            "sample_count": N,
            "v21_plus_v22": {
                "at_least_one_correct_count": int(np.sum(pair_any)),
                "oracle_accuracy": float(np.mean(pair_any)),
                "both_correct_count": int(np.sum(pair_both_c)),
                "both_correct_fraction": float(np.mean(pair_both_c)),
                "all_models_wrong_count": int(np.sum(pair_both_w)),
                "all_models_wrong_fraction": float(np.mean(pair_both_w)),
                "only_one_model_correct_count": int(np.sum(pair_only_one)),
                "only_one_model_correct_fraction": float(np.mean(pair_only_one)),
                "prediction_disagreement_count": int(np.sum(pair_disagree)),
                "prediction_disagreement_fraction": float(np.mean(pair_disagree)),
            },
            "four_models_all": {
                "at_least_one_correct_count": int(np.sum(quad_any)),
                "oracle_accuracy": float(np.mean(quad_any)),
                "all_correct_count": int(np.sum(quad_all_c)),
                "all_correct_fraction": float(np.mean(quad_all_c)),
                "all_models_wrong_count": int(np.sum(quad_all_w)),
                "all_models_wrong_fraction": float(np.mean(quad_all_w)),
                "only_one_model_correct_count": int(np.sum(quad_only_one)),
                "only_one_model_correct_fraction": float(np.mean(quad_only_one)),
                "prediction_disagreement_count": int(np.sum(quad_disagree)),
                "prediction_disagreement_fraction": float(np.mean(quad_disagree)),
                "consensus_distribution": {
                    f"{k}_of_4_correct": int(np.sum(num_c_arr == k)) for k in range(5)
                }
            }
        }

    # Save CSV
    csv_path = AUDIT_DIR / "a5_model_consensus.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(consensus_rows[0].keys()))
        writer.writeheader()
        writer.writerows(consensus_rows)
    print(f"\nSaved {csv_path} ({len(consensus_rows)} rows).")

    # Save Oracle JSON
    oracle_path = AUDIT_DIR / "a5_oracle_complementarity.json"
    oracle_path.write_text(json.dumps(oracle_summary, indent=2), encoding="utf-8")
    print(f"Saved {oracle_path}")

    # Save NPZ
    npz_path = AUDIT_DIR / "a5_multi_model_predictions.npz"
    np.savez_compressed(npz_path, **npz_save_dict)
    print(f"Saved {npz_path}")

    print(f"\nA5.2 finished in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
