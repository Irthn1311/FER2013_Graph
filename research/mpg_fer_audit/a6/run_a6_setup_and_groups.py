"""A6.0 Setup, Provenance Verification, and Sample Group Partitioning.

Verifies:
1. v2.1 and v2.2 source tree hashes and checkpoint SHA-256 digests.
2. Canonical FP32 single-view raw inference on PublicTest and PrivateTest.
3. Classifies all 7,178 test samples into:
   - V21_CORRECT_V22_WRONG
   - V22_CORRECT_V21_WRONG
   - BOTH_CORRECT
   - BOTH_WRONG
4. Applies exclusion masks to create:
   - MODEL_RESOLVABLE_ALL
   - MODEL_RESOLVABLE_CLEAN (excludes exact Train duplicates, label mismatches, benchmark conflicts)
   - MODEL_RESOLVABLE_STRICT_CLEAN (additionally excludes visual ambiguity candidates)
5. Produces:
   - a6_provenance.json
   - a6_sample_groups.csv
   - a6_clean_sample_groups.csv
"""

from __future__ import annotations

import csv
import hashlib
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
A5_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a5"
A5H_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a5h"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)

V21_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"
V22_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src"

sys.path.insert(0, str(V21_SRC))
sys.path.insert(0, str(V22_SRC))

from mpg_fer_v2_1.config import MPGConfig as MPGConfigV21
from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path
from mpg_fer_v2_1.model import MPGFER as MPGFERV21
from mpg_fer_v2_1.train import source_tree_hash as source_tree_hash_v21

from mpg_fer_v2_2.config import MPGConfig as MPGConfigV22
from mpg_fer_v2_2.model import MPGFER as MPGFERV22
from mpg_fer_v2_2.train import source_tree_hash as source_tree_hash_v22

V21_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "best_val_acc.pt"
V22_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "best_val_acc.pt"

EXPECTED_V21_SHA = "4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75"
EXPECTED_V22_SHA = "a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4"
EXPECTED_V21_SRC = "d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679"
EXPECTED_V22_SRC = "a8dc77db29e997c4c3ab69bb862704c8a948f940a4636e1c01e0d96bab40de65"

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running A6.0 Setup on {device}...", flush=True)

    # 1. Provenance Gate
    v21_sha = sha256_file(V21_CKPT)
    v22_sha = sha256_file(V22_CKPT)
    v21_src_h = source_tree_hash_v21(V21_SRC / "mpg_fer_v2_1")
    v22_src_h = source_tree_hash_v22(V22_SRC / "mpg_fer_v2_2")

    print(f"v2.1 SHA-256: {v21_sha} (matches expected: {v21_sha == EXPECTED_V21_SHA})")
    print(f"v2.2 SHA-256: {v22_sha} (matches expected: {v22_sha == EXPECTED_V22_SHA})")
    print(f"v2.1 Source : {v21_src_h} (matches expected: {v21_src_h == EXPECTED_V21_SRC})")
    print(f"v2.2 Source : {v22_src_h} (matches expected: {v22_src_h == EXPECTED_V22_SRC})")

    if not (v21_sha == EXPECTED_V21_SHA and v22_sha == EXPECTED_V22_SHA and
            v21_src_h == EXPECTED_V21_SRC and v22_src_h == EXPECTED_V22_SRC):
        print("STOP: A6_BLOCKED_PROVENANCE")
        sys.exit(1)

    # 2. Strict load models
    m21 = MPGFERV21(MPGConfigV21()).to(device).eval()
    m22 = MPGFERV22(MPGConfigV22()).to(device).eval()

    c21 = torch.load(V21_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    c22 = torch.load(V22_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]

    res21 = m21.load_state_dict(c21, strict=True)
    res22 = m22.load_state_dict(c22, strict=True)
    p21 = sum(p.numel() for p in m21.parameters())
    p22 = sum(p.numel() for p in m22.parameters())
    print(f"v2.1 strict load: {res21}, params: {p21:,}")
    print(f"v2.2 strict load: {res22}, params: {p22:,}")

    # 3. Load datasets
    val_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val"), split="val", augment=False)
    test_ds = FER2013Dataset(validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test"), split="test", augment=False)

    # 4. Load exclusion sets from A5-R and A5-H
    exact_dup_df = pd.read_csv(A5_DIR / "a5_exact_duplicates.csv")
    train_hashes = set(exact_dup_df[exact_dup_df["split"] == "train"]["sha256"])

    pub_train_dups = set(exact_dup_df[(exact_dup_df["split"] == "public") & (exact_dup_df["sha256"].isin(train_hashes))]["row_index"].astype(int))
    priv_train_dups = set(exact_dup_df[(exact_dup_df["split"] == "private") & (exact_dup_df["sha256"].isin(train_hashes))]["row_index"].astype(int))

    mismatch_df = pd.read_csv(A5H_DIR / "a5h_label_mismatch_candidates.csv")
    pub_mismatch = set(mismatch_df[mismatch_df["split"] == "public"]["row_index"].astype(int))
    priv_mismatch = set(mismatch_df[mismatch_df["split"] == "private"]["row_index"].astype(int))

    bench_df = pd.read_csv(A5H_DIR / "a5h_benchmark_conflict_candidates.csv")
    pub_bench = set(bench_df[bench_df["held_out_split"] == "public"]["held_out_row_index"].astype(int))
    priv_bench = set(bench_df[bench_df["held_out_split"] == "private"]["held_out_row_index"].astype(int))

    amb_df = pd.read_csv(A5H_DIR / "a5h_visual_ambiguity_candidates.csv")
    pub_amb = set(amb_df[amb_df["split"] == "public"]["row_index"].astype(int))
    priv_amb = set(amb_df[amb_df["split"] == "private"]["row_index"].astype(int))

    clean_excl = {
        "public": pub_train_dups | pub_mismatch | pub_bench,
        "private": priv_train_dups | priv_mismatch | priv_bench,
    }
    strict_clean_excl = {
        "public": clean_excl["public"] | pub_amb,
        "private": clean_excl["private"] | priv_amb,
    }

    print(f"Public clean exclusions: {len(clean_excl['public'])} (strict: {len(strict_clean_excl['public'])})")
    print(f"Private clean exclusions: {len(clean_excl['private'])} (strict: {len(strict_clean_excl['private'])})")

    # 5. Raw Single-View Inference & Sample Grouping
    sample_records = []
    clean_sample_records = []

    group_counts_all = {"public": {}, "private": {}}
    group_counts_clean = {"public": {}, "private": {}}

    for split_name, ds in [("public", val_ds), ("private", test_ds)]:
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        raw_l21, raw_l22 = [], []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                l21, _ = m21(x)
                l22, _ = m22(x)
                raw_l21.append(l21.cpu().numpy().astype(np.float32))
                raw_l22.append(l22.cpu().numpy().astype(np.float32))

        raw_l21 = np.concatenate(raw_l21, axis=0)
        raw_l22 = np.concatenate(raw_l22, axis=0)

        # Softmax in float64
        def softmax(arr):
            e = np.exp(arr.astype(np.float64) - np.max(arr, axis=-1, keepdims=True))
            return e / np.sum(e, axis=-1, keepdims=True)

        p21 = softmax(raw_l21)
        p22 = softmax(raw_l22)

        pred21 = np.argmax(p21, axis=-1)
        pred22 = np.argmax(p22, axis=-1)
        conf21 = np.max(p21, axis=-1)
        conf22 = np.max(p22, axis=-1)

        y_true = ds.labels.astype(int)
        N = len(y_true)

        c21 = (pred21 == y_true)
        c22 = (pred22 == y_true)

        print(f"\n{split_name.upper()} Raw Single-View FP32 Accuracies:")
        print(f"  v2.1: {np.mean(c21):.6f} ({int(np.sum(c21))}/{N})")
        print(f"  v2.2: {np.mean(c22):.6f} ({int(np.sum(c22))}/{N})")

        for idx in range(N):
            y_i = int(y_true[idx])
            p21_i = int(pred21[idx])
            p22_i = int(pred22[idx])
            c21_i = bool(c21[idx])
            c22_i = bool(c22[idx])

            if c21_i and not c22_i:
                cat = "V21_CORRECT_V22_WRONG"
            elif c22_i and not c21_i:
                cat = "V22_CORRECT_V21_WRONG"
            elif c21_i and c22_i:
                cat = "BOTH_CORRECT"
            else:
                cat = "BOTH_WRONG"

            group_counts_all[split_name][cat] = group_counts_all[split_name].get(cat, 0) + 1

            is_clean = (idx not in clean_excl[split_name])
            is_strict_clean = (idx not in strict_clean_excl[split_name])

            if is_clean:
                group_counts_clean[split_name][cat] = group_counts_clean[split_name].get(cat, 0) + 1

            rec = {
                "split": split_name,
                "row_index": idx,
                "true_label": y_i,
                "true_class": CLASS_NAMES[y_i],
                "v21_pred": p21_i,
                "v21_pred_class": CLASS_NAMES[p21_i],
                "v21_confidence": float(conf21[idx]),
                "v22_pred": p22_i,
                "v22_pred_class": CLASS_NAMES[p22_i],
                "v22_confidence": float(conf22[idx]),
                "sample_category": cat,
                "is_model_resolvable": bool(cat in ["V21_CORRECT_V22_WRONG", "V22_CORRECT_V21_WRONG"]),
                "is_clean": is_clean,
                "is_strict_clean": is_strict_clean,
                "has_train_duplicate": bool(idx in (pub_train_dups if split_name == "public" else priv_train_dups)),
            }
            sample_records.append(rec)
            if is_clean and rec["is_model_resolvable"]:
                clean_sample_records.append(rec)

    # Save all sample groups CSV
    all_csv_path = AUDIT_DIR / "a6_sample_groups.csv"
    pd.DataFrame(sample_records).to_csv(all_csv_path, index=False)
    print(f"\nSaved {all_csv_path} ({len(sample_records)} rows)")

    # Save clean model-resolvable CSV
    clean_csv_path = AUDIT_DIR / "a6_clean_sample_groups.csv"
    pd.DataFrame(clean_sample_records).to_csv(clean_csv_path, index=False)
    print(f"Saved {clean_csv_path} ({len(clean_sample_records)} clean model-resolvable rows)")

    # Print summary counts
    print("\n--- Model-Resolvable Counts Summary ---")
    for s in ["public", "private"]:
        all_res = group_counts_all[s]["V21_CORRECT_V22_WRONG"] + group_counts_all[s]["V22_CORRECT_V21_WRONG"]
        clean_res = group_counts_clean[s]["V21_CORRECT_V22_WRONG"] + group_counts_clean[s]["V22_CORRECT_V21_WRONG"]
        print(f"{s.upper()}:")
        print(f"  ALL Model-Resolvable  : {all_res} (V21_corr: {group_counts_all[s]['V21_CORRECT_V22_WRONG']}, V22_corr: {group_counts_all[s]['V22_CORRECT_V21_WRONG']})")
        print(f"  CLEAN Model-Resolvable: {clean_res} (V21_corr: {group_counts_clean[s]['V21_CORRECT_V22_WRONG']}, V22_corr: {group_counts_clean[s]['V22_CORRECT_V21_WRONG']})")

    # Provenance JSON
    prov_doc = {
        "audit": "MPG-FER A6",
        "device": device,
        "runtime_seconds": float(time.time() - start_time),
        "v2_1": {
            "source_hash": v21_src_h,
            "checkpoint_sha256": v21_sha,
            "parameters": p21,
            "weights_type": "EMA",
            "selected_epoch": 57,
        },
        "v2_2": {
            "source_hash": v22_src_h,
            "checkpoint_sha256": v22_sha,
            "parameters": p22,
            "weights_type": "EMA",
            "selected_epoch": 64,
            "topk_schedule": [8, 16, 16, 16, 24],
        },
        "inference_policy": "RAW_SINGLE_VIEW_FP32",
        "sample_counts": {
            "public_total": int(len(val_ds)),
            "private_total": int(len(test_ds)),
            "public_groups_all": {k: int(v) for k, v in group_counts_all["public"].items()},
            "private_groups_all": {k: int(v) for k, v in group_counts_all["private"].items()},
            "public_groups_clean": {k: int(v) for k, v in group_counts_clean["public"].items()},
            "private_groups_clean": {k: int(v) for k, v in group_counts_clean["private"].items()},
        }
    }
    def default_conv(o):
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    (AUDIT_DIR / "a6_provenance.json").write_text(json.dumps(prov_doc, indent=2, default=default_conv), encoding="utf-8")
    print(f"Saved {AUDIT_DIR / 'a6_provenance.json'}")


if __name__ == "__main__":
    main()
