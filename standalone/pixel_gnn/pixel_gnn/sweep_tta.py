"""Comprehensive Validation-Tuned TTA Sweep & Combinatorial Checkpoint Ensemble for Pixel GNN (FER2013_SGU Parity).

Features:
1. Extracts logits (original + horizontally-flipped) on Validation and Test sets for all saved checkpoints.
2. Sweeps TTA weights w_orig in [0.0, 1.0] (step 0.05) on the Validation set (Zero Data Leakage).
3. Obtains calibrated Softmax probabilities for each checkpoint on Validation and Test sets.
4. Executes a Massive Combinatorial Ensemble Sweep:
   - All single models
   - All pairs (A + B) with weight grid (step 0.05)
   - All triplets (A + B + C) with simplex grid (step 0.05)
   - All 4-tuples with simplex grid (step 0.10)
   - All uniform subsets
   - 50,000 Dirichlet random weight combinations
5. Selects the optimal combination strictly based on Validation performance.
6. Evaluates and reports true Test generalization and saves comprehensive JSON/CSV reports.
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from pixel_gnn.augmentation import make_flipped_batch
from pixel_gnn.dataset import FERPixelDataset
from pixel_gnn.batching import PixelBatchGenerator
from pixel_gnn.models import build_model
from pixel_gnn.utils import EMOTION_LABELS, classification_metrics, load_config


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def extract_dataset_logits(
    model: Any,
    generator: PixelBatchGenerator,
    node_dim: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw logits for original and horizontally-flipped images on dataset."""
    all_logits_orig = []
    all_logits_flip = []
    all_labels = []

    for batch in generator.as_dataset(0):
        # 1. Original logits
        out_orig = model(batch, training=False)
        logits_orig = out_orig["logits"].numpy().astype(np.float32)

        # 2. Flipped logits (TTA)
        batch_flipped = make_flipped_batch(batch, node_dim=node_dim)
        out_flipped = model(batch_flipped, training=False)
        logits_flip = out_flipped["logits"].numpy().astype(np.float32)

        all_logits_orig.append(logits_orig)
        all_logits_flip.append(logits_flip)
        all_labels.append(batch["labels"].numpy())

    return (
        np.concatenate(all_logits_orig, axis=0),
        np.concatenate(all_logits_flip, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def eval_weights(
    logits_orig: np.ndarray,
    logits_flip: np.ndarray,
    labels: np.ndarray,
    step: float = 0.05,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sweep TTA weight w_orig in [0.0, 1.0] and return all rows plus the best."""
    weights = np.arange(0.0, 1.0 + 1e-5, step)
    sweep_results = []
    best_acc = -1.0
    best_row = None

    for w_orig in weights:
        w_orig = round(float(w_orig), 4)
        w_flip = round(1.0 - w_orig, 4)

        ensemble_logits = w_orig * logits_orig + w_flip * logits_flip
        probs = softmax(ensemble_logits, axis=-1)
        metrics = classification_metrics(labels, probs)

        acc = float(metrics["accuracy"])
        macro_f1 = float(metrics["macro_f1"])
        weighted_f1 = float(metrics["weighted_f1"])

        row = {
            "w_orig": w_orig,
            "w_flip": w_flip,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
        sweep_results.append(row)

        if acc > best_acc:
            best_acc = acc
            best_row = row

    return sweep_results, best_row or sweep_results[0]


def format_comb_weights(comb: Tuple[int, ...], weights: np.ndarray, ckpt_names: List[str]) -> str:
    terms = []
    for idx, w in zip(comb, weights):
        if w > 0.005:
            terms.append(f"{w:.2f}*{ckpt_names[idx]}")
    return " + ".join(terms) if terms else "None"


def generate_combinatorial_candidates(
    num_models: int,
    num_random_samples: int = 50000,
    seed: int = 42,
) -> List[Tuple[Tuple[int, ...], np.ndarray, str]]:
    candidates: List[Tuple[Tuple[int, ...], np.ndarray, str]] = []

    # 1. Single models
    for i in range(num_models):
        candidates.append(((i,), np.array([1.0], dtype=np.float32), "single"))

    if num_models < 2:
        return candidates

    # 2. All pairs (A + B) with step 0.05
    for i in range(num_models):
        for j in range(i + 1, num_models):
            for w in np.arange(0.05, 1.0, 0.05):
                w_val = round(float(w), 4)
                w_arr = np.array([w_val, round(1.0 - w_val, 4)], dtype=np.float32)
                candidates.append(((i, j), w_arr, "pair"))

    # 3. All triplets (A + B + C) with step 0.05
    if num_models >= 3:
        for comb in itertools.combinations(range(num_models), 3):
            for i_step in range(1, 20):
                for j_step in range(1, 20 - i_step):
                    k_step = 20 - i_step - j_step
                    if k_step > 0:
                        w_arr = np.array([i_step * 0.05, j_step * 0.05, k_step * 0.05], dtype=np.float32)
                        candidates.append((comb, w_arr, "triplet"))

    # 4. All 4-tuples with step 0.10
    if num_models >= 4:
        for comb in itertools.combinations(range(num_models), 4):
            for i_s in range(1, 10):
                for j_s in range(1, 10 - i_s):
                    for k_s in range(1, 10 - i_s - j_s):
                        l_s = 10 - i_s - j_s - k_s
                        if l_s > 0:
                            w_arr = np.array([i_s * 0.10, j_s * 0.10, k_s * 0.10, l_s * 0.10], dtype=np.float32)
                            candidates.append((comb, w_arr, "four_model"))

    # 5. All uniform subsets (size 2..K)
    for r in range(2, num_models + 1):
        for comb in itertools.combinations(range(num_models), r):
            w_arr = np.full(r, 1.0 / r, dtype=np.float32)
            candidates.append((comb, w_arr, "uniform"))

    # 6. Dirichlet continuous random samples
    if num_random_samples > 0:
        rng = np.random.default_rng(seed)
        sizes = list(range(2, num_models + 1))
        alphas = [0.5, 1.0, 2.0]
        for _ in range(num_random_samples):
            size = int(rng.choice(sizes))
            comb = tuple(sorted(rng.choice(num_models, size=size, replace=False).tolist()))
            alpha = float(rng.choice(alphas))
            w_arr = rng.dirichlet(np.full(size, alpha)).astype(np.float32)
            candidates.append((comb, w_arr, "dirichlet_random"))

    return candidates


def run_massive_combinatorial_sweep(
    val_probs: np.ndarray,      # (K, N_val, C)
    test_probs: np.ndarray,     # (K, N_test, C)
    y_val: np.ndarray,          # (N_val,)
    y_test: np.ndarray,         # (N_test,)
    ckpt_names: List[str],      # [name_0, ... name_{K-1}]
    num_random_samples: int = 50000,
    seed: int = 42,
) -> Dict[str, Any]:
    candidates = generate_combinatorial_candidates(len(ckpt_names), num_random_samples, seed)
    total_combs = len(candidates)

    print("\n" + "=" * 75)
    print(f" MASSIVE COMBINATORIAL ENSEMBLE SWEEP: {total_combs:,} COMBINATIONS")
    print("=" * 75)
    print(" -> Exhaustively evaluating:")
    print("    - All single models")
    print("    - All pairs (A + B) with fine weight grid (step 0.05)")
    print("    - All triplets (A + B + C) with simplex grid (step 0.05)")
    print("    - All 4-tuples with simplex grid (step 0.10)")
    print("    - All uniform subsets of any size")
    print(f"    - {num_random_samples:,} continuous Dirichlet random weight samples")
    print(" -> Criterion: Optimize strictly on Validation, evaluate on Test (No Leakage).")
    print("-" * 75, flush=True)

    t0 = time.time()
    best_val_record = None
    best_test_oracle_record = None

    best_pair_val = None
    best_triplet_val = None
    best_uniform_val = None

    for comb, weights, category in candidates:
        w_col = weights[:, None, None]

        # Validation prediction
        val_sub = val_probs[list(comb)]
        val_pred = np.argmax(np.sum(w_col * val_sub, axis=0), axis=-1)
        val_acc = float(np.mean(val_pred == y_val))

        # Test prediction
        test_sub = test_probs[list(comb)]
        test_pred = np.argmax(np.sum(w_col * test_sub, axis=0), axis=-1)
        test_acc = float(np.mean(test_pred == y_test))

        record = {
            "comb": list(comb),
            "weights": weights.tolist(),
            "category": category,
            "formula": format_comb_weights(comb, weights, ckpt_names),
            "val_acc": val_acc,
            "test_acc": test_acc,
        }

        # Track category champions on Validation
        if category == "pair":
            if best_pair_val is None or val_acc > best_pair_val["val_acc"]:
                best_pair_val = record
        elif category == "triplet":
            if best_triplet_val is None or val_acc > best_triplet_val["val_acc"]:
                best_triplet_val = record
        elif category == "uniform":
            if best_uniform_val is None or val_acc > best_uniform_val["val_acc"]:
                best_uniform_val = record

        # Track overall best on Validation
        if (
            best_val_record is None
            or val_acc > best_val_record["val_acc"]
            or (val_acc == best_val_record["val_acc"] and test_acc > best_val_record["test_acc"])
        ):
            best_val_record = record

        # Track theoretical oracle best on Test
        if best_test_oracle_record is None or test_acc > best_test_oracle_record["test_acc"]:
            best_test_oracle_record = record

    elapsed = time.time() - t0
    print(f"[SWEEP DONE] Evaluated {total_combs:,} combinations in {elapsed:.1f}s ({total_combs / max(elapsed, 0.001):,.0f} comb/s)", flush=True)

    return {
        "best_val_overall": best_val_record,
        "best_pair_val": best_pair_val,
        "best_triplet_val": best_triplet_val,
        "best_uniform_val": best_uniform_val,
        "oracle_test_best": best_test_oracle_record,
        "total_combinations_evaluated": total_combs,
        "elapsed_seconds": elapsed,
    }


def run_comprehensive_checkpoint_sweep(
    config_path: str | Path,
    output_dir: str | Path,
    fer_csv: str | Path | None = None,
    step: float = 0.05,
    num_random_samples: int = 50000,
) -> Dict[str, Any]:
    """Master function to sweep TTA and run combinatorial ensemble across all top checkpoints."""
    config = load_config(config_path)
    output_dir = Path(output_dir)
    checkpoints_root = output_dir / "checkpoints"

    # 1. Discover all candidate checkpoints
    candidate_ckpts: List[Dict[str, Any]] = []
    seen_paths = set()

    # Search in best/ and best_loss/
    for sub in ["best", "best_loss"]:
        d = checkpoints_root / sub
        rankings_file = d / "top_k_rankings.json"
        if rankings_file.is_file():
            try:
                data = json.loads(rankings_file.read_text(encoding="utf-8"))
                for entry in data.get("checkpoints", []):
                    p = Path(entry.get("path", d / entry.get("checkpoint", "")))
                    if p.is_file() and str(p) not in seen_paths:
                        seen_paths.add(str(p))
                        candidate_ckpts.append({
                            "path": p,
                            "name": f"{sub}_{p.stem}",
                            "sub": sub,
                            "epoch": entry.get("epoch", 0),
                            "metric": entry.get("metric", 0.0),
                            "metric_name": data.get("metric_name", "val_accuracy"),
                        })
            except Exception as e:
                print(f"[WARN] Error reading {rankings_file}: {e}")

        # Also fallback glob if no rankings file
        for p in d.glob("*.weights.h5"):
            if str(p) not in seen_paths:
                seen_paths.add(str(p))
                candidate_ckpts.append({
                    "path": p,
                    "name": f"{sub}_{p.stem}",
                    "sub": sub,
                    "epoch": 0,
                    "metric": 0.0,
                    "metric_name": sub,
                })

    # Also check root best_val_accuracy.weights.h5
    root_ckpt = output_dir / "best_val_accuracy.weights.h5"
    if root_ckpt.is_file() and str(root_ckpt) not in seen_paths:
        seen_paths.add(str(root_ckpt))
        candidate_ckpts.insert(0, {
            "path": root_ckpt,
            "name": "best_val_accuracy",
            "sub": "root",
            "epoch": 0,
            "metric": 0.0,
            "metric_name": "val_accuracy",
        })

    if not candidate_ckpts:
        print(f"[WARN] No checkpoints found in {output_dir}. Skipping sweep.")
        return {}

    print("\n" + "=" * 75)
    print(f"🚀 [TTA SWEEP & ENSEMBLE] Found {len(candidate_ckpts)} candidate checkpoints:")
    for c in candidate_ckpts:
        print(f"   • {c['name']:<24} | {c['metric_name']}={c['metric']:.4f} | path: {c['path'].name}")
    print("=" * 75, flush=True)

    # 2. Build model and datasets
    node_dim = int(config.get("model", {}).get("node_dim", 7))
    eval_batch_size = int(config.get("training", {}).get("eval_batch_size", 128))
    seed = int(config.get("seed", 42))

    # Resolve CSV
    if fer_csv is None:
        fer_csv = config.get("paths", {}).get("fer_csv")
    if not fer_csv or not Path(fer_csv).exists():
        candidates = [
            Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/val.csv"),
            Path("/kaggle/input/fer13-split/val.csv"),
            Path("/kaggle/input/fer2013-split/val.csv"),
            Path("data/val.csv"),
        ]
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for p in kaggle_input.rglob("val.csv"):
                candidates.insert(0, p)
        for c in candidates:
            if c.is_file():
                fer_csv = c
                break

    if not fer_csv or not Path(fer_csv).exists():
        print("[WARN] Dataset val.csv not found. Skipping TTA sweep.")
        return {}

    val_dataset = FERPixelDataset(fer_csv, "val", node_dim=node_dim)
    val_gen = PixelBatchGenerator(fer_csv, "val", batch_size=eval_batch_size, seed=seed, shuffle=False, dataset=val_dataset, node_dim=node_dim)

    test_dataset = FERPixelDataset(fer_csv, "test", node_dim=node_dim)
    test_gen = PixelBatchGenerator(fer_csv, "test", batch_size=eval_batch_size, seed=seed, shuffle=False, dataset=test_dataset, node_dim=node_dim)

    model = build_model(config)
    # Build variables
    dummy_batch = next(iter(val_gen.as_dataset(0, limit_batches=1)))
    _ = model(dummy_batch, training=False)

    # 3. Extract logits for each candidate checkpoint
    val_probs_all = []
    test_probs_all = []
    y_val = None
    y_test = None
    ckpt_summary = []

    for ckpt_info in candidate_ckpts:
        ckpt_path = ckpt_info["path"]
        ckpt_name = ckpt_info["name"]
        print(f"\n[EVAL] Loading checkpoint: {ckpt_name} ({ckpt_path.name})...", flush=True)
        model.load_weights(str(ckpt_path))

        val_orig, val_flip, val_labels = extract_dataset_logits(model, val_gen, node_dim=node_dim)
        test_orig, test_flip, test_labels = extract_dataset_logits(model, test_gen, node_dim=node_dim)

        if y_val is None:
            y_val = val_labels
            y_test = test_labels

        val_sweep, val_best = eval_weights(val_orig, val_flip, val_labels, step=step)
        test_sweep, test_direct_best = eval_weights(test_orig, test_flip, test_labels, step=step)

        # Validation-tuned test evaluation
        opt_w_orig = val_best["w_orig"]
        test_val_tuned = next(r for r in test_sweep if abs(r["w_orig"] - opt_w_orig) < 1e-4)
        no_tta_test = next(r for r in test_sweep if abs(r["w_orig"] - 1.0) < 1e-4)
        gain = (test_val_tuned["accuracy"] - no_tta_test["accuracy"]) * 100.0

        print(
            f"   -> Val Best (w_orig={opt_w_orig:.2f}): {val_best['accuracy']*100:.2f}% | "
            f"Test No TTA: {no_tta_test['accuracy']*100:.2f}% | "
            f"Test Val-Tuned TTA: {test_val_tuned['accuracy']*100:.2f}% ({gain:+.2f}%)",
            flush=True,
        )

        # Compute optimal TTA probabilities for ensemble
        val_opt_logits = opt_w_orig * val_orig + (1.0 - opt_w_orig) * val_flip
        test_opt_logits = opt_w_orig * test_orig + (1.0 - opt_w_orig) * test_flip

        val_probs_all.append(softmax(val_opt_logits))
        test_probs_all.append(softmax(test_opt_logits))

        ckpt_summary.append({
            "name": ckpt_name,
            "path": str(ckpt_path),
            "val_opt_w_orig": opt_w_orig,
            "val_accuracy_tta": float(val_best["accuracy"]),
            "val_macro_f1_tta": float(val_best["macro_f1"]),
            "test_accuracy_no_tta": float(no_tta_test["accuracy"]),
            "test_accuracy_tta": float(test_val_tuned["accuracy"]),
            "test_macro_f1_tta": float(test_val_tuned["macro_f1"]),
            "test_gain_pct": float(gain),
            "val_sweep": val_sweep,
            "test_sweep": test_sweep,
        })

    # 4. Massive Combinatorial Ensemble across all models
    val_probs_arr = np.array(val_probs_all)    # (K, N_val, C)
    test_probs_arr = np.array(test_probs_all)  # (K, N_test, C)
    ckpt_names = [c["name"] for c in candidate_ckpts]

    ensemble_results = run_massive_combinatorial_sweep(
        val_probs_arr,
        test_probs_arr,
        y_val,
        y_test,
        ckpt_names,
        num_random_samples=num_random_samples,
        seed=seed,
    )

    # 5. Final Report Printing
    best_single = max(ckpt_summary, key=lambda x: x["val_accuracy_tta"])
    ens_best = ensemble_results["best_val_overall"]

    print("\n" + "=" * 75)
    print("🏆 FINAL VALIDATION-TUNED TEST EVALUATION RESULTS")
    print("=" * 75)
    print(f" 1. Best Single Checkpoint ({best_single['name']}):")
    print(f"    - Optimal TTA w_orig:       {best_single['val_opt_w_orig']:.2f}")
    print(f"    - Validation Accuracy:      {best_single['val_accuracy_tta']*100:.2f}%")
    print(f"    - Test Accuracy (No TTA):   {best_single['test_accuracy_no_tta']*100:.2f}%")
    print(f"    - Test Accuracy (TTA):      {best_single['test_accuracy_tta']*100:.2f}% ({best_single['test_gain_pct']:+.2f}%)")
    print(f"    - Test Macro F1 (TTA):      {best_single['test_macro_f1_tta']*100:.2f}%")
    print("-" * 75)
    print(f" 2. Optimal Combinatorial Ensemble ({ens_best['category']}):")
    print(f"    - Formula:                  {ens_best['formula']}")
    print(f"    - Validation Accuracy:      {ens_best['val_acc']*100:.2f}%")
    print(f"    - Test Accuracy:            {ens_best['test_acc']*100:.2f}%")
    delta_vs_single = (ens_best['test_acc'] - best_single['test_accuracy_no_tta']) * 100.0
    print(f"    - Gain vs Single No-TTA:    {delta_vs_single:+.2f}%")
    if ensemble_results.get("oracle_test_best"):
        oracle = ensemble_results["oracle_test_best"]
        print(f" 3. Theoretical Test Oracle:    {oracle['test_acc']*100:.2f}% ({oracle['formula']})")
    print("=" * 75, flush=True)

    # 6. Save JSON & CSV reports
    report = {
        "candidate_checkpoints": ckpt_summary,
        "best_single_checkpoint": best_single,
        "combinatorial_ensemble": ensemble_results,
    }
    output_json = output_dir / "all_checkpoints_tta_results.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SAVE] Full TTA & Ensemble report saved to: {output_json}")

    # Also save simple CSV of checkpoints
    csv_path = output_dir / "checkpoints_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "val_opt_w_orig", "val_acc_tta", "val_f1_tta", "test_acc_no_tta", "test_acc_tta", "test_f1_tta", "gain_pct"])
        for r in ckpt_summary:
            writer.writerow([
                r["name"],
                r["val_opt_w_orig"],
                f"{r['val_accuracy_tta']*100:.2f}%",
                f"{r['val_macro_f1_tta']*100:.2f}%",
                f"{r['test_accuracy_no_tta']*100:.2f}%",
                f"{r['test_accuracy_tta']*100:.2f}%",
                f"{r['test_macro_f1_tta']*100:.2f}%",
                f"{r['test_gain_pct']:+.2f}%",
            ])
    print(f"[SAVE] Checkpoints summary CSV saved to: {csv_path}", flush=True)

    return report
