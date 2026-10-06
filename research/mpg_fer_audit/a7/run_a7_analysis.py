"""A7 Cross-Depth Information Complementarity & Representation-Preservation Audit.

FROZEN POST-HOC AUDIT.
Strictly respects the Private-Test Firewall:
Phase 1: Single-stage probe reproduction & validation.
Phase 2: Public-only cross-depth analysis, controls, CKA, error sets, statistics, decision rules.
Phase 3: Write and lock a7_public_decision_lock.json BEFORE opening PrivateTest.
Phase 4: Post-freeze confirmatory Private evaluation against locked findings (predict only, no re-fitting).
Phase 5: Master results generation and consistency output.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
A7_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a7"
CACHE_DIR = A7_DIR / "cache"
A7_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

V22_ROOT = PROJECT_ROOT / "research" / "mpg_fer_v2_2"
V23_ROOT = PROJECT_ROOT / "research" / "mpg_fer_v2_3"

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

STAGE_MAPPING = {
    "PRE": "R1",
    "L1": "R2",
    "L2": "R3",
    "L5": "R6",
    "Motif Readout": "R7",
    "Fusion": "R8",
}

STAGE_DIMS = {
    "PRE": 192,
    "L1": 192,
    "L2": 192,
    "L5": 192,
    "Motif Readout": 384,
    "Fusion": 512,
}

CONFUSION_PAIRS = [
    ("Fear", "Sad"),
    ("Fear", "Neutral"),
    ("Fear", "Angry"),
    ("Sad", "Neutral"),
    ("Neutral", "Sad"),
    ("Angry", "Sad"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_head() -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return res.stdout.strip()


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    XtY = X_c.T @ Y_c
    hsic_xy = float(np.sum(XtY ** 2))
    XtX = X_c.T @ X_c
    hsic_xx = float(np.sum(XtX ** 2))
    YtY = Y_c.T @ Y_c
    hsic_yy = float(np.sum(YtY ** 2))
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0


def mcnemar_exact_test(y_true: np.ndarray, pred_base: np.ndarray, pred_cand: np.ndarray) -> dict:
    base_correct = (pred_base == y_true)
    cand_correct = (pred_cand == y_true)
    b = int(np.sum(base_correct & ~cand_correct))  # base correct, cand wrong
    c = int(np.sum(~base_correct & cand_correct))  # base wrong, cand correct
    n = b + c
    if n == 0:
        p_val = 1.0
    else:
        res = stats.binomtest(b, n=n, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)
    return {
        "b_base_correct_cand_wrong": b,
        "c_base_wrong_cand_correct": c,
        "discordant_total": n,
        "p_value": p_val,
    }


def paired_bootstrap_ci(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_cand: np.ndarray,
    b_reps: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx_mat = rng.integers(0, n, size=(b_reps, n))

    y_mat = y_true[idx_mat]
    pb_mat = pred_base[idx_mat]
    pc_mat = pred_cand[idx_mat]

    acc_base = np.mean(pb_mat == y_mat, axis=1)
    acc_cand = np.mean(pc_mat == y_mat, axis=1)
    delta_accs = acc_cand - acc_base

    def compute_macro_f1_mat(p_mat, y_m, num_classes=7):
        f1_sum = np.zeros(p_mat.shape[0], dtype=np.float64)
        for c in range(num_classes):
            tp = np.sum((p_mat == c) & (y_m == c), axis=1)
            fp = np.sum((p_mat == c) & (y_m != c), axis=1)
            fn = np.sum((p_mat != c) & (y_m == c), axis=1)
            denom = 2 * tp + fp + fn
            f1_c = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
            f1_sum += f1_c
        return f1_sum / num_classes

    f1_base = compute_macro_f1_mat(pb_mat, y_mat)
    f1_cand = compute_macro_f1_mat(pc_mat, y_mat)
    delta_f1s = f1_cand - f1_base

    ci_acc_low = float(np.percentile(delta_accs, 2.5))
    ci_acc_high = float(np.percentile(delta_accs, 97.5))
    ci_f1_low = float(np.percentile(delta_f1s, 2.5))
    ci_f1_high = float(np.percentile(delta_f1s, 97.5))

    return {
        "b_reps": b_reps,
        "seed": seed,
        "delta_acc_ci_95": [ci_acc_low, ci_acc_high],
        "delta_acc_ci_excludes_zero": bool(ci_acc_low > 0 or ci_acc_high < 0),
        "delta_f1_ci_95": [ci_f1_low, ci_f1_high],
        "delta_f1_ci_excludes_zero": bool(ci_f1_low > 0 or ci_f1_high < 0),
    }


class ProbeRunner:
    def __init__(self):
        print("Loading cached representations...", flush=True)
        self.v23_cache = np.load(V23_ROOT / "official_runs" / "analysis_cache" / "v23_frozen_features.npz")
        self.a6_tr = np.load(PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6" / "a6_train_features.npz")
        self.a6_pub = np.load(PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6" / "a6_public_features.npz")
        self.a6_priv = np.load(PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6" / "a6_private_features.npz")

        self.y_train = self.v23_cache["train_targets"]
        self.y_public = self.v23_cache["public_targets"]
        self.y_private = self.v23_cache["private_targets"]

        # Validate alignment
        assert np.array_equal(self.y_train, self.a6_tr["targets"]), "Train target alignment mismatch!"
        assert np.array_equal(self.y_public, self.a6_pub["targets"]), "Public target alignment mismatch!"
        assert np.array_equal(self.y_private, self.a6_priv["targets"]), "Private target alignment mismatch!"

    def get_stage_feat(self, model: str, stage: str, split: str) -> np.ndarray:
        stage_key = STAGE_MAPPING[stage]
        if model == "v23":
            key = f"{split}_{stage_key}"
            return self.v23_cache[key]
        elif model == "v22":
            if split == "train":
                return self.a6_tr[f"v22_{stage_key}"]
            elif split == "public":
                return self.a6_pub[f"v22_{stage_key}"]
            elif split == "private":
                return self.a6_priv[f"v22_{stage_key}"]
            else:
                raise ValueError(f"Unknown split {split}")
        else:
            raise ValueError(f"Unknown model {model}")

    def get_or_fit_probe(self, cache_key: str, X_tr: np.ndarray) -> tuple:
        ckpt_path = CACHE_DIR / f"probe_{cache_key}.npz"
        if ckpt_path.exists():
            data = np.load(ckpt_path)
            return data["coef"], data["intercept"], data["mean"], data["scale"], int(data["n_iter"]), float(data["fit_duration_sec"])

        t0 = time.time()
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        clf = LogisticRegression(
            C=1.0,
            max_iter=5000,
            tol=1e-6,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X_tr_s, self.y_train)
        duration = time.time() - t0

        np.savez_compressed(
            ckpt_path,
            coef=clf.coef_.astype(np.float32),
            intercept=clf.intercept_.astype(np.float32),
            mean=scaler.mean_.astype(np.float32),
            scale=scaler.scale_.astype(np.float32),
            n_iter=clf.n_iter_[0],
            fit_duration_sec=duration,
        )
        return clf.coef_, clf.intercept_, scaler.mean_, scaler.scale_, int(clf.n_iter_[0]), duration

    def eval_probe(self, coef: np.ndarray, intercept: np.ndarray, mean: np.ndarray, scale: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple:
        Xs = (X - mean) / scale
        logits = Xs @ coef.T + intercept
        preds = np.argmax(logits, axis=-1)
        acc = float(np.mean(preds == y))
        f1 = float(f1_score(y, preds, average="macro", zero_division=0))
        return acc, f1, preds

    def get_or_fit_pca(self, cache_key: str, X_tr: np.ndarray, target_dim: int) -> tuple:
        ckpt_path = CACHE_DIR / f"pca_{cache_key}.npz"
        if ckpt_path.exists():
            data = np.load(ckpt_path)
            return (
                data["components"],
                data["pca_mean"],
                data["prep_mean"],
                data["prep_scale"],
                float(data["explained_variance_ratio_sum"]),
            )

        scaler_prep = StandardScaler()
        X_tr_p = scaler_prep.fit_transform(X_tr)
        pca = PCA(n_components=target_dim, svd_solver="full", random_state=42)
        pca.fit(X_tr_p)

        np.savez_compressed(
            ckpt_path,
            components=pca.components_.astype(np.float32),
            pca_mean=pca.mean_.astype(np.float32),
            prep_mean=scaler_prep.mean_.astype(np.float32),
            prep_scale=scaler_prep.scale_.astype(np.float32),
            explained_variance_ratio_sum=float(np.sum(pca.explained_variance_ratio_)),
        )
        return pca.components_, pca.mean_, scaler_prep.mean_, scaler_prep.scale_, float(np.sum(pca.explained_variance_ratio_))

    def transform_pca(self, components: np.ndarray, pca_mean: np.ndarray, prep_mean: np.ndarray, prep_scale: np.ndarray, X: np.ndarray) -> np.ndarray:
        X_p = (X - prep_mean) / prep_scale
        return (X_p - pca_mean) @ components.T

    def get_or_fit_ridge(self, cache_key: str, L5_tr: np.ndarray, E_tr: np.ndarray) -> tuple:
        ckpt_path = CACHE_DIR / f"ridge_{cache_key}.npz"
        if ckpt_path.exists():
            data = np.load(ckpt_path)
            return data["coef"], data["intercept"]

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(L5_tr, E_tr)
        np.savez_compressed(
            ckpt_path,
            coef=ridge.coef_.astype(np.float32),
            intercept=ridge.intercept_.astype(np.float32),
        )
        return ridge.coef_, ridge.intercept_

    def predict_ridge(self, coef: np.ndarray, intercept: np.ndarray, X: np.ndarray) -> np.ndarray:
        return X @ coef.T + intercept


def main():
    print("============================================================", flush=True)
    print("MPG-FER A7 — CROSS-DEPTH AUDIT PIPELINE STARTING", flush=True)
    print("============================================================", flush=True)
    t_start = time.time()
    runner = ProbeRunner()

    with open(PROJECT_ROOT / "research" / "mpg_fer_v2_3" / "v23_probe_analysis.json") as f:
        auth_v23_doc = json.load(f)

    # -------------------------------------------------------------
    # 1. REPRODUCE SINGLE-STAGE PROBES
    # -------------------------------------------------------------
    print("\n--- PHASE 1: SINGLE-STAGE PROBE REPRODUCTION ---", flush=True)
    single_stage_metrics_path = A7_DIR / "a7_single_stage_probe_metrics.json"

    stages_order = ["PRE", "L1", "L2", "L5", "Motif Readout", "Fusion"]
    single_stage_metrics = {"v22": {}, "v23": {}}
    single_stage_preds_pub = {"v22": {}, "v23": {}}
    single_stage_preds_priv = {"v22": {}, "v23": {}}

    for model in ["v23", "v22"]:
        print(f"\nModel {model}: Single-Stage Probes:", flush=True)
        for stage in stages_order:
            X_tr = runner.get_stage_feat(model, stage, "train")
            X_pub = runner.get_stage_feat(model, stage, "public")
            X_priv = runner.get_stage_feat(model, stage, "private")

            coef, intercept, mean, scale, n_iter, duration = runner.get_or_fit_probe(f"{model}_{stage}", X_tr)

            acc_tr, f1_tr, pred_tr = runner.eval_probe(coef, intercept, mean, scale, X_tr, runner.y_train)
            acc_pub, f1_pub, pred_pub = runner.eval_probe(coef, intercept, mean, scale, X_pub, runner.y_public)
            acc_priv, f1_priv, pred_priv = runner.eval_probe(coef, intercept, mean, scale, X_priv, runner.y_private)

            single_stage_preds_pub[model][stage] = pred_pub
            single_stage_preds_priv[model][stage] = pred_priv

            if model == "v23":
                auth = auth_v23_doc["v23"][stage]
                auth_pub_acc = auth["public_accuracy"]
                auth_priv_acc = auth["private_accuracy"]
            else:
                auth = [x for x in auth_v23_doc["v22_v23_depth_specialization"] if x["stage"] == stage][0]
                auth_pub_acc = auth["v22_public"]
                auth_priv_acc = auth["v22_private"]

            diff_pub = abs(acc_pub - auth_pub_acc) * 100
            diff_priv = abs(acc_priv - auth_priv_acc) * 100

            print(
                f"  {stage:14s} | Dim: {X_tr.shape[1]:3d} | "
                f"Pub Acc: {acc_pub*100:.4f}% (auth: {auth_pub_acc*100:.4f}%, diff: {diff_pub:.4f}pp) | "
                f"Priv Acc: {acc_priv*100:.4f}% (auth: {auth_priv_acc*100:.4f}%, diff: {diff_priv:.4f}pp) | "
                f"Iters: {n_iter:4d}",
                flush=True,
            )

            if diff_pub > 0.10 or diff_priv > 0.10:
                raise RuntimeError(
                    f"Reproduction FAILED for {model} {stage}: Pub diff {diff_pub:.4f}pp, Priv diff {diff_priv:.4f}pp > 0.10pp"
                )

            single_stage_metrics[model][stage] = {
                "stage": stage,
                "a6_stage": STAGE_MAPPING[stage],
                "dim": int(X_tr.shape[1]),
                "train_acc": acc_tr,
                "train_macro_f1": f1_tr,
                "public_acc": acc_pub,
                "public_macro_f1": f1_pub,
                "private_acc": acc_priv,
                "private_macro_f1": f1_priv,
                "train_public_gap": acc_tr - acc_pub,
                "train_private_gap": acc_tr - acc_priv,
                "iterations": n_iter,
                "fit_duration_sec": duration,
                "auth_diff_pub_pp": diff_pub,
                "auth_diff_priv_pp": diff_priv,
                "status": "PASS",
            }

    with open(single_stage_metrics_path, "w", encoding="utf-8") as f:
        json.dump(single_stage_metrics, f, indent=2)
    print(f"\nWrote: {single_stage_metrics_path}", flush=True)

    # -------------------------------------------------------------
    # 2. PHASE 2: PUBLIC-ONLY CROSS-DEPTH EXPLORATION & CONTROLS
    # -------------------------------------------------------------
    print("\n============================================================", flush=True)
    print("--- PHASE 2: PUBLIC-ONLY CROSS-DEPTH ANALYSIS (STRICT FIREWALL) ---", flush=True)
    print("============================================================", flush=True)

    primary_pairs_config = [
        {"name": "L1+L5", "stages": ["L1", "L5"], "baseline_stage": "L5", "control_dup": "L5+L5", "pca_target_dim": 192},
        {"name": "L2+L5", "stages": ["L2", "L5"], "baseline_stage": "L5", "control_dup": "L5+L5", "pca_target_dim": 192},
        {"name": "PRE+L5", "stages": ["PRE", "L5"], "baseline_stage": "L5", "control_dup": "L5+L5", "pca_target_dim": 192},
        {"name": "L1+L2+L5", "stages": ["L1", "L2", "L5"], "baseline_stage": "L5", "control_dup": "L5+L5+L5", "pca_target_dim": 192},
        {"name": "L2+Motif Readout", "stages": ["L2", "Motif Readout"], "baseline_stage": "Motif Readout", "control_dup": "Motif Readout+Motif Readout", "pca_target_dim": 384},
        {"name": "L2+Fusion", "stages": ["L2", "Fusion"], "baseline_stage": "Fusion", "control_dup": "Fusion+Fusion", "pca_target_dim": 512},
        {"name": "L5+Motif Readout", "stages": ["L5", "Motif Readout"], "baseline_stage": "Motif Readout", "control_dup": "Motif Readout+Motif Readout", "pca_target_dim": 384},
        {"name": "L5+Fusion", "stages": ["L5", "Fusion"], "baseline_stage": "Fusion", "control_dup": "Fusion+Fusion", "pca_target_dim": 512},
    ]

    raw_concat_results = {"v22": {}, "v23": {}}
    duplicate_results = {"v22": {}, "v23": {}}
    pca_results = {"v22": {}, "v23": {}}
    residual_results = {"v22": {}, "v23": {}}

    preds_for_csv_pub = {}

    for model in ["v23", "v22"]:
        print(f"\n>>> Running Public Analysis for Model {model} <<<", flush=True)

        # 2.1 Duplicate controls
        print("\n  [Evaluating Duplicate Controls]", flush=True)
        dup_specs = [
            ("L5+L5", ["L5", "L5"], "L5"),
            ("L5+L5+L5", ["L5", "L5", "L5"], "L5"),
            ("Motif Readout+Motif Readout", ["Motif Readout", "Motif Readout"], "Motif Readout"),
            ("Fusion+Fusion", ["Fusion", "Fusion"], "Fusion"),
        ]
        for dup_name, d_stages, b_stage in dup_specs:
            X_tr_dup = np.hstack([runner.get_stage_feat(model, s, "train") for s in d_stages])
            X_pub_dup = np.hstack([runner.get_stage_feat(model, s, "public") for s in d_stages])

            coef, intercept, mean, scale, n_iter, duration = runner.get_or_fit_probe(f"{model}_dup_{dup_name}", X_tr_dup)

            acc_tr, f1_tr, _ = runner.eval_probe(coef, intercept, mean, scale, X_tr_dup, runner.y_train)
            acc_pub, f1_pub, _ = runner.eval_probe(coef, intercept, mean, scale, X_pub_dup, runner.y_public)

            base_res = single_stage_metrics[model][b_stage]
            delta_acc = acc_pub - base_res["public_acc"]
            delta_f1 = f1_pub - base_res["public_macro_f1"]

            duplicate_results[model][dup_name] = {
                "name": dup_name,
                "dim": int(X_tr_dup.shape[1]),
                "baseline_stage": b_stage,
                "train_acc": acc_tr,
                "train_macro_f1": f1_tr,
                "public_acc": acc_pub,
                "public_macro_f1": f1_pub,
                "delta_acc_vs_late": delta_acc,
                "delta_f1_vs_late": delta_f1,
                "iterations": n_iter,
            }
            print(
                f"    {dup_name:28s} | Dim: {X_tr_dup.shape[1]:4d} | "
                f"Pub Acc: {acc_pub*100:.4f}% | Delta Acc: {delta_acc*100:+.4f}pp | "
                f"Pub F1: {f1_pub*100:.4f}%",
                flush=True,
            )

        # 2.2 Raw Concatenation and PCA Controls
        print("\n  [Evaluating Raw Concat & PCA Controls]", flush=True)
        for pair in primary_pairs_config:
            p_name = pair["name"]
            b_stage = pair["baseline_stage"]
            pca_dim = pair["pca_target_dim"]

            # Raw concat
            X_tr_raw = np.hstack([runner.get_stage_feat(model, s, "train") for s in pair["stages"]])
            X_pub_raw = np.hstack([runner.get_stage_feat(model, s, "public") for s in pair["stages"]])

            coef_raw, intercept_raw, mean_raw, scale_raw, n_iter_raw, dur_raw = runner.get_or_fit_probe(f"{model}_raw_{p_name}", X_tr_raw)

            acc_tr_raw, f1_tr_raw, _ = runner.eval_probe(coef_raw, intercept_raw, mean_raw, scale_raw, X_tr_raw, runner.y_train)
            acc_pub_raw, f1_pub_raw, pred_cand_pub = runner.eval_probe(coef_raw, intercept_raw, mean_raw, scale_raw, X_pub_raw, runner.y_public)

            base_res = single_stage_metrics[model][b_stage]
            delta_acc_raw = acc_pub_raw - base_res["public_acc"]
            delta_f1_raw = f1_pub_raw - base_res["public_macro_f1"]

            pred_base_pub = single_stage_preds_pub[model][b_stage]
            preds_for_csv_pub[f"{model}_{p_name}_pred"] = pred_cand_pub

            boot_raw = paired_bootstrap_ci(runner.y_public, pred_base_pub, pred_cand_pub, b_reps=2000, seed=42)
            mcnemar_raw = mcnemar_exact_test(runner.y_public, pred_base_pub, pred_cand_pub)

            raw_concat_results[model][p_name] = {
                "name": p_name,
                "stages": pair["stages"],
                "baseline_stage": b_stage,
                "dim": int(X_tr_raw.shape[1]),
                "train_acc": acc_tr_raw,
                "train_macro_f1": f1_tr_raw,
                "public_acc": acc_pub_raw,
                "public_macro_f1": f1_pub_raw,
                "delta_acc_vs_late": delta_acc_raw,
                "delta_f1_vs_late": delta_f1_raw,
                "iterations": n_iter_raw,
                "bootstrap_ci": boot_raw,
                "mcnemar": mcnemar_raw,
            }

            # Train-Only PCA Control
            components, pca_mean, prep_mean, prep_scale, exp_var = runner.get_or_fit_pca(f"{model}_{p_name}_{pca_dim}", X_tr_raw, pca_dim)
            X_tr_pca = runner.transform_pca(components, pca_mean, prep_mean, prep_scale, X_tr_raw)
            X_pub_pca = runner.transform_pca(components, pca_mean, prep_mean, prep_scale, X_pub_raw)

            coef_pca, intercept_pca, mean_pca, scale_pca, n_iter_pca, dur_pca = runner.get_or_fit_probe(f"{model}_pca_{p_name}_{pca_dim}", X_tr_pca)

            acc_tr_pca, f1_tr_pca, _ = runner.eval_probe(coef_pca, intercept_pca, mean_pca, scale_pca, X_tr_pca, runner.y_train)
            acc_pub_pca, f1_pub_pca, pred_pca_pub = runner.eval_probe(coef_pca, intercept_pca, mean_pca, scale_pca, X_pub_pca, runner.y_public)

            delta_acc_pca = acc_pub_pca - base_res["public_acc"]
            delta_f1_pca = f1_pub_pca - base_res["public_macro_f1"]

            preds_for_csv_pub[f"{model}_PCA_{p_name}_pred"] = pred_pca_pub

            boot_pca = paired_bootstrap_ci(runner.y_public, pred_base_pub, pred_pca_pub, b_reps=2000, seed=42)
            mcnemar_pca = mcnemar_exact_test(runner.y_public, pred_base_pub, pred_pca_pub)

            pca_results[model][p_name] = {
                "name": f"PCA({p_name})->{pca_dim}",
                "original_dim": int(X_tr_raw.shape[1]),
                "reduced_dim": pca_dim,
                "baseline_stage": b_stage,
                "train_acc": acc_tr_pca,
                "train_macro_f1": f1_tr_pca,
                "public_acc": acc_pub_pca,
                "public_macro_f1": f1_pub_pca,
                "delta_acc_vs_late": delta_acc_pca,
                "delta_f1_vs_late": delta_f1_pca,
                "iterations": n_iter_pca,
                "bootstrap_ci": boot_pca,
                "mcnemar": mcnemar_pca,
                "explained_variance_ratio_sum": exp_var,
            }

            print(
                f"    {p_name:18s} | Raw: Pub Acc {acc_pub_raw*100:.4f}% ({delta_acc_raw*100:+.4f}pp, CI [{boot_raw['delta_acc_ci_95'][0]*100:+.2f}, {boot_raw['delta_acc_ci_95'][1]*100:+.2f}], p={mcnemar_raw['p_value']:.4f}) | "
                f"PCA->{pca_dim:3d}: Pub Acc {acc_pub_pca*100:.4f}% ({delta_acc_pca*100:+.4f}pp, CI [{boot_pca['delta_acc_ci_95'][0]*100:+.2f}, {boot_pca['delta_acc_ci_95'][1]*100:+.2f}], p={mcnemar_pca['p_value']:.4f})",
                flush=True,
            )

        # 2.3 Conditional Unique-Information Analysis (Ridge residualization)
        print("\n  [Evaluating Conditional Unique-Information (Ridge Residualization)]", flush=True)
        L5_tr = runner.get_stage_feat(model, "L5", "train")
        L5_pub = runner.get_stage_feat(model, "L5", "public")

        for e_stage in ["PRE", "L1", "L2"]:
            E_tr = runner.get_stage_feat(model, e_stage, "train")
            E_pub = runner.get_stage_feat(model, e_stage, "public")

            r_coef, r_intercept = runner.get_or_fit_ridge(f"{model}_L5_to_{e_stage}", L5_tr, E_tr)

            E_res_tr = E_tr - runner.predict_ridge(r_coef, r_intercept, L5_tr)
            E_res_pub = E_pub - runner.predict_ridge(r_coef, r_intercept, L5_pub)

            X_tr_res = np.hstack([L5_tr, E_res_tr])
            X_pub_res = np.hstack([L5_pub, E_res_pub])

            c_res, i_res, m_res, s_res, it_res, _ = runner.get_or_fit_probe(f"{model}_L5+{e_stage}_res", X_tr_res)

            acc_tr_res, _, _ = runner.eval_probe(c_res, i_res, m_res, s_res, X_tr_res, runner.y_train)
            acc_pub_res, f1_pub_res, _ = runner.eval_probe(c_res, i_res, m_res, s_res, X_pub_res, runner.y_public)

            # Shuffled control
            rng_tr = np.random.default_rng(42)
            perm_tr = rng_tr.permutation(len(E_res_tr))
            E_res_tr_shuf = E_res_tr[perm_tr]

            rng_pub = np.random.default_rng(42)
            perm_pub = rng_pub.permutation(len(E_res_pub))
            E_res_pub_shuf = E_res_pub[perm_pub]

            X_tr_shuf = np.hstack([L5_tr, E_res_tr_shuf])
            X_pub_shuf = np.hstack([L5_pub, E_res_pub_shuf])

            c_shuf, i_shuf, m_shuf, s_shuf, it_shuf, _ = runner.get_or_fit_probe(f"{model}_L5+{e_stage}_shuf", X_tr_shuf)
            acc_pub_shuf, _, _ = runner.eval_probe(c_shuf, i_shuf, m_shuf, s_shuf, X_pub_shuf, runner.y_public)

            base_res = single_stage_metrics[model]["L5"]
            delta_true = acc_pub_res - base_res["public_acc"]
            delta_shuf = acc_pub_shuf - base_res["public_acc"]

            residual_results[model][f"L5+{e_stage}_res"] = {
                "early_stage": e_stage,
                "late_stage": "L5",
                "dim": int(X_tr_res.shape[1]),
                "train_acc": acc_tr_res,
                "public_acc": acc_pub_res,
                "public_macro_f1": f1_pub_res,
                "delta_acc_vs_l5": delta_true,
                "shuffled_control_public_acc": acc_pub_shuf,
                "shuffled_control_delta_acc": delta_shuf,
            }
            print(
                f"    L5 + {e_stage}_res: Pub Acc {acc_pub_res*100:.4f}% ({delta_true*100:+.4f}pp) | "
                f"Shuffled Control: {acc_pub_shuf*100:.4f}% ({delta_shuf*100:+.4f}pp)",
                flush=True,
            )

    with open(A7_DIR / "a7_cross_depth_raw_concat.json", "w", encoding="utf-8") as f:
        json.dump(raw_concat_results, f, indent=2)
    with open(A7_DIR / "a7_duplicate_control.json", "w", encoding="utf-8") as f:
        json.dump(duplicate_results, f, indent=2)
    with open(A7_DIR / "a7_pca_dimension_control.json", "w", encoding="utf-8") as f:
        json.dump(pca_results, f, indent=2)
    with open(A7_DIR / "a7_residualized_information.json", "w", encoding="utf-8") as f:
        json.dump(residual_results, f, indent=2)

    # -------------------------------------------------------------
    # 3. PROBE COMPLEMENTARITY SETS & CLASSWISE ANALYSIS
    # -------------------------------------------------------------
    print("\n--- PHASE 2.4: PROBE COMPLEMENTARITY SETS & CLASSWISE ANALYSIS ---", flush=True)

    complementarity_sets = {"v22": {}, "v23": {}}
    classwise_summary = {"v22": {}, "v23": {}}
    confusion_analysis = {"v22": {}, "v23": {}}
    cka_results = {"v22": {}, "v23": {}}

    for model in ["v23", "v22"]:
        L5_preds = single_stage_preds_pub[model]["L5"]
        y_pub = runner.y_public

        # CKA Analysis
        cka_results[model] = {
            "train": {
                "PRE_vs_L5": linear_cka(runner.get_stage_feat(model, "PRE", "train"), runner.get_stage_feat(model, "L5", "train")),
                "L1_vs_L5": linear_cka(runner.get_stage_feat(model, "L1", "train"), runner.get_stage_feat(model, "L5", "train")),
                "L2_vs_L5": linear_cka(runner.get_stage_feat(model, "L2", "train"), runner.get_stage_feat(model, "L5", "train")),
                "L5_vs_MotifReadout": linear_cka(runner.get_stage_feat(model, "L5", "train"), runner.get_stage_feat(model, "Motif Readout", "train")),
                "L5_vs_Fusion": linear_cka(runner.get_stage_feat(model, "L5", "train"), runner.get_stage_feat(model, "Fusion", "train")),
            },
            "public": {
                "PRE_vs_L5": linear_cka(runner.get_stage_feat(model, "PRE", "public"), runner.get_stage_feat(model, "L5", "public")),
                "L1_vs_L5": linear_cka(runner.get_stage_feat(model, "L1", "public"), runner.get_stage_feat(model, "L5", "public")),
                "L2_vs_L5": linear_cka(runner.get_stage_feat(model, "L2", "public"), runner.get_stage_feat(model, "L5", "public")),
                "L5_vs_MotifReadout": linear_cka(runner.get_stage_feat(model, "L5", "public"), runner.get_stage_feat(model, "Motif Readout", "public")),
                "L5_vs_Fusion": linear_cka(runner.get_stage_feat(model, "L5", "public"), runner.get_stage_feat(model, "Fusion", "public")),
            },
        }

        # Error overlap and set analysis for early stages vs L5
        for e_stage in ["PRE", "L1", "L2"]:
            e_preds = single_stage_preds_pub[model][e_stage]
            joint_raw_preds = preds_for_csv_pub[f"{model}_{e_stage}+L5_pred"]
            joint_pca_preds = preds_for_csv_pub[f"{model}_PCA_{e_stage}+L5_pred"]

            e_correct = (e_preds == y_pub)
            l5_correct = (L5_preds == y_pub)
            oracle_correct = (e_correct | l5_correct)

            early_c_late_w = (e_correct & ~l5_correct)
            early_w_late_c = (~e_correct & l5_correct)
            both_c = (e_correct & l5_correct)
            both_w = (~e_correct & ~l5_correct)

            num_early_c_late_w = int(np.sum(early_c_late_w))
            num_rescued_raw = int(np.sum((joint_raw_preds == y_pub) & early_c_late_w)) if num_early_c_late_w > 0 else 0
            num_rescued_pca = int(np.sum((joint_pca_preds == y_pub) & early_c_late_w)) if num_early_c_late_w > 0 else 0

            e_errors = set(np.where(~e_correct)[0])
            l5_errors = set(np.where(~l5_correct)[0])
            jaccard = len(e_errors & l5_errors) / len(e_errors | l5_errors)
            frac_l5_errors_solved_by_e = len(l5_errors - e_errors) / len(l5_errors) if len(l5_errors) > 0 else 0.0
            frac_e_errors_solved_by_l5 = len(e_errors - l5_errors) / len(e_errors) if len(e_errors) > 0 else 0.0

            complementarity_sets[model][f"{e_stage}_vs_L5"] = {
                "early_stage": e_stage,
                "total_samples": len(y_pub),
                "both_correct": int(np.sum(both_c)),
                "both_wrong": int(np.sum(both_w)),
                "early_correct_late_wrong": num_early_c_late_w,
                "early_wrong_late_correct": int(np.sum(early_w_late_c)),
                "oracle_union_accuracy": float(np.mean(oracle_correct)),
                "oracle_union_gain_vs_l5": float(np.mean(oracle_correct) - np.mean(l5_correct)),
                "rescued_by_raw_concat": num_rescued_raw,
                "rescued_by_raw_concat_pct": float(num_rescued_raw / num_early_c_late_w) if num_early_c_late_w > 0 else 0.0,
                "rescued_by_pca": num_rescued_pca,
                "rescued_by_pca_pct": float(num_rescued_pca / num_early_c_late_w) if num_early_c_late_w > 0 else 0.0,
                "error_set_jaccard": float(jaccard),
                "fraction_l5_errors_solved_by_early": float(frac_l5_errors_solved_by_e),
                "fraction_early_errors_solved_by_l5": float(frac_e_errors_solved_by_l5),
            }

        # Classwise complementarity for L2 vs L5
        L2_preds = single_stage_preds_pub[model]["L2"]
        L2_L5_raw_preds = preds_for_csv_pub[f"{model}_L2+L5_pred"]
        L2_L5_pca_preds = preds_for_csv_pub[f"{model}_PCA_L2+L5_pred"]

        classwise_summary[model] = {}
        for c_idx, c_name in enumerate(CLASS_NAMES):
            c_mask = (y_pub == c_idx)
            n_c = int(np.sum(c_mask))
            if n_c == 0:
                continue

            acc_l5 = float(np.mean(L5_preds[c_mask] == c_idx))
            acc_l2 = float(np.mean(L2_preds[c_mask] == c_idx))
            acc_raw = float(np.mean(L2_L5_raw_preds[c_mask] == c_idx))
            acc_pca = float(np.mean(L2_L5_pca_preds[c_mask] == c_idx))

            c_early_c_late_w = (c_mask & (L2_preds == c_idx) & (L5_preds != c_idx))
            n_e_c_l_w = int(np.sum(c_early_c_late_w))
            n_rescued = int(np.sum(c_early_c_late_w & (L2_L5_raw_preds == c_idx)))

            classwise_summary[model][c_name] = {
                "class_index": c_idx,
                "class_name": c_name,
                "num_samples": n_c,
                "L5_probe_acc": acc_l5,
                "L2_probe_acc": acc_l2,
                "L2_L5_raw_acc": acc_raw,
                "L2_L5_pca_acc": acc_pca,
                "delta_raw_vs_l5": acc_raw - acc_l5,
                "delta_pca_vs_l5": acc_pca - acc_l5,
                "early_correct_late_wrong": n_e_c_l_w,
                "rescued_by_joint_raw": n_rescued,
                "rescue_rate": float(n_rescued / n_e_c_l_w) if n_e_c_l_w > 0 else 0.0,
            }

        # Confusion-pair analysis
        confusion_analysis[model] = {}
        for true_name, pred_name in CONFUSION_PAIRS:
            true_idx = CLASS_NAMES.index(true_name)
            pred_idx = CLASS_NAMES.index(pred_name)

            mask = (y_pub == true_idx)
            count_l5 = int(np.sum(mask & (L5_preds == pred_idx)))
            count_raw = int(np.sum(mask & (L2_L5_raw_preds == pred_idx)))
            count_pca = int(np.sum(mask & (L2_L5_pca_preds == pred_idx)))

            confusion_analysis[model][f"{true_name}->{pred_name}"] = {
                "true_class": true_name,
                "confused_as": pred_name,
                "L5_probe_confusions": count_l5,
                "L2_L5_raw_confusions": count_raw,
                "L2_L5_pca_confusions": count_pca,
                "delta_raw_vs_l5": count_raw - count_l5,
                "delta_pca_vs_l5": count_pca - count_l5,
            }

    with open(A7_DIR / "a7_probe_complementarity_sets.json", "w", encoding="utf-8") as f:
        json.dump(complementarity_sets, f, indent=2)
    with open(A7_DIR / "a7_classwise_complementarity.json", "w", encoding="utf-8") as f:
        json.dump(classwise_summary, f, indent=2)
    with open(A7_DIR / "a7_confusion_pair_analysis.json", "w", encoding="utf-8") as f:
        json.dump(confusion_analysis, f, indent=2)
    with open(A7_DIR / "a7_representation_cka.json", "w", encoding="utf-8") as f:
        json.dump(cka_results, f, indent=2)

    # -------------------------------------------------------------
    # 4. MODEL-RESOLVABLE SENSITIVITY (v2.2 vs v2.3 raw predictions)
    # -------------------------------------------------------------
    v23_raw_pub = np.argmax(runner.v23_cache["public_R10"], axis=-1)
    v22_raw_pub = np.argmax(runner.a6_pub["v22_R10"], axis=-1)
    y_pub = runner.y_public

    v23_c = (v23_raw_pub == y_pub)
    v22_c = (v22_raw_pub == y_pub)

    v23_c_v22_w = (v23_c & ~v22_c)
    v22_c_v23_w = (~v23_c & v22_c)
    both_c = (v23_c & v22_c)
    both_w = (~v23_c & ~v22_c)

    v23_l2_l5_raw = preds_for_csv_pub["v23_L2+L5_pred"]
    v23_l5 = single_stage_preds_pub["v23"]["L5"]

    resolvable_sensitivity = {
        "group_counts": {
            "v23_correct_v22_wrong": int(np.sum(v23_c_v22_w)),
            "v22_correct_v23_wrong": int(np.sum(v22_c_v23_w)),
            "both_correct": int(np.sum(both_c)),
            "both_wrong": int(np.sum(both_w)),
        },
        "v23_l2_l5_probe_acc_by_group": {
            "v23_correct_v22_wrong": {
                "L5_acc": float(np.mean(v23_l5[v23_c_v22_w] == y_pub[v23_c_v22_w])),
                "L2_L5_raw_acc": float(np.mean(v23_l2_l5_raw[v23_c_v22_w] == y_pub[v23_c_v22_w])),
            },
            "v22_correct_v23_wrong": {
                "L5_acc": float(np.mean(v23_l5[v22_c_v23_w] == y_pub[v22_c_v23_w])),
                "L2_L5_raw_acc": float(np.mean(v23_l2_l5_raw[v22_c_v23_w] == y_pub[v22_c_v23_w])),
            },
        },
    }

    # -------------------------------------------------------------
    # 5. EVALUATE PREREGISTERED DECISION RULES ON PUBLIC
    # -------------------------------------------------------------
    print("\n--- PHASE 2.5: PREREGISTERED DECISION RULES EVALUATION (PUBLIC ONLY) ---", flush=True)

    actionable_pairs_v23 = {}
    actionable_pairs_v22 = {}

    for model, act_dict in [("v23", actionable_pairs_v23), ("v22", actionable_pairs_v22)]:
        for p_name in ["L1+L5", "L2+L5", "PRE+L5", "L1+L2+L5"]:
            raw_info = raw_concat_results[model][p_name]
            pca_info = pca_results[model][p_name]
            dup_ctrl_name = "L5+L5" if p_name != "L1+L2+L5" else "L5+L5+L5"
            dup_info = duplicate_results[model][dup_ctrl_name]

            delta_acc_raw = raw_info["delta_acc_vs_late"]
            ci_low = raw_info["bootstrap_ci"]["delta_acc_ci_95"][0]
            delta_acc_dup = dup_info["delta_acc_vs_late"]
            delta_acc_pca = pca_info["delta_acc_vs_late"]

            rule_a = bool(delta_acc_raw >= 0.0075)
            rule_b = bool(ci_low > 0.0)
            rule_c = bool(delta_acc_raw > delta_acc_dup)
            rule_d = bool(delta_acc_pca >= 0.0050 or (pca_info["delta_f1_vs_late"] > 0.005 and delta_acc_pca >= -0.001))

            passed_all = bool(rule_a and rule_b and rule_c and rule_d)

            act_dict[p_name] = {
                "pair": p_name,
                "delta_acc_raw_pp": delta_acc_raw * 100,
                "rule_a_ge_075pp": rule_a,
                "rule_b_ci_excludes_zero": rule_b,
                "rule_c_exceeds_duplicate": rule_c,
                "rule_d_pca_retains_ge_050pp": rule_d,
                "actionable": passed_all,
            }
            print(f"  {model} {p_name:10s} Actionable Complementarity: {passed_all} (A={rule_a}, B={rule_b}, C={rule_c}, D={rule_d})", flush=True)

    # Readout Information-Loss Rule (Section 28)
    readout_raw = raw_concat_results["v23"]["L5+Motif Readout"]
    readout_pca = pca_results["v23"]["L5+Motif Readout"]
    readout_dup = duplicate_results["v23"]["Motif Readout+Motif Readout"]
    readout_loss_supported = bool(
        readout_raw["delta_acc_vs_late"] >= 0.0050
        and readout_raw["delta_acc_vs_late"] > readout_dup["delta_acc_vs_late"]
        and readout_pca["delta_acc_vs_late"] >= 0.0030
    )
    print(f"  Readout Information Loss Supported: {readout_loss_supported} (Delta raw: {readout_raw['delta_acc_vs_late']*100:+.4f}pp, PCA: {readout_pca['delta_acc_vs_late']*100:+.4f}pp)", flush=True)

    # Fusion Information-Loss Rule (Section 29)
    fusion_raw_l2 = raw_concat_results["v23"]["L2+Fusion"]
    fusion_pca_l2 = pca_results["v23"]["L2+Fusion"]
    fusion_dup = duplicate_results["v23"]["Fusion+Fusion"]
    fusion_loss_supported = bool(
        fusion_raw_l2["delta_acc_vs_late"] >= 0.0050
        and fusion_raw_l2["delta_acc_vs_late"] > fusion_dup["delta_acc_vs_late"]
        and fusion_pca_l2["delta_acc_vs_late"] >= 0.0030
    )
    print(f"  Fusion Information Loss Supported: {fusion_loss_supported} (Delta raw: {fusion_raw_l2['delta_acc_vs_late']*100:+.4f}pp, PCA: {fusion_pca_l2['delta_acc_vs_late']*100:+.4f}pp)", flush=True)

    v23_has_actionable = any(v["actionable"] for v in actionable_pairs_v23.values())
    v22_has_actionable = any(v["actionable"] for v in actionable_pairs_v22.values())

    if v23_has_actionable:
        h_cross_depth = "SUPPORTED"
    elif any(pca_results["v23"][p]["delta_acc_vs_late"] > 0 for p in ["L1+L5", "L2+L5", "PRE+L5"]):
        h_cross_depth = "MIXED"
    else:
        h_cross_depth = "NOT_SUPPORTED"

    res_gains_v23 = [residual_results["v23"][k]["delta_acc_vs_l5"] for k in residual_results["v23"]]
    if any(g >= 0.005 for g in res_gains_v23):
        h_early_unique = "SUPPORTED"
    elif any(g > 0 for g in res_gains_v23):
        h_early_unique = "MIXED"
    else:
        h_early_unique = "NOT_SUPPORTED"

    h_readout = "SUPPORTED" if readout_loss_supported else "NOT_SUPPORTED"
    h_fusion = "SUPPORTED" if fusion_loss_supported else "NOT_SUPPORTED"

    if v23_has_actionable and not v22_has_actionable:
        h_v23_specific = "SUPPORTED"
    elif not v23_has_actionable and not v22_has_actionable:
        h_v23_specific = "NOT_SUPPORTED"
    else:
        h_v23_specific = "MIXED"

    hypothesis_statuses = {
        "H-A7-CROSS-DEPTH-COMPLEMENTARITY": h_cross_depth,
        "H-A7-EARLY-UNIQUE-INFORMATION": h_early_unique,
        "H-A7-READOUT-INFORMATION-LOSS": h_readout,
        "H-A7-FUSION-INFORMATION-LOSS": h_fusion,
        "H-A7-V23-SPECIFIC-COMPLEMENTARITY": h_v23_specific,
    }

    if v23_has_actionable:
        locked_target = "CROSS_DEPTH_INFORMATION_PRESERVATION_TARGET"
    elif readout_loss_supported:
        locked_target = "READOUT_INFORMATION_PRESERVATION_TARGET"
    elif fusion_loss_supported:
        locked_target = "FUSION_INFORMATION_PRESERVATION_TARGET"
    else:
        locked_target = "NO_V24_ARCHITECTURAL_TARGET"

    print(f"\n============================================================", flush=True)
    print(f"LOCKED PUBLIC NEXT-TARGET DECISION: {locked_target}", flush=True)
    print(f"HYPOTHESIS STATUSES: {hypothesis_statuses}", flush=True)
    print(f"============================================================", flush=True)

    # -------------------------------------------------------------
    # 6. WRITE PUBLIC DECISION LOCK (MANDATORY FIREWALL BOUNDARY)
    # -------------------------------------------------------------
    public_lock_doc = {
        "audit_phase": "A7_PUBLIC_DECISION_LOCK",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": get_git_head(),
        "script_sha256": sha256_file(Path(__file__)),
        "v22_checkpoint_sha256": "a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4",
        "v23_checkpoint_sha256": "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e",
        "PRIVATE_USED_FOR_DECISION": False,
        "locked_public_target": locked_target,
        "hypothesis_statuses": hypothesis_statuses,
        "actionable_rules_v23": actionable_pairs_v23,
        "actionable_rules_v22": actionable_pairs_v22,
        "single_stage_metrics_public": {
            "v23": {s: {"acc": m["public_acc"], "f1": m["public_macro_f1"]} for s, m in single_stage_metrics["v23"].items()},
            "v22": {s: {"acc": m["public_acc"], "f1": m["public_macro_f1"]} for s, m in single_stage_metrics["v22"].items()},
        },
        "raw_concat_public": raw_concat_results,
        "duplicate_controls_public": duplicate_results,
        "pca_controls_public": pca_results,
        "residual_controls_public": residual_results,
        "classwise_summary_public": classwise_summary,
        "confusion_analysis_public": confusion_analysis,
        "cka_public": cka_results,
        "resolvable_sensitivity_public": resolvable_sensitivity,
        "v24_go_no_go_public": bool(locked_target != "NO_V24_ARCHITECTURAL_TARGET"),
    }

    lock_path = A7_DIR / "a7_public_decision_lock.json"
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(public_lock_doc, f, indent=2)
    print(f"\n*** Successfully saved and locked: {lock_path} ***", flush=True)
    print("PRIVATE FIREWALL PRESERVED: PrivateTest will now be evaluated strictly as confirmatory evidence.", flush=True)

    # -------------------------------------------------------------
    # 7. PHASE 3: POST-FREEZE CONFIRMATORY PRIVATE EVALUATION
    # -------------------------------------------------------------
    print("\n============================================================", flush=True)
    print("--- PHASE 3: CONFIRMATORY PRIVATE EVALUATION ---", flush=True)
    print("============================================================", flush=True)

    private_results = {"v22": {}, "v23": {}}
    preds_for_csv_priv = {}

    for model in ["v23", "v22"]:
        print(f"\nEvaluating Private on Model {model}...", flush=True)
        private_results[model]["single_stage"] = {
            s: {"acc": single_stage_metrics[model][s]["private_acc"], "f1": single_stage_metrics[model][s]["private_macro_f1"]}
            for s in stages_order
        }

        # Duplicate controls on Private
        private_results[model]["duplicate_controls"] = {}
        for dup_name, d_stages, b_stage in dup_specs:
            X_priv_dup = np.hstack([runner.get_stage_feat(model, s, "private") for s in d_stages])
            coef, intercept, mean, scale, _, _ = runner.get_or_fit_probe(f"{model}_dup_{dup_name}", X_priv_dup)
            acc_priv, f1_priv, _ = runner.eval_probe(coef, intercept, mean, scale, X_priv_dup, runner.y_private)

            base_priv_acc = single_stage_metrics[model][b_stage]["private_acc"]
            base_priv_f1 = single_stage_metrics[model][b_stage]["private_macro_f1"]

            private_results[model]["duplicate_controls"][dup_name] = {
                "name": dup_name,
                "dim": int(X_priv_dup.shape[1]),
                "baseline_stage": b_stage,
                "private_acc": acc_priv,
                "private_macro_f1": f1_priv,
                "delta_acc_vs_late": acc_priv - base_priv_acc,
                "delta_f1_vs_late": f1_priv - base_priv_f1,
            }

        # Raw Concat and PCA on Private
        private_results[model]["raw_concat"] = {}
        private_results[model]["pca_controls"] = {}

        for pair in primary_pairs_config:
            p_name = pair["name"]
            b_stage = pair["baseline_stage"]
            pca_dim = pair["pca_target_dim"]
            base_priv_acc = single_stage_metrics[model][b_stage]["private_acc"]
            base_priv_f1 = single_stage_metrics[model][b_stage]["private_macro_f1"]
            pred_base_priv = single_stage_preds_priv[model][b_stage]

            # Raw eval
            X_priv_raw = np.hstack([runner.get_stage_feat(model, s, "private") for s in pair["stages"]])
            coef_raw, intercept_raw, mean_raw, scale_raw, _, _ = runner.get_or_fit_probe(f"{model}_raw_{p_name}", X_priv_raw)
            acc_raw, f1_raw, pred_cand_priv = runner.eval_probe(coef_raw, intercept_raw, mean_raw, scale_raw, X_priv_raw, runner.y_private)
            preds_for_csv_priv[f"{model}_{p_name}_pred"] = pred_cand_priv

            boot_raw = paired_bootstrap_ci(runner.y_private, pred_base_priv, pred_cand_priv, b_reps=2000, seed=42)
            mcnemar_raw = mcnemar_exact_test(runner.y_private, pred_base_priv, pred_cand_priv)

            private_results[model]["raw_concat"][p_name] = {
                "name": p_name,
                "dim": int(X_priv_raw.shape[1]),
                "baseline_stage": b_stage,
                "private_acc": acc_raw,
                "private_macro_f1": f1_raw,
                "delta_acc_vs_late": acc_raw - base_priv_acc,
                "delta_f1_vs_late": f1_raw - base_priv_f1,
                "bootstrap_ci": boot_raw,
                "mcnemar": mcnemar_raw,
            }

            # PCA eval
            components, pca_mean, prep_mean, prep_scale, exp_var = runner.get_or_fit_pca(f"{model}_{p_name}_{pca_dim}", X_priv_raw, pca_dim)
            X_priv_pca = runner.transform_pca(components, pca_mean, prep_mean, prep_scale, X_priv_raw)

            coef_pca, intercept_pca, mean_pca, scale_pca, _, _ = runner.get_or_fit_probe(f"{model}_pca_{p_name}_{pca_dim}", X_priv_pca)
            acc_pca, f1_pca, pred_pca_priv = runner.eval_probe(coef_pca, intercept_pca, mean_pca, scale_pca, X_priv_pca, runner.y_private)
            preds_for_csv_priv[f"{model}_PCA_{p_name}_pred"] = pred_pca_priv

            boot_pca = paired_bootstrap_ci(runner.y_private, pred_base_priv, pred_pca_priv, b_reps=2000, seed=42)
            mcnemar_pca = mcnemar_exact_test(runner.y_private, pred_base_priv, pred_pca_priv)

            private_results[model]["pca_controls"][p_name] = {
                "name": f"PCA({p_name})->{pca_dim}",
                "reduced_dim": pca_dim,
                "baseline_stage": b_stage,
                "private_acc": acc_pca,
                "private_macro_f1": f1_pca,
                "delta_acc_vs_late": acc_pca - base_priv_acc,
                "delta_f1_vs_late": f1_pca - base_priv_f1,
                "bootstrap_ci": boot_pca,
                "mcnemar": mcnemar_pca,
            }

            print(
                f"    {p_name:18s} | Raw: Priv Acc {acc_raw*100:.4f}% ({(acc_raw - base_priv_acc)*100:+.4f}pp, CI [{boot_raw['delta_acc_ci_95'][0]*100:+.2f}, {boot_raw['delta_acc_ci_95'][1]*100:+.2f}], p={mcnemar_raw['p_value']:.4f}) | "
                f"PCA->{pca_dim:3d}: Priv Acc {acc_pca*100:.4f}% ({(acc_pca - base_priv_acc)*100:+.4f}pp, CI [{boot_pca['delta_acc_ci_95'][0]*100:+.2f}, {boot_pca['delta_acc_ci_95'][1]*100:+.2f}], p={mcnemar_pca['p_value']:.4f})",
                flush=True,
            )

        # Residualization on Private
        private_results[model]["residual_controls"] = {}
        L5_priv = runner.get_stage_feat(model, "L5", "private")
        for e_stage in ["PRE", "L1", "L2"]:
            E_priv = runner.get_stage_feat(model, e_stage, "private")
            r_coef, r_intercept = runner.get_or_fit_ridge(f"{model}_L5_to_{e_stage}", L5_priv, E_priv)

            E_res_priv = E_priv - runner.predict_ridge(r_coef, r_intercept, L5_priv)
            X_priv_res = np.hstack([L5_priv, E_res_priv])

            c_res, i_res, m_res, s_res, _, _ = runner.get_or_fit_probe(f"{model}_L5+{e_stage}_res", X_priv_res)
            acc_res, f1_res, _ = runner.eval_probe(c_res, i_res, m_res, s_res, X_priv_res, runner.y_private)

            base_priv = single_stage_metrics[model]["L5"]["private_acc"]

            private_results[model]["residual_controls"][f"L5+{e_stage}_res"] = {
                "early_stage": e_stage,
                "private_acc": acc_res,
                "private_macro_f1": f1_res,
                "delta_acc_vs_l5": acc_res - base_priv,
            }

    # Assess Confirmation Status
    if locked_target == "NO_V24_ARCHITECTURAL_TARGET":
        confirmatory_status = "PUBLIC_NEGATIVE_PRIVATE_OBSERVATIONAL_ONLY"
    else:
        v23_priv_l2_l5_pca = private_results["v23"]["pca_controls"]["L2+L5"]["delta_acc_vs_late"]
        if v23_priv_l2_l5_pca >= 0.0050:
            confirmatory_status = "PUBLIC_PRIVATE_CONFIRMED"
        elif v23_priv_l2_l5_pca > 0.0:
            confirmatory_status = "PUBLIC_SUPPORTED_PRIVATE_PARTIAL"
        else:
            confirmatory_status = "PUBLIC_SUPPORTED_PRIVATE_NOT_CONFIRMED"

    if locked_target == "CROSS_DEPTH_INFORMATION_PRESERVATION_TARGET" and confirmatory_status in ["PUBLIC_PRIVATE_CONFIRMED", "PUBLIC_SUPPORTED_PRIVATE_PARTIAL"]:
        final_scientific_decision = "CROSS_DEPTH_COMPLEMENTARITY_SUPPORTED"
    elif locked_target == "READOUT_INFORMATION_PRESERVATION_TARGET":
        final_scientific_decision = "READOUT_INFORMATION_LOSS_SUPPORTED"
    elif locked_target == "FUSION_INFORMATION_PRESERVATION_TARGET":
        final_scientific_decision = "FUSION_INFORMATION_LOSS_SUPPORTED"
    elif locked_target == "NO_V24_ARCHITECTURAL_TARGET":
        final_scientific_decision = "NO_ACTIONABLE_COMPLEMENTARITY"
    else:
        final_scientific_decision = "MIXED_INFORMATION_PRESERVATION_EVIDENCE"

    if (
        locked_target != "NO_V24_ARCHITECTURAL_TARGET"
        and v23_has_actionable
        and confirmatory_status in ["PUBLIC_PRIVATE_CONFIRMED", "PUBLIC_SUPPORTED_PRIVATE_PARTIAL"]
    ):
        v24_gate = "V24_JUSTIFIED"
        operational_verdict = "A7_COMPLETE_V24_GATE_READY"
    else:
        v24_gate = "V24_NOT_JUSTIFIED"
        operational_verdict = "A7_COMPLETE_NO_V24_TARGET"

    print("\n============================================================", flush=True)
    print(f"CONFIRMATORY STATUS: {confirmatory_status}", flush=True)
    print(f"FINAL SCIENTIFIC DECISION: {final_scientific_decision}", flush=True)
    print(f"V2.4 GATE: {v24_gate}", flush=True)
    print(f"FINAL OPERATIONAL VERDICT: {operational_verdict}", flush=True)
    print("============================================================", flush=True)

    # -------------------------------------------------------------
    # 8. SAVE ALL REQUIRED ARTIFACTS
    # -------------------------------------------------------------
    private_confirmation_doc = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": get_git_head(),
        "public_decision_locked_before_private": True,
        "locked_public_target": locked_target,
        "confirmatory_status": confirmatory_status,
        "final_scientific_decision": final_scientific_decision,
        "v24_gate": v24_gate,
        "operational_verdict": operational_verdict,
        "private_results": private_results,
    }

    with open(A7_DIR / "a7_private_confirmation.json", "w", encoding="utf-8") as f:
        json.dump(private_confirmation_doc, f, indent=2)

    hypothesis_decisions_doc = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hypotheses": hypothesis_statuses,
        "confirmatory_status": confirmatory_status,
        "final_scientific_decision": final_scientific_decision,
        "v24_gate": v24_gate,
        "operational_verdict": operational_verdict,
    }
    with open(A7_DIR / "a7_hypothesis_decisions.json", "w", encoding="utf-8") as f:
        json.dump(hypothesis_decisions_doc, f, indent=2)

    final_target_doc = {
        "locked_public_target": locked_target,
        "confirmatory_status": confirmatory_status,
        "final_scientific_decision": final_scientific_decision,
        "v24_gate": v24_gate,
        "operational_verdict": operational_verdict,
    }
    with open(A7_DIR / "a7_final_target.json", "w", encoding="utf-8") as f:
        json.dump(final_target_doc, f, indent=2)

    public_stats_doc = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "v23": {
            p_name: {
                "raw": {
                    "delta_acc": raw_concat_results["v23"][p_name]["delta_acc_vs_late"],
                    "delta_f1": raw_concat_results["v23"][p_name]["delta_f1_vs_late"],
                    "bootstrap_95_ci": raw_concat_results["v23"][p_name]["bootstrap_ci"]["delta_acc_ci_95"],
                    "mcnemar_p": raw_concat_results["v23"][p_name]["mcnemar"]["p_value"],
                },
                "pca": {
                    "delta_acc": pca_results["v23"][p_name]["delta_acc_vs_late"],
                    "delta_f1": pca_results["v23"][p_name]["delta_f1_vs_late"],
                    "bootstrap_95_ci": pca_results["v23"][p_name]["bootstrap_ci"]["delta_acc_ci_95"],
                    "mcnemar_p": pca_results["v23"][p_name]["mcnemar"]["p_value"],
                },
            }
            for p_name in raw_concat_results["v23"]
        },
        "v22": {
            p_name: {
                "raw": {
                    "delta_acc": raw_concat_results["v22"][p_name]["delta_acc_vs_late"],
                    "delta_f1": raw_concat_results["v22"][p_name]["delta_f1_vs_late"],
                    "bootstrap_95_ci": raw_concat_results["v22"][p_name]["bootstrap_ci"]["delta_acc_ci_95"],
                    "mcnemar_p": raw_concat_results["v22"][p_name]["mcnemar"]["p_value"],
                },
                "pca": {
                    "delta_acc": pca_results["v22"][p_name]["delta_acc_vs_late"],
                    "delta_f1": pca_results["v22"][p_name]["delta_f1_vs_late"],
                    "bootstrap_95_ci": pca_results["v22"][p_name]["bootstrap_ci"]["delta_acc_ci_95"],
                    "mcnemar_p": pca_results["v22"][p_name]["mcnemar"]["p_value"],
                },
            }
            for p_name in raw_concat_results["v22"]
        },
    }
    with open(A7_DIR / "a7_public_statistics.json", "w", encoding="utf-8") as f:
        json.dump(public_stats_doc, f, indent=2)

    master_results = {
        "provenance": json.load(open(A7_DIR / "a7_provenance.json")),
        "hook_validation": json.load(open(A7_DIR / "a7_hook_validation.json")),
        "single_stage": single_stage_metrics,
        "raw_concat": raw_concat_results,
        "duplicate_controls": duplicate_results,
        "pca_controls": pca_results,
        "residual_controls": residual_results,
        "probe_complementarity_sets": complementarity_sets,
        "classwise_summary": classwise_summary,
        "confusion_analysis": confusion_analysis,
        "cka": cka_results,
        "public_statistics": public_stats_doc,
        "resolvable_sensitivity": resolvable_sensitivity,
        "public_decision_lock": public_lock_doc,
        "private_confirmation": private_confirmation_doc,
        "hypothesis_decisions": hypothesis_decisions_doc,
        "final_target": final_target_doc,
    }
    with open(A7_DIR / "a7_master_results.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print("\nWriting sample-level CSVs...", flush=True)
    df_pub = pd.DataFrame({
        "row_index": np.arange(len(runner.y_public)),
        "true_label": runner.y_public,
        "true_class_name": [CLASS_NAMES[y] for y in runner.y_public],
        "v22_model_raw_pred": np.argmax(runner.a6_pub["v22_R10"], axis=-1),
        "v23_model_raw_pred": np.argmax(runner.v23_cache["public_R10"], axis=-1),
        "v22_L5_probe_pred": single_stage_preds_pub["v22"]["L5"],
        "v23_L5_probe_pred": single_stage_preds_pub["v23"]["L5"],
        "v22_L2_probe_pred": single_stage_preds_pub["v22"]["L2"],
        "v23_L2_probe_pred": single_stage_preds_pub["v23"]["L2"],
        "v22_L2_L5_raw_pred": preds_for_csv_pub["v22_L2+L5_pred"],
        "v23_L2_L5_raw_pred": preds_for_csv_pub["v23_L2+L5_pred"],
        "v22_PCA_L2_L5_pred": preds_for_csv_pub["v22_PCA_L2+L5_pred"],
        "v23_PCA_L2_L5_pred": preds_for_csv_pub["v23_PCA_L2+L5_pred"],
    })
    df_pub.to_csv(A7_DIR / "a7_public_predictions.csv", index=False)

    df_priv = pd.DataFrame({
        "row_index": np.arange(len(runner.y_private)),
        "true_label": runner.y_private,
        "true_class_name": [CLASS_NAMES[y] for y in runner.y_private],
        "v22_model_raw_pred": np.argmax(runner.a6_priv["v22_R10"], axis=-1),
        "v23_model_raw_pred": np.argmax(runner.v23_cache["private_R10"], axis=-1),
        "v22_L5_probe_pred": single_stage_preds_priv["v22"]["L5"],
        "v23_L5_probe_pred": single_stage_preds_priv["v23"]["L5"],
        "v22_L2_probe_pred": single_stage_preds_priv["v22"]["L2"],
        "v23_L2_probe_pred": single_stage_preds_priv["v23"]["L2"],
        "v22_L2_L5_raw_pred": preds_for_csv_priv["v22_L2+L5_pred"],
        "v23_L2_L5_raw_pred": preds_for_csv_priv["v23_L2+L5_pred"],
        "v22_PCA_L2_L5_pred": preds_for_csv_priv["v22_PCA_L2+L5_pred"],
        "v23_PCA_L2_L5_pred": preds_for_csv_priv["v23_PCA_L2+L5_pred"],
    })
    df_priv.to_csv(A7_DIR / "a7_private_predictions.csv", index=False)

    l2_p = single_stage_preds_pub["v23"]["L2"]
    l5_p = single_stage_preds_pub["v23"]["L5"]
    joint_p = preds_for_csv_pub["v23_L2+L5_pred"]
    pca_p = preds_for_csv_pub["v23_PCA_L2+L5_pred"]

    categories = []
    rescued_raw_list = []
    rescued_pca_list = []

    for idx in range(len(runner.y_public)):
        y = runner.y_public[idx]
        e_c = (l2_p[idx] == y)
        l_c = (l5_p[idx] == y)
        if e_c and not l_c:
            cat = "EARLY_CORRECT_LATE_WRONG"
        elif not e_c and l_c:
            cat = "EARLY_WRONG_LATE_CORRECT"
        elif e_c and l_c:
            cat = "BOTH_CORRECT"
        else:
            cat = "BOTH_WRONG"
        categories.append(cat)
        rescued_raw_list.append(bool(cat == "EARLY_CORRECT_LATE_WRONG" and joint_p[idx] == y))
        rescued_pca_list.append(bool(cat == "EARLY_CORRECT_LATE_WRONG" and pca_p[idx] == y))

    df_comp = pd.DataFrame({
        "row_index": np.arange(len(runner.y_public)),
        "true_label": runner.y_public,
        "true_class_name": [CLASS_NAMES[y] for y in runner.y_public],
        "v23_L2_pred": l2_p,
        "v23_L5_pred": l5_p,
        "v23_L2_L5_raw_pred": joint_p,
        "v23_PCA_L2_L5_pred": pca_p,
        "complementarity_category": categories,
        "rescued_by_joint_raw": rescued_raw_list,
        "rescued_by_joint_pca": rescued_pca_list,
    })
    df_comp.to_csv(A7_DIR / "a7_probe_complementarity_samples.csv", index=False)

    print(f"\nAll A7 analysis completed in {time.time() - t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
