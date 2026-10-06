"""Train and cache frozen layerwise linear probes on Train split (R0-R9).

Fixed protocol:
- StandardScaler fit Train only
- Multinomial LogisticRegression (C=1.0, L2, lbfgs, max_iter=5000, tol=1e-6)
- Evaluates on Train, Public, Private
- Saves per-probe checkpoint in research/mpg_fer_audit/a6/probe_cache/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a6"
CACHE_DIR = AUDIT_DIR / "probe_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAGES = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]


def train_single_probe(model_key: str, stage: str, train_data, pub_data, priv_data):
    ckpt_file = CACHE_DIR / f"probe_{model_key}_{stage}.npz"
    meta_file = CACHE_DIR / f"meta_{model_key}_{stage}.json"

    if ckpt_file.exists() and meta_file.exists():
        print(f"Probe {model_key} {stage} already cached; loading.", flush=True)
        return

    t0 = time.time()
    X_tr = train_data[f"{model_key}_{stage}"]
    y_tr = train_data["targets"]

    X_pub = pub_data[f"{model_key}_{stage}"]
    y_pub = pub_data["targets"]

    X_priv = priv_data[f"{model_key}_{stage}"]
    y_priv = priv_data["targets"]

    dim = X_tr.shape[1]
    print(f"\n--- Training Probe {model_key.upper()} Stage {stage} [{dim}d] ---", flush=True)

    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr)

    clf = LogisticRegression(
        C=1.0,
        max_iter=5000,
        tol=1e-6,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_tr_std, y_tr)
    fit_duration = time.time() - t0
    converged = bool(clf.n_iter_[0] < clf.max_iter)

    # Save probe parameters
    np.savez_compressed(
        ckpt_file,
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )

    # Evaluate on Train, Public, Private
    def eval_split(X, y):
        Xs = (X - scaler.mean_) / scaler.scale_
        logits = Xs @ clf.coef_.T + clf.intercept_
        preds = np.argmax(logits, axis=-1)
        acc = float(accuracy_score(y, preds))
        f1 = float(f1_score(y, preds, average="macro", zero_division=0))
        return acc, f1

    acc_tr, f1_tr = eval_split(X_tr, y_tr)
    acc_pub, f1_pub = eval_split(X_pub, y_pub)
    acc_priv, f1_priv = eval_split(X_priv, y_priv)

    meta = {
        "model": model_key,
        "stage": stage,
        "dimensions": dim,
        "fit_duration_sec": fit_duration,
        "iterations": int(clf.n_iter_[0]),
        "converged": converged,
        "train_accuracy": acc_tr,
        "train_macro_f1": f1_tr,
        "public_accuracy": acc_pub,
        "public_macro_f1": f1_pub,
        "private_accuracy": acc_priv,
        "private_macro_f1": f1_priv,
        "train_public_gap": float(acc_tr - acc_pub),
        "train_private_gap": float(acc_tr - acc_priv),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  {stage} fit in {fit_duration:.1f}s ({clf.n_iter_[0]} iter, conv: {converged}) | "
          f"Train: {acc_tr:.4f} | Public: {acc_pub:.4f} | Private: {acc_priv:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "v21", "v22"], default="all")
    parser.add_argument("--stage", default="all")
    args = parser.parse_args()

    t_all = time.time()
    print("Loading feature NPZs for probe training...", flush=True)
    train_data = np.load(AUDIT_DIR / "a6_train_features.npz")
    pub_data = np.load(AUDIT_DIR / "a6_public_features.npz")
    priv_data = np.load(AUDIT_DIR / "a6_private_features.npz")

    models = ["v21", "v22"] if args.model == "all" else [args.model]
    stages = STAGES if args.stage == "all" else [args.stage]

    for m in models:
        for s in stages:
            train_single_probe(m, s, train_data, pub_data, priv_data)

    print(f"\nAll requested probes completed in {time.time() - t_all:.1f}s.")


if __name__ == "__main__":
    main()
