"""Zero-fit registered M0 aggregation and independent recomputation primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .e02_runner import PUBLIC_ROWS, sha256_array, sha256_file
from .m0_config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID,
    G_CONFIG_SHA256,
    ISSUE_NUMBER,
    L_CONFIG_SHA256,
    M_CONFIG_SHA256,
    PUBLIC_IDS_SHA256,
    SEEDS,
    TRAIN_IDS_SHA256,
)


VERDICT_SUPPORTED = "M0 SUPPORTED — GRAPH-STRUCTURED PROCESSING ADDS REGISTERED INCREMENTAL VALUE"
VERDICT_MIXED = "M0 MIXED — NOT SUPPORTED BY REGISTERED JOINT CRITERION"
VERDICT_NOT_SUPPORTED = "M0 NOT SUPPORTED"


def _metric(labels: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return float(accuracy_score(labels, prediction)), float(f1_score(labels, prediction, average="macro", labels=np.arange(7), zero_division=0))


def registered_verdict(arrays: dict[str, np.ndarray]) -> str:
    passed = [float(np.quantile(arrays[key], 0.025)) > 0 for key in ("g_l_accuracy", "g_l_macro_f1", "g_m_accuracy", "g_m_macro_f1")]
    if all(passed):
        return VERDICT_SUPPORTED
    if any(passed):
        return VERDICT_MIXED
    return VERDICT_NOT_SUPPORTED


def registered_bootstrap(labels: np.ndarray, l_prediction: np.ndarray, m_predictions: np.ndarray, g_predictions: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    if labels.shape != (PUBLIC_ROWS,) or l_prediction.shape != (PUBLIC_ROWS,) or m_predictions.shape != (5, PUBLIC_ROWS) or g_predictions.shape != (5, PUBLIC_ROWS):
        raise ValueError("registered prediction shape mismatch")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = {key: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for key in ("g_l_accuracy", "g_l_macro_f1", "g_m_accuracy", "g_m_macro_f1")}
    for b in range(BOOTSTRAP_REPLICATES):
        take = rng.integers(0, PUBLIC_ROWS, size=PUBLIC_ROWS)
        y = labels[take]
        l_acc, l_f1 = _metric(y, l_prediction[take])
        m_metric = np.asarray([_metric(y, prediction[take]) for prediction in m_predictions])
        g_metric = np.asarray([_metric(y, prediction[take]) for prediction in g_predictions])
        result["g_l_accuracy"][b] = g_metric[:, 0].mean() - l_acc
        result["g_l_macro_f1"][b] = g_metric[:, 1].mean() - l_f1
        result["g_m_accuracy"][b] = g_metric[:, 0].mean() - m_metric[:, 0].mean()
        result["g_m_macro_f1"][b] = g_metric[:, 1].mean() - m_metric[:, 1].mean()
    return result


def _text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("expected scalar text")
    return str(array.item())


def _load_prediction(path: Path, *, family: str, seed: int | None, scientific_sha: str, wrapper_sha: str, substrate_sha: str) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as artifact:
        expected_config = {"L": L_CONFIG_SHA256, "M": M_CONFIG_SHA256, "G": G_CONFIG_SHA256}[family]
        if _text(artifact["experiment"]) != EXPERIMENT_ID or int(artifact["issue"]) != ISSUE_NUMBER:
            raise ValueError("prediction provenance mismatch")
        if _text(artifact["scientific_sha"]) != scientific_sha or _text(artifact["wrapper_sha"]) != wrapper_sha or _text(artifact["substrate_sha256"]) != substrate_sha:
            raise ValueError("prediction source/substrate mismatch")
        if _text(artifact["model_family"]) != family or int(artifact["seed"]) != (-1 if seed is None else seed):
            raise ValueError("prediction family/seed mismatch")
        if _text(artifact["model_config_sha256"]) != expected_config or _text(artifact["public_ids_sha256"]) != PUBLIC_IDS_SHA256 or _text(artifact["train_ids_sha256"]) != TRAIN_IDS_SHA256:
            raise ValueError("prediction config/ID mismatch")
        if bool(artifact["public_metrics_present"]):
            raise ValueError("per-fit artifact contains Public metric")
        if family != "L" and int(artifact["final_epoch"]) != 50:
            raise ValueError("seed fit did not complete epoch 50")
        prediction = np.asarray(artifact["public_predictions"])
        if prediction.dtype != np.int8 or prediction.shape != (PUBLIC_ROWS,) or np.any((prediction < 0) | (prediction > 6)):
            raise ValueError("invalid Public prediction")
        record = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "family": family, "seed": seed}
        if family == "L":
            record.update({"n_iter": np.asarray(artifact["n_iter"]).astype(int).tolist(), "converged": bool(artifact["converged"]), "warnings": json.loads(_text(artifact["warnings_json"]))})
        else:
            record.update({"epochs": int(artifact["final_epoch"]), "parameter_count": int(artifact["parameter_count"]), "loss": np.asarray(artifact["training_loss"]).tolist()})
        return prediction.copy(), record


def aggregate(
    public_labels: np.ndarray,
    public_ids: np.ndarray,
    l_path: str | Path,
    m_paths: Iterable[str | Path],
    g_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    scientific_sha: str,
    wrapper_sha: str,
    substrate_sha: str,
) -> dict:
    labels = np.asarray(public_labels, dtype=np.int8)
    ids = np.asarray(public_ids, dtype=np.int32)
    if labels.shape != (PUBLIC_ROWS,) or np.any((labels < 0) | (labels > 6)) or sha256_array(ids) != PUBLIC_IDS_SHA256:
        raise ValueError("canonical Public labels/IDs mismatch")
    m_map, g_map = {}, {}
    for family, paths, target in (("M", m_paths, m_map), ("G", g_paths, g_map)):
        for raw in paths:
            path = Path(raw)
            with np.load(path, allow_pickle=False) as artifact:
                seed = int(artifact["seed"])
            if seed in target:
                raise ValueError(f"duplicate {family} seed")
            target[seed] = path
    if set(m_map) != set(SEEDS) or set(g_map) != set(SEEDS):
        raise ValueError("M/G seeds must be exactly 42..46")
    l_prediction, l_record = _load_prediction(Path(l_path), family="L", seed=None, scientific_sha=scientific_sha, wrapper_sha=wrapper_sha, substrate_sha=substrate_sha)
    m_loaded = [_load_prediction(m_map[s], family="M", seed=s, scientific_sha=scientific_sha, wrapper_sha=wrapper_sha, substrate_sha=substrate_sha) for s in SEEDS]
    g_loaded = [_load_prediction(g_map[s], family="G", seed=s, scientific_sha=scientific_sha, wrapper_sha=wrapper_sha, substrate_sha=substrate_sha) for s in SEEDS]
    m_predictions = np.stack([item[0] for item in m_loaded])
    g_predictions = np.stack([item[0] for item in g_loaded])
    l_metric = np.asarray(_metric(labels, l_prediction))
    m_metrics = np.asarray([_metric(labels, p) for p in m_predictions])
    g_metrics = np.asarray([_metric(labels, p) for p in g_predictions])
    bootstrap = registered_bootstrap(labels, l_prediction, m_predictions, g_predictions)
    verdict = registered_verdict(bootstrap)
    comparisons = {}
    for name, values in bootstrap.items():
        comparisons[name] = {
            "observed": float((g_metrics.mean(axis=0) - (l_metric if name.startswith("g_l") else m_metrics.mean(axis=0)))[0 if name.endswith("accuracy") else 1]),
            "q2_5_q50_q97_5": np.quantile(values, [0.025, 0.5, 0.975]).tolist(),
            "fraction_le_zero": float(np.mean(values <= 0)),
            "strict_pass": bool(np.quantile(values, 0.025) > 0),
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = output / "m0_results.npz"
    with results.open("wb") as stream:
        np.savez_compressed(
            stream, experiment=np.asarray(EXPERIMENT_ID), issue=np.asarray(ISSUE_NUMBER, np.int32),
            scientific_sha=np.asarray(scientific_sha), wrapper_sha=np.asarray(wrapper_sha), substrate_sha256=np.asarray(substrate_sha),
            public_ids=ids, public_labels=labels, seeds=np.asarray(SEEDS, np.int32),
            l_prediction=l_prediction, m_predictions=m_predictions, g_predictions=g_predictions,
            l_metrics=l_metric, m_metrics=m_metrics, g_metrics=g_metrics,
            bootstrap_g_l_accuracy=bootstrap["g_l_accuracy"], bootstrap_g_l_macro_f1=bootstrap["g_l_macro_f1"],
            bootstrap_g_m_accuracy=bootstrap["g_m_accuracy"], bootstrap_g_m_macro_f1=bootstrap["g_m_macro_f1"],
            registered_verdict=np.asarray(verdict),
        )
    summary = {
        "experiment": EXPERIMENT_ID, "issue": ISSUE_NUMBER, "scientific_sha": scientific_sha, "wrapper_sha": wrapper_sha,
        "substrate_sha256": substrate_sha, "public_status": "PublicTest developmental evidence only",
        "historical_e0r2_descriptive": {"accuracy": 0.42017275006965726, "macro_f1": 0.41720028899607875},
        "L": {"accuracy": l_metric[0], "macro_f1": l_metric[1], "artifact": l_record},
        "M": {"seed_metrics": [{"seed": s, "accuracy": row[0], "macro_f1": row[1]} for s, row in zip(SEEDS, m_metrics)], "mean": m_metrics.mean(axis=0).tolist(), "sample_sd": m_metrics.std(axis=0, ddof=1).tolist(), "artifacts": [item[1] for item in m_loaded]},
        "G": {"seed_metrics": [{"seed": s, "accuracy": row[0], "macro_f1": row[1]} for s, row in zip(SEEDS, g_metrics)], "mean": g_metrics.mean(axis=0).tolist(), "sample_sd": g_metrics.std(axis=0, ddof=1).tolist(), "artifacts": [item[1] for item in g_loaded]},
        "comparisons": comparisons, "registered_verdict": verdict, "private_test_read": False, "model_fits_executed": 0,
    }
    summary_path = output / "m0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    manifest = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in (results, summary_path)}
    manifest_path = output / "m0_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return summary


def independent_recompute(results_path: str | Path) -> dict:
    with np.load(results_path, allow_pickle=False) as artifact:
        labels = np.asarray(artifact["public_labels"])
        l_prediction = np.asarray(artifact["l_prediction"])
        m_predictions = np.asarray(artifact["m_predictions"])
        g_predictions = np.asarray(artifact["g_predictions"])
        stored_bootstrap = {
            "g_l_accuracy": np.asarray(artifact["bootstrap_g_l_accuracy"]),
            "g_l_macro_f1": np.asarray(artifact["bootstrap_g_l_macro_f1"]),
            "g_m_accuracy": np.asarray(artifact["bootstrap_g_m_accuracy"]),
            "g_m_macro_f1": np.asarray(artifact["bootstrap_g_m_macro_f1"]),
        }
        stored_l = np.asarray(artifact["l_metrics"])
        stored_m = np.asarray(artifact["m_metrics"])
        stored_g = np.asarray(artifact["g_metrics"])
        stored_verdict = _text(artifact["registered_verdict"])
    recomputed = registered_bootstrap(labels, l_prediction, m_predictions, g_predictions)
    if any(not np.array_equal(recomputed[key], stored_bootstrap[key]) for key in recomputed):
        raise AssertionError("bootstrap arrays differ")
    metrics_l = np.asarray(_metric(labels, l_prediction))
    metrics_m = np.asarray([_metric(labels, p) for p in m_predictions])
    metrics_g = np.asarray([_metric(labels, p) for p in g_predictions])
    maximum = max(float(np.max(np.abs(metrics_l - stored_l))), float(np.max(np.abs(metrics_m - stored_m))), float(np.max(np.abs(metrics_g - stored_g))))
    if registered_verdict(recomputed) != stored_verdict:
        raise AssertionError("registered verdict differs")
    return {"prediction_equality": True, "bootstrap_array_equality": True, "maximum_absolute_metric_difference": maximum, "registered_verdict": stored_verdict}
