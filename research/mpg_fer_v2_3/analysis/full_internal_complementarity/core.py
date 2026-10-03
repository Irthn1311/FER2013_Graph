"""Pure computations for the frozen MPG-FER internal-complementarity audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import torch


CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
HEAD_PAIRS = (("fused", "pixel"), ("fused", "motif"), ("pixel", "motif"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state_or_model: Any) -> str:
    state = (
        state_or_model.state_dict()
        if hasattr(state_or_model, "state_dict")
        else state_or_model
    )
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(labels: np.ndarray, logits: np.ndarray) -> float:
    probability = softmax(logits)
    return float(-np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-300, 1.0)).mean())


def metric_block(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    prediction = logits.argmax(axis=1)
    recalls = []
    for class_id in range(len(CLASS_NAMES)):
        mask = labels == class_id
        recalls.append(None if not mask.any() else float(np.mean(prediction[mask] == class_id)))
    return {
        "loss": cross_entropy(labels, logits),
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": float(
            f1_score(
                labels,
                prediction,
                average="macro",
                labels=np.arange(len(CLASS_NAMES)),
                zero_division=0,
            )
        ),
        "correct_count": int(np.sum(prediction == labels)),
        "per_class_recall": recalls,
    }


def head_diagnostics(raw: np.ndarray, flip: np.ndarray) -> dict[str, np.ndarray]:
    tta = 0.5 * (raw + flip)
    raw_probability = softmax(raw)
    flip_probability = softmax(flip)
    tta_probability = softmax(tta)
    ordered = np.sort(tta_probability, axis=1)
    mixture = 0.5 * (raw_probability + flip_probability)
    eps = np.finfo(np.float64).tiny
    js = 0.5 * (
        np.sum(raw_probability * (np.log(np.clip(raw_probability, eps, 1.0)) - np.log(np.clip(mixture, eps, 1.0))), axis=1)
        + np.sum(flip_probability * (np.log(np.clip(flip_probability, eps, 1.0)) - np.log(np.clip(mixture, eps, 1.0))), axis=1)
    )
    return {
        "tta_logits": tta.astype(np.float32),
        "raw_prediction": raw.argmax(axis=1).astype(np.int64),
        "tta_prediction": tta.argmax(axis=1).astype(np.int64),
        "tta_confidence": tta_probability.max(axis=1),
        "tta_margin": ordered[:, -1] - ordered[:, -2],
        "tta_entropy": -np.sum(tta_probability * np.log(np.clip(tta_probability, eps, 1.0)), axis=1),
        "flip_js_divergence": js,
    }


def disagreement(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    first_ok = first == labels
    second_ok = second == labels
    cc = first_ok & second_ok
    cw = first_ok & ~second_ok
    wc = ~first_ok & second_ok
    ww = ~first_ok & ~second_ok
    equal = first == second
    return {
        "N_CC": int(cc.sum()),
        "N_CW": int(cw.sum()),
        "N_WC": int(wc.sum()),
        "N_WW": int(ww.sum()),
        "prediction_agreement_count": int(equal.sum()),
        "prediction_disagreement_count": int((~equal).sum()),
        "prediction_disagreement_rate": float(np.mean(~equal)),
        "correctness_disagreement_count": int(cw.sum() + wc.sum()),
        "first_only_correct_indices": np.flatnonzero(cw).astype(int).tolist(),
        "second_only_correct_indices": np.flatnonzero(wc).astype(int).tolist(),
        "both_wrong_indices": np.flatnonzero(ww).astype(int).tolist(),
    }


def oracle(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    correct = (first == labels) | (second == labels)
    return {
        "correct_count": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "headroom_over_first_pp": float(100.0 * (correct.mean() - np.mean(first == labels))),
        "headroom_over_second_pp": float(100.0 * (correct.mean() - np.mean(second == labels))),
    }


def classwise_pair(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        mask = labels == class_id
        first_ok = (first == labels) & mask
        second_ok = (second == labels) & mask
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "support": int(mask.sum()),
                "first_correct": int(first_ok.sum()),
                "second_correct": int(second_ok.sum()),
                "both_correct": int((first_ok & second_ok).sum()),
                "first_only_correct": int((first_ok & ~second_ok).sum()),
                "second_only_correct": int((~first_ok & second_ok).sum()),
                "both_wrong": int((mask & ~first_ok & ~second_ok).sum()),
                "first_recall": float(first_ok.sum() / mask.sum()),
                "second_recall": float(second_ok.sum() / mask.sum()),
                "oracle_recall": float(((first_ok | second_ok).sum()) / mask.sum()),
            }
        )
    return rows


def summarize(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = values[mask]
    return {
        "mean": None if not len(selected) else float(selected.mean()),
        "median": None if not len(selected) else float(np.median(selected)),
    }


def pair_confidence(labels: np.ndarray, first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, Any]:
    first_ok = first["tta_prediction"] == labels
    second_ok = second["tta_prediction"] == labels
    result: dict[str, Any] = {}
    for name, mask in (
        ("first_only_correct", first_ok & ~second_ok),
        ("second_only_correct", ~first_ok & second_ok),
        ("both_correct", first_ok & second_ok),
        ("both_wrong", ~first_ok & ~second_ok),
    ):
        result[name] = {
            "count": int(mask.sum()),
            "first": {key: summarize(first[key], mask) for key in ("tta_confidence", "tta_margin", "tta_entropy", "flip_js_divergence")},
            "second": {key: summarize(second[key], mask) for key in ("tta_confidence", "tta_margin", "tta_entropy", "flip_js_divergence")},
            "directional_fractions": {
                "first_higher_confidence": None if not mask.any() else float(np.mean(first["tta_confidence"][mask] > second["tta_confidence"][mask])),
                "first_higher_margin": None if not mask.any() else float(np.mean(first["tta_margin"][mask] > second["tta_margin"][mask])),
                "first_lower_entropy": None if not mask.any() else float(np.mean(first["tta_entropy"][mask] < second["tta_entropy"][mask])),
                "first_lower_flip_js": None if not mask.any() else float(np.mean(first["flip_js_divergence"][mask] < second["flip_js_divergence"][mask])),
            },
        }
    return result


def raw_to_tta(labels: np.ndarray, diagnostic: dict[str, np.ndarray]) -> dict[str, int]:
    raw_ok = diagnostic["raw_prediction"] == labels
    tta_ok = diagnostic["tta_prediction"] == labels
    return {
        "raw_correct_count": int(raw_ok.sum()),
        "tta_correct_count": int(tta_ok.sum()),
        "net_gain": int(tta_ok.sum() - raw_ok.sum()),
        "corrected_by_tta": int((~raw_ok & tta_ok).sum()),
        "broken_by_tta": int((raw_ok & ~tta_ok).sum()),
    }


def scalar_fusion(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    return (float(alpha) * first + (1.0 - float(alpha)) * second).astype(np.float32)


def scalar_sweep(labels: np.ndarray, heads: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first_name, second_name in HEAD_PAIRS:
        for view in ("raw", "tta"):
            first = heads[first_name][f"{view}_logits"]
            second = heads[second_name][f"{view}_logits"]
            for step in range(101):
                alpha = step / 100.0
                metrics = metric_block(labels, scalar_fusion(first, second, alpha))
                rows.append({"first_head": first_name, "second_head": second_name, "view": view, "alpha_first": alpha, **metrics})
    return rows


def fit_classwise_coefficients(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    def objective(theta: np.ndarray) -> float:
        alpha = 1.0 / (1.0 + np.exp(-theta))
        return cross_entropy(labels, alpha[None, :] * first + (1.0 - alpha[None, :]) * second)

    fit = minimize(objective, np.zeros(first.shape[1], dtype=np.float64), method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-14})
    alpha = 1.0 / (1.0 + np.exp(-fit.x))
    logits = alpha[None, :] * first + (1.0 - alpha[None, :]) * second
    return {
        "alpha_first_by_class": alpha.tolist(),
        "optimization_success": bool(fit.success),
        "optimization_message": str(fit.message),
        "iterations": int(fit.nit),
        "objective_ce": float(fit.fun),
        "metrics": metric_block(labels, logits),
        "logits": logits.astype(np.float32),
    }


def crossfit_classwise_fusion(labels: np.ndarray, first: np.ndarray, second: np.ndarray, seed: int = 42) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_logits = np.empty_like(first, dtype=np.float32)
    fold_id = np.full(len(labels), -1, dtype=np.int64)
    folds = []
    for fold, (train_index, test_index) in enumerate(splitter.split(first, labels)):
        fitted = fit_classwise_coefficients(labels[train_index], first[train_index], second[train_index])
        alpha = np.asarray(fitted["alpha_first_by_class"])
        fold_logits = alpha[None, :] * first[test_index] + (1.0 - alpha[None, :]) * second[test_index]
        oof_logits[test_index] = fold_logits
        fold_id[test_index] = fold
        folds.append(
            {
                "fold": fold,
                "train_indices": train_index.astype(int).tolist(),
                "test_indices": test_index.astype(int).tolist(),
                "alpha_first_by_class": alpha.tolist(),
                "train_metrics": fitted["metrics"],
                "test_metrics": metric_block(labels[test_index], fold_logits),
            }
        )
    if np.any(fold_id < 0):
        raise RuntimeError("crossfit did not cover every sample exactly once")
    fold_summary = {}
    for metric in ("accuracy", "macro_f1", "loss"):
        values = np.asarray([fold["test_metrics"][metric] for fold in folds], dtype=np.float64)
        fold_summary[metric] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
    return {
        "scheme": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "folds": folds,
        "oof_fold_id": fold_id.tolist(),
        "oof_metrics": metric_block(labels, oof_logits),
        "fold_metric_summary": fold_summary,
        "oof_logits": oof_logits,
    }


def subsystem_for_key(name: str) -> str:
    prefix = name.split(".", 1)[0]
    mapping = {
        "pixel_extractor": "input_pixel_descriptor_projection",
        "pixel_proj": "input_pixel_descriptor_projection",
        "pixel_gnn": "pixel_gnn",
        "motif_composer": "spatial_motif_composer",
        "motif_gnn": "motif_graph",
        "pixel_attn_pool": "pixel_readout",
        "pixel_readout_proj": "pixel_readout",
        "motif_attn_pool": "motif_readout",
        "motif_readout_proj": "motif_readout",
        "aux_pixel_head": "auxiliary_heads",
        "aux_motif_head": "auxiliary_heads",
        "supcon_head": "auxiliary_heads",
        "classifier": "final_classifier",
    }
    return mapping.get(prefix, "other")


def weight_distance(full: Mapping[str, torch.Tensor], npf: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if set(full) != set(npf):
        raise RuntimeError("state_dict key mismatch")
    layers = []
    aggregate: dict[str, dict[str, float | int]] = {}
    total_diff2 = 0.0
    total_full2 = 0.0
    for name in sorted(full):
        a, b = full[name].detach().cpu(), npf[name].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"state tensor mismatch: {name}")
        subsystem = subsystem_for_key(name)
        entry = aggregate.setdefault(subsystem, {"parameter_count": 0, "floating_parameter_count": 0, "difference_l2_squared": 0.0, "full_l2_squared": 0.0, "dot": 0.0, "npf_l2_squared": 0.0})
        entry["parameter_count"] = int(entry["parameter_count"]) + a.numel()
        if not (torch.is_floating_point(a) or torch.is_complex(a)):
            equal = bool(torch.equal(a, b))
            layers.append({"name": name, "subsystem": subsystem, "shape": list(a.shape), "dtype": str(a.dtype), "floating": False, "equal": equal})
            continue
        av, bv = a.double().reshape(-1), b.double().reshape(-1)
        diff2 = float(torch.dot(av - bv, av - bv))
        a2 = float(torch.dot(av, av))
        b2 = float(torch.dot(bv, bv))
        dot = float(torch.dot(av, bv))
        diff = math.sqrt(diff2)
        norm = math.sqrt(a2)
        cosine = None if a2 == 0.0 or b2 == 0.0 else dot / math.sqrt(a2 * b2)
        layers.append({"name": name, "subsystem": subsystem, "shape": list(a.shape), "dtype": str(a.dtype), "floating": True, "parameter_count": a.numel(), "difference_l2": diff, "full_l2": norm, "relative_l2": None if norm == 0.0 else diff / norm, "cosine_similarity": cosine})
        entry["floating_parameter_count"] = int(entry["floating_parameter_count"]) + a.numel()
        entry["difference_l2_squared"] = float(entry["difference_l2_squared"]) + diff2
        entry["full_l2_squared"] = float(entry["full_l2_squared"]) + a2
        entry["npf_l2_squared"] = float(entry["npf_l2_squared"]) + b2
        entry["dot"] = float(entry["dot"]) + dot
        total_diff2 += diff2
        total_full2 += a2
    subsystems = {}
    weighted_relative = 0.0
    weighted_count = 0
    for name, item in aggregate.items():
        diff = math.sqrt(float(item["difference_l2_squared"]))
        anorm = math.sqrt(float(item["full_l2_squared"]))
        bnorm = math.sqrt(float(item["npf_l2_squared"]))
        relative = None if anorm == 0 else diff / anorm
        cosine = None if anorm == 0 or bnorm == 0 else float(item["dot"]) / (anorm * bnorm)
        count = int(item["floating_parameter_count"])
        if relative is not None:
            weighted_relative += count * relative
            weighted_count += count
        subsystems[name] = {"parameter_count": int(item["parameter_count"]), "floating_parameter_count": count, "difference_l2": diff, "full_l2": anorm, "relative_l2": relative, "cosine_similarity": cosine}
    return {
        "keys_exact_match": True,
        "tensor_count": len(full),
        "layers": layers,
        "subsystems": subsystems,
        "global": {
            "difference_l2": math.sqrt(total_diff2),
            "full_l2": math.sqrt(total_full2),
            "relative_l2": math.sqrt(total_diff2) / math.sqrt(total_full2),
            "parameter_count_weighted_mean_subsystem_relative_l2": weighted_relative / weighted_count,
        },
    }


def interpolate_state(full: Mapping[str, torch.Tensor], npf: Mapping[str, torch.Tensor], lambda_full: float) -> dict[str, torch.Tensor]:
    """Return lambda_full*FULL + (1-lambda_full)*NPF.

    At the midpoint, semantically discrete state is deterministically taken
    from FULL; integer/discrete tensors are never averaged.
    """
    if not 0.0 <= lambda_full <= 1.0 or set(full) != set(npf):
        raise ValueError("invalid interpolation contract")
    result = {}
    for name in full:
        a, b = full[name], npf[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"state tensor mismatch: {name}")
        if torch.is_floating_point(a) or torch.is_complex(a):
            result[name] = (lambda_full * a + (1.0 - lambda_full) * b).to(dtype=a.dtype)
        else:
            result[name] = (a if lambda_full >= 0.5 else b).clone()
    return result


def finite_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite scientific JSON value")
    if isinstance(value, Mapping):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
