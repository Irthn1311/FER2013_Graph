"""D13A diagnostics and reporting utilities.

These helpers describe reduction health only. They intentionally avoid any
motif language because D13A is a reduction baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from data.labels import EMOTION_NAMES


def compute_pred_count(y_pred: Sequence[int], num_classes: int = 7) -> Dict[str, int]:
    arr = np.asarray(list(y_pred), dtype=np.int64)
    counts = np.bincount(arr, minlength=int(num_classes))
    return {f"pred_count_{idx}_{EMOTION_NAMES[idx]}": int(counts[idx]) for idx in range(int(num_classes))}


def compute_per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 7) -> Dict[str, float]:
    scores = f1_score(
        np.asarray(list(y_true), dtype=np.int64),
        np.asarray(list(y_pred), dtype=np.int64),
        labels=list(range(int(num_classes))),
        average=None,
        zero_division=0,
    )
    return {f"f1_{idx}_{EMOTION_NAMES[idx]}": float(scores[idx]) for idx in range(int(num_classes))}


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def compute_assignment_stats(aux: Dict[str, Any]) -> Dict[str, float]:
    keys = (
        "assignment_entropy",
        "effective_regions",
        "empty_region_ratio",
        "region_area_min",
        "region_area_mean",
        "region_area_max",
        "region_area_std",
        "balance_loss",
        "compactness_loss",
        "entropy_loss",
        "area_loss",
    )
    out: Dict[str, float] = {}
    for key in keys:
        if key in aux:
            out[key] = _to_float(aux[key])
    return out


def compute_effective_regions(region_area: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    area = region_area.float()
    return area.sum(dim=-1).pow(2) / area.pow(2).sum(dim=-1).clamp_min(float(eps))


def compute_encoder_oversmoothing_score(layer_embeddings: Iterable[torch.Tensor] | None) -> Dict[str, float]:
    if not layer_embeddings:
        return {}
    scores: Dict[str, float] = {}
    for idx, h in enumerate(layer_embeddings):
        if not torch.is_tensor(h) or h.ndim != 2 or h.shape[0] < 2:
            continue
        rows = h.detach().float()
        if rows.shape[0] > 2048:
            step = max(rows.shape[0] // 2048, 1)
            rows = rows[::step]
        normed = F.normalize(rows, dim=1, eps=1e-8)
        sim = normed @ normed.t()
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        scores[f"encoder_layer{idx}_mean_cosine"] = float(sim[mask].mean().cpu()) if mask.any() else 0.0
    return scores


def write_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: str | Path,
    num_classes: int = 7,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(int(num_classes))))
    df = pd.DataFrame(cm, index=EMOTION_NAMES[:num_classes], columns=EMOTION_NAMES[:num_classes])
    df.to_csv(output_path)


def _read_last_csv_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def _summarize_pooling(pooling_csv: Path) -> Dict[str, float]:
    if not pooling_csv.exists():
        return {}
    df = pd.read_csv(pooling_csv)
    if df.empty:
        return {}
    keys = [
        "assignment_entropy",
        "effective_regions",
        "empty_region_ratio",
        "region_area_min",
        "region_area_mean",
        "region_area_max",
        "region_area_std",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        if key in df:
            out[f"{key}_mean"] = float(df[key].mean())
            out[f"{key}_last"] = float(df[key].iloc[-1])
    return out


def write_d13_report(
    output_dir: str | Path,
    config: Dict[str, Any],
    final_val: Dict[str, Any] | None = None,
    final_test: Dict[str, Any] | None = None,
    decision: str | None = None,
    warnings: List[str] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_val = final_val or _read_last_csv_row(output_dir / "val_metrics.csv")
    final_test = final_test or _read_last_csv_row(output_dir / "test_metrics.csv")
    pool_summary = _summarize_pooling(output_dir / "pooling_stats.csv")
    warnings = list(warnings or [])
    if pool_summary.get("empty_region_ratio_last", 0.0) > 0.25:
        warnings.append("High empty-region ratio; reduction may be collapsing.")
    if pool_summary.get("effective_regions_last", 999.0) < 72.0:
        warnings.append("Effective regions below half of K=144; check assignment balance.")
    if decision is None:
        if warnings:
            decision = "D13A_FAIL_COLLAPSE"
        elif not final_val and not final_test:
            decision = "D13A_FAIL_TRAINING_UNSTABLE"
        else:
            decision = "D13A_PASS_REDUCTION_BASELINE"

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    lines = [
        "# D13A Hierarchical Reduction Report",
        "",
        "## Config Summary",
        f"- model: {model_cfg.get('name', 'd13_hierarchical_reduction')}",
        f"- encoder: {model_cfg.get('encoder', {})}",
        f"- pooling: {model_cfg.get('pooling', {})}",
        f"- hidden_dim: {model_cfg.get('hidden_dim')}",
        f"- pixel_layers: {model_cfg.get('pixel_layers')}",
        f"- region_layers: {model_cfg.get('region_layers')}",
        f"- epochs: {train_cfg.get('epochs', train_cfg.get('max_epochs'))}",
        f"- amp: {train_cfg.get('amp', False)}",
        "",
        "## Final Metrics",
        "### Validation",
        "```json",
        json.dumps(final_val, indent=2, default=str),
        "```",
        "### Test",
        "```json",
        json.dumps(final_test, indent=2, default=str),
        "```",
        "",
        "## Pooling Stats Summary",
        "```json",
        json.dumps(pool_summary, indent=2, default=str),
        "```",
        "",
        "## Collapse Warnings",
    ]
    if warnings:
        lines.extend([f"- {w}" for w in warnings])
    else:
        lines.append("- none")
    lines.extend(["", "## Decision", decision, "", "No motif claim is made for D13A."])
    path = output_dir / "d13a_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

