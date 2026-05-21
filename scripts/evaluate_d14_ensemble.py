"""Evaluate D14 checkpoint ensembles without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import build_dataloader, load_config, resolve_device
from data.labels import EMOTION_NAMES
from evaluation.d13_diagnostics import write_confusion_matrix
from models.d13c_supcon_model import D13CSupConModel
from training.train_d13b import _metrics, _pred_count_row


def _load_model(config_path: str, checkpoint_path: str, device: torch.device, environment: str | None) -> torch.nn.Module:
    cfg = load_config(config_path, environment=environment)
    model = D13CSupConModel.from_config(cfg.get("model", {})).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _predict_model(model: torch.nn.Module, loader, device: torch.device, max_batches: int | None = None) -> Dict[str, np.ndarray]:
    logits_all, probs_all, labels_all, graph_ids = [], [], [], []
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        logits = out["logits"].detach().float()
        probs = torch.softmax(logits, dim=1)
        logits_all.append(logits.cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        labels_all.append(batch["y"].detach().cpu().numpy())
        graph_ids.append(batch.get("graph_id", batch.get("sample_idx")).detach().cpu().numpy())
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "probs": np.concatenate(probs_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "graph_id": np.concatenate(graph_ids, axis=0),
    }


def _safe_softmax(logits: np.ndarray) -> np.ndarray:
    work = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(work)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _metric_row(name: str, probs: np.ndarray, labels: np.ndarray, weights: List[float] | None = None) -> Dict[str, Any]:
    pred = probs.argmax(axis=1)
    row = {"method": name, **_metrics(labels, pred)}
    row.update({"weights": "" if weights is None else json.dumps([round(float(w), 4) for w in weights])})
    return row


def _read_val_weight(checkpoint_path: str) -> float:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = ckpt.get("metrics", {})
        return float(metrics.get("val_macro_f1", metrics.get("macro_f1", 1.0)))
    except Exception:
        return 1.0


def _grid_weights(num_models: int) -> List[List[float]]:
    if num_models < 2:
        return [[1.0]]
    grids: List[List[float]] = []
    for w0 in (0.4, 0.5, 0.6):
        for w1 in (0.2, 0.3, 0.4):
            rem = 1.0 - w0 - w1
            if rem < -1e-9:
                continue
            weights = [w0, w1]
            if num_models == 2:
                weights = [w0, 1.0 - w0]
            else:
                weights.extend([rem / (num_models - 2)] * (num_models - 2))
            if min(weights) >= -1e-9 and abs(sum(weights) - 1.0) < 1e-6:
                grids.append([float(max(0.0, w)) for w in weights])
    return grids


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No data."
    use = df.head(max_rows).copy()
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(args.configs) != len(args.checkpoints):
        raise ValueError("--configs and --checkpoints must have the same length")
    names = args.names or [Path(c).stem for c in args.configs]
    if len(names) != len(args.configs):
        raise ValueError("--names length must match --configs")
    cfg0 = load_config(args.configs[0], environment=args.environment)
    device = resolve_device(args.device, cfg0)
    loader = build_dataloader(cfg0, args.split, shuffle=False)

    model_outputs = []
    for name, cfg_path, ckpt_path in zip(names, args.configs, args.checkpoints):
        model = _load_model(cfg_path, ckpt_path, device, args.environment)
        pred = _predict_model(model, loader, device, max_batches=args.max_batches)
        np.savez_compressed(output_dir / f"{name}_test_logits_probs.npz", **pred)
        model_outputs.append(pred)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = model_outputs[0]["labels"]
    graph_ids = model_outputs[0]["graph_id"]
    logits_stack = np.stack([m["logits"] for m in model_outputs], axis=0)
    probs_stack = np.stack([m["probs"] for m in model_outputs], axis=0)
    rows: List[Dict[str, Any]] = []
    rows.append(_metric_row("equal_logits", _safe_softmax(logits_stack.mean(axis=0)), labels, [1.0 / len(model_outputs)] * len(model_outputs)))
    rows.append(_metric_row("equal_probs", probs_stack.mean(axis=0), labels, [1.0 / len(model_outputs)] * len(model_outputs)))

    raw_val = np.asarray([_read_val_weight(p) for p in args.checkpoints], dtype=np.float64)
    val_weights = raw_val / np.clip(raw_val.sum(), 1e-12, None)
    rows.append(_metric_row("val_weighted_probs", np.tensordot(val_weights, probs_stack, axes=(0, 0)), labels, val_weights.tolist()))

    for weights in _grid_weights(len(model_outputs)):
        probs = np.tensordot(np.asarray(weights, dtype=np.float64), probs_stack, axes=(0, 0))
        rows.append(_metric_row("grid_probs", probs, labels, weights))

    metrics = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    metrics.to_csv(output_dir / "ensemble_metrics.csv", index=False)
    best = metrics.iloc[0].to_dict()
    best_weights = json.loads(best.get("weights") or "[]")
    if best_weights:
        best_probs = np.tensordot(np.asarray(best_weights, dtype=np.float64), probs_stack, axes=(0, 0))
    elif best["method"] == "equal_logits":
        best_probs = _safe_softmax(logits_stack.mean(axis=0))
    else:
        best_probs = probs_stack.mean(axis=0)
    best_pred = best_probs.argmax(axis=1)

    pd.DataFrame({"graph_id": graph_ids, "y_true": labels, "y_pred": best_pred}).to_csv(output_dir / "ensemble_predictions.csv", index=False)
    pd.DataFrame([_pred_count_row(0, "test", _metrics(labels, best_pred))]).to_csv(output_dir / "ensemble_pred_count.csv", index=False)
    per_class = {"method": best["method"]}
    for key, value in _metrics(labels, best_pred).items():
        if str(key).startswith("f1_"):
            per_class[key] = value
    pd.DataFrame([per_class]).to_csv(output_dir / "ensemble_per_class_metrics.csv", index=False)
    write_confusion_matrix(labels, best_pred, output_dir / "ensemble_confusion_matrix.csv")

    lines = [
        "# D14 Ensemble Evaluation Report",
        "",
        "D14 performance-first ensemble evaluation only. No prototype, no motif-level SupCon, no evidence or interpretability claim.",
        "",
        f"- split: `{args.split}`",
        f"- models: {', '.join(names)}",
        f"- best_method: `{best['method']}`",
        f"- best_macro_f1: {float(best.get('macro_f1', 0.0)):.6f}",
        f"- best_accuracy: {float(best.get('accuracy', 0.0)):.6f}",
        f"- best_weights: `{best.get('weights', '')}`",
        "",
        "## Ranking",
        _md_table(metrics),
        "",
        "Decision hint: `D14_ENSEMBLE_PROMISING` if this clearly improves over the 0.6481 D13C M8 control; `D14_REACHED_0P70_TARGET` if accuracy reaches 0.70.",
        "",
    ]
    (output_dir / "ensemble_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
