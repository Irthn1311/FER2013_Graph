"""Export D16 test embeddings for read-only audits.

The exported representation is ``out["z_image"]`` from ``D16Model.forward``,
which is the image-level representation immediately before the classifier for
the current D16 readout. This script does not train or change checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from d16.data.graph_builder import collate_d16_graphs
from d16.models.d16_model import D16Model
from d16.training.train_d16 import attach_hard_proto_loss_if_needed, build_dataset, load_checkpoint, resolve_device


def _read_config(run_dir: Path) -> Dict[str, Any]:
    for name in ("resolved_config.yaml", "resolved_config.json"):
        path = run_dir / name
        if path.exists():
            if path.suffix == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raise FileNotFoundError(f"Missing resolved_config.yaml/json in {run_dir}")


def _loader_kwargs(cfg: Dict[str, Any], num_workers: int, batch_size: int | None) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    bs = int(batch_size or training_cfg.get("batch_size", data_cfg.get("batch_size", 16)) or 16)
    return {
        "batch_size": bs,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": False,
        "collate_fn": collate_d16_graphs,
    }


def _read_existing_predictions(path: Path) -> Dict[int, int]:
    preds: Dict[int, int] = {}
    if not path.exists():
        return preds
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds[int(float(row["sample_index"]))] = int(float(row["y_pred"]))
    return preds


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    name = checkpoint
    if checkpoint in {"best", "last"}:
        name = f"{checkpoint}.pt"
    path = run_dir / "checkpoints" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


@torch.no_grad()
def export_embeddings(
    run_dir: Path,
    prior_dir: Path,
    output_npz: Path,
    checkpoint: str = "best",
    device_name: str = "cuda:0",
    num_workers: int = 0,
    batch_size: int | None = None,
    max_prediction_mismatch: int = 0,
    prediction_csv: Path | None = None,
) -> Dict[str, Any]:
    cfg = _read_config(run_dir)
    device = resolve_device(device_name)
    training_cfg = cfg.get("training", {}) or {}
    if device.type == "cuda":
        allow_tf32 = bool(training_cfg.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    test_ds = build_dataset(cfg, prior_dir, "test")
    first_batch = next(iter(DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_d16_graphs)))
    model = D16Model.from_config(cfg, input_dim=first_batch.x_cat.size(1)).to(device)
    hard_proto_loss = attach_hard_proto_loss_if_needed(
        model,
        cfg.get("loss", {}) or {},
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if hard_proto_loss is not None:
        hard_proto_loss.to(device)
    ckpt_path = _checkpoint_path(run_dir, checkpoint)
    ckpt = load_checkpoint(ckpt_path, model, device)
    model.eval()

    loader = DataLoader(test_ds, **_loader_kwargs(cfg, num_workers=num_workers, batch_size=batch_size))
    amp_enabled = bool(training_cfg.get("amp", training_cfg.get("mixed_precision", False))) and device.type == "cuda"

    sample_indices: List[np.ndarray] = []
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    detected: List[np.ndarray] = []
    missing: List[np.ndarray] = []
    z_final: List[np.ndarray] = []
    logits_out: List[np.ndarray] = []
    probs_out: List[np.ndarray] = []
    confidence: List[np.ndarray] = []
    true_prob: List[np.ndarray] = []
    margin: List[np.ndarray] = []
    top2_pred: List[np.ndarray] = []
    major_tokens: List[np.ndarray] = []
    micro_tokens: List[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            out = model(batch)
            logits = out["logits"]
        probs = torch.softmax(logits, dim=1)
        top2 = torch.topk(probs, k=2, dim=1)
        pred = logits.argmax(dim=1)
        z = out["z_image"]

        sample_indices.append(batch.sample_index.detach().cpu().numpy().astype(np.int64))
        y = batch.y.detach().cpu().numpy().astype(np.int64)
        y_true.append(y)
        y_pred.append(pred.detach().cpu().numpy().astype(np.int64))
        detected.append(batch.landmark_missing_flag.detach().cpu().long().eq(0).numpy().astype(np.int64))
        missing.append(batch.landmark_missing_flag.detach().cpu().numpy().astype(np.int64))
        z_final.append(z.detach().float().cpu().numpy().astype(np.float32))
        logits_out.append(logits.detach().float().cpu().numpy().astype(np.float32))
        probs_np = probs.detach().float().cpu().numpy().astype(np.float32)
        probs_out.append(probs_np)
        confidence.append(top2.values[:, 0].detach().float().cpu().numpy().astype(np.float32))
        top2_pred.append(top2.indices[:, 1].detach().cpu().numpy().astype(np.int64))
        margin.append((top2.values[:, 0] - top2.values[:, 1]).detach().float().cpu().numpy().astype(np.float32))
        true_prob.append(probs.detach().float().cpu()[torch.arange(batch.y.numel()), batch.y.detach().cpu()].numpy().astype(np.float32))
        if isinstance(out.get("micro_major_motif_tokens"), torch.Tensor):
            major_tokens.append(out["micro_major_motif_tokens"].detach().float().cpu().numpy().astype(np.float32))
        if isinstance(out.get("micro_motif_tokens"), torch.Tensor):
            micro_tokens.append(out["micro_motif_tokens"].detach().float().cpu().numpy().astype(np.float32))

    arrays: Dict[str, np.ndarray] = {
        "sample_index": np.concatenate(sample_indices, axis=0),
        "y_true": np.concatenate(y_true, axis=0),
        "y_pred": np.concatenate(y_pred, axis=0),
        "detected": np.concatenate(detected, axis=0),
        "landmark_missing_flag": np.concatenate(missing, axis=0),
        "z_final_before_classifier": np.concatenate(z_final, axis=0),
        "logits": np.concatenate(logits_out, axis=0),
        "probs": np.concatenate(probs_out, axis=0),
        "confidence": np.concatenate(confidence, axis=0),
        "true_prob": np.concatenate(true_prob, axis=0),
        "margin_top1_top2": np.concatenate(margin, axis=0),
        "top2_pred": np.concatenate(top2_pred, axis=0),
    }
    if major_tokens:
        arrays["major_motif_tokens"] = np.concatenate(major_tokens, axis=0)
    if micro_tokens:
        arrays["micro_motif_tokens"] = np.concatenate(micro_tokens, axis=0)
    if hard_proto_loss is not None:
        arrays["hard_proto_prototypes"] = hard_proto_loss.prototypes.detach().float().cpu().numpy().astype(np.float32)

    if arrays["sample_index"].shape[0] != 3589:
        raise ValueError(f"Expected 3589 rows, got {arrays['sample_index'].shape[0]}")
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise ValueError(f"NaN/Inf found in {name}")

    prediction_csv = prediction_csv or (run_dir / ("last_predictions.csv" if checkpoint == "last" else "predictions.csv"))
    existing = _read_existing_predictions(prediction_csv)
    mismatch = 0
    if existing:
        artifact_pred = np.array(
            [existing.get(int(sample_idx), -1) for sample_idx in arrays["sample_index"].tolist()],
            dtype=np.int64,
        )
        arrays["artifact_y_pred"] = artifact_pred
        mismatch = int(np.sum(artifact_pred != arrays["y_pred"]))
    if mismatch > int(max_prediction_mismatch):
        raise ValueError(f"Exported predictions do not match predictions.csv: mismatches={mismatch}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", 0) or 0),
        "best_monitor_metric": str(ckpt.get("best_monitor_metric", "")),
        "best_monitor_score": float(ckpt.get("best_monitor_score", float("nan"))),
        "output_npz": str(output_npz),
        "row_count": int(arrays["sample_index"].shape[0]),
        "embedding_dim": int(arrays["z_final_before_classifier"].shape[1]),
        "prediction_mismatches": int(mismatch),
        "prediction_csv": str(prediction_csv),
        "max_prediction_mismatch_allowed": int(max_prediction_mismatch),
        "has_artifact_y_pred": "artifact_y_pred" in arrays,
        "has_major_motif_tokens": "major_motif_tokens" in arrays,
        "has_micro_motif_tokens": "micro_motif_tokens" in arrays,
        "has_hard_proto_prototypes": "hard_proto_prototypes" in arrays,
    }
    (output_npz.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_prediction_mismatch", type=int, default=0)
    parser.add_argument("--prediction_csv", default=None)
    args = parser.parse_args()
    summary = export_embeddings(
        run_dir=Path(args.run_dir),
        prior_dir=Path(args.prior_dir),
        output_npz=Path(args.output_npz),
        checkpoint=args.checkpoint,
        device_name=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_prediction_mismatch=args.max_prediction_mismatch,
        prediction_csv=Path(args.prediction_csv) if args.prediction_csv else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
