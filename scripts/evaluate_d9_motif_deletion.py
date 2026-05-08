"""Light motif deletion test for D9 relation motif checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import build_dataloader, load_config, resolve_device, resolve_existing_path, resolve_path  # noqa: E402
from evaluation.metrics import classification_report_dict, compute_metrics, confusion_matrix_array  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims  # noqa: E402


def _parse_top_k(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint


def _forward_deleted(model: torch.nn.Module, batch: Dict[str, torch.Tensor], top_k: int) -> torch.Tensor:
    if not all(hasattr(model, name) for name in ("pixel_encoder", "motif_discovery", "motif_relation_classifier")):
        raise TypeError("Motif deletion currently supports D9RelationMotifClassifier-style models only")
    x = batch.get("x", batch.get("node_features"))
    h_pixel = model.pixel_encoder(
        x,
        edge_index=batch["edge_index"],
        edge_attr=batch["edge_attr"],
        node_mask=batch.get("node_mask"),
    )
    motif_out = model.motif_discovery(h_pixel, image_hw=(model.height, model.width), node_mask=batch.get("node_mask"))
    selection_weights = torch.softmax(motif_out["motif_scores"] / float(model.selection_temperature), dim=1)
    k = max(1, min(int(top_k), int(selection_weights.shape[1])))
    top_idx = torch.topk(selection_weights, k=k, dim=1).indices
    keep = torch.ones_like(selection_weights)
    keep.scatter_(dim=1, index=top_idx, value=0.0)
    deleted_embeddings = motif_out["motif_embeddings"] * keep.unsqueeze(-1)
    deleted_weights = selection_weights * keep
    deleted_weights = deleted_weights / deleted_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    out = model.motif_relation_classifier(
        motif_embeddings=deleted_embeddings,
        motif_maps=motif_out["motif_assignment_maps"],
        selection_weights=deleted_weights,
        centers=motif_out.get("motif_centers"),
        area=motif_out.get("motif_area"),
    )
    return out["logits"]


@torch.no_grad()
def evaluate_deletion(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: Dict[str, Any],
    top_k_values: list[int],
    max_samples: int,
) -> Dict[str, Any]:
    model.eval()
    model_cfg = dict(config.get("model", {}) or {})
    feature_cfg = dict(config.get("feature_ablation", {}) or {})
    normal_true: list[int] = []
    normal_pred: list[int] = []
    normal_prob_true: list[float] = []
    graph_ids: list[int] = []
    deleted_pred = {k: [] for k in top_k_values}
    deleted_prob_true = {k: [] for k in top_k_values}
    processed = 0
    for raw_batch in loader:
        if processed >= int(max_samples):
            break
        raw_batch = move_to_device(raw_batch, device)
        batch = apply_feature_ablation(dict(raw_batch), feature_cfg)
        assert_feature_dims(
            batch,
            node_dim=int(model_cfg.get("node_dim", 3)),
            edge_dim=int(model_cfg.get("edge_dim", 5)),
        )
        remaining = int(max_samples) - processed
        if int(batch["y"].shape[0]) > remaining:
            batch = _slice_batch(batch, remaining)
        labels = batch["y"].long()
        out = model(batch)
        normal_logits = out["logits"].detach().float()
        normal_probs = torch.softmax(normal_logits, dim=1)
        pred = normal_logits.argmax(dim=1)
        normal_true.extend(labels.detach().cpu().tolist())
        normal_pred.extend(pred.detach().cpu().tolist())
        normal_prob_true.extend(normal_probs.gather(1, labels.view(-1, 1)).squeeze(1).cpu().tolist())
        graph_ids.extend(batch["graph_id"].detach().cpu().tolist())
        for k in top_k_values:
            logits_k = _forward_deleted(model, batch, top_k=k).detach().float()
            probs_k = torch.softmax(logits_k, dim=1)
            deleted_pred[k].extend(logits_k.argmax(dim=1).cpu().tolist())
            deleted_prob_true[k].extend(probs_k.gather(1, labels.view(-1, 1)).squeeze(1).cpu().tolist())
        processed += int(labels.shape[0])
    normal_metrics = compute_metrics(normal_true, normal_pred)
    result: Dict[str, Any] = {
        "normal": {
            **normal_metrics,
            "classification_report": classification_report_dict(normal_true, normal_pred),
            "confusion_matrix": confusion_matrix_array(normal_true, normal_pred).tolist(),
            "pred_distribution": _pred_distribution(normal_pred),
            "mean_true_prob": float(np.mean(normal_prob_true)) if normal_prob_true else 0.0,
        },
        "deletion": {},
        "samples": int(len(normal_true)),
    }
    normal_pred_np = np.asarray(normal_pred, dtype=np.int64)
    true_prob_np = np.asarray(normal_prob_true, dtype=np.float64)
    for k in top_k_values:
        metrics_k = compute_metrics(normal_true, deleted_pred[k])
        pred_k_np = np.asarray(deleted_pred[k], dtype=np.int64)
        prob_k_np = np.asarray(deleted_prob_true[k], dtype=np.float64)
        result["deletion"][f"top{k}"] = {
            **metrics_k,
            "macro_f1_drop": float(normal_metrics["macro_f1"] - metrics_k["macro_f1"]),
            "accuracy_drop": float(normal_metrics["accuracy"] - metrics_k["accuracy"]),
            "correct_prob_drop": float(true_prob_np.mean() - prob_k_np.mean()) if len(prob_k_np) else 0.0,
            "prediction_change_rate": float((pred_k_np != normal_pred_np).mean()) if len(pred_k_np) else 0.0,
            "classification_report": classification_report_dict(normal_true, deleted_pred[k]),
            "confusion_matrix": confusion_matrix_array(normal_true, deleted_pred[k]).tolist(),
            "pred_distribution": _pred_distribution(deleted_pred[k]),
        }
    result["graph_ids"] = graph_ids
    return result


def _slice_batch(batch: Dict[str, Any], n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == batch["y"].shape[:1]:
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _pred_distribution(values: list[int]) -> list[int]:
    return np.bincount(np.asarray(values, dtype=np.int64), minlength=7).tolist() if values else [0] * 7


def _save_outputs(result: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "motif_deletion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with (output_dir / "motif_deletion_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "macro_f1",
                "accuracy",
                "weighted_f1",
                "macro_f1_drop",
                "accuracy_drop",
                "correct_prob_drop",
                "prediction_change_rate",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "case": "normal",
            "macro_f1": result["normal"]["macro_f1"],
            "accuracy": result["normal"]["accuracy"],
            "weighted_f1": result["normal"]["weighted_f1"],
        })
        for name, item in result["deletion"].items():
            writer.writerow({
                "case": name,
                "macro_f1": item["macro_f1"],
                "accuracy": item["accuracy"],
                "weighted_f1": item["weighted_f1"],
                "macro_f1_drop": item["macro_f1_drop"],
                "accuracy_drop": item["accuracy_drop"],
                "correct_prob_drop": item["correct_prob_drop"],
                "prediction_change_rate": item["prediction_change_rate"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--environment", "--env", default=None, choices=["local", "kaggle"])
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--top_k", default="1,3")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, environment=args.environment)
    if args.graph_repo_path:
        config.setdefault("paths", {})["graph_repo_path"] = args.graph_repo_path
        config.setdefault("data", {})["graph_repo_path"] = args.graph_repo_path
    set_seed(int(config.get("training", {}).get("seed", 42)))
    device = resolve_device(args.device, config=config)
    model = build_model(dict(config.get("model", {}) or {})).to(device)
    checkpoint_path = resolve_existing_path(args.checkpoint)
    _load_checkpoint(model, checkpoint_path, device)
    loader = build_dataloader(config, split=str(args.split), shuffle=False)
    result = evaluate_deletion(
        model=model,
        loader=loader,
        device=device,
        config=config,
        top_k_values=_parse_top_k(args.top_k),
        max_samples=int(args.max_samples),
    )
    if args.output_dir:
        output_dir = resolve_path(args.output_dir) or Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / "motif_deletion"
    _save_outputs(result, output_dir)
    print(f"[MotifDeletion] samples={result['samples']} output_dir={output_dir}")
    for name, item in result["deletion"].items():
        print(
            f"[MotifDeletion] {name} macro_f1={item['macro_f1']:.6f} "
            f"drop={item['macro_f1_drop']:.6f} change_rate={item['prediction_change_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
