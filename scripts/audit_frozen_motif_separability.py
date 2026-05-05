"""Audit class separability of frozen D8M motif representations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, resolve_path, save_config  # noqa: E402
from evaluation.metrics import classification_report_dict, confusion_matrix_array  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims, log_feature_ablation  # noqa: E402
from utils.motif_graph_builder import build_motif_graph  # noqa: E402
from utils.motif_stage1_loader import load_frozen_motif_model  # noqa: E402


EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = apply_cli_overrides(config, args)
    paths = dict(cfg.get("paths", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    output = dict(cfg.get("output", {}) or {})
    if getattr(args, "output_dir", None):
        paths["resolved_output_root"] = str(args.output_dir)
        output["dir"] = str(args.output_dir)
    elif output.get("dir"):
        paths["resolved_output_root"] = str(output["dir"])
    if getattr(args, "graph_repo_path", None):
        paths["graph_repo_path"] = str(args.graph_repo_path)
    if getattr(args, "batch_size", None) is not None:
        data["batch_size"] = int(args.batch_size)
    if sys.platform.startswith("win") and getattr(args, "num_workers", None) is None:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    cfg["paths"] = paths
    cfg["data"] = data
    cfg["output"] = output
    return cfg


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _normalized(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1, eps=1e-8)


def _pool_selected_embeddings(selected_embeddings: torch.Tensor, selected_weights: torch.Tensor) -> torch.Tensor:
    weights = selected_weights.to(dtype=selected_embeddings.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mean_pool = selected_embeddings.mean(dim=1)
    max_pool = selected_embeddings.max(dim=1).values
    weighted_pool = (selected_embeddings * weights.unsqueeze(-1)).sum(dim=1)
    return torch.cat([mean_pool, max_pool, weighted_pool], dim=1)


def _extract_split(
    *,
    stage1_model: torch.nn.Module,
    stage1_model_cfg: Dict[str, Any],
    feature_ablation_cfg: Dict[str, Any],
    loader,
    stage2_cfg: Dict[str, Any],
    device: torch.device,
    max_batches: int | None,
    split: str,
) -> Dict[str, torch.Tensor]:
    selected_embeddings: list[torch.Tensor] = []
    selected_weights: list[torch.Tensor] = []
    pooled_repr: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    selected_indices: list[torch.Tensor] = []
    graph_ids: list[torch.Tensor] = []

    stage1_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, device)
            batch = apply_feature_ablation(batch, feature_ablation_cfg)
            assert_feature_dims(
                batch,
                node_dim=int(stage1_model_cfg.get("node_dim", 7)),
                edge_dim=int(stage1_model_cfg.get("edge_dim", 5)),
            )
            outputs = {k: v.detach() if torch.is_tensor(v) else v for k, v in stage1_model(batch).items()}
            motif_graph = build_motif_graph(outputs, stage2_cfg)
            embeddings = motif_graph["selected_embeddings"].detach().float().cpu()
            weights = motif_graph["selected_weights"].detach().float().cpu()
            selected_embeddings.append(embeddings)
            selected_weights.append(weights)
            pooled_repr.append(_pool_selected_embeddings(embeddings, weights).cpu())
            labels.append(batch["y"].detach().long().cpu())
            selected_indices.append(motif_graph["selected_indices"].detach().long().cpu())
            if "graph_id" in batch:
                gid = batch["graph_id"]
                if torch.is_tensor(gid):
                    graph_ids.append(gid.detach().cpu())
    if not labels:
        raise RuntimeError(f"No samples extracted for split={split!r}")
    out = {
        "selected_embeddings": torch.cat(selected_embeddings, dim=0),
        "selected_weights": torch.cat(selected_weights, dim=0),
        "pooled_repr": torch.cat(pooled_repr, dim=0),
        "labels": torch.cat(labels, dim=0),
        "selected_indices": torch.cat(selected_indices, dim=0),
    }
    if graph_ids:
        out["graph_ids"] = torch.cat(graph_ids, dim=0)
    print(
        f"[Extract {split}] samples={out['labels'].numel()} "
        f"selected_embeddings={tuple(out['selected_embeddings'].shape)} pooled_repr={tuple(out['pooled_repr'].shape)}"
    )
    return out


def _class_centroids(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    centroids = []
    present = []
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if bool(mask.any()):
            centroids.append(features[mask].mean(dim=0))
            present.append(class_idx)
    if not centroids:
        raise RuntimeError("No class centroids could be computed")
    return torch.stack(centroids, dim=0), torch.tensor(present, dtype=torch.long)


def _nearest_centroid(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, Dict[str, Any]]:
    train_n = _normalized(train_x)
    val_n = _normalized(val_x)
    centroids, present = _class_centroids(train_n, train_y, num_classes)
    centroids = _normalized(centroids)
    sims = val_n @ centroids.t()
    pred = present[sims.argmax(dim=1)]
    centroid_cos = centroids @ centroids.t()
    if centroid_cos.shape[0] > 1:
        mask = ~torch.eye(centroid_cos.shape[0], dtype=torch.bool)
        mean_centroid_cosine = float(centroid_cos[mask].mean().cpu())
        mean_centroid_distance = float((1.0 - centroid_cos[mask]).mean().cpu())
    else:
        mean_centroid_cosine = float("nan")
        mean_centroid_distance = float("nan")
    return pred, {
        "present_classes": present.tolist(),
        "centroid_cosine_matrix": _as_numpy(centroid_cos).tolist(),
        "mean_centroid_cosine": mean_centroid_cosine,
        "mean_centroid_cosine_distance": mean_centroid_distance,
    }


def _intra_inter_cosine(features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    x = _normalized(features)
    sim = x @ x.t()
    same = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(labels.numel(), dtype=torch.bool)
    intra_mask = same & ~eye
    inter_mask = ~same
    return {
        "intra_class_cosine_mean": float(sim[intra_mask].mean().cpu()) if bool(intra_mask.any()) else float("nan"),
        "inter_class_cosine_mean": float(sim[inter_mask].mean().cpu()) if bool(inter_mask.any()) else float("nan"),
    }


def _knn_predict(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, k: int) -> torch.Tensor:
    train_n = _normalized(train_x)
    val_n = _normalized(val_x)
    sims = val_n @ train_n.t()
    top_idx = sims.topk(k=min(int(k), train_y.numel()), dim=1).indices
    top_labels = train_y[top_idx]
    preds = []
    for row_labels, row_sims in zip(top_labels, sims.gather(dim=1, index=top_idx)):
        scores: Dict[int, float] = {}
        for label, sim in zip(row_labels.tolist(), row_sims.tolist()):
            scores[int(label)] = scores.get(int(label), 0.0) + float(sim)
        preds.append(max(scores.items(), key=lambda item: (item[1], -item[0]))[0])
    return torch.tensor(preds, dtype=torch.long)


def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(pred.eq(target).float().mean().cpu()) if target.numel() else 0.0


def _pred_dist(pred: torch.Tensor, num_classes: int) -> list[int]:
    return [int((pred == class_idx).sum().item()) for class_idx in range(num_classes)]


def _evaluate(train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor], num_classes: int) -> Dict[str, Any]:
    train_x = train["pooled_repr"].float()
    train_y = train["labels"].long()
    val_x = val["pooled_repr"].float()
    val_y = val["labels"].long()

    centroid_pred, centroid_stats = _nearest_centroid(train_x, train_y, val_x, num_classes)
    metrics: Dict[str, Any] = {
        "num_train": int(train_y.numel()),
        "num_val": int(val_y.numel()),
        "train_class_counts": [int((train_y == i).sum().item()) for i in range(num_classes)],
        "val_class_counts": [int((val_y == i).sum().item()) for i in range(num_classes)],
        "nearest_centroid_accuracy": _accuracy(centroid_pred, val_y),
        "nearest_centroid_pred_dist": _pred_dist(centroid_pred, num_classes),
        "nearest_centroid_classification_report": classification_report_dict(val_y.tolist(), centroid_pred.tolist()),
        "nearest_centroid_confusion_matrix": confusion_matrix_array(val_y.tolist(), centroid_pred.tolist()).tolist(),
    }
    metrics.update(centroid_stats)
    metrics["train_cosine"] = _intra_inter_cosine(train_x, train_y)
    metrics["val_cosine"] = _intra_inter_cosine(val_x, val_y)
    for k in (1, 3, 5):
        pred = _knn_predict(train_x, train_y, val_x, k=k)
        metrics[f"knn{k}_accuracy"] = _accuracy(pred, val_y)
        metrics[f"knn{k}_pred_dist"] = _pred_dist(pred, num_classes)
        metrics[f"knn{k}_classification_report"] = classification_report_dict(val_y.tolist(), pred.tolist())
        metrics[f"knn{k}_confusion_matrix"] = confusion_matrix_array(val_y.tolist(), pred.tolist()).tolist()
    return metrics


def _write_npz(path: Path, train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor]) -> None:
    payload = {}
    for split, data in (("train", train), ("val", val)):
        for key, value in data.items():
            payload[f"{split}_{key}"] = _as_numpy(value)
    np.savez_compressed(path, **payload)


def _write_summary_csv(path: Path, metrics: Dict[str, Any]) -> None:
    rows = [
        {"metric": "nearest_centroid_accuracy", "value": metrics["nearest_centroid_accuracy"]},
        {"metric": "mean_centroid_cosine", "value": metrics["mean_centroid_cosine"]},
        {"metric": "mean_centroid_cosine_distance", "value": metrics["mean_centroid_cosine_distance"]},
        {"metric": "train_intra_class_cosine_mean", "value": metrics["train_cosine"]["intra_class_cosine_mean"]},
        {"metric": "train_inter_class_cosine_mean", "value": metrics["train_cosine"]["inter_class_cosine_mean"]},
        {"metric": "val_intra_class_cosine_mean", "value": metrics["val_cosine"]["intra_class_cosine_mean"]},
        {"metric": "val_inter_class_cosine_mean", "value": metrics["val_cosine"]["inter_class_cosine_mean"]},
        {"metric": "knn1_accuracy", "value": metrics["knn1_accuracy"]},
        {"metric": "knn3_accuracy", "value": metrics["knn3_accuracy"]},
        {"metric": "knn5_accuracy", "value": metrics["knn5_accuracy"]},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_pca_plot(path: Path, train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional artifact
        print(f"[PCA] skipped: matplotlib unavailable ({exc})")
        return
    x = torch.cat([train["pooled_repr"], val["pooled_repr"]], dim=0).float()
    labels = torch.cat([train["labels"], val["labels"]], dim=0).long()
    split = np.asarray(["train"] * train["labels"].numel() + ["val"] * val["labels"].numel())
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=2, center=False)
    coords = (x @ v[:, :2]).cpu().numpy()
    labels_np = labels.cpu().numpy()
    plt.figure(figsize=(8, 6))
    for class_idx, name in enumerate(EMOTION_NAMES):
        mask = labels_np == class_idx
        if not np.any(mask):
            continue
        marker = "o"
        plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.65, marker=marker, label=name)
    val_mask = split == "val"
    plt.scatter(coords[val_mask, 0], coords[val_mask, 1], s=42, facecolors="none", edgecolors="black", linewidths=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--train_batches", type=int, default=60)
    parser.add_argument("--val_batches", type=int, default=60)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--no_pca", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _update_config(load_config(args.config, environment=args.environment), args)
    set_seed(int(config.get("training", {}).get("seed", 42)))
    device = resolve_device(args.device, config=config)
    stage1_cfg = dict(config.get("stage1", {}) or {})
    stage2_cfg = dict(config.get("stage2", {}) or {})
    output_root = resolve_path(config.get("paths", {}).get("resolved_output_root") or config.get("output", {}).get("dir"))
    if output_root is None:
        output_root = PROJECT_ROOT / "outputs" / f"{Path(args.config).stem}_separability"
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)

    stage1_config = load_config(stage1_cfg["config"], environment=args.environment)
    stage1_model_cfg = dict(stage1_config.get("model", {}) or {})
    feature_ablation_cfg = dict(stage1_config.get("feature_ablation", {}) or {})
    log_feature_ablation(
        feature_ablation_cfg,
        model_node_dim=int(stage1_model_cfg.get("node_dim", 7)),
        model_edge_dim=int(stage1_model_cfg.get("edge_dim", 5)),
        prefix="[FeatureAblation audit stage1]",
    )
    stage1_model = load_frozen_motif_model(stage1_cfg["config"], stage1_cfg["checkpoint"], device)
    train_loader = build_dataloader(config, split="train", shuffle=False)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    train = _extract_split(
        stage1_model=stage1_model,
        stage1_model_cfg=stage1_model_cfg,
        feature_ablation_cfg=feature_ablation_cfg,
        loader=train_loader,
        stage2_cfg=stage2_cfg,
        device=device,
        max_batches=args.train_batches,
        split="train",
    )
    val = _extract_split(
        stage1_model=stage1_model,
        stage1_model_cfg=stage1_model_cfg,
        feature_ablation_cfg=feature_ablation_cfg,
        loader=val_loader,
        stage2_cfg=stage2_cfg,
        device=device,
        max_batches=args.val_batches,
        split="val",
    )
    metrics = _evaluate(train, val, num_classes=int(args.num_classes))

    features_pt = output_root / "features.pt"
    features_npz = output_root / "features.npz"
    metrics_path = output_root / "metrics.json"
    summary_csv = output_root / "summary.csv"
    torch.save({"train": train, "val": val, "metrics": metrics, "config": config}, features_pt)
    _write_npz(features_npz, train, val)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _write_summary_csv(summary_csv, metrics)
    if not args.no_pca:
        _write_pca_plot(output_root / "figures" / "pca_pooled_repr.png", train, val)

    print(f"[Output] features_pt={features_pt}")
    print(f"[Output] features_npz={features_npz}")
    print(f"[Output] metrics={metrics_path}")
    print(f"[Output] summary={summary_csv}")
    print(
        "[Summary] "
        f"centroid_acc={metrics['nearest_centroid_accuracy']:.4f} "
        f"knn1={metrics['knn1_accuracy']:.4f} "
        f"knn3={metrics['knn3_accuracy']:.4f} "
        f"knn5={metrics['knn5_accuracy']:.4f} "
        f"train_intra={metrics['train_cosine']['intra_class_cosine_mean']:.4f} "
        f"train_inter={metrics['train_cosine']['inter_class_cosine_mean']:.4f} "
        f"val_intra={metrics['val_cosine']['intra_class_cosine_mean']:.4f} "
        f"val_inter={metrics['val_cosine']['inter_class_cosine_mean']:.4f}"
    )
    if not math.isfinite(float(metrics["nearest_centroid_accuracy"])):
        raise FloatingPointError("Non-finite separability metric produced")


if __name__ == "__main__":
    main()
