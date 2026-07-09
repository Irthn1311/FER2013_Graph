"""Train/evaluate D17 Evidence-Preserving Pixel Graph models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d17.data.collate import EPPBatch, collate_epp_graphs
from d17.data.epp_dataset import EPPPixelDataset
from d17.models.epp_gnn import EPPGNN


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def read_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_csv(path: Path, row: Dict[str, Any], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(text: str | None) -> torch.device:
    requested = text or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_dataset(cfg: Dict[str, Any], split: str, max_samples: int | None = None) -> EPPPixelDataset:
    data_cfg = cfg.get("data", {}) or {}
    return EPPPixelDataset(
        prior_dir=data_cfg.get("prior_dir", "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"),
        split=split,
        graph=cfg.get("graph", {}) or {},
        max_samples=max_samples,
    )


def loader_kwargs(cfg: Dict[str, Any], shuffle: bool) -> Dict[str, Any]:
    train_cfg = cfg.get("training", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    num_workers = int(train_cfg.get("num_workers", data_cfg.get("num_workers", 0)) or 0)
    kwargs: Dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", data_cfg.get("batch_size", 16)) or 16),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "collate_fn": collate_epp_graphs,
        "pin_memory": bool(train_cfg.get("pin_memory", data_cfg.get("pin_memory", False))),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", data_cfg.get("persistent_workers", False)))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", data_cfg.get("prefetch_factor", 2)) or 2)
    return kwargs


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int = 7) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_predictions(y_true: List[int], y_pred: List[int], loss_sum: float, count: int) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, len(CLASS_NAMES))
    total = int(cm.sum())
    acc = float(np.trace(cm) / max(total, 1))
    f1s = []
    for i in range(len(CLASS_NAMES)):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
    return {
        "loss": float(loss_sum / max(count, 1)),
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": cm,
        "per_class_f1": f1s,
    }


def detected_fallback_metrics(y_true: List[int], y_pred: List[int], detected: List[bool]) -> List[Dict[str, Any]]:
    rows = []
    arr_d = np.asarray(detected, dtype=bool)
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    for name, mask in [("detected", arr_d), ("fallback", ~arr_d)]:
        if not bool(mask.any()):
            rows.append({"group": name, "total": 0, "accuracy": math.nan, "macro_f1": math.nan})
            continue
        m = metrics_from_predictions(y_t[mask].tolist(), y_p[mask].tolist(), 0.0, 1)
        rows.append({"group": name, "total": int(mask.sum()), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    return rows


def apply_dropedge(batch: EPPBatch, p: float) -> EPPBatch:
    p = float(p)
    if p <= 0.0 or batch.edge_index_cat.size(1) <= 1:
        return batch
    keep = torch.rand((batch.edge_index_cat.size(1),), device=batch.edge_index_cat.device) >= p
    if not bool(keep.any()):
        keep[int(torch.randint(0, keep.numel(), (1,), device=keep.device).item())] = True
    return EPPBatch(
        x_cat=batch.x_cat,
        edge_index_cat=batch.edge_index_cat[:, keep],
        edge_attr_cat=batch.edge_attr_cat[keep],
        batch_index=batch.batch_index,
        ptr=batch.ptr,
        y=batch.y,
        sample_index=batch.sample_index,
        pos_cat=batch.pos_cat,
        detected=batch.detected,
        landmark_missing_flag=batch.landmark_missing_flag,
        image_48=batch.image_48,
        local_edge_count=batch.local_edge_count,
        knn_edge_count=batch.knn_edge_count,
        total_edge_count=batch.total_edge_count,
        node_feature_names=batch.node_feature_names,
        edge_feature_names=batch.edge_feature_names,
    )


def train_one_epoch(
    model: EPPGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: torch.nn.Module,
    drop_edge_p: float,
    progress_interval: int,
    epoch: int,
) -> Dict[str, Any]:
    model.train()
    loss_sum = 0.0
    count = 0
    node_counts: List[float] = []
    edge_counts: List[float] = []
    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        train_batch = apply_dropedge(batch, drop_edge_p)
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_batch)["logits"]
        loss = loss_fn(logits, train_batch.y)
        loss.backward()
        optimizer.step()
        bs = int(train_batch.y.numel())
        loss_sum += float(loss.detach().item()) * bs
        count += bs
        node_counts.extend(((batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()).tolist())
        edge_counts.extend(batch.total_edge_count.detach().cpu().numpy().tolist())
        if progress_interval > 0 and (batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == len(loader)):
            print(
                json.dumps(
                    {
                        "event": "d17_train_progress",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": len(loader),
                        "avg_loss_so_far": loss_sum / max(count, 1),
                    }
                ),
                flush=True,
            )
    return {
        "train_loss": loss_sum / max(count, 1),
        "node_count_mean": float(np.mean(node_counts)) if node_counts else math.nan,
        "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else math.nan,
    }


@torch.no_grad()
def evaluate(model: EPPGNN, loader: DataLoader, device: torch.device, loss_fn: torch.nn.Module) -> tuple[Dict[str, Any], Dict[str, Any]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    detected: List[bool] = []
    sample_index: List[int] = []
    loss_sum = 0.0
    count = 0
    node_counts: List[float] = []
    local_counts: List[float] = []
    knn_counts: List[float] = []
    edge_counts: List[float] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)["logits"]
        loss = loss_fn(logits, batch.y)
        pred = logits.argmax(dim=1)
        bs = int(batch.y.numel())
        loss_sum += float(loss.item()) * bs
        count += bs
        y_true.extend(batch.y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        detected.extend(batch.detected.detach().cpu().tolist())
        sample_index.extend(batch.sample_index.detach().cpu().tolist())
        node_counts.extend(((batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy()).tolist())
        local_counts.extend(batch.local_edge_count.detach().cpu().numpy().tolist())
        knn_counts.extend(batch.knn_edge_count.detach().cpu().numpy().tolist())
        edge_counts.extend(batch.total_edge_count.detach().cpu().numpy().tolist())
    row = metrics_from_predictions(y_true, y_pred, loss_sum, count)
    row.update(
        {
            "node_count_mean": float(np.mean(node_counts)) if node_counts else math.nan,
            "local_edge_count_mean": float(np.mean(local_counts)) if local_counts else math.nan,
            "knn_edge_count_mean": float(np.mean(knn_counts)) if knn_counts else math.nan,
            "edge_count_mean": float(np.mean(edge_counts)) if edge_counts else math.nan,
        }
    )
    detail = {"y_true": y_true, "y_pred": y_pred, "detected": detected, "sample_index": sample_index}
    return row, detail


def save_checkpoint(path: Path, model: EPPGNN, optimizer: torch.optim.Optimizer, epoch: int, best_score: float, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "config": cfg,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_checkpoint(path: Path, model: EPPGNN, optimizer: torch.optim.Optimizer | None = None, device: torch.device | str = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def write_eval_outputs(output_dir: Path, prefix: str, row: Dict[str, Any], detail: Dict[str, Any]) -> None:
    cm = row["confusion_matrix"]
    with (output_dir / f"{prefix}confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + [int(x) for x in cm[i].tolist()])
    with (output_dir / f"{prefix}per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "support", "f1"])
        writer.writeheader()
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow({"class_id": i, "class_name": name, "support": int(cm[i].sum()), "f1": float(row["per_class_f1"][i])})
    with (output_dir / f"{prefix}detected_vs_fallback_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "total", "accuracy", "macro_f1"])
        writer.writeheader()
        for r in detected_fallback_metrics(detail["y_true"], detail["y_pred"], detail["detected"]):
            writer.writerow(r)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"D17 confusion matrix, acc: {row['accuracy']*100:.2f}%")
        ax.set_xlabel("Pred label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        for i in range(cm.shape[0]):
            denom = max(int(cm[i].sum()), 1)
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i,j]}\\n{cm[i,j]/denom*100:.1f}%", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}confusion_matrix.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[D17] skipped confusion png: {exc}", flush=True)


def write_graph_schema(output_dir: Path, cfg: Dict[str, Any], eval_row: Dict[str, Any]) -> None:
    graph_cfg = cfg.get("graph", {}) or {}
    payload = {
        "node_support_mode": graph_cfg.get("node_support_mode", "detail_topN_knn"),
        "target_node_count": int(graph_cfg.get("target_node_count", 1800)),
        "actual_node_count_mean": eval_row.get("node_count_mean"),
        "local_edge_count_mean": eval_row.get("local_edge_count_mean"),
        "knn_edge_count_mean": eval_row.get("knn_edge_count_mean"),
        "total_edge_count_mean": eval_row.get("edge_count_mean"),
        "node_feature_names": [
            "intensity",
            "gx",
            "gy",
            "x_norm",
            "y_norm",
            "grad_mag",
            "local_mean_3x3",
            "local_std_3x3",
            "laplacian_abs",
            "center_surround",
        ],
        "edge_feature_names": [
            "dx",
            "dy",
            "spatial_dist",
            "abs_intensity_diff",
            "abs_grad_mag_diff",
            "abs_laplacian_diff",
        ],
        "uses_face_mask_for_node_selection": False,
        "uses_part_prior_for_node_selection": False,
        "uses_anchor_nodes": False,
        "uses_log_prior_bias": False,
    }
    write_json(output_dir / "graph_schema.json", payload)
    write_json(output_dir / "feature_schema.json", {"node_feature_names": payload["node_feature_names"], "edge_feature_names": payload["edge_feature_names"]})


def run_train(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = read_config(args.config)
    train_cfg = cfg.get("training", {}) or {}
    seed = int(train_cfg.get("seed", cfg.get("seed", 42)) or 42)
    set_seed(seed)
    device = resolve_device(args.device or train_cfg.get("device"))
    output_dir = Path(args.output_dir or cfg.get("output_dir") or Path("outputs/d17_runs/ofix15") / str(cfg.get("run_name", Path(args.config).stem)))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", cfg)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    max_samples = args.max_samples
    train_ds = build_dataset(cfg, "train", max_samples=max_samples)
    val_ds = build_dataset(cfg, "val", max_samples=max_samples)
    test_ds = build_dataset(cfg, "test", max_samples=max_samples)
    train_loader = DataLoader(train_ds, **loader_kwargs(cfg, shuffle=True))
    val_loader = DataLoader(val_ds, **loader_kwargs(cfg, shuffle=False))
    test_loader = DataLoader(test_ds, **loader_kwargs(cfg, shuffle=False))

    first_batch = next(iter(DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_epp_graphs)))
    model = EPPGNN.from_config(cfg, input_dim=int(first_batch.x_cat.size(1)), edge_attr_dim=int(first_batch.edge_attr_cat.size(1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-3)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float((train_cfg.get("scheduler", {}) or {}).get("factor", 0.5)),
        patience=int((train_cfg.get("scheduler", {}) or {}).get("patience", 5)),
        min_lr=float((train_cfg.get("scheduler", {}) or {}).get("min_lr", 3e-5)),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    best_score = -math.inf
    best_epoch = 0
    best_val_loss = math.inf
    best_val_loss_epoch = 0
    patience = int((train_cfg.get("early_stopping", {}) or {}).get("patience", 15))
    min_epochs = int((train_cfg.get("early_stopping", {}) or {}).get("min_epochs_before_stop", 30))
    epochs_wo = 0
    max_epochs = int(train_cfg.get("max_epochs", 90))
    drop_edge_p = float(train_cfg.get("drop_edge_p", 0.0) or 0.0)
    fields = [
        "epoch",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "best_val_macro_f1",
        "lr",
        "node_count_mean",
        "local_edge_count_mean",
        "knn_edge_count_mean",
        "edge_count_mean",
        "drop_edge_p",
    ]
    for epoch in range(1, max_epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn,
            drop_edge_p=drop_edge_p,
            progress_interval=int(train_cfg.get("progress_interval_batches", 500) or 500),
            epoch=epoch,
        )
        train_eval, _ = evaluate(model, train_loader, device, loss_fn)
        val_row, _ = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(float(val_row["loss"]))
        improved = float(val_row["macro_f1"]) > best_score
        if improved:
            best_score = float(val_row["macro_f1"])
            best_epoch = epoch
            epochs_wo = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_score, cfg)
        else:
            epochs_wo += 1
        if float(val_row["loss"]) < best_val_loss:
            best_val_loss = float(val_row["loss"])
            best_val_loss_epoch = epoch
            save_checkpoint(output_dir / "checkpoints" / "best_val_loss.pt", model, optimizer, epoch, best_score, cfg)
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best_score, cfg)
        log_row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "train_eval_loss": train_eval["loss"],
            "train_accuracy": train_eval["accuracy"],
            "train_macro_f1": train_eval["macro_f1"],
            "val_loss": val_row["loss"],
            "val_accuracy": val_row["accuracy"],
            "val_macro_f1": val_row["macro_f1"],
            "best_val_macro_f1": best_score,
            "lr": optimizer.param_groups[0]["lr"],
            "node_count_mean": val_row["node_count_mean"],
            "local_edge_count_mean": val_row["local_edge_count_mean"],
            "knn_edge_count_mean": val_row["knn_edge_count_mean"],
            "edge_count_mean": val_row["edge_count_mean"],
            "drop_edge_p": drop_edge_p,
        }
        append_csv(output_dir / "train_log.csv", log_row, fields)
        print(json.dumps({"event": "d17_epoch", **log_row, "best_epoch": best_epoch}), flush=True)
        if epoch >= min_epochs and epochs_wo >= patience:
            print(json.dumps({"early_stopped": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    # Evaluate best and last.
    load_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer=None, device=device)
    test_row, test_detail = evaluate(model, test_loader, device, loss_fn)
    write_eval_outputs(output_dir, "", test_row, test_detail)
    load_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer=None, device=device)
    last_row, last_detail = evaluate(model, test_loader, device, loss_fn)
    write_eval_outputs(output_dir, "last_", last_row, last_detail)
    write_graph_schema(output_dir, cfg, test_row)
    summary = {
        "output_dir": str(output_dir),
        "run_name": cfg.get("run_name", output_dir.name),
        "device": str(device),
        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "test_accuracy": test_row["accuracy"],
        "test_macro_f1": test_row["macro_f1"],
        "last_test_accuracy": last_row["accuracy"],
        "last_test_macro_f1": last_row["macro_f1"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }
    write_json(output_dir / "d17_train_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D17 EPP-GNN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run_train(parse_args())


if __name__ == "__main__":
    main()

