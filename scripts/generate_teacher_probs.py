"""Generate split-aligned teacher probabilities for D9-TGMS distillation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import build_dataloader, load_config, resolve_device, resolve_existing_path, resolve_path  # noqa: E402
from data.graph_repository import GraphRepositoryReader  # noqa: E402
from evaluation.metrics import classification_report_dict, compute_metrics  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims  # noqa: E402


NUM_CLASSES = 7


def _as_list(values: List[str] | None, appended: List[str] | None) -> List[str]:
    merged: List[str] = []
    if values:
        merged.extend(values)
    if appended:
        merged.extend(appended)
    return merged


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _extract_logits(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    for key in ("logits", "logits_fused", "logits_swin", "aux_logits"):
        value = out.get(key)
        if torch.is_tensor(value) and value.ndim == 2:
            return value
    available = ", ".join(sorted(out.keys()))
    raise KeyError(f"Teacher output has no 2D logits tensor; available keys: {available}")


def _load_teacher(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = load_config(config_path)
    model_cfg = dict(cfg.get("model", {}) or {})
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(resolve_existing_path(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, cfg


def _loader_config(args: argparse.Namespace, teacher_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(teacher_cfg or {})
    paths = dict(cfg.get("paths", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    training = dict(cfg.get("training", {}) or {})
    paths["graph_repo_path"] = str(args.graph_repo_path)
    data["batch_size"] = int(args.batch_size)
    data["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    training["device"] = str(args.device)
    cfg["paths"] = paths
    cfg["data"] = data
    cfg["training"] = training
    return cfg


def _read_split_labels_from_csv(graph_repo_path: str | Path, split: str, max_samples: int | None, progress_interval: int) -> np.ndarray | None:
    repo = Path(graph_repo_path)
    candidates = [
        PROJECT_ROOT / "data" / f"{split}.csv",
        PROJECT_ROOT / "data" / "data" / f"{split}.csv",
        repo.parent / "data" / f"{split}.csv",
        repo.parent.parent / "data" / f"{split}.csv",
    ]
    csv_path = next((path for path in candidates if path.exists()), None)
    if csv_path is None:
        return None
    labels: List[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().strip().split(",")
        try:
            emotion_idx = header.index("emotion")
        except ValueError:
            emotion_idx = 0
        for line_idx, line in enumerate(f):
            if max_samples is not None and len(labels) >= int(max_samples):
                break
            if not line.strip():
                continue
            if emotion_idx == 0:
                label_text = line.split(",", 1)[0]
            else:
                row = next(csv.reader([line]))
                label_text = row[emotion_idx]
            labels.append(int(label_text))
            if progress_interval > 0 and len(labels) % int(progress_interval) == 0:
                print(f"[DummyTeacher] split={split} labels={len(labels)} source=csv")
    print(f"[DummyTeacher] split={split} label_source={csv_path} labels={len(labels)}")
    return np.asarray(labels, dtype=np.int64)


def _read_split_labels_from_chunks(graph_repo_path: str | Path, split: str, max_samples: int | None, progress_interval: int) -> np.ndarray:
    from data.graph_repository import torch_load

    reader = GraphRepositoryReader(graph_repo_path)
    labels: List[int] = []
    for chunk_idx, path in enumerate(reader.chunk_paths(split)):
        chunk = torch_load(path)
        for sample in chunk:
            if max_samples is not None and len(labels) >= int(max_samples):
                break
            labels.append(int(getattr(sample, "label")))
            if progress_interval > 0 and len(labels) % int(progress_interval) == 0:
                print(f"[DummyTeacher] split={split} labels={len(labels)} source=chunk_{chunk_idx:03d}")
        if max_samples is not None and len(labels) >= int(max_samples):
            break
    print(f"[DummyTeacher] split={split} label_source=graph_chunks labels={len(labels)}")
    return np.asarray(labels, dtype=np.int64)


def _dummy_probs_from_labels(labels: np.ndarray, smoothing: float) -> np.ndarray:
    smoothing = float(smoothing)
    if smoothing < 0.0 or smoothing >= 1.0:
        raise ValueError(f"--dummy_smoothing must be in [0,1), got {smoothing}")
    probs = np.full((labels.shape[0], NUM_CLASSES), smoothing / max(NUM_CLASSES - 1, 1), dtype=np.float32)
    probs[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1.0 - smoothing
    return probs


def _run_dummy_label_only(args: argparse.Namespace, splits: List[str]) -> None:
    start = time.perf_counter()
    output_dir = resolve_path(args.output_dir) or Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = GraphRepositoryReader(args.graph_repo_path)
    split_metrics: Dict[str, Any] = {}
    for split in splits:
        split_size = int(reader.split_size(split))
        target_n = split_size if args.max_samples is None else min(split_size, int(args.max_samples))
        labels = _read_split_labels_from_csv(
            args.graph_repo_path,
            split,
            max_samples=target_n,
            progress_interval=int(args.progress_interval),
        )
        if labels is None or labels.shape[0] < target_n:
            labels = _read_split_labels_from_chunks(
                args.graph_repo_path,
                split,
                max_samples=target_n,
                progress_interval=int(args.progress_interval),
            )
        if labels.shape[0] != target_n:
            raise RuntimeError(f"Dummy label count mismatch for split={split}: got={labels.shape[0]} expected={target_n}")
        indices = np.arange(target_n, dtype=np.int64)
        probs = _dummy_probs_from_labels(labels, smoothing=float(args.dummy_smoothing))
        logits = np.log(np.clip(probs, 1e-12, None)).astype(np.float32)
        _validate_arrays(probs, logits, labels, indices)
        np.save(output_dir / f"{split}_probs.npy", probs)
        np.save(output_dir / f"{split}_logits.npy", logits)
        np.save(output_dir / f"{split}_labels.npy", labels)
        np.save(output_dir / f"{split}_indices.npy", indices)
        payload = _metrics_payload(labels, probs, [], [])
        payload["dummy_from_labels"] = True
        payload["partial"] = bool(target_n < split_size)
        payload["split_size"] = split_size
        payload["max_samples"] = args.max_samples
        split_metrics[split] = payload
        print(f"[DummyTeacher] wrote split={split} n={target_n} partial={target_n < split_size}")
    manifest = {
        "graph_repo_path": str(args.graph_repo_path),
        "splits": list(map(str, splits)),
        "teacher_configs": [],
        "teacher_checkpoints": [],
        "ensemble_method": "dummy_from_labels",
        "dummy_from_labels": True,
        "dummy_smoothing": float(args.dummy_smoothing),
        "max_samples": args.max_samples,
        "num_classes": NUM_CLASSES,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "teacher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "teacher_metrics.json").write_text(
        json.dumps(
            {
                "teacher_configs": [],
                "teacher_checkpoints": [],
                "dummy_from_labels": True,
                "splits": split_metrics,
                "elapsed_seconds": manifest["elapsed_seconds"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DummyTeacher] wrote {output_dir} elapsed_seconds={manifest['elapsed_seconds']:.3f}")


def _validate_arrays(probs: np.ndarray, logits: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise RuntimeError(f"Expected probs [N,{NUM_CLASSES}], got {probs.shape}")
    if logits.shape != probs.shape:
        raise RuntimeError(f"Expected logits shape {probs.shape}, got {logits.shape}")
    if labels.shape != (probs.shape[0],):
        raise RuntimeError(f"Expected labels [{probs.shape[0]}], got {labels.shape}")
    if indices.shape != (probs.shape[0],):
        raise RuntimeError(f"Expected indices [{probs.shape[0]}], got {indices.shape}")
    if not np.isfinite(probs).all() or not np.isfinite(logits).all():
        raise RuntimeError("Teacher arrays contain NaN/Inf")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise RuntimeError(f"Teacher probs row sums are not close to 1.0: min={row_sums.min()} max={row_sums.max()}")
    expected = np.arange(probs.shape[0], dtype=np.int64)
    if not np.array_equal(np.sort(indices.astype(np.int64)), expected):
        raise RuntimeError("Teacher indices must cover 0..N-1 exactly once")


def _metrics_payload(labels: np.ndarray, probs: np.ndarray, teacher_configs: List[str], teacher_checkpoints: List[str]) -> Dict[str, Any]:
    preds = probs.argmax(axis=1).astype(np.int64)
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    report = classification_report_dict(labels.tolist(), preds.tolist())
    per_class_f1 = [
        float(report.get(name, {}).get("f1-score", 0.0))
        for name in ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
    ]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "per_class_f1": per_class_f1,
        "pred_distribution": np.bincount(preds, minlength=NUM_CLASSES).astype(int).tolist(),
        "label_distribution": np.bincount(labels.astype(np.int64), minlength=NUM_CLASSES).astype(int).tolist(),
        "num_samples": int(labels.shape[0]),
        "teacher_configs": list(teacher_configs),
        "teacher_checkpoints": list(teacher_checkpoints),
    }


def _run_split(
    *,
    split: str,
    args: argparse.Namespace,
    models: List[torch.nn.Module],
    teacher_configs: List[str],
    teacher_checkpoints: List[str],
    reference_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    loader = build_dataloader(_loader_config(args, reference_cfg), split=split, shuffle=False)
    feature_ablation_cfg = dict(reference_cfg.get("feature_ablation", {}) or {})
    model_cfg = dict(reference_cfg.get("model", {}) or {})
    seen: set[int] = set()
    probs_by_idx: Dict[int, np.ndarray] = {}
    logits_by_idx: Dict[int, np.ndarray] = {}
    labels_by_idx: Dict[int, int] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            if "sample_idx" not in batch:
                raise RuntimeError("Batch has no sample_idx. Update FullGraphDataset/collate before generating teacher probs.")
            indices = batch["sample_idx"].detach().cpu().numpy().astype(np.int64)
            if args.max_samples is not None:
                max_samples = int(args.max_samples)
                keep = indices < max_samples
                if not np.any(keep):
                    break
                if not np.all(keep):
                    keep_t = torch.from_numpy(keep)
                    batch = {
                        key: value[keep_t] if torch.is_tensor(value) and value.shape[:1] == keep_t.shape else value
                        for key, value in batch.items()
                    }
                    indices = indices[keep]
            duplicates = [int(i) for i in indices.tolist() if int(i) in seen]
            if duplicates:
                raise RuntimeError(f"Duplicate sample_idx in split={split}: {duplicates[:5]}")
            batch = move_to_device(batch, device)
            batch = apply_feature_ablation(batch, feature_ablation_cfg)
            if bool(feature_ablation_cfg.get("enabled", False)):
                assert_feature_dims(
                    batch,
                    node_dim=int(model_cfg.get("node_dim", batch["x"].shape[-1])),
                    edge_dim=int(model_cfg.get("edge_dim", batch["edge_attr"].shape[-1])),
                )

            if args.dummy_from_labels:
                labels_t = batch["y"].detach().long()
                probs_t = torch.nn.functional.one_hot(labels_t, num_classes=NUM_CLASSES).float()
                logits_t = probs_t.clamp_min(1e-12).log()
            else:
                logits_tensors = [_extract_logits(model(batch)).detach().float() for model in models]
                stacked_logits = torch.stack(logits_tensors, dim=0)
                if args.ensemble_method == "logit_average":
                    logits_t = stacked_logits.mean(dim=0)
                    probs_t = torch.softmax(logits_t, dim=1)
                elif args.ensemble_method == "probability_average":
                    probs_t = torch.softmax(stacked_logits, dim=2).mean(dim=0)
                    probs_t = probs_t / probs_t.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    logits_t = probs_t.clamp_min(1e-12).log()
                else:
                    raise ValueError(f"Unsupported ensemble_method={args.ensemble_method!r}")
            probs_np = probs_t.detach().cpu().numpy().astype(np.float32)
            logits_np = logits_t.detach().cpu().numpy().astype(np.float32)
            labels_np = batch["y"].detach().cpu().numpy().astype(np.int64)
            for row_idx, sample_idx in enumerate(indices.tolist()):
                sample_idx = int(sample_idx)
                seen.add(sample_idx)
                probs_by_idx[sample_idx] = probs_np[row_idx]
                logits_by_idx[sample_idx] = logits_np[row_idx]
                labels_by_idx[sample_idx] = int(labels_np[row_idx])
            if args.progress_interval > 0 and len(seen) % int(args.progress_interval) == 0:
                print(f"[TeacherProbs] split={split} generated={len(seen)}")
            if args.max_samples is not None and len(seen) >= int(args.max_samples):
                break

    if not seen:
        raise RuntimeError(f"No samples were generated for split={split}")
    if args.max_samples is not None:
        expected_len = min(len(loader.dataset), int(args.max_samples))
    else:
        expected_len = len(loader.dataset) if args.max_batches is None else max(seen) + 1
    expected = set(range(int(expected_len)))
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise RuntimeError(
            f"sample_idx coverage failed for split={split}: missing={missing[:5]} extra={extra[:5]} "
            f"expected_len={expected_len} seen={len(seen)}"
        )

    probs = np.zeros((expected_len, NUM_CLASSES), dtype=np.float32)
    logits = np.zeros((expected_len, NUM_CLASSES), dtype=np.float32)
    labels = np.zeros((expected_len,), dtype=np.int64)
    indices = np.arange(expected_len, dtype=np.int64)
    for sample_idx in range(expected_len):
        probs[sample_idx] = probs_by_idx[sample_idx]
        logits[sample_idx] = logits_by_idx[sample_idx]
        labels[sample_idx] = labels_by_idx[sample_idx]
    _validate_arrays(probs, logits, labels, indices)

    output_dir = resolve_path(args.output_dir) or Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{split}_probs.npy", probs)
    np.save(output_dir / f"{split}_logits.npy", logits)
    np.save(output_dir / f"{split}_labels.npy", labels)
    np.save(output_dir / f"{split}_indices.npy", indices)
    payload = _metrics_payload(labels, probs, teacher_configs, teacher_checkpoints)
    payload["partial"] = bool(args.max_batches is not None)
    payload["max_batches"] = args.max_batches
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_repo_path", required=True)
    parser.add_argument("--split", nargs="+", default=None, help="Split(s) to generate; alias of --splits.")
    parser.add_argument("--splits", nargs="+", default=None, help="Split(s) to generate.")
    parser.add_argument("--teacher_configs", nargs="*", default=None)
    parser.add_argument("--teacher_config", action="append", default=None)
    parser.add_argument("--teacher_checkpoints", nargs="*", default=None)
    parser.add_argument("--teacher_checkpoint", action="append", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=0)
    parser.add_argument("--environment", "--env", default=None)
    parser.add_argument("--ensemble_method", choices=["probability_average", "logit_average"], default="probability_average")
    parser.add_argument("--dummy_from_labels", action="store_true", help="Smoke-only: create one-hot teacher probs from labels.")
    parser.add_argument("--dummy_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    splits = args.splits or args.split or ["train", "val", "test"]
    teacher_configs = _as_list(args.teacher_configs, args.teacher_config)
    teacher_checkpoints = _as_list(args.teacher_checkpoints, args.teacher_checkpoint)
    if args.dummy_from_labels:
        if teacher_configs or teacher_checkpoints:
            raise ValueError("--dummy_from_labels does not use teacher configs/checkpoints")
        _run_dummy_label_only(args, list(map(str, splits)))
        return
    else:
        if len(teacher_configs) != len(teacher_checkpoints):
            raise ValueError(
                f"teacher_configs and teacher_checkpoints length mismatch: "
                f"{len(teacher_configs)} vs {len(teacher_checkpoints)}"
            )
        if not teacher_configs:
            raise ValueError("Provide at least one teacher config/checkpoint, or use --dummy_from_labels for smoke only")
        reference_cfg = load_config(teacher_configs[0], environment=args.environment)
        device = resolve_device(args.device, config=reference_cfg)
        models = []
        for cfg_path, ckpt_path in zip(teacher_configs, teacher_checkpoints):
            model, _ = _load_teacher(cfg_path, ckpt_path, device)
            models.append(model)
    device = resolve_device(args.device, config=reference_cfg)

    split_metrics: Dict[str, Any] = {}
    for split in splits:
        print(f"[TeacherProbs] split={split} dummy={bool(args.dummy_from_labels)} max_batches={args.max_batches}")
        split_metrics[str(split)] = _run_split(
            split=str(split),
            args=args,
            models=models,
            teacher_configs=teacher_configs,
            teacher_checkpoints=teacher_checkpoints,
            reference_cfg=reference_cfg,
            device=device,
        )
    output_dir = resolve_path(args.output_dir) or Path(args.output_dir)
    manifest = {
        "graph_repo_path": str(args.graph_repo_path),
        "splits": list(map(str, splits)),
        "teacher_configs": teacher_configs,
        "teacher_checkpoints": teacher_checkpoints,
        "ensemble_method": str(args.ensemble_method),
        "dummy_from_labels": bool(args.dummy_from_labels),
        "max_batches": args.max_batches,
        "num_classes": NUM_CLASSES,
    }
    (output_dir / "teacher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "teacher_metrics.json").write_text(
        json.dumps(
            {
                "teacher_configs": teacher_configs,
                "teacher_checkpoints": teacher_checkpoints,
                "dummy_from_labels": bool(args.dummy_from_labels),
                "splits": split_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[TeacherProbs] wrote {output_dir}")


if __name__ == "__main__":
    main()
