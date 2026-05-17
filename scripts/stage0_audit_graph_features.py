"""Stage 0 graph/data/feature audit for FER full-pixel graph artifacts.

This script is intentionally read-only for graph artifacts and training config:
it loads existing train/val/test graph chunks, writes audit outputs, and keeps
going when optional checks cannot run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_checkpoint_model, load_config, resolve_path
from data.graph_config import GraphConfig
from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.labels import EMOTION_NAMES, NUM_CLASSES

SPLITS = ("train", "val", "test")
PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)
UNKNOWN = "UNKNOWN"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["note"]
        rows = [{"note": "NO_ROWS"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def fmt(value: Any) -> str:
    value = safe_float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


class Notes:
    def __init__(self) -> None:
        self.items: List[str] = []

    def add(self, text: str) -> None:
        print(f"[Stage0] {text}")
        self.items.append(text)


class FeatureAccumulator:
    def __init__(
        self,
        names: Sequence[str],
        reservoir_size: int,
        seed: int,
    ) -> None:
        self.names = list(names)
        self.dim = len(self.names)
        self.count = np.zeros(self.dim, dtype=np.int64)
        self.total = np.zeros(self.dim, dtype=np.int64)
        self.nan = np.zeros(self.dim, dtype=np.int64)
        self.inf = np.zeros(self.dim, dtype=np.int64)
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)
        self.reservoir = [np.empty((0,), dtype=np.float32) for _ in range(self.dim)]
        self.reservoir_size = int(reservoir_size)
        self.rng = np.random.default_rng(seed)

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().cpu().reshape(-1, values.shape[-1]).to(torch.float64)
        if values.shape[-1] != self.dim:
            raise ValueError(f"feature dim mismatch: expected {self.dim}, got {values.shape[-1]}")
        finite = torch.isfinite(values)
        self.total += int(values.shape[0])
        self.nan += torch.isnan(values).sum(0).numpy().astype(np.int64)
        self.inf += torch.isinf(values).sum(0).numpy().astype(np.int64)
        self.count += finite.sum(0).numpy().astype(np.int64)
        clean = torch.where(finite, values, torch.zeros_like(values))
        self.sum += clean.sum(0).numpy()
        self.sumsq += (clean * clean).sum(0).numpy()
        mins = torch.where(finite, values, torch.full_like(values, float("inf"))).min(0).values.numpy()
        maxs = torch.where(finite, values, torch.full_like(values, float("-inf"))).max(0).values.numpy()
        self.min = np.minimum(self.min, mins)
        self.max = np.maximum(self.max, maxs)
        self._sample(values.to(torch.float32), finite)

    def _sample(self, values: torch.Tensor, finite: torch.Tensor) -> None:
        for idx in range(self.dim):
            col = values[:, idx][finite[:, idx]]
            if col.numel() == 0:
                continue
            if col.numel() > 5000:
                take = self.rng.choice(int(col.numel()), size=5000, replace=False)
                arr = col[take].numpy()
            else:
                arr = col.numpy()
            merged = np.concatenate([self.reservoir[idx], arr.astype(np.float32, copy=False)])
            if merged.size > self.reservoir_size:
                keep = self.rng.choice(merged.size, size=self.reservoir_size, replace=False)
                merged = merged[keep]
            self.reservoir[idx] = merged

    def rows(self, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, name in enumerate(self.names):
            count = int(self.count[idx])
            mean = self.sum[idx] / max(count, 1)
            var = max(self.sumsq[idx] / max(count, 1) - mean * mean, 0.0)
            pct = percentile_values(self.reservoir[idx])
            rows.append(
                {
                    **scope,
                    "feature_index": idx,
                    "feature": name,
                    "count": count,
                    "total_count": int(self.total[idx]),
                    "nan_count": int(self.nan[idx]),
                    "inf_count": int(self.inf[idx]),
                    "min": self.min[idx] if np.isfinite(self.min[idx]) else float("nan"),
                    "max": self.max[idx] if np.isfinite(self.max[idx]) else float("nan"),
                    "mean": mean,
                    "std": math.sqrt(var),
                    **pct,
                    "reservoir_count": int(self.reservoir[idx].size),
                }
            )
        return rows


class RowReservoir:
    def __init__(self, dim: int, size: int, seed: int) -> None:
        self.dim = int(dim)
        self.size = int(size)
        self.rows = np.empty((0, self.dim), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().cpu().reshape(-1, values.shape[-1])
        if values.shape[-1] != self.dim:
            return
        finite = torch.isfinite(values).all(1)
        values = values[finite]
        if values.numel() == 0:
            return
        if values.shape[0] > 5000:
            take = self.rng.choice(int(values.shape[0]), size=5000, replace=False)
            values = values[take]
        merged = np.concatenate([self.rows, values.float().numpy()], axis=0)
        if merged.shape[0] > self.size:
            keep = self.rng.choice(merged.shape[0], size=self.size, replace=False)
            merged = merged[keep]
        self.rows = merged

    def corr(self) -> np.ndarray:
        if self.rows.shape[0] < 2:
            return np.full((self.dim, self.dim), np.nan)
        return np.corrcoef(self.rows.astype(np.float64), rowvar=False)


def percentile_values(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {f"p{p}": float("nan") for p in PERCENTILES}
    pct = np.percentile(values.astype(np.float64), PERCENTILES)
    return {f"p{p}": float(v) for p, v in zip(PERCENTILES, pct)}


def corr_rows(kind: str, names: Sequence[str], corr: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            rows.append({"kind": kind, "feature_a": a, "feature_b": b, "correlation": corr[i, j]})
    return rows


def unknown_names(prefix: str, dim: int) -> List[str]:
    return [f"{UNKNOWN}_{prefix}_{idx}" for idx in range(int(dim))]


def normalize_names(names: Optional[Sequence[Any]], dim: int, prefix: str) -> Optional[List[str]]:
    if not names:
        return None
    out = [str(v) for v in names]
    if len(out) != int(dim):
        return None
    return out


def infer_feature_names(
    reader: GraphRepositoryReader,
    config: Dict[str, Any],
    sample: Any,
    node_dim: int,
    edge_dim: int,
    notes: Notes,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    manifest = reader.manifest if isinstance(reader.manifest, dict) else {}
    manifest_cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    cfg_graph = config.get("graph", {}) if isinstance(config.get("graph", {}), dict) else {}
    sources: Dict[str, str] = {}

    node_candidates: List[Tuple[str, Optional[Sequence[Any]]]] = [
        ("manifest.node_feature_names", manifest.get("node_feature_names")),
        ("manifest.config.node_feature_names", manifest_cfg.get("node_feature_names")),
        ("sample.node_feature_names", getattr(sample, "node_feature_names", None)),
        ("config.graph.node_feature_names", cfg_graph.get("node_feature_names")),
    ]
    try:
        graph_cfg_names = GraphConfig.from_dict(cfg_graph).node_feature_names
        node_candidates.append(("GraphConfig.from_dict(config.graph).node_feature_names", graph_cfg_names))
    except Exception as exc:
        notes.add(f"Could not trace node names through GraphConfig: {exc}")

    node_names = None
    for source, values in node_candidates:
        node_names = normalize_names(values, node_dim, "node")
        if node_names:
            sources["node_feature_names"] = source
            break
    if node_names is None:
        node_names = unknown_names("node", node_dim)
        sources["node_feature_names"] = "UNKNOWN"
        notes.add("node_feature_names missing or wrong length; wrote UNKNOWN names instead of guessing.")

    shared = reader.load_shared()
    sample_dynamic = getattr(sample, "dynamic_feature_names", None)
    edge_candidates: List[Tuple[str, Optional[Sequence[Any]]]] = [
        ("manifest.edge_feature_names", manifest.get("edge_feature_names")),
        (
            "manifest.config.edge_static_feature_names+edge_dynamic_feature_names",
            list(manifest_cfg.get("edge_static_feature_names") or [])
            + list(manifest_cfg.get("edge_dynamic_feature_names") or []),
        ),
        (
            "shared.static_feature_names+sample.dynamic_feature_names",
            list(getattr(shared, "static_feature_names", []) or []) + list(sample_dynamic or []),
        ),
        (
            "config.graph.edge_static_feature_names+edge_dynamic_feature_names",
            list(cfg_graph.get("edge_static_feature_names") or []) + list(cfg_graph.get("edge_dynamic_feature_names") or []),
        ),
    ]
    try:
        graph_cfg = GraphConfig.from_dict(cfg_graph)
        edge_candidates.append(
            (
                "GraphConfig.from_dict(config.graph).edge_static_feature_names+edge_dynamic_feature_names",
                list(graph_cfg.edge_static_feature_names) + list(graph_cfg.edge_dynamic_feature_names),
            )
        )
    except Exception as exc:
        notes.add(f"Could not trace edge names through GraphConfig: {exc}")

    edge_names = None
    for source, values in edge_candidates:
        edge_names = normalize_names(values, edge_dim, "edge")
        if edge_names:
            sources["edge_feature_names"] = source
            break
    if edge_names is None:
        edge_names = unknown_names("edge", edge_dim)
        sources["edge_feature_names"] = "UNKNOWN"
        notes.add("edge_feature_names missing or wrong length; wrote UNKNOWN names instead of guessing.")
    return node_names, edge_names, sources


def resolve_graph_repo(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    candidates: List[Path] = []
    if args.graph_repo_path:
        candidates.append(Path(args.graph_repo_path))
    cfg_path = config.get("paths", {}).get("graph_repo_path") if isinstance(config.get("paths", {}), dict) else None
    if cfg_path:
        resolved = resolve_path(cfg_path)
        if resolved:
            candidates.append(resolved)
    candidates.extend(
        [
            PROJECT_ROOT / "artifacts" / "graph-repo" / "graph_repo",
            PROJECT_ROOT / "artifacts" / "graph_repo",
            PROJECT_ROOT / "artifacts" / "graph_repo_local",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No graph repo found. Checked: {[str(c) for c in candidates]}")


def pick_sample(reader: GraphRepositoryReader) -> Any:
    for split in SPLITS:
        paths = reader.chunk_paths(split)
        if paths:
            chunk = reader.load_chunk(split, 0)
            if chunk:
                return chunk[0]
    raise FileNotFoundError("No graph samples found in train/val/test chunks.")


def iter_limited(reader: GraphRepositoryReader, split: str, max_samples: Optional[int]) -> Iterable[Any]:
    seen = 0
    for sample in reader.iter_split(split):
        yield sample
        seen += 1
        if max_samples is not None and seen >= int(max_samples):
            break


def class_name(label: int) -> str:
    return EMOTION_NAMES[label] if 0 <= int(label) < len(EMOTION_NAMES) else f"label_{label}"


def add_schema_split(schema: Dict[str, Any], split: str, resolved: Any, sample: Any) -> None:
    info = schema["splits"].setdefault(
        split,
        {
            "num_samples_scanned": 0,
            "class_counts": Counter(),
            "node_shapes": Counter(),
            "edge_attr_shapes": Counter(),
            "num_nodes": Counter(),
            "num_edges": Counter(),
            "node_feature_name_variants": Counter(),
            "edge_feature_name_variants": Counter(),
        },
    )
    info["num_samples_scanned"] += 1
    info["class_counts"][int(resolved.label)] += 1
    info["node_shapes"][tuple(resolved.node_features.shape)] += 1
    info["edge_attr_shapes"][tuple(resolved.edge_attr.shape)] += 1
    info["num_nodes"][int(resolved.node_features.shape[0])] += 1
    info["num_edges"][int(resolved.edge_attr.shape[0])] += 1
    info["node_feature_name_variants"][tuple(getattr(sample, "node_feature_names", []) or [])] += 1
    info["edge_feature_name_variants"][tuple(resolved.edge_feature_names or [])] += 1


def finalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(schema)
    splits = {}
    for split, info in schema["splits"].items():
        splits[split] = {
            "num_samples_scanned": info["num_samples_scanned"],
            "class_counts": {str(k): int(v) for k, v in sorted(info["class_counts"].items())},
            "class_counts_named": {class_name(k): int(v) for k, v in sorted(info["class_counts"].items())},
            "node_shapes": {str(k): int(v) for k, v in info["node_shapes"].items()},
            "edge_attr_shapes": {str(k): int(v) for k, v in info["edge_attr_shapes"].items()},
            "num_nodes": {str(k): int(v) for k, v in info["num_nodes"].items()},
            "num_edges": {str(k): int(v) for k, v in info["num_edges"].items()},
            "node_feature_name_variants": {str(k): int(v) for k, v in info["node_feature_name_variants"].items()},
            "edge_feature_name_variants": {str(k): int(v) for k, v in info["edge_feature_name_variants"].items()},
        }
    out["splits"] = splits
    signatures = []
    for split, info in splits.items():
        signatures.append((split, tuple(info["node_shapes"]), tuple(info["edge_attr_shapes"]), tuple(info["num_nodes"]), tuple(info["num_edges"])))
    out["schema_consistent_train_val_test"] = len({sig[1:] for sig in signatures}) <= 1
    return out


def topology_rows(edge_index: torch.Tensor, num_nodes: int, split: str = "shared") -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    edge_index = edge_index.detach().cpu().long()
    src, dst = edge_index[0], edge_index[1]
    invalid = (src < 0) | (src >= num_nodes) | (dst < 0) | (dst >= num_nodes)
    self_loop = src == dst
    pairs = list(zip(src.tolist(), dst.tolist()))
    duplicate_count = len(pairs) - len(set(pairs))
    indeg = torch.bincount(dst.clamp(0, max(num_nodes - 1, 0)), minlength=num_nodes).float()
    outdeg = torch.bincount(src.clamp(0, max(num_nodes - 1, 0)), minlength=num_nodes).float()
    degree = indeg + outdeg
    rows = [
        {
            "split": split,
            "metric": "num_nodes",
            "value": int(num_nodes),
        },
        {"split": split, "metric": "num_edges", "value": int(edge_index.shape[1])},
        {"split": split, "metric": "invalid_edges", "value": int(invalid.sum().item())},
        {"split": split, "metric": "duplicate_edges", "value": int(duplicate_count)},
        {"split": split, "metric": "self_loops", "value": int(self_loop.sum().item())},
    ]
    for name, values in (("in_degree", indeg), ("out_degree", outdeg), ("total_degree", degree)):
        arr = values.numpy()
        rows.append(
            {
                "split": split,
                "metric": name,
                "min": float(arr.min()) if arr.size else float("nan"),
                "max": float(arr.max()) if arr.size else float("nan"),
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "std": float(arr.std()) if arr.size else float("nan"),
                **percentile_values(arr.astype(np.float32)),
            }
        )
    return rows, degree


def pooled_features(node: torch.Tensor, edge: torch.Tensor, node_names: Sequence[str], edge_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    pieces: List[np.ndarray] = []
    names: List[str] = []
    for prefix, tensor, feature_names in (("node", node, node_names), ("edge", edge, edge_names)):
        arr = tensor.detach().cpu().float().numpy()
        stats = {
            "mean": np.nanmean(arr, axis=0),
            "std": np.nanstd(arr, axis=0),
            "min": np.nanmin(arr, axis=0),
            "max": np.nanmax(arr, axis=0),
        }
        for stat_name, values in stats.items():
            pieces.append(values.astype(np.float32, copy=False))
            names.extend([f"{prefix}_{name}_{stat_name}" for name in feature_names])
    return np.concatenate(pieces, axis=0), names


def simple_probe(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    notes: Notes,
    variant: str,
    feature_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] in {"val", "test"}]
    if len(train) < 2 or len(val) < 1:
        return {"variant": variant, "status": "SKIPPED", "reason": "not enough train/val rows"}
    x_train = np.stack([r["features"] for r in train])
    y_train = np.asarray([r["label"] for r in train], dtype=np.int64)
    x_val = np.stack([r["features"] for r in val])
    y_val = np.asarray([r["label"] for r in val], dtype=np.int64)
    if feature_mask is not None:
        x_train = x_train[:, feature_mask]
        x_val = x_val[:, feature_mask]
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        clf = LogisticRegression(max_iter=400, class_weight="balanced", random_state=42)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)
        cm = confusion_matrix(y_val, pred, labels=list(range(NUM_CLASSES)))
        write_csv(
            output_dir / "figures" / f"confusion_matrix_{variant}.csv",
            [
                {"true_label": i, "pred_label": j, "count": int(cm[i, j])}
                for i in range(NUM_CLASSES)
                for j in range(NUM_CLASSES)
            ],
        )
        return {
            "variant": variant,
            "status": "OK",
            "model": "LogisticRegression",
            "num_train": len(train),
            "num_eval": len(val),
            "num_features": int(x_train.shape[1]),
            "accuracy": float(accuracy_score(y_val, pred)),
            "macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        }
    except Exception as exc:
        notes.add(f"Probe {variant} skipped: {exc}")
        return {"variant": variant, "status": "SKIPPED", "reason": str(exc)}


def transformed_pooled_rows(
    rows: Sequence[Dict[str, Any]],
    edge_mask: np.ndarray,
    mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    features = [np.array(row["features"], copy=True) for row in rows]
    if mode == "edge_zeroed":
        for arr in features:
            arr[edge_mask] = 0.0
    elif mode == "edge_shuffled":
        if features:
            edge_values = np.stack([arr[edge_mask] for arr in features], axis=0)
            perm = rng.permutation(edge_values.shape[0])
            shuffled = edge_values[perm]
            for arr, edge_arr in zip(features, shuffled):
                arr[edge_mask] = edge_arr
    else:
        raise ValueError(f"Unknown pooled transform mode: {mode}")
    for row, arr in zip(rows, features):
        new_row = dict(row)
        new_row["features"] = arr
        out.append(new_row)
    return out


def save_heatmap(array: np.ndarray, path: Path, title: str, notes: Notes) -> None:
    try:
        import matplotlib.pyplot as plt

        ensure_dir(path.parent)
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        im = ax.imshow(array, cmap="viridis")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception as exc:
        notes.add(f"Could not save heatmap {path.name}: {exc}")


def save_corr_heatmap(corr: np.ndarray, names: Sequence[str], path: Path, title: str, notes: Notes) -> None:
    try:
        import matplotlib.pyplot as plt

        ensure_dir(path.parent)
        fig, ax = plt.subplots(figsize=(max(4, len(names) * 0.55), max(3.5, len(names) * 0.45)))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        notes.add(f"Could not save correlation heatmap {path.name}: {exc}")


def edge_to_node_map(edge_index: torch.Tensor, edge_values: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    values = edge_values.detach().cpu().float()
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros((num_nodes,), dtype=torch.float32)
    cnt = torch.zeros((num_nodes,), dtype=torch.float32)
    out.index_add_(0, src, values)
    out.index_add_(0, dst, values)
    one = torch.ones_like(values)
    cnt.index_add_(0, src, one)
    cnt.index_add_(0, dst, one)
    return out / cnt.clamp_min(1.0)


def make_figures(
    samples_by_class: Dict[int, List[Any]],
    edge_index: torch.Tensor,
    node_names: Sequence[str],
    edge_names: Sequence[str],
    height: int,
    width: int,
    output_dir: Path,
    degree: torch.Tensor,
    notes: Notes,
) -> None:
    figures = ensure_dir(output_dir / "figures")
    if degree.numel() == height * width:
        save_heatmap(degree.reshape(height, width).numpy(), figures / "topology_degree_map.png", "degree map", notes)
    for label, samples in samples_by_class.items():
        cname = class_name(label).lower()
        for sample_idx, resolved in enumerate(samples):
            for feat_idx, feat_name in enumerate(node_names):
                arr = resolved.node_features[:, feat_idx].detach().cpu().reshape(height, width).numpy()
                save_heatmap(
                    arr,
                    figures / "node_feature_heatmaps" / f"class_{label}_{cname}_sample_{sample_idx}_node_{feat_idx}_{feat_name}.png",
                    f"{cname} {feat_name}",
                    notes,
                )
            for feat_idx, feat_name in enumerate(edge_names):
                node_map = edge_to_node_map(edge_index, resolved.edge_attr[:, feat_idx], height * width)
                save_heatmap(
                    node_map.reshape(height, width).numpy(),
                    figures / "edge_to_node_heatmaps" / f"class_{label}_{cname}_sample_{sample_idx}_edge_{feat_idx}_{feat_name}.png",
                    f"{cname} edge {feat_name}",
                    notes,
                )


def gate_diagnostics(
    args: argparse.Namespace,
    config: Dict[str, Any],
    probe_batch: Optional[Dict[str, torch.Tensor]],
    output_dir: Path,
    edge_names: Sequence[str],
    notes: Notes,
) -> Dict[str, Any]:
    if not args.checkpoint:
        reason = "No --checkpoint provided; checkpoint/encoder gate diagnostics skipped."
        notes.add(reason)
        return {"status": "SKIPPED", "reason": reason}
    if probe_batch is None:
        reason = "No probe batch available for forward gate diagnostics."
        notes.add(reason)
        return {"status": "SKIPPED", "reason": reason}
    try:
        model, device, _ = load_checkpoint_model(config, args.checkpoint)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in probe_batch.items()}
        with torch.no_grad():
            out = model(batch)
        rows: List[Dict[str, Any]] = []
        output_items = out.items() if isinstance(out, dict) else []
        for key, value in output_items:
            if "gate" not in str(key).lower() or not torch.is_tensor(value):
                continue
            arr = value.detach().float().cpu().reshape(-1).numpy()
            if arr.size == 0:
                continue
            rows.append(
                {
                    "gate_key": key,
                    "count": int(arr.size),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    **percentile_values(arr.astype(np.float32)),
                }
            )
        if not rows:
            reason = "Forward succeeded but no tensor output key containing 'gate' was found."
            notes.add(reason)
            return {"status": "SKIPPED", "reason": reason}
        write_csv(output_dir / "gate_diagnostics.csv", rows)
        return {"status": "OK", "gate_keys": [r["gate_key"] for r in rows]}
    except Exception as exc:
        reason = f"Gate diagnostics failed and were skipped: {exc}"
        notes.add(reason)
        return {"status": "SKIPPED", "reason": reason}


def collate_probe_batch(samples: Sequence[Any]) -> Optional[Dict[str, torch.Tensor]]:
    if not samples:
        return None
    x = torch.stack([s.node_features for s in samples], dim=0).float()
    edge_attr = torch.stack([s.edge_attr for s in samples], dim=0).float()
    y = torch.tensor([int(s.label) for s in samples], dtype=torch.long)
    edge_index = samples[0].edge_index.long().unsqueeze(0).expand(len(samples), -1, -1).contiguous()
    node_mask = torch.ones((len(samples), x.shape[1]), dtype=torch.bool)
    graph_id = torch.tensor([int(s.graph_id) for s in samples], dtype=torch.long)
    return {"x": x, "node_features": x, "edge_attr": edge_attr, "edge_index": edge_index, "node_mask": node_mask, "y": y, "label": y, "graph_id": graph_id}


def write_report(
    output_dir: Path,
    schema: Dict[str, Any],
    node_rows: Sequence[Dict[str, Any]],
    edge_rows: Sequence[Dict[str, Any]],
    topology: Sequence[Dict[str, Any]],
    probe_rows: Sequence[Dict[str, Any]],
    ablation_rows: Sequence[Dict[str, Any]],
    gate_info: Dict[str, Any],
    notes: Notes,
    sample_limit: Optional[int],
) -> None:
    warnings = list(schema.get("warnings", [])) + notes.items
    all_node = [r for r in node_rows if r.get("scope") == "all" and r.get("split") == "train"]
    all_edge = [r for r in edge_rows if r.get("scope") == "all" and r.get("split") == "train"]

    def signal_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
        ranked = sorted(rows, key=lambda r: abs(safe_float(r.get("std"))), reverse=True)
        return [f"- {r['feature']}: std={fmt(r.get('std'))}, range=[{fmt(r.get('min'))}, {fmt(r.get('max'))}]" for r in ranked[:8]]

    bad_features = [
        f"{r.get('feature')} ({r.get('split')})"
        for r in list(node_rows) + list(edge_rows)
        if int(r.get("nan_count", 0) or 0) > 0 or int(r.get("inf_count", 0) or 0) > 0
    ]
    suspect = [
        f"{r.get('feature')} ({r.get('split')})"
        for r in list(node_rows) + list(edge_rows)
        if r.get("scope") == "all" and safe_float(r.get("std")) < 1e-8
    ]
    probe_ok = [r for r in probe_rows if r.get("status") == "OK"]
    best_probe = sorted(probe_ok, key=lambda r: safe_float(r.get("macro_f1")), reverse=True)[:1]
    schema_ok = bool(schema.get("schema_consistent_train_val_test"))
    topo_problem = any(
        str(r.get("metric")) in {"invalid_edges", "duplicate_edges", "self_loops"} and safe_float(r.get("value")) > 0
        for r in topology
    )
    can_stage1 = schema_ok and not bad_features and not topo_problem and bool(probe_ok)
    if sample_limit is not None:
        readiness = "PARTIAL, smoke subset only; run without --max_samples_per_split before Stage 1 freeze."
    elif can_stage1:
        readiness = "YES"
    else:
        readiness = "NO, can xu ly cac warning/skip o tren truoc."

    lines = [
        "# Stage 0 Graph Feature Audit Report",
        "",
        "## Scope",
        "- Read existing graph artifacts only; no full model training, no model/loss/config mutation.",
        f"- Sample limit per split: {sample_limit if sample_limit is not None else 'FULL'}",
        f"- Graph repo: `{schema.get('graph_repo_path')}`",
        "",
        "## Schema",
        f"- node_dim: {schema.get('node_dim')}; edge_dim: {schema.get('edge_dim')}",
        f"- num_nodes: {schema.get('num_nodes')}; num_edges: {schema.get('num_edges')}",
        f"- node_feature_names source: {schema.get('feature_name_sources', {}).get('node_feature_names')}",
        f"- edge_feature_names source: {schema.get('feature_name_sources', {}).get('edge_feature_names')}",
        f"- train/val/test schema consistency: {schema_ok}",
        "",
        "## Feature nao chac chan dung",
    ]
    known_node = [n for n in schema.get("node_feature_names", []) if not str(n).startswith(UNKNOWN)]
    known_edge = [n for n in schema.get("edge_feature_names", []) if not str(n).startswith(UNKNOWN)]
    lines.append(f"- Node features co metadata ro: {', '.join(known_node) if known_node else 'NONE'}")
    lines.append(f"- Edge features co metadata ro: {', '.join(known_edge) if known_edge else 'NONE'}")
    lines.append("")
    lines.append("## Feature nao co tin hieu")
    lines.extend(signal_rows(all_node) or ["- No node stats available."])
    lines.extend(signal_rows(all_edge) or ["- No edge stats available."])
    if best_probe:
        row = best_probe[0]
        lines.append(f"- Probe tot nhat: {row['variant']} macro_f1={fmt(row.get('macro_f1'))}, accuracy={fmt(row.get('accuracy'))}")
    lines.append("")
    lines.append("## Feature nao nghi ngo")
    if bad_features:
        lines.append(f"- Co NaN/Inf: {', '.join(bad_features[:20])}")
    if suspect:
        lines.append(f"- Gan nhu hang so hoac static theo split: {', '.join(suspect[:20])}")
    if not bad_features and not suspect:
        lines.append("- Chua thay NaN/Inf hay feature degenerate trong pham vi scan.")
    lines.append("")
    lines.append("## Feature can ablation them")
    if ablation_rows:
        for row in ablation_rows:
            if row.get("status") == "OK":
                lines.append(f"- {row['variant']}: macro_f1={fmt(row.get('macro_f1'))}, accuracy={fmt(row.get('accuracy'))}")
            else:
                lines.append(f"- {row['variant']}: SKIPPED ({row.get('reason')})")
    else:
        lines.append("- feature_ablation.csv khong co row; xem notes.")
    lines.append("")
    lines.append("## Schema/scale/topology errors")
    lines.append(f"- Schema consistent: {schema_ok}")
    lines.append(f"- Topology problem detected: {topo_problem}")
    lines.append(f"- Gate diagnostics: {gate_info.get('status')} ({gate_info.get('reason', gate_info.get('gate_keys', ''))})")
    if warnings:
        lines.append("")
        lines.append("Notes/warnings:")
        lines.extend(f"- {w}" for w in warnings[:80])
    lines.append("")
    lines.append("## Stage 1 readiness")
    lines.append("- Du dieu kien chuyen Stage 1 pixel/region selection: " + readiness)
    lines.append("")
    lines.append("## Output index")
    for name in (
        "graph_schema.json",
        "node_feature_stats.csv",
        "edge_feature_stats.csv",
        "topology_stats.csv",
        "probe_metrics.csv",
        "feature_ablation.csv",
        "figures/",
    ):
        lines.append(f"- `{name}`")
    (output_dir / "stage0_graph_feature_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 0 graph/data/feature audit for FER-GNN graph artifacts.")
    parser.add_argument("--config", default="configs/experiments/d12a_global_local_motif.yaml")
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default="outputs/stage0_graph_feature_audit")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--figure_samples_per_class", type=int, default=2)
    parser.add_argument("--reservoir_size", type=int, default=120_000)
    parser.add_argument("--probe_max_train", type=int, default=4000)
    parser.add_argument("--probe_max_eval", type=int, default=1500)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for read-only forward gate diagnostics.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    notes = Notes()
    output_dir = ensure_dir(Path(args.output_dir))
    ensure_dir(output_dir / "figures")
    config = load_config(args.config) if args.config else {}
    graph_repo = resolve_graph_repo(args, config)
    reader = GraphRepositoryReader(graph_repo)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)
    first_sample = pick_sample(reader)
    first_resolved = resolver.resolve(first_sample)
    height = int(getattr(shared, "height", first_sample.height))
    width = int(getattr(shared, "width", first_sample.width))
    num_nodes = int(first_resolved.node_features.shape[0])
    num_edges = int(first_resolved.edge_attr.shape[0])
    node_dim = int(first_resolved.node_features.shape[1])
    edge_dim = int(first_resolved.edge_attr.shape[1])
    node_names, edge_names, name_sources = infer_feature_names(reader, config, first_sample, node_dim, edge_dim, notes)

    schema: Dict[str, Any] = {
        "graph_repo_path": str(graph_repo),
        "manifest_present": bool((graph_repo / "manifest.pt").exists()),
        "height": height,
        "width": width,
        "connectivity": int(getattr(shared, "connectivity", -1)),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_feature_names": node_names,
        "edge_feature_names": edge_names,
        "feature_name_sources": name_sources,
        "manifest": reader.manifest,
        "splits": {},
        "warnings": [],
    }
    if num_nodes != height * width:
        schema["warnings"].append(f"num_nodes {num_nodes} != height*width {height * width}")

    node_acc = {(split, "all"): FeatureAccumulator(node_names, args.reservoir_size, args.seed + i) for i, split in enumerate(SPLITS)}
    edge_acc = {(split, "all"): FeatureAccumulator(edge_names, args.reservoir_size, args.seed + 100 + i) for i, split in enumerate(SPLITS)}
    for split in SPLITS:
        for label in range(NUM_CLASSES):
            node_acc[(split, label)] = FeatureAccumulator(node_names, max(10_000, args.reservoir_size // 4), args.seed + 1000 + 31 * label)
            edge_acc[(split, label)] = FeatureAccumulator(edge_names, max(10_000, args.reservoir_size // 4), args.seed + 2000 + 31 * label)
    node_corr = RowReservoir(node_dim, args.reservoir_size, args.seed + 3001)
    edge_corr = RowReservoir(edge_dim, args.reservoir_size, args.seed + 3002)
    samples_by_class: Dict[int, List[Any]] = defaultdict(list)
    pooled_rows: List[Dict[str, Any]] = []
    pooled_names: Optional[List[str]] = None
    probe_batch_samples: List[Any] = []

    for split in SPLITS:
        paths = reader.chunk_paths(split)
        if not paths:
            schema["warnings"].append(f"Missing split chunks: {split}")
            continue
        print(f"[Stage0] Reading split={split}, chunks={len(paths)}")
        for sample in iter_limited(reader, split, args.max_samples_per_split):
            try:
                resolved = resolver.resolve(sample)
            except Exception as exc:
                schema["warnings"].append(f"Could not resolve sample split={split}: {exc}")
                continue
            add_schema_split(schema, split, resolved, sample)
            label = int(resolved.label)
            node_acc[(split, "all")].update(resolved.node_features)
            edge_acc[(split, "all")].update(resolved.edge_attr)
            if 0 <= label < NUM_CLASSES:
                node_acc[(split, label)].update(resolved.node_features)
                edge_acc[(split, label)].update(resolved.edge_attr)
                if len(samples_by_class[label]) < int(args.figure_samples_per_class):
                    samples_by_class[label].append(resolved)
            node_corr.update(resolved.node_features)
            edge_corr.update(resolved.edge_attr)
            features, feature_names = pooled_features(resolved.node_features, resolved.edge_attr, node_names, edge_names)
            pooled_names = feature_names
            if split == "train" and sum(r["split"] == "train" for r in pooled_rows) >= args.probe_max_train:
                pass
            elif split in {"val", "test"} and sum(r["split"] in {"val", "test"} for r in pooled_rows) >= args.probe_max_eval:
                pass
            else:
                pooled_rows.append({"split": split, "label": label, "graph_id": int(resolved.graph_id), "features": features})
            if len(probe_batch_samples) < 4:
                probe_batch_samples.append(resolved)

    schema = finalize_schema(schema)
    node_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        node_rows.extend(node_acc[(split, "all")].rows({"split": split, "scope": "all", "label": "all", "class_name": "all"}))
        edge_rows.extend(edge_acc[(split, "all")].rows({"split": split, "scope": "all", "label": "all", "class_name": "all"}))
        for label in range(NUM_CLASSES):
            scope = {"split": split, "scope": "class", "label": label, "class_name": class_name(label)}
            node_rows.extend(node_acc[(split, label)].rows(scope))
            edge_rows.extend(edge_acc[(split, label)].rows(scope))

    topo_rows, degree = topology_rows(shared.edge_index, num_nodes)
    write_csv(output_dir / "node_feature_stats.csv", node_rows)
    write_csv(output_dir / "edge_feature_stats.csv", edge_rows)
    write_csv(output_dir / "topology_stats.csv", topo_rows)
    node_c = node_corr.corr()
    edge_c = edge_corr.corr()
    write_csv(output_dir / "node_feature_correlation.csv", corr_rows("node", node_names, node_c))
    write_csv(output_dir / "edge_feature_correlation.csv", corr_rows("edge", edge_names, edge_c))
    save_corr_heatmap(node_c, node_names, output_dir / "figures" / "node_feature_correlation.png", "Node feature correlation", notes)
    save_corr_heatmap(edge_c, edge_names, output_dir / "figures" / "edge_feature_correlation.png", "Edge feature correlation", notes)
    make_figures(samples_by_class, shared.edge_index, node_names, edge_names, height, width, output_dir, degree, notes)

    probe_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    probe_rows.append(simple_probe(pooled_rows, output_dir, notes, "global_node_edge_pooled_all"))
    if pooled_names:
        pooled_names_np = np.asarray(pooled_names)
        intensity_mask = np.asarray(["node_intensity_" in name for name in pooled_names_np])
        xy_mask = np.asarray([("node_x_norm_" in name or "node_y_norm_" in name or "node_intensity_" in name) for name in pooled_names_np])
        all_node_mask = np.asarray([name.startswith("node_") for name in pooled_names_np])
        edge_mask = np.asarray([name.startswith("edge_") for name in pooled_names_np])
        for variant, mask in (
            ("only_intensity", intensity_mask),
            ("intensity_plus_xy", xy_mask),
            ("all_node_features", all_node_mask),
            ("all_edge_features", edge_mask),
            ("all_node_plus_edge", None),
        ):
            if mask is not None and not bool(mask.any()):
                ablation_rows.append({"variant": variant, "status": "SKIPPED", "reason": "feature names unavailable or mask empty"})
            else:
                ablation_rows.append(simple_probe(pooled_rows, output_dir, notes, variant, mask))
        if bool(edge_mask.any()):
            ablation_rows.append(
                simple_probe(
                    transformed_pooled_rows(pooled_rows, edge_mask=edge_mask, mode="edge_zeroed", seed=args.seed + 5100),
                    output_dir,
                    notes,
                    "edge_attr_zeroed_pooled",
                )
            )
            ablation_rows.append(
                simple_probe(
                    transformed_pooled_rows(pooled_rows, edge_mask=edge_mask, mode="edge_shuffled", seed=args.seed + 5200),
                    output_dir,
                    notes,
                    "edge_attr_shuffled_pooled",
                )
            )
        else:
            ablation_rows.append({"variant": "edge_attr_zeroed_pooled", "status": "SKIPPED", "reason": "edge feature mask empty"})
            ablation_rows.append({"variant": "edge_attr_shuffled_pooled", "status": "SKIPPED", "reason": "edge feature mask empty"})
    else:
        ablation_rows.append({"variant": "all", "status": "SKIPPED", "reason": "pooled feature names unavailable"})
    write_csv(output_dir / "probe_metrics.csv", probe_rows)
    write_csv(output_dir / "feature_ablation.csv", ablation_rows)

    probe_batch = collate_probe_batch(probe_batch_samples)
    gate_info = gate_diagnostics(args, config, probe_batch, output_dir, edge_names, notes)
    write_json(output_dir / "graph_schema.json", schema)
    write_report(output_dir, schema, node_rows, edge_rows, topo_rows, probe_rows, ablation_rows, gate_info, notes, args.max_samples_per_split)
    print(f"[Stage0] Wrote audit outputs to {output_dir}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
