"""Build static OFIX14 k-NN edge cache from D16 prior NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import yaml

from d16.data.graph_builder import (
    _knn_cache_path,
    _node_coord_hash,
    build_pixel_graph,
    compute_knn_dst,
)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _knn_cfg_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    graph = config.get("graph", {}) or {}
    cfg = dict(graph.get("knn_edges", {}) or {})
    if not cfg:
        cfg = {
            "enabled": True,
            "k": 6,
            "metric": "standardized_euclidean",
            "feature_names": [
                "intensity",
                "gx",
                "gy",
                "grad_mag",
                "local_mean_3x3",
                "local_std_3x3",
                "laplacian_abs",
                "center_surround",
            ],
        }
    cfg["enabled"] = True
    cfg["cache_enabled"] = False
    cfg.pop("cache_dir", None)
    cfg.pop("cache_root", None)
    cfg.pop("cache_split", None)
    return cfg


def _graph_without_knn(config: Dict[str, Any], prior: Dict[str, np.ndarray]):
    graph = config.get("graph", {}) or {}
    return build_pixel_graph(
        prior,
        graph_mode=str(graph.get("graph_mode", "face_plus_context")),
        face_threshold=float(graph.get("face_threshold", 0.15)),
        context_pixels=int(graph.get("context_pixels", 2)),
        detail_features=graph.get("detail_features", {}) or {},
        edge_features={},
        anchor_nodes={"enabled": False},
        node_features=graph.get("node_features", {}) or {},
        knn_edges={"enabled": False},
        prior_usage=graph.get("prior_usage"),
    )


def _existing_valid(path: Path, node_count: int, k: int, coord_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(np.asarray(data["node_count"]).item()) == int(node_count)
                and int(np.asarray(data["k"]).item()) == int(k)
                and str(np.asarray(data["coord_hash"]).item()) == str(coord_hash)
                and "knn_dst" in data.files
            )
    except Exception:
        return False


def _write_cache_file(
    path: Path,
    knn_dst: np.ndarray,
    *,
    sample_index: int,
    node_count: int,
    k: int,
    coord_hash: str,
    compressed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "knn_dst": np.asarray(knn_dst, dtype=np.uint16),
        "sample_index": np.asarray(int(sample_index), dtype=np.int64),
        "node_count": np.asarray(int(node_count), dtype=np.int32),
        "k": np.asarray(int(k), dtype=np.int16),
        "coord_hash": np.asarray(str(coord_hash)),
    }
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def _split_files(prior_dir: Path, split: str, limit: int | None) -> list[Path]:
    files = sorted((prior_dir / split).glob("*.npz"))
    if limit is not None:
        files = files[: int(limit)]
    if not files:
        raise FileNotFoundError(f"No NPZ files found for split={split}: {prior_dir / split}")
    return files


def build_cache(
    config: Dict[str, Any],
    prior_dir: Path,
    output_dir: Path,
    splits: Iterable[str],
    *,
    limit_per_split: int | None = None,
    compressed: bool = False,
    overwrite: bool = False,
    progress_interval: int = 500,
) -> list[Dict[str, Any]]:
    knn_cfg = _knn_cfg_from_config(config)
    rows: list[Dict[str, Any]] = []
    start_all = time.perf_counter()
    for split in splits:
        files = _split_files(prior_dir, split, limit_per_split)
        split_start = time.perf_counter()
        for idx, npz_path in enumerate(files, start=1):
            prior = _load_npz(npz_path)
            graph = _graph_without_knn(config, prior)
            sample_index = int(graph.sample_index.item())
            node_count = int(graph.x.size(0))
            k_eff = min(int(knn_cfg.get("k", 6) or 6), max(node_count - 1, 0))
            coords = np.rint(((graph.pos.numpy() + 1.0) * 47.0 / 2.0)).astype(np.int16)
            coords = coords[:, [1, 0]]
            coord_hash = _node_coord_hash(coords)
            cache_path = _knn_cache_path(output_dir, split, sample_index)
            if not overwrite and _existing_valid(cache_path, node_count, k_eff, coord_hash):
                status = "skipped_existing"
                knn_edges = node_count * k_eff
            else:
                knn_dst = compute_knn_dst(graph.x.numpy(), graph.node_feature_names or [], knn_cfg)
                _write_cache_file(
                    cache_path,
                    knn_dst,
                    sample_index=sample_index,
                    node_count=node_count,
                    k=k_eff,
                    coord_hash=coord_hash,
                    compressed=compressed,
                )
                status = "written"
                knn_edges = int(knn_dst.size)
            rows.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "source_file": str(npz_path),
                    "cache_file": str(cache_path),
                    "node_count": node_count,
                    "k": k_eff,
                    "knn_edges": knn_edges,
                    "coord_hash": coord_hash,
                    "status": status,
                }
            )
            if progress_interval > 0 and (idx == 1 or idx % int(progress_interval) == 0 or idx == len(files)):
                elapsed = time.perf_counter() - split_start
                print(
                    json.dumps(
                        {
                            "event": "d16_knn_cache_progress",
                            "split": split,
                            "index": idx,
                            "total": len(files),
                            "elapsed_sec": elapsed,
                            "sec_per_sample": elapsed / max(idx, 1),
                        }
                    ),
                    flush=True,
                )
    elapsed_all = time.perf_counter() - start_all
    summary = {
        "prior_dir": str(prior_dir),
        "output_dir": str(output_dir),
        "splits": list(splits),
        "total_files": len(rows),
        "elapsed_sec": elapsed_all,
        "sec_per_sample": elapsed_all / max(len(rows), 1),
        "knn_edges": knn_cfg,
        "compressed": bool(compressed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "sample_index", "source_file", "cache_file", "node_count", "k", "knn_edges", "coord_hash", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "d16_knn_cache_done", **summary}, indent=2), flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prior_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limit_per_split", type=int, default=None)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress_interval", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    build_cache(
        config,
        args.prior_dir,
        args.output_dir,
        splits,
        limit_per_split=args.limit_per_split,
        compressed=bool(args.compressed),
        overwrite=bool(args.overwrite),
        progress_interval=int(args.progress_interval),
    )


if __name__ == "__main__":
    main()
