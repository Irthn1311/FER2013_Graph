"""Build reusable D18 graph cache from D16 retry-rescue prior npz files.

CPU-only. The output can be uploaded as a Kaggle Dataset and mounted through
`graph.cache.dir` in D18 configs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d18.data.structure_dataset import StructurePixelDataset
from d18.data.structure_graph_cache import graph_cache_path, load_d18_graph_cache, save_d18_graph_cache


def read_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def graph_cache_signature(cfg: Dict[str, Any]) -> Dict[str, Any]:
    graph = dict(cfg.get("graph", {}) or {})
    graph.pop("cache", None)
    edge_schema = str(graph.get("edge_schema", "base6"))
    edge_dim = 9 if edge_schema in {"structure9", "relation9", "d18b"} else 6
    return {
        "run_name": cfg.get("run_name"),
        "graph": graph,
        "node_feature_dim": 10,
        "edge_feature_dim": edge_dim,
        "edge_schema": edge_schema,
        "cache_tensor_dtypes": {
            "x": "float16",
            "edge_attr": "float16",
            "pos": "float16",
            "edge_index": "uint16",
            "image_48": "float16",
        },
        "cache_reuse_note": "D18A and D18B can share base6 cache when graph config is identical except drop_edge_p. D18C needs structure9 cache.",
    }


def build_split(
    cfg: Dict[str, Any],
    split: str,
    output_dir: Path,
    max_samples: int | None,
    overwrite: bool,
    compressed: bool,
    progress_interval: int,
) -> Dict[str, Any]:
    graph_cfg = dict(cfg.get("graph", {}) or {})
    graph_cfg["cache"] = {"enabled": False}
    prior_dir = cfg.get("data", {}).get("prior_dir", "outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    ds = StructurePixelDataset(prior_dir=prior_dir, split=split, graph=graph_cfg, max_samples=max_samples)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{split}.csv"
    start = time.perf_counter()
    built = 0
    skipped = 0
    rows = []
    total_bytes = 0
    for i, prior_file in enumerate(ds.files, start=1):
        cache_file = graph_cache_path(output_dir, split, prior_file)
        if cache_file.exists() and not overwrite:
            try:
                load_d18_graph_cache(cache_file)
                skipped += 1
                total_bytes += cache_file.stat().st_size
                continue
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "d18_graph_cache_rebuild_corrupt",
                            "split": split,
                            "index": i,
                            "cache_file": str(cache_file),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )
        graph = ds[i - 1]
        save_d18_graph_cache(graph, cache_file, compressed=compressed)
        size_bytes = cache_file.stat().st_size
        total_bytes += size_bytes
        built += 1
        rows.append(
            {
                "split": split,
                "source_file": prior_file.name,
                "cache_file": str(cache_file.relative_to(output_dir)),
                "sample_index": int(graph.sample_index),
                "label": int(graph.y),
                "detected": int(graph.detected),
                "landmark_missing_flag": int(graph.landmark_missing_flag),
                "node_count": int(graph.x.size(0)),
                "edge_dim": int(graph.edge_attr.size(1)),
                "local_edge_count": int(graph.local_edge_count),
                "knn_edge_count": int(graph.knn_edge_count),
                "structure_edge_count": int(graph.structure_edge_count),
                "total_edge_count": int(graph.total_edge_count),
                "cache_size_bytes": int(size_bytes),
            }
        )
        if progress_interval > 0 and (i == 1 or i % progress_interval == 0 or i == len(ds)):
            elapsed = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "event": "d18_graph_cache_progress",
                        "split": split,
                        "index": i,
                        "total": len(ds),
                        "built": built,
                        "skipped": skipped,
                        "elapsed_sec": elapsed,
                        "sec_per_built_graph": elapsed / max(built, 1),
                        "total_cache_mb_so_far": total_bytes / (1024**2),
                    }
                ),
                flush=True,
            )
    if rows:
        exists = manifest_path.exists() and not overwrite
        with manifest_path.open("a" if exists else "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
    elapsed = time.perf_counter() - start
    return {
        "split": split,
        "total": len(ds),
        "built": built,
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "sec_per_built_graph": elapsed / max(built, 1),
        "cache_size_mb": total_bytes / (1024**2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D18 graph cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed. Smaller, but slower to build/load.")
    parser.add_argument("--progress_interval", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    if args.prior_dir:
        cfg.setdefault("data", {})["prior_dir"] = args.prior_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = graph_cache_signature(cfg)
    signature.update({"source_config": args.config, "prior_dir": cfg.get("data", {}).get("prior_dir"), "compressed": bool(args.compressed)})
    (output_dir / "cache_config.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
    summaries = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        summaries.append(build_split(cfg, split, output_dir, args.max_samples, args.overwrite, args.compressed, args.progress_interval))
    summary = {"event": "d18_graph_cache_done", "output_dir": str(output_dir), "summaries": summaries}
    (output_dir / "cache_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
