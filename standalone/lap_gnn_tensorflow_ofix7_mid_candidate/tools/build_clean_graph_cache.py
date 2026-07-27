"""Build a sharded clean graph cache for the TensorFlow OFIX7-mid runner.

The cache is intentionally clean: no train-time prior corruption is applied.
The runtime uses it for validation/test and for train samples whose deterministic
corruption draw says "clean"; corrupted train samples still use the live path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from lap_gnn_tf.config import load_config, validate_locked_config
from lap_gnn_tf.data.clean_graph_cache import (
    CACHE_SCHEMA_VERSION,
    clean_graph_config_payload,
    clean_graph_config_sha256,
)
from lap_gnn_tf.priors.loader import PixelPriorDataset


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_size(graph) -> int:
    arrays = [
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        graph.pos,
        graph.y,
        graph.sample_index,
        graph.part_soft,
        graph.face_mask,
        graph.valid_part_mask,
        graph.valid_anchor_mask,
        graph.detected,
        graph.landmark_missing_flag,
        graph.image_48,
        graph.anchor_mask,
    ]
    return sum(int(array.nbytes) for array in arrays if array is not None)


def _serialize_graph(graph) -> bytes:
    stream = BytesIO()
    np.savez_compressed(
        stream,
        x=np.asarray(graph.x, dtype=np.float32),
        edge_index=np.asarray(graph.edge_index, dtype=np.int64),
        edge_attr=np.asarray(graph.edge_attr, dtype=np.float32),
        pos=np.asarray(graph.pos, dtype=np.float32),
        y=np.asarray(graph.y, dtype=np.int64),
        sample_index=np.asarray(graph.sample_index, dtype=np.int64),
        part_soft=np.asarray(graph.part_soft, dtype=np.float32),
        face_mask=np.asarray(graph.face_mask, dtype=np.float32),
        valid_part_mask=np.asarray(graph.valid_part_mask, dtype=np.float32),
        valid_anchor_mask=np.asarray(graph.valid_anchor_mask, dtype=np.float32),
        detected=np.asarray(graph.detected, dtype=np.bool_),
        landmark_missing_flag=np.asarray(graph.landmark_missing_flag, dtype=np.int64),
        image_48=np.asarray(graph.image_48, dtype=np.float32),
        anchor_mask=np.asarray(graph.anchor_mask, dtype=np.bool_),
    )
    return stream.getvalue()


def _write_shard(path: Path, graphs: list[Any]) -> list[int]:
    temporary = path.with_suffix(".tmp.bin")
    if temporary.exists():
        temporary.unlink()
    offsets = [0]
    with temporary.open("wb") as stream:
        for graph in graphs:
            payload = _serialize_graph(graph)
            stream.write(payload)
            offsets.append(offsets[-1] + len(payload))
    temporary.replace(path)
    return offsets


def _build_split(
    prior_root: Path,
    split: str,
    graph_config: dict[str, Any],
    split_root: Path,
    shard_size: int,
    workers: int,
    max_samples: int | None,
) -> dict[str, Any]:
    dataset = PixelPriorDataset(
        prior_root,
        split=split,
        graph_mode=graph_config["graph_mode"],
        face_threshold=graph_config["face_threshold"],
        context_pixels=graph_config["context_pixels"],
        detail_features=graph_config.get("detail_features"),
        edge_features=graph_config.get("edge_features"),
        anchor_nodes=graph_config.get("anchor_nodes"),
        node_features=graph_config.get("node_features"),
        knn_edges=graph_config.get("knn_edges"),
        prior_usage=graph_config.get("prior_usage"),
        prior_corruption=None,
        max_samples=max_samples,
    )
    split_root.mkdir(parents=True, exist_ok=True)
    shards = []
    total_bytes = 0
    started = time.perf_counter()
    for start in range(0, len(dataset), int(shard_size)):
        end = min(start + int(shard_size), len(dataset))
        indices = list(range(start, end))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                graphs = list(executor.map(dataset.__getitem__, indices))
        else:
            graphs = [dataset[index] for index in indices]
        shard_name = f"shard-{start // int(shard_size):05d}.bin"
        shard_path = split_root / shard_name
        offsets = _write_shard(shard_path, graphs)
        size = int(shard_path.stat().st_size)
        total_bytes += size
        shards.append({
            "path": shard_name,
            "start": start,
            "end": end,
            "samples": end - start,
            "offsets": offsets,
            "bytes": size,
            "sha256": _sha256_path(shard_path),
        })
        if (len(shards) == 1 or len(shards) % 10 == 0 or end == len(dataset)):
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "event": "clean_graph_cache_shard",
                "split": split,
                "start": start,
                "end": end,
                "total": len(dataset),
                "shards": len(shards),
                "cache_bytes": total_bytes,
                "elapsed_sec": elapsed,
            }), flush=True)
    index = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "split": split,
        "sample_count": len(dataset),
        "shard_size": int(shard_size),
        "shards": shards,
    }
    (split_root / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "split": split,
        "sample_count": len(dataset),
        "shards": len(shards),
        "bytes": total_bytes,
        "elapsed_sec": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.shard_size <= 0 or args.workers <= 0:
        raise ValueError("shard-size and workers must be positive")

    prior_root = Path(args.prior_root).resolve()
    output_root = Path(args.output_root).resolve()
    config = load_config(args.config)
    validate_locked_config(config)
    graph_config = config["graph"]
    graph_payload = clean_graph_config_payload(graph_config)
    graph_sha = clean_graph_config_sha256(graph_config)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"Cache output is not empty: {output_root}; pass --force to rebuild"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for split in args.splits:
        results.append(_build_split(
            prior_root,
            str(split),
            graph_config,
            output_root / str(split),
            args.shard_size,
            args.workers,
            args.max_samples,
        ))

    first_split = str(args.splits[0])
    first_index = json.loads(
        (output_root / first_split / "index.json").read_text(encoding="utf-8")
    )
    probe_dataset = PixelPriorDataset(
        prior_root,
        split=first_split,
        graph_mode=graph_config["graph_mode"],
        face_threshold=graph_config["face_threshold"],
        context_pixels=graph_config["context_pixels"],
        detail_features=graph_config.get("detail_features"),
        edge_features=graph_config.get("edge_features"),
        anchor_nodes=graph_config.get("anchor_nodes"),
        node_features=graph_config.get("node_features"),
        knn_edges=graph_config.get("knn_edges"),
        prior_usage=graph_config.get("prior_usage"),
        prior_corruption=None,
        max_samples=1,
    )
    probe_graph = probe_dataset[0]
    node_dim = int(probe_graph.x.shape[1])
    edge_dim = int(probe_graph.edge_attr.shape[1])
    complete = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": "clean_graph_arrays",
        "prior_root": str(prior_root),
        "config_path": str(Path(args.config).resolve()),
        "graph_config": graph_payload,
        "graph_config_sha256": graph_sha,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "node_feature_names": list(probe_graph.node_feature_names or []),
        "edge_feature_names": list(probe_graph.edge_feature_names or []),
        "splits": results,
        "created_at_epoch_sec": time.time(),
    }
    (output_root / "cache_manifest.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True), encoding="utf-8"
    )
    complete_marker = dict(complete)
    complete_marker["complete"] = True
    complete_marker["cache_manifest_sha256"] = _sha256_path(output_root / "cache_manifest.json")
    (output_root / "CACHE_COMPLETE.json").write_text(
        json.dumps(complete_marker, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "event": "clean_graph_cache_complete",
        "output_root": str(output_root),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "splits": results,
        "total_bytes": sum(item["bytes"] for item in results),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
