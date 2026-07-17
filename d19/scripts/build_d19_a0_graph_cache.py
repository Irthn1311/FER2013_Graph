"""Build a resumable landmark-free graph cache for D19-A0 from FER CSV splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.structure_dataset import StructurePixelDataset
from d18.data.structure_graph_cache import (
    EVIDENCE_CACHE_SCHEMA,
    evidence_cache_signature,
    evidence_cache_signature_payload,
    evidence_graph_cache_path,
    load_d18_graph_cache,
    save_d18_graph_cache,
)


def read_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_split(
    cfg: dict[str, Any],
    split: str,
    output_dir: Path,
    overwrite: bool,
    max_samples: int | None,
    progress_interval: int,
) -> dict[str, Any]:
    graph_cfg = dict(cfg.get("graph", {}) or {})
    graph_cfg["cache"] = {"enabled": False, "schema": EVIDENCE_CACHE_SCHEMA}
    evidence_dir = (cfg.get("data") or {}).get("evidence_dir")
    dataset = StructurePixelDataset(
        prior_dir=None,
        split=split,
        graph=graph_cfg,
        max_samples=max_samples,
        evidence_dir=evidence_dir,
    )
    manifest_path = output_dir / f"manifest_{split}.csv"
    rows: list[dict[str, Any]] = []
    built = skipped = rebuilt = total_bytes = 0
    started = time.perf_counter()
    for index in range(len(dataset)):
        evidence = dataset._load_evidence(index)
        cache_file = evidence_graph_cache_path(
            output_dir,
            split,
            index,
            evidence["image_48"],
            int(evidence["label"]),
            graph_cfg,
        )
        cache_valid = False
        if cache_file.exists() and not overwrite:
            try:
                graph = load_d18_graph_cache(cache_file)
                if int(graph.sample_index) != index or int(graph.y) != int(evidence["label"]):
                    raise RuntimeError("cached sample identity mismatch")
                if graph.structure_edge_count != 0 or bool((graph.edge_type == 2).any()):
                    raise RuntimeError("cached A0 graph contains structure edges")
                skipped += 1
                cache_valid = True
            except Exception:
                rebuilt += 1
        if not cache_valid:
            graph = dataset[index]
            if graph.structure_edge_count != 0 or bool((graph.edge_type == 2).any()):
                raise RuntimeError(f"A0 cache build produced structure edges for {split}/{index}")
            save_d18_graph_cache(graph, cache_file, compressed=False)
            built += 1
        size = cache_file.stat().st_size
        total_bytes += size
        rows.append({
            "split": split,
            "sample_index": index,
            "label": int(graph.y),
            "cache_file": str(cache_file.relative_to(output_dir)),
            "node_count": int(graph.x.size(0)),
            "local_edge_count": int(graph.local_edge_count),
            "knn_edge_count": int(graph.knn_edge_count),
            "structure_edge_count": int(graph.structure_edge_count),
            "total_edge_count": int(graph.total_edge_count),
            "cache_size_bytes": size,
        })
        current = index + 1
        if progress_interval > 0 and (current == 1 or current % progress_interval == 0 or current == len(dataset)):
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "event": "d19_a0_cache_progress",
                "split": split,
                "current": current,
                "total": len(dataset),
                "built": built,
                "skipped": skipped,
                "rebuilt": rebuilt,
                "elapsed_sec": elapsed,
                "sec_per_new_graph": elapsed / max(built, 1),
                "cache_gb_so_far": total_bytes / (1024**3),
            }), flush=True)
    if rows:
        # Rewrite the complete manifest on every invocation. This keeps a
        # resumed build self-contained even when most graph files were skipped.
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elapsed = time.perf_counter() - started
    return {
        "split": split,
        "total": len(dataset),
        "built": built,
        "skipped": skipped,
        "rebuilt": rebuilt,
        "elapsed_sec": elapsed,
        "cache_gb": total_bytes / (1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = read_config(Path(args.config))
    if args.evidence_dir:
        cfg.setdefault("data", {})["evidence_dir"] = args.evidence_dir
    graph_cfg = cfg.get("graph", {}) or {}
    if str(graph_cfg.get("graph_mode")) != "evidence_only":
        raise RuntimeError("D19 A0 cache builder requires graph_mode=evidence_only")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = evidence_cache_signature(graph_cfg)
    signature_payload = {
        "cache_schema": EVIDENCE_CACHE_SCHEMA,
        "namespace_sha256": signature,
        "namespace_dir": signature[:16],
        "evidence_dir": str((cfg.get("data") or {}).get("evidence_dir")),
        "signature_payload": evidence_cache_signature_payload(graph_cfg),
        "landmark_dependencies": [],
        "structure_dependencies": [],
    }
    (output / "cache_signature.json").write_text(json.dumps(signature_payload, indent=2), encoding="utf-8")
    summaries = [
        build_split(cfg, split.strip(), output, args.overwrite, args.max_samples, args.progress_interval)
        for split in str(args.splits).split(",") if split.strip()
    ]
    result = {"status": "COMPLETE", "output_dir": str(output), "signature": signature_payload, "splits": summaries}
    (output / "cache_build_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "CACHE_COMPLETE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
