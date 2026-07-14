"""Build the OFIX17 D18 graph caches needed for Kaggle training.

OFIX17 has three distinct graph-cache signatures:
- base6_shared: used by OFIX17A and OFIX17B.
- purified_base6: used by OFIX17C because purification changes structure topology.
- structure9_capped_gate: used by OFIX17D because edge_attr_dim is 9.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d18.scripts.build_d18_graph_cache import build_split, graph_cache_signature, read_config


CACHE_JOBS = [
    {
        "key": "base6_shared",
        "config": "configs/d18/ofix17_structure_reg/d18_ofix17a_drop_structure_seed42.yaml",
        "used_by": ["d18_ofix17a_drop_structure_seed42", "d18_ofix17b_structure_mode_mix_seed42"],
        "note": "A/B share identical graph schema; training regularization differs after collate.",
    },
    {
        "key": "purified_base6",
        "config": "configs/d18/ofix17_structure_reg/d18_ofix17c_purified_structure_seed42.yaml",
        "used_by": ["d18_ofix17c_purified_structure_seed42"],
        "note": "C needs its own cache because purification filters structure edges before edge_attr.",
    },
    {
        "key": "structure9_capped_gate",
        "config": "configs/d18/ofix17_structure_reg/d18_ofix17d_capped_gate_seed42.yaml",
        "used_by": ["d18_ofix17d_capped_gate_seed42"],
        "note": "D needs structure9 edge attributes for scalar edge gate.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all OFIX17 graph caches")
    parser.add_argument("--output_root", default="outputs/d18_graph_cache/ofix17_structure_reg")
    parser.add_argument("--prior_dir", default=None)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--jobs", default="all", help="Comma list: all, base6_shared, purified_base6, structure9_capped_gate")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--progress_interval", type=int, default=1000)
    return parser.parse_args()


def selected_jobs(keys: str) -> List[Dict[str, Any]]:
    wanted = {x.strip() for x in str(keys).split(",") if x.strip()}
    if not wanted or "all" in wanted:
        return list(CACHE_JOBS)
    known = {job["key"] for job in CACHE_JOBS}
    missing = sorted(wanted - known)
    if missing:
        raise ValueError(f"Unknown OFIX17 cache jobs: {missing}. Known: {sorted(known)}")
    return [job for job in CACHE_JOBS if job["key"] in wanted]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    started = time.perf_counter()
    all_rows: List[Dict[str, Any]] = []
    plan: List[Dict[str, Any]] = []
    for job in selected_jobs(args.jobs):
        cfg = read_config(job["config"])
        if args.prior_dir:
            cfg.setdefault("data", {})["prior_dir"] = args.prior_dir
        job_output = output_root / job["key"]
        job_output.mkdir(parents=True, exist_ok=True)
        signature = graph_cache_signature(cfg)
        signature.update(
            {
                "source_config": job["config"],
                "prior_dir": cfg.get("data", {}).get("prior_dir"),
                "compressed": bool(args.compressed),
                "ofix17_cache_key": job["key"],
                "used_by": job["used_by"],
                "note": job["note"],
                "cache_format_requires_edge_type": True,
            }
        )
        (job_output / "cache_config.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
        print(json.dumps({"event": "ofix17_cache_job_start", "key": job["key"], "output_dir": str(job_output), "used_by": job["used_by"]}), flush=True)
        summaries = []
        for split in splits:
            summaries.append(
                build_split(
                    cfg=cfg,
                    split=split,
                    output_dir=job_output,
                    max_samples=args.max_samples,
                    overwrite=bool(args.overwrite),
                    compressed=bool(args.compressed),
                    progress_interval=int(args.progress_interval),
                )
            )
        row = {"key": job["key"], "output_dir": str(job_output), "used_by": job["used_by"], "summaries": summaries}
        all_rows.append(row)
        plan.append({"key": job["key"], "local_cache_dir": str(job_output), "used_by": job["used_by"], "config": job["config"]})
        (job_output / "cache_build_summary.json").write_text(json.dumps({"event": "ofix17_cache_job_done", **row}, indent=2), encoding="utf-8")
        print(json.dumps({"event": "ofix17_cache_job_done", "key": job["key"], "summaries": summaries}, indent=2), flush=True)
    final = {"event": "ofix17_all_cache_done", "output_root": str(output_root), "elapsed_sec": time.perf_counter() - started, "jobs": all_rows}
    (output_root / "OFIX17_CACHE_USAGE.json").write_text(json.dumps({"cache_jobs": plan}, indent=2), encoding="utf-8")
    (output_root / "cache_build_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
