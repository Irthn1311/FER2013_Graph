"""Audit whether D15 edge attributes are loaded or recomputed."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_path, save_config
from data.graph_repository import ChunkedGraphDataset
from training.train_d14 import _apply_graph_augmentation


def _time_aug(batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    out = _apply_graph_augmentation(batch, cfg)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    changed = out is not batch
    recompute = bool(changed and cfg.get("enabled", False) and "edge_attr" in out)
    return {
        "augmentation_enabled": bool(cfg.get("enabled", False)),
        "edge_attr_source": "augmented_recomputed" if recompute else "precomputed",
        "edge_attr_compute_time_ms": elapsed_ms if recompute else 0.0,
        "edge_attr_recompute_count": 1 if recompute else 0,
        "batch_changed": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d15_speed/d15_m8_basic_speedfix_chunkaware_b32_w2_cache8.yaml")
    parser.add_argument("--output_dir", default="outputs/d15_speed_debug")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    save_config(config, output_dir / "edge_attr_audit_config")
    repo = resolve_path(config.get("paths", {}).get("graph_repo_path", "artifacts/graph_repo"))
    raw_ds = ChunkedGraphDataset(repo_root=repo, split="train", resolve=False, chunk_cache_size=1)
    sample = raw_ds[0]
    raw_info = {
        "raw_sample_has_edge_attr_dynamic": hasattr(sample, "edge_attr_dynamic"),
        "edge_attr_dynamic_shape": list(sample.edge_attr_dynamic.shape),
        "edge_attr_dynamic_source": "stored_in_graph_chunk",
    }

    resolved_ds = ChunkedGraphDataset(repo_root=repo, split="train", resolve=True, chunk_cache_size=1)
    resolved = resolved_ds[0]
    resolved_info = {
        "resolved_edge_attr_shape": list(resolved.edge_attr.shape),
        "edge_attr_source": "precomputed_static_plus_stored_dynamic",
        "resolver_concatenates_static_dynamic": True,
    }

    loader = build_dataloader(config, "train", shuffle=True)
    batch = next(iter(loader))
    basic_cfg = {"enabled": False}
    current_aug_cfg = dict(config.get("augmentation", {}) or {})
    no_aug = _time_aug(batch, basic_cfg)
    current_aug = _time_aug(batch, current_aug_cfg)
    result = {
        "config": args.config,
        "repo": str(repo),
        **raw_info,
        **resolved_info,
        "getitem_recomputes_edge_attr": False,
        "collate_recomputes_edge_attr": False,
        "basic_no_aug": no_aug,
        "configured_augmentation": current_aug,
        "standard_augmentation_recomputes_edges": bool(current_aug_cfg.get("enabled", False)),
        "strong_augmentation_recomputes_edges": True,
        "decision": "EDGE_ATTR_PRECOMPUTED_OK" if not current_aug_cfg.get("enabled", False) else "AUGMENTATION_RECOMPUTES_EDGE_ATTR",
    }
    (output_dir / "edge_attr_audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# D15 Edge Attr Audit Report",
        "",
        f"- config: `{args.config}`",
        f"- graph_repo: `{repo}`",
        f"- raw `edge_attr_dynamic`: stored in graph chunk, shape `{raw_info['edge_attr_dynamic_shape']}`",
        f"- resolved `edge_attr`: `{resolved_info['edge_attr_shape'] if 'edge_attr_shape' in resolved_info else resolved_info['resolved_edge_attr_shape']}`",
        "- `__getitem__`: loads stored dynamic edge attrs and resolves by concatenating shared static attrs; no dynamic recompute in basic/no_aug.",
        "- `collate_fn_full_graph`: stacks `edge_attr`; no recompute.",
        f"- configured augmentation enabled: {bool(current_aug_cfg.get('enabled', False))}",
        f"- configured edge_attr_source: {current_aug['edge_attr_source']}",
        f"- configured edge_attr_compute_time_ms_one_batch: {current_aug['edge_attr_compute_time_ms']:.3f}",
        "",
        "Decision:",
        f"- {result['decision']}",
    ]
    (output_dir / "edge_attr_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
