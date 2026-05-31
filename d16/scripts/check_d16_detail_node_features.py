"""Check D16R-A5a detail-aware node feature construction."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.data.pixel_prior_dataset import D16PixelPriorDataset
from d16.models.d16_model import D16Model


EXPECTED_BASE_DIM = 32
EXPECTED_DETAIL_DIM = 37
DETAIL_NAMES = ["grad_mag", "local_mean_3x3", "local_std_3x3", "laplacian_abs", "center_surround"]


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()) if values.size else float("nan"),
        "std": float(values.std()) if values.size else float("nan"),
        "min": float(values.min()) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
    }


def _dataset(cfg: Dict[str, Any], prior_dir: Path, split: str, detail_features: Dict[str, Any], max_samples: int):
    graph_cfg = cfg.get("graph", {}) or {}
    return D16PixelPriorDataset(
        prior_dir=prior_dir,
        split=split,
        graph_mode=str(graph_cfg.get("graph_mode", (cfg.get("data", {}) or {}).get("graph_mode", "face_plus_context"))),
        face_threshold=float(graph_cfg.get("face_threshold", 0.15)),
        context_pixels=int(graph_cfg.get("context_pixels", 2)),
        detail_features=detail_features,
        max_samples=max_samples,
    )


def run_check(config_path: Path, prior_dir: Path, output_dir: Path, max_samples_per_split: int = 4) -> Dict[str, Any]:
    cfg = load_config(config_path)
    graph_cfg = cfg.get("graph", {}) or {}
    detail_cfg = graph_cfg.get("detail_features", {}) or {}
    expected_names = list(detail_cfg.get("features") or DETAIL_NAMES)
    failures: List[str] = []
    warnings: List[str] = []

    if not bool(detail_cfg.get("enabled", False)):
        failures.append("graph.detail_features.enabled is not true")
    if not bool(detail_cfg.get("append_to_x", True)):
        failures.append("graph.detail_features.append_to_x must be true")
    if str(graph_cfg.get("graph_mode", (cfg.get("data", {}) or {}).get("graph_mode", ""))) != "face_plus_context":
        failures.append("graph_mode is not face_plus_context")
    if (cfg.get("data", {}) or {}).get("graph_cache_dir") not in (None, "", "null"):
        failures.append("data.graph_cache_dir must be null for A5a detail feature check")

    split_summaries: Dict[str, Any] = {}
    all_detail_values: Dict[str, List[np.ndarray]] = {name: [] for name in expected_names}
    first_batch_dim = None
    first_model_input_dim = None
    forward_ok = False

    for split in ("train", "val", "test"):
        ds = _dataset(cfg, prior_dir, split, detail_cfg, max_samples=max_samples_per_split)
        base_detail = copy.deepcopy(detail_cfg)
        base_detail["enabled"] = False
        base_ds = _dataset(cfg, prior_dir, split, base_detail, max_samples=1)
        base_graph = base_ds[0]
        graph = ds[0]
        if int(base_graph.x.size(1)) != EXPECTED_BASE_DIM:
            failures.append(f"{split}: disabled detail x dim={int(base_graph.x.size(1))}, expected {EXPECTED_BASE_DIM}")
        if int(graph.x.size(1)) != EXPECTED_DETAIL_DIM:
            failures.append(f"{split}: A5a x dim={int(graph.x.size(1))}, expected {EXPECTED_DETAIL_DIM}")
        if graph.x.size(1) >= EXPECTED_DETAIL_DIM:
            detail = graph.x[:, EXPECTED_BASE_DIM:EXPECTED_DETAIL_DIM].detach().cpu().numpy()
            if not np.isfinite(detail).all():
                failures.append(f"{split}: detail feature block contains NaN/Inf")
            for idx, name in enumerate(expected_names):
                vals = detail[:, idx]
                all_detail_values[name].append(vals)
                if np.allclose(vals, 0.0):
                    failures.append(f"{split}: detail feature {name} is all zero in first sample")

        loader = DataLoader(ds, batch_size=min(2, len(ds)), shuffle=False, collate_fn=collate_d16_graphs)
        batch = next(iter(loader))
        batch_dim = int(batch.x_cat.size(1))
        first_batch_dim = batch_dim if first_batch_dim is None else first_batch_dim
        if batch_dim != EXPECTED_DETAIL_DIM:
            failures.append(f"{split}: collated x_cat dim={batch_dim}, expected {EXPECTED_DETAIL_DIM}")
        if not torch.isfinite(batch.x_cat).all().item():
            failures.append(f"{split}: collated x_cat contains NaN/Inf")
        split_summaries[split] = {
            "num_samples_checked": len(ds),
            "first_sample_nodes": int(graph.x.size(0)),
            "first_sample_x_dim": int(graph.x.size(1)),
            "disabled_detail_x_dim": int(base_graph.x.size(1)),
            "collated_x_cat_dim": batch_dim,
            "graph_mode": str(graph_cfg.get("graph_mode", (cfg.get("data", {}) or {}).get("graph_mode", ""))),
        }
        if split == "train":
            model = D16Model.from_config(cfg, input_dim=batch.x_cat.size(1))
            first_linear = model.encoder.net[0]
            first_model_input_dim = int(first_linear.in_features)
            if first_model_input_dim != EXPECTED_DETAIL_DIM:
                failures.append(f"model first encoder layer in_features={first_model_input_dim}, expected {EXPECTED_DETAIL_DIM}")
            with torch.no_grad():
                out = model(batch)
            logits = out.get("logits")
            forward_ok = isinstance(logits, torch.Tensor) and tuple(logits.shape) == (batch.num_graphs, 7)
            if not forward_ok:
                failures.append(f"model forward logits shape invalid: {None if logits is None else tuple(logits.shape)}")
            elif not torch.isfinite(logits).all().item():
                failures.append("model forward logits contain NaN/Inf")

    feature_stats: Dict[str, Dict[str, float]] = {}
    for name, chunks in all_detail_values.items():
        values = np.concatenate(chunks, axis=0) if chunks else np.asarray([], dtype=np.float32)
        feature_stats[name] = _stats(values)
        if values.size and math.isclose(float(values.std()), 0.0, abs_tol=1e-8):
            warnings.append(f"detail feature {name} has near-zero std across checked first samples")
        if not np.isfinite(values).all():
            failures.append(f"detail feature {name} aggregate contains NaN/Inf")

    summary = {
        "config": str(config_path),
        "prior_dir": str(prior_dir),
        "output_dir": str(output_dir),
        "decision": "PASS" if not failures else "FAIL",
        "expected_base_dim": EXPECTED_BASE_DIM,
        "expected_detail_dim": EXPECTED_DETAIL_DIM,
        "detail_feature_names": expected_names,
        "first_batch_x_cat_dim": first_batch_dim,
        "model_first_layer_input_dim": first_model_input_dim,
        "model_forward_ok": forward_ok,
        "split_summaries": split_summaries,
        "feature_stats": feature_stats,
        "failures": failures,
        "warnings": warnings,
    }
    _write_json(output_dir / "detail_node_feature_check_summary.json", summary)
    lines = [
        "# D16R-A5a Detail Node Feature Check",
        "",
        f"- decision: `{summary['decision']}`",
        f"- config: `{summary['config']}`",
        f"- prior_dir: `{summary['prior_dir']}`",
        f"- expected_base_dim: `{EXPECTED_BASE_DIM}`",
        f"- expected_detail_dim: `{EXPECTED_DETAIL_DIM}`",
        f"- first_batch_x_cat_dim: `{first_batch_dim}`",
        f"- model_first_layer_input_dim: `{first_model_input_dim}`",
        f"- model_forward_ok: `{forward_ok}`",
        "",
        "## Feature Stats",
        "| feature | mean | std | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in feature_stats.items():
        lines.append(
            f"| {name} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |"
        )
    lines.extend(["", "## Failures"])
    lines.extend([f"- {item}" for item in failures] if failures else ["- none"])
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- none"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "D16R_A5A_DETAIL_NODE_FEATURE_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples_per_split", type=int, default=4)
    args = parser.parse_args()
    summary = run_check(
        config_path=Path(args.config),
        prior_dir=Path(args.prior_dir),
        output_dir=Path(args.output_dir),
        max_samples_per_split=int(args.max_samples_per_split),
    )
    print(json.dumps(summary, indent=2, default=str))
    if summary["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
