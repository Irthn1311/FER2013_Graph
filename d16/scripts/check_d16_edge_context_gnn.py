"""Check D16R-A5b edge-context GNN data/model wiring."""

from __future__ import annotations

import argparse
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


EXPECTED_X_DIM = 37
EXPECTED_EDGE_DIM = 8


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _stats(tensor: torch.Tensor) -> Dict[str, float]:
    arr = tensor.detach().cpu().float().numpy()
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
    }


def _dataset(cfg: Dict[str, Any], prior_dir: Path, split: str, max_samples: int):
    data_cfg = cfg.get("data", {}) or {}
    graph_cfg = cfg.get("graph", {}) or {}
    return D16PixelPriorDataset(
        prior_dir=prior_dir,
        split=split,
        graph_mode=str(graph_cfg.get("graph_mode", data_cfg.get("graph_mode", "face_plus_context"))),
        face_threshold=float(graph_cfg.get("face_threshold", 0.15)),
        context_pixels=int(graph_cfg.get("context_pixels", 2)),
        detail_features=graph_cfg.get("detail_features", {}) or {},
        edge_features=graph_cfg.get("edge_features", {}) or {},
        max_samples=max_samples,
    )


def _finite_diag(diag: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    for key, value in diag.items():
        if torch.is_tensor(value):
            ok = bool(torch.isfinite(value).all().item())
            if not ok:
                failures.append(f"diagnostic {key} is non-finite")
        elif isinstance(value, (float, int)):
            if not math.isfinite(float(value)):
                failures.append(f"diagnostic {key} is non-finite")
    return failures


def run_check(config_path: Path, prior_dir: Path, output_dir: Path, max_samples_per_split: int = 4) -> Dict[str, Any]:
    cfg = load_config(config_path)
    graph_cfg = cfg.get("graph", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    detail_cfg = graph_cfg.get("detail_features", {}) or {}
    edge_cfg = graph_cfg.get("edge_features", {}) or {}
    edge_names = list(edge_cfg.get("features") or [])
    failures: List[str] = []
    warnings: List[str] = []

    if model_cfg.get("gnn_type") != "edge_context_gnn":
        failures.append(f"model.gnn_type must be edge_context_gnn, got {model_cfg.get('gnn_type')!r}")
    if not bool(detail_cfg.get("enabled", False)):
        failures.append("graph.detail_features.enabled must be true")
    if not bool(edge_cfg.get("enabled", False)):
        failures.append("graph.edge_features.enabled must be true")
    if (cfg.get("data", {}) or {}).get("graph_cache_dir") not in (None, "", "null"):
        failures.append("data.graph_cache_dir must be null for A5b edge-context check")

    split_summaries: Dict[str, Any] = {}
    edge_chunks: List[torch.Tensor] = []
    first_forward_diag: Dict[str, Any] = {}
    forward_ok = False
    backward_ok = False
    vectorized_context_max_abs_diff: float | None = None

    for split in ("train", "val", "test"):
        ds = _dataset(cfg, prior_dir, split, max_samples=max_samples_per_split)
        graph = ds[0]
        if int(graph.x.size(1)) != EXPECTED_X_DIM:
            failures.append(f"{split}: x dim={int(graph.x.size(1))}, expected {EXPECTED_X_DIM}")
        if graph.edge_attr is None:
            failures.append(f"{split}: edge_attr is missing")
        else:
            if int(graph.edge_attr.size(1)) != EXPECTED_EDGE_DIM:
                failures.append(f"{split}: edge_attr dim={int(graph.edge_attr.size(1))}, expected {EXPECTED_EDGE_DIM}")
            if graph.edge_attr.size(0) != graph.edge_index.size(1):
                failures.append(f"{split}: edge_attr row count does not match edge_index")
            if not torch.isfinite(graph.edge_attr).all().item():
                failures.append(f"{split}: edge_attr contains NaN/Inf")
            if torch.allclose(graph.edge_attr, torch.zeros_like(graph.edge_attr)):
                failures.append(f"{split}: edge_attr is all zero")
            edge_chunks.append(graph.edge_attr)

        loader = DataLoader(ds, batch_size=min(2, len(ds)), shuffle=False, collate_fn=collate_d16_graphs)
        batch = next(iter(loader))
        if int(batch.x_cat.size(1)) != EXPECTED_X_DIM:
            failures.append(f"{split}: batch x_cat dim={int(batch.x_cat.size(1))}, expected {EXPECTED_X_DIM}")
        if batch.edge_attr_cat is None:
            failures.append(f"{split}: batch edge_attr_cat is missing")
        elif int(batch.edge_attr_cat.size(1)) != EXPECTED_EDGE_DIM:
            failures.append(f"{split}: batch edge_attr_cat dim={int(batch.edge_attr_cat.size(1))}, expected {EXPECTED_EDGE_DIM}")
        if batch.edge_attr_cat is not None and batch.edge_attr_cat.size(0) != batch.edge_index_cat.size(1):
            failures.append(f"{split}: batch edge_attr_cat row count does not match edge_index_cat")
        split_summaries[split] = {
            "num_samples_checked": len(ds),
            "first_sample_nodes": int(graph.x.size(0)),
            "first_sample_edges": int(graph.edge_index.size(1)),
            "first_sample_x_dim": int(graph.x.size(1)),
            "first_sample_edge_attr_dim": None if graph.edge_attr is None else int(graph.edge_attr.size(1)),
            "batch_x_cat_dim": int(batch.x_cat.size(1)),
            "batch_edge_attr_dim": None if batch.edge_attr_cat is None else int(batch.edge_attr_cat.size(1)),
            "graph_mode": str(graph_cfg.get("graph_mode", (cfg.get("data", {}) or {}).get("graph_mode", ""))),
        }
        if split == "train":
            model = D16Model.from_config(cfg, input_dim=batch.x_cat.size(1))
            model.train()
            out = model(batch)
            logits = out.get("logits")
            forward_ok = isinstance(logits, torch.Tensor) and tuple(logits.shape) == (batch.num_graphs, 7)
            if not forward_ok:
                failures.append(f"model forward logits shape invalid: {None if logits is None else tuple(logits.shape)}")
            elif not torch.isfinite(logits).all().item():
                failures.append("model forward logits contain NaN/Inf")
            node_embeddings = out.get("node_embeddings")
            if not torch.is_tensor(node_embeddings) or not torch.isfinite(node_embeddings).all().item():
                failures.append("node_embeddings missing or non-finite")
            diag = out.get("edge_context_gnn_diagnostics") or {}
            first_forward_diag = {
                key: (float(value.detach().cpu().item()) if torch.is_tensor(value) and value.numel() == 1 else str(value))
                for key, value in diag.items()
            }
            failures.extend(_finite_diag(diag))
            context_block = getattr(getattr(model, "gnn", None), "context_block", None)
            if context_block is not None and hasattr(context_block, "_forward_loop_reference"):
                was_training = bool(context_block.training)
                context_block.eval()
                with torch.no_grad():
                    torch.manual_seed(123)
                    h_probe = torch.randn(
                        batch.x_cat.size(0),
                        int(getattr(context_block, "hidden_dim", 96)),
                        dtype=torch.float32,
                    )
                    ref = context_block._forward_loop_reference(
                        h_probe,
                        batch.part_soft_cat.float(),
                        batch.batch_index.long(),
                        batch.num_graphs,
                    )
                    vec, _ = context_block(
                        h_probe,
                        batch.part_soft_cat.float(),
                        batch.batch_index.long(),
                        batch.num_graphs,
                        collect_diagnostics=False,
                    )
                    vectorized_context_max_abs_diff = float((ref - vec).abs().max().item())
                context_block.train(was_training)
                if vectorized_context_max_abs_diff > 1.0e-4:
                    failures.append(
                        "vectorized context injection differs from loop reference: "
                        f"max_abs_diff={vectorized_context_max_abs_diff:.6g}"
                    )
            loss = logits.float().sum() if torch.is_tensor(logits) else torch.tensor(0.0)
            loss.backward()
            backward_ok = True

    edge_stats: Dict[str, Dict[str, float]] = {}
    if edge_chunks:
        all_edges = torch.cat(edge_chunks, dim=0)
        for idx, name in enumerate(edge_names):
            edge_stats[name] = _stats(all_edges[:, idx])
            if torch.allclose(all_edges[:, idx], torch.zeros_like(all_edges[:, idx])):
                warnings.append(f"edge feature {name} is all zero across checked first samples")

    summary = {
        "config": str(config_path),
        "prior_dir": str(prior_dir),
        "output_dir": str(output_dir),
        "decision": "PASS" if not failures else "FAIL",
        "expected_x_dim": EXPECTED_X_DIM,
        "expected_edge_attr_dim": EXPECTED_EDGE_DIM,
        "edge_feature_names": edge_names,
        "split_summaries": split_summaries,
        "edge_feature_stats": edge_stats,
        "model_forward_ok": forward_ok,
        "model_backward_ok": backward_ok,
        "vectorized_context_max_abs_diff": vectorized_context_max_abs_diff,
        "edge_context_gnn_diagnostics": first_forward_diag,
        "failures": failures,
        "warnings": warnings,
    }
    _write_json(output_dir / "edge_context_gnn_check_summary.json", summary)
    lines = [
        "# D16R-A5b Edge-Context GNN Check",
        "",
        f"- decision: `{summary['decision']}`",
        f"- expected_x_dim: `{EXPECTED_X_DIM}`",
        f"- expected_edge_attr_dim: `{EXPECTED_EDGE_DIM}`",
        f"- model_forward_ok: `{forward_ok}`",
        f"- model_backward_ok: `{backward_ok}`",
        f"- vectorized_context_max_abs_diff: `{vectorized_context_max_abs_diff}`",
        "",
        "## Edge Feature Stats",
        "| feature | mean | std | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in edge_stats.items():
        lines.append(f"| {name} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |")
    lines.extend(["", "## Edge-Context Diagnostics"])
    if first_forward_diag:
        for key, value in first_forward_diag.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Failures"])
    lines.extend([f"- {item}" for item in failures] if failures else ["- none"])
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- none"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "D16R_A5B_EDGE_CONTEXT_GNN_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
