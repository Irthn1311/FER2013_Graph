"""Check D16R-A5c multi-scale EdgeContextGNN fusion wiring."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _dataset(cfg: Dict[str, Any], prior_dir: Path, split: str, max_samples: int) -> D16PixelPriorDataset:
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
            if not bool(torch.isfinite(value).all().item()):
                failures.append(f"diagnostic {key} is non-finite")
        elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
            failures.append(f"diagnostic {key} is non-finite")
    return failures


def run_check(config_path: Path, prior_dir: Path, output_dir: Path, max_samples: int = 4) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    model_cfg = cfg.get("model", {}) or {}
    edge_cfg = model_cfg.get("edge_context_gnn", {}) or {}
    fusion_cfg = edge_cfg.get("multiscale_fusion", {}) or {}
    hidden_dim = int(model_cfg.get("hidden_dim", 96))
    failures: List[str] = []
    warnings: List[str] = []

    if model_cfg.get("gnn_type") != "edge_context_gnn":
        failures.append(f"model.gnn_type must be edge_context_gnn, got {model_cfg.get('gnn_type')!r}")
    if not bool(edge_cfg.get("layer_output_concat", False)):
        failures.append("model.edge_context_gnn.layer_output_concat must be true for A5c")
    if not bool(fusion_cfg.get("enabled", False)):
        failures.append("model.edge_context_gnn.multiscale_fusion.enabled must be true for A5c")
    if list(fusion_cfg.get("layers") or []) != [1, 3]:
        failures.append(f"model.edge_context_gnn.multiscale_fusion.layers must be [1, 3], got {fusion_cfg.get('layers')!r}")
    if fusion_cfg.get("mode") != "concat_project":
        failures.append("model.edge_context_gnn.multiscale_fusion.mode must be concat_project")

    train_ds = _dataset(cfg, prior_dir, "train", max_samples=max_samples)
    val_ds = _dataset(cfg, prior_dir, "val", max_samples=max_samples)
    split_dims: Dict[str, Any] = {}
    for split, ds in [("train", train_ds), ("val", val_ds)]:
        graph = ds[0]
        if int(graph.x.size(1)) != EXPECTED_X_DIM:
            failures.append(f"{split}: x dim={int(graph.x.size(1))}, expected {EXPECTED_X_DIM}")
        if graph.edge_attr is None or int(graph.edge_attr.size(1)) != EXPECTED_EDGE_DIM:
            failures.append(f"{split}: edge_attr dim invalid")
        if graph.edge_attr is not None and not torch.isfinite(graph.edge_attr).all().item():
            failures.append(f"{split}: edge_attr has NaN/Inf")
        split_dims[split] = {
            "nodes": int(graph.x.size(0)),
            "edges": int(graph.edge_index.size(1)),
            "x_dim": int(graph.x.size(1)),
            "edge_attr_dim": None if graph.edge_attr is None else int(graph.edge_attr.size(1)),
        }

    loader = DataLoader(train_ds, batch_size=min(2, len(train_ds)), shuffle=False, collate_fn=collate_d16_graphs)
    batch = next(iter(loader))
    model = D16Model.from_config(cfg, input_dim=int(batch.x_cat.size(1)))
    gnn = getattr(model, "gnn", None)
    if not bool(getattr(gnn, "multiscale_enabled", False)):
        failures.append("constructed EdgeContextGNNEncoder.multiscale_enabled is false")
    if list(getattr(gnn, "multiscale_layers", [])) != [1, 3]:
        failures.append(f"constructed multiscale_layers={getattr(gnn, 'multiscale_layers', None)!r}, expected [1, 3]")

    model.eval()
    with torch.no_grad():
        eval_out = model(batch)
    logits = eval_out.get("logits")
    node_embeddings = eval_out.get("node_embeddings")
    diag = eval_out.get("edge_context_gnn_diagnostics") or {}
    forward_ok = torch.is_tensor(logits) and tuple(logits.shape) == (batch.num_graphs, 7)
    if not forward_ok:
        failures.append(f"logits shape invalid: {None if logits is None else tuple(logits.shape)}")
    elif not torch.isfinite(logits).all().item():
        failures.append("logits has NaN/Inf")
    fused_dim_ok = torch.is_tensor(node_embeddings) and int(node_embeddings.size(1)) == hidden_dim
    if not fused_dim_ok:
        failures.append(f"fused node embedding dim invalid: {None if node_embeddings is None else tuple(node_embeddings.shape)}")
    elif not torch.isfinite(node_embeddings).all().item():
        failures.append("fused node embeddings have NaN/Inf")
    fusion_enabled_diag = diag.get("multiscale_fusion_enabled", 0.0)
    if torch.is_tensor(fusion_enabled_diag):
        fusion_enabled_diag = float(fusion_enabled_diag.detach().cpu().item())
    if float(fusion_enabled_diag or 0.0) != 1.0:
        failures.append("multiscale_fusion_enabled diagnostic missing or not 1")
    failures.extend(_finite_diag(diag))

    model.train()
    train_out = model(batch)
    train_logits = train_out.get("logits")
    loss = train_logits.float().sum() if torch.is_tensor(train_logits) else torch.tensor(0.0)
    loss.backward()
    backward_ok = True
    if any(not torch.isfinite(p).all().item() for p in model.parameters() if p.grad is not None):
        failures.append("parameter gradient contains NaN/Inf")

    base_cfg_path = config_path.with_name("d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150.yaml")
    a5b_unaffected = None
    if base_cfg_path.exists():
        base_cfg = _load_yaml(base_cfg_path)
        base_edge = (base_cfg.get("model", {}) or {}).get("edge_context_gnn", {}) or {}
        a5b_unaffected = not bool(base_edge.get("layer_output_concat", False)) and not bool((base_edge.get("multiscale_fusion", {}) or {}).get("enabled", False))
        if not a5b_unaffected:
            failures.append("A5b seed42 config appears to enable multiscale fusion")
    else:
        warnings.append(f"Cannot verify A5b config unaffected; missing {base_cfg_path}")

    diag_json = {
        key: (float(value.detach().cpu().item()) if torch.is_tensor(value) and value.numel() == 1 else str(value))
        for key, value in diag.items()
    }
    summary = {
        "config": str(config_path),
        "prior_dir": str(prior_dir),
        "output_dir": str(output_dir),
        "decision": "PASS" if not failures else "FAIL",
        "expected_x_dim": EXPECTED_X_DIM,
        "expected_edge_attr_dim": EXPECTED_EDGE_DIM,
        "hidden_dim": hidden_dim,
        "split_dims": split_dims,
        "logits_shape": None if logits is None else list(logits.shape),
        "fused_node_embedding_shape": None if node_embeddings is None else list(node_embeddings.shape),
        "fused_node_embedding_dim_ok": bool(fused_dim_ok),
        "model_forward_ok": bool(forward_ok),
        "model_backward_ok": bool(backward_ok),
        "a4_readout_diagnostics_finite": not _finite_diag(train_out.get("micro_motif_diagnostics", {}) or {}),
        "a5b_config_unaffected": a5b_unaffected,
        "edge_context_gnn_diagnostics": diag_json,
        "failures": failures,
        "warnings": warnings,
    }
    _write_json(output_dir / "a5c_multiscale_check_summary.json", summary)
    lines = [
        "# D16R-A5c Multi-scale EdgeContextGNN Check",
        "",
        f"- decision: `{summary['decision']}`",
        f"- expected_x_dim: `{EXPECTED_X_DIM}`",
        f"- expected_edge_attr_dim: `{EXPECTED_EDGE_DIM}`",
        f"- hidden_dim: `{hidden_dim}`",
        f"- logits_shape: `{summary['logits_shape']}`",
        f"- fused_node_embedding_shape: `{summary['fused_node_embedding_shape']}`",
        f"- model_forward_ok: `{forward_ok}`",
        f"- model_backward_ok: `{backward_ok}`",
        f"- a5b_config_unaffected: `{a5b_unaffected}`",
        "",
        "## Diagnostics",
    ]
    lines.extend([f"- {key}: `{value}`" for key, value in diag_json.items()] or ["- none"])
    lines.extend(["", "## Failures"])
    lines.extend([f"- {item}" for item in failures] if failures else ["- none"])
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- none"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "D16R_A5C_MULTISCALE_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=4)
    args = parser.parse_args()
    summary = run_check(
        config_path=Path(args.config),
        prior_dir=Path(args.prior_dir),
        output_dir=Path(args.output_dir),
        max_samples=int(args.max_samples),
    )
    print(json.dumps(summary, indent=2, default=str))
    if summary["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
