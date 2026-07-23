"""Standalone package validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lap_gnn.cli.common import add_data_arguments, validate_inputs
from lap_gnn.constants import PARAMETER_COUNT
from lap_gnn.data.pixel_prior_dataset import D16PixelPriorDataset
from lap_gnn.model.d16_model import D16Model


def main() -> None:
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    parser.add_argument("--package-root")
    args = parser.parse_args()
    cfg, prior_root = validate_inputs(args)
    dataset = D16PixelPriorDataset(
        prior_root, split="train",
        graph_mode=cfg["graph"]["graph_mode"],
        face_threshold=cfg["graph"]["face_threshold"],
        context_pixels=cfg["graph"]["context_pixels"],
        detail_features=cfg["graph"]["detail_features"],
        edge_features=cfg["graph"]["edge_features"],
        anchor_nodes=cfg["graph"]["anchor_nodes"],
        prior_corruption={"enabled": False},
        max_samples=1,
    )
    graph = dataset[0]
    model = D16Model.from_config(cfg, input_dim=graph.x.shape[1])
    count = sum(parameter.numel() for parameter in model.parameters())
    failures = []
    if graph.x.shape[1] != 37:
        failures.append(f"node_dim={graph.x.shape[1]}")
    if graph.edge_attr is None or graph.edge_attr.shape[1] != 8:
        failures.append("edge_dim")
    if count != PARAMETER_COUNT:
        failures.append(f"parameter_count={count}")
    package_root = Path(args.package_root).resolve() if args.package_root else Path(__file__).resolve().parents[4]
    manifest = package_root / "package_manifest.json"
    checksums = package_root / "CHECKSUMS.sha256"
    if not manifest.is_file():
        failures.append("package_manifest")
    if not checksums.is_file():
        failures.append("checksums")
    decision = "READY_FOR_STANDALONE_SMOKE" if not failures else "HOLD"
    print(json.dumps({
        "decision": decision,
        "failures": failures,
        "node_dim": int(graph.x.shape[1]),
        "edge_dim": int(graph.edge_attr.shape[1]),
        "parameter_count": count,
        "torch": torch.__version__,
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
