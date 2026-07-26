"""Export converted standalone weights without requiring PyTorch."""

from __future__ import annotations

from pathlib import Path

from lap_gnn_tf.conversion.state_mapping import load_pytorch_npz
from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import build_model


def convert(state_path: str | Path, graph_batch_path: str | Path, output_path: str | Path) -> dict:
    batch = load_golden_batch(str(graph_batch_path))
    model = build_model(batch)
    result = load_pytorch_npz(model, state_path, strict=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(output_path)
    result["output_path"] = str(output_path)
    return result

