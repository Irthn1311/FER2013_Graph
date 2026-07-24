from pathlib import Path

import numpy as np

from lap_gnn_tf.conversion import load_pytorch_npz
from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import build_model


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "validation_assets" / "golden"


def loaded():
    batch = load_golden_batch(str(GOLDEN / "graph_batch.npz"))
    model = build_model(batch)
    load_pytorch_npz(model, GOLDEN / "model_state.npz")
    return model, batch


def golden(name):
    return np.load(GOLDEN / name, allow_pickle=False)

