import numpy as np

from _helpers import GOLDEN, loaded


def test_readout_parity():
    model, batch = loaded()
    actual = model(batch, training=False, collect_intermediates=True)["intermediates"]
    names = [
        "micro_major_motif_tokens", "micro_major_motif_transformed_tokens",
        "micro_motif_tokens", "micro_motif_transformed_tokens",
        "micro_support_gate", "pooled_graph_embedding",
    ]
    with np.load(GOLDEN / "layer_outputs.npz") as expected:
        for name in names:
            assert np.max(np.abs(actual[name].numpy() - expected[name])) <= 1e-5

