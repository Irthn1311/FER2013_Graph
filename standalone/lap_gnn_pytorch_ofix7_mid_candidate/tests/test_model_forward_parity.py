import numpy as np
import torch

from _helpers import golden_array, golden_batch, loaded_model


def test_forward_matches_parent_golden_logits():
    with torch.no_grad():
        output = loaded_model()(golden_batch())
    expected = golden_array("logits.npy")
    np.testing.assert_allclose(output["logits"].numpy(), expected, rtol=0, atol=1e-6)
    assert np.array_equal(output["logits"].argmax(1).numpy(), expected.argmax(1))
