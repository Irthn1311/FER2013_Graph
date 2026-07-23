import json

import numpy as np
import torch

from _helpers import ROOT, golden_array, golden_batch, loaded_model


def test_forward_matches_parent_golden_logits():
    with torch.no_grad():
        output = loaded_model()(golden_batch())
    actual = output["logits"].numpy()
    expected = golden_array("logits.npy")
    manifest = json.loads(
        (ROOT / "validation_assets/manifest.json").read_text(encoding="utf-8")
    )
    tolerance = float(manifest["tolerances"]["maximum_allowed"])

    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    assert np.array_equal(actual.argmax(1), expected.argmax(1))
