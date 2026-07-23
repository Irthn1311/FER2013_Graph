import tempfile
from pathlib import Path

import torch

from _helpers import loaded_model


def test_checkpoint_roundtrip():
    model = loaded_model()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "bounded.pt"
        torch.save({"model_state": model.state_dict(), "epoch": 0}, path)
        state = torch.load(path, map_location="cpu", weights_only=False)["model_state"]
        assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
