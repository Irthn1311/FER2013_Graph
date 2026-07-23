import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from _helpers import golden_batch, loaded_model


def test_bounded_two_step_smoke():
    torch.set_num_threads(1)
    model = loaded_model(eval_mode=False)
    batch = golden_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)["logits"]
        loss = F.cross_entropy(logits, batch.y)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        optimizer.step()
        losses.append(float(loss.detach()))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "smoke.pt"
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "steps": 2}, path)
        assert torch.load(path, map_location="cpu", weights_only=False)["steps"] == 2
    assert len(losses) == 2
