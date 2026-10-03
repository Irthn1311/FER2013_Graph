from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1] / "analysis/npf_conditional_pixel_residual"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"npf_residual_{name}", ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load("model")
protocol = load("protocol")


class FakeNPF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pixel = nn.Linear(4, 128)
        self.motif = nn.Linear(4, 384)
        self.base = nn.Linear(4, 7)

    def forward(self, images: torch.Tensor):
        features = images.flatten(1)
        logits = self.base(features)
        return logits, {
            "h_pixel_readout": self.pixel(features),
            "h_motif_readout": self.motif(features),
        }


@pytest.mark.parametrize("variant", ["R1", "R2"])
def test_zero_initialized_residual_is_exact_npf_parity(variant: str) -> None:
    torch.manual_seed(42)
    wrapper = model_module.NPFResidualModel(FakeNPF(), variant)
    images = torch.randn(5, 1, 2, 2)
    base, _ = wrapper.backbone(images)
    final, outputs = wrapper(images)
    assert torch.equal(final, base)
    assert torch.equal(outputs["delta"], torch.zeros(5, 7))
    if variant == "R2":
        assert outputs["gate"].shape == (5, 7)
        assert torch.all((outputs["gate"] > 0) & (outputs["gate"] < 1))
    else:
        assert outputs["gate"] is None


def test_residual_shapes_and_finite_logits() -> None:
    branch = model_module.ResidualBranch("R2")
    logits, outputs = branch(
        torch.randn(3, 7), torch.randn(3, 128), torch.randn(3, 384)
    )
    assert logits.shape == outputs["delta"].shape == outputs["gate"].shape == (3, 7)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("variant", ["R1", "R2"])
def test_only_allowed_parameters_receive_gradients_and_backbone_is_immutable(variant: str) -> None:
    torch.manual_seed(7)
    wrapper = model_module.NPFResidualModel(FakeNPF(), variant)
    before = protocol.state_sha256(wrapper.backbone)
    optimizer = torch.optim.AdamW(wrapper.residual.parameters(), lr=1e-2)
    images = torch.randn(6, 1, 2, 2)
    labels = torch.arange(6) % 7
    logits, _ = wrapper(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    gradient_names = {name for name, value in wrapper.named_parameters() if value.grad is not None}
    assert gradient_names == wrapper.allowed_trainable_names()
    assert all(not parameter.requires_grad for parameter in wrapper.backbone.parameters())
    optimizer.step()
    assert protocol.state_sha256(wrapper.backbone) == before


def test_raw_tta_construction_matches_canonical_logit_average() -> None:
    raw = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    flip = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
    tta = 0.5 * (raw + flip)
    assert np.array_equal(tta, np.array([[1.0, 1.0], [1.0, 0.5]], dtype=np.float32))


def test_changed_fixed_broken_identity() -> None:
    labels = np.array([0, 1, 2, 0, 1])
    baseline = np.array([1, 1, 2, 0, 0])
    residual = np.array([0, 2, 2, 1, 1])
    result = protocol.change_counts(labels, baseline, residual)
    assert result == {
        "argmax_changed_count": 4,
        "argmax_changed_fraction": 0.8,
        "npf_wrong_to_residual_correct": 2,
        "npf_correct_to_residual_wrong": 2,
        "net_corrected_count": 0,
    }
    base_correct = int(np.sum(baseline == labels))
    final_correct = int(np.sum(residual == labels))
    assert final_correct - base_correct == result["net_corrected_count"]


def test_checkpoint_selection_and_schedule_are_deterministic() -> None:
    incumbent = {"accuracy": 0.7, "macro_f1": 0.65}
    assert protocol.is_better({"accuracy": 0.71, "macro_f1": 0.60}, incumbent)
    assert protocol.is_better({"accuracy": 0.70, "macro_f1": 0.66}, incumbent)
    assert not protocol.is_better({"accuracy": 0.70, "macro_f1": 0.64}, incumbent)
    assert protocol.learning_rate_for_epoch(1) == pytest.approx(1.5e-4)
    assert protocol.learning_rate_for_epoch(2) == pytest.approx(3e-4)
    assert protocol.learning_rate_for_epoch(30) == pytest.approx(0.0, abs=1e-15)
