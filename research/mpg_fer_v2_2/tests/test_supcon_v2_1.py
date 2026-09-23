from __future__ import annotations

import torch
import torch.nn.functional as F

from mpg_fer_v2_2.losses import supervised_contrastive_loss
from mpg_fer_v2_2.model import MPGFER


def test_same_label_positives_and_different_label_negatives_are_finite() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss, stats = supervised_contrastive_loss(embeddings, labels, temperature=0.10)
    assert torch.isfinite(loss) and loss > 0
    assert stats["valid_supcon_anchor_fraction"] == 1.0
    assert stats["mean_positive_count"] == 1.0
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert torch.any(embeddings.grad != 0)


def test_self_pairs_are_excluded() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    loss, stats = supervised_contrastive_loss(
        embeddings, torch.tensor([3, 3]), temperature=0.10
    )
    # Each anchor sees only the other sample. Its positive probability is one.
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)
    assert stats["mean_positive_count"] == 1.0


def test_anchors_without_positives_are_skipped() -> None:
    embeddings = torch.randn(3, 8, requires_grad=True)
    loss, stats = supervised_contrastive_loss(
        embeddings, torch.tensor([0, 0, 1]), temperature=0.10
    )
    assert torch.isfinite(loss)
    assert stats["valid_supcon_anchor_fraction"] == torch.tensor(2 / 3)
    assert stats["mean_positive_count"] == 1.0


def test_all_unique_labels_return_differentiable_zero() -> None:
    embeddings = torch.randn(4, 8, requires_grad=True)
    loss, stats = supervised_contrastive_loss(
        embeddings, torch.arange(4), temperature=0.10
    )
    assert loss.requires_grad
    assert loss == 0.0
    assert stats["valid_supcon_anchor_fraction"] == 0.0
    assert stats["mean_positive_count"] == 0.0
    loss.backward()
    assert embeddings.grad is not None
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


def test_supcon_gradient_reaches_fusion_projection_and_both_readout_branches() -> None:
    torch.manual_seed(21)
    model = MPGFER()
    pixel_readout = torch.randn(4, 128, requires_grad=True)
    motif_readout = torch.randn(4, 384, requires_grad=True)
    fusion = torch.cat([pixel_readout, motif_readout], dim=-1)
    fusion.retain_grad()
    embeddings = F.normalize(model.supcon_head(fusion), dim=-1)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(4), atol=1e-6)
    loss, _ = supervised_contrastive_loss(
        embeddings, torch.tensor([0, 0, 1, 1]), temperature=0.10
    )
    loss.backward()
    for gradient in (
        fusion.grad,
        pixel_readout.grad,
        motif_readout.grad,
        model.supcon_head[0].weight.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.any(gradient != 0)
