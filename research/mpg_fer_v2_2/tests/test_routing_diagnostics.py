from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from mpg_fer_v2_2.model import GeometryAwareMotifTransformerBlock
from mpg_fer_v2_2.train import (
    FIXED_ROUTING_DIAGNOSTIC_INDICES,
    ROUTING_DIAGNOSTIC_FIELDS,
    _write_routing_diagnostics,
    build_fixed_routing_diagnostic_batch,
    collect_fixed_routing_diagnostics,
)


class TinyRoutingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, return_routing_supports=False):
        logits = self.projection(self.dropout(images))
        outputs = {}
        support = torch.tensor(
            [[[[1, 2], [0, 2], [0, 1], [0, 1]]]],
            dtype=torch.int16,
            device=images.device,
        ).expand(images.shape[0], -1, -1, -1)
        for layer in range(1, 6):
            prefix = f"motif_l{layer}"
            outputs.update({
                f"{prefix}_selected_k": torch.tensor(2, device=images.device),
                f"{prefix}_entropy": torch.tensor(0.5, device=images.device),
                f"{prefix}_top1_mass": torch.tensor(0.75, device=images.device),
                f"{prefix}_boundary_tie_count": torch.tensor(0, device=images.device),
                f"{prefix}_local_share": torch.tensor(0.5, device=images.device),
                f"{prefix}_meso_share": torch.tensor(0.25, device=images.device),
                f"{prefix}_far_share": torch.tensor(0.25, device=images.device),
                f"{prefix}_edge_universe_coverage": torch.tensor(0.5, device=images.device),
            })
            if return_routing_supports:
                outputs[f"{prefix}_selected_indices"] = support
        return logits, outputs


class ArrayDataset:
    def __init__(self) -> None:
        self.images = np.arange(20 * 48 * 48, dtype=np.uint8).reshape(20, 48, 48)

    def __len__(self) -> int:
        return len(self.images)


def test_fixed_batch_bypasses_augmentation_and_uses_registered_indices() -> None:
    dataset = ArrayDataset()
    indices, images = build_fixed_routing_diagnostic_batch(dataset)
    assert indices == FIXED_ROUTING_DIAGNOSTIC_INDICES
    assert images.shape == (16, 1, 48, 48)
    assert torch.equal(
        images[7, 0], torch.from_numpy(dataset.images[7]).float() / 255.0
    )


def test_fixed_diagnostic_preserves_rng_modes_and_gradients() -> None:
    model = TinyRoutingModel().train()
    model.dropout.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes_before = [module.training for module in model.modules()]
    grads_before = [parameter.grad.clone() for parameter in model.parameters()]
    rng_before = torch.get_rng_state().clone()

    first, supports = collect_fixed_routing_diagnostics(
        model, torch.ones(2, 3), "cpu"
    )

    assert torch.equal(torch.get_rng_state(), rng_before)
    assert [module.training for module in model.modules()] == modes_before
    for parameter, expected in zip(model.parameters(), grads_before):
        assert torch.equal(parameter.grad, expected)
    assert all(
        not value.requires_grad
        for value in supports.values()
    )

    second, _ = collect_fixed_routing_diagnostics(
        model, torch.ones(2, 3), "cpu", supports
    )
    for layer in first["layers"]:
        assert first["layers"][layer]["support_jaccard_previous_epoch"] is None
        assert second["layers"][layer]["support_jaccard_previous_epoch"] == 1.0
        assert second["layers"][layer]["support_turnover_previous_epoch"] == 0.0


def test_spatial_shares_use_occurrence_grid_chebyshev_bins() -> None:
    block = GeometryAwareMotifTransformerBlock(
        d_motif=12, geom_dim=6, num_heads=3, dropout=0.0, topk=48
    ).eval()
    _, diagnostics = block(
        torch.randn(1, 49, 12),
        torch.randn(1, 49, 49, 6),
        return_diagnostics=True,
    )
    shares = [
        diagnostics["local_share"],
        diagnostics["meso_share"],
        diagnostics["far_share"],
    ]
    assert sum(float(value) for value in shares) == pytest.approx(1.0)
    assert float(diagnostics["edge_universe_coverage"]) == 1.0
    assert int(diagnostics["topk"]) == 48
    for name in (
        "topk", "entropy", "top1_mass", "boundary_tie_count",
        "local_share", "meso_share", "far_share", "edge_universe_coverage",
    ):
        assert not diagnostics[name].requires_grad


def test_routing_artifact_contains_best_final_and_full_trajectory(tmp_path) -> None:
    entry = {
        "epoch": 1,
        "routing_diagnostics": {"layers": {}},
    }
    for layer in range(1, 6):
        prefix = f"motif_l{layer}"
        fixed = {field: 0.0 for field in ROUTING_DIAGNOSTIC_FIELDS}
        fixed.update({
            "support_jaccard_previous_epoch": None,
            "support_turnover_previous_epoch": None,
        })
        entry["routing_diagnostics"]["layers"][prefix] = fixed
        for field in ROUTING_DIAGNOSTIC_FIELDS:
            entry[f"{prefix}_{field}"] = 0.0
    _write_routing_diagnostics([entry], tmp_path, best_epoch=1)
    payload = json.loads((tmp_path / "routing_diagnostics.json").read_text())
    assert payload["best_epoch"] == payload["final_epoch"] == 1
    assert payload["checkpoint_selection_used_routing_diagnostics"] is False
    assert payload["fixed_batch"]["sample_indices"] == list(range(16))
    assert len(payload["trajectory"]) == 1
