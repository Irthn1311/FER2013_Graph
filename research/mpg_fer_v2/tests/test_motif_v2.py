from __future__ import annotations

import math
import torch

from mpg_fer_v2.losses import motif_mutual_information_loss, symmetric_js_divergence
from mpg_fer_v2.model import GeometryAwareMotifTransformerBlock, compute_motif_geometry
from mpg_fer_v2.motif import (
    SpatialMotifComposer,
    aligned_logical_starts,
    build_aligned_support_indices,
)


def test_learnable_temperature_is_bounded_and_initialized() -> None:
    composer = SpatialMotifComposer()
    assert torch.isclose(composer.temperature, torch.tensor(0.70), atol=1e-6)
    with torch.no_grad():
        composer.raw_tau.fill_(-1_000.0)
    assert composer.tau_min <= float(composer.temperature.detach()) <= composer.tau_max
    with torch.no_grad():
        composer.raw_tau.fill_(1_000.0)
    assert composer.tau_min <= float(composer.temperature.detach()) <= composer.tau_max


def test_mi_entropy_math_uniform_and_specialized() -> None:
    uniform = torch.full((2, 5, 4), 0.25)
    loss, diagnostics = motif_mutual_information_loss(uniform)
    assert torch.isclose(diagnostics["H_local_normalized"], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(diagnostics["H_global_normalized"], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    collapsed = torch.zeros(1, 8, 4)
    collapsed[..., 0] = 1.0
    collapsed_loss, collapsed_diagnostics = motif_mutual_information_loss(collapsed)
    assert collapsed_diagnostics["H_local_normalized"] < 1e-5
    assert collapsed_diagnostics["H_global_normalized"] < 1e-5
    assert torch.isclose(collapsed_loss, torch.tensor(0.0), atol=1e-5)
    specialized = torch.eye(4).view(1, 4, 4)
    specialized_loss, specialized_diagnostics = motif_mutual_information_loss(specialized)
    assert specialized_diagnostics["H_local_normalized"] < 1e-5
    assert torch.isclose(specialized_diagnostics["H_global_normalized"], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(specialized_loss, torch.tensor(-1.0), atol=1e-5)
    assert specialized_loss < collapsed_loss
    assert specialized_loss < loss


def test_mi_gradients_reach_prototypes_and_assignment_projections() -> None:
    composer = SpatialMotifComposer(d_pixel=12, d_type=4, d_motif=16)
    pixels = torch.randn(2, 2304, 12, requires_grad=True)
    _, _, diagnostics = composer(pixels)
    diagnostics["loss_mi"].backward()
    for parameter in (
        composer.prototypes, composer.assignment_query.weight,
        composer.prototype_key.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.any(parameter.grad != 0)


def test_multiscale_support_shapes_and_aligned_conceptual_centers() -> None:
    supports, centers = build_aligned_support_indices()
    assert supports[8].shape == (49, 64)
    assert supports[12].shape == (49, 144)
    assert supports[16].shape == (49, 256)
    expected_axis = torch.tensor([5.5, 11.5, 17.5, 23.5, 29.5, 35.5, 41.5])
    assert torch.equal(centers[:, 0].unique(), expected_axis)
    assert torch.equal(centers[:, 1].unique(), expected_axis)
    assert all(int(index.min()) >= 0 and int(index.max()) < 2304 for index in supports.values())
    logical = aligned_logical_starts()
    # top-left, center, and bottom-right anchors preserve s+2, s, s-2.
    for anchor, medium_start in ((0, 0), (24, 18), (48, 36)):
        assert logical[8][anchor].tolist() == [medium_start + 2, medium_start + 2]
        assert logical[12][anchor].tolist() == [medium_start, medium_start]
        assert logical[16][anchor].tolist() == [medium_start - 2, medium_start - 2]


def test_coarse_border_reflection_is_symmetric_not_wrap_or_clamp() -> None:
    supports, _ = build_aligned_support_indices()
    coarse = supports[16]
    top_left_rows = (coarse[0] // 48).view(16, 16)[:, 0].tolist()
    top_left_columns = (coarse[0] % 48).view(16, 16)[0].tolist()
    bottom_right_rows = (coarse[-1] // 48).view(16, 16)[:, 0].tolist()
    bottom_right_columns = (coarse[-1] % 48).view(16, 16)[0].tolist()
    assert top_left_rows == [2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert top_left_columns == top_left_rows
    assert bottom_right_rows == [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 46, 45]
    assert bottom_right_columns == bottom_right_rows


def test_scale_weights_sum_to_one_and_output_stays_at_49_nodes() -> None:
    composer = SpatialMotifComposer(d_pixel=12, d_type=4, d_motif=16)
    motif, assignments, diagnostics = composer(torch.randn(2, 2304, 12))
    assert motif.shape == (2, 49, 16)
    assert assignments.shape == (2, 2304, 48)
    assert diagnostics["scale_weights"].shape == (2, 49, 3)
    assert torch.allclose(diagnostics["scale_weights"].sum(-1), torch.ones(2, 49), atol=1e-6)


def test_fused_center_keeps_gradient_to_occurrence_pooling_and_scale_gate() -> None:
    composer = SpatialMotifComposer(d_pixel=12, d_type=4, d_motif=16)
    _, _, diagnostics = composer(torch.randn(2, 2304, 12))
    objective = diagnostics["learned_centers_x"].square().mean() + diagnostics["learned_centers_y"].square().mean()
    objective.backward()
    assert composer.scale_gate.weight.grad is not None
    assert torch.any(composer.scale_gate.weight.grad != 0)
    assert composer.scale_saliency["8"].weight.grad is not None
    assert torch.any(composer.scale_saliency["8"].weight.grad != 0)


def test_motif_graph_geometry_loss_reaches_scale_gate_and_occurrence_weights() -> None:
    torch.manual_seed(13)
    composer = SpatialMotifComposer(d_pixel=12, d_type=4, d_motif=24)
    block = GeometryAwareMotifTransformerBlock(
        d_motif=24, num_heads=6, dropout=0.0
    ).eval()
    motif, _, diagnostics = composer(torch.randn(2, 2304, 12))
    geometry = compute_motif_geometry(
        diagnostics["learned_centers_x"], diagnostics["learned_centers_y"]
    )
    # Detach node content so composer gradients can only travel through geometry.
    output = block(motif.detach(), geometry)
    coefficients = torch.linspace(0.2, 1.1, output.numel()).reshape_as(output)
    (output * coefficients).sum().backward()
    assert composer.scale_gate.weight.grad is not None
    assert torch.any(composer.scale_gate.weight.grad != 0)
    for scale in ("8", "12", "16"):
        gradient = composer.scale_saliency[scale].weight.grad
        assert gradient is not None and torch.any(gradient != 0)


def test_raw_temperature_has_gradient_and_is_part_of_ema_state() -> None:
    from mpg_fer_v2.ema import ModelEMA

    composer = SpatialMotifComposer(d_pixel=12, d_type=4, d_motif=16)
    ema = ModelEMA(composer, decay=0.5)
    _, _, diagnostics = composer(torch.randn(2, 2304, 12))
    diagnostics["loss_mi"].backward()
    assert composer.raw_tau.grad is not None
    assert torch.isfinite(composer.raw_tau.grad)
    assert "raw_tau" in ema.module.state_dict()
    before = ema.module.raw_tau.detach().clone()
    with torch.no_grad():
        composer.raw_tau.add_(0.4)
    ema.update(composer)
    assert torch.allclose(
        ema.module.raw_tau, before * 0.5 + composer.raw_tau.detach() * 0.5
    )


def test_flip_consistency_is_finite_differentiable_and_zero_for_equal_logits() -> None:
    logits_a = torch.randn(4, 7, requires_grad=True)
    logits_b = torch.randn(4, 7, requires_grad=True)
    loss = symmetric_js_divergence(logits_a, logits_b)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert logits_a.grad is not None and logits_b.grad is not None
    same = symmetric_js_divergence(logits_a.detach(), logits_a.detach())
    assert torch.isclose(same, torch.tensor(0.0), atol=1e-7)
