from __future__ import annotations

from dataclasses import asdict, replace
import gc
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mpg_fer_v2_3.checkpoint import config_hash
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.evaluate import evaluate_raw_and_tta
from mpg_fer_v2_3.model import GeometryAwareMotifTransformerBlock, MPGFER


ROOT = Path(__file__).resolve().parents[3]
V22_SOURCE = ROOT / "research" / "mpg_fer_v2_2" / "src"
LOCKED_SCHEDULE = (8, 16, 16, 16, 24)
LOCKED_RESIDUAL_SCALES = (0.5, 0.5, 1.0, 1.0, 1.0)
V22_EQUIVALENT_RESIDUAL_SCALES = (1.0, 1.0, 1.0, 1.0, 1.0)


@pytest.fixture(autouse=True)
def _release_large_model_allocations():
    """Keep model-heavy regression tests isolated on memory-constrained hosts."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _v22_modules():
    sys.path.insert(0, str(V22_SOURCE))
    try:
        from mpg_fer_v2_2.config import MPGConfig as MPGConfigV22
        from mpg_fer_v2_2.model import MPGFER as MPGFERV22
    finally:
        sys.path.remove(str(V22_SOURCE))
    return MPGConfigV22, MPGFERV22


def _v22_state() -> dict[str, torch.Tensor]:
    MPGConfigV22, MPGFERV22 = _v22_modules()
    torch.manual_seed(42)
    return MPGFERV22(MPGConfigV22()).state_dict()


def test_scale_one_is_exact_v22_equivalence_with_strict_state_load() -> None:
    MPGConfigV22, MPGFERV22 = _v22_modules()
    state = _v22_state()
    equivalent = MPGFER(
        MPGConfig(motif_residual_scale_schedule=V22_EQUIVALENT_RESIDUAL_SCALES)
    ).eval()
    equivalent.load_state_dict(state, strict=True)
    reference = MPGFERV22(MPGConfigV22()).eval()
    reference.load_state_dict(state, strict=True)

    torch.manual_seed(42)
    images = torch.randn(2, 1, 48, 48)
    with torch.no_grad():
        reference_logits, _ = reference(images)
        equivalent_logits, _ = equivalent(images)

    assert torch.equal(reference_logits, equivalent_logits)
    assert (reference_logits - equivalent_logits).abs().max().item() == 0.0


def test_exact_selected_degree_per_sample_head_query_and_layer() -> None:
    model = MPGFER(MPGConfig(motif_topk_schedule=LOCKED_SCHEDULE)).eval()
    h_motif = torch.randn(2, 49, 192)
    geometry = torch.randn(2, 49, 49, 6)

    for layer_index, block in enumerate(model.motif_gnn):
        _, diagnostics = block(h_motif, geometry, return_diagnostics=True)
        selected = diagnostics["selected_mask"]
        expected_k = LOCKED_SCHEDULE[layer_index]
        assert selected.shape == (2, 6, 49, 49)
        assert torch.all(selected.sum(dim=-1) == expected_k)
        assert not selected.diagonal(dim1=-2, dim2=-1).any()


def test_exact_ties_still_select_exactly_k_without_score_perturbation() -> None:
    block = GeometryAwareMotifTransformerBlock(
        d_motif=12, geom_dim=6, num_heads=3, dropout=0.0, topk=8
    ).eval()
    with torch.no_grad():
        block.q_proj.weight.zero_()
        block.q_proj.bias.zero_()
        block.k_proj.weight.zero_()
        block.k_proj.bias.zero_()
        block.geom_proj.weight.zero_()
        block.geom_proj.bias.zero_()
    _, diagnostics = block(
        torch.zeros(2, 49, 12),
        torch.zeros(2, 49, 49, 6),
        return_diagnostics=True,
    )
    selected = diagnostics["selected_mask"]
    assert torch.all(selected.sum(dim=-1) == 8)
    assert not selected.diagonal(dim1=-2, dim2=-1).any()
    assert diagnostics["boundary_tie_count"].item() > 0


def test_pre_dropout_attention_is_normalized_and_excluded_edges_are_zero() -> None:
    block = GeometryAwareMotifTransformerBlock(
        d_motif=192, dropout=0.5, topk=8
    ).train()
    output, diagnostics = block(
        torch.randn(2, 49, 192),
        torch.randn(2, 49, 49, 6),
        return_diagnostics=True,
    )
    attention = diagnostics["attention_pre_dropout"]
    selected = diagnostics["selected_mask"]
    assert torch.isfinite(output).all()
    assert torch.allclose(
        attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1))
    )
    assert torch.count_nonzero(attention.masked_select(~selected)) == 0


def test_fp32_masking_is_finite_and_exactly_zero_off_support() -> None:
    block = GeometryAwareMotifTransformerBlock(topk=8).eval()
    _, diagnostics = block(
        torch.randn(1, 49, 192),
        torch.randn(1, 49, 49, 6),
        return_diagnostics=True,
    )
    attention = diagnostics["attention_pre_dropout"]
    selected = diagnostics["selected_mask"]
    assert torch.isfinite(attention).all()
    assert torch.count_nonzero(attention.masked_select(~selected)) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_amp_fp16_masking_forward_backward_is_finite() -> None:
    model = MPGFER(MPGConfig(motif_topk_schedule=LOCKED_SCHEDULE)).cuda().train()
    images = torch.randn(2, 1, 48, 48, device="cuda")
    targets = torch.tensor([0, 1], device="cuda")
    with torch.amp.autocast("cuda", enabled=True):
        logits, _ = model(images)
        loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_repeated_eval_forward_is_deterministic() -> None:
    torch.manual_seed(9)
    model = MPGFER(MPGConfig(motif_topk_schedule=LOCKED_SCHEDULE)).eval()
    images = torch.randn(2, 1, 48, 48)
    with torch.no_grad():
        logits_first, outputs_first = model(images)
        logits_second, outputs_second = model(images)
    assert torch.equal(logits_first, logits_second)
    for layer_index in range(1, 6):
        assert torch.equal(
            outputs_first[f"motif_l{layer_index}_entropy"],
            outputs_second[f"motif_l{layer_index}_entropy"],
        )
        assert torch.equal(
            outputs_first[f"motif_l{layer_index}_boundary_tie_count"],
            outputs_second[f"motif_l{layer_index}_boundary_tie_count"],
        )


def test_selected_qkv_geometry_paths_receive_finite_nonzero_gradients() -> None:
    torch.manual_seed(12)
    model = MPGFER(MPGConfig(motif_topk_schedule=LOCKED_SCHEDULE)).train()
    logits, _ = model(torch.randn(2, 1, 48, 48))
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()

    for layer_index, block in enumerate(model.motif_gnn):
        for name, parameter in (
            ("q_proj", block.q_proj.weight),
            ("k_proj", block.k_proj.weight),
            ("v_proj", block.v_proj.weight),
            ("geom_proj", block.geom_proj.weight),
        ):
            assert parameter.grad is not None, (
                f"layer {layer_index + 1} {name} has no gradient"
            )
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum().item() > 0.0


def test_parameter_count_and_per_module_reconciliation_match_v22() -> None:
    MPGConfigV22, MPGFERV22 = _v22_modules()
    v22 = MPGFERV22(MPGConfigV22())
    v23 = MPGFER(MPGConfig())
    v22_parameters = {name: value.numel() for name, value in v22.named_parameters()}
    v23_parameters = {name: value.numel() for name, value in v23.named_parameters()}
    assert v23_parameters == v22_parameters
    assert sum(v23_parameters.values()) == 2_304_528


def test_full_config_matches_v22_except_residual_scale_schedule() -> None:
    MPGConfigV22, _ = _v22_modules()
    v22 = asdict(MPGConfigV22())
    v23 = asdict(MPGConfig())
    assert v23.pop("motif_residual_scale_schedule") == LOCKED_RESIDUAL_SCALES
    assert v23 == v22


def test_residual_intervention_is_localized_to_motif_layers_one_and_two() -> None:
    model = MPGFER(MPGConfig())
    assert tuple(layer.residual_scale for layer in model.motif_gnn) == (
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
    )
    assert tuple(layer.topk for layer in model.motif_gnn) == LOCKED_SCHEDULE


@pytest.mark.parametrize(
    "schedule",
    [
        (),
        (8, 16, 16, 16),
        (8, 16, 16, 16, 24, 24),
        (0, 16, 16, 16, 24),
        (8, 16, 16, 16, 49),
        (8, 16, 16, 16, 24.0),
        (True, 16, 16, 16, 24),
    ],
)
def test_invalid_schedule_fails_closed(schedule) -> None:
    with pytest.raises(ValueError):
        MPGConfig(motif_topk_schedule=schedule)


@pytest.mark.parametrize(
    "schedule",
    [
        (),
        (0.5, 0.5, 1.0, 1.0),
        (0.5, 0.5, 1.0, 1.0, 1.0, 1.0),
        (0.0, 0.5, 1.0, 1.0, 1.0),
        (0.5, 0.5, 1.0, 1.0, 1.1),
        (0.5, 0.5, 1.0, 1.0, True),
        (0.5, 0.5, 1.0, 1.0, "1.0"),
    ],
)
def test_invalid_residual_scale_schedule_fails_closed(schedule) -> None:
    with pytest.raises(ValueError):
        MPGConfig(motif_residual_scale_schedule=schedule)


@pytest.mark.parametrize("topk", [0, -1, 1.5, True])
def test_invalid_block_topk_fails_closed(topk) -> None:
    with pytest.raises(ValueError):
        GeometryAwareMotifTransformerBlock(topk=topk)


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.1, True, "0.5"])
def test_invalid_block_residual_scale_fails_closed(scale) -> None:
    with pytest.raises(ValueError):
        GeometryAwareMotifTransformerBlock(residual_scale=scale)


def test_block_rejects_topk_above_available_non_self_keys() -> None:
    block = GeometryAwareMotifTransformerBlock(
        d_motif=12, num_heads=3, topk=9
    )
    with pytest.raises(ValueError, match="exceeds"):
        block(torch.randn(1, 9, 12), torch.randn(1, 9, 9, 6))


def test_config_hash_includes_scientific_schedules_and_schema_remains_three() -> None:
    config = MPGConfig()
    changed_topk = replace(config, motif_topk_schedule=(16, 16, 16, 16, 24))
    changed_residual = replace(
        config, motif_residual_scale_schedule=V22_EQUIVALENT_RESIDUAL_SCALES
    )
    assert (
        config.resume_schema_version
        == changed_topk.resume_schema_version
        == changed_residual.resume_schema_version
        == 3
    )
    assert config_hash(config) != config_hash(changed_topk)
    assert config_hash(config) != config_hash(changed_residual)


def test_tta_evaluation_path_works_with_sparse_model() -> None:
    model = MPGFER(MPGConfig(motif_topk_schedule=LOCKED_SCHEDULE)).eval()
    loader = DataLoader(
        TensorDataset(torch.rand(2, 1, 48, 48), torch.tensor([0, 1])),
        batch_size=2,
    )
    metrics = evaluate_raw_and_tta(model, loader, device="cpu", use_amp=False)
    assert set(metrics) == {"raw", "tta"}
    assert metrics["raw"]["support"] == metrics["tta"]["support"]
    assert 0.0 <= metrics["tta"]["accuracy"] <= 1.0
