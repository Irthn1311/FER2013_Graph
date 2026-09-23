from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mpg_fer_v2_2.checkpoint import config_hash
from mpg_fer_v2_2.config import MPGConfig
from mpg_fer_v2_2.evaluate import evaluate_raw_and_tta
from mpg_fer_v2_2.model import GeometryAwareMotifTransformerBlock, MPGFER


ROOT = Path(__file__).resolve().parents[3]
V21_SOURCE = ROOT / "research" / "mpg_fer_v2_1" / "src"
OFFICIAL_V21_CHECKPOINT = (
    ROOT
    / "research"
    / "mpg_fer_v2_1"
    / "official_runs"
    / "segment_02"
    / "mpg_fer_v2_1_run"
    / "best_val_acc.pt"
)
LOCKED_SCHEDULE = (8, 16, 16, 16, 24)


def _v21_modules():
    sys.path.insert(0, str(V21_SOURCE))
    try:
        from mpg_fer_v2_1.config import MPGConfig as MPGConfigV21
        from mpg_fer_v2_1.model import MPGFER as MPGFERV21
    finally:
        sys.path.remove(str(V21_SOURCE))
    return MPGConfigV21, MPGFERV21


def _v21_state() -> dict[str, torch.Tensor]:
    if OFFICIAL_V21_CHECKPOINT.is_file():
        checkpoint = torch.load(
            OFFICIAL_V21_CHECKPOINT, map_location="cpu", weights_only=False
        )
        return checkpoint["model_state_dict"]
    MPGConfigV21, MPGFERV21 = _v21_modules()
    torch.manual_seed(42)
    return MPGFERV21(MPGConfigV21()).state_dict()


def test_k48_dense_equivalence_and_strict_v21_load() -> None:
    MPGConfigV21, MPGFERV21 = _v21_modules()
    state = _v21_state()
    dense = MPGFER(
        MPGConfig(motif_topk_schedule=(48, 48, 48, 48, 48))
    ).eval()
    dense.load_state_dict(state, strict=True)
    reference = MPGFERV21(MPGConfigV21()).eval()
    reference.load_state_dict(state, strict=True)

    torch.manual_seed(42)
    images = torch.randn(2, 1, 48, 48)
    with torch.no_grad():
        reference_logits, _ = reference(images)
        dense_logits, _ = dense(images)

    assert torch.equal(reference_logits, dense_logits)
    assert (reference_logits - dense_logits).abs().max().item() <= 1e-6


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


def test_parameter_count_and_per_module_reconciliation_match_v21() -> None:
    MPGConfigV21, MPGFERV21 = _v21_modules()
    v21 = MPGFERV21(MPGConfigV21())
    v22 = MPGFER(MPGConfig(motif_topk_schedule=LOCKED_SCHEDULE))
    v21_parameters = {name: value.numel() for name, value in v21.named_parameters()}
    v22_parameters = {name: value.numel() for name, value in v22.named_parameters()}
    assert v22_parameters == v21_parameters
    assert sum(v22_parameters.values()) == 2_304_528


def test_full_config_matches_v21_except_sparse_schedule() -> None:
    MPGConfigV21, _ = _v21_modules()
    v21 = asdict(MPGConfigV21())
    v22 = asdict(MPGConfig())
    assert v22.pop("motif_topk_schedule") == LOCKED_SCHEDULE
    assert v22 == v21


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


@pytest.mark.parametrize("topk", [0, -1, 1.5, True])
def test_invalid_block_topk_fails_closed(topk) -> None:
    with pytest.raises(ValueError):
        GeometryAwareMotifTransformerBlock(topk=topk)


def test_block_rejects_topk_above_available_non_self_keys() -> None:
    block = GeometryAwareMotifTransformerBlock(
        d_motif=12, num_heads=3, topk=9
    )
    with pytest.raises(ValueError, match="exceeds"):
        block(torch.randn(1, 9, 12), torch.randn(1, 9, 9, 6))


def test_config_hash_includes_sparse_schedule_and_schema_remains_three() -> None:
    config = MPGConfig()
    changed = replace(config, motif_topk_schedule=(16, 16, 16, 16, 24))
    assert config.resume_schema_version == changed.resume_schema_version == 3
    assert config_hash(config) != config_hash(changed)


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
