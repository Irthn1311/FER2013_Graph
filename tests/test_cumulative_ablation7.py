"""Tests for MPG-FER 7-configuration cumulative ablation ladder."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from mpg_fer_cumulative_ablation7.model import (
    CUMULATIVE_ABLATION_ORDER,
    CUMULATIVE_REGISTRY,
    CumulativeAblationMode,
    CumulativeAblationMPGFER,
)
from mpg_fer_cumulative_ablation7.protocol import (
    CANONICAL_DATASET_HASHES,
    CANONICAL_DATASET_ROWS,
    compute_pairwise_configuration_diffs,
    validate_split_identity,
)
from mpg_fer_v2_3.checkpoint import sha256_file
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.model import MPGFER


def test_cumulative_ablation_registry_order() -> None:
    expected_order = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
    actual_order = tuple(m.value for m in CUMULATIVE_ABLATION_ORDER)
    assert actual_order == expected_order, f"Order mismatch: {actual_order} != {expected_order}"


def test_pairwise_configuration_diffs() -> None:
    diffs = compute_pairwise_configuration_diffs()
    assert len(diffs) == 6
    for d in diffs:
        assert d["status"] == "VALID"


def test_model_instantiation_and_invariants() -> None:
    cfg = MPGConfig()
    x = torch.randn(2, 1, 48, 48)

    for mode in CumulativeAblationMode:
        model = CumulativeAblationMPGFER(config=cfg, mode=mode).eval()
        with torch.no_grad():
            logits, outputs = model(x)

        assert logits.shape == (2, 7), f"{mode} logits shape mismatch: {logits.shape}"
        assert torch.isfinite(logits).all(), f"{mode} produced non-finite logits"
        assert outputs["fusion_representation"].shape == (2, 512)
        assert outputs["h_pixel_readout"].shape == (2, 128)
        assert outputs["h_motif_readout"].shape == (2, 384)


def test_a0_zero_motif_feature_shape() -> None:
    cfg = MPGConfig()
    x = torch.randn(2, 1, 48, 48)
    model = CumulativeAblationMPGFER(config=cfg, mode=CumulativeAblationMode.A0).eval()
    with torch.no_grad():
        logits, outputs = model(x)

    # Check motif readout is exactly zero
    assert (outputs["h_motif_readout"] == 0).all()
    assert outputs["motif_assignments"] is None
    assert outputs["motif_geometry"] is None
    # Pixel readout must be non-zero
    assert not (outputs["h_pixel_readout"] == 0).all()
    # Loss applicability
    assert model.spec.applicable_losses["motif_aux"] == 0.0
    assert model.spec.applicable_losses["pixel_aux"] == 0.05


def test_a1_no_motif_gnn() -> None:
    cfg = MPGConfig()
    x = torch.randn(2, 1, 48, 48)
    model = CumulativeAblationMPGFER(config=cfg, mode=CumulativeAblationMode.A1).eval()
    with torch.no_grad():
        logits, outputs = model(x)

    # In A1, occurrence nodes bypass motif GNN: initial == final
    assert torch.equal(outputs["h_motif_nodes_initial"], outputs["h_motif_nodes_final"])
    assert model.spec.composer == "single_scale_12"
    assert model.spec.composer_scales == [12]
    assert model.spec.motif_gnn is False


def test_a2_motif_gnn_dense_no_geom() -> None:
    cfg = MPGConfig()
    model = CumulativeAblationMPGFER(config=cfg, mode=CumulativeAblationMode.A2)
    assert model.spec.motif_gnn is True
    assert model.spec.geometry_bias is False
    assert model.spec.relation_mode == "dense"
    assert model.spec.motif_topk_schedule == [48, 48, 48, 48, 48]
    assert model.spec.composer_scales == [12]


def test_a3_geometry_bias_enabled() -> None:
    cfg = MPGConfig()
    model = CumulativeAblationMPGFER(config=cfg, mode=CumulativeAblationMode.A3)
    assert model.spec.motif_gnn is True
    assert model.spec.geometry_bias is True
    assert model.spec.relation_mode == "dense"
    assert model.spec.composer_scales == [12]


def test_a4_multiscale_enabled() -> None:
    cfg = MPGConfig()
    x = torch.randn(2, 1, 48, 48)
    model = CumulativeAblationMPGFER(config=cfg, mode=CumulativeAblationMode.A4).eval()
    with torch.no_grad():
        logits, outputs = model(x)

    assert model.spec.composer == "multi_scale_8_12_16"
    assert model.spec.composer_scales == [8, 12, 16]
    assert model.spec.relation_mode == "dense"
    assert outputs["scale_weights"] is not None
    assert outputs["scale_weights"].shape[-1] == 3


def test_a5_dynamic_topk_enabled() -> None:
    cfg = MPGConfig()
    model = CumulativeAblationMPGFER(config=cfg, mode=CumulativeAblationMode.A5)
    assert model.spec.relation_mode == "dynamic_topk"
    assert model.spec.motif_topk_schedule == [8, 16, 16, 16, 24]
    assert model.spec.motif_pooling == "fixed"


def test_a6_full_model_exact_parity() -> None:
    cfg = MPGConfig()
    x = torch.randn(3, 1, 48, 48)

    torch.manual_seed(12345)
    full_canonical = MPGFER(cfg).eval()
    torch.manual_seed(12345)
    a6_model = CumulativeAblationMPGFER(cfg, mode=CumulativeAblationMode.A6).eval()

    with torch.no_grad():
        can_logits, can_out = full_canonical(x)
        a6_logits, a6_out = a6_model(x)

    diff = (can_logits - a6_logits).abs().max().item()
    assert diff == 0.0, f"A6 must execute exact canonical FULL forward pass, got max diff={diff}"
    assert a6_model.spec.motif_pooling == "learnable_attention"


def test_a6_strict_checkpoint_load() -> None:
    official_ckpt = Path(
        r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3\official_runs\segment_02\mpg_fer_v2_3_run\best_val_acc.pt"
    )
    if not official_ckpt.is_file():
        pytest.skip("Official checkpoint not present on disk")

    digest = sha256_file(official_ckpt)
    assert digest == "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"

    payload = torch.load(official_ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)

    model = CumulativeAblationMPGFER(mode=CumulativeAblationMode.A6)
    incompatible = model.load_state_dict(state, strict=True)
    assert len(incompatible.missing_keys) == 0
    assert len(incompatible.unexpected_keys) == 0


def test_dataset_split_identities() -> None:
    data_dir = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\data")
    if not (data_dir / "train.csv").is_file():
        pytest.skip("Data directory not present")

    for fname in ("train.csv", "val.csv", "test.csv"):
        res = validate_split_identity(fname, data_dir / fname)
        assert res["status"] == "PASS"
        assert res["sha256"] == CANONICAL_DATASET_HASHES[fname]
        assert res["rows"] == CANONICAL_DATASET_ROWS[fname]
