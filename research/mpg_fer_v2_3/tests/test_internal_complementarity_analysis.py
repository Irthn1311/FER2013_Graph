from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


CORE = Path(__file__).resolve().parents[1] / "analysis/full_internal_complementarity/core.py"
SPEC = importlib.util.spec_from_file_location("internal_complementarity_core", CORE)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def test_tta_and_pair_partition_are_exact() -> None:
    labels = np.array([0, 1, 2, 0])
    raw = np.array([[4, 0, 0], [0, 4, 0], [0, 3, 2], [0, 4, 0]], dtype=np.float32)
    flip = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 4], [4, 0, 0]], dtype=np.float32)
    diagnostic = core.head_diagnostics(raw, flip)
    assert np.array_equal(diagnostic["tta_logits"], 0.5 * (raw + flip))
    first = np.array([0, 1, 1, 1])
    second = np.array([0, 2, 2, 0])
    result = core.disagreement(labels, first, second)
    assert [result[key] for key in ("N_CC", "N_CW", "N_WC", "N_WW")] == [1, 1, 2, 0]
    assert sum(result[key] for key in ("N_CC", "N_CW", "N_WC", "N_WW")) == len(labels)
    assert core.oracle(labels, first, second)["correct_count"] == 4
    assert np.all(diagnostic["tta_margin"] >= 0)
    assert np.all(diagnostic["tta_entropy"] >= 0)
    assert np.all(diagnostic["flip_js_divergence"] >= 0)
    identical = core.head_diagnostics(raw, raw)
    assert np.array_equal(identical["flip_js_divergence"], np.zeros(len(raw)))


def test_scalar_fusion_endpoints_and_sweep_contract() -> None:
    labels = np.array([0, 1, 2])
    first = np.eye(3, dtype=np.float32) * 3
    second = first[:, ::-1].copy()
    assert np.array_equal(core.scalar_fusion(first, second, 1), first)
    assert np.array_equal(core.scalar_fusion(first, second, 0), second)
    heads = {name: {"raw_logits": first, "tta_logits": second} for name in ("fused", "pixel", "motif")}
    rows = core.scalar_sweep(labels, heads)
    assert len(rows) == 3 * 2 * 101
    assert {row["alpha_first"] for row in rows} == {step / 100 for step in range(101)}


def test_classwise_crossfit_is_deterministic_and_exhaustive() -> None:
    generator = np.random.default_rng(42)
    labels = np.tile(np.arange(7), 10)
    first = generator.normal(size=(70, 7)).astype(np.float32)
    second = generator.normal(size=(70, 7)).astype(np.float32)
    one = core.crossfit_classwise_fusion(labels, first, second)
    two = core.crossfit_classwise_fusion(labels, first, second)
    assert one["oof_fold_id"] == two["oof_fold_id"]
    assert np.array_equal(one["oof_logits"], two["oof_logits"])
    test_indices = [index for fold in one["folds"] for index in fold["test_indices"]]
    assert sorted(test_indices) == list(range(70))
    assert all(len(fold["alpha_first_by_class"]) == 7 for fold in one["folds"])
    alpha = np.arange(7, dtype=np.float32) / 6
    expected = alpha * first + (1 - alpha) * second
    assert np.array_equal(core.scalar_fusion(first, second, 0.5), (0.5 * first + 0.5 * second).astype(np.float32))
    assert expected.shape == first.shape
    assert set(one["fold_metric_summary"]) == {"accuracy", "macro_f1", "loss"}


def test_weight_distance_and_interpolation_preserve_discrete_state() -> None:
    full = {
        "pixel_gnn.weight": torch.tensor([1.0, 2.0]),
        "classifier.weight": torch.tensor([3.0]),
        "counter": torch.tensor(4, dtype=torch.int64),
    }
    npf = {
        "pixel_gnn.weight": torch.tensor([3.0, 4.0]),
        "classifier.weight": torch.tensor([5.0]),
        "counter": torch.tensor(9, dtype=torch.int64),
    }
    report = core.weight_distance(full, npf)
    assert report["keys_exact_match"]
    assert set(report["subsystems"]) == {"pixel_gnn", "final_classifier", "other"}
    midpoint = core.interpolate_state(full, npf, 0.5)
    assert torch.equal(midpoint["pixel_gnn.weight"], torch.tensor([2.0, 3.0]))
    assert midpoint["counter"].item() == 4
    assert core.interpolate_state(full, npf, 0.25)["counter"].item() == 9
    assert torch.equal(core.interpolate_state(full, npf, 1.0)["pixel_gnn.weight"], full["pixel_gnn.weight"])
    assert torch.equal(core.interpolate_state(full, npf, 0.0)["pixel_gnn.weight"], npf["pixel_gnn.weight"])
    with pytest.raises(ValueError):
        core.interpolate_state(full, npf, 1.1)


def test_state_hash_detects_change_and_is_order_independent() -> None:
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    reordered = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    changed = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    assert core.state_sha256(first) == core.state_sha256(reordered)
    assert core.state_sha256(first) != core.state_sha256(changed)
