"""Focused unit regressions for the D19-A1-ID paired treatment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import D18Batch
from d18.models.structure_gnn import StructureGNN
from d18.training.train_d18 import canonical_state_manifest, scientific_resume_signature, set_seed

def config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs/d19" / name).read_text(encoding="utf-8"))


def batch() -> D18Batch:
    generator = torch.Generator().manual_seed(9)
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2, 1, 3], [1, 2, 3, 0, 2, 0, 3, 1]], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return D18Batch(
        x_cat=torch.randn((4, 10), generator=generator), edge_index_cat=edge_index,
        edge_attr_cat=torch.randn((8, 6), generator=generator), batch_index=torch.zeros(4, dtype=torch.long),
        ptr=torch.tensor([0, 4]), y=torch.tensor([2]), sample_index=torch.tensor([0]),
        pos_cat=torch.randn((4, 2), generator=generator), detected=torch.tensor([False]),
        landmark_missing_flag=torch.tensor([1]), image_48=torch.randn((1, 48, 48), generator=generator),
        edge_type_cat=edge_type, structure_relation_id_cat=torch.full((8,), -1, dtype=torch.long),
        local_edge_count=torch.tensor([4]), knn_edge_count=torch.tensor([4]), structure_edge_count=torch.tensor([0]),
        total_edge_count=torch.tensor([8]), structure_edge_count_before_purification=torch.tensor([0]),
        structure_edge_count_after_purification=torch.tensor([0]),
        purification_compatibility_kept_mean=torch.tensor([float("nan")]),
        purification_compatibility_dropped_mean=torch.tensor([float("nan")]),
        node_feature_names=[f"x{i}" for i in range(10)], edge_feature_names=[f"e{i}" for i in range(6)],
    )


def models() -> tuple[StructureGNN, StructureGNN]:
    set_seed(42)
    null = StructureGNN.from_config(config("d19_a1_id_null_evidence_only_seed42.yaml"), 10, 6)
    set_seed(42)
    correct = StructureGNN.from_config(config("d19_a1_id_correct_evidence_only_seed42.yaml"), 10, 6)
    return null, correct


def test_parameter_initialization_and_schema_parity() -> None:
    null, correct = models()
    assert sum(p.numel() for p in null.parameters()) == 266_616
    assert sum(p.numel() for p in correct.parameters()) == 266_616
    assert list(null.state_dict()) == list(correct.state_dict())
    assert canonical_state_manifest(null)["canonical_state_sha256"] == canonical_state_manifest(correct)["canonical_state_sha256"]


def test_null_is_exactly_invariant_to_valid_id_swap() -> None:
    null, _ = models()
    null.eval()
    value = batch()
    swapped = replace(value, edge_type_cat=1 - value.edge_type_cat)
    with torch.no_grad():
        true = null(value)["logits"]
        other = null(swapped)["logits"]
    assert torch.equal(true, other)


def test_correct_conditioning_changes_edge_input_and_activation() -> None:
    _, correct = models()
    correct.eval()
    value = batch()
    swapped_type = 1 - value.edge_type_cat
    true_attr, _ = correct.conditioned_edge_attributes(value.edge_attr_cat, value.edge_type_cat)
    swapped_attr, _ = correct.conditioned_edge_attributes(value.edge_attr_cat, swapped_type)
    with torch.no_grad():
        true = correct(value)["node_embeddings"]
        other = correct(replace(value, edge_type_cat=swapped_type))["node_embeddings"]
    assert not torch.equal(true_attr, swapped_attr)
    assert float((true - other).abs().max()) > 1e-8


def test_gradient_row_routing() -> None:
    null, correct = models()
    value = batch()
    for model in (null, correct):
        torch.nn.functional.cross_entropy(model(value)["logits"], value.y).backward()
    assert float(null.edge_type_embedding.weight.grad[0].norm()) > 0.0
    assert float(null.edge_type_embedding.weight.grad[1].norm()) == 0.0
    assert float(correct.edge_type_embedding.weight.grad[0].norm()) > 0.0
    assert float(correct.edge_type_embedding.weight.grad[1].norm()) > 0.0


def test_legacy_default_and_resume_signatures() -> None:
    baseline = config("d19_a0_evidence_only_matched_seed42.yaml")
    null_cfg = config("d19_a1_id_null_evidence_only_seed42.yaml")
    correct_cfg = config("d19_a1_id_correct_evidence_only_seed42.yaml")
    legacy = StructureGNN.from_config(baseline, 10, 6)
    assert sum(p.numel() for p in legacy.parameters()) == 265_832
    assert legacy.edge_type_embedding is None
    signatures = {scientific_resume_signature(value) for value in (baseline, null_cfg, correct_cfg)}
    assert len(signatures) == 3


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
