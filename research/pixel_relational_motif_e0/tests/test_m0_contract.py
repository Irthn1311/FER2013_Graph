import inspect

import numpy as np
from scipy import sparse
import torch

from pixel_relational_motif_e0.e0r_geometry import GEOMETRY_DIM, build_geometry_csr, nested_csr, sparse_sha256
from pixel_relational_motif_e0.e0r_occurrence import CompactOccurrences, occurrence_histograms
from pixel_relational_motif_e0.m0_aggregate import (
    VERDICT_MIXED,
    VERDICT_NOT_SUPPORTED,
    VERDICT_SUPPORTED,
    registered_bootstrap,
    registered_verdict,
)
from pixel_relational_motif_e0.m0_config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DISTANCE_MEDIAN,
    EPOCHS,
    G_CONFIG,
    L_CONFIG,
    M_CONFIG,
    SEEDS,
    X_DIM,
)
from pixel_relational_motif_e0.m0_models import MinimalRelationGraphModel, SparseAggregateMLP, parameter_count
from pixel_relational_motif_e0.m0_substrate import (
    edge_type_matrix,
    graph_primitives,
    reconstruct_aggregate_from_graph,
)
from pixel_relational_motif_e0.m0_train import scipy_batch_to_torch


def _occ(components, xy):
    return CompactOccurrences(
        np.asarray(components, dtype=np.int16),
        np.ones(len(components), dtype=np.float64),
        np.asarray([item[1] for item in xy], dtype=np.int16),
        np.asarray([item[0] for item in xy], dtype=np.int16),
    )


def test_preregistered_constants_are_exact():
    assert DISTANCE_MEDIAN == 0.5057389074044559
    assert SEEDS == (42, 43, 44, 45, 46)
    assert EPOCHS == 50 and BOOTSTRAP_REPLICATES == 2000 and BOOTSTRAP_SEED == 42
    assert X_DIM == 131200
    assert L_CONFIG == {
        "family": "L", "scaler": "StandardScaler(with_mean=False)", "penalty": "l2",
        "C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-6,
    }
    assert M_CONFIG["batch_size"] == 256 and G_CONFIG["batch_size"] == 64
    assert M_CONFIG["epochs"] == G_CONFIG["epochs"] == 50
    assert parameter_count(SparseAggregateMLP()) == M_CONFIG["expected_parameter_count"] == 4199719
    assert parameter_count(MinimalRelationGraphModel()) == G_CONFIG["expected_parameter_count"] == 22759


def test_graph_primitives_exhaustively_reproduce_exact_o_g_x():
    occurrences = [
        _occ([], []),
        _occ([4], [(3, 7)]),
        _occ([1, 2, 1], [(2, 2), (42, 4), (11, 39)]),
        _occ([127, 0], [(0, 47), (47, 0)]),
    ]
    ids = np.arange(len(occurrences), dtype=np.int32)
    graph = graph_primitives(occurrences, DISTANCE_MEDIAN)
    reconstructed_o, reconstructed_g = reconstruct_aggregate_from_graph(graph)
    direct_o = occurrence_histograms(occurrences)
    direct_g = build_geometry_csr(occurrences, ids, distance_median=DISTANCE_MEDIAN)
    assert np.array_equal(reconstructed_o, direct_o)
    assert (reconstructed_g != direct_g).nnz == 0
    assert sparse_sha256(reconstructed_g) == sparse_sha256(direct_g)
    assert sparse_sha256(nested_csr(reconstructed_o, reconstructed_g)) == sparse_sha256(nested_csr(direct_o, direct_g))
    assert reconstructed_g.shape == (4, GEOMETRY_DIM)
    assert set(graph) == {"node_offsets", "components", "relation_offsets", "relations"}


def test_confidence_does_not_enter_frozen_graph_primitives():
    first = _occ([1, 2], [(2, 3), (40, 41)])
    second = CompactOccurrences(first.components, np.asarray([0.01, 0.99]), first.y, first.x)
    one = graph_primitives([first], DISTANCE_MEDIAN)
    two = graph_primitives([second], DISTANCE_MEDIAN)
    assert all(np.array_equal(one[key], two[key]) for key in one)


def test_zero_and_one_node_semantics_have_no_fake_edges():
    zero, one = _occ([], []), _occ([5], [(9, 9)])
    assert edge_type_matrix(zero, DISTANCE_MEDIAN).shape == (0, 0)
    assert np.array_equal(edge_type_matrix(one, DISTANCE_MEDIAN), np.asarray([[255]], dtype=np.uint8))
    graph = graph_primitives([zero, one], DISTANCE_MEDIAN)
    o, g = reconstruct_aggregate_from_graph(graph)
    assert np.count_nonzero(o[0]) == 0 and o[1, 5] == 1
    assert g[0].nnz == g[1].nnz == 0
    model = MinimalRelationGraphModel()
    component = torch.tensor([[0], [5]])
    relation = torch.tensor([[[255]], [[255]]], dtype=torch.uint8)
    mask = torch.tensor([[False], [True]])
    logits = model(component, relation, mask)
    assert torch.equal(logits[0], model.classifier.bias)
    assert logits.shape == (2, 7)


def test_sparse_first_layer_is_exactly_sparse_matrix_times_weight():
    matrix = sparse.csr_matrix(np.asarray([[0, 2, 0], [3, 0, 4]], dtype=np.float32))
    padded = sparse.hstack([matrix, sparse.csr_matrix((2, X_DIM - 3), dtype=np.float32)], format="csr")
    model = SparseAggregateMLP()
    x = scipy_batch_to_torch(padded, np.asarray([0, 1]), torch.device("cpu"))
    sparse_result = torch.sparse.mm(x, model.sparse_linear.weight.t()) + model.sparse_linear.bias
    dense_reference = torch.from_numpy(padded.toarray()) @ model.sparse_linear.weight.t() + model.sparse_linear.bias
    assert torch.equal(sparse_result, dense_reference)


def test_graph_model_has_exact_fixed_structure_and_no_forbidden_layers():
    model = MinimalRelationGraphModel()
    assert model.embedding.num_embeddings == 128 and model.embedding.embedding_dim == 32
    assert len(model.message_layers) == 2
    assert all(tuple(layer.relation_weight.shape) == (8, 32, 32) for layer in model.message_layers)
    forbidden = (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.MultiheadAttention)
    assert not any(isinstance(module, forbidden) for module in model.modules())


def test_registered_gate_is_strict_joint():
    positive = np.asarray([0.01, 0.02, 0.03])
    zero = np.asarray([0.0, 0.01, 0.02])
    negative = np.asarray([-0.03, -0.02, -0.01])
    keys = ("g_l_accuracy", "g_l_macro_f1", "g_m_accuracy", "g_m_macro_f1")
    assert registered_verdict({key: positive for key in keys}) == VERDICT_SUPPORTED
    assert registered_verdict({keys[0]: positive, keys[1]: zero, keys[2]: negative, keys[3]: negative}) == VERDICT_MIXED
    assert registered_verdict({key: negative for key in keys}) == VERDICT_NOT_SUPPORTED


def test_bootstrap_uses_mean_of_seed_metrics_not_ensemble(monkeypatch):
    import pixel_relational_motif_e0.m0_aggregate as module

    monkeypatch.setattr(module, "PUBLIC_ROWS", 14)
    monkeypatch.setattr(module, "BOOTSTRAP_REPLICATES", 11)
    labels = np.asarray([0, 1, 2, 3, 4, 5, 6] * 2, dtype=np.int8)
    l = np.zeros(14, dtype=np.int8)
    m = np.stack([np.roll(labels, shift) for shift in range(5)])
    g = np.stack([labels.copy() for _ in range(5)])
    first = registered_bootstrap(labels, l, m, g)
    second = registered_bootstrap(labels, l, m, g)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert all(value.shape == (11,) for value in first.values())


def test_public_labels_are_not_part_of_training_or_substrate_artifact_contract():
    import pixel_relational_motif_e0.m0_substrate as substrate_module
    import pixel_relational_motif_e0.m0_train as train_module

    train_source = inspect.getsource(train_module)
    assert "public_labels" not in train_source
    assert "public_metrics_present" in train_source
    assert "public_labels.npy" not in inspect.getsource(substrate_module)
