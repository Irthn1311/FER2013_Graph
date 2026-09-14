import inspect

import numpy as np
import pytest
from scipy import sparse

from pixel_relational_motif_e0.e02_runner import sha256_array
from pixel_relational_motif_e0.e0r_continuation import (
    R1_OCCURRENCES_SHA256,
    R1_RESULTS_SHA256,
    SCIENTIFIC_SHA,
    SHARD_PLAN,
    FrozenR1Substrate,
    _atomic_savez,
    _checkpoint_payload,
    build_nested_arm,
    fit_nested_arm,
    validate_checkpoint,
)
from pixel_relational_motif_e0.e0r_geometry import (
    build_geometry_csr,
    nested_csr,
    sparse_sha256,
    train_pair_distance_median,
)
from pixel_relational_motif_e0.e0r_occurrence import CompactOccurrences, occurrence_histograms
from pixel_relational_motif_e0.e0r_runner import _fit_predict_probe, _metric_record


def _occ(label: int, index: int) -> CompactOccurrences:
    components = np.asarray([label, (label + 1) % 7, (label + 2) % 7], dtype=np.int16)
    x = np.asarray([2 + label, 12 + index % 5, 31 - label], dtype=np.int16)
    y = np.asarray([3 + index % 7, 21 - label, 40 - index % 9], dtype=np.int16)
    return CompactOccurrences(components, np.asarray([0.99, 0.98, 0.97]), y, x)


def _substrate() -> FrozenR1Substrate:
    train_labels = np.tile(np.arange(7, dtype=np.int8), 6)
    public_labels = np.tile(np.arange(7, dtype=np.int8), 2)
    train_occurrences = [_occ(int(label), i) for i, label in enumerate(train_labels)]
    public_occurrences = [_occ(int(label), i + 100) for i, label in enumerate(public_labels)]
    train_o = occurrence_histograms(train_occurrences)
    public_o = occurrence_histograms(public_occurrences)
    distance = train_pair_distance_median(train_occurrences)
    assert distance is not None
    return FrozenR1Substrate(
        train_occurrences=train_occurrences,
        public_occurrences=public_occurrences,
        train_ids=np.arange(1000, 1000 + len(train_occurrences), dtype=np.int32),
        public_ids=np.arange(2000, 2000 + len(public_occurrences), dtype=np.int32),
        train_labels=train_labels,
        public_labels=public_labels,
        train_o=train_o,
        public_o=public_o,
        distance_median=distance,
        o_train_sha256=sha256_array(train_o),
        o_public_sha256=sha256_array(public_o),
    )


def test_execution_plan_and_original_scientific_identity_are_frozen():
    assert SCIENTIFIC_SHA == "671e3c2f69607778f08a923026e76744561e13b2"
    assert R1_OCCURRENCES_SHA256 == "30f1a5b642af2ecdfc29ae73960fb01f90c391964db845bd4b33cf3c017f7fa9"
    assert R1_RESULTS_SHA256 == "b6675ce694f5a607cfea07abb3ed1065753f42ee848f7595d1cb8e682361a0ed"
    assert SHARD_PLAN == {
        "A": {"actual": True, "seeds": (42, 43, 44, 45)},
        "B": {"actual": False, "seeds": (46, 47, 48, 49, 50)},
        "C": {"actual": False, "seeds": (51, 52, 53, 54, 55)},
        "D": {"actual": False, "seeds": (56, 57, 58, 59, 60, 61)},
    }


@pytest.mark.parametrize("seed", [None, 42, 47])
def test_shard_nested_csr_and_probe_exactly_equal_original_functions(seed):
    substrate = _substrate()
    shard_train, shard_public = build_nested_arm(substrate, seed=seed)

    direct_train_g = build_geometry_csr(
        substrate.train_occurrences,
        substrate.train_ids,
        distance_median=train_pair_distance_median(substrate.train_occurrences),
        shuffle_seed=seed,
    )
    direct_public_g = build_geometry_csr(
        substrate.public_occurrences,
        substrate.public_ids,
        distance_median=train_pair_distance_median(substrate.train_occurrences),
        shuffle_seed=seed,
    )
    direct_train = nested_csr(occurrence_histograms(substrate.train_occurrences), direct_train_g)
    direct_public = nested_csr(occurrence_histograms(substrate.public_occurrences), direct_public_g)

    assert np.array_equal(substrate.train_o, occurrence_histograms(substrate.train_occurrences))
    assert substrate.distance_median == train_pair_distance_median(substrate.train_occurrences)
    assert shard_train.shape == direct_train.shape
    assert shard_public.shape == direct_public.shape
    assert (shard_train != direct_train).nnz == 0
    assert (shard_public != direct_public).nnz == 0
    assert sparse_sha256(shard_train) == sparse_sha256(direct_train)
    assert sparse_sha256(shard_public) == sparse_sha256(direct_public)

    condition = "actual_O_G" if seed is None else f"geometry_shuffle_{seed}"
    shard_prediction, shard_diagnostic, shard_metric = fit_nested_arm(
        shard_train,
        substrate.train_labels,
        shard_public,
        substrate.public_labels,
        condition=condition,
    )
    direct_prediction, direct_diagnostic = _fit_predict_probe(
        direct_train,
        substrate.train_labels,
        direct_public,
        condition=condition,
    )
    assert np.array_equal(shard_prediction, direct_prediction)
    assert shard_diagnostic == direct_diagnostic
    assert shard_metric == _metric_record(substrate.public_labels, direct_prediction)


def test_execution_wrapper_imports_scientific_math_instead_of_reimplementing_it():
    from pixel_relational_motif_e0 import e0r_continuation

    source = inspect.getsource(e0r_continuation)
    assert "LogisticRegression" not in source
    assert "StandardScaler" not in source
    assert ".fit(" not in source
    assert "_fit_predict_probe(" in source
    assert "build_geometry_csr(" in source
    assert "nested_csr(" in source
    assert "train_pair_distance_median(" in source


def test_per_fit_checkpoint_roundtrip_is_strict_and_immediate(tmp_path):
    train_ids = np.arange(28709, dtype=np.int32)
    public_ids = np.arange(28709, 32298, dtype=np.int32)
    train_o = np.zeros((28709, 128), dtype=np.float32)
    public_o = np.zeros((3589, 128), dtype=np.float32)
    substrate = FrozenR1Substrate(
        train_occurrences=[],
        public_occurrences=[],
        train_ids=train_ids,
        public_ids=public_ids,
        train_labels=np.zeros(28709, dtype=np.int8),
        public_labels=np.zeros(3589, dtype=np.int8),
        train_o=train_o,
        public_o=public_o,
        distance_median=0.5,
        o_train_sha256=sha256_array(train_o),
        o_public_sha256=sha256_array(public_o),
    )
    train_nested = sparse.csr_matrix((28709, 131200), dtype=np.float32)
    public_nested = sparse.csr_matrix((3589, 131200), dtype=np.float32)
    prediction = np.zeros(3589, dtype=np.int8)
    payload = _checkpoint_payload(
        substrate,
        seed=47,
        execution_wrapper_sha="1" * 40,
        train_nested=train_nested,
        public_nested=public_nested,
        prediction=prediction,
        diagnostic={"n_iter": [2000], "converged": False, "warnings": ["registered warning"]},
        metric={"accuracy": 1.0, "macro_f1": 1.0},
    )
    path = tmp_path / "e0r_r2_seed_47.npz"
    _atomic_savez(path, payload)
    assert path.is_file() and not path.with_suffix(".npz.tmp").exists()
    record = validate_checkpoint(
        path,
        seed=47,
        execution_wrapper_sha="1" * 40,
        substrate=substrate,
    )
    assert record["seed"] == 47
    assert record["n_iter"] == [2000]
    assert record["converged"] is False
    assert record["warnings"] == ["registered warning"]
