import inspect
from pathlib import Path

import numpy as np
import pytest

from pixel_relational_motif_e0.e02_runner import E01_V533_DICTIONARY_SHA256, TRAIN_SHA256, sha256_file
from pixel_relational_motif_e0.e0r_frozen import (
    CONTROL_ARTIFACT_SHA256,
    MATCHED_POOL_SHA256,
    load_frozen_control,
    load_frozen_conditions,
)


def _control_file(path: Path, seed: int):
    np.savez_compressed(
        path,
        control_seed=np.asarray(seed),
        selected_k=np.asarray(128),
        train_sha256=np.asarray(TRAIN_SHA256),
        source_dictionary_sha256=np.asarray(E01_V533_DICTIONARY_SHA256),
        pool_identity_sha256=np.asarray(MATCHED_POOL_SHA256),
        variance_floor=np.ones(12),
        gmm_weights=np.full(128, 1 / 128),
        gmm_means=np.zeros((128, 12)),
        gmm_variances=np.ones((128, 12)),
        pca_mean=np.zeros(24),
        pca_components=np.zeros((11, 24)),
        log_sigma_mean=np.asarray(0.0),
        log_sigma_std=np.asarray(1.0),
    )


def test_frozen_control_loader_assigns_state_without_refit(tmp_path, monkeypatch):
    path = tmp_path / "control.npz"
    _control_file(path, 42)
    monkeypatch.setitem(CONTROL_ARTIFACT_SHA256, 42, sha256_file(path))
    condition = load_frozen_control(path, expected_seed=42)
    assert condition.control_seed == 42
    assert condition.model.n_components == 128
    assert condition.model.weights_.shape == (128,)
    assert condition.transform.pca_components.shape == (11, 24)


def test_frozen_control_wrong_seed_mapping_fails(tmp_path, monkeypatch):
    path = tmp_path / "control.npz"
    _control_file(path, 43)
    monkeypatch.setitem(CONTROL_ARTIFACT_SHA256, 42, sha256_file(path))
    with pytest.raises(ValueError, match="seed-to-artifact"):
        load_frozen_control(path, expected_seed=42)


def test_all_six_mapping_requires_exact_five_controls():
    with pytest.raises(ValueError, match="exact control seed mapping"):
        load_frozen_conditions("actual.npz", {42: "one.npz"})


def test_r2_gate_and_complete_source_exist_before_execution():
    import pixel_relational_motif_e0.e0r_runner as runner

    source = inspect.getsource(runner)
    assert '!= "E0.R1 SUPPORTED"' in source
    assert '"NOT_RUN_R1_GATE"' in source
    assert "build_geometry_csr" in source
    assert "GEOMETRY_SEEDS" in source
    assert "e0r_r2_results.npz" in source


def test_public_labels_are_downstream_of_all_six_occurrence_arms():
    from pixel_relational_motif_e0.e0r_runner import run_e0r

    source = inspect.getsource(run_e0r)
    assert source.index("for condition in conditions:") < source.index("load_labels_downstream(")
    assert source.index("run_occurrence_condition(") < source.index("load_labels_downstream(")


def test_r2_reuses_actual_r1_occurrences_and_train_only_distance_source():
    from pixel_relational_motif_e0.e0r_runner import run_e0r

    source = inspect.getsource(run_e0r)
    assert "train_pair_distance_median(actual_result.train_occurrences)" in source
    assert "actual_result.public_occurrences" in source
    assert "tau_actual=np.asarray(actual_result.tau" in source


def test_e0r_runner_keeps_strict_json_and_never_names_components_validated_motifs():
    from pixel_relational_motif_e0.e0r_runner import run_e0r

    source = inspect.getsource(run_e0r)
    assert "allow_nan=False" in source
    assert "validated motif" not in source.lower()
