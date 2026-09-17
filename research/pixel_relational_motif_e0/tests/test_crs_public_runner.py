import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

from pixel_relational_motif_e0 import crs_public_runner as r
from pixel_relational_motif_e0 import crs_stage as c
from pixel_relational_motif_e0.e02_runner import PixelData, bind_e02_split, sha256_file


def _unit_centers() -> np.ndarray:
    centers = np.zeros((512, 1152), dtype=np.float32)
    centers[np.arange(512), np.arange(512)] = 1.0
    return centers


def _model_payload() -> dict[str, np.ndarray]:
    payload = {
        "technical_amendment": np.asarray("A1"),
        "master_seed": np.asarray(42, dtype=np.int32),
        "dictionary_samples_per_image": np.asarray(24, dtype=np.int32),
        "crs_k": np.asarray(512, dtype=np.int32),
        "crs_descriptor_dim": np.asarray(1152, dtype=np.int32),
        "skm_n_init": np.asarray(3, dtype=np.int32),
        "skm_max_iter": np.asarray(500, dtype=np.int32),
        "skm_tol": np.asarray(1e-6, dtype=np.float64),
        "source_dictionary_sha256": np.asarray(r.E01_V533_DICTIONARY_SHA256),
        "train_sha256": np.asarray(r.TRAIN_SHA256),
        "m_crs_centers": _unit_centers(),
        "c_crs_centers": _unit_centers(),
        "logreg_c": np.asarray(1.0, dtype=np.float64),
        "logreg_solver": np.asarray("lbfgs"),
        "logreg_class_weight": np.asarray("balanced"),
        "logreg_max_iter": np.asarray(5000, dtype=np.int32),
        "logreg_tol": np.asarray(1e-4, dtype=np.float64),
    }
    for arm, dimension in (("p", 640), ("m", 2560), ("c", 2560)):
        payload[f"{arm}_classes"] = np.arange(7, dtype=np.int64)
        payload[f"{arm}_coef"] = np.zeros((7, dimension), dtype=np.float64)
        payload[f"{arm}_intercept"] = np.arange(7, dtype=np.float64)
        payload[f"{arm}_n_iter"] = np.asarray([1], dtype=np.int64)
    return payload


def _write_model(path: Path, payload: dict[str, np.ndarray]) -> str:
    np.savez_compressed(path, **payload)
    return sha256_file(path)


def _load_mutated_model(tmp_path, monkeypatch, mutate):
    payload = _model_payload()
    mutate(payload)
    path = tmp_path / "model.npz"
    digest = _write_model(path, payload)
    monkeypatch.setattr(r, "TRAIN_MODEL_SHA256", digest)
    return r.load_frozen_train_model(path)


def test_public_runner_registered_constants():
    assert r.TRAIN_MODEL_SHA256 == "77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a"
    assert r.E01_V533_DICTIONARY_SHA256 == "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"
    assert r.PUBLIC_ROWS == 3589
    assert r.PUBLIC_SHA256 == "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08"
    assert r.BOOTSTRAP_REPLICATES == 2000
    assert r.BOOTSTRAP_SEED == 42
    assert r.BOOTSTRAP_GENERATOR == "numpy.random.Generator(PCG64)"
    assert r.BOOTSTRAP_CI_METHOD == "linear"


def test_wrong_train_model_sha_is_rejected(tmp_path):
    path = tmp_path / "model.npz"
    _write_model(path, _model_payload())
    with pytest.raises(ValueError, match="Train model SHA256"):
        r.load_frozen_train_model(path)


def test_wrong_dictionary_sha_is_rejected(tmp_path):
    path = tmp_path / "e01_dictionary.npz"
    path.write_bytes(b"not the frozen dictionary")
    with pytest.raises(ValueError, match="v533 dictionary SHA256"):
        r.verify_frozen_dictionary(path)


def test_valid_train_model_schema_loads(tmp_path, monkeypatch):
    state = _load_mutated_model(tmp_path, monkeypatch, lambda payload: None)
    assert state.m_centers.shape == (512, 1152)
    assert state.c_centers.shape == (512, 1152)
    assert set(state.probes) == {"P", "M", "C"}


def test_missing_train_model_field_is_rejected(tmp_path, monkeypatch):
    def mutate(payload):
        del payload["m_crs_centers"]

    with pytest.raises(ValueError, match="missing required fields"):
        _load_mutated_model(tmp_path, monkeypatch, mutate)


def test_wrong_train_model_shape_is_rejected(tmp_path, monkeypatch):
    def mutate(payload):
        payload["c_crs_centers"] = payload["c_crs_centers"][:-1]

    with pytest.raises(ValueError, match="C center shape"):
        _load_mutated_model(tmp_path, monkeypatch, mutate)


def test_nonfinite_train_model_state_is_rejected(tmp_path, monkeypatch):
    def mutate(payload):
        payload["m_coef"][0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        _load_mutated_model(tmp_path, monkeypatch, mutate)


def test_nonunit_train_centers_are_rejected(tmp_path, monkeypatch):
    def mutate(payload):
        payload["m_crs_centers"][0] *= 2.0

    with pytest.raises(ValueError, match="not unit normalized"):
        _load_mutated_model(tmp_path, monkeypatch, mutate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("technical_amendment", np.asarray("A0")),
        ("master_seed", np.asarray(43, dtype=np.int32)),
        ("dictionary_samples_per_image", np.asarray(23, dtype=np.int32)),
        ("crs_k", np.asarray(511, dtype=np.int32)),
        ("crs_descriptor_dim", np.asarray(1151, dtype=np.int32)),
        ("skm_n_init", np.asarray(2, dtype=np.int32)),
        ("skm_max_iter", np.asarray(499, dtype=np.int32)),
        ("skm_tol", np.asarray(1e-5, dtype=np.float64)),
    ],
)
def test_wrong_a1_or_registered_config_is_rejected(
    tmp_path, monkeypatch, field, value
):
    def mutate(payload):
        payload[field] = value

    with pytest.raises(ValueError, match=field):
        _load_mutated_model(tmp_path, monkeypatch, mutate)


def test_public_role_aliases_and_canonical_identity_contract():
    for name in ("test.csv", "PrivateTest.csv", "private_test.csv", "final_test.csv"):
        with pytest.raises(ValueError, match="must not open"):
            bind_e02_split("public", name)

    images = np.empty((3589, 48, 48), dtype=np.uint8)
    ids = np.arange(28709, 32298, dtype=np.int32)
    data = PixelData(images, ids, r.PUBLIC_SHA256, "public")
    r.validate_public_identity(data)
    assert ids[0] == 28709 and ids[-1] == 32297

    with pytest.raises(ValueError, match="canonical-ID"):
        r.validate_public_identity(
            PixelData(images, np.arange(3589, dtype=np.int32), r.PUBLIC_SHA256, "public")
        )


def test_public_control_key_is_deterministic_split_specific_and_preserves_blocks():
    primitive = (
        np.arange(c.PRIMITIVE_SIDE * c.PRIMITIVE_SIDE, dtype=np.int32)
        .reshape(c.PRIMITIVE_SIDE, c.PRIMITIVE_SIDE)
        % c.PRIMITIVE_K
    )
    descriptors = c.composition_descriptors(primitive, normalize=True)
    indices = np.arange(c.CRS_VALID_POSITIONS, dtype=np.int32)
    public_a = c.permute_cell_blocks(
        descriptors,
        split_id="public",
        image_id=28709,
        center_indices=indices,
        master_seed=42,
    )
    public_b = c.permute_cell_blocks(
        descriptors,
        split_id="public",
        image_id=28709,
        center_indices=indices,
        master_seed=42,
    )
    train = c.permute_cell_blocks(
        descriptors,
        split_id="train",
        image_id=28709,
        center_indices=indices,
        master_seed=42,
    )
    assert np.array_equal(public_a, public_b)
    assert not np.array_equal(public_a, train)
    assert public_a.shape == (1296, 1152)
    original_blocks = descriptors.reshape(1296, 9, 128)
    public_blocks = public_a.reshape(1296, 9, 128)
    for row in range(1296):
        assert sorted(x.tobytes() for x in original_blocks[row]) == sorted(
            x.tobytes() for x in public_blocks[row]
        )


def test_exact_cosine_helper_matches_existing_predict_dense_and_csr():
    rng = np.random.default_rng(7)
    x = rng.random((31, 17), dtype=np.float32)
    centers = rng.random((9, 17), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    reference = c.SphericalKMeansResult(
        cluster_centers_=centers,
        labels_=np.zeros(31, dtype=np.int32),
        objective_=0.0,
        n_iter_=1,
        converged_=True,
        init_index_=0,
        empty_reseeds_=0,
        init_diagnostics_=(),
    )
    expected_dense = reference.predict(x, batch_size=11)
    expected_sparse = reference.predict(sparse.csr_matrix(x), batch_size=11)
    assert np.array_equal(
        r.exact_cosine_nearest_centers(x, centers, batch_size=11), expected_dense
    )
    assert np.array_equal(
        r.exact_cosine_nearest_centers(
            sparse.csr_matrix(x), centers, batch_size=11
        ),
        expected_sparse,
    )


def test_public_representation_dimensions_use_all_1296_assignments():
    primitive = (
        np.arange(c.PRIMITIVE_SIDE * c.PRIMITIVE_SIDE, dtype=np.int32)
        .reshape(c.PRIMITIVE_SIDE, c.PRIMITIVE_SIDE)
        % c.PRIMITIVE_K
    )
    descriptors = c.composition_descriptors(primitive, normalize=True)
    assert descriptors.shape == (1296, 1152)
    p = c.p_image_feature(primitive)
    m = c.crs_image_feature(np.arange(1296, dtype=np.int32) % 512)
    control = c.permute_cell_blocks(
        descriptors,
        split_id="public",
        image_id=28709,
        center_indices=np.arange(1296, dtype=np.int32),
    )
    control_ids = np.argmax(control[:, :512], axis=1).astype(np.int32)
    z = c.crs_image_feature(control_ids)
    assert p.shape == (640,)
    assert m.shape == z.shape == (2560,)
    assert all(np.isfinite(x).all() for x in (p, m, z))
    assert all(np.isclose(np.linalg.norm(x), 1.0) for x in (p, m, z))


def test_stored_probe_inference_is_exact_logits_argmax():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(13, 8)).astype(np.float32)
    coef = rng.normal(size=(7, 8))
    intercept = rng.normal(size=7)
    probe = r.FrozenProbe(np.arange(7), coef, intercept, np.asarray([3]))
    expected = np.argmax(features @ coef.T + intercept[None, :], axis=1)
    assert np.array_equal(r.frozen_probe_predict(probe, features), expected)


def test_public_execution_path_contains_no_fit_and_loads_labels_after_predictions():
    module_source = inspect.getsource(r)
    run_source = inspect.getsource(r.run_public_stage)
    assert ".fit(" not in module_source
    assert run_source.index("build_public_representations(") < run_source.index(
        "freeze_predictions_before_labels("
    )
    assert run_source.index("freeze_predictions_before_labels(") < run_source.index(
        "load_labels_downstream("
    )


def test_bootstrap_matrix_is_exact_deterministic_pcg64_contract():
    observed = r.make_bootstrap_indices()
    expected = np.random.Generator(np.random.PCG64(42)).integers(
        0, 3589, size=(2000, 3589), dtype=np.int32
    )
    assert observed.shape == (2000, 3589)
    assert observed.dtype == np.int32
    assert np.array_equal(observed, expected)


def test_confusion_metrics_match_registered_sklearn_metrics():
    rng = np.random.default_rng(12)
    y = rng.integers(0, 7, size=3589, dtype=np.int64)
    pred = rng.integers(0, 7, size=3589, dtype=np.int64)
    accuracy, macro_f1 = r._metrics_from_confusion(y, pred)
    assert accuracy == accuracy_score(y, pred)
    assert macro_f1 == f1_score(
        y, pred, labels=np.arange(7), average="macro", zero_division=0
    )


def test_all_four_deltas_use_one_shared_matrix_and_registered_ci(monkeypatch):
    monkeypatch.setattr(r, "PUBLIC_ROWS", 12)
    monkeypatch.setattr(r, "BOOTSTRAP_REPLICATES", 5)
    y = np.asarray([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4], dtype=np.int64)
    predictions = {
        "P": np.asarray([0, 1, 0, 3, 0, 5, 0, 0, 1, 0, 3, 0]),
        "M": y.copy(),
        "C": np.asarray([0, 0, 2, 0, 4, 0, 6, 0, 0, 2, 0, 4]),
    }
    shared = np.asarray(
        [
            np.arange(12),
            np.zeros(12),
            np.ones(12),
            np.arange(11, -1, -1),
            np.asarray([0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2]),
        ],
        dtype=np.int32,
    )
    deltas = r.paired_bootstrap_deltas(y, predictions, shared)
    assert set(deltas) == {
        "delta_acc_m_c",
        "delta_f1_m_c",
        "delta_acc_m_p",
        "delta_f1_m_p",
    }
    assert all(value.shape == (5,) for value in deltas.values())
    for value in deltas.values():
        assert r.percentile_interval(value) == tuple(
            np.quantile(value, [0.025, 0.975], method="linear")
        )


@pytest.mark.parametrize(
    ("gate_a", "gate_b", "expected"),
    [
        (False, False, "COMPOSITIONAL FORMULATION NOT SUPPORTED"),
        (
            False,
            True,
            "UTILITY WITHOUT ARRANGEMENT EVIDENCE — NOT COMPOSITIONAL SUPPORT",
        ),
        (
            True,
            False,
            "ARRANGEMENT SUPPORTED — ABSTRACTION UTILITY NOT SUPPORTED",
        ),
        (True, True, "COMPOSITIONAL STAGE SUPPORTED"),
    ],
)
def test_exact_registered_decision_table(gate_a, gate_b, expected):
    assert r.registered_verdict(gate_a, gate_b) == expected


def test_cli_is_public_only_and_has_no_scientific_overrides():
    parser = r._build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert options.issuperset(
        {"--public-csv", "--dictionary-npz", "--train-model-npz", "--output-dir"}
    )
    forbidden = ("train-csv", "private", "test-csv", "seed", "bootstrap", "tol", "max-iter", "k")
    assert not any(any(token in option.lower() for token in forbidden) for option in options)


def test_registered_outputs_refuse_overwrite_before_reading_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv(r.PUBLIC_SOURCE_SHA_ENV, "a" * 40)
    (tmp_path / r.PREDICTIONS_FILENAME).write_bytes(b"existing registered look")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        r.run_public_stage(
            public_csv="missing-public.csv",
            dictionary_npz="missing-dictionary.npz",
            train_model_npz="missing-model.npz",
            output_dir=tmp_path,
        )


def test_prediction_artifact_roundtrip_validates_independently(tmp_path):
    source_sha = "b" * 40
    indices = r.make_bootstrap_indices()
    y = np.arange(3589, dtype=np.int64) % 7
    payload = {
        "experiment": np.asarray("PGM_CRS_PUBLIC_REGISTERED_VALIDATION"),
        "issue": np.asarray(84, dtype=np.int32),
        "draft_pr": np.asarray(85, dtype=np.int32),
        "source_sha": np.asarray(source_sha),
        "public_sha256": np.asarray(r.PUBLIC_SHA256),
        "train_model_sha256": np.asarray(r.TRAIN_MODEL_SHA256),
        "dictionary_sha256": np.asarray(r.E01_V533_DICTIONARY_SHA256),
        "canonical_public_ids": np.arange(28709, 32298, dtype=np.int32),
        "y_true": y.astype(np.int8),
        "pred_P": y.astype(np.int8),
        "pred_M": y.astype(np.int8),
        "pred_C": y.astype(np.int8),
        "bootstrap_indices": indices,
        "bootstrap_indices_sha256": np.asarray(r.sha256_array(indices)),
        "delta_acc_m_c": np.zeros(2000),
        "delta_f1_m_c": np.zeros(2000),
        "delta_acc_m_p": np.zeros(2000),
        "delta_f1_m_p": np.zeros(2000),
    }
    path = tmp_path / r.PREDICTIONS_FILENAME
    r._atomic_savez(path, payload)
    r._validate_prediction_artifact(path, source_sha=source_sha)
    with np.load(path, allow_pickle=False) as artifact:
        assert np.array_equal(artifact["bootstrap_indices"], indices)
        assert np.array_equal(artifact["pred_M"], y)


def test_private_access_flag_is_literal_false_in_public_summary_source():
    source = inspect.getsource(r.run_public_stage)
    assert '"private_test_accessed": False' in source

