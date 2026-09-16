import inspect

import numpy as np
from scipy import sparse

from pixel_relational_motif_e0 import crs_train_runner as r


def _toy_multiclass():
    rng = np.random.default_rng(123)
    y = np.repeat(np.arange(7, dtype=np.int64), 30)
    x = rng.normal(scale=0.05, size=(len(y), 14)).astype(np.float64)
    for cls in range(7):
        rows = y == cls
        x[rows, cls] += 2.0
        x[rows, cls + 7] += 1.0
    return x, y


def test_train_runner_registered_constants():
    assert r.SOURCE_BASE_SHA == "49e65e1032f1af13fd615a2e6f7a64f20760684b"
    assert r.SKM_N_INIT == 3
    assert r.SKM_MAX_ITER == 50
    assert r.SKM_TOL == 1e-6
    assert r.SPARSE_POOL_CHUNK_IMAGES == 256
    assert r.LOGREG_C == 1.0
    assert r.LOGREG_SOLVER == "lbfgs"
    assert r.LOGREG_CLASS_WEIGHT == "balanced"
    assert r.LOGREG_MAX_ITER == 5000
    assert r.LOGREG_TOL == 1e-4


def test_fixed_probe_is_seven_way_converged_and_serializable_by_parameters():
    x, y = _toy_multiclass()
    state = r._fit_fixed_probe(x, y, arm="toy")
    assert state.converged
    assert np.array_equal(state.classes, np.arange(7))
    assert state.coef.shape == (7, x.shape[1])
    assert state.intercept.shape == (7,)
    pred = r.probe_predict(state, x)
    assert pred.shape == y.shape
    assert np.mean(pred == y) > 0.95


def test_cluster_diagnostics_cover_all_registered_clusters():
    labels = np.tile(np.arange(512, dtype=np.int32), 3)
    diag = r._cluster_diagnostics(labels)
    assert diag["n_assignments"] == 1536
    assert diag["empty_clusters"] == 0
    assert diag["min_occupancy"] == 3
    assert diag["max_occupancy"] == 3
    assert np.isclose(diag["normalized_entropy"], 1.0)


def test_sparse_pool_chunk_roundtrip_is_exact(tmp_path):
    rng = np.random.default_rng(8)
    x = np.zeros((11, 1152), dtype=np.float32)
    for row in range(len(x)):
        ids = rng.choice(1152, size=30, replace=False)
        x[row, ids] = rng.random(30, dtype=np.float32)
    block = sparse.csr_matrix(x)
    path = tmp_path / "chunk.npz"
    r._save_sparse_pool_chunk([block], path=path)
    restored = sparse.load_npz(path)
    assert sparse.isspmatrix_csr(restored)
    assert restored.shape == x.shape
    assert np.allclose(restored.toarray(), x)


def test_train_runner_cli_has_no_public_or_private_input():
    parser = r._build_parser()
    option_strings = {
        opt
        for action in parser._actions
        for opt in action.option_strings
    }
    assert "--train-csv" in option_strings
    assert "--dictionary-npz" in option_strings
    assert "--output-dir" in option_strings
    assert not any("public" in opt.lower() for opt in option_strings)
    assert not any("private" in opt.lower() for opt in option_strings)
    assert "--test-csv" not in option_strings


def test_labels_are_read_only_after_dense_representations_are_built():
    source = inspect.getsource(r.run_train_stage)
    dense_call = source.index("_build_dense_features(")
    labels_call = source.index("load_labels_downstream(")
    assert dense_call < labels_call
    assert 'role="train"' in source
    assert 'role="public"' not in source
