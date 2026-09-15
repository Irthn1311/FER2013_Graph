"""Canonical label-isolated substrate construction for preregistered PGM M0."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from .e02_runner import PUBLIC_ROWS, TRAIN_ROWS, sha256_array, sha256_file
from .e0r_continuation import _unflatten_occurrences
from .e0r_geometry import _quadrants, build_geometry_csr, nested_csr, sparse_sha256, train_pair_distance_median
from .e0r_occurrence import CompactOccurrences, occurrence_histograms
from .m0_config import (
    DISTANCE_MEDIAN,
    E0R_SCIENTIFIC_SHA,
    EXPERIMENT_ID,
    G_DIM,
    ISSUE_NUMBER,
    OCCURRENCES_SHA256,
    PUBLIC_IDS_SHA256,
    R1_RESULTS_SHA256,
    TRAIN_IDS_SHA256,
    X_DIM,
)


SUBSTRATE_FILES = (
    "train_x_scaled.npz",
    "public_x_scaled.npz",
    "train_graphs.npz",
    "public_graphs.npz",
    "train_ids.npy",
    "public_ids.npy",
    "train_labels.npy",
    "scaler.npz",
)


def _atomic_write(path: Path, writer) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_array(path: Path, value: np.ndarray) -> None:
    _atomic_write(path, lambda stream: np.save(stream, value, allow_pickle=False))


def _save_npz(path: Path, **values: np.ndarray) -> None:
    for key, value in values.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"nonfinite substrate field {key}")
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **values))


def _save_sparse(path: Path, value: sparse.csr_matrix) -> None:
    matrix = value.tocsr(copy=True)
    matrix.sort_indices()
    _save_npz(
        path,
        data=matrix.data.astype(np.float32),
        indices=matrix.indices.astype(np.int32),
        indptr=matrix.indptr.astype(np.int64),
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )


def _load_sparse(path: Path) -> sparse.csr_matrix:
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"data", "indices", "indptr", "shape"}:
            raise ValueError(f"invalid sparse artifact keys: {path.name}")
        shape = tuple(int(v) for v in np.asarray(artifact["shape"]).tolist())
        matrix = sparse.csr_matrix(
            (
                np.asarray(artifact["data"], dtype=np.float32),
                np.asarray(artifact["indices"], dtype=np.int32),
                np.asarray(artifact["indptr"], dtype=np.int64),
            ),
            shape=shape,
        )
    if matrix.shape[1] != X_DIM or not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"invalid X matrix: {path.name}")
    return matrix


def edge_type_matrix(occurrences: CompactOccurrences, distance_median: float) -> np.ndarray:
    """Return exact E0.R type for every ordered local-index pair; diagonal is sentinel 255."""
    n = len(occurrences)
    if n == 0:
        return np.empty((0, 0), dtype=np.uint8)
    x = occurrences.x.astype(np.float64) / 47.0
    y = occurrences.y.astype(np.float64) / 47.0
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    relation = ((np.hypot(dx, dy) > distance_median).astype(np.int8) * 4 + _quadrants(dx, dy)).astype(np.uint8)
    np.fill_diagonal(relation, np.uint8(255))
    return relation


def graph_primitives(all_occurrences: list[CompactOccurrences], distance_median: float) -> dict[str, np.ndarray]:
    node_offsets = np.zeros(len(all_occurrences) + 1, dtype=np.int64)
    relation_offsets = np.zeros(len(all_occurrences) + 1, dtype=np.int64)
    nodes: list[np.ndarray] = []
    relations: list[np.ndarray] = []
    for row, occurrences in enumerate(all_occurrences):
        component = np.asarray(occurrences.components, dtype=np.int16)
        relation = edge_type_matrix(occurrences, distance_median)
        nodes.append(component)
        relations.append(relation.reshape(-1))
        node_offsets[row + 1] = node_offsets[row] + len(component)
        relation_offsets[row + 1] = relation_offsets[row] + relation.size
    return {
        "node_offsets": node_offsets,
        "components": np.concatenate(nodes) if nodes else np.empty(0, dtype=np.int16),
        "relation_offsets": relation_offsets,
        "relations": np.concatenate(relations) if relations else np.empty(0, dtype=np.uint8),
    }


def reconstruct_aggregate_from_graph(graph: dict[str, np.ndarray]) -> tuple[np.ndarray, sparse.csr_matrix]:
    rows = len(graph["node_offsets"]) - 1
    o = np.zeros((rows, 128), dtype=np.float32)
    indptr = np.zeros(rows + 1, dtype=np.int64)
    index_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for row in range(rows):
        lo, hi = int(graph["node_offsets"][row]), int(graph["node_offsets"][row + 1])
        components = graph["components"][lo:hi].astype(np.int64)
        n = len(components)
        if n:
            o[row] = np.bincount(components, minlength=128).astype(np.float32) / np.float32(n)
        rlo, rhi = int(graph["relation_offsets"][row]), int(graph["relation_offsets"][row + 1])
        relation = graph["relations"][rlo:rhi].reshape(n, n)
        if n >= 2:
            src = np.repeat(components, n)
            dst = np.tile(components, n)
            mask = relation.reshape(-1) != 255
            columns = ((src[mask] * 128 + dst[mask]) * 8 + relation.reshape(-1)[mask]).astype(np.int32)
            unique, counts = np.unique(columns, return_counts=True)
            index_parts.append(unique)
            data_parts.append(counts.astype(np.float32) / np.float32(n * (n - 1)))
        else:
            index_parts.append(np.empty(0, dtype=np.int32))
            data_parts.append(np.empty(0, dtype=np.float32))
        indptr[row + 1] = indptr[row] + len(index_parts[-1])
    indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int32)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float32)
    return o, sparse.csr_matrix((data, indices, indptr), shape=(rows, G_DIM))


def _graph_sha256(graph: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("node_offsets", "components", "relation_offsets", "relations"):
        value = np.ascontiguousarray(graph[key])
        digest.update(key.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _canonical_ids(role: str) -> np.ndarray:
    return np.arange(TRAIN_ROWS, dtype=np.int32) if role == "train" else np.arange(TRAIN_ROWS, TRAIN_ROWS + PUBLIC_ROWS, dtype=np.int32)


def build_substrate(occurrences_path: str | Path, r1_results_path: str | Path, output_dir: str | Path, *, m0_scientific_sha: str) -> dict:
    if len(m0_scientific_sha) != 40:
        raise ValueError("M0 scientific SHA must be a full commit")
    occurrences_path, r1_results_path = Path(occurrences_path), Path(r1_results_path)
    if sha256_file(occurrences_path) != OCCURRENCES_SHA256 or sha256_file(r1_results_path) != R1_RESULTS_SHA256:
        raise ValueError("frozen E0.R input SHA mismatch")
    with np.load(occurrences_path, allow_pickle=False) as artifact:
        train_ids = np.asarray(artifact["train_ids"], dtype=np.int32)
        public_ids = np.asarray(artifact["public_ids"], dtype=np.int32)
        if not np.array_equal(train_ids, _canonical_ids("train")) or not np.array_equal(public_ids, _canonical_ids("public")):
            raise ValueError("canonical IDs mismatch")
        train_occurrences = _unflatten_occurrences(artifact, "train")
        public_occurrences = _unflatten_occurrences(artifact, "public")

    distance_median = train_pair_distance_median(train_occurrences)
    if distance_median != DISTANCE_MEDIAN:
        raise ValueError(f"frozen distance median mismatch: {distance_median!r}")

    train_o, public_o = occurrence_histograms(train_occurrences), occurrence_histograms(public_occurrences)
    train_g = build_geometry_csr(train_occurrences, train_ids, distance_median=distance_median)
    public_g = build_geometry_csr(public_occurrences, public_ids, distance_median=distance_median)
    train_x, public_x = nested_csr(train_o, train_g), nested_csr(public_o, public_g)
    train_graph, public_graph = graph_primitives(train_occurrences, distance_median), graph_primitives(public_occurrences, distance_median)
    parity_train_o, parity_train_g = reconstruct_aggregate_from_graph(train_graph)
    parity_public_o, parity_public_g = reconstruct_aggregate_from_graph(public_graph)
    if not np.array_equal(train_o, parity_train_o) or not np.array_equal(public_o, parity_public_o):
        raise RuntimeError("exhaustive O parity failure")
    if (train_g != parity_train_g).nnz or (public_g != parity_public_g).nnz:
        raise RuntimeError("exhaustive G parity failure")

    scaler = StandardScaler(with_mean=False)
    train_scaled = scaler.fit_transform(train_x).astype(np.float32).tocsr()
    public_scaled = scaler.transform(public_x).astype(np.float32).tocsr()

    # Labels are accessed only after every representation and parity check is complete.
    # Public labels are deliberately never indexed and are excluded from the substrate.
    with np.load(r1_results_path, allow_pickle=False) as result:
        if not np.array_equal(result["train_ids"], train_ids) or not np.array_equal(result["public_ids"], public_ids):
            raise ValueError("R1 result IDs mismatch")
        train_labels = np.asarray(result["train_labels"], dtype=np.int8)
    if train_labels.shape != (TRAIN_ROWS,) or np.any((train_labels < 0) | (train_labels >= 7)):
        raise ValueError("invalid Train labels")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _save_sparse(output / "train_x_scaled.npz", train_scaled)
    _save_sparse(output / "public_x_scaled.npz", public_scaled)
    _save_npz(output / "train_graphs.npz", **train_graph)
    _save_npz(output / "public_graphs.npz", **public_graph)
    _save_array(output / "train_ids.npy", train_ids)
    _save_array(output / "public_ids.npy", public_ids)
    _save_array(output / "train_labels.npy", train_labels)
    _save_npz(
        output / "scaler.npz",
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        var=np.asarray(scaler.var_, dtype=np.float64),
        n_samples_seen=np.asarray(scaler.n_samples_seen_, dtype=np.int64),
    )
    files = {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size} for name in SUBSTRATE_FILES}
    logical = json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False)
    substrate_sha = hashlib.sha256(logical.encode("utf-8")).hexdigest()
    manifest = {
        "experiment": EXPERIMENT_ID,
        "issue": ISSUE_NUMBER,
        "m0_scientific_sha": m0_scientific_sha,
        "e0r_scientific_sha": E0R_SCIENTIFIC_SHA,
        "occurrences_sha256": OCCURRENCES_SHA256,
        "r1_results_sha256": R1_RESULTS_SHA256,
        "distance_median": distance_median,
        "dimensions": {"O": 128, "G": G_DIM, "X": X_DIM},
        "train_ids_sha256": sha256_array(train_ids),
        "public_ids_sha256": sha256_array(public_ids),
        "o_train_sha256": sha256_array(train_o),
        "o_public_sha256": sha256_array(public_o),
        "g_train_sparse_sha256": sparse_sha256(train_g),
        "g_public_sparse_sha256": sparse_sha256(public_g),
        "x_train_sparse_sha256": sparse_sha256(train_x),
        "x_public_sparse_sha256": sparse_sha256(public_x),
        "x_scaled_train_sparse_sha256": sparse_sha256(train_scaled),
        "x_scaled_public_sparse_sha256": sparse_sha256(public_scaled),
        "graph_train_sha256": _graph_sha256(train_graph),
        "graph_public_sha256": _graph_sha256(public_graph),
        "files": files,
        "substrate_sha256": substrate_sha,
        "public_labels_included": False,
        "private_test_read": False,
        "parity": "EXHAUSTIVE_EXACT_PASS",
    }
    (output / "m0_substrate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class LoadedM0Substrate:
    root: Path
    manifest: dict
    train_x: sparse.csr_matrix
    public_x: sparse.csr_matrix
    train_ids: np.ndarray
    public_ids: np.ndarray
    train_labels: np.ndarray
    train_graph: dict[str, np.ndarray]
    public_graph: dict[str, np.ndarray]


def load_substrate(root: str | Path) -> LoadedM0Substrate:
    root = Path(root)
    manifest = json.loads((root / "m0_substrate_manifest.json").read_text(encoding="utf-8"))
    if manifest["experiment"] != EXPERIMENT_ID or manifest["issue"] != ISSUE_NUMBER:
        raise ValueError("substrate provenance mismatch")
    if manifest["public_labels_included"] or manifest["private_test_read"]:
        raise ValueError("substrate isolation violation")
    for name in SUBSTRATE_FILES:
        record = manifest["files"][name]
        path = root / name
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"substrate file mismatch: {name}")
    logical = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(logical.encode("utf-8")).hexdigest() != manifest["substrate_sha256"]:
        raise ValueError("substrate identity mismatch")
    train_ids = np.load(root / "train_ids.npy", allow_pickle=False)
    public_ids = np.load(root / "public_ids.npy", allow_pickle=False)
    if sha256_array(train_ids) != TRAIN_IDS_SHA256 or sha256_array(public_ids) != PUBLIC_IDS_SHA256:
        raise ValueError("canonical ID hash mismatch")
    def graph(name: str) -> dict[str, np.ndarray]:
        with np.load(root / name, allow_pickle=False) as artifact:
            return {key: np.asarray(artifact[key]).copy() for key in artifact.files}
    return LoadedM0Substrate(
        root=root,
        manifest=manifest,
        train_x=_load_sparse(root / "train_x_scaled.npz"),
        public_x=_load_sparse(root / "public_x_scaled.npz"),
        train_ids=train_ids,
        public_ids=public_ids,
        train_labels=np.load(root / "train_labels.npy", allow_pickle=False),
        train_graph=graph("train_graphs.npz"),
        public_graph=graph("public_graphs.npz"),
    )
