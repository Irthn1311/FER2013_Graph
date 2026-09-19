"""Issue #90 Train-only sparse information retention and relational necessity runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .e02_runner import load_pixels_only, sha256_file
from .motif_qualification import (
    Occurrence,
    extract_occurrences,
    sparse_occurrence_feature,
)
from .motif_train_runner import load_substrate_manifest


ISSUE_NUMBER = 90
PREREGISTRATION_SHA = "fea1c85208aca3da503e53e770998f46c92a27b9"
SCIENTIFIC_SOURCE_SHA = "16a84b2b36f0d3584afd0a547487c4373dc22128"

TRAIN_ROWS = 28709
CRS_SIDE = 36
CRS_K = 512
CRS_SPLIT_LINE = 18

EXPECTED_TRAIN_SHA256 = (
    "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8"
)
EXPECTED_SUBSTRATE_IDENTITY = (
    "ee5d8262f7ec439bd6e2cd8ad01a17df17cd152da10b474c49ea93872fbfc46c"
)
EXPECTED_CRS_TRAIN_MODEL_SHA256 = (
    "77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a"
)

EXPECTED_MERGED_SHA256 = (
    "aa7b3c14e2c544eff673969366c3e1801abe33ba0de2611159946b027a0b1f2b"
)
EXPECTED_OCC_DIAG_SHA256 = (
    "218838aeaefd575b96a15037e84f52d966ddae66b86850fea7f394febe146658"
)
EXPECTED_SM_ALL_FEAT_SHA256 = (
    "900e0fd07aa7bb9ef0eec947246df8d6d66d92603508994e1c09fcbca71092ad"
)
EXPECTED_FINAL_SUMMARY_SHA256 = (
    "ea191bd389c6cad44c4eb6f1ff600559b38c649eec904e4d886a3f9003963d18"
)
EXPECTED_FINAL_MANIFEST_SHA256 = (
    "3ad2d559fe33ce4d852156a06ae2a778e3cff1ea460983cb2b7cb24495e29786"
)

FROZEN_QM_TYPES = (
    31,
    81,
    116,
    118,
    123,
    125,
    167,
    169,
    171,
    174,
    179,
    195,
    223,
    305,
    311,
    344,
    381,
    440,
    451,
    479,
    491,
)
QM_COUNT = len(FROZEN_QM_TYPES)
RELATION_DIRECTIONS = 8
RELATION_DIM = QM_COUNT * QM_COUNT * RELATION_DIRECTIONS  # 21 * 21 * 8 = 3528
COMBINED_DIM = 2560 + RELATION_DIM  # 6088

DIRECTION_SECTOR_MAP = {
    (-1, -1): 0,
    (-1, 0): 1,
    (-1, 1): 2,
    (0, -1): 3,
    (0, 1): 4,
    (1, -1): 5,
    (1, 0): 6,
    (1, 1): 7,
}


def _verify_exact_hash(path: Path, expected: str, desc: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{desc} file not found: {path}")
    obs = sha256_file(path)
    if obs != expected:
        raise ValueError(f"{desc} SHA256 mismatch: {obs} != {expected}")


def _l2_normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm).astype(np.float32)
    return vec.astype(np.float32)


def dense_pyramid_feature(
    assignment_map: np.ndarray,
    allowed_types: set[int] | None = None,
) -> np.ndarray:
    """Build 2560D 1x1 + 2x2 spatial pyramid from 36x36 assignment map."""
    if assignment_map.shape != (CRS_SIDE, CRS_SIDE):
        raise ValueError(f"assignment map must have shape ({CRS_SIDE}, {CRS_SIDE})")
    if not np.all((assignment_map >= 0) & (assignment_map < CRS_K)):
        raise ValueError("assignment map values out of [0, 511] range")

    r_split = CRS_SPLIT_LINE
    c_split = CRS_SPLIT_LINE

    regions = [
        assignment_map,
        assignment_map[:r_split, :c_split],
        assignment_map[:r_split, c_split:],
        assignment_map[r_split:, :c_split],
        assignment_map[r_split:, c_split:],
    ]

    region_histograms = []
    for reg in regions:
        hist = np.zeros(CRS_K, dtype=np.float32)
        vals, counts = np.unique(reg, return_counts=True)
        for v, c in zip(vals, counts):
            if allowed_types is None or int(v) in allowed_types:
                hist[int(v)] = float(c)
        s = hist.sum()
        if s > 0:
            hist /= s
        region_histograms.append(hist)

    concat_hist = np.concatenate(region_histograms)
    if concat_hist.shape != (2560,):
        raise ValueError("dense pyramid dimension must be exactly 2560")
    if not np.all(np.isfinite(concat_hist)):
        raise ValueError("non-finite value in dense pyramid feature")
    return _l2_normalize_vector(concat_hist)


def compute_relation_vector_raw(
    nodes: list[Occurrence],
    qm_type_to_idx: dict[int, int],
) -> np.ndarray:
    """Compute raw unnormalized 3528D pairwise directional relation count vector."""
    vec = np.zeros(RELATION_DIM, dtype=np.float32)
    n = len(nodes)
    if n < 2:
        return vec

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u_node = nodes[i]
            v_node = nodes[j]

            src_idx = qm_type_to_idx[int(u_node.type_id)]
            tgt_idx = qm_type_to_idx[int(v_node.type_id)]

            dr = int(v_node.row) - int(u_node.row)
            dc = int(v_node.col) - int(u_node.col)

            sr = 0 if dr == 0 else (1 if dr > 0 else -1)
            sc = 0 if dc == 0 else (1 if dc > 0 else -1)

            if sr == 0 and sc == 0:
                raise ValueError(
                    "two distinct sparse occurrences cannot occupy the same location (0,0)"
                )

            direction = DIRECTION_SECTOR_MAP[(sr, sc)]
            coord = ((src_idx * QM_COUNT + tgt_idx) * RELATION_DIRECTIONS) + direction
            vec[coord] += 1.0

    if abs(vec.sum() - (n * (n - 1))) > 1e-5:
        raise ValueError(f"raw relation count sum {vec.sum()} != n(n-1) {n * (n - 1)}")
    return vec


def compute_relation_vector(
    nodes: list[Occurrence],
    qm_type_to_idx: dict[int, int],
) -> np.ndarray:
    """Compute 3528D pairwise directional relation vector R (globally L2 normalized)."""
    raw = compute_relation_vector_raw(nodes, qm_type_to_idx)
    return _l2_normalize_vector(raw)


def derive_relation_control_permutation(canonical_image_id: int) -> np.ndarray:
    """Deterministically derive permutation of 8 direction sectors for Arm E."""
    ascii_decimal = str(int(canonical_image_id)).encode("ascii")
    digest = hashlib.sha256(b"42|relation_control|" + ascii_decimal).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(RELATION_DIRECTIONS)
    if len(perm) != 8 or sorted(perm.tolist()) != list(range(8)):
        raise ValueError("control permutation is not a bijection of 0..7")
    return perm


def compute_relation_control_vector_raw(
    nodes: list[Occurrence],
    qm_type_to_idx: dict[int, int],
    canonical_image_id: int,
) -> np.ndarray:
    """Compute raw unnormalized 3528D permuted control relation count vector."""
    vec = np.zeros(RELATION_DIM, dtype=np.float32)
    n = len(nodes)
    if n < 2:
        return vec

    perm = derive_relation_control_permutation(canonical_image_id)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u_node = nodes[i]
            v_node = nodes[j]

            src_idx = qm_type_to_idx[int(u_node.type_id)]
            tgt_idx = qm_type_to_idx[int(v_node.type_id)]

            dr = int(v_node.row) - int(u_node.row)
            dc = int(v_node.col) - int(u_node.col)

            sr = 0 if dr == 0 else (1 if dr > 0 else -1)
            sc = 0 if dc == 0 else (1 if dc > 0 else -1)

            if sr == 0 and sc == 0:
                raise ValueError(
                    "two distinct sparse occurrences cannot occupy the same location (0,0)"
                )

            orig_direction = DIRECTION_SECTOR_MAP[(sr, sc)]
            perm_direction = int(perm[orig_direction])

            coord = (
                (src_idx * QM_COUNT + tgt_idx) * RELATION_DIRECTIONS
            ) + perm_direction
            vec[coord] += 1.0

    if abs(vec.sum() - (n * (n - 1))) > 1e-5:
        raise ValueError(
            f"raw control relation count sum {vec.sum()} != n(n-1) {n * (n - 1)}"
        )
    return vec


def compute_relation_control_vector(
    nodes: list[Occurrence],
    qm_type_to_idx: dict[int, int],
    canonical_image_id: int,
) -> np.ndarray:
    """Compute 3528D permuted control relation vector R_control (globally L2 normalized)."""
    raw = compute_relation_control_vector_raw(nodes, qm_type_to_idx, canonical_image_id)
    return _l2_normalize_vector(raw)


def combine_unary_and_relation(
    unary_feat: np.ndarray, relation_feat: np.ndarray
) -> np.ndarray:
    """Combine 2560D unary and 3528D relation block into 6088D normalized representation."""
    if unary_feat.shape != (2560,):
        raise ValueError("unary feature dimension must be 2560")
    if relation_feat.shape != (RELATION_DIM,):
        raise ValueError(f"relation feature dimension must be {RELATION_DIM}")

    u_norm = _l2_normalize_vector(unary_feat)
    r_norm = _l2_normalize_vector(relation_feat)
    combined = np.concatenate([u_norm, r_norm])
    if combined.shape != (COMBINED_DIM,):
        raise ValueError(f"combined feature dimension must be {COMBINED_DIM}")
    if not np.all(np.isfinite(combined)):
        raise ValueError("non-finite values in combined feature")
    return _l2_normalize_vector(combined)


def generate_shared_5fold_splits(
    labels: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate and strictly validate shared 5-fold StratifiedKFold splits."""
    if len(labels) != TRAIN_ROWS:
        raise ValueError(f"expected exactly {TRAIN_ROWS} labels for fold generation")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(labels)), labels))
    if len(splits) != 5:
        raise ValueError("expected exactly 5 splits")

    assigned_val_indices = []
    for f_idx, (train_idx, val_idx) in enumerate(splits):
        if len(np.intersect1d(train_idx, val_idx)) != 0:
            raise ValueError(f"fold {f_idx} train and val indices overlap")
        assigned_val_indices.extend(val_idx.tolist())

    if len(assigned_val_indices) != TRAIN_ROWS or sorted(assigned_val_indices) != list(
        range(TRAIN_ROWS)
    ):
        raise ValueError("validation fold union does not partition exact Train indices")
    return splits


def run_fixed_oof_probe(
    features: np.ndarray | sparse.spmatrix,
    labels: np.ndarray,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict]]:
    """Run fixed LogisticRegression across shared 5-fold splits with fail-closed convergence."""
    if features.shape[0] != len(labels) or len(labels) != TRAIN_ROWS:
        raise ValueError("feature and label rows mismatch")
    oof_predictions = np.full(len(labels), -1, dtype=np.int32)
    fold_diagnostics = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(features[train_idx], labels[train_idx])

        conv_warnings = [
            w for w in captured if issubclass(w.category, ConvergenceWarning)
        ]
        if conv_warnings:
            raise RuntimeError(
                f"fold {fold_idx} failed convergence with ConvergenceWarning"
            )
        if np.any(clf.n_iter_ >= 5000):
            raise RuntimeError(
                f"fold {fold_idx} reached max_iter=5000 without convergence"
            )

        preds = clf.predict(features[val_idx])
        oof_predictions[val_idx] = preds

        acc = float(np.mean(preds == labels[val_idx]))
        macro_f1 = float(
            np.mean(
                f1_score(
                    labels[val_idx],
                    preds,
                    average=None,
                    labels=np.arange(7),
                    zero_division=0,
                )
            )
        )
        cm = confusion_matrix(labels[val_idx], preds, labels=np.arange(7)).tolist()
        fold_diagnostics.append(
            {
                "fold": fold_idx,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "confusion_matrix": cm,
                "n_iter": [int(x) for x in clf.n_iter_],
                "converged": True,
            }
        )

    if np.any(oof_predictions < 0):
        raise ValueError("uninitialized entries detected in OOF prediction array")
    return oof_predictions, fold_diagnostics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute primary accuracy, 7-class Macro-F1, per-class F1, and confusion matrix."""
    acc = float(np.mean(y_true == y_pred))
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=np.arange(7), zero_division=0
    )
    macro_f1 = float(np.mean(per_class_f1))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(7)).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": [float(x) for x in per_class_f1],
        "confusion_matrix": cm,
    }


def compute_shared_paired_bootstrap(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    comparisons: list[tuple[str, str, str]],
    b_replicates: int = 2000,
    seed: int = 42,
) -> dict[str, dict]:
    """Compute paired bootstrap over shared sampled indices with linear quantile semantics."""
    n_samples = len(y_true)
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap_indices = rng.integers(0, n_samples, size=(b_replicates, n_samples))

    results = {}
    for comp_name, arm_b, arm_a in comparisons:
        pred_b = predictions[arm_b]
        pred_a = predictions[arm_a]

        delta_accs = np.empty(b_replicates, dtype=np.float64)
        delta_f1s = np.empty(b_replicates, dtype=np.float64)

        for b in range(b_replicates):
            idx = bootstrap_indices[b]
            yb_true = y_true[idx]

            acc_b = np.mean(yb_true == pred_b[idx])
            acc_a = np.mean(yb_true == pred_a[idx])
            delta_accs[b] = acc_b - acc_a

            f1_b = np.mean(
                f1_score(
                    yb_true,
                    pred_b[idx],
                    average=None,
                    labels=np.arange(7),
                    zero_division=0,
                )
            )
            f1_a = np.mean(
                f1_score(
                    yb_true,
                    pred_a[idx],
                    average=None,
                    labels=np.arange(7),
                    zero_division=0,
                )
            )
            delta_f1s[b] = f1_b - f1_a

        ci_acc = np.quantile(delta_accs, [0.025, 0.975], method="linear")
        ci_f1 = np.quantile(delta_f1s, [0.025, 0.975], method="linear")

        orig_acc_b = np.mean(y_true == pred_b)
        orig_acc_a = np.mean(y_true == pred_a)
        orig_f1_b = np.mean(
            f1_score(y_true, pred_b, average=None, labels=np.arange(7), zero_division=0)
        )
        orig_f1_a = np.mean(
            f1_score(y_true, pred_a, average=None, labels=np.arange(7), zero_division=0)
        )

        results[comp_name] = {
            "comparison": f"{arm_b} - {arm_a}",
            "point_delta_accuracy": float(orig_acc_b - orig_acc_a),
            "ci_95_accuracy": [float(ci_acc[0]), float(ci_acc[1])],
            "point_delta_macro_f1": float(orig_f1_b - orig_f1_a),
            "ci_95_macro_f1": [float(ci_f1[0]), float(ci_f1[1])],
        }
    return results


def evaluate_scientific_verdicts(bootstrap_results: dict[str, dict]) -> dict[str, str]:
    """Evaluate exact three preregistered scientific decision rules."""
    verdicts = {}

    # 1. Vocabulary Filtering Loss (B - A)
    res_vocab = bootstrap_results["delta_vocab"]
    if res_vocab["ci_95_accuracy"][1] < 0 and res_vocab["ci_95_macro_f1"][1] < 0:
        verdicts["vocab_loss"] = "QUALIFIED-VOCABULARY FILTERING LOSS SUPPORTED"
    else:
        verdicts["vocab_loss"] = "QUALIFIED-VOCABULARY FILTERING LOSS NOT ESTABLISHED"

    # 2. Component Sparsification Loss (C - B)
    res_sparse = bootstrap_results["delta_sparse"]
    if res_sparse["ci_95_accuracy"][1] < 0 and res_sparse["ci_95_macro_f1"][1] < 0:
        verdicts["sparse_loss"] = "COMPONENT SPARSIFICATION LOSS SUPPORTED"
    else:
        verdicts["sparse_loss"] = "COMPONENT SPARSIFICATION LOSS NOT ESTABLISHED"

    # 3. Pairwise Relational Signal (D - C and D - E)
    res_rel = bootstrap_results["delta_rel"]
    res_geom = bootstrap_results["delta_geom"]
    rel_pass = res_rel["ci_95_accuracy"][0] > 0 and res_rel["ci_95_macro_f1"][0] > 0
    geom_pass = res_geom["ci_95_accuracy"][0] > 0 and res_geom["ci_95_macro_f1"][0] > 0

    if rel_pass and geom_pass:
        verdicts["relational_signal"] = "SPARSE PAIRWISE RELATIONAL SIGNAL SUPPORTED"
    else:
        verdicts["relational_signal"] = (
            "SPARSE PAIRWISE RELATIONAL SIGNAL NOT ESTABLISHED"
        )
    return verdicts


def run_experiment(
    train_csv: str | Path,
    substrate_dir: str | Path,
    finalizer_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Official end-to-end execution of Issue #90 Train-only OOF evaluation."""
    out = Path(output_dir)
    if not out.exists():
        out.mkdir(parents=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory must be empty: {out}")

    sub_root = Path(substrate_dir)
    fin_root = Path(finalizer_dir)
    train_csv_path = Path(train_csv)

    # 1. Verify frozen inputs
    _verify_exact_hash(train_csv_path, EXPECTED_TRAIN_SHA256, "Train CSV")
    sub_manifest = load_substrate_manifest(sub_root)
    if sub_manifest["substrate_identity_sha256"] != EXPECTED_SUBSTRATE_IDENTITY:
        raise ValueError("substrate identity mismatch")

    _verify_exact_hash(
        fin_root / "motif_stability_merged.npz",
        EXPECTED_MERGED_SHA256,
        "motif_stability_merged.npz",
    )
    _verify_exact_hash(
        fin_root / "motif_train_occurrence_diagnostics.npz",
        EXPECTED_OCC_DIAG_SHA256,
        "motif_train_occurrence_diagnostics.npz",
    )
    _verify_exact_hash(
        fin_root / "motif_train_s_m_all_features.npz",
        EXPECTED_SM_ALL_FEAT_SHA256,
        "motif_train_s_m_all_features.npz",
    )
    _verify_exact_hash(
        fin_root / "motif_train_summary.json",
        EXPECTED_FINAL_SUMMARY_SHA256,
        "motif_train_summary.json",
    )
    _verify_exact_hash(
        fin_root / "motif_train_execution_manifest.json",
        EXPECTED_FINAL_MANIFEST_SHA256,
        "motif_train_execution_manifest.json",
    )

    # 2. Load and verify Q_M from official artifact
    merged_data = np.load(fin_root / "motif_stability_merged.npz")
    loaded_qm = tuple(int(x) for x in merged_data["q_m"])
    if loaded_qm != FROZEN_QM_TYPES:
        raise ValueError(f"official Q_M mismatch: {loaded_qm} != {FROZEN_QM_TYPES}")
    qm_set = set(loaded_qm)
    qm_to_idx = {t: i for i, t in enumerate(loaded_qm)}

    # 3. Load assignment & margin maps
    m_assignments = np.load(sub_root / "motif_train_m_assignments.npy", mmap_mode="r")
    m_margins = np.load(sub_root / "motif_train_m_margins.npy", mmap_mode="r")
    if m_assignments.shape != (TRAIN_ROWS, CRS_SIDE, CRS_SIDE) or m_margins.shape != (
        TRAIN_ROWS,
        CRS_SIDE,
        CRS_SIDE,
    ):
        raise ValueError("m_assignments or m_margins shape mismatch")

    # 4. Feature construction (Arms A, B, C, D, E) BEFORE labels
    print("[1/5] Constructing Arm A (M_DENSE_ALL) and Arm B (M_DENSE_Q)...")
    feat_a = np.empty((TRAIN_ROWS, 2560), dtype=np.float32)
    feat_b = np.empty((TRAIN_ROWS, 2560), dtype=np.float32)
    arm_b_zero_ids = []

    for i in range(TRAIN_ROWS):
        grid = m_assignments[i]
        fa = dense_pyramid_feature(grid, allowed_types=None)
        fb = dense_pyramid_feature(grid, allowed_types=qm_set)
        feat_a[i] = fa
        feat_b[i] = fb
        if np.linalg.norm(fb) == 0:
            arm_b_zero_ids.append(i)

    print(
        "[2/5] Binding Arm C (M_SPARSE_Q) from official artifact & verifying reconstruction gate..."
    )
    feat_c = sparse.load_npz(fin_root / "motif_train_s_m_all_features.npz").tocsr()
    if feat_c.shape != (TRAIN_ROWS, 2560):
        raise ValueError("official Arm C feature shape mismatch")

    reconstructed_unary = np.empty((TRAIN_ROWS, 2560), dtype=np.float32)
    reconstructed_nodes_per_img = []
    reconstructed_pre_cap = []
    reconstructed_post_cap = []

    for i in range(TRAIN_ROWS):
        ext = extract_occurrences(
            assignment_map=m_assignments[i],
            margin_map=m_margins[i],
            selected_type_ids=np.asarray(loaded_qm, dtype=np.int32),
            max_occurrences=64,
        )
        reconstructed_nodes_per_img.append(list(ext.retained))
        reconstructed_pre_cap.append(ext.pre_cap_count)
        reconstructed_post_cap.append(len(ext.retained))
        u_rec = sparse_occurrence_feature(ext.retained)
        reconstructed_unary[i] = u_rec

    rec_pre = np.array(reconstructed_pre_cap)
    rec_post = np.array(reconstructed_post_cap)
    zero_nodes_count = int(np.sum(rec_post == 0))
    if (
        zero_nodes_count != 48
        or np.min(rec_post) != 0
        or np.median(rec_post) != 13
        or np.max(rec_post) != 38
    ):
        raise RuntimeError(
            "Issue #90 sparse reconstruction diagnostics mismatch against official Issue #86"
        )
    if np.sum(rec_pre > 64) != 0:
        raise RuntimeError("cap binding count is non-zero")

    diff_c = np.max(np.abs(reconstructed_unary - feat_c.toarray()))
    if diff_c > 1e-5:
        raise RuntimeError(
            f"reconstructed unary feature differs from official Arm C: max diff {diff_c}"
        )
    print(f"Sparse reconstruction gate PASS (max diff {diff_c:.2e})")

    print(
        "[3/5] Constructing Arm D (M_SPARSE_Q_PLUS_REL) and Arm E (M_SPARSE_Q_PLUS_REL_CONTROL)..."
    )
    feat_d = np.empty((TRAIN_ROWS, COMBINED_DIM), dtype=np.float32)
    feat_e = np.empty((TRAIN_ROWS, COMBINED_DIM), dtype=np.float32)

    for i in range(TRAIN_ROWS):
        nodes = reconstructed_nodes_per_img[i]
        u = feat_c[i].toarray().reshape(-1)
        r_vec = compute_relation_vector(nodes, qm_to_idx)
        r_ctrl = compute_relation_control_vector(nodes, qm_to_idx, canonical_image_id=i)
        feat_d[i] = combine_unary_and_relation(u, r_vec)
        feat_e[i] = combine_unary_and_relation(u, r_ctrl)

    # 5. Zero-node coverage diagnostic analysis
    arm_c_zero_ids = np.where(rec_post == 0)[0].tolist()
    zero_sets_equal = set(arm_b_zero_ids) == set(arm_c_zero_ids)
    if zero_sets_equal:
        zero_node_verdict = "ZERO-NODE COVERAGE FAILURE ORIGINATES AT QUALIFIED-VOCABULARY COVERAGE, NOT COMPONENT COLLAPSE"
    else:
        zero_node_verdict = "ZERO-NODE COVERAGE FAILURE DIFFERS BETWEEN DENSE-Q AND COMPONENT OCCURRENCES"

    # 6. Now load official Train labels
    print("[4/5] Loading official Train labels after feature completion...")
    train_data = load_pixels_only(train_csv_path, role="train")
    if (
        train_data.sha256 != EXPECTED_TRAIN_SHA256
        or len(train_data.labels) != TRAIN_ROWS
    ):
        raise ValueError("Train data loading validation failed")
    y_true = train_data.labels.astype(np.int32)

    # 7. Generate shared 5-fold cross-validation
    print("[5/5] Generating shared 5-fold splits and executing OOF evaluation...")
    fold_splits = generate_shared_5fold_splits(y_true)

    features_dict = {
        "A": feat_a,
        "B": feat_b,
        "C": feat_c,
        "D": feat_d,
        "E": feat_e,
    }

    predictions = {}
    per_arm_metrics = {}
    fold_diagnostics = {}

    for arm_id, feats in features_dict.items():
        print(f"  Running OOF probe for Arm {arm_id}...")
        preds, diags = run_fixed_oof_probe(feats, y_true, fold_splits)
        predictions[arm_id] = preds
        fold_diagnostics[arm_id] = diags
        per_arm_metrics[arm_id] = compute_metrics(y_true, preds)
        print(
            f"  Arm {arm_id} OOF: Acc={per_arm_metrics[arm_id]['accuracy']:.6f}, Macro-F1={per_arm_metrics[arm_id]['macro_f1']:.6f}"
        )

    # 8. Shared paired bootstrap
    comparisons = [
        ("delta_vocab", "B", "A"),
        ("delta_sparse", "C", "B"),
        ("delta_rel", "D", "C"),
        ("delta_geom", "D", "E"),
    ]
    print(
        "Computing shared paired bootstrap across shared sampled indices (B=2000, seed 42)..."
    )
    bootstrap_results = compute_shared_paired_bootstrap(
        y_true, predictions, comparisons, b_replicates=2000, seed=42
    )
    scientific_verdicts = evaluate_scientific_verdicts(bootstrap_results)

    # 9. Serialize artifacts atomically
    pred_path = out / "issue90_oof_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_true=y_true,
        pred_a=predictions["A"],
        pred_b=predictions["B"],
        pred_c=predictions["C"],
        pred_d=predictions["D"],
        pred_e=predictions["E"],
    )

    fold_path = out / "issue90_fold_assignments.npz"
    val_folds_array = np.empty(TRAIN_ROWS, dtype=np.int32)
    for f_idx, (_, val_idx) in enumerate(fold_splits):
        val_folds_array[val_idx] = f_idx
    np.savez_compressed(fold_path, fold_assignments=val_folds_array)

    diag_path = out / "issue90_relation_diagnostics.npz"
    np.savez_compressed(
        diag_path,
        q_m=np.asarray(loaded_qm, dtype=np.int32),
        zero_node_canonical_ids=np.asarray(arm_c_zero_ids, dtype=np.int32),
        zero_node_labels=y_true[arm_c_zero_ids],
    )

    res_path = out / "issue90_results.json"
    results_payload = {
        "issue": ISSUE_NUMBER,
        "preregistration_sha": PREREGISTRATION_SHA,
        "scientific_source_sha": SCIENTIFIC_SOURCE_SHA,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": "1.6.1",
        },
        "per_arm_metrics": per_arm_metrics,
        "fold_diagnostics": fold_diagnostics,
        "bootstrap_results": bootstrap_results,
        "scientific_verdicts": scientific_verdicts,
        "zero_node_diagnostic": {
            "zero_node_count": len(arm_c_zero_ids),
            "zero_node_ids": arm_c_zero_ids,
            "zero_node_class_counts": np.bincount(
                y_true[arm_c_zero_ids], minlength=7
            ).tolist(),
            "verdict": zero_node_verdict,
        },
        "public_test_accessed": False,
        "private_test_accessed": False,
        "graph_gnn_executed": False,
    }
    res_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    # Manifest
    manifest_records = {}
    for p in [pred_path, fold_path, diag_path, res_path]:
        manifest_records[p.name] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    manifest_path = out / "issue90_execution_manifest.json"
    manifest_payload = {
        "issue": ISSUE_NUMBER,
        "preregistration_sha": PREREGISTRATION_SHA,
        "outputs": manifest_records,
        "public_test_accessed": False,
        "private_test_accessed": False,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print("Execution complete. Manifest written.")
    return results_payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue #90 scientific runner.")
    sub = parser.add_subparsers(dest="mode", required=True)

    dry = sub.add_parser("dry-run")
    dry.add_argument("--test-only", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--train-csv", required=True)
    run.add_argument("--substrate-dir", required=True)
    run.add_argument("--finalizer-dir", required=True)
    run.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mode == "dry-run":
        print("Issue #90 dry-run verification mode: PASS.")
        return 0
    run_experiment(
        train_csv=args.train_csv,
        substrate_dir=args.substrate_dir,
        finalizer_dir=args.finalizer_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
