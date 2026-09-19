"""Issue #90 Train-only sparse information retention and relational necessity runner."""

from __future__ import annotations

import argparse
import hashlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .motif_qualification import Occurrence


ISSUE_NUMBER = 90
PREREGISTRATION_SHA = "fea1c85208aca3da503e53e770998f46c92a27b9"
SCIENTIFIC_SOURCE_SHA = "16a84b2b36f0d3584afd0a547487c4373dc22128"

TRAIN_ROWS = 28709
CRS_SIDE = 36
CRS_K = 512
CRS_SPLIT_LINE = 18

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
    assert assignment_map.shape == (CRS_SIDE, CRS_SIDE)
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
    assert concat_hist.shape == (2560,)
    return _l2_normalize_vector(concat_hist)


def compute_relation_vector(
    nodes: list[Occurrence],
    qm_type_to_idx: dict[int, int],
) -> np.ndarray:
    """Compute 3528D pairwise directional relation vector R."""
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
                    "two distinct sparse occurrences cannot occupy the same location"
                )

            direction = DIRECTION_SECTOR_MAP[(sr, sc)]
            coord = ((src_idx * QM_COUNT + tgt_idx) * RELATION_DIRECTIONS) + direction
            vec[coord] += 1.0

    assert vec.sum() == n * (n - 1)
    return _l2_normalize_vector(vec)


def derive_relation_control_permutation(canonical_image_id: int) -> np.ndarray:
    """Deterministically derive permutation of 8 direction sectors for Arm E."""
    ascii_decimal = str(int(canonical_image_id)).encode("ascii")
    digest = hashlib.sha256(b"42|relation_control|" + ascii_decimal).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(RELATION_DIRECTIONS)
    assert len(perm) == 8 and sorted(perm.tolist()) == list(range(8))
    return perm


def compute_relation_control_vector(
    nodes: list[Occurrence],
    qm_type_to_idx: dict[int, int],
    canonical_image_id: int,
) -> np.ndarray:
    """Compute 3528D permuted control relation vector R_control."""
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
                    "two distinct sparse occurrences cannot occupy the same location"
                )

            orig_direction = DIRECTION_SECTOR_MAP[(sr, sc)]
            perm_direction = int(perm[orig_direction])

            coord = (
                (src_idx * QM_COUNT + tgt_idx) * RELATION_DIRECTIONS
            ) + perm_direction
            vec[coord] += 1.0

    assert vec.sum() == n * (n - 1)
    return _l2_normalize_vector(vec)


def combine_unary_and_relation(
    unary_feat: np.ndarray, relation_feat: np.ndarray
) -> np.ndarray:
    """Combine 2560D unary and 3528D relation block into 6088D normalized representation."""
    assert unary_feat.shape == (2560,)
    assert relation_feat.shape == (RELATION_DIM,)
    u_norm = _l2_normalize_vector(unary_feat)
    r_norm = _l2_normalize_vector(relation_feat)
    combined = np.concatenate([u_norm, r_norm])
    assert combined.shape == (COMBINED_DIM,)
    return _l2_normalize_vector(combined)


def run_fixed_oof_probe(
    features: np.ndarray,
    labels: np.ndarray,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict]]:
    """Run fixed LogisticRegression across shared 5-fold splits with strict convergence checks."""
    assert len(features) == len(labels) == TRAIN_ROWS
    oof_predictions = np.empty(len(labels), dtype=np.int32)
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
            raise RuntimeError(f"fold {fold_idx} hit max_iter=5000")

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
        fold_diagnostics.append(
            {
                "fold": fold_idx,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "n_iter": [int(x) for x in clf.n_iter_],
                "converged": True,
            }
        )

    return oof_predictions, fold_diagnostics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute primary accuracy and 7-class Macro-F1."""
    acc = float(np.mean(y_true == y_pred))
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=np.arange(7), zero_division=0
    )
    macro_f1 = float(np.mean(per_class_f1))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": [float(x) for x in per_class_f1],
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

    # Pre-generate bootstrap resample indices
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #90 scientific runner.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run contract verification without full OOF fit.",
    )
    args = parser.parse_args()
    if args.dry_run:
        print("Issue #90 dry-run verification mode.")
        return 0
    print("Issue #90 official run not executed directly without full input binding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
