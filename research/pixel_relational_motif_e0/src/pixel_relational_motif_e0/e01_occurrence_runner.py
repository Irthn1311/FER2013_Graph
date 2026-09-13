from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .descriptor import extract_raw_relations
from .diag_gmm import DiagonalGaussianMixture
from .e01_runner import BASE_SHA, ISSUE_NUMBER, TRAIN_ROWS, load_official_train_images
from .occurrence import local_max_nms, stable_score_map

MEDIAN_NODE_BUDGET = 64
NODE_CAP = 80
MINIMUM_NODES = 2
NMS_RADIUS = 2


@dataclass(frozen=True)
class FrozenDescriptorTransform:
    pca_mean: np.ndarray
    pca_components: np.ndarray
    log_sigma_mean: float
    log_sigma_std: float

    def transform(self, s: np.ndarray, log_sigma: np.ndarray) -> np.ndarray:
        x = np.asarray(s, dtype=np.float64)
        l = np.asarray(log_sigma, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != 24 or len(x) != len(l):
            raise ValueError("expected S=(n,24) with matching log_sigma")
        if self.pca_mean.shape != (24,) or self.pca_components.shape != (11, 24):
            raise ValueError("invalid frozen PCA shape")
        if not np.isfinite(self.log_sigma_std) or self.log_sigma_std <= 0:
            raise ValueError("invalid frozen log-sigma scale")
        p = (x - self.pca_mean[None, :]) @ self.pca_components.T
        z = (l - self.log_sigma_mean) / self.log_sigma_std
        r = np.concatenate([p, z[:, None]], axis=1)
        if not np.all(np.isfinite(r)):
            raise FloatingPointError("non-finite frozen descriptor transform")
        return r


@dataclass(frozen=True)
class DictionaryArtifact:
    train_sha256: str
    transform: FrozenDescriptorTransform
    model: DiagonalGaussianMixture
    stable_components: np.ndarray
    selected_k: int


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def load_dictionary_artifact(path: str | Path) -> DictionaryArtifact:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        if _scalar_text(z["base_sha"]) != BASE_SHA:
            raise ValueError("dictionary base SHA does not match locked E0 source base")
        if int(np.asarray(z["issue"]).reshape(())) != ISSUE_NUMBER:
            raise ValueError("dictionary issue provenance mismatch")
        selected_k = int(np.asarray(z["selected_k"]).reshape(()))
        weights = np.asarray(z["canonical_weights"], dtype=np.float64)
        means = np.asarray(z["canonical_means"], dtype=np.float64)
        variances = np.asarray(z["canonical_variances"], dtype=np.float64)
        floor = np.asarray(z["variance_floor"], dtype=np.float64)
        stable = np.asarray(z["canonical_stable_components"], dtype=np.int64).reshape(-1)
        if weights.shape != (selected_k,) or means.shape != (selected_k, 12):
            raise ValueError("canonical GMM artifact shape mismatch")
        if variances.shape != means.shape or floor.shape != (12,):
            raise ValueError("canonical variance artifact shape mismatch")
        if len(stable) and (stable.min() < 0 or stable.max() >= selected_k):
            raise ValueError("stable component IDs outside canonical dictionary")
        model = DiagonalGaussianMixture(selected_k, floor)
        model.weights_ = weights
        model.means_ = means
        model.variances_ = variances
        transform = FrozenDescriptorTransform(
            pca_mean=np.asarray(z["pca_mean"], dtype=np.float64),
            pca_components=np.asarray(z["pca_components"], dtype=np.float64),
            log_sigma_mean=float(np.asarray(z["log_sigma_mean"]).reshape(())),
            log_sigma_std=float(np.asarray(z["log_sigma_std"]).reshape(())),
        )
        train_sha = _scalar_text(z["train_sha256"])
    return DictionaryArtifact(train_sha, transform, model, stable, selected_k)


def calibrate_tau_from_scores(
    score_arrays: list[np.ndarray],
    *,
    median_budget: int = MEDIAN_NODE_BUDGET,
) -> float:
    """Exact lowest observed confidence threshold with median count <= budget.

    Official FER2013 Train has 28,709 images (odd), so its sample median is the
    center order statistic. For that locked case, each image contributes only
    the (budget+1)-th confidence as a sufficient statistic for deciding when
    its count drops to the budget. This is exactly equivalent to scanning all
    unique confidences as `choose_tau_for_budget` does, without an O(N*T)
    threshold scan over all images.
    """
    if not score_arrays:
        raise ValueError("need at least one image")
    if median_budget < 0:
        raise ValueError("median_budget must be non-negative")
    arrays = [np.asarray(x, dtype=np.float64).reshape(-1) for x in score_arrays]
    n = len(arrays)
    if n % 2 == 0:
        raise ValueError("exact fast median calibration requires an odd image count")
    total_counts = np.asarray([len(x) for x in arrays], dtype=np.int64)
    if float(np.median(total_counts)) <= median_budget:
        return 0.0

    critical = np.full(n, -np.inf, dtype=np.float64)
    for i, scores in enumerate(arrays):
        if len(scores) > median_budget:
            ordered = np.sort(scores)[::-1]
            critical[i] = ordered[median_budget]
    m = (n + 1) // 2
    q = float(np.partition(critical, m - 1)[m - 1])
    successors = [x[x > q].min() for x in arrays if np.any(x > q)]
    if not successors:
        return 1.0
    return float(min(successors))


def occurrence_count_diagnostics(
    score_arrays: list[np.ndarray],
    *,
    tau: float,
    cap: int = NODE_CAP,
    minimum_nodes: int = MINIMUM_NODES,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cap <= 0 or minimum_nodes <= 0:
        raise ValueError("cap and minimum_nodes must be positive")
    candidate_counts = np.asarray([len(x) for x in score_arrays], dtype=np.int32)
    threshold_counts = np.asarray(
        [int(np.sum(np.asarray(x) >= tau)) for x in score_arrays], dtype=np.int32
    )
    cap_flags = threshold_counts > cap
    fallback_flags = threshold_counts < minimum_nodes
    final_counts = np.minimum(threshold_counts, cap).astype(np.int32)
    for i in np.flatnonzero(fallback_flags):
        final_counts[i] = min(minimum_nodes, int(candidate_counts[i]))
    diagnostics = {
        "median_nodes": float(np.median(final_counts)),
        "p90_nodes": float(np.percentile(final_counts, 90)),
        "p95_nodes": float(np.percentile(final_counts, 95)),
        "cap_rate": float(np.mean(cap_flags)),
        "fallback_rate": float(np.mean(fallback_flags)),
    }
    return diagnostics, final_counts, threshold_counts, candidate_counts, fallback_flags


def run_occurrence_calibration(
    train_csv: str | Path,
    dictionary_npz: str | Path,
    output_dir: str | Path,
) -> dict:
    """Calibrate locked E0 occurrence threshold on official Train only."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_dictionary_artifact(dictionary_npz)
    train = load_official_train_images(train_csv)
    if train.sha256 != artifact.train_sha256:
        raise ValueError("Train SHA256 does not match E0.1 dictionary provenance")

    if len(artifact.stable_components) == 0:
        counts_path = out_dir / "e01_occurrence_counts.npz"
        np.savez_compressed(
            counts_path,
            status=np.asarray("NO_STABLE_COMPONENTS"),
            train_sha256=np.asarray(train.sha256),
            stable_components=np.empty(0, dtype=np.int32),
            tau_star=np.empty(0, dtype=np.float64),
            final_node_counts=np.empty(0, dtype=np.int32),
            threshold_counts=np.empty(0, dtype=np.int32),
            candidate_counts=np.empty(0, dtype=np.int32),
            cap_flags=np.empty(0, dtype=np.bool_),
            fallback_flags=np.empty(0, dtype=np.bool_),
        )
        summary = {
            "experiment": "PGM_E0.1_occurrence_calibration",
            "issue": ISSUE_NUMBER,
            "base_sha": BASE_SHA,
            "train": {"rows": TRAIN_ROWS, "sha256": train.sha256},
            "private_test_read": False,
            "public_test_read": False,
            "selected_k": artifact.selected_k,
            "stable_component_count": 0,
            "status": "NO_STABLE_COMPONENTS",
            "tau_star": None,
            "diagnostics": None,
            "counts_artifact": counts_path.name,
        }
        (out_dir / "e01_occurrence_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    score_arrays: list[np.ndarray] = []
    for image_id in range(TRAIN_ROWS):
        image = np.asarray(train.images_uint8[image_id], dtype=np.float64) / 255.0
        s, l, _ = extract_raw_relations(image)
        r = artifact.transform.transform(s, l)
        posterior = artifact.model.predict_proba(r)
        score_map, motif_map = stable_score_map(posterior, artifact.stable_components)
        candidates = local_max_nms(score_map, motif_map, radius=NMS_RADIUS)
        score_arrays.append(np.asarray([o.confidence for o in candidates], dtype=np.float64))
        if (image_id + 1) % 1000 == 0 or image_id + 1 == TRAIN_ROWS:
            print(
                f"[PGM-E0.1-occurrence] processed {image_id + 1}/{TRAIN_ROWS} Train images",
                flush=True,
            )

    tau = calibrate_tau_from_scores(score_arrays, median_budget=MEDIAN_NODE_BUDGET)
    diagnostics, final_counts, threshold_counts, candidate_counts, fallback_flags = (
        occurrence_count_diagnostics(
            score_arrays,
            tau=tau,
            cap=NODE_CAP,
            minimum_nodes=MINIMUM_NODES,
        )
    )
    cap_flags = threshold_counts > NODE_CAP

    counts_path = out_dir / "e01_occurrence_counts.npz"
    np.savez_compressed(
        counts_path,
        train_sha256=np.asarray(train.sha256),
        tau_star=np.asarray(tau, dtype=np.float64),
        final_node_counts=final_counts,
        threshold_counts=threshold_counts,
        candidate_counts=candidate_counts,
        cap_flags=cap_flags.astype(np.bool_),
        fallback_flags=fallback_flags.astype(np.bool_),
    )
    summary = {
        "experiment": "PGM_E0.1_occurrence_calibration",
        "issue": ISSUE_NUMBER,
        "base_sha": BASE_SHA,
        "train": {"rows": TRAIN_ROWS, "sha256": train.sha256},
        "private_test_read": False,
        "public_test_read": False,
        "selected_k": artifact.selected_k,
        "stable_component_count": int(len(artifact.stable_components)),
        "stable_components": artifact.stable_components.tolist(),
        "posterior_rule": "max canonical stable-component posterior without renormalization",
        "local_maximum": "3x3",
        "nms_radius": NMS_RADIUS,
        "median_node_budget": MEDIAN_NODE_BUDGET,
        "node_cap": NODE_CAP,
        "minimum_nodes_fallback": MINIMUM_NODES,
        "tau_star": tau,
        "diagnostics": diagnostics,
        "counts_artifact": counts_path.name,
        "status": "OK",
    }
    (out_dir / "e01_occurrence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[PGM-E0.1-occurrence] tau*={tau:.8f} median={diagnostics['median_nodes']:.1f} "
        f"p90={diagnostics['p90_nodes']:.1f} p95={diagnostics['p95_nodes']:.1f} "
        f"cap_rate={diagnostics['cap_rate']:.6f} fallback_rate={diagnostics['fallback_rate']:.6f}",
        flush=True,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked PGM E0.1 occurrence calibration on official Train only."
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--dictionary-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    run_occurrence_calibration(args.train_csv, args.dictionary_npz, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
