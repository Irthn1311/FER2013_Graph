from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .e01b_diagnostic import finite_spearman
from .e02_runner import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PROBE_SPEC,
    PixelData,
    load_labels_downstream,
    load_pixels_only,
    sha256_array,
    sha256_file,
)
from .e0r_frozen import (
    CONTROL_ARTIFACT_SHA256,
    E02_RESULTS_SHA256,
    E02_SUMMARY_SHA256,
    MATCHED_POOL_SHA256,
    FrozenCondition,
    load_frozen_conditions,
)
from .e0r_geometry import (
    GEOMETRY_SEEDS,
    build_geometry_csr,
    nested_csr,
    sparse_sha256,
    train_pair_distance_median,
)
from .e0r_occurrence import (
    NODE_CAP,
    CompactOccurrences,
    apply_no_fallback,
    exact_tau,
    extract_candidates,
    occurrence_diagnostics,
    occurrence_histograms,
)
from .probe import fit_probe, metrics, paired_bootstrap_vs_control_mean


ISSUE_NUMBER = 80
BASE_SHA = "6e6685bb7e56c9b55999eea9a2c810bf42966b7e"
CONTROL_SEEDS = tuple(sorted(CONTROL_ARTIFACT_SHA256))


@dataclass
class ConditionResult:
    tau: float
    train_features: np.ndarray
    public_features: np.ndarray
    train_diagnostics: dict[str, object]
    public_diagnostics: dict[str, object]
    train_occurrences: list[CompactOccurrences] | None
    public_occurrences: list[CompactOccurrences] | None


def _log(message: str) -> None:
    print(f"[PGM-E0.R] {message}", flush=True)


def registered_joint_verdict(
    accuracy_ci95: np.ndarray,
    macro_f1_ci95: np.ndarray,
    *,
    stage: str,
) -> str:
    if stage not in {"R1", "R2"}:
        raise ValueError("stage must be R1 or R2")
    accuracy_pass = float(np.asarray(accuracy_ci95)[0]) > 0.0
    f1_pass = float(np.asarray(macro_f1_ci95)[0]) > 0.0
    if accuracy_pass and f1_pass:
        return f"E0.{stage} SUPPORTED"
    if accuracy_pass != f1_pass:
        return f"E0.{stage} MIXED — NOT SUPPORTED BY REGISTERED JOINT CRITERION"
    return f"E0.{stage} NOT SUPPORTED"


def _extract_split_candidates(
    data: PixelData,
    condition: FrozenCondition,
) -> list[CompactOccurrences]:
    out: list[CompactOccurrences] = []
    for position, canonical_id in enumerate(data.canonical_ids):
        out.append(
            extract_candidates(
                data.images_uint8[position],
                canonical_image_id=int(canonical_id),
                condition=condition,
            )
        )
        if (position + 1) % 500 == 0 or position + 1 == len(data.canonical_ids):
            _log(f"{condition.name} {data.role} candidates {position + 1}/{len(data.canonical_ids)}")
    return out


def run_occurrence_condition(
    train: PixelData,
    public: PixelData,
    condition: FrozenCondition,
    *,
    retain_occurrences: bool,
) -> ConditionResult:
    train_candidates = _extract_split_candidates(train, condition)
    tau = exact_tau([item.confidence for item in train_candidates])
    train_occurrences: list[CompactOccurrences] = []
    train_cap = np.zeros(len(train_candidates), dtype=bool)
    for index, candidates in enumerate(train_candidates):
        retained, cap_used = apply_no_fallback(candidates, tau=tau, cap=NODE_CAP)
        train_occurrences.append(retained)
        train_cap[index] = cap_used
    del train_candidates
    train_features = occurrence_histograms(train_occurrences)
    train_diagnostics = occurrence_diagnostics(train_occurrences, train_cap)

    public_candidates = _extract_split_candidates(public, condition)
    public_occurrences: list[CompactOccurrences] = []
    public_cap = np.zeros(len(public_candidates), dtype=bool)
    for index, candidates in enumerate(public_candidates):
        retained, cap_used = apply_no_fallback(candidates, tau=tau, cap=NODE_CAP)
        public_occurrences.append(retained)
        public_cap[index] = cap_used
    del public_candidates
    public_features = occurrence_histograms(public_occurrences)
    public_diagnostics = occurrence_diagnostics(public_occurrences, public_cap)
    return ConditionResult(
        tau=tau,
        train_features=train_features,
        public_features=public_features,
        train_diagnostics=train_diagnostics,
        public_diagnostics=public_diagnostics,
        train_occurrences=train_occurrences if retain_occurrences else None,
        public_occurrences=public_occurrences if retain_occurrences else None,
    )


def _fit_predict_probe(x_train, labels, x_public, *, condition: str) -> tuple[np.ndarray, dict]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe = fit_probe(x_train, labels, spec=PROBE_SPEC)
    classifier = probe.named_steps["logisticregression"]
    prediction = np.asarray(probe.predict(x_public), dtype=np.int8)
    return prediction, {
        "condition": condition,
        "n_iter": np.asarray(classifier.n_iter_, dtype=np.int64).tolist(),
        "converged": bool(np.all(np.asarray(classifier.n_iter_) < PROBE_SPEC.max_iter)),
        "warnings": [f"{type(item.message).__name__}: {item.message}" for item in caught],
    }


def _metric_record(labels, prediction) -> dict[str, float]:
    accuracy, macro_f1 = metrics(labels, prediction)
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def _comparison(
    labels: np.ndarray,
    actual_prediction: np.ndarray,
    control_predictions: np.ndarray,
    control_names: list[str],
    *,
    stage: str,
) -> tuple[dict, dict[str, np.ndarray | float]]:
    actual = _metric_record(labels, actual_prediction)
    controls = {
        name: _metric_record(labels, prediction)
        for name, prediction in zip(control_names, control_predictions)
    }
    accuracy = np.asarray([value["accuracy"] for value in controls.values()])
    macro_f1 = np.asarray([value["macro_f1"] for value in controls.values()])
    bootstrap = paired_bootstrap_vs_control_mean(
        labels,
        actual_prediction,
        control_predictions,
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    da = np.asarray(bootstrap["delta_accuracy"])
    df = np.asarray(bootstrap["delta_macro_f1"])
    verdict = registered_joint_verdict(
        np.asarray(bootstrap["accuracy_ci95"]),
        np.asarray(bootstrap["macro_f1_ci95"]),
        stage=stage,
    )
    report = {
        "actual": actual,
        "controls": controls,
        "control_mean": {
            "accuracy": float(accuracy.mean()),
            "macro_f1": float(macro_f1.mean()),
        },
        "control_sample_sd": {
            "accuracy": float(accuracy.std(ddof=1)),
            "macro_f1": float(macro_f1.std(ddof=1)),
        },
        "actual_minus_control_mean": {
            "accuracy": float(actual["accuracy"] - accuracy.mean()),
            "macro_f1": float(actual["macro_f1"] - macro_f1.mean()),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "accuracy": {
                "quantiles_2.5_50_97.5": np.percentile(da, [2.5, 50, 97.5]).tolist(),
                "fraction_le_zero": float(np.mean(da <= 0)),
            },
            "macro_f1": {
                "quantiles_2.5_50_97.5": np.percentile(df, [2.5, 50, 97.5]).tolist(),
                "fraction_le_zero": float(np.mean(df <= 0)),
            },
        },
        "registered_verdict": verdict,
    }
    return report, bootstrap


def _serializable_diagnostics(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "node_counts"}


def _flatten_occurrences(all_occurrences: list[CompactOccurrences]) -> dict[str, np.ndarray]:
    offsets = np.zeros(len(all_occurrences) + 1, dtype=np.int64)
    for index, occurrences in enumerate(all_occurrences):
        offsets[index + 1] = offsets[index] + len(occurrences)
    return {
        "offsets": offsets,
        "components": np.concatenate([x.components for x in all_occurrences]),
        "confidence": np.concatenate([x.confidence for x in all_occurrences]),
        "y": np.concatenate([x.y for x in all_occurrences]),
        "x": np.concatenate([x.x for x in all_occurrences]),
    }


def _write_manifest(output_dir: Path) -> Path:
    manifest = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "e0r_artifact_manifest.json"
    }
    path = output_dir / "e0r_artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def run_e0r(
    train_csv: str | Path,
    public_csv: str | Path,
    actual_dictionary: str | Path,
    control_paths: dict[int, str | Path],
    e02_summary_path: str | Path,
    e02_results_path: str | Path,
    output_dir: str | Path,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if sha256_file(e02_summary_path) != E02_SUMMARY_SHA256:
        raise ValueError("frozen E0.2 summary SHA mismatch")
    if sha256_file(e02_results_path) != E02_RESULTS_SHA256:
        raise ValueError("frozen E0.2 results SHA mismatch")
    conditions = load_frozen_conditions(actual_dictionary, control_paths)
    _log("all six frozen v535 representations loaded without PCA/GMM refit")

    train = load_pixels_only(train_csv, role="train")
    public = load_pixels_only(public_csv, role="public")
    condition_results: dict[str, ConditionResult] = {}
    for condition in conditions:
        _log(f"extracting R1 occurrences for {condition.name}")
        condition_results[condition.name] = run_occurrence_condition(
            train,
            public,
            condition,
            retain_occurrences=condition.name == "actual",
        )

    actual_result = condition_results["actual"]
    assert actual_result.train_occurrences is not None
    assert actual_result.public_occurrences is not None
    train_flat = _flatten_occurrences(actual_result.train_occurrences)
    public_flat = _flatten_occurrences(actual_result.public_occurrences)
    occurrences_path = out / "e0r_occurrences_actual.npz"
    np.savez_compressed(
        occurrences_path,
        experiment=np.asarray("PGM_E0.R"),
        issue=np.asarray(ISSUE_NUMBER, np.int32),
        train_ids=train.canonical_ids,
        public_ids=public.canonical_ids,
        tau_actual=np.asarray(actual_result.tau, np.float64),
        train_offsets=train_flat["offsets"],
        train_components=train_flat["components"],
        train_confidence=train_flat["confidence"],
        train_y=train_flat["y"],
        train_x=train_flat["x"],
        public_offsets=public_flat["offsets"],
        public_components=public_flat["components"],
        public_confidence=public_flat["confidence"],
        public_y=public_flat["y"],
        public_x=public_flat["x"],
    )

    condition_names = [condition.name for condition in conditions]
    diagnostics_path = out / "e0r_r1_occurrence_diagnostics.npz"
    np.savez_compressed(
        diagnostics_path,
        condition_names=np.asarray(condition_names),
        taus=np.asarray([condition_results[name].tau for name in condition_names]),
        train_node_counts=np.stack(
            [condition_results[name].train_diagnostics["node_counts"] for name in condition_names]
        ),
        public_node_counts=np.stack(
            [condition_results[name].public_diagnostics["node_counts"] for name in condition_names]
        ),
        train_component_prevalence=np.stack(
            [condition_results[name].train_diagnostics["component_prevalence"] for name in condition_names]
        ),
        public_component_prevalence=np.stack(
            [condition_results[name].public_diagnostics["component_prevalence"] for name in condition_names]
        ),
        train_component_support_rate=np.stack(
            [condition_results[name].train_diagnostics["component_image_support_rate"] for name in condition_names]
        ),
        public_component_support_rate=np.stack(
            [condition_results[name].public_diagnostics["component_image_support_rate"] for name in condition_names]
        ),
    )

    # This is intentionally the first label access, after all six occurrence representations freeze.
    _log("R1 unsupervised occurrence features frozen; reading Train/Public labels downstream")
    train_labels = load_labels_downstream(train_csv, role="train", expected_sha256=train.sha256)
    public_labels = load_labels_downstream(public_csv, role="public", expected_sha256=public.sha256)
    r1_predictions: dict[str, np.ndarray] = {}
    r1_probe_diagnostics: dict[str, dict] = {}
    for name in condition_names:
        _log(f"fitting fixed R1 probe for {name}")
        result = condition_results[name]
        r1_predictions[name], r1_probe_diagnostics[name] = _fit_predict_probe(
            result.train_features,
            train_labels,
            result.public_features,
            condition=name,
        )
    r1_controls = np.stack([r1_predictions[f"control_{seed}"] for seed in CONTROL_SEEDS])
    r1_report, r1_bootstrap = _comparison(
        public_labels,
        r1_predictions["actual"],
        r1_controls,
        [str(seed) for seed in CONTROL_SEEDS],
        stage="R1",
    )
    r1_results_path = out / "e0r_r1_results.npz"
    np.savez_compressed(
        r1_results_path,
        train_ids=train.canonical_ids,
        public_ids=public.canonical_ids,
        train_labels=train_labels,
        public_labels=public_labels,
        condition_names=np.asarray(condition_names),
        taus=np.asarray([condition_results[name].tau for name in condition_names]),
        actual_public_predictions=r1_predictions["actual"],
        control_public_predictions=r1_controls,
        bootstrap_delta_accuracy=np.asarray(r1_bootstrap["delta_accuracy"]),
        bootstrap_delta_macro_f1=np.asarray(r1_bootstrap["delta_macro_f1"]),
        registered_verdict=np.asarray(r1_report["registered_verdict"]),
    )

    actual_train_diag = actual_result.train_diagnostics
    actual_public_diag = actual_result.public_diagnostics
    recurrence = {
        "component_prevalence_spearman": finite_spearman(
            np.asarray(actual_train_diag["component_prevalence"]),
            np.asarray(actual_public_diag["component_prevalence"]),
        ),
        "component_support_rate_spearman": finite_spearman(
            np.asarray(actual_train_diag["component_image_support_rate"]),
            np.asarray(actual_public_diag["component_image_support_rate"]),
        ),
    }

    r2_summary: dict[str, object]
    r2_artifact: str | None = None
    if r1_report["registered_verdict"] != "E0.R1 SUPPORTED":
        r2_summary = {
            "status": "NOT_RUN_R1_GATE",
            "report_text": "E0.R2 NOT RUN — R1 REGISTERED GATE NOT MET",
            "probes_executed": 0,
            "m0_gate": "BLOCKED",
        }
        (out / "e0r_r2_status.json").write_text(
            json.dumps(r2_summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
    else:
        distance_median = train_pair_distance_median(actual_result.train_occurrences)
        if distance_median is None:
            r2_summary = {
                "status": "E0.R2 NOT SUPPORTED — INSUFFICIENT OCCURRENCE PAIRS",
                "probes_executed": 0,
                "m0_gate": "BLOCKED",
            }
            (out / "e0r_r2_status.json").write_text(
                json.dumps(r2_summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
        else:
            _log(f"R1 gate passed; frozen Train pair-distance median={distance_median:.12g}")
            train_geometry = build_geometry_csr(
                actual_result.train_occurrences,
                train.canonical_ids,
                distance_median=distance_median,
            )
            public_geometry = build_geometry_csr(
                actual_result.public_occurrences,
                public.canonical_ids,
                distance_median=distance_median,
            )
            actual_train_nested = nested_csr(actual_result.train_features, train_geometry)
            actual_public_nested = nested_csr(actual_result.public_features, public_geometry)
            actual_nested_hashes = {
                "train": sparse_sha256(actual_train_nested),
                "public": sparse_sha256(actual_public_nested),
            }
            _log("fitting fixed R2 actual [O||G] probe")
            r2_actual_prediction, r2_actual_probe = _fit_predict_probe(
                actual_train_nested,
                train_labels,
                actual_public_nested,
                condition="actual_O_G",
            )
            r2_control_predictions: list[np.ndarray] = []
            r2_probe_diagnostics: dict[str, dict] = {"actual": r2_actual_probe}
            r2_sparse_hashes: dict[str, dict[str, str]] = {"actual": actual_nested_hashes}
            unary_hashes = {
                "train": sha256_array(actual_result.train_features),
                "public": sha256_array(actual_result.public_features),
            }
            for seed in GEOMETRY_SEEDS:
                _log(f"fitting fixed R2 geometry-shuffle probe seed={seed}")
                train_shuffled = build_geometry_csr(
                    actual_result.train_occurrences,
                    train.canonical_ids,
                    distance_median=distance_median,
                    shuffle_seed=seed,
                )
                public_shuffled = build_geometry_csr(
                    actual_result.public_occurrences,
                    public.canonical_ids,
                    distance_median=distance_median,
                    shuffle_seed=seed,
                )
                train_nested = nested_csr(actual_result.train_features, train_shuffled)
                public_nested = nested_csr(actual_result.public_features, public_shuffled)
                prediction, diagnostic = _fit_predict_probe(
                    train_nested,
                    train_labels,
                    public_nested,
                    condition=f"geometry_shuffle_{seed}",
                )
                r2_control_predictions.append(prediction)
                r2_probe_diagnostics[str(seed)] = diagnostic
                r2_sparse_hashes[str(seed)] = {
                    "train": sparse_sha256(train_nested),
                    "public": sparse_sha256(public_nested),
                }
                del train_shuffled, public_shuffled, train_nested, public_nested
            r2_controls = np.stack(r2_control_predictions)
            r2_report, r2_bootstrap = _comparison(
                public_labels,
                r2_actual_prediction,
                r2_controls,
                [str(seed) for seed in GEOMETRY_SEEDS],
                stage="R2",
            )
            r2_results_path = out / "e0r_r2_results.npz"
            np.savez_compressed(
                r2_results_path,
                public_ids=public.canonical_ids,
                public_labels=public_labels,
                distance_median=np.asarray(distance_median),
                geometry_seeds=np.asarray(GEOMETRY_SEEDS, np.int32),
                actual_public_predictions=r2_actual_prediction,
                control_public_predictions=r2_controls,
                bootstrap_delta_accuracy=np.asarray(r2_bootstrap["delta_accuracy"]),
                bootstrap_delta_macro_f1=np.asarray(r2_bootstrap["delta_macro_f1"]),
                registered_verdict=np.asarray(r2_report["registered_verdict"]),
            )
            r2_artifact = r2_results_path.name
            r2_summary = {
                "status": "COMPLETE",
                "distance_median_train_actual": distance_median,
                "geometry_seeds": list(GEOMETRY_SEEDS),
                "unary_feature_sha256_identical_all_arms": unary_hashes,
                "nested_sparse_sha256": r2_sparse_hashes,
                "probe_diagnostics": r2_probe_diagnostics,
                "comparison": r2_report,
                "results_artifact": r2_artifact,
                "m0_gate": (
                    "M0 PREREGISTRATION UNLOCKED"
                    if r2_report["registered_verdict"] == "E0.R2 SUPPORTED"
                    else "BLOCKED"
                ),
            }

    summary = {
        "experiment": "PGM_E0.R_relational_component_occurrence_and_nested_spatial_qualification",
        "issue": ISSUE_NUMBER,
        "base_sha": BASE_SHA,
        "history": {
            "e01": "E0.1 NEGATIVE — MOTIF EXISTENCE NOT SUPPORTED",
            "e01b": "post-hoc and separate",
            "e02": "E0.2 INDEPENDENT REVIEW — STRONG POSITIVE EVIDENCE FOR LOCAL RELATIONAL NECESSITY.",
            "e0r_first_automatic_joint_ci_threshold": True,
        },
        "private_test_read": False,
        "public_status": "developmental replication evidence only",
        "frozen_inputs": {
            "train_sha256": train.sha256,
            "public_sha256": public.sha256,
            "matched_pool_sha256": MATCHED_POOL_SHA256,
            "e02_summary_sha256": E02_SUMMARY_SHA256,
            "e02_results_sha256": E02_RESULTS_SHA256,
            "condition_artifact_sha256": {
                condition.name: condition.artifact_sha256 for condition in conditions
            },
            "pca_or_gmm_refit": False,
        },
        "r1": {
            "taus": {name: condition_results[name].tau for name in condition_names},
            "diagnostics": {
                name: {
                    "train": _serializable_diagnostics(condition_results[name].train_diagnostics),
                    "public": _serializable_diagnostics(condition_results[name].public_diagnostics),
                }
                for name in condition_names
            },
            "actual_train_public_recurrence": recurrence,
            "feature_sha256": {
                name: {
                    "train": sha256_array(condition_results[name].train_features),
                    "public": sha256_array(condition_results[name].public_features),
                }
                for name in condition_names
            },
            "probe_diagnostics": r1_probe_diagnostics,
            "comparison": r1_report,
            "results_artifact": r1_results_path.name,
            "occurrences_artifact": occurrences_path.name,
            "diagnostics_artifact": diagnostics_path.name,
            "failure_interpretation_boundary": "The full frozen v533 K=128 vocabulary under this preregistered sparse extraction rule did not meet the registered joint criterion.",
        },
        "r2": r2_summary,
    }
    summary_path = out / "e0r_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    manifest_path = _write_manifest(out)
    _log(f"E0.R complete: R1={r1_report['registered_verdict']} R2={r2_summary['status']}")
    _log(f"wrote {summary_path.name} and {manifest_path.name}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preregistered PGM E0.R sequential runner")
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--public-csv", required=True, type=Path)
    parser.add_argument("--actual-dictionary", required=True, type=Path)
    parser.add_argument("--e02-summary", required=True, type=Path)
    parser.add_argument("--e02-results", required=True, type=Path)
    parser.add_argument("--control-42", required=True, type=Path)
    parser.add_argument("--control-43", required=True, type=Path)
    parser.add_argument("--control-44", required=True, type=Path)
    parser.add_argument("--control-45", required=True, type=Path)
    parser.add_argument("--control-46", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    run_e0r(
        args.train_csv,
        args.public_csv,
        args.actual_dictionary,
        {seed: getattr(args, f"control_{seed}") for seed in CONTROL_SEEDS},
        args.e02_summary,
        args.e02_results,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
