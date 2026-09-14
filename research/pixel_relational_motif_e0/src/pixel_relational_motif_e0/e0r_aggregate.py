"""Zero-fit aggregation for completed checkpointed PGM E0.R R2 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .e02_runner import sha256_file
from .e0r_continuation import (
    GEOMETRY_SEEDS,
    ISSUE_NUMBER,
    PUBLIC_IDS_SHA256,
    R1_DIAGNOSTICS_SHA256,
    R1_OCCURRENCES_SHA256,
    R1_RESULTS_SHA256,
    SCIENTIFIC_SHA,
    _scalar_text,
    load_frozen_r1,
    validate_checkpoint,
)
from .e0r_runner import _comparison


def aggregate_r2(
    occurrences_path: str | Path,
    diagnostics_path: str | Path,
    results_path: str | Path,
    actual_path: str | Path,
    seed_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    execution_wrapper_sha: str,
) -> dict:
    if sha256_file(diagnostics_path) != R1_DIAGNOSTICS_SHA256:
        raise ValueError("canonical R1 diagnostics SHA mismatch")
    substrate = load_frozen_r1(occurrences_path, results_path)
    if len(seed_paths) != len(GEOMETRY_SEEDS):
        raise ValueError("aggregation requires exactly 20 control artifacts")
    actual_record = validate_checkpoint(
        actual_path,
        seed=None,
        execution_wrapper_sha=execution_wrapper_sha,
        substrate=substrate,
    )
    seed_artifacts: dict[int, Path] = {}
    for raw_path in seed_paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as artifact:
            seed = int(artifact["seed"])
        if seed in seed_artifacts:
            raise ValueError(f"duplicate control seed {seed}")
        seed_artifacts[seed] = path
    if set(seed_artifacts) != set(GEOMETRY_SEEDS):
        raise ValueError("control seed set must be exactly 42..61")
    control_records = []
    control_predictions = []
    nested_train_hashes = {"actual": actual_record["nested_train_sparse_sha256"]}
    nested_public_hashes = {"actual": actual_record["nested_public_sparse_sha256"]}
    with np.load(actual_path, allow_pickle=False) as actual_artifact:
        actual_prediction = np.asarray(actual_artifact["public_predictions"]).copy()
        common_distance = float(actual_artifact["distance_median"])
        common_o_train = _scalar_text(actual_artifact["o_train_sha256"])
        common_o_public = _scalar_text(actual_artifact["o_public_sha256"])
    for seed in GEOMETRY_SEEDS:
        path = seed_artifacts[seed]
        record = validate_checkpoint(
            path,
            seed=seed,
            execution_wrapper_sha=execution_wrapper_sha,
            substrate=substrate,
        )
        with np.load(path, allow_pickle=False) as artifact:
            if float(artifact["distance_median"]) != common_distance:
                raise ValueError("distance median differs across checkpoints")
            if _scalar_text(artifact["o_train_sha256"]) != common_o_train or _scalar_text(
                artifact["o_public_sha256"]
            ) != common_o_public:
                raise ValueError("O hashes differ across checkpoints")
            control_predictions.append(np.asarray(artifact["public_predictions"]).copy())
        control_records.append(record)
        nested_train_hashes[str(seed)] = record["nested_train_sparse_sha256"]
        nested_public_hashes[str(seed)] = record["nested_public_sparse_sha256"]
    controls = np.stack(control_predictions)
    comparison, bootstrap = _comparison(
        substrate.public_labels,
        actual_prediction,
        controls,
        [str(seed) for seed in GEOMETRY_SEEDS],
        stage="R2",
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_out = out / "e0r_r2_results.npz"
    with results_out.open("wb") as stream:
        np.savez_compressed(
            stream,
            experiment=np.asarray("PGM_E0.R_R2_AGGREGATE"),
            issue=np.asarray(ISSUE_NUMBER, dtype=np.int32),
            scientific_sha=np.asarray(SCIENTIFIC_SHA),
            execution_wrapper_sha=np.asarray(execution_wrapper_sha),
            r1_occurrences_sha256=np.asarray(R1_OCCURRENCES_SHA256),
            r1_results_sha256=np.asarray(R1_RESULTS_SHA256),
            public_ids_sha256=np.asarray(PUBLIC_IDS_SHA256),
            public_ids=substrate.public_ids,
            public_labels=substrate.public_labels,
            distance_median=np.asarray(common_distance, dtype=np.float64),
            o_train_sha256=np.asarray(common_o_train),
            o_public_sha256=np.asarray(common_o_public),
            geometry_seeds=np.asarray(GEOMETRY_SEEDS, dtype=np.int32),
            actual_public_predictions=actual_prediction,
            control_public_predictions=controls,
            bootstrap_delta_accuracy=np.asarray(bootstrap["delta_accuracy"]),
            bootstrap_delta_macro_f1=np.asarray(bootstrap["delta_macro_f1"]),
            registered_verdict=np.asarray(comparison["registered_verdict"]),
        )
    aggregate_summary = {
        "experiment": "PGM_E0.R_R2_AGGREGATE",
        "issue": ISSUE_NUMBER,
        "scientific_sha": SCIENTIFIC_SHA,
        "execution_wrapper_sha": execution_wrapper_sha,
        "r1_occurrences_sha256": R1_OCCURRENCES_SHA256,
        "r1_diagnostics_sha256": R1_DIAGNOSTICS_SHA256,
        "r1_results_sha256": R1_RESULTS_SHA256,
        "distance_median": common_distance,
        "o_train_sha256": common_o_train,
        "o_public_sha256": common_o_public,
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "actual_checkpoint": actual_record,
        "control_checkpoints": control_records,
        "nested_train_sparse_sha256": nested_train_hashes,
        "nested_public_sparse_sha256": nested_public_hashes,
        "comparison": comparison,
        "private_test_read": False,
        "model_fits_executed": 0,
        "m0_gate": "M0 PREREGISTRATION UNLOCKED" if comparison["registered_verdict"] == "E0.R2 SUPPORTED" else "M0 BLOCKED",
    }
    aggregate_summary_path = out / "e0r_r2_aggregate_summary.json"
    aggregate_summary_path.write_text(
        json.dumps(aggregate_summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    with np.load(results_path, allow_pickle=False) as r1:
        r1_controls = np.asarray(r1["control_public_predictions"])
        r1_comparison, _ = _comparison(
            np.asarray(r1["public_labels"]),
            np.asarray(r1["actual_public_predictions"]),
            r1_controls,
            [str(seed) for seed in range(42, 47)],
            stage="R1",
        )
        r1_taus = np.asarray(r1["taus"]).tolist()
    canonical_summary = {
        "experiment": "PGM_E0.R_relational_component_occurrence_and_nested_spatial_qualification",
        "issue": ISSUE_NUMBER,
        "scientific_sha": SCIENTIFIC_SHA,
        "execution_wrapper_sha": execution_wrapper_sha,
        "v538_classification": "TECHNICAL TIMEOUT DURING REGISTERED R2 EXECUTION; R2 SCIENTIFIC VERDICT UNAVAILABLE",
        "private_test_read": False,
        "public_status": "developmental replication evidence only",
        "r1": {
            "canonical_occurrences_sha256": R1_OCCURRENCES_SHA256,
            "canonical_diagnostics_sha256": R1_DIAGNOSTICS_SHA256,
            "canonical_results_sha256": R1_RESULTS_SHA256,
            "condition_names": ["actual", "control_42", "control_43", "control_44", "control_45", "control_46"],
            "taus": r1_taus,
            "comparison": r1_comparison,
        },
        "r2": aggregate_summary,
        "m0_gate": aggregate_summary["m0_gate"],
    }
    summary_path = out / "e0r_summary.json"
    summary_path.write_text(json.dumps(canonical_summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    manifest = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "e0r_r2_aggregate_manifest.json"
    }
    manifest_path = out / "e0r_r2_aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return aggregate_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-fit PGM E0.R R2 checkpoint aggregator")
    parser.add_argument("--r1-occurrences", required=True, type=Path)
    parser.add_argument("--r1-diagnostics", required=True, type=Path)
    parser.add_argument("--r1-results", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-wrapper-sha", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    aggregate_r2(
        args.r1_occurrences,
        args.r1_diagnostics,
        args.r1_results,
        args.actual,
        args.seeds,
        args.output_dir,
        execution_wrapper_sha=args.execution_wrapper_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
