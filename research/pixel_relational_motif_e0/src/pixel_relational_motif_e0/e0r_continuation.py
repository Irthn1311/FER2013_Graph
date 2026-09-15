"""Execution-only, checkpointed continuation for the frozen PGM E0.R R2 protocol.

Scientific definitions are imported from the original E0.R modules.  This file
only changes scheduling and persistence after Kaggle v538 timed out.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from .e02_runner import PUBLIC_ROWS, TRAIN_ROWS, PUBLIC_SHA256, TRAIN_SHA256, sha256_array, sha256_file
from .e0r_geometry import GEOMETRY_SEEDS, build_geometry_csr, nested_csr, sparse_sha256, train_pair_distance_median
from .e0r_occurrence import CompactOccurrences, K, occurrence_histograms
from .e0r_runner import ISSUE_NUMBER, _fit_predict_probe, _metric_record


SCIENTIFIC_SHA = "671e3c2f69607778f08a923026e76744561e13b2"
R1_OCCURRENCES_SHA256 = "30f1a5b642af2ecdfc29ae73960fb01f90c391964db845bd4b33cf3c017f7fa9"
R1_DIAGNOSTICS_SHA256 = "e1b3d6c71ef39b96b8431ed6c9bdf0039bccc47a63cc71e98c82790472adb99f"
R1_RESULTS_SHA256 = "b6675ce694f5a607cfea07abb3ed1065753f42ee848f7595d1cb8e682361a0ed"
TRAIN_IDS_SHA256 = "5020e75f45ec0b8ea45fcb867c566367a232bb7245838a121f22b602c51e3be7"
PUBLIC_IDS_SHA256 = "3b82a8c2ab1c92c8469f3741bb2fa8cdf4003a4ba7bb2cf5ce1c2608565c6df7"
CONDITION_NAMES = ("actual", "control_42", "control_43", "control_44", "control_45", "control_46")
SHARD_PLAN = {
    "A": {"actual": True, "seeds": (42, 43, 44, 45)},
    "B": {"actual": False, "seeds": (46, 47, 48, 49, 50)},
    "C": {"actual": False, "seeds": (51, 52, 53, 54, 55)},
    "D": {"actual": False, "seeds": (56, 57, 58, 59, 60, 61)},
}


@dataclass(frozen=True)
class FrozenR1Substrate:
    train_occurrences: list[CompactOccurrences]
    public_occurrences: list[CompactOccurrences]
    train_ids: np.ndarray
    public_ids: np.ndarray
    train_labels: np.ndarray
    public_labels: np.ndarray
    train_o: np.ndarray
    public_o: np.ndarray
    distance_median: float
    o_train_sha256: str
    o_public_sha256: str


def _log(message: str) -> None:
    print(f"[PGM-E0.R-CONTINUATION] {message}", flush=True)


def _scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("expected scalar text field")
    return str(array.item())


def _expected_ids(role: str) -> np.ndarray:
    if role == "train":
        return np.arange(TRAIN_ROWS, dtype=np.int32)
    if role == "public":
        return np.arange(TRAIN_ROWS, TRAIN_ROWS + PUBLIC_ROWS, dtype=np.int32)
    raise ValueError("role must be train or public")


def _unflatten_occurrences(artifact: np.lib.npyio.NpzFile, prefix: str) -> list[CompactOccurrences]:
    expected_rows = TRAIN_ROWS if prefix == "train" else PUBLIC_ROWS
    offsets = np.asarray(artifact[f"{prefix}_offsets"])
    if offsets.dtype != np.int64 or offsets.shape != (expected_rows + 1,):
        raise ValueError(f"invalid {prefix} occurrence offsets")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(f"invalid {prefix} occurrence offset order")
    total = int(offsets[-1])
    components = np.asarray(artifact[f"{prefix}_components"])
    confidence = np.asarray(artifact[f"{prefix}_confidence"])
    y = np.asarray(artifact[f"{prefix}_y"])
    x = np.asarray(artifact[f"{prefix}_x"])
    if components.dtype != np.int16 or confidence.dtype != np.float64 or y.dtype != np.int16 or x.dtype != np.int16:
        raise ValueError(f"invalid {prefix} occurrence dtypes")
    if any(value.shape != (total,) for value in (components, confidence, y, x)):
        raise ValueError(f"invalid {prefix} occurrence array length")
    if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError(f"invalid {prefix} occurrence confidence")
    if np.any((components < 0) | (components >= K)) or np.any((x < 0) | (x >= 48)) or np.any((y < 0) | (y >= 48)):
        raise ValueError(f"invalid {prefix} occurrence identity or coordinates")
    return [
        CompactOccurrences(
            components[int(offsets[i]) : int(offsets[i + 1])].copy(),
            confidence[int(offsets[i]) : int(offsets[i + 1])].copy(),
            y[int(offsets[i]) : int(offsets[i + 1])].copy(),
            x[int(offsets[i]) : int(offsets[i + 1])].copy(),
        )
        for i in range(expected_rows)
    ]


def load_frozen_r1(
    occurrences_path: str | Path,
    results_path: str | Path,
) -> FrozenR1Substrate:
    occurrences_path = Path(occurrences_path)
    results_path = Path(results_path)
    if sha256_file(occurrences_path) != R1_OCCURRENCES_SHA256:
        raise ValueError("canonical R1 occurrence artifact SHA mismatch")
    if sha256_file(results_path) != R1_RESULTS_SHA256:
        raise ValueError("canonical R1 results artifact SHA mismatch")
    with np.load(occurrences_path, allow_pickle=False) as occurrence_artifact, np.load(
        results_path, allow_pickle=False
    ) as result_artifact:
        if _scalar_text(occurrence_artifact["experiment"]) != "PGM_E0.R" or int(occurrence_artifact["issue"]) != ISSUE_NUMBER:
            raise ValueError("R1 occurrence experiment provenance mismatch")
        train_ids = np.asarray(result_artifact["train_ids"])
        public_ids = np.asarray(result_artifact["public_ids"])
        if train_ids.dtype != np.int32 or not np.array_equal(train_ids, _expected_ids("train")):
            raise ValueError("canonical Train IDs mismatch")
        if public_ids.dtype != np.int32 or not np.array_equal(public_ids, _expected_ids("public")):
            raise ValueError("canonical Public IDs mismatch")
        if sha256_array(train_ids) != TRAIN_IDS_SHA256 or sha256_array(public_ids) != PUBLIC_IDS_SHA256:
            raise ValueError("canonical ID hash mismatch")
        if not np.array_equal(occurrence_artifact["train_ids"], train_ids) or not np.array_equal(
            occurrence_artifact["public_ids"], public_ids
        ):
            raise ValueError("occurrence/results canonical IDs differ")
        names = tuple(str(value) for value in result_artifact["condition_names"].tolist())
        if names != CONDITION_NAMES or _scalar_text(result_artifact["registered_verdict"]) != "E0.R1 SUPPORTED":
            raise ValueError("canonical R1 gate evidence mismatch")
        taus = np.asarray(result_artifact["taus"])
        if taus.dtype != np.float64 or taus.shape != (6,) or not np.all(np.isfinite(taus)):
            raise ValueError("invalid frozen R1 thresholds")
        if float(occurrence_artifact["tau_actual"]) != float(taus[0]):
            raise ValueError("actual threshold differs across R1 artifacts")
        train_labels = np.asarray(result_artifact["train_labels"])
        public_labels = np.asarray(result_artifact["public_labels"])
        if train_labels.dtype != np.int8 or train_labels.shape != (TRAIN_ROWS,) or np.any((train_labels < 0) | (train_labels > 6)):
            raise ValueError("invalid frozen Train labels")
        if public_labels.dtype != np.int8 or public_labels.shape != (PUBLIC_ROWS,) or np.any((public_labels < 0) | (public_labels > 6)):
            raise ValueError("invalid frozen Public labels")
        train_occurrences = _unflatten_occurrences(occurrence_artifact, "train")
        public_occurrences = _unflatten_occurrences(occurrence_artifact, "public")
    train_o = occurrence_histograms(train_occurrences)
    public_o = occurrence_histograms(public_occurrences)
    distance_median = train_pair_distance_median(train_occurrences)
    if distance_median is None or not np.isfinite(distance_median):
        raise ValueError("canonical R1 occurrences have no finite Train pair-distance median")
    return FrozenR1Substrate(
        train_occurrences=train_occurrences,
        public_occurrences=public_occurrences,
        train_ids=train_ids.copy(),
        public_ids=public_ids.copy(),
        train_labels=train_labels.copy(),
        public_labels=public_labels.copy(),
        train_o=train_o,
        public_o=public_o,
        distance_median=float(distance_median),
        o_train_sha256=sha256_array(train_o),
        o_public_sha256=sha256_array(public_o),
    )


def build_nested_arm(
    substrate: FrozenR1Substrate,
    *,
    seed: int | None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if seed is not None and seed not in GEOMETRY_SEEDS:
        raise ValueError("unregistered R2 geometry seed")
    train_geometry = build_geometry_csr(
        substrate.train_occurrences,
        substrate.train_ids,
        distance_median=substrate.distance_median,
        shuffle_seed=seed,
    )
    public_geometry = build_geometry_csr(
        substrate.public_occurrences,
        substrate.public_ids,
        distance_median=substrate.distance_median,
        shuffle_seed=seed,
    )
    return nested_csr(substrate.train_o, train_geometry), nested_csr(substrate.public_o, public_geometry)


def fit_nested_arm(
    train_nested: sparse.csr_matrix,
    train_labels: np.ndarray,
    public_nested: sparse.csr_matrix,
    public_labels: np.ndarray,
    *,
    condition: str,
) -> tuple[np.ndarray, dict, dict[str, float]]:
    prediction, diagnostic = _fit_predict_probe(
        train_nested, train_labels, public_nested, condition=condition
    )
    return prediction, diagnostic, _metric_record(public_labels, prediction)


def _checkpoint_name(seed: int | None) -> str:
    return "e0r_r2_actual.npz" if seed is None else f"e0r_r2_seed_{seed}.npz"


def _checkpoint_payload(
    substrate: FrozenR1Substrate,
    *,
    seed: int | None,
    execution_wrapper_sha: str,
    train_nested: sparse.csr_matrix,
    public_nested: sparse.csr_matrix,
    prediction: np.ndarray,
    diagnostic: dict,
    metric: dict[str, float],
) -> dict[str, np.ndarray]:
    if len(execution_wrapper_sha) != 40:
        raise ValueError("execution wrapper SHA must be a full 40-character commit")
    n_iter = np.asarray(diagnostic["n_iter"], dtype=np.int64)
    warnings_json = json.dumps(diagnostic["warnings"], ensure_ascii=False, allow_nan=False)
    return {
        "experiment": np.asarray("PGM_E0.R_R2_CHECKPOINT"),
        "issue": np.asarray(ISSUE_NUMBER, dtype=np.int32),
        "scientific_sha": np.asarray(SCIENTIFIC_SHA),
        "execution_wrapper_sha": np.asarray(execution_wrapper_sha),
        "fit_kind": np.asarray("actual" if seed is None else "geometry_shuffle"),
        "seed": np.asarray(-1 if seed is None else seed, dtype=np.int32),
        "r1_occurrences_sha256": np.asarray(R1_OCCURRENCES_SHA256),
        "r1_results_sha256": np.asarray(R1_RESULTS_SHA256),
        "train_csv_sha256": np.asarray(TRAIN_SHA256),
        "public_csv_sha256": np.asarray(PUBLIC_SHA256),
        "train_ids_sha256": np.asarray(TRAIN_IDS_SHA256),
        "public_ids_sha256": np.asarray(PUBLIC_IDS_SHA256),
        "distance_median": np.asarray(substrate.distance_median, dtype=np.float64),
        "o_train_sha256": np.asarray(substrate.o_train_sha256),
        "o_public_sha256": np.asarray(substrate.o_public_sha256),
        "nested_train_sparse_sha256": np.asarray(sparse_sha256(train_nested)),
        "nested_public_sparse_sha256": np.asarray(sparse_sha256(public_nested)),
        "public_ids": substrate.public_ids,
        "public_predictions": np.asarray(prediction, dtype=np.int8),
        "n_iter": n_iter,
        "converged": np.asarray(bool(diagnostic["converged"])),
        "warnings_json": np.asarray(warnings_json),
        "public_accuracy": np.asarray(metric["accuracy"], dtype=np.float64),
        "public_macro_f1": np.asarray(metric["macro_f1"], dtype=np.float64),
    }


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    for value in payload.values():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"nonfinite checkpoint field for {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_checkpoint(
    path: str | Path,
    *,
    seed: int | None,
    execution_wrapper_sha: str,
    substrate: FrozenR1Substrate,
) -> dict[str, object]:
    path = Path(path)
    expected_keys = {
        "experiment", "issue", "scientific_sha", "execution_wrapper_sha", "fit_kind", "seed",
        "r1_occurrences_sha256", "r1_results_sha256", "train_csv_sha256", "public_csv_sha256",
        "train_ids_sha256", "public_ids_sha256", "distance_median", "o_train_sha256", "o_public_sha256",
        "nested_train_sparse_sha256", "nested_public_sparse_sha256", "public_ids", "public_predictions",
        "n_iter", "converged", "warnings_json", "public_accuracy", "public_macro_f1",
    }
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != expected_keys:
            raise ValueError("checkpoint key set mismatch")
        scalar_expected = {
            "experiment": "PGM_E0.R_R2_CHECKPOINT",
            "scientific_sha": SCIENTIFIC_SHA,
            "execution_wrapper_sha": execution_wrapper_sha,
            "fit_kind": "actual" if seed is None else "geometry_shuffle",
            "r1_occurrences_sha256": R1_OCCURRENCES_SHA256,
            "r1_results_sha256": R1_RESULTS_SHA256,
            "train_csv_sha256": TRAIN_SHA256,
            "public_csv_sha256": PUBLIC_SHA256,
            "train_ids_sha256": TRAIN_IDS_SHA256,
            "public_ids_sha256": PUBLIC_IDS_SHA256,
            "o_train_sha256": substrate.o_train_sha256,
            "o_public_sha256": substrate.o_public_sha256,
        }
        for key, expected in scalar_expected.items():
            if _scalar_text(artifact[key]) != expected:
                raise ValueError(f"checkpoint {key} mismatch")
        if int(artifact["issue"]) != ISSUE_NUMBER or int(artifact["seed"]) != (-1 if seed is None else seed):
            raise ValueError("checkpoint issue/seed mismatch")
        if float(artifact["distance_median"]) != substrate.distance_median:
            raise ValueError("checkpoint distance median mismatch")
        public_ids = np.asarray(artifact["public_ids"])
        prediction = np.asarray(artifact["public_predictions"])
        if not np.array_equal(public_ids, substrate.public_ids) or prediction.dtype != np.int8 or prediction.shape != (PUBLIC_ROWS,):
            raise ValueError("checkpoint Public identity/prediction mismatch")
        if np.any((prediction < 0) | (prediction > 6)):
            raise ValueError("checkpoint prediction outside registered classes")
        n_iter = np.asarray(artifact["n_iter"])
        if n_iter.dtype != np.int64 or n_iter.shape != (1,) or np.any(n_iter <= 0):
            raise ValueError("checkpoint solver diagnostics invalid")
        metric = _metric_record(substrate.public_labels, prediction)
        if metric["accuracy"] != float(artifact["public_accuracy"]) or metric["macro_f1"] != float(artifact["public_macro_f1"]):
            raise ValueError("checkpoint metric does not reproduce from prediction")
        json.loads(_scalar_text(artifact["warnings_json"]))
        result = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "seed": seed,
            "nested_train_sparse_sha256": _scalar_text(artifact["nested_train_sparse_sha256"]),
            "nested_public_sparse_sha256": _scalar_text(artifact["nested_public_sparse_sha256"]),
            "n_iter": n_iter.tolist(),
            "converged": bool(artifact["converged"]),
            "warnings": json.loads(_scalar_text(artifact["warnings_json"])),
            "metrics": metric,
        }
    return result


def run_r2_shard(
    occurrences_path: str | Path,
    results_path: str | Path,
    output_dir: str | Path,
    *,
    execution_wrapper_sha: str,
    shard_name: str,
    run_actual: bool,
    seeds: Sequence[int],
) -> dict[str, object]:
    if len(set(seeds)) != len(seeds) or tuple(seeds) != tuple(sorted(seeds)):
        raise ValueError("shard seeds must be unique and ascending")
    if any(seed not in GEOMETRY_SEEDS for seed in seeds):
        raise ValueError("shard contains an unregistered geometry seed")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    substrate = load_frozen_r1(occurrences_path, results_path)
    _log(
        f"canonical R1 substrate verified; distance_median={substrate.distance_median:.15g}; "
        f"O_train={substrate.o_train_sha256}; O_public={substrate.o_public_sha256}"
    )
    completed: list[dict[str, object]] = []
    arms: list[int | None] = ([None] if run_actual else []) + list(seeds)
    for seed in arms:
        checkpoint = out / _checkpoint_name(seed)
        if checkpoint.exists():
            try:
                record = validate_checkpoint(
                    checkpoint,
                    seed=seed,
                    execution_wrapper_sha=execution_wrapper_sha,
                    substrate=substrate,
                )
                record["resume_action"] = "SKIPPED_VALID_CHECKPOINT"
                completed.append(record)
                _log(f"SKIP valid checkpoint {checkpoint.name} sha256={record['sha256']}")
                continue
            except Exception as exc:
                corrupt_hash = sha256_file(checkpoint)
                quarantine = checkpoint.with_name(f"{checkpoint.name}.corrupt-{corrupt_hash[:16]}")
                checkpoint.replace(quarantine)
                _log(f"quarantined corrupt checkpoint {checkpoint.name}: {type(exc).__name__}: {exc}")
        condition = "actual_O_G" if seed is None else f"geometry_shuffle_{seed}"
        _log(f"building exact nested CSR for {condition}")
        train_nested, public_nested = build_nested_arm(substrate, seed=seed)
        _log(f"fitting fixed registered probe for {condition}")
        prediction, diagnostic, metric = fit_nested_arm(
            train_nested,
            substrate.train_labels,
            public_nested,
            substrate.public_labels,
            condition=condition,
        )
        payload = _checkpoint_payload(
            substrate,
            seed=seed,
            execution_wrapper_sha=execution_wrapper_sha,
            train_nested=train_nested,
            public_nested=public_nested,
            prediction=prediction,
            diagnostic=diagnostic,
            metric=metric,
        )
        _atomic_savez(checkpoint, payload)
        record = validate_checkpoint(
            checkpoint,
            seed=seed,
            execution_wrapper_sha=execution_wrapper_sha,
            substrate=substrate,
        )
        record["resume_action"] = "COMPUTED_AND_CHECKPOINTED"
        completed.append(record)
        _log(f"CHECKPOINT {checkpoint.name} sha256={record['sha256']} bytes={record['bytes']}")
        del train_nested, public_nested, prediction, payload
    manifest = {
        "experiment": "PGM_E0.R_R2_EXECUTION_SHARD",
        "issue": ISSUE_NUMBER,
        "scientific_sha": SCIENTIFIC_SHA,
        "execution_wrapper_sha": execution_wrapper_sha,
        "shard_name": shard_name,
        "run_actual": run_actual,
        "seeds": list(seeds),
        "r1_occurrences_sha256": R1_OCCURRENCES_SHA256,
        "r1_results_sha256": R1_RESULTS_SHA256,
        "distance_median": substrate.distance_median,
        "o_train_sha256": substrate.o_train_sha256,
        "o_public_sha256": substrate.o_public_sha256,
        "private_test_read": False,
        "scientific_verdict_written": False,
        "fits": completed,
    }
    manifest_path = out / f"e0r_r2_shard_{shard_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    _log(f"shard {shard_name} complete; no scientific R2 verdict calculated")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execution-only checkpointed PGM E0.R R2 shard")
    parser.add_argument("--r1-occurrences", required=True, type=Path)
    parser.add_argument("--r1-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-wrapper-sha", required=True)
    parser.add_argument("--shard-name", required=True)
    parser.add_argument("--r2-actual", action="store_true")
    parser.add_argument("--r2-seeds", nargs="*", type=int, default=[])
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    run_r2_shard(
        args.r1_occurrences,
        args.r1_results,
        args.output_dir,
        execution_wrapper_sha=args.execution_wrapper_sha,
        shard_name=args.shard_name,
        run_actual=args.r2_actual,
        seeds=args.r2_seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
