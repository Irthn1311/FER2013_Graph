"""Independent fail-closed validator for complementarity runtime artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
import torch


N = 3589
EXPECTED_FULL = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"
EXPECTED_NPF = "f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972"
EXPECTED_FULL_METRICS = {
    "raw": {"accuracy": 0.6876567288938423, "macro_f1": 0.6734818933345407},
    "tta": {"accuracy": 0.7066035107272220, "macro_f1": 0.6981583150632577},
}
FILES = (
    "FULL_INTERNAL_COMPLEMENTARITY_AUDIT.json",
    "FULL_INTERNAL_COMPLEMENTARITY_AUDIT.md",
    "FULL_INTERNAL_HEAD_LOGITS.npz",
    "FULL_INTERNAL_READOUTS.npz",
    "FULL_INTERNAL_SAMPLE_DIAGNOSTICS.csv",
    "FULL_INTERNAL_CLASSWISE.csv",
    "FULL_INTERNAL_FUSION_SWEEP.csv",
    "FULL_INTERNAL_CROSSFIT_FUSION.json",
    "FULL_NPF_WEIGHT_DISTANCE.json",
    "FULL_NPF_WEIGHT_INTERPOLATION.csv",
    "environment.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = logits.argmax(axis=1)
    return {
        "loss": float(-np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-300, 1.0)).mean()),
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", labels=np.arange(7), zero_division=0)),
        "correct_count": int(np.sum(prediction == labels)),
    }


def assert_metrics(actual: dict[str, Any], expected: dict[str, Any], tolerance: float = 2e-9) -> None:
    for key in ("loss", "accuracy", "macro_f1"):
        if abs(float(actual[key]) - float(expected[key])) > tolerance:
            raise RuntimeError(f"metric mismatch for {key}: {actual[key]} != {expected[key]}")
    if int(actual["correct_count"]) != int(expected["correct_count"]):
        raise RuntimeError("correct-count mismatch")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pair_counts(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    first_ok = first == labels
    second_ok = second == labels
    equal = first == second
    return {
        "N_CC": int((first_ok & second_ok).sum()),
        "N_CW": int((first_ok & ~second_ok).sum()),
        "N_WC": int((~first_ok & second_ok).sum()),
        "N_WW": int((~first_ok & ~second_ok).sum()),
        "prediction_agreement_count": int(equal.sum()),
        "prediction_disagreement_count": int((~equal).sum()),
        "prediction_disagreement_rate": float(np.mean(~equal)),
        "correctness_disagreement_count": int((first_ok ^ second_ok).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    staging = Path(r"D:\KaggleStaging\mpg-fer-ablation7-20261002")
    repo = Path(__file__).resolve().parents[4]
    checkpoint_repo = repo.parent if repo.name.startswith(".codex-") else repo
    parser.add_argument("--output", type=Path, default=staging / "analysis/full_internal_complementarity")
    parser.add_argument("--full-checkpoint", type=Path, default=checkpoint_repo / "research/mpg_fer_v2_3/official_runs/segment_02/mpg_fer_v2_3_run/best_val_acc.pt")
    parser.add_argument("--npf-checkpoint", type=Path, default=staging / "resume_segment2/verified_outputs/NO_PIXEL_FUSION/mpg-fer-table-vi/mpgfer-ablation-no-pixel-fusion-s42/best_val_acc.pt")
    args = parser.parse_args()
    for name in (*FILES, "checksums.sha256"):
        if not (args.output / name).is_file():
            raise FileNotFoundError(name)

    checksum_rows = {}
    for line in (args.output / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksum_rows[name] = digest
    if set(checksum_rows) != set(FILES):
        raise RuntimeError("checksum manifest file set mismatch")
    for name in FILES:
        if sha(args.output / name) != checksum_rows[name]:
            raise RuntimeError(f"checksum mismatch: {name}")

    audit = json.loads((args.output / FILES[0]).read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["scope"]["training"] is not False:
        raise RuntimeError("audit scope/status mismatch")
    if not audit["model_immutability"]["FULL"]["unchanged"] or not audit["model_immutability"]["interpolation_all_unchanged"]:
        raise RuntimeError("reported model-state immutability failure")
    if sha(args.full_checkpoint) != EXPECTED_FULL or sha(args.npf_checkpoint) != EXPECTED_NPF:
        raise RuntimeError("checkpoint bytes changed")

    with np.load(args.output / "FULL_INTERNAL_HEAD_LOGITS.npz") as arrays:
        labels = arrays["true_label"]
        if labels.shape != (N,) or not np.array_equal(arrays["row_index"], np.arange(N)):
            raise RuntimeError("row alignment mismatch")
        for head in ("fused", "pixel", "motif"):
            raw = arrays[f"{head}_raw_logits"]
            flip = arrays[f"{head}_flip_logits"]
            tta = arrays[f"{head}_tta_logits"]
            if raw.shape != (N, 7) or not np.isfinite(raw).all() or not np.array_equal(tta, (0.5 * (raw + flip)).astype(np.float32)):
                raise RuntimeError(f"malformed or inconsistent logits: {head}")
            for view, logits in (("raw", raw), ("tta", tta)):
                actual = metrics(labels, logits)
                assert_metrics(actual, audit["head_metrics"][head][view])
                if head == "fused":
                    for key, expected in EXPECTED_FULL_METRICS[view].items():
                        if abs(float(actual[key]) - expected) > 1e-9:
                            raise RuntimeError(f"frozen FULL parity mismatch: {view}/{key}")

        for first_name, second_name in (("fused", "pixel"), ("fused", "motif"), ("pixel", "motif")):
            pair_name = f"{first_name}_vs_{second_name}"
            for view in ("raw", "tta"):
                first_logits = arrays[f"{first_name}_{view}_logits"]
                second_logits = arrays[f"{second_name}_{view}_logits"]
                first_prediction = first_logits.argmax(axis=1)
                second_prediction = second_logits.argmax(axis=1)
                recorded = audit["pairwise"][pair_name][view]
                for key, value in pair_counts(labels, first_prediction, second_prediction).items():
                    expected = recorded["disagreement"][key]
                    if isinstance(value, float):
                        if abs(value - float(expected)) > 1e-15:
                            raise RuntimeError(f"pairwise mismatch: {pair_name}/{view}/{key}")
                    elif value != int(expected):
                        raise RuntimeError(f"pairwise mismatch: {pair_name}/{view}/{key}")
                either_correct = (first_prediction == labels) | (second_prediction == labels)
                if int(either_correct.sum()) != int(recorded["oracle"]["correct_count"]):
                    raise RuntimeError(f"oracle mismatch: {pair_name}/{view}")
                assert_metrics(metrics(labels, 0.5 * (first_logits + second_logits)), recorded["equal_logit_average"])

        crossfit = json.loads((args.output / "FULL_INTERNAL_CROSSFIT_FUSION.json").read_text(encoding="utf-8"))
        first = arrays["fused_tta_logits"]
        second = arrays["motif_tta_logits"]
        full_alpha = np.asarray(crossfit["full_data_fit"]["alpha_first_by_class"])
        expected_full_fit = (full_alpha * first + (1.0 - full_alpha) * second).astype(np.float32)
        if not np.array_equal(arrays["fused_motif_full_fit_logits"], expected_full_fit):
            raise RuntimeError("full-data coefficient fusion mismatch")
        assert_metrics(metrics(labels, expected_full_fit), crossfit["full_data_fit"]["metrics"])
        oof = np.empty_like(first)
        coverage = np.zeros(N, dtype=np.int64)
        fold_ids = np.full(N, -1, dtype=np.int64)
        for fold in crossfit["crossfit"]["folds"]:
            test = np.asarray(fold["test_indices"], dtype=np.int64)
            train = np.asarray(fold["train_indices"], dtype=np.int64)
            if np.intersect1d(train, test).size or len(train) + len(test) != N:
                raise RuntimeError("invalid crossfit partition")
            alpha = np.asarray(fold["alpha_first_by_class"])
            oof[test] = alpha * first[test] + (1.0 - alpha) * second[test]
            coverage[test] += 1
            fold_ids[test] = int(fold["fold"])
            assert_metrics(metrics(labels[test], oof[test]), fold["test_metrics"])
        if not np.all(coverage == 1) or not np.array_equal(fold_ids, arrays["fused_motif_crossfit_fold_id"]):
            raise RuntimeError("crossfit coverage/fold mismatch")
        if not np.allclose(oof, arrays["fused_motif_crossfit_oof_logits"], rtol=0, atol=2e-7):
            raise RuntimeError("crossfit OOF logits mismatch")
        assert_metrics(metrics(labels, oof), crossfit["crossfit"]["oof_metrics"])

        sweep = read_rows(args.output / "FULL_INTERNAL_FUSION_SWEEP.csv")
        if len(sweep) != 606:
            raise RuntimeError("scalar sweep must contain 3*2*101 rows")
        for row in sweep:
            a = row["first_head"]
            b = row["second_head"]
            view = row["view"]
            alpha = float(row["alpha_first"])
            fused = alpha * arrays[f"{a}_{view}_logits"] + (1.0 - alpha) * arrays[f"{b}_{view}_logits"]
            assert_metrics(metrics(labels, fused), row, tolerance=2e-9)

    with np.load(args.output / "FULL_INTERNAL_READOUTS.npz") as readouts:
        expected_shapes = {"pixel_readout_raw": (N, 128), "pixel_readout_flip": (N, 128), "motif_readout_raw": (N, 384), "motif_readout_flip": (N, 384)}
        for name, shape in expected_shapes.items():
            if readouts[name].shape != shape or not np.isfinite(readouts[name]).all():
                raise RuntimeError(f"readout shape/finite mismatch: {name}")

    if len(read_rows(args.output / "FULL_INTERNAL_SAMPLE_DIAGNOSTICS.csv")) != N:
        raise RuntimeError("sample diagnostics row count mismatch")
    if len(read_rows(args.output / "FULL_INTERNAL_CLASSWISE.csv")) != 42:
        raise RuntimeError("classwise row count mismatch")

    full_state = torch.load(args.full_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    npf_state = torch.load(args.npf_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    distance = json.loads((args.output / "FULL_NPF_WEIGHT_DISTANCE.json").read_text(encoding="utf-8"))
    if set(full_state) != set(npf_state) or not distance["keys_exact_match"]:
        raise RuntimeError("weight key compatibility mismatch")
    if state_sha(full_state) != distance["full_state_sha256"] or state_sha(npf_state) != distance["npf_state_sha256"]:
        raise RuntimeError("state hash mismatch")
    diff2 = full2 = 0.0
    for key in full_state:
        a, b = full_state[key], npf_state[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"tensor compatibility mismatch: {key}")
        if torch.is_floating_point(a) or torch.is_complex(a):
            av, bv = a.double().reshape(-1), b.double().reshape(-1)
            diff2 += float(torch.dot(av - bv, av - bv))
            full2 += float(torch.dot(av, av))
    relative = math.sqrt(diff2) / math.sqrt(full2)
    if abs(relative - float(distance["global"]["relative_l2"])) > 1e-12:
        raise RuntimeError("global weight distance mismatch")

    interpolation = read_rows(args.output / "FULL_NPF_WEIGHT_INTERPOLATION.csv")
    expected_grid = {(semantic, lam, view) for semantic in ("FULL", "NO_PIXEL_FUSION") for lam in (0.0, 0.25, 0.5, 0.75, 1.0) for view in ("raw", "tta")}
    actual_grid = {(row["semantics"], float(row["lambda_full"]), row["view"]) for row in interpolation}
    if actual_grid != expected_grid or any(row["state_unchanged"].lower() != "true" or row["finite_logits"].lower() != "true" for row in interpolation):
        raise RuntimeError("interpolation grid/immutability mismatch")
    for row in interpolation:
        for key in ("loss", "accuracy", "macro_f1", "runtime_seconds"):
            if not math.isfinite(float(row[key])):
                raise RuntimeError("non-finite interpolation result")
    print(json.dumps({"status": "PASS", "files_verified": len(FILES), "samples": N, "sweep_rows": len(sweep), "interpolation_rows": len(interpolation)}, indent=2))


if __name__ == "__main__":
    main()
