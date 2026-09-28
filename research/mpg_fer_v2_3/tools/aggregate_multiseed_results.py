"""Build the verified six-seed MPG-FER v2.3 result registry and handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import zipfile

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


SEEDS = (0, 1, 42, 43, 123, 3047)
NEW_SEEDS = (0, 1, 43, 123, 3047)
SOURCE_SHA256 = "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87"
CONFIG_SHA256 = {
    0: "fc57b7c057df4226ac2eadddb3b902ad157bd426710c6dc413ab80feaf920565",
    1: "5086767014b1938f7783597498a3ea1fdb99fbd6178e8b42fb890d88d009d26e",
    42: "8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2",
    43: "97f594b19295822946e5632cff9535d630a3b505a2a0375cf04fc9a36dbece4e",
    123: "e85608767dc1b4c57897c6273b6014089b7df9f80005345c4da9b595a21823f8",
    3047: "895abb86818e7ac77c31186bac12efc2353ac79265715e14387fbc0ab135f779",
}
CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
METRICS = (
    "public_raw_acc",
    "public_raw_macro_f1",
    "private_raw_acc",
    "private_raw_macro_f1",
    "public_tta_acc",
    "public_tta_macro_f1",
    "private_tta_acc",
    "private_tta_macro_f1",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_verified_final(zip_path: Path, destination: Path, seed: int) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"Corrupt final ZIP for seed {seed}")
        names = set(archive.namelist())
        required = {
            "RUN_MANIFEST.json",
            "execution_manifest.json",
            "final_selection_manifest.json",
            "best_val_acc.pt",
            "hashes/SHA256SUMS.txt",
            "evaluation/private_raw_predictions.csv",
        }
        if not required.issubset(names):
            raise RuntimeError(f"Incomplete final ZIP for seed {seed}: {required - names}")
        manifest = json.loads(archive.read("RUN_MANIFEST.json"))
        execution = json.loads(archive.read("execution_manifest.json"))
        selection = json.loads(archive.read("final_selection_manifest.json"))
        checks = {
            "seed": manifest.get("seed") == seed,
            "status": manifest.get("run_status") == "TRAINING_COMPLETED"
            and execution.get("status") == "TRAINING_COMPLETED",
            "source": manifest.get("source_sha256") == SOURCE_SHA256,
            "config": manifest.get("config_sha256") == CONFIG_SHA256[seed],
            "private_firewall": manifest.get("private_evaluated_only_after_freeze") is True
            and execution.get("PRIVATE_EVALUATED") is True,
            "checkpoint": sha256_bytes(archive.read("best_val_acc.pt"))
            == manifest.get("checkpoint_sha256")
            == selection.get("checkpoint_sha256"),
        }
        declared = {
            rel: digest
            for digest, rel in (
                line.split("  ", 1)
                for line in archive.read("hashes/SHA256SUMS.txt")
                .decode("utf-8")
                .splitlines()
                if line
            )
        }
        checks["internal_hashes"] = all(
            rel in names and sha256_bytes(archive.read(rel)) == digest
            for rel, digest in declared.items()
        )
        if not all(checks.values()):
            raise RuntimeError(f"Final identity failure seed {seed}: {checks}")

        manifest["run_id"] = execution["run_id"]

        destination.mkdir(parents=True, exist_ok=True)
        for name in sorted(names):
            if name.endswith("/") or name.endswith(".pt") or name.endswith(".zip"):
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        write_json(destination / "RUN_MANIFEST.json", manifest)
        write_json(
            destination / "archive_identity.json",
            {
                "source_archive": str(zip_path.resolve()),
                "archive_sha256": sha256_file(zip_path),
                "verified_internal_files": len(declared),
                "checkpoint_omitted_from_handoff": True,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            },
        )
        return manifest


def seed42_documents(project_root: Path, destination: Path) -> dict:
    research_root = project_root / "research" / "mpg_fer_v2_3"
    run = research_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run"
    summary = json.loads(
        (research_root / "official_runs" / "multiseed" / "seed_42"
         / "canonical_fp32_summary.json").read_text(encoding="utf-8")
    )
    public_official = json.loads((research_root / "v23_public_results.json").read_text(encoding="utf-8"))
    private_official = json.loads((research_root / "v23_private_results.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "final_selection_manifest.json").read_text(encoding="utf-8"))
    if selection["checkpoint_sha256"] != summary["checkpoint_sha256"]:
        raise RuntimeError("Seed-42 checkpoint identity mismatch")
    if sha256_file(run / "best_val_acc.pt") != summary["checkpoint_sha256"]:
        raise RuntimeError("Seed-42 checkpoint file hash mismatch")

    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.suffix in {".pt", ".zip"}:
            continue
        target = destination / "official_run" / path.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ("v23_public_results.json", "v23_private_results.json"):
        shutil.copy2(research_root / name, destination / name)

    cache = np.load(
        research_root / "official_runs" / "analysis_cache" / "v23_frozen_features.npz"
    )
    targets = cache["private_targets"].astype(np.int64)
    logits = cache["private_R10"].astype(np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    prediction_rows = [
        {
            "row_index": index,
            "target": int(targets[index]),
            "prediction": int(predictions[index]),
            "confidence": float(probabilities[index].max()),
        }
        for index in range(len(targets))
    ]
    write_csv(
        destination / "evaluation" / "private_raw_predictions.csv",
        ["row_index", "target", "prediction", "confidence"],
        prediction_rows,
    )
    matrix = confusion_matrix(targets, predictions, labels=np.arange(7))
    write_csv(
        destination / "evaluation" / "confusion_matrix_raw.csv",
        ["true_class", *CLASS_NAMES],
        [
            {"true_class": CLASS_NAMES[row], **{
                CLASS_NAMES[col]: int(matrix[row, col]) for col in range(7)
            }}
            for row in range(7)
        ],
    )
    precision = precision_score(
        targets, predictions, labels=np.arange(7), average=None, zero_division=0
    )
    recall = recall_score(
        targets, predictions, labels=np.arange(7), average=None, zero_division=0
    )
    f1 = f1_score(targets, predictions, labels=np.arange(7), average=None, zero_division=0)
    write_csv(
        destination / "evaluation" / "per_class_raw.csv",
        ["class", "precision", "recall", "f1", "support"],
        [
            {
                "class": CLASS_NAMES[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(np.sum(targets == index)),
            }
            for index in range(7)
        ],
    )
    for split in ("public", "private"):
        for view in ("raw", "tta"):
            metric = summary[split][view]
            write_json(
                destination / "evaluation" / f"{split}_{view}_canonical_fp32.json",
                {
                    "split": "PublicTest" if split == "public" else "PrivateTest",
                    "view": view,
                    "evaluation_precision": "canonical_fp32_no_autocast_tf32_disabled",
                    "weights_type": "EMA",
                    **metric,
                },
            )
    write_json(destination / "canonical_fp32_summary.json", summary)
    manifest = {
        "method": "MPG-FER",
        "version": "2.3",
        "seed": 42,
        "run_id": public_official["identity"]["run_id"],
        "source_sha256": public_official["identity"]["source_sha256"],
        "config_sha256": public_official["identity"]["config_sha256"],
        "selected_epoch": summary["selected_epoch"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "run_status": "TRAINING_COMPLETED",
        "private_evaluated_only_after_freeze": True,
        "canonical_fp32": {
            "public_raw": summary["public"]["raw"],
            "public_tta": summary["public"]["tta"],
            "private_raw": summary["private"]["raw"],
            "private_tta": summary["private"]["tta"],
        },
        "official_runtime_amp": {
            "public_raw": public_official["raw"],
            "public_tta": public_official["tta"],
            "private_raw": private_official["raw"],
            "private_tta": private_official["tta"],
        },
        "checkpoint_omitted_from_handoff": True,
    }
    write_json(destination / "RUN_MANIFEST.json", manifest)
    return manifest


def registry_row(seed: int, manifest: dict, total_runtime_seconds: float | None) -> dict:
    canonical = manifest["canonical_fp32"]
    return {
        "seed": seed,
        "selected_epoch": manifest["selected_epoch"],
        "evaluation_precision": "canonical_fp32_no_autocast_tf32_disabled",
        "public_raw_acc": canonical["public_raw"]["accuracy"],
        "public_raw_macro_f1": canonical["public_raw"]["macro_f1"],
        "private_raw_acc": canonical["private_raw"]["accuracy"],
        "private_raw_macro_f1": canonical["private_raw"]["macro_f1"],
        "public_tta_acc": canonical["public_tta"]["accuracy"],
        "public_tta_macro_f1": canonical["public_tta"]["macro_f1"],
        "private_tta_acc": canonical["private_tta"]["accuracy"],
        "private_tta_macro_f1": canonical["private_tta"]["macro_f1"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
        "source_sha256": manifest["source_sha256"],
        "run_id": manifest.get("run_id"),
        "total_runtime_seconds": total_runtime_seconds,
        "run_status": manifest["run_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--handoff-zip", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    research_root = project_root / "research" / "mpg_fer_v2_3"
    runs_root = research_root / "official_runs" / "multiseed"
    output_dir = (args.output_dir or research_root / "MPG_V23_MULTI_SEED_RESULTS").resolve()
    handoff_zip = (args.handoff_zip or project_root / "MPG_V23_MULTI_SEED_HANDOFF.zip").resolve()
    if output_dir.exists() or handoff_zip.exists():
        raise FileExistsError("Refusing to overwrite an existing result directory or handoff ZIP")

    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        build = Path(temporary) / output_dir.name
        manifests: dict[int, dict] = {}
        seed42_segment_01 = json.loads(
            (research_root / "official_runs" / "segment_01" / "mpg_fer_v2_3_run"
             / "segment_manifest.json").read_text(encoding="utf-8")
        )
        seed42_segment_02 = json.loads(
            (research_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run"
             / "segment_manifest.json").read_text(encoding="utf-8")
        )
        runtimes: dict[int, float | None] = {
            42: float(seed42_segment_01["wallclock_seconds"])
            + float(seed42_segment_02["wallclock_seconds"])
        }
        for seed in NEW_SEEDS:
            final_dir = runs_root / f"seed_{seed}" / "segment_02"
            archives = list(final_dir.glob(f"mpg_v2_3_seed_{seed}_artifacts.zip"))
            if len(archives) != 1:
                raise RuntimeError(f"Expected one final archive for seed {seed}")
            manifests[seed] = extract_verified_final(
                archives[0], build / f"seed_{seed}", seed
            )
            s1 = json.loads(zipfile.ZipFile(next(
                (runs_root / f"seed_{seed}" / "segment_01").glob(
                    f"mpg_v2_3_seed_{seed}_artifacts.zip"
                )
            )).read("segment_manifest.json"))
            s2 = json.loads((build / f"seed_{seed}" / "segment_manifest.json").read_text())
            runtimes[seed] = float(s1["wallclock_seconds"]) + float(s2["wallclock_seconds"])
        manifests[42] = seed42_documents(project_root, build / "seed_42")

        rows = [registry_row(seed, manifests[seed], runtimes[seed]) for seed in SEEDS]
        fields = list(rows[0])
        write_csv(build / "multi_seed_registry.csv", fields, rows)
        aggregate = {
            "method": "MPG-FER",
            "version": "2.3",
            "seed_set": list(SEEDS),
            "n": len(SEEDS),
            "evaluation_precision": "canonical_fp32_no_autocast_tf32_disabled",
            "standard_deviation": "sample SD (ddof=1)",
            "metrics": {},
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            aggregate["metrics"][metric] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
                "values_by_seed": {str(row["seed"]): float(row[metric]) for row in rows},
            }
        write_json(build / "aggregate_statistics.json", aggregate)
        write_json(
            build / "scientific_integrity.json",
            {
                "no_seed_selection": True,
                "official_seed_set": list(SEEDS),
                "private_test_used_for_model_selection": False,
                "hyperparameter_changes_between_seeds": False,
                "result_based_reruns": False,
                "infrastructure_retries": [
                    "Segment 1 required exact continuation because of Kaggle wall-clock limits.",
                    "One local artifact download for seed 123 was retried after Windows network error 10053; training was not rerun.",
                ],
                "interpretation_boundary": "Multi-seed replication only; no superiority or SOTA claim.",
            },
        )
        headline = aggregate["metrics"]
        lines = [
            "# MPG-FER v2.3 Six-Seed Replication Results",
            "",
            "Canonical evaluation: FP32, autocast disabled, TF32 disabled, frozen EMA checkpoint.",
            "",
            "## Headline",
            "",
            f"- Private Raw Accuracy: {headline['private_raw_acc']['mean']:.6f} +/- {headline['private_raw_acc']['sample_sd']:.6f}",
            f"- Private Raw Macro-F1: {headline['private_raw_macro_f1']['mean']:.6f} +/- {headline['private_raw_macro_f1']['sample_sd']:.6f}",
            f"- Private TTA Accuracy: {headline['private_tta_acc']['mean']:.6f} +/- {headline['private_tta_acc']['sample_sd']:.6f}",
            f"- Private TTA Macro-F1: {headline['private_tta_macro_f1']['mean']:.6f} +/- {headline['private_tta_macro_f1']['sample_sd']:.6f}",
            "",
            "## Individual seeds",
            "",
            "| Seed | Epoch | Private raw acc | Private raw macro-F1 | Private TTA acc | Private TTA macro-F1 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['seed']} | {row['selected_epoch']} | {row['private_raw_acc']:.6f} | "
                f"{row['private_raw_macro_f1']:.6f} | {row['private_tta_acc']:.6f} | "
                f"{row['private_tta_macro_f1']:.6f} |"
            )
        lines += [
            "",
            "All six registered seeds are retained. PrivateTest did not select checkpoints, "
            "and no run was repeated because of its metric value.",
        ]
        (build / "MULTI_SEED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        package_hashes = []
        for path in sorted(build.rglob("*")):
            if path.is_file() and path.name != "PACKAGE_SHA256SUMS.txt":
                package_hashes.append(
                    f"{sha256_file(path)}  {path.relative_to(build).as_posix()}"
                )
        (build / "PACKAGE_SHA256SUMS.txt").write_text(
            "\n".join(package_hashes) + "\n", encoding="utf-8"
        )
        build.rename(output_dir)

    archive_base = handoff_zip.with_suffix("")
    created = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_dir))
    if created != handoff_zip:
        raise RuntimeError(f"Unexpected handoff archive path: {created}")
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(output_dir),
        "handoff_zip": str(handoff_zip),
        "handoff_sha256": sha256_file(handoff_zip),
        "rows": len(SEEDS),
        "private_raw_accuracy_mean": aggregate["metrics"]["private_raw_acc"]["mean"],
        "private_raw_accuracy_sample_sd": aggregate["metrics"]["private_raw_acc"]["sample_sd"],
    }, indent=2))


if __name__ == "__main__":
    main()
