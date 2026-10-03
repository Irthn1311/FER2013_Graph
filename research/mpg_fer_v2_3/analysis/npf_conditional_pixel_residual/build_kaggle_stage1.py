"""Build the private, hash-locked Kaggle Stage-1 launch bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil


WORKTREE = Path(
    r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\.codex-internal-complementarity-worktree"
)
RESEARCH_ROOT = WORKTREE / "research" / "mpg_fer_v2_3"
EXPERIMENT_ROOT = RESEARCH_ROOT / "analysis" / "npf_conditional_pixel_residual"
STAGING_ROOT = Path(r"D:\KaggleStaging\mpg-fer-npf-residual-20261003")
INPUT_ROOT = STAGING_ROOT / "input_dataset"
KERNEL_ROOT = STAGING_ROOT / "kernel"
PHASE0_ROOT = Path(
    r"D:\KaggleStaging\mpg-fer-ablation7-20261002\analysis\npf_conditional_pixel_residual"
)
FULL_CHECKPOINT = Path(
    r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3\official_runs"
    r"\segment_02\mpg_fer_v2_3_run\best_val_acc.pt"
)
NPF_CHECKPOINT = Path(
    r"D:\KaggleStaging\mpg-fer-ablation7-20261002\resume_segment2\verified_outputs"
    r"\NO_PIXEL_FUSION\mpg-fer-table-vi\mpgfer-ablation-no-pixel-fusion-s42"
    r"\best_val_acc.pt"
)

KAGGLE_USER = "irthn1311"
DATASET_SLUG = "mpg-fer-npf-residual-stage1-input"
KERNEL_SLUG = "mpg-fer-npf-residual-stage1-seed42"

EXPECTED = {
    "full_checkpoint_sha256": "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e",
    "npf_checkpoint_sha256": "f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972",
    "train_sha256": "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8",
    "public_sha256": "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08",
    "private_sha256": "be385344b93606c75cb65cac3d39b13cdedb5b91621640ee08f73b9a551fbd9d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def add_file(manifest: dict[str, dict[str, int | str]], source: Path, relative: str) -> None:
    target = INPUT_ROOT / relative
    copy_file(source, target)
    manifest[relative] = {"sha256": sha256(target), "size_bytes": target.stat().st_size}


def notebook() -> dict:
    code = r'''from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/npf-residual-stage1-runtime")
OUTPUT_ROOT = Path("/kaggle/working/npf_conditional_pixel_residual")
EXPECTED_DATA = {
    "train.csv": "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8",
    "val.csv": "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08",
    "test.csv": "be385344b93606c75cb65cac3d39b13cdedb5b91621640ee08f73b9a551fbd9d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}, found {len(matches)}")
    return matches[0]


manifest_path = unique("INPUT_MANIFEST.json")
bundle_root = manifest_path.parent
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest["schema_version"] != 1 or manifest["purpose"] != "NPF_RESIDUAL_STAGE1":
    raise RuntimeError("Input manifest contract mismatch")
for relative, identity in manifest["files"].items():
    path = bundle_root / relative
    if not path.is_file() or path.stat().st_size != identity["size_bytes"]:
        raise RuntimeError(f"Missing or size-mismatched input: {relative}")
    if sha256(path) != identity["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {relative}")

# Resolve the official mounted split only by path names here. The runner hashes
# Train/Public immediately and deliberately defers opening/hash-checking Test
# until both Public selectors are frozen.
train_candidates = [
    path for path in INPUT_ROOT.rglob("train.csv")
    if path.parent.name == "fer13-split" and (path.parent / "val.csv").is_file()
    and (path.parent / "test.csv").is_file()
]
if len(train_candidates) != 1:
    raise RuntimeError(f"Expected one fer13-split mount, found {len(train_candidates)}")
train_csv = train_candidates[0]
public_csv = train_csv.parent / "val.csv"
private_csv = train_csv.parent / "test.csv"

if WORK_ROOT.exists() or OUTPUT_ROOT.exists():
    raise RuntimeError("Fail-closed: Kaggle working output already exists")
shutil.copytree(bundle_root / "src", WORK_ROOT / "src")
shutil.copytree(bundle_root / "experiment", WORK_ROOT / "experiment")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
for name in (
    "PUBLIC_FULL_NPF_AUDIT.json",
    "PUBLIC_FULL_NPF_AUDIT.md",
    "PUBLIC_FULL_NPF_OUTPUTS.npz",
    "PUBLIC_FULL_NPF_FUSION_SWEEP.csv",
):
    shutil.copy2(bundle_root / "phase0" / name, OUTPUT_ROOT / name)

launch = {
    "status": "STARTED",
    "seed": 42,
    "variants": ["R1", "R2"],
    "epochs": 30,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "backbone_microbatch_size": 16,
    "private_opened_before_selector_freeze": False,
    "input_manifest_sha256": sha256(manifest_path),
}
print(json.dumps(launch, indent=2), flush=True)

runner = WORK_ROOT / "experiment" / "run_experiment.py"
command = [
    sys.executable, "-u", str(runner),
    "--repo", str(WORK_ROOT),
    "--source-root", str(WORK_ROOT / "src"),
    "--train-csv", str(train_csv),
    "--public-csv", str(public_csv),
    "--private-csv", str(private_csv),
    "--full-checkpoint", str(bundle_root / "checkpoints" / "full_best_val_acc.pt"),
    "--npf-checkpoint", str(bundle_root / "checkpoints" / "npf_best_val_acc.pt"),
    "--output", str(OUTPUT_ROOT),
    "--batch-size", "16",
    "--eval-batch-size", "16",
    "--backbone-microbatch-size", "16",
    "--num-workers", "2",
    "--device", "cuda",
    "--max-epochs", "30",
    "--resume-phase0",
]
subprocess.run(command, check=True)
subprocess.run(
    [sys.executable, "-u", str(WORK_ROOT / "experiment" / "validate_experiment.py"),
     "--output", str(OUTPUT_ROOT)],
    check=True,
)
archive = shutil.make_archive(
    "/kaggle/working/npf_conditional_pixel_residual_artifacts", "zip", OUTPUT_ROOT
)
summary = json.loads((OUTPUT_ROOT / "RESIDUAL_EXPERIMENT_SUMMARY.json").read_text(encoding="utf-8"))
print(json.dumps({
    "status": "COMPLETE_AND_VALIDATED",
    "decision": summary["decision"],
    "selected_variant": summary["selected_variant_by_public"],
    "go_stage_2": summary["go_stage_2"],
    "archive": archive,
}, indent=2), flush=True)
'''
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MPG-FER NPF conditional pixel residual — Stage 1\n",
                    "Hash-locked R1/R2 training. Public selection precedes one final Test evaluation.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "experiment": "NPF_RESIDUAL_STAGE1",
            "seed": 42,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    if STAGING_ROOT.exists():
        raise FileExistsError(f"Fail-closed staging root exists: {STAGING_ROOT}")
    INPUT_ROOT.mkdir(parents=True)
    KERNEL_ROOT.mkdir(parents=True)

    if sha256(FULL_CHECKPOINT) != EXPECTED["full_checkpoint_sha256"]:
        raise RuntimeError("FULL checkpoint identity mismatch")
    if sha256(NPF_CHECKPOINT) != EXPECTED["npf_checkpoint_sha256"]:
        raise RuntimeError("NPF checkpoint identity mismatch")

    files: dict[str, dict[str, int | str]] = {}
    for package in ("mpg_fer_v2_3", "mpg_fer_table_vi"):
        for source in sorted((RESEARCH_ROOT / "src" / package).glob("*.py")):
            add_file(files, source, f"src/{package}/{source.name}")
    for name in (
        "__init__.py",
        "model.py",
        "protocol.py",
        "run_experiment.py",
        "validate_experiment.py",
    ):
        add_file(files, EXPERIMENT_ROOT / name, f"experiment/{name}")
    add_file(files, FULL_CHECKPOINT, "checkpoints/full_best_val_acc.pt")
    add_file(files, NPF_CHECKPOINT, "checkpoints/npf_best_val_acc.pt")
    for name in (
        "PUBLIC_FULL_NPF_AUDIT.json",
        "PUBLIC_FULL_NPF_AUDIT.md",
        "PUBLIC_FULL_NPF_OUTPUTS.npz",
        "PUBLIC_FULL_NPF_FUSION_SWEEP.csv",
    ):
        add_file(files, PHASE0_ROOT / name, f"phase0/{name}")

    manifest = {
        "schema_version": 1,
        "purpose": "NPF_RESIDUAL_STAGE1",
        "source_base_commit": "1757dde108c71f756fffe51620a1c84bb003a7bd",
        "seed": 42,
        "variants": ["R1", "R2"],
        "dataset_expected_sha256": {
            "train.csv": EXPECTED["train_sha256"],
            "val.csv": EXPECTED["public_sha256"],
            "test.csv": EXPECTED["private_sha256"],
        },
        "files": files,
    }
    (INPUT_ROOT / "INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (INPUT_ROOT / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MPG-FER NPF residual Stage1 input",
                "id": f"{KAGGLE_USER}/{DATASET_SLUG}",
                "licenses": [{"name": "unknown"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (KERNEL_ROOT / "npf_residual_stage1.ipynb").write_text(
        json.dumps(notebook(), indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (KERNEL_ROOT / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{KAGGLE_USER}/{KERNEL_SLUG}",
                "title": "MPG FER NPF residual Stage1 seed42",
                "code_file": "npf_residual_stage1.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "dataset_sources": [
                    "doduyquynii/fer13-split",
                    f"{KAGGLE_USER}/{DATASET_SLUG}",
                ],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "BUILT",
                "staging_root": str(STAGING_ROOT),
                "input_file_count": len(files),
                "input_manifest_sha256": sha256(INPUT_ROOT / "INPUT_MANIFEST.json"),
                "notebook_sha256": sha256(KERNEL_ROOT / "npf_residual_stage1.ipynb"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
