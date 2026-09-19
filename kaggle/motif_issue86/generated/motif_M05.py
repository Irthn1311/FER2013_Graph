"""Issue #86 Execution Amendment E1 — hard-coded execution unit M05."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
BRANCH = "research/motif-qualification-sparsification"
SCIENTIFIC_SHA = "16a84b2b36f0d3584afd0a547487c4373dc22128"
TRAIN_SHA256 = "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8"
DICTIONARY_SHA256 = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"
TRAIN_MODEL_SHA256 = "77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a"

UNIT_ID = "M05"
EXECUTION_MODE = "RUN_REPLICATE"
STABILITY_ARM = "M"
STABILITY_REPLICATE_ID = 5

OUTPUT_DIR = Path("/kaggle/working/motif_issue86_outputs")
REPO_DIR = Path("/kaggle/working/FER2013_Graph")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(candidates: tuple[Path, ...], expected_sha: str) -> Path:
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected_sha:
            return candidate
    if Path("/kaggle/input").exists():
        for filename in {c.name for c in candidates}:
            for match in Path("/kaggle/input").glob(f"**/{filename}"):
                if match.is_file() and sha256_file(match) == expected_sha:
                    return match
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _resolve_substrate_dir(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "motif_train_substrate_manifest.json").is_file()
        ):
            return candidate
    if Path("/kaggle/input").exists():
        for match in Path("/kaggle/input").glob(
            "**/motif_train_substrate_manifest.json"
        ):
            if match.is_file():
                return match.parent
    return candidates[0]


def _resolve_replicates_dir(candidates: tuple[Path, ...]) -> Path:
    expected_names = {
        f"motif_stability_{arm}_rep_{replicate_id:02d}.npz"
        for arm in ("M", "C")
        for replicate_id in range(20)
    }

    def _is_valid(directory: Path) -> bool:
        if not directory.is_dir():
            return False
        npz_names = {
            p.name for p in directory.glob("motif_stability_[MC]_rep_[0-9][0-9].npz")
        }
        if npz_names != expected_names:
            return False
        for name in expected_names:
            if not (directory / name.replace(".npz", ".manifest.json")).is_file():
                return False
        return True

    for candidate in candidates:
        if _is_valid(candidate):
            return candidate

    if Path("/kaggle/input").exists():
        for match in Path("/kaggle/input").glob("**/motif_stability_M_rep_00.npz"):
            if _is_valid(match.parent):
                return match.parent

    raise FileNotFoundError(
        "cannot resolve canonical replicate directory containing exact 40 M/C NPZ and manifest files"
    )


def run_unit() -> int:
    print(
        json.dumps(
            {
                "unit_id": UNIT_ID,
                "execution_mode": EXECUTION_MODE,
                "arm": STABILITY_ARM,
                "replicate_id": STABILITY_REPLICATE_ID,
                "scientific_source_sha": SCIENTIFIC_SHA,
                "internet_required_for_clone": True,
                "gpu_required": False,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if REPO_DIR.exists():
        raise FileExistsError(f"refusing pre-existing repository path: {REPO_DIR}")

    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            BRANCH,
            "--single-branch",
            REPO_URL,
            str(REPO_DIR),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", SCIENTIFIC_SHA], cwd=REPO_DIR, check=True
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
    ).strip()
    if head != SCIENTIFIC_SHA:
        raise RuntimeError(
            f"scientific source lock mismatch: {head} != {SCIENTIFIC_SHA}"
        )
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_DIR, text=True
    ).strip():
        raise RuntimeError("scientific source worktree is not clean")
    print(f"Detached checkout of scientific source {SCIENTIFIC_SHA}: PASS")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(REPO_DIR / "research/pixel_relational_motif_e0"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(REPO_DIR / "research/pixel_relational_motif_e0/tests"),
            "-q",
        ],
        check=True,
    )
    print("Pre-data full pytest PASS")

    train_csv_candidates = (
        Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/train.csv"),
        Path("/kaggle/input/fer13-split/fer13-split/train.csv"),
        Path("/kaggle/input/fer13-split/train.csv"),
    )
    dictionary_npz_candidates = (
        Path(
            "/kaggle/input/datasets/nuyntai/pgm-e01-v533-dictionary-crs/e01_dictionary.npz"
        ),
        Path("/kaggle/input/pgm-e01-v533-dictionary-crs/e01_dictionary.npz"),
    )
    train_model_npz_candidates = (
        Path(
            "/kaggle/input/datasets/nuyntai/pgm-crs-train-v2-artifacts/crs_train_model.npz"
        ),
        Path("/kaggle/input/pgm-crs-train-v2-artifacts/crs_train_model.npz"),
    )
    substrate_dir_candidates = (
        Path("/kaggle/input/pgm-motif-train-substrate"),
        Path("/kaggle/input/datasets/irthn1311/pgm-motif-train-substrate"),
        Path("/kaggle/input/datasets/nuyntai/pgm-motif-train-substrate"),
        Path("/kaggle/input/datasets/thanhhhgng/pgm-motif-train-substrate"),
        Path("/kaggle/input/datasets/trngthngcnhi/pgm-motif-train-substrate"),
    )
    replicates_dir_candidates = (
        Path("/kaggle/input/pgm-motif-stability-replicates"),
        Path("/kaggle/input/datasets/irthn1311/pgm-motif-stability-replicates"),
        Path("/kaggle/input/datasets/nuyntai/pgm-motif-stability-replicates"),
        Path("/kaggle/input/datasets/thanhhhgng/pgm-motif-stability-replicates"),
        Path("/kaggle/input/datasets/trngthngcnhi/pgm-motif-stability-replicates"),
    )

    train_model_npz = _resolve_file(train_model_npz_candidates, TRAIN_MODEL_SHA256)
    train_csv = _resolve_file(train_csv_candidates, TRAIN_SHA256)
    dictionary_npz = _resolve_file(dictionary_npz_candidates, DICTIONARY_SHA256)
    substrate_dir = _resolve_substrate_dir(substrate_dir_candidates)
    replicates_dir = _resolve_replicates_dir(replicates_dir_candidates)

    required_hashes: list[tuple[Path, str]] = [(train_model_npz, TRAIN_MODEL_SHA256)]
    if EXECUTION_MODE in ("BUILD_SUBSTRATE", "FINALIZE_TRAIN"):
        required_hashes.append((train_csv, TRAIN_SHA256))
    if EXECUTION_MODE == "BUILD_SUBSTRATE":
        required_hashes.append((dictionary_npz, DICTIONARY_SHA256))

    for path, expected in required_hashes:
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"input lock mismatch for {path}: {observed} != {expected}"
            )
        print(f"INPUT LOCK {path} sha256={observed}")

    if EXECUTION_MODE in ("RUN_REPLICATE", "FINALIZE_TRAIN"):
        manifest_file = substrate_dir / "motif_train_substrate_manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(
                f"canonical substrate manifest missing at {manifest_file}"
            )

    if EXECUTION_MODE == "FINALIZE_TRAIN":
        if not replicates_dir.is_dir():
            raise FileNotFoundError(
                f"replicate artifact directory missing at {replicates_dir}"
            )

    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing existing output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    command = [sys.executable, "-m", "pixel_relational_motif_e0.motif_train_runner"]
    if EXECUTION_MODE == "BUILD_SUBSTRATE":
        command += [
            "build-substrate",
            "--train-csv",
            str(train_csv),
            "--dictionary-npz",
            str(dictionary_npz),
            "--train-model-npz",
            str(train_model_npz),
            "--output-dir",
            str(OUTPUT_DIR),
        ]
    elif EXECUTION_MODE == "RUN_REPLICATE":
        command += [
            "run-replicate",
            "--arm",
            str(STABILITY_ARM),
            "--replicate-id",
            str(STABILITY_REPLICATE_ID),
            "--substrate-dir",
            str(substrate_dir),
            "--train-model-npz",
            str(train_model_npz),
            "--output-dir",
            str(OUTPUT_DIR),
        ]
    elif EXECUTION_MODE == "FINALIZE_TRAIN":
        command += [
            "finalize-train",
            "--train-csv",
            str(train_csv),
            "--substrate-dir",
            str(substrate_dir),
            "--replicates-dir",
            str(replicates_dir),
            "--output-dir",
            str(OUTPUT_DIR),
        ]
    else:
        raise ValueError(f"unknown execution mode: {EXECUTION_MODE}")

    environment = os.environ.copy()
    environment["MOTIF_SCIENTIFIC_SHA"] = SCIENTIFIC_SHA
    subprocess.run(command, cwd=REPO_DIR, env=environment, check=True)

    inventory = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(OUTPUT_DIR.iterdir())
        if path.is_file()
    }
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "unit_id": UNIT_ID,
                "execution_mode": EXECUTION_MODE,
                "outputs": inventory,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_unit())
