"""Deterministic generator for Issue #86 Execution Amendment E1 wrappers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import black


SCIENTIFIC_SHA = "16a84b2b36f0d3584afd0a547487c4373dc22128"
BRANCH = "research/motif-qualification-sparsification"
REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
TRAIN_SHA256 = "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8"
DICTIONARY_SHA256 = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"
TRAIN_MODEL_SHA256 = "77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a"


def _script_template(
    *,
    unit_id: str,
    execution_mode: str,
    arm: str | None,
    replicate_id: int | None,
) -> str:
    arm_repr = repr(arm)
    rep_repr = repr(replicate_id)

    return f'''"""Issue #86 Execution Amendment E1 — hard-coded execution unit {unit_id}."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_URL = "{REPO_URL}"
BRANCH = "{BRANCH}"
SCIENTIFIC_SHA = "{SCIENTIFIC_SHA}"
TRAIN_SHA256 = "{TRAIN_SHA256}"
DICTIONARY_SHA256 = "{DICTIONARY_SHA256}"
TRAIN_MODEL_SHA256 = "{TRAIN_MODEL_SHA256}"

UNIT_ID = "{unit_id}"
EXECUTION_MODE = "{execution_mode}"
STABILITY_ARM = {arm_repr}
STABILITY_REPLICATE_ID = {rep_repr}

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
        for filename in {{c.name for c in candidates}}:
            for match in Path("/kaggle/input").glob(f"**/{{filename}}"):
                if match.is_file() and sha256_file(match) == expected_sha:
                    return match
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _resolve_substrate_dir(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "motif_train_substrate_manifest.json").is_file():
            return candidate
    if Path("/kaggle/input").exists():
        for match in Path("/kaggle/input").glob("**/motif_train_substrate_manifest.json"):
            if match.is_file():
                return match.parent
    return candidates[0]


def _resolve_replicates_dir(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("motif_stability_replicate_*.npz")):
            return candidate
    if Path("/kaggle/input").exists():
        for match in Path("/kaggle/input").glob("**/motif_stability_replicate_*.npz"):
            if match.is_file():
                return match.parent
    return candidates[0]


def run_unit() -> int:
    print(
        json.dumps(
            {{
                "unit_id": UNIT_ID,
                "execution_mode": EXECUTION_MODE,
                "arm": STABILITY_ARM,
                "replicate_id": STABILITY_REPLICATE_ID,
                "scientific_source_sha": SCIENTIFIC_SHA,
                "internet_required_for_clone": True,
                "gpu_required": False,
            }},
            indent=2,
            sort_keys=True,
        )
    )

    if REPO_DIR.exists():
        raise FileExistsError(f"refusing pre-existing repository path: {{REPO_DIR}}")

    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)],
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", SCIENTIFIC_SHA], cwd=REPO_DIR, check=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
    ).strip()
    if head != SCIENTIFIC_SHA:
        raise RuntimeError(f"scientific source lock mismatch: {{head}} != {{SCIENTIFIC_SHA}}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_DIR, text=True
    ).strip():
        raise RuntimeError("scientific source worktree is not clean")
    print(f"Detached checkout of scientific source {{SCIENTIFIC_SHA}}: PASS")

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
        Path("/kaggle/input/datasets/nuyntai/pgm-e01-v533-dictionary-crs/e01_dictionary.npz"),
        Path("/kaggle/input/pgm-e01-v533-dictionary-crs/e01_dictionary.npz"),
    )
    train_model_npz_candidates = (
        Path("/kaggle/input/datasets/nuyntai/pgm-crs-train-v2-artifacts/crs_train_model.npz"),
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
            raise ValueError(f"input lock mismatch for {{path}}: {{observed}} != {{expected}}")
        print(f"INPUT LOCK {{path}} sha256={{observed}}")

    if EXECUTION_MODE in ("RUN_REPLICATE", "FINALIZE_TRAIN"):
        manifest_file = substrate_dir / "motif_train_substrate_manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"canonical substrate manifest missing at {{manifest_file}}")

    if EXECUTION_MODE == "FINALIZE_TRAIN":
        if not replicates_dir.is_dir():
            raise FileNotFoundError(f"replicate artifact directory missing at {{replicates_dir}}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing existing output directory: {{OUTPUT_DIR}}")
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
        raise ValueError(f"unknown execution mode: {{EXECUTION_MODE}}")

    environment = os.environ.copy()
    environment["MOTIF_SCIENTIFIC_SHA"] = SCIENTIFIC_SHA
    subprocess.run(command, cwd=REPO_DIR, env=environment, check=True)

    inventory = {{
        path.name: {{"bytes": path.stat().st_size, "sha256": sha256_file(path)}}
        for path in sorted(OUTPUT_DIR.iterdir())
        if path.is_file()
    }}
    print(
        json.dumps(
            {{
                "status": "COMPLETE",
                "unit_id": UNIT_ID,
                "execution_mode": EXECUTION_MODE,
                "outputs": inventory,
            }},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_unit())
'''


def get_unit_definitions() -> list[dict]:
    units = []

    # 1. BUILD_SUBSTRATE
    units.append(
        {
            "unit_id": "BUILD_SUBSTRATE",
            "filename": "kaggle/motif_issue86/generated/motif_build_substrate.py",
            "mode": "BUILD_SUBSTRATE",
            "arm": None,
            "replicate_id": None,
        }
    )

    # 2. Replicates M00..M19
    for i in range(20):
        label = f"M{i:02d}"
        units.append(
            {
                "unit_id": label,
                "filename": f"kaggle/motif_issue86/generated/motif_{label}.py",
                "mode": "RUN_REPLICATE",
                "arm": "M",
                "replicate_id": i,
            }
        )

    # 3. Replicates C00..C19
    for i in range(20):
        label = f"C{i:02d}"
        units.append(
            {
                "unit_id": label,
                "filename": f"kaggle/motif_issue86/generated/motif_{label}.py",
                "mode": "RUN_REPLICATE",
                "arm": "C",
                "replicate_id": i,
            }
        )

    # 4. FINALIZE_TRAIN
    units.append(
        {
            "unit_id": "FINALIZE_TRAIN",
            "filename": "kaggle/motif_issue86/generated/motif_finalize_train.py",
            "mode": "FINALIZE_TRAIN",
            "arm": None,
            "replicate_id": None,
        }
    )

    return units


def generate_all(base_dir: Path | None = None) -> dict:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    generated_dir = base_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    units_def = get_unit_definitions()
    manifest: dict[str, dict] = {}

    for u in units_def:
        raw_code = _script_template(
            unit_id=u["unit_id"],
            execution_mode=u["mode"],
            arm=u["arm"],
            replicate_id=u["replicate_id"],
        )
        script_code = black.format_str(raw_code, mode=black.FileMode())
        script_name = Path(u["filename"]).name
        file_path = generated_dir / script_name
        file_path.write_text(script_code, encoding="utf-8", newline="\n")
        sha256 = hashlib.sha256(script_code.encode("utf-8")).hexdigest()

        manifest[u["unit_id"]] = {
            "unit_id": u["unit_id"],
            "filename": f"kaggle/motif_issue86/generated/{script_name}",
            "mode": u["mode"],
            "arm": u["arm"],
            "replicate_id": u["replicate_id"],
            "scientific_sha": SCIENTIFIC_SHA,
            "sha256": sha256,
        }

    manifest_path = base_dir / "execution_units_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Issue #86 execution units.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory to generate units (defaults to directory of this script).",
    )
    args = parser.parse_args()
    target_dir = args.output_dir or Path(__file__).resolve().parent
    m = generate_all(target_dir)
    print(f"Generated {len(m)} execution units and manifest at {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
