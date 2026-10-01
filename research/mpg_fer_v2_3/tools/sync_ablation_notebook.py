"""Generate the one source-locked Kaggle notebook for Issue #101."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
PACKAGE_NAMES = ("mpg_fer_v2_3", "mpg_fer_table_vi")
NOTEBOOK = ROOT / "notebooks" / "MPG_FER_Table_VI_Ablation_Kaggle_T4.ipynb"
IMPLEMENTATION_COMMIT_FILE = ROOT / "ABLATION_IMPLEMENTATION_COMMIT.txt"


def source_payload() -> tuple[dict[str, str], str]:
    encoded = {}
    digest = hashlib.sha256()
    for package_name in PACKAGE_NAMES:
        for path in sorted((SOURCE_ROOT / package_name).glob("*.py")):
            data = path.read_bytes().replace(b"\r\n", b"\n")
            relative = f"{package_name}/{path.name}"
            encoded[relative] = base64.b64encode(data).decode("ascii")
            digest.update(relative.encode("utf-8"))
            digest.update(data)
    return encoded, digest.hexdigest()


def _cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def implementation_commit() -> str:
    if not IMPLEMENTATION_COMMIT_FILE.is_file():
        raise RuntimeError("ABLATION_IMPLEMENTATION_COMMIT.txt is absent")
    value = IMPLEMENTATION_COMMIT_FILE.read_text(encoding="utf-8").strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("Invalid ablation implementation commit lock")
    return value


def build_notebook(implementation_sha: str | None = None) -> dict:
    implementation_sha = implementation_sha or implementation_commit()
    embedded, source_sha = source_payload()
    cells = [
        _cell(
            "markdown",
            "# MPG-FER seven-run ablation execution (Issue #106)\n\n"
            "One canonical source-locked notebook for all seven variants and FULL. "
            "Train and checkpoint selection use only Train/PublicTest. After training "
            "completes and the selected checkpoint SHA-256 is frozen, the notebook "
            "evaluates PrivateTest once for final paper reporting.\n",
        ),
        _cell(
            "code",
            """from pathlib import Path
import hashlib
import json
import os
import shutil

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
ABLATION_MODE = secrets.get_secret("MPG_FER_ABLATION_MODE")
RUN_ID = secrets.get_secret("MPG_FER_ABLATION_RUN_ID")
RESUME_MODE = secrets.get_secret("MPG_FER_ABLATION_RESUME_MODE") or "fresh"
SEGMENT_NUMBER = int(secrets.get_secret("MPG_FER_ABLATION_SEGMENT_NUMBER") or "1")
if not ABLATION_MODE or not RUN_ID:
    raise RuntimeError("ABLATION_MODE and RUN_ID secrets are required")
if RESUME_MODE not in {"fresh", "required"}:
    raise RuntimeError("RESUME_MODE must be fresh or required")
OUTPUT_DIR = Path("/kaggle/working/mpg-fer-table-vi") / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print({"ABLATION_MODE": ABLATION_MODE, "RUN_ID": RUN_ID, "RESUME_MODE": RESUME_MODE, "OUTPUT_DIR": str(OUTPUT_DIR)})
""",
        ),
        _cell(
            "code",
            f"""import base64
import sys

EMBEDDED_SOURCES = {embedded!r}
EXPECTED_SOURCE_TREE_SHA256 = {source_sha!r}
SOURCE_ROOT = Path("/kaggle/working/mpg_fer_table_vi_source")
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
source_digest = hashlib.sha256()
for name in sorted(EMBEDDED_SOURCES):
    payload = base64.b64decode(EMBEDDED_SOURCES[name])
    source_digest.update(name.encode("utf-8"))
    source_digest.update(payload)
    target = SOURCE_ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
if source_digest.hexdigest() != EXPECTED_SOURCE_TREE_SHA256:
    raise RuntimeError("ABLATION_SOURCE_LOCK_MISMATCH")
sys.path.insert(0, str(SOURCE_ROOT))
os.environ["MPG_FER_ARCHITECTURE_BASE_COMMIT"] = "232e7a9f09251e7c3353684d34351356bd2b023b"
os.environ["MPG_FER_ABLATION_IMPLEMENTATION_COMMIT"] = {implementation_sha!r}
os.environ["MPG_FER_SOURCE_GIT_COMMIT"] = {implementation_sha!r}
print("source lock verified", EXPECTED_SOURCE_TREE_SHA256)
""",
        ),
        _cell(
            "code",
            """from mpg_fer_table_vi.model import AblationMode
from mpg_fer_table_vi.protocol import (
    config_from_final_recipe,
    validate_ablation_data_paths,
    validate_final_recipe_lock,
    validate_kaggle_mounted_input_contract,
)
from mpg_fer_table_vi.train import (
    run_ablation_micro_overfit_preflight,
    run_ablation_training,
)
from mpg_fer_v2_3.checkpoint import sha256_file

INPUT_ROOT = Path("/kaggle/input")
mounted_input_contract = validate_kaggle_mounted_input_contract(INPUT_ROOT)
print(mounted_input_contract)

def exactly_one(filename):
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}, found {len(matches)}")
    return matches[0]

recipe_path = exactly_one("FINAL_RECIPE_LOCK.json")
design_lock_path = exactly_one("ABLATION_DESIGN_LOCK.json")

resume_path = None
resume_sha256 = None
if RESUME_MODE == "required":
    candidates = [
        path for path in INPUT_ROOT.rglob("resume_latest.pt")
        if any(part.startswith("mpg-fer-table-vi-resume-") for part in path.parts)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one explicit ablation resume, found {len(candidates)}")
    resume_path = candidates[0]
    metadata_path = resume_path.with_name("resume_latest.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    resume_sha256 = metadata["checkpoint_sha256"]
    if sha256_file(resume_path) != resume_sha256:
        raise RuntimeError("ABLATION_RESUME_SHA256_MISMATCH")

config = config_from_final_recipe(
    recipe_path,
    mode=AblationMode(ABLATION_MODE),
    run_id=RUN_ID,
    output_dir=OUTPUT_DIR,
    resume_path=resume_path,
    segment_number=SEGMENT_NUMBER,
)
# Authorization is checked before any FER CSV path is resolved.
authorization = validate_final_recipe_lock(recipe_path, design_lock_path, config)
train_csv = exactly_one("train.csv")
public_csv = exactly_one("val.csv")
private_csv = exactly_one("test.csv")
validate_ablation_data_paths(train_csv, public_csv, private_csv)
print({
    "recipe_sha256": authorization["sha256"],
    "train_csv": str(train_csv),
    "public_csv": str(public_csv),
    "private_csv": str(private_csv),
    "resume_path": None if resume_path is None else str(resume_path),
})
""",
        ),
        _cell(
            "code",
            """if RESUME_MODE == "fresh":
    preflight = run_ablation_micro_overfit_preflight(train_csv, config)
else:
    preflight = {"resume_compatibility_passed": True}
result = run_ablation_training(
    train_csv,
    public_csv,
    private_csv,
    OUTPUT_DIR,
    preflight,
    config=config,
    final_recipe_lock=recipe_path,
    design_lock=design_lock_path,
    resume_path=resume_path,
    resume_sha256=resume_sha256,
)
print(json.dumps(result, indent=2, default=str))
""",
        ),
        _cell(
            "code",
            """archive_base = Path("/kaggle/working") / f"{RUN_ID}-artifacts"
archive = shutil.make_archive(str(archive_base), "zip", root_dir=OUTPUT_DIR)
print("artifact_zip", archive)
""",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "mpg_fer_ablation_issue": 106,
            "source_tree_sha256": source_sha,
            "architecture_base_commit": "232e7a9f09251e7c3353684d34351356bd2b023b",
            "ablation_implementation_commit": implementation_sha,
            "private_test_permitted": True,
            "private_test_use": "one_shot_final_reporting_after_checkpoint_sha_freeze",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(implementation_sha: str | None = None) -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(
        json.dumps(build_notebook(implementation_sha), indent=1) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
