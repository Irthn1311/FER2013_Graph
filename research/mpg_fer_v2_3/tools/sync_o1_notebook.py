"""Generate the one source-locked MPG-FER O1 Wave-1 notebook."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
PACKAGE_NAMES = ("mpg_fer_v2_3", "mpg_fer_o1")
NOTEBOOK = ROOT / "notebooks" / "MPG_FER_O1_Wave1_Kaggle_T4.ipynb"
IMPLEMENTATION_COMMIT_FILE = ROOT / "O1_IMPLEMENTATION_COMMIT.txt"


def implementation_commit() -> str:
    if not IMPLEMENTATION_COMMIT_FILE.is_file():
        raise RuntimeError("O1_IMPLEMENTATION_COMMIT.txt is absent")
    value = IMPLEMENTATION_COMMIT_FILE.read_text(encoding="utf-8").strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("Invalid O1 implementation commit lock")
    return value


def source_payload() -> tuple[dict[str, str], str]:
    encoded: dict[str, str] = {}
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
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build_notebook(implementation_sha: str | None = None) -> dict:
    implementation_sha = implementation_sha or implementation_commit()
    embedded, source_sha = source_payload()
    cells = [
        _cell(
            "markdown",
            "# MPG-FER O1 / Wave 1 (Issue #103)\n\n"
            "One canonical source-locked notebook for all 14 registered jobs. "
            "It refuses mounted PrivateTest markers before resolving any input "
            "and refuses execution until the reviewed design lock authorizes Wave 1.\n",
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
CONFIG_ID = secrets.get_secret("MPG_FER_O1_CONFIG_ID")
RUN_ID = secrets.get_secret("MPG_FER_O1_RUN_ID")
RESUME_MODE = secrets.get_secret("MPG_FER_O1_RESUME_MODE") or "fresh"
SEGMENT_NUMBER = int(secrets.get_secret("MPG_FER_O1_SEGMENT_NUMBER") or "1")
if not CONFIG_ID or not RUN_ID:
    raise RuntimeError("MPG_FER_O1_CONFIG_ID and MPG_FER_O1_RUN_ID are required")
if RESUME_MODE not in {"fresh", "required"}:
    raise RuntimeError("RESUME_MODE must be fresh or required")
OUTPUT_DIR = Path("/kaggle/working/mpg-fer-o1") / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print({"CONFIG_ID": CONFIG_ID, "RUN_ID": RUN_ID, "RESUME_MODE": RESUME_MODE})
""",
        ),
        _cell(
            "code",
            f"""import base64
import sys

EMBEDDED_SOURCES = {embedded!r}
EXPECTED_SOURCE_TREE_SHA256 = {source_sha!r}
SOURCE_ROOT = Path("/kaggle/working/mpg_fer_o1_source")
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
digest = hashlib.sha256()
for name in sorted(EMBEDDED_SOURCES):
    payload = base64.b64decode(EMBEDDED_SOURCES[name])
    digest.update(name.encode("utf-8"))
    digest.update(payload)
    target = SOURCE_ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
if digest.hexdigest() != EXPECTED_SOURCE_TREE_SHA256:
    raise RuntimeError("O1_SOURCE_LOCK_MISMATCH")
sys.path.insert(0, str(SOURCE_ROOT))
os.environ["MPG_FER_ARCHITECTURE_BASE_COMMIT"] = "232e7a9f09251e7c3353684d34351356bd2b023b"
os.environ["MPG_FER_O1_IMPLEMENTATION_COMMIT"] = {implementation_sha!r}
os.environ["MPG_FER_SOURCE_GIT_COMMIT"] = {implementation_sha!r}
print("O1 source lock verified", EXPECTED_SOURCE_TREE_SHA256)
""",
        ),
        _cell(
            "code",
            """from mpg_fer_o1.protocol import (
    O1_REGISTRY,
    o1_source_tree_hash,
    validate_design_lock,
    validate_mounted_input_firewall,
)
from mpg_fer_o1.train import run_o1_training
from mpg_fer_v2_3.checkpoint import sha256_file
from mpg_fer_v2_3.train import run_micro_overfit_preflight

INPUT_ROOT = Path("/kaggle/input")
mounted_firewall = validate_mounted_input_firewall(INPUT_ROOT)
print(mounted_firewall)
if CONFIG_ID not in O1_REGISTRY:
    raise RuntimeError("CONFIG_ID is not one of the 14 immutable O1 jobs")

def exactly_one(filename):
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}, found {len(matches)}")
    return matches[0]

design_lock_path = exactly_one("O1_HPO_DESIGN_LOCK.json")
design = json.loads(design_lock_path.read_text(encoding="utf-8"))
validate_design_lock(
    design_lock_path,
    expected_source_sha256=o1_source_tree_hash(SOURCE_ROOT),
    expected_notebook_sha256=design["notebook_sha256"],
)

resume_path = None
resume_sha256 = None
if RESUME_MODE == "required":
    candidates = [
        path for path in INPUT_ROOT.rglob("resume_latest.pt")
        if any(part.startswith("mpg-fer-o1-resume-") for part in path.parts)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one explicit O1 resume, found {len(candidates)}")
    resume_path = candidates[0]
    metadata = json.loads(resume_path.with_name("resume_latest.json").read_text(encoding="utf-8"))
    resume_sha256 = metadata["checkpoint_sha256"]
    if sha256_file(resume_path) != resume_sha256:
        raise RuntimeError("O1_RESUME_SHA256_MISMATCH")

train_csv = exactly_one("train.csv")
public_csv = exactly_one("val.csv")
""",
        ),
        _cell(
            "code",
            """if RESUME_MODE == "fresh":
    from mpg_fer_o1.protocol import resolve_o1_config
    preflight = run_micro_overfit_preflight(train_csv, resolve_o1_config(CONFIG_ID))
else:
    preflight = {"resume_compatibility_passed": True}
result = run_o1_training(
    train_csv,
    public_csv,
    OUTPUT_DIR,
    preflight,
    config_id=CONFIG_ID,
    design_lock=design_lock_path,
    notebook_sha256=design["notebook_sha256"],
    run_id=RUN_ID,
    resume_path=resume_path,
    resume_sha256=resume_sha256,
    segment_number=SEGMENT_NUMBER,
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
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "mpg_fer_o1_issue": 103,
            "source_tree_sha256": source_sha,
            "architecture_base_commit": "232e7a9f09251e7c3353684d34351356bd2b023b",
            "o1_implementation_commit": implementation_sha,
            "private_test_permitted": False,
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
