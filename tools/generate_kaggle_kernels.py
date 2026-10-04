"""Generate self-contained Kaggle kernels for cumulative ablation ladder A0..A6."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
V23_SOURCE_ROOT = ROOT / "research" / "mpg_fer_v2_3" / "src"
PACKAGES = [
    (SOURCE_ROOT / "mpg_fer_cumulative_ablation7", "mpg_fer_cumulative_ablation7"),
    (V23_SOURCE_ROOT / "mpg_fer_table_vi", "mpg_fer_table_vi"),
    (V23_SOURCE_ROOT / "mpg_fer_v2_3", "mpg_fer_v2_3"),
]


def encode_sources() -> tuple[dict[str, str], str]:
    encoded = {}
    digest = hashlib.sha256()
    for src_dir, pkg_name in PACKAGES:
        for p in sorted(src_dir.glob("*.py")):
            rel_name = f"{pkg_name}/{p.name}"
            data = p.read_bytes().replace(b"\r\n", b"\n")
            encoded[rel_name] = base64.b64encode(data).decode("ascii")
            digest.update(rel_name.encode("utf-8"))
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


def build_kernel_notebook(mode_str: str, slug: str, encoded_sources: dict, source_sha: str) -> dict:
    cells = [
        _cell(
            "markdown",
            f"# MPG-FER Cumulative Ablation - Configuration {mode_str}\n\n"
            f"Automated standalone execution for {mode_str}.\n"
            "Evaluates Train/PublicTest during training and selects best checkpoint via EMA TTA Accuracy. "
            "Evaluates PrivateTest exactly once in canonical FP32 after checkpoint freeze.\n",
        ),
        _cell(
            "code",
            f"""import os
from pathlib import Path

CUMULATIVE_MODE = {mode_str!r}
RUN_ID = {slug!r}
OUTPUT_DIR = Path("/kaggle/working/mpg_fer_cumulative_ablation7") / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print({{"CUMULATIVE_MODE": CUMULATIVE_MODE, "RUN_ID": RUN_ID, "OUTPUT_DIR": str(OUTPUT_DIR)}})
""",
        ),
        _cell(
            "code",
            f"""import base64
import hashlib
import sys

EMBEDDED_SOURCES = {encoded_sources!r}
EXPECTED_SOURCE_TREE_SHA256 = {source_sha!r}
SOURCE_ROOT = Path("/kaggle/working/source_root")
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

digest = hashlib.sha256()
for name, b64payload in sorted(EMBEDDED_SOURCES.items()):
    payload = base64.b64decode(b64payload)
    digest.update(name.encode("utf-8"))
    digest.update(payload)
    target = SOURCE_ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

actual_sha = digest.hexdigest()
if actual_sha != EXPECTED_SOURCE_TREE_SHA256:
    raise RuntimeError(f"Source SHA mismatch: expected {{EXPECTED_SOURCE_TREE_SHA256}}, got {{actual_sha}}")

sys.path.insert(0, str(SOURCE_ROOT))
print("Source unpacked and verified successfully:", actual_sha)
""",
        ),
        _cell(
            "code",
            """from pathlib import Path
from mpg_fer_cumulative_ablation7.model import CumulativeAblationMode, CumulativeAblationMPGFER
from mpg_fer_cumulative_ablation7.protocol import validate_all_splits, CANONICAL_DATASET_HASHES
from mpg_fer_cumulative_ablation7.train import run_cumulative_training_job

INPUT_ROOT = Path("/kaggle/input")
def find_unique_csv(filename):
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly 1 {filename}, found {len(matches)}: {matches}")
    return matches[0]

train_csv = find_unique_csv("train.csv")
val_csv = find_unique_csv("val.csv")
test_csv = find_unique_csv("test.csv")

print("Found dataset CSVs:")
print("  train:", train_csv)
print("  val:  ", val_csv)
print("  test: ", test_csv)

# Fail closed dataset identity verification
audit = validate_all_splits(train_csv, val_csv, test_csv)
print("Dataset validation audit PASS:", audit)
""",
        ),
        _cell(
            "code",
            """# Execute end-to-end training and evaluation
result = run_cumulative_training_job(
    mode=CumulativeAblationMode(CUMULATIVE_MODE),
    run_id=RUN_ID,
    train_csv=train_csv,
    val_csv=val_csv,
    test_csv=test_csv,
    output_dir=OUTPUT_DIR,
)
print("Job completed successfully!")
print("Selected Best Epoch:", result["best_epoch"])
print("Checkpoint SHA256:", result["checkpoint_sha256"])
print("Public TTA Accuracy:", result["public_metrics"]["tta"]["accuracy"])
print("Private TTA Accuracy:", result["private_metrics"]["tta"]["accuracy"])
""",
        ),
        _cell(
            "code",
            """import shutil
import json

# Write final result json at root of output
res_path = Path("/kaggle/working") / f"{CUMULATIVE_MODE}_RESULT.json"
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print("Wrote summary result to", res_path)
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
            "gpu": True,
            "internet": False,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    encoded_sources, source_sha = encode_sources()
    print(f"Encoded {len(encoded_sources)} source files. Digest: {source_sha}")

    notebooks_dir = ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    configs = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    for mode in configs:
        slug = f"mpg-fer-cumabl7-opus-{mode.lower()}"
        nb = build_kernel_notebook(mode, slug, encoded_sources, source_sha)
        nb_path = notebooks_dir / f"{slug}.ipynb"
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Generated notebook:", nb_path)


if __name__ == "__main__":
    main()
