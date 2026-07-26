from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "d16_analysis"
    / "lap_gnn_tensorflow_adamw_arithmetic_closure"
)
GATE = 2e-8


def load_json(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def primitive(step: int, name: str, candidate: str) -> dict[str, str]:
    with (OUTPUT_DIR / "07_tensorflow_primitive_candidates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["step"]) == step
                and row["primitive"] == name
                and row["candidate"] == candidate
            ):
                return row
    raise AssertionError(
        f"Missing primitive row: step={step}, {name}, {candidate}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
