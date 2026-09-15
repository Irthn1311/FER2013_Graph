"""Kaggle entry point for the source-locked, representation-only PGM M0 substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys


REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
SCIENTIFIC_SHA = "8910e9757210674a1504837c5b8bc66e2da9208d"
E0R_OCCURRENCES_SHA256 = "30f1a5b642af2ecdfc29ae73960fb01f90c391964db845bd4b33cf3c017f7fa9"
E0R_R1_RESULTS_SHA256 = "b6675ce694f5a607cfea07abb3ed1065753f42ee848f7595d1cb8e682361a0ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_lock() -> tuple[Path, str]:
    wrapper_project = Path(__file__).resolve().parents[1]
    wrapper_sha = subprocess.check_output(["git", "-C", str(wrapper_project), "rev-parse", "HEAD"], text=True).strip()
    science = Path("/kaggle/working/FER2013_Graph_M0_SCIENCE")
    if science.exists():
        shutil.rmtree(science)
    subprocess.run(["git", "clone", REPO_URL, str(science)], check=True)
    subprocess.run(["git", "-C", str(science), "checkout", "--detach", SCIENTIFIC_SHA], check=True)
    if subprocess.check_output(["git", "-C", str(science), "rev-parse", "HEAD"], text=True).strip() != SCIENTIFIC_SHA:
        raise RuntimeError("M0 scientific source-lock mismatch")
    subprocess.run(["git", "-C", str(science), "diff", "--quiet"], check=True)
    scientific_paths = (
        "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/m0_config.py",
        "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/m0_substrate.py",
        "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/m0_models.py",
        "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/m0_train.py",
        "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/m0_aggregate.py",
    )
    changed = subprocess.check_output(["git", "-C", str(wrapper_project), "diff", "--name-only", SCIENTIFIC_SHA, wrapper_sha, "--", *scientific_paths], text=True).strip()
    if changed:
        raise RuntimeError(f"scientific M0 files changed after lock: {changed}")
    package = science / "research/pixel_relational_motif_e0"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package / "src")
    test = subprocess.run([sys.executable, "-m", "pytest", str(package / "tests"), "-q"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(test.stdout, flush=True)
    if test.returncode:
        raise RuntimeError("pre-data M0 tests failed")
    sys.path.insert(0, str(package / "src"))
    return science, wrapper_sha


def _r1_root() -> Path:
    candidates = (
        Path("/kaggle/input/pgm-e0r-v538-r1-frozen"),
        Path("/kaggle/input/datasets/irthn1311/pgm-e0r-v538-r1-frozen"),
        Path("/kaggle/input/datasets/nuyntai/pgm-e0r-v538-r1-frozen"),
    )
    matches = [root for root in candidates if (root / "e0r_occurrences_actual.npz").is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"need exactly one canonical R1 mount: {matches}")
    return matches[0]


def main() -> int:
    _science, wrapper_sha = _source_lock()
    root = _r1_root()
    occurrences, results = root / "e0r_occurrences_actual.npz", root / "e0r_r1_results.npz"
    if sha256(occurrences) != E0R_OCCURRENCES_SHA256 or sha256(results) != E0R_R1_RESULTS_SHA256:
        raise RuntimeError("frozen R1 input hash mismatch")
    from pixel_relational_motif_e0.m0_substrate import build_substrate

    output = Path("/kaggle/working/outputs/pgm_m0_substrate")
    manifest = build_substrate(occurrences, results, output, m0_scientific_sha=SCIENTIFIC_SHA)
    environment = {
        "python": sys.version, "platform": platform.platform(), "scientific_sha": SCIENTIFIC_SHA,
        "execution_wrapper_sha": wrapper_sha, "private_test_read": False, "public_labels_read": False,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (output / "source_lock_provenance.json").write_text(json.dumps({
        "issue": 82, "scientific_sha": SCIENTIFIC_SHA, "execution_wrapper_sha": wrapper_sha,
        "occurrences_sha256": E0R_OCCURRENCES_SHA256, "r1_results_sha256": E0R_R1_RESULTS_SHA256,
        "representation_only": True, "public_labels_read": False, "private_test_read": False,
    }, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
