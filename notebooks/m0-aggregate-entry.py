"""Kaggle entry point for the source-locked zero-fit PGM M0 aggregator."""

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
R1_RESULTS_SHA256 = "b6675ce694f5a607cfea07abb3ed1065753f42ee848f7595d1cb8e682361a0ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", action="append", required=True, type=Path)
    parser.add_argument("--substrate-root", required=True, type=Path)
    parser.add_argument("--r1-results", required=True, type=Path)
    parser.add_argument("--expected-wrapper-sha", required=True)
    args = parser.parse_args()
    wrapper = Path(__file__).resolve().parents[1]
    actual_wrapper_sha = subprocess.check_output(["git", "-C", str(wrapper), "rev-parse", "HEAD"], text=True).strip()
    if actual_wrapper_sha != args.expected_wrapper_sha:
        raise RuntimeError("aggregation wrapper source-lock mismatch")
    science = Path("/kaggle/working/FER2013_Graph_M0_SCIENCE")
    if science.exists(): shutil.rmtree(science)
    subprocess.run(["git", "clone", REPO_URL, str(science)], check=True)
    subprocess.run(["git", "-C", str(science), "checkout", "--detach", SCIENTIFIC_SHA], check=True)
    package = science / "research/pixel_relational_motif_e0"
    env = os.environ.copy(); env["PYTHONPATH"] = str(package / "src")
    test = subprocess.run([sys.executable, "-m", "pytest", str(package / "tests"), "-q"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(test.stdout, flush=True)
    if test.returncode: raise RuntimeError("pre-data M0 aggregate tests failed")
    sys.path.insert(0, str(package / "src"))
    from pixel_relational_motif_e0.m0_aggregate import aggregate, independent_recompute
    from pixel_relational_motif_e0.m0_substrate import load_substrate
    if sha256(args.r1_results) != R1_RESULTS_SHA256:
        raise RuntimeError("canonical R1 result hash mismatch")
    substrate = load_substrate(args.substrate_root)
    found: dict[str, list[Path]] = {}
    for root in args.prediction_root:
        for path in root.rglob("m0_*_prediction.npz"):
            found.setdefault(path.name, []).append(path)
        for path in root.rglob("m0_[MG]_seed_*.npz"):
            found.setdefault(path.name, []).append(path)
    expected = {"m0_L_prediction.npz", *(f"m0_M_seed_{s}.npz" for s in range(42,47)), *(f"m0_G_seed_{s}.npz" for s in range(42,47))}
    if set(found) != expected or any(len(paths) != 1 for paths in found.values()):
        raise RuntimeError(f"prediction set mismatch: {sorted(found)}")
    with __import__("numpy").load(args.r1_results, allow_pickle=False) as result:
        public_ids = __import__("numpy").asarray(result["public_ids"]).copy()
        public_labels = __import__("numpy").asarray(result["public_labels"]).copy()
    output = Path("/kaggle/working/outputs/pgm_m0_aggregate")
    summary = aggregate(public_labels, public_ids, found["m0_L_prediction.npz"][0], [found[f"m0_M_seed_{s}.npz"][0] for s in range(42,47)], [found[f"m0_G_seed_{s}.npz"][0] for s in range(42,47)], output, scientific_sha=SCIENTIFIC_SHA, wrapper_sha=args.expected_wrapper_sha, substrate_sha=substrate.manifest["substrate_sha256"])
    verification = independent_recompute(output / "m0_results.npz")
    (output / "m0_independent_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    environment = {"python": sys.version, "platform": platform.platform(), "scientific_sha": SCIENTIFIC_SHA, "execution_wrapper_sha": args.expected_wrapper_sha, "model_fits_executed": 0, "private_test_read": False}
    (output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    manifest = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(output.iterdir()) if path.is_file() and path.name != "m0_artifact_manifest.json"}
    (output / "m0_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": summary, "independent_verification": verification}, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
