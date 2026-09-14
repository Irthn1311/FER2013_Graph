"""Kaggle entry point for one frozen PGM E0.R checkpoint shard."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
SCIENTIFIC_SHA = "671e3c2f69607778f08a923026e76744561e13b2"
TRAIN_SHA256 = "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8"
PUBLIC_SHA256 = "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08"
R1_OCCURRENCES_SHA256 = "30f1a5b642af2ecdfc29ae73960fb01f90c391964db845bd4b33cf3c017f7fa9"
R1_DIAGNOSTICS_SHA256 = "e1b3d6c71ef39b96b8431ed6c9bdf0039bccc47a63cc71e98c82790472adb99f"
R1_RESULTS_SHA256 = "b6675ce694f5a607cfea07abb3ed1065753f42ee848f7595d1cb8e682361a0ed"
R1_MANIFEST_SHA256 = "f1ea79d8c31f084a4cbf76161c348985edffd72e0eee7315b8688772c98d564d"
PROTECTED_SCIENTIFIC_PATHS = (
    "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/e0r_occurrence.py",
    "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/e0r_geometry.py",
    "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/e0r_runner.py",
    "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/probe.py",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def heartbeat(label: str, interval_seconds: int = 180):
    stop = threading.Event()
    started = time.time()

    def worker():
        while not stop.wait(interval_seconds):
            print(f"[heartbeat] {label}: {(time.time() - started) / 60:.1f} min", flush=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
        print(f"[heartbeat] {label}: complete {(time.time() - started) / 60:.1f} min", flush=True)


def _load_execution_module(wrapper_project: Path, science_project: Path):
    science_src = science_project / "research/pixel_relational_motif_e0/src"
    sys.path.insert(0, str(science_src))
    import pixel_relational_motif_e0

    imported = Path(pixel_relational_motif_e0.__file__).resolve()
    if science_src.resolve() not in imported.parents:
        raise RuntimeError(f"scientific import isolation violation: {imported}")
    module_path = (
        wrapper_project
        / "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/e0r_continuation.py"
    )
    module_name = "pixel_relational_motif_e0.e0r_continuation_execution"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load execution-only continuation wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if module.SCIENTIFIC_SHA != SCIENTIFIC_SHA:
        raise RuntimeError("execution wrapper scientific SHA mismatch")
    return module


def execute_shard(shard_name: str) -> dict:
    working = Path("/kaggle/working")
    wrapper_project = Path(__file__).resolve().parents[1]
    wrapper_sha = subprocess.check_output(
        ["git", "-C", str(wrapper_project), "rev-parse", "HEAD"], text=True
    ).strip()
    science_project = working / "FER2013_Graph_E0R_SCIENCE"
    if science_project.exists():
        shutil.rmtree(science_project)
    subprocess.run(["git", "clone", REPO_URL, str(science_project)], check=True)
    subprocess.run(["git", "-C", str(science_project), "checkout", "--detach", SCIENTIFIC_SHA], check=True)
    science_head = subprocess.check_output(
        ["git", "-C", str(science_project), "rev-parse", "HEAD"], text=True
    ).strip()
    if science_head != SCIENTIFIC_SHA:
        raise RuntimeError(f"scientific source lock mismatch: {science_head}")
    subprocess.run(["git", "-C", str(science_project), "diff", "--quiet"], check=True)
    protected_diff = subprocess.check_output(
        ["git", "-C", str(wrapper_project), "diff", "--name-only", SCIENTIFIC_SHA, wrapper_sha, "--", *PROTECTED_SCIENTIFIC_PATHS],
        text=True,
    ).strip()
    if protected_diff:
        raise RuntimeError(f"protected scientific files changed: {protected_diff}")

    package = wrapper_project / "research/pixel_relational_motif_e0"
    test_env = os.environ.copy()
    test_env["PYTHONPATH"] = str(package / "src")
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", str(package / "tests"), "-q"],
        env=test_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(test_result.stdout, flush=True)
    if test_result.returncode != 0:
        raise RuntimeError(f"pre-data continuation tests failed: {test_result.returncode}")
    print("Pre-data continuation tests PASS", flush=True)

    continuation = _load_execution_module(wrapper_project, science_project)
    if shard_name not in continuation.SHARD_PLAN:
        raise ValueError(f"unknown frozen shard {shard_name}")
    plan = continuation.SHARD_PLAN[shard_name]

    fer_root = Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split")
    train_csv = fer_root / "train.csv"
    public_csv = fer_root / "val.csv"
    r1_root = Path("/kaggle/input/pgm-e0r-v538-r1-frozen")
    occurrences_path = r1_root / "e0r_occurrences_actual.npz"
    diagnostics_path = r1_root / "e0r_r1_occurrence_diagnostics.npz"
    results_path = r1_root / "e0r_r1_results.npz"
    r1_manifest_path = r1_root / "e0r_r1_frozen_manifest.json"
    expected = {
        train_csv: TRAIN_SHA256,
        public_csv: PUBLIC_SHA256,
        occurrences_path: R1_OCCURRENCES_SHA256,
        diagnostics_path: R1_DIAGNOSTICS_SHA256,
        results_path: R1_RESULTS_SHA256,
        r1_manifest_path: R1_MANIFEST_SHA256,
    }
    for path, expected_sha in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"input SHA mismatch for {path}: {actual_sha}")
        print(f"INPUT LOCK {path} sha256={actual_sha}", flush=True)
    print("No PrivateTest path is configured or inspected.", flush=True)

    output_dir = working / f"outputs/pixel_relational_motif_e0r_r2_shard_{shard_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import scipy
    import sklearn

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "scientific_sha": SCIENTIFIC_SHA,
        "execution_wrapper_sha": wrapper_sha,
        "shard_name": shard_name,
    }
    environment_path = output_dir / "environment.json"
    environment_path.write_text(json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8")
    provenance = {
        "issue": 80,
        "classification": "execution-only continuation/checkpoint wrapper",
        "scientific_sha": SCIENTIFIC_SHA,
        "scientific_head": science_head,
        "execution_wrapper_sha": wrapper_sha,
        "protected_scientific_diff_empty": True,
        "shard_name": shard_name,
        "run_actual": bool(plan["actual"]),
        "seeds": list(plan["seeds"]),
        "private_test_read": False,
        "scientific_verdict_written": False,
    }
    provenance_path = output_dir / "source_lock_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    with heartbeat(f"PGM E0.R R2 checkpoint shard {shard_name}"):
        manifest = continuation.run_r2_shard(
            occurrences_path,
            results_path,
            output_dir,
            execution_wrapper_sha=wrapper_sha,
            shard_name=shard_name,
            run_actual=bool(plan["actual"]),
            seeds=tuple(plan["seeds"]),
        )
    manifest_path = output_dir / f"e0r_r2_shard_{shard_name}_manifest.json"
    manifest["supporting_artifacts"] = {
        environment_path.name: {"sha256": sha256(environment_path), "bytes": environment_path.stat().st_size},
        provenance_path.name: {"sha256": sha256(provenance_path), "bytes": provenance_path.stat().st_size},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"shard": shard_name, "fits": manifest["fits"]}, indent=2), flush=True)
    print("Shard complete. No R2 scientific verdict and no M0 execution.", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True, choices=("A", "B", "C", "D"))
    args = parser.parse_args()
    execute_shard(args.shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
