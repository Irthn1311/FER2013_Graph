"""Kaggle entry point for the zero-fit PGM E0.R R2 aggregation job."""

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


REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
SCIENTIFIC_SHA = "671e3c2f69607778f08a923026e76744561e13b2"
CONTINUATION_WRAPPER_SHA = "4abcdeaefb95d73038d9dab5299407dcb6c534fa"
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
CONTINUATION_PATHS = (
    "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/e0r_continuation.py",
    "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/e0r_aggregate.py",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _science_import(science_project: Path):
    science_src = science_project / "research/pixel_relational_motif_e0/src"
    sys.path.insert(0, str(science_src))
    try:
        import pixel_relational_motif_e0

        imported = Path(pixel_relational_motif_e0.__file__).resolve()
        if science_src.resolve() not in imported.parents:
            raise RuntimeError(f"scientific import isolation violation: {imported}")
        yield
    finally:
        sys.path.remove(str(science_src))


def _load_wrapper_module(wrapper_project: Path, name: str, filename: str):
    path = wrapper_project / "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0" / filename
    qualified = f"pixel_relational_motif_e0.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load execution-only module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    if module.SCIENTIFIC_SHA != SCIENTIFIC_SHA:
        raise RuntimeError(f"execution-only module scientific SHA mismatch: {filename}")
    return module


def _resolve_r1_root() -> Path:
    candidates = (
        Path("/kaggle/input/pgm-e0r-v538-r1-frozen"),
        Path("/kaggle/input/pgm-e0-r-v538-r1-frozen"),
        Path("/kaggle/input/datasets/irthn1311/pgm-e0r-v538-r1-frozen"),
        Path("/kaggle/input/datasets/irthn1311/pgm-e0-r-v538-r1-frozen"),
        Path("/kaggle/input/datasets/nuyntai/pgm-e0r-v538-r1-frozen"),
        Path("/kaggle/input/datasets/nuyntai/pgm-e0-r-v538-r1-frozen"),
    )
    matches = [root for root in candidates if (root / "e0r_occurrences_actual.npz").is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"need exactly one frozen R1 dataset mount; matches={matches}")
    return matches[0]


def _resolve_checkpoints(roots: list[Path]) -> tuple[Path, list[Path]]:
    if len(roots) != 4 or len({root.resolve() for root in roots}) != 4:
        raise RuntimeError("aggregation requires exactly four distinct shard roots")
    found: dict[str, list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("e0r_r2_*.npz"):
            found.setdefault(path.name, []).append(path)
    expected = {"e0r_r2_actual.npz", *(f"e0r_r2_seed_{seed}.npz" for seed in range(42, 62))}
    if set(found) != expected:
        raise RuntimeError(f"checkpoint filename set mismatch: observed={sorted(found)}")
    duplicates = {name: paths for name, paths in found.items() if len(paths) != 1}
    if duplicates:
        raise RuntimeError(f"duplicate checkpoint artifacts: {duplicates}")
    actual = found["e0r_r2_actual.npz"][0]
    seeds = [found[f"e0r_r2_seed_{seed}.npz"][0] for seed in range(42, 62)]
    return actual, seeds


def execute_aggregate(checkpoint_roots: list[Path]) -> dict:
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
    continuation_diff = subprocess.check_output(
        ["git", "-C", str(wrapper_project), "diff", "--name-only", CONTINUATION_WRAPPER_SHA, wrapper_sha, "--", *CONTINUATION_PATHS],
        text=True,
    ).strip()
    if continuation_diff:
        raise RuntimeError(f"continuation wrapper changed after its locked commit: {continuation_diff}")

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

    r1_root = _resolve_r1_root()
    occurrences_path = r1_root / "e0r_occurrences_actual.npz"
    diagnostics_path = r1_root / "e0r_r1_occurrence_diagnostics.npz"
    results_path = r1_root / "e0r_r1_results.npz"
    r1_manifest_path = r1_root / "e0r_r1_frozen_manifest.json"
    for path, expected_sha in {
        occurrences_path: R1_OCCURRENCES_SHA256,
        diagnostics_path: R1_DIAGNOSTICS_SHA256,
        results_path: R1_RESULTS_SHA256,
        r1_manifest_path: R1_MANIFEST_SHA256,
    }.items():
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"frozen R1 input lock mismatch: {path}")
        print(f"INPUT LOCK {path} sha256={expected_sha}", flush=True)

    actual_path, seed_paths = _resolve_checkpoints(checkpoint_roots)
    print("Resolved exactly one actual and seeds 42..61 from four shard roots.", flush=True)
    print("No PrivateTest path is configured or inspected.", flush=True)
    print("This aggregation executes zero model fits.", flush=True)
    with _science_import(science_project):
        _load_wrapper_module(wrapper_project, "e0r_continuation", "e0r_continuation.py")
        aggregate = _load_wrapper_module(wrapper_project, "e0r_aggregate_execution", "e0r_aggregate.py")
        output_dir = working / "outputs/pixel_relational_motif_e0r_r2_aggregate"
        summary = aggregate.aggregate_r2(
            occurrences_path,
            diagnostics_path,
            results_path,
            actual_path,
            seed_paths,
            output_dir,
            execution_wrapper_sha=CONTINUATION_WRAPPER_SHA,
        )
    provenance = {
        "issue": 80,
        "classification": "zero-fit registered R2 aggregation",
        "scientific_sha": SCIENTIFIC_SHA,
        "scientific_head": science_head,
        "execution_wrapper_sha": CONTINUATION_WRAPPER_SHA,
        "orchestration_sha": wrapper_sha,
        "protected_scientific_diff_empty": True,
        "continuation_wrapper_diff_empty": True,
        "checkpoint_roots": [str(root) for root in checkpoint_roots],
        "private_test_read": False,
        "model_fits_executed": 0,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    provenance_path = output_dir / "source_lock_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", action="append", required=True, type=Path)
    args = parser.parse_args()
    execute_aggregate(args.checkpoint_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
