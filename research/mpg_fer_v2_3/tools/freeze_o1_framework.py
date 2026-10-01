"""Freeze MPG-FER O1 source/notebook/design and run implementation preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tools"))

from build_o1_baseline_reference import build_reference  # noqa: E402
from mpg_fer_o1.protocol import (  # noqa: E402
    BASE_COMMIT,
    CHECKPOINT_SELECTION,
    FROZEN_SCIENTIFIC_SOURCE_SHA256,
    O1_CONFIG_ORDER,
    PRIVATE_PATH_MARKERS,
    SCREEN_STOP_EPOCH,
    o1_source_tree_hash,
    registry_document,
    write_json,
)
from mpg_fer_v2_3.checkpoint import sha256_file  # noqa: E402
from mpg_fer_v2_3.model import MPGFER  # noqa: E402
import sync_o1_notebook  # noqa: E402


def _canonical_text_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _frozen_source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((SRC / "mpg_fer_v2_3").glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _compile_notebook(path: Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
            count += 1
    return count


def _strict_checkpoint(checkpoint: Path, expected_sha: str) -> dict[str, Any]:
    actual = sha256_file(checkpoint)
    if actual != expected_sha:
        raise RuntimeError(f"Historical checkpoint SHA mismatch: {actual}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    incompatible = MPGFER().load_state_dict(state, strict=True)
    return {
        "sha256": actual,
        "strict_load_passed": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _source_manifest(notebook: Path, baseline: Path) -> dict[str, Any]:
    selected = [
        *sorted((SRC / "mpg_fer_v2_3").glob("*.py")),
        *sorted((SRC / "mpg_fer_o1").glob("*.py")),
        ROOT / "tools" / "sync_o1_notebook.py",
        ROOT / "tools" / "build_o1_baseline_reference.py",
        ROOT / "tools" / "freeze_o1_framework.py",
        *sorted(ROOT.glob("tests/test_o1_*.py")),
        ROOT / "O1_IMPLEMENTATION_COMMIT.txt",
        baseline,
    ]
    return {
        "schema_version": 1,
        "architecture_base_commit": BASE_COMMIT,
        "o1_implementation_commit": (
            ROOT / "O1_IMPLEMENTATION_COMMIT.txt"
        ).read_text(encoding="utf-8").strip(),
        "frozen_scientific_source_sha256": _frozen_source_hash(),
        "o1_source_tree_sha256": o1_source_tree_hash(SRC),
        "notebook": {
            "path": notebook.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(notebook),
        },
        "baseline_reference": {
            "path": baseline.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(baseline),
        },
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _canonical_text_sha(path),
                "bytes": path.stat().st_size,
            }
            for path in selected
        ],
    }


def _run_tests(checkpoint: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["MPG_FER_O1_BASELINE_CHECKPOINT"] = str(checkpoint.resolve())
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result = {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "output": completed.stdout.strip(),
    }
    if completed.returncode:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-public-results", type=Path, required=True)
    parser.add_argument("--baseline-history-summary", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    (ROOT / "O1_IMPLEMENTATION_COMMIT.txt").write_text(
        implementation_commit + "\n", encoding="utf-8"
    )
    if _frozen_source_hash() != FROZEN_SCIENTIFIC_SOURCE_SHA256:
        raise RuntimeError("Frozen mpg_fer_v2_3 package changed")
    baseline_path = write_json(
        ROOT / "O1_BASELINE_REFERENCE.json",
        build_reference(
            args.baseline_public_results,
            args.baseline_history_summary,
            args.baseline_checkpoint,
        ),
    )
    registry_path = write_json(ROOT / "O1_REGISTRY.json", registry_document())
    sync_o1_notebook.main(implementation_commit)
    notebook = sync_o1_notebook.NOTEBOOK
    compiled_cells = _compile_notebook(notebook)
    manifest = _source_manifest(notebook, baseline_path)
    manifest_path = write_json(ROOT / "O1_SOURCE_MANIFEST.json", manifest)
    design = {
        "schema_version": 1,
        "issue": 103,
        "architecture_base_commit": BASE_COMMIT,
        "o1_implementation_commit": implementation_commit,
        "frozen_scientific_source_sha256": FROZEN_SCIENTIFIC_SOURCE_SHA256,
        "o1_source_tree_sha256": manifest["o1_source_tree_sha256"],
        "notebook_sha256": manifest["notebook"]["sha256"],
        "registry_sha256": sha256_file(registry_path),
        "baseline_reference_sha256": manifest["baseline_reference"]["sha256"],
        "new_config_ids": list(O1_CONFIG_ORDER),
        "historical_control_id": "O1_C0_BASELINE",
        "screen_stop_epoch": SCREEN_STOP_EPOCH,
        "scientific_max_epochs": 120,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "private_test_permitted": False,
        "mounted_input_firewall": {
            "root": "/kaggle/input",
            "forbidden_markers": sorted(PRIVATE_PATH_MARKERS),
            "inspection": "path_names_only_no_file_open",
        },
        "wave1_execution_authorized": False,
        "authorization_blocker": "independent review has not authorized Wave 1",
    }
    design_path = write_json(ROOT / "O1_HPO_DESIGN_LOCK.json", design)
    checkpoint = _strict_checkpoint(
        args.baseline_checkpoint,
        expected_sha=json.loads(baseline_path.read_text(encoding="utf-8"))[
            "checkpoint_sha256"
        ],
    )
    tests = None if args.skip_tests else _run_tests(args.baseline_checkpoint)
    gate_status = "PASS" if tests is not None else "NOT_RUN"
    report = {
        "schema_version": 1,
        "issue": 103,
        "architecture_base_commit": BASE_COMMIT,
        "o1_implementation_commit": implementation_commit,
        "source_tree_sha256": manifest["o1_source_tree_sha256"],
        "notebook_sha256": manifest["notebook"]["sha256"],
        "hpo_design_lock_sha256": sha256_file(design_path),
        "registry_sha256": sha256_file(registry_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "baseline_reference_sha256": sha256_file(baseline_path),
        "baseline_reference_status": "VERIFIED",
        "notebook_code_cells_compiled": compiled_cells,
        "official_checkpoint_compatibility": checkpoint,
        "test_run": tests,
        "gates": {f"O1-T{index}": gate_status for index in range(1, 12)},
        "private_test_accessed": False,
        "fer2013_accessed": False,
        "kaggle_launched": False,
        "wave1_execution_authorized": False,
        "status": (
            "MPG_FER_O1_FRAMEWORK_READY_PENDING_INDEPENDENT_REVIEW"
            if tests is not None
            else "O1_PREFLIGHT_INCOMPLETE"
        ),
    }
    write_json(ROOT / "O1_PREFLIGHT_REPORT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
