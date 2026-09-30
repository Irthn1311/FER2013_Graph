"""Generate Issue #101 locks and execute the release preflight."""

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

from mpg_fer_table_vi.model import (  # noqa: E402
    TABLE_VI_ORDER,
    AblationMPGFER,
    AblationMode,
    registry_document,
)
from mpg_fer_table_vi.protocol import (  # noqa: E402
    BASE_COMMIT,
    CHECKPOINT_SELECTION,
    TABLE_INFERENCE,
    ablation_source_tree_hash,
    write_json,
)
from mpg_fer_v2_3.checkpoint import sha256_file  # noqa: E402

import sync_ablation_notebook  # noqa: E402


FROZEN_V23_SOURCE_SHA256 = (
    "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87"
)
EXPECTED_OFFICIAL_SEED42_CHECKPOINT_SHA256 = (
    "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"
)


def _canonical_text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _compile_notebook(path: Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
        count += 1
    return count


def _source_manifest(notebook: Path) -> dict[str, Any]:
    files = []
    selected = [
        *sorted((SRC / "mpg_fer_v2_3").glob("*.py")),
        *sorted((SRC / "mpg_fer_table_vi").glob("*.py")),
        ROOT / "tools" / "sync_ablation_notebook.py",
        ROOT / "tools" / "freeze_ablation_framework.py",
        *sorted(ROOT.glob("tests/test_ablation_*.py")),
        ROOT / "tests" / "test_multiseed_protocol.py",
        ROOT / "tests" / "test_v23_contract.py",
        ROOT / "ABLATION_IMPLEMENTATION_COMMIT.txt",
    ]
    for path in selected:
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _canonical_text_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "architecture_base_commit": BASE_COMMIT,
        "ablation_implementation_commit": (
            ROOT / "ABLATION_IMPLEMENTATION_COMMIT.txt"
        ).read_text(encoding="utf-8").strip(),
        "ablation_source_tree_sha256": ablation_source_tree_hash(SRC),
        "notebook": {
            "path": notebook.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(notebook),
        },
        "files": files,
    }


def _strict_checkpoint(checkpoint: Path) -> dict[str, Any]:
    digest = sha256_file(checkpoint)
    if digest != EXPECTED_OFFICIAL_SEED42_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Official seed-42 checkpoint SHA mismatch: "
            f"expected {EXPECTED_OFFICIAL_SEED42_CHECKPOINT_SHA256}, got {digest}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    incompatible = AblationMPGFER(mode=AblationMode.FULL).load_state_dict(
        state, strict=True
    )
    return {
        "path": str(checkpoint.resolve()),
        "sha256": digest,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict_load_passed": True,
    }


def _run_pytest(checkpoint: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q"]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["MPG_FER_V23_OFFICIAL_CHECKPOINT"] = str(checkpoint.resolve())
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
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    (ROOT / "ABLATION_IMPLEMENTATION_COMMIT.txt").write_text(
        implementation_commit + "\n", encoding="utf-8"
    )
    registry_path = write_json(ROOT / "ablation_registry.json", registry_document())
    sync_ablation_notebook.main(implementation_commit)
    notebook = sync_ablation_notebook.NOTEBOOK
    compiled_cells = _compile_notebook(notebook)
    manifest = _source_manifest(notebook)
    source_manifest_path = write_json(ROOT / "source_checksum_manifest.json", manifest)
    design = {
        "schema_version": 1,
        "issue": 101,
        "method": "MPG-FER",
        "seed": 42,
        "ablation_split": "PublicTest",
        "table_inference": TABLE_INFERENCE,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "private_test_permitted": False,
        "variant_ids": [
            mode.value for mode in TABLE_VI_ORDER if mode is not AblationMode.FULL
        ],
        "full_control_id": AblationMode.FULL.value,
        "architecture_base_commit": BASE_COMMIT,
        "ablation_implementation_commit": implementation_commit,
        "architecture_provenance": {
            "name": "MPG-FER v2.3",
            "frozen_v2_3_source_sha256": FROZEN_V23_SOURCE_SHA256,
            "parameter_count": 2304528,
            "residual_scale_schedule": [0.5, 0.5, 1.0, 1.0, 1.0],
            "topk_schedule": [8, 16, 16, 16, 24],
        },
        "final_recipe_lock_sha256": None,
        "ablation_source_sha256": manifest["ablation_source_tree_sha256"],
        "notebook_sha256": manifest["notebook"]["sha256"],
        "scientific_training_authorized": False,
        "authorization_blocker": "FINAL_RECIPE_LOCK.json is not yet frozen and independently reviewed",
    }
    design_path = write_json(ROOT / "ABLATION_DESIGN_LOCK.json", design)
    checkpoint = _strict_checkpoint(args.official_checkpoint)
    pytest_result = None if args.skip_tests else _run_pytest(args.official_checkpoint)
    test_evidence = (
        "not run (--skip-tests)" if pytest_result is None else pytest_result["output"]
    )
    gates = {
        "T1_full_parity": "PASS" if pytest_result else "NOT_RUN",
        "T2_shape_contract": "PASS" if pytest_result else "NOT_RUN",
        "T3_forward_backward_optimizer_ema": "PASS" if pytest_result else "NOT_RUN",
        "T4_delta_specific_assertions": "PASS" if pytest_result else "NOT_RUN",
        "T5_official_full_checkpoint_strict_load": "PASS",
        "T6_private_firewall": "PASS" if pytest_result else "NOT_RUN",
        "T7_resume_determinism_full_and_single_scale_12": (
            "PASS" if pytest_result else "NOT_RUN"
        ),
        "T8_canonical_fp32_evaluator": "PASS" if pytest_result else "NOT_RUN",
        "T9_registry_integrity": "PASS" if pytest_result else "NOT_RUN",
    }
    report = {
        "schema_version": 1,
        "issue": 101,
        "architecture_base_commit": BASE_COMMIT,
        "ablation_implementation_commit": implementation_commit,
        "source_tree_sha256": manifest["ablation_source_tree_sha256"],
        "notebook_sha256": manifest["notebook"]["sha256"],
        "design_lock_sha256": sha256_file(design_path),
        "registry_sha256": sha256_file(registry_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "notebook_code_cells_compiled": compiled_cells,
        "official_checkpoint_compatibility": checkpoint,
        "test_run": pytest_result,
        "test_evidence": test_evidence,
        "gates": gates,
        "private_test_accessed": False,
        "fer2013_accessed": False,
        "kaggle_training_launched": False,
        "final_recipe_lock_present": False,
        "scientific_training_authorized": False,
        "status": (
            "MPG_FER_ABLATION_FRAMEWORK_READY_PENDING_INDEPENDENT_REVIEW"
            if pytest_result is not None
            else "PREFLIGHT_INCOMPLETE"
        ),
    }
    write_json(ROOT / "preflight_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
