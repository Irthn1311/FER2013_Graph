"""Prepare the fail-closed Issue #105 Wave-1 launch authorization candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from mpg_fer_o1.protocol import (  # noqa: E402
    BASELINE_ID,
    CHECKPOINT_SELECTION,
    FROZEN_SCIENTIFIC_SOURCE_SHA256,
    O1_CONFIG_ORDER,
    O1_REGISTRY,
    SCREEN_STOP_EPOCH,
    o1_source_tree_hash,
    write_json,
)
from mpg_fer_v2_3.checkpoint import sha256_file  # noqa: E402


ISSUE_NUMBER = 105
ISSUE_URL = "https://github.com/Irthn1311/FER2013_Graph/issues/105"
FRAMEWORK_PR_NUMBER = 104
FRAMEWORK_PR_URL = "https://github.com/Irthn1311/FER2013_Graph/pull/104"
REVIEWED_FRAMEWORK_HEAD = "e7ea4f1a969d50b34b7e0930338d92f1263bdb96"
REVIEWED_IMPLEMENTATION_COMMIT = "b4a3563ca284dae5dc6ad5aad7023eda1253590b"
REVIEWED_O1_SOURCE_SHA256 = (
    "800803493415562b3d533a8631d79203a181757220f10bad7c4326387d360595"
)
REVIEWED_NOTEBOOK_SHA256 = (
    "752cec59a824092ab3286f6aa1a37a26abc81916152db04e6cf62b7c3c268a81"
)
REVIEWED_REGISTRY_SHA256 = (
    "6a839be43ca5da9542d6bcd1f6a18d6c3c2e0326898ce59750c9c8560bb968aa"
)
REVIEWED_BASELINE_SHA256 = (
    "1b0aca11d49af9bb9e3ee170753666e0b446b435eecdc273a98e8f95c8c01f4e"
)
REVIEWED_DESIGN_LOCK_SHA256 = (
    "a520c4ca7533d20715f9fdcda7f4b204bdeb8c22e7573a50349878cd5b58282c"
)
AUTHORIZATION_STATE = "CANDIDATE_PENDING_FINAL_REVIEW"
READY_STATUS = (
    "MPG_FER_O1_WAVE1_AUTHORIZATION_CANDIDATE_READY_PENDING_FINAL_REVIEW"
)

NOTEBOOK_PATH = ROOT / "notebooks" / "MPG_FER_O1_Wave1_Kaggle_T4.ipynb"
REGISTRY_PATH = ROOT / "O1_REGISTRY.json"
BASELINE_PATH = ROOT / "O1_BASELINE_REFERENCE.json"
IMPLEMENTATION_LOCK_PATH = ROOT / "O1_IMPLEMENTATION_COMMIT.txt"
DESIGN_LOCK_PATH = ROOT / "O1_HPO_DESIGN_LOCK.json"
AUTHORIZATION_PATH = ROOT / "O1_WAVE1_LAUNCH_AUTHORIZATION.json"
PREFLIGHT_PATH = ROOT / "O1_WAVE1_LAUNCH_PREFLIGHT_REPORT.json"

LAUNCH_ROWS = (
    ("01", "O1_01", 1.5e-4, 65, "mpgfer-o1-01-lr00015-end65-s42"),
    ("02", "O1_02", 2.0e-4, 65, "mpgfer-o1-02-lr00020-end65-s42"),
    ("03", "O1_03", 2.5e-4, 65, "mpgfer-o1-03-lr00025-end65-s42"),
    ("04", "O1_04", 3.0e-4, 65, "mpgfer-o1-04-lr00030-end65-s42"),
    ("05", "O1_05", 4.0e-4, 65, "mpgfer-o1-05-lr00040-end65-s42"),
    ("06", "O1_06", 1.5e-4, 75, "mpgfer-o1-06-lr00015-end75-s42"),
    ("07", "O1_07", 2.0e-4, 75, "mpgfer-o1-07-lr00020-end75-s42"),
    ("08", "O1_08", 2.5e-4, 75, "mpgfer-o1-08-lr00025-end75-s42"),
    ("09", "O1_09", 3.0e-4, 75, "mpgfer-o1-09-lr00030-end75-s42"),
    ("10", "O1_10", 4.0e-4, 75, "mpgfer-o1-10-lr00040-end75-s42"),
    ("11", "O1_11", 1.5e-4, 85, "mpgfer-o1-11-lr00015-end85-s42"),
    ("12", "O1_12", 2.0e-4, 85, "mpgfer-o1-12-lr00020-end85-s42"),
    ("13", "O1_13", 2.5e-4, 85, "mpgfer-o1-13-lr00025-end85-s42"),
    ("14", "O1_14", 4.0e-4, 85, "mpgfer-o1-14-lr00040-end85-s42"),
)

PROTECTED_PATHS = (
    "research/mpg_fer_v2_3/src/mpg_fer_v2_3",
    "research/mpg_fer_v2_3/src/mpg_fer_o1",
    "research/mpg_fer_v2_3/notebooks/MPG_FER_O1_Wave1_Kaggle_T4.ipynb",
    "research/mpg_fer_v2_3/O1_REGISTRY.json",
    "research/mpg_fer_v2_3/O1_BASELINE_REFERENCE.json",
    "research/mpg_fer_v2_3/O1_IMPLEMENTATION_COMMIT.txt",
)

DESIGN_ALLOWED_ADDITIONS = {
    "launch_authorization_issue",
    "launch_authorization_path",
    "launch_authorization_sha256",
    "launch_job_count",
    "authorization_state",
}


def _frozen_source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((SRC / "mpg_fer_v2_3").glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _git_json_at(commit: str, relative_path: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return json.loads(completed.stdout)


def _assert_protected_identities() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "diff", "--quiet", REVIEWED_FRAMEWORK_HEAD, "--", *PROTECTED_PATHS],
        cwd=REPO_ROOT,
        check=False,
    )
    identities = {
        "o1_implementation_commit": IMPLEMENTATION_LOCK_PATH.read_text(
            encoding="utf-8"
        ).strip(),
        "o1_source_tree_sha256": o1_source_tree_hash(SRC),
        "canonical_notebook_sha256": sha256_file(NOTEBOOK_PATH),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "baseline_reference_sha256": sha256_file(BASELINE_PATH),
        "frozen_v2_3_scientific_source_sha256": _frozen_source_hash(),
    }
    expected = {
        "o1_implementation_commit": REVIEWED_IMPLEMENTATION_COMMIT,
        "o1_source_tree_sha256": REVIEWED_O1_SOURCE_SHA256,
        "canonical_notebook_sha256": REVIEWED_NOTEBOOK_SHA256,
        "registry_sha256": REVIEWED_REGISTRY_SHA256,
        "baseline_reference_sha256": REVIEWED_BASELINE_SHA256,
        "frozen_v2_3_scientific_source_sha256": FROZEN_SCIENTIFIC_SOURCE_SHA256,
    }
    if completed.returncode or identities != expected:
        raise RuntimeError("Issue #105 protected scientific/framework identity changed")
    return {
        **identities,
        "reviewed_framework_head": REVIEWED_FRAMEWORK_HEAD,
        "protected_paths_git_diff_exit_code": completed.returncode,
        "canonical_notebook_bytes_unchanged": True,
        "scientific_source_bytes_unchanged": True,
    }


def _launch_jobs() -> list[dict[str, Any]]:
    jobs = []
    for slot, config_id, learning_rate, decay_end, run_id in LAUNCH_ROWS:
        jobs.append(
            {
                "slot": slot,
                "config_id": config_id,
                "learning_rate": learning_rate,
                "lr_decay_end_epoch": decay_end,
                "run_id": run_id,
                "seed": 42,
                "resume_mode": "fresh",
                "segment_number": 1,
                "launch_controls": {
                    "MPG_FER_O1_CONFIG_ID": config_id,
                    "MPG_FER_O1_RUN_ID": run_id,
                    "MPG_FER_O1_RESUME_MODE": "fresh",
                    "MPG_FER_O1_SEGMENT_NUMBER": "1",
                },
            }
        )
    return jobs


def build_authorization() -> dict[str, Any]:
    identities = _assert_protected_identities()
    jobs = _launch_jobs()
    if [job["config_id"] for job in jobs] != list(O1_CONFIG_ORDER):
        raise RuntimeError("Launch order does not equal O1_CONFIG_ORDER")
    for job in jobs:
        spec = O1_REGISTRY[job["config_id"]]
        if (
            job["learning_rate"] != spec.learning_rate
            or job["lr_decay_end_epoch"] != spec.lr_decay_end_epoch
        ):
            raise RuntimeError(f"Launch registry mismatch: {job['config_id']}")
    return {
        "schema_version": 1,
        "issue": {"number": ISSUE_NUMBER, "url": ISSUE_URL},
        "framework_pr": {"number": FRAMEWORK_PR_NUMBER, "url": FRAMEWORK_PR_URL},
        "reviewed_framework_head": REVIEWED_FRAMEWORK_HEAD,
        "reviewed_framework_design_lock_sha256": REVIEWED_DESIGN_LOCK_SHA256,
        "immutable_identities": identities,
        "screen_stop_epoch": SCREEN_STOP_EPOCH,
        "scientific_max_epochs": 120,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "private_test_permitted": False,
        "data_scope": "Train+PublicTest_only",
        "required_mounted_inputs": {
            "train_csv_count": 1,
            "val_csv_count": 1,
            "reviewed_design_lock_count": 1,
            "forbidden_markers": [
                "test.csv",
                "PrivateTest",
                "private_test",
                "private-test",
            ],
        },
        "canonical_notebook": {
            "expected_local_reviewed_path": (
                "notebooks/MPG_FER_O1_Wave1_Kaggle_T4.ipynb"
            ),
            "expected_sha256": REVIEWED_NOTEBOOK_SHA256,
            "operator_requirement": (
                "upload_and_use_only_the_byte_identical_reviewed_notebook"
            ),
        },
        "launch_job_count": 14,
        "launch_jobs": jobs,
        "historical_control": {
            "config_id": BASELINE_ID,
            "launched": False,
            "runnable": False,
            "role": "external_historical_control_only",
        },
        "fresh_launch_defaults": {"resume_mode": "fresh", "segment_number": 1},
        "exact_resume_retry_policy": {
            "trigger": "documented_technical_or_infrastructure_interruption_only",
            "preserve_exactly": ["config_id", "run_id", "scientific_config"],
            "permitted_changes": {
                "resume_mode": {"from": "fresh", "to": "required"},
                "segment_number": "increment_by_one_only",
                "resume_artifact": (
                    "exact_prior_resume_latest.pt_plus_resume_latest.json_metadata"
                ),
            },
            "metric_based_rerun_forbidden": True,
            "hyperparameter_change_after_partial_trajectory_forbidden": True,
            "restart_with_different_config_or_run_id_forbidden": True,
            "retain_failed_and_partial_artifacts": True,
        },
        "metric_based_rerun_forbidden": True,
        "scientific_field_changes": [],
        "authorization_state": AUTHORIZATION_STATE,
        "wave1_execution_authorized": False,
        "status": READY_STATUS,
    }


def _write_design_binding(authorization_sha256: str) -> dict[str, Any]:
    relative_design = "research/mpg_fer_v2_3/O1_HPO_DESIGN_LOCK.json"
    reviewed = _git_json_at(REVIEWED_FRAMEWORK_HEAD, relative_design)
    current = json.loads(DESIGN_LOCK_PATH.read_text(encoding="utf-8"))
    current_without_additions = {
        key: value for key, value in current.items() if key not in DESIGN_ALLOWED_ADDITIONS
    }
    if current_without_additions != reviewed:
        raise RuntimeError("Design lock changed outside Issue #105 allowed additions")
    if reviewed.get("wave1_execution_authorized") is not False:
        raise RuntimeError("Reviewed design lock is not fail-closed")
    updated = dict(reviewed)
    updated.update(
        {
            "launch_authorization_issue": {"number": ISSUE_NUMBER, "url": ISSUE_URL},
            "launch_authorization_path": AUTHORIZATION_PATH.relative_to(ROOT).as_posix(),
            "launch_authorization_sha256": authorization_sha256,
            "launch_job_count": 14,
            "authorization_state": AUTHORIZATION_STATE,
        }
    )
    if updated["wave1_execution_authorized"] is not False:
        raise RuntimeError("Issue #105 candidate must remain fail-closed")
    write_json(DESIGN_LOCK_PATH, updated)
    return updated


def _run_tests() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
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
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    candidate_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    authorization = build_authorization()
    write_json(AUTHORIZATION_PATH, authorization)
    authorization_sha256 = sha256_file(AUTHORIZATION_PATH)
    design = _write_design_binding(authorization_sha256)
    tests = None if args.skip_tests else _run_tests()
    gate_status = "PASS" if tests is not None else "NOT_RUN"
    report = {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "framework_pr": FRAMEWORK_PR_NUMBER,
        "reviewed_framework_head": REVIEWED_FRAMEWORK_HEAD,
        "candidate_preparation_commit": candidate_commit,
        "launch_authorization_sha256": authorization_sha256,
        "design_lock_sha256": sha256_file(DESIGN_LOCK_PATH),
        "immutable_identities": authorization["immutable_identities"],
        "launch_job_count": len(authorization["launch_jobs"]),
        "authorization_state": design["authorization_state"],
        "wave1_execution_authorized": design["wave1_execution_authorized"],
        "test_run": tests,
        "gates": {
            f"O1-LA{index}": gate_status for index in range(1, 13)
        },
        "fer2013_accessed": False,
        "private_test_accessed": False,
        "kaggle_launched": False,
        "status": READY_STATUS if tests is not None else "O1_LA_PREFLIGHT_INCOMPLETE",
    }
    write_json(PREFLIGHT_PATH, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
