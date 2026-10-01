from __future__ import annotations

import json
from pathlib import Path
import subprocess

from mpg_fer_o1.protocol import (
    BASELINE_ID,
    FROZEN_SCIENTIFIC_SOURCE_SHA256,
    O1_CONFIG_ORDER,
    O1_REGISTRY,
    o1_source_tree_hash,
)
from mpg_fer_v2_3.checkpoint import sha256_file


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
REVIEWED_HEAD = "e7ea4f1a969d50b34b7e0930338d92f1263bdb96"
REVIEWED_SOURCE = "800803493415562b3d533a8631d79203a181757220f10bad7c4326387d360595"
REVIEWED_NOTEBOOK = "752cec59a824092ab3286f6aa1a37a26abc81916152db04e6cf62b7c3c268a81"
REVIEWED_REGISTRY = "6a839be43ca5da9542d6bcd1f6a18d6c3c2e0326898ce59750c9c8560bb968aa"
REVIEWED_BASELINE = "1b0aca11d49af9bb9e3ee170753666e0b446b435eecdc273a98e8f95c8c01f4e"
REVIEWED_IMPLEMENTATION = "b4a3563ca284dae5dc6ad5aad7023eda1253590b"
AUTHORIZATION_STATE = "FINAL_AUTHORIZED"
REVIEW_VERDICT = "MPG_FER_O1_WAVE1_AUTHORIZATION_CANDIDATE_REVIEW_PASS"
REVIEW_COMMENT_ID = 5929994015

EXPECTED_ROWS = (
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


def _load(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )


def _authorization() -> dict:
    return _load(ROOT / "O1_WAVE1_LAUNCH_AUTHORIZATION.json")


def test_launch_matrix_is_exact_ordered_unique_and_registry_bound() -> None:
    authorization = _authorization()
    jobs = authorization["launch_jobs"]
    actual = tuple(
        (
            job["slot"],
            job["config_id"],
            job["learning_rate"],
            job["lr_decay_end_epoch"],
            job["run_id"],
        )
        for job in jobs
    )
    assert authorization["launch_job_count"] == len(jobs) == 14
    assert actual == EXPECTED_ROWS
    assert tuple(job["config_id"] for job in jobs) == O1_CONFIG_ORDER
    assert len({job["config_id"] for job in jobs}) == 14
    assert len({job["run_id"] for job in jobs}) == 14
    for job in jobs:
        spec = O1_REGISTRY[job["config_id"]]
        assert job["learning_rate"] == spec.learning_rate
        assert job["lr_decay_end_epoch"] == spec.lr_decay_end_epoch
        assert job["seed"] == 42
        assert job["resume_mode"] == "fresh"
        assert job["segment_number"] == 1
        assert job["launch_controls"] == {
            "MPG_FER_O1_CONFIG_ID": job["config_id"],
            "MPG_FER_O1_RUN_ID": job["run_id"],
            "MPG_FER_O1_RESUME_MODE": "fresh",
            "MPG_FER_O1_SEGMENT_NUMBER": "1",
        }


def test_historical_control_is_not_launched_and_run_ids_encode_pairs() -> None:
    authorization = _authorization()
    assert BASELINE_ID not in {job["config_id"] for job in authorization["launch_jobs"]}
    assert authorization["historical_control"] == {
        "config_id": BASELINE_ID,
        "launched": False,
        "runnable": False,
        "role": "external_historical_control_only",
    }
    for slot, config_id, learning_rate, decay_end, expected_run_id in EXPECTED_ROWS:
        expected_lr_code = f"{round(learning_rate * 100000):05d}"
        assert expected_run_id == (
            f"mpgfer-o1-{slot}-lr{expected_lr_code}-end{decay_end}-s42"
        )
        assert config_id == f"O1_{slot}"


def test_exact_resume_retry_preserves_identity_and_forbids_metric_reruns() -> None:
    authorization = _authorization()
    policy = authorization["exact_resume_retry_policy"]
    fresh = dict(authorization["launch_jobs"][0]["launch_controls"])
    retry = {
        **fresh,
        "MPG_FER_O1_RESUME_MODE": "required",
        "MPG_FER_O1_SEGMENT_NUMBER": "2",
        "RESUME_ARTIFACT": "exact-prior/resume_latest.pt",
    }
    assert retry["MPG_FER_O1_CONFIG_ID"] == fresh["MPG_FER_O1_CONFIG_ID"]
    assert retry["MPG_FER_O1_RUN_ID"] == fresh["MPG_FER_O1_RUN_ID"]
    assert set(retry).difference(fresh) == {"RESUME_ARTIFACT"}
    assert {
        key for key in fresh if retry[key] != fresh[key]
    } == {"MPG_FER_O1_RESUME_MODE", "MPG_FER_O1_SEGMENT_NUMBER"}
    assert policy["preserve_exactly"] == ["config_id", "run_id", "scientific_config"]
    assert policy["metric_based_rerun_forbidden"] is True
    assert policy["hyperparameter_change_after_partial_trajectory_forbidden"] is True
    assert policy["restart_with_different_config_or_run_id_forbidden"] is True
    assert authorization["metric_based_rerun_forbidden"] is True
    assert authorization["scientific_field_changes"] == []


def test_reviewed_hashes_and_protected_bytes_are_unchanged() -> None:
    authorization = _authorization()
    identities = authorization["immutable_identities"]
    assert identities["reviewed_framework_head"] == REVIEWED_HEAD
    assert identities["o1_implementation_commit"] == REVIEWED_IMPLEMENTATION
    assert identities["o1_source_tree_sha256"] == REVIEWED_SOURCE
    assert identities["canonical_notebook_sha256"] == REVIEWED_NOTEBOOK
    assert identities["registry_sha256"] == REVIEWED_REGISTRY
    assert identities["baseline_reference_sha256"] == REVIEWED_BASELINE
    assert identities["frozen_v2_3_scientific_source_sha256"] == (
        FROZEN_SCIENTIFIC_SOURCE_SHA256
    )
    assert o1_source_tree_hash(ROOT / "src") == REVIEWED_SOURCE
    assert sha256_file(
        ROOT / "notebooks" / "MPG_FER_O1_Wave1_Kaggle_T4.ipynb"
    ) == REVIEWED_NOTEBOOK
    assert sha256_file(ROOT / "O1_REGISTRY.json") == REVIEWED_REGISTRY
    assert sha256_file(ROOT / "O1_BASELINE_REFERENCE.json") == REVIEWED_BASELINE
    assert (ROOT / "O1_IMPLEMENTATION_COMMIT.txt").read_text(
        encoding="utf-8"
    ).strip() == REVIEWED_IMPLEMENTATION
    completed = subprocess.run(
        ["git", "diff", "--quiet", REVIEWED_HEAD, "--", *PROTECTED_PATHS],
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0
    assert identities["canonical_notebook_bytes_unchanged"] is True
    assert identities["scientific_source_bytes_unchanged"] is True


def test_notebook_identity_and_final_review_authorization_are_bound() -> None:
    authorization_path = ROOT / "O1_WAVE1_LAUNCH_AUTHORIZATION.json"
    authorization = _load(authorization_path)
    design = _load(ROOT / "O1_HPO_DESIGN_LOCK.json")
    assert authorization["canonical_notebook"] == {
        "expected_local_reviewed_path": (
            "notebooks/MPG_FER_O1_Wave1_Kaggle_T4.ipynb"
        ),
        "expected_sha256": REVIEWED_NOTEBOOK,
        "operator_requirement": (
            "upload_and_use_only_the_byte_identical_reviewed_notebook"
        ),
    }
    assert authorization["authorization_state"] == AUTHORIZATION_STATE
    assert authorization["wave1_execution_authorized"] is True
    assert authorization["status"] == "MPG_FER_O1_WAVE1_FINAL_AUTHORIZED"
    assert design["authorization_state"] == AUTHORIZATION_STATE
    assert design["wave1_execution_authorized"] is True
    assert design["authorization_blocker"] is None
    assert design["launch_authorization_issue"]["number"] == 105
    assert design["launch_authorization_path"] == authorization_path.name
    assert design["launch_authorization_sha256"] == sha256_file(authorization_path)
    assert design["launch_job_count"] == 14
    assert design["o1_source_tree_sha256"] == REVIEWED_SOURCE
    assert design["notebook_sha256"] == REVIEWED_NOTEBOOK
    assert design["registry_sha256"] == REVIEWED_REGISTRY
    assert design["baseline_reference_sha256"] == REVIEWED_BASELINE
    final_review = {
        "reviewed_candidate_head": "9c8cc8804b362e13c99a240a996775dfbaf55aa0",
        "reviewer_verdict": REVIEW_VERDICT,
        "review_comment_id": REVIEW_COMMENT_ID,
        "review_pr": 104,
    }
    for key, value in final_review.items():
        assert authorization[key] == value
        assert design[key] == value
