"""Contract tests for Issue #86 Execution Amendment E1 wrappers."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


SCIENTIFIC_SHA = "16a84b2b36f0d3584afd0a547487c4373dc22128"
HISTORICAL_NOTEBOOK_SHA256 = (
    "b2791ad54d9b69a1740fce19c8fa380be5dab5c9f22d05eb7585d544e4313e66"
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KAGGLE_E1_DIR = PROJECT_ROOT / "kaggle" / "motif_issue86"
MANIFEST_PATH = KAGGLE_E1_DIR / "execution_units_manifest.json"
GENERATOR_PATH = KAGGLE_E1_DIR / "generate_execution_units.py"


def _constant(code: str, name: str):
    match = re.search(rf"(?m)^{name} = (.+)$", code)
    assert match, f"constant {name} not found in code"
    return ast.literal_eval(match.group(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_1_historical_blocked_notebook_is_preserved_byte_identical():
    notebook_path = (
        PROJECT_ROOT / "notebooks" / "motif-qualification-train-kaggle.ipynb"
    )
    assert notebook_path.is_file(), "historical blocked notebook missing"
    raw = notebook_path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    assert (
        hashlib.sha256(normalized).hexdigest() == HISTORICAL_NOTEBOOK_SHA256
    ), "historical notebook SHA mismatch"


def test_2_e1_manifest_contains_exactly_42_units():
    assert MANIFEST_PATH.is_file(), "E1 manifest missing"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest) == 42, f"expected exactly 42 units, got {len(manifest)}"


def test_3_exact_unit_id_set_and_no_duplicates():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    unit_ids = list(manifest.keys())
    assert len(unit_ids) == len(set(unit_ids)), "duplicate unit ID found in manifest"

    expected_replicate_ids = {f"M{i:02d}" for i in range(20)} | {
        f"C{i:02d}" for i in range(20)
    }
    expected_all = expected_replicate_ids | {"BUILD_SUBSTRATE", "FINALIZE_TRAIN"}
    assert (
        set(unit_ids) == expected_all
    ), f"unit set mismatch: extra={set(unit_ids) - expected_all}, missing={expected_all - set(unit_ids)}"


def test_4_unit_arm_and_replicate_literals():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # BUILD_SUBSTRATE
    sub = manifest["BUILD_SUBSTRATE"]
    assert sub["mode"] == "BUILD_SUBSTRATE"
    assert sub["arm"] is None
    assert sub["replicate_id"] is None

    # FINALIZE_TRAIN
    fin = manifest["FINALIZE_TRAIN"]
    assert fin["mode"] == "FINALIZE_TRAIN"
    assert fin["arm"] is None
    assert fin["replicate_id"] is None

    # M00..M19
    for i in range(20):
        key = f"M{i:02d}"
        entry = manifest[key]
        assert entry["mode"] == "RUN_REPLICATE"
        assert entry["arm"] == "M"
        assert entry["replicate_id"] == i

    # C00..C19
    for i in range(20):
        key = f"C{i:02d}"
        entry = manifest[key]
        assert entry["mode"] == "RUN_REPLICATE"
        assert entry["arm"] == "C"
        assert entry["replicate_id"] == i


def test_5_all_scripts_pin_scientific_sha():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for unit_id, entry in manifest.items():
        assert (
            entry["scientific_sha"] == SCIENTIFIC_SHA
        ), f"manifest scientific_sha mismatch for {unit_id}"
        script_path = PROJECT_ROOT / entry["filename"]
        assert script_path.is_file(), f"script file missing for {unit_id}"
        code = script_path.read_text(encoding="utf-8")
        assert (
            _constant(code, "SCIENTIFIC_SHA") == SCIENTIFIC_SHA
        ), f"script scientific_sha mismatch for {unit_id}"


def test_6_manifest_hashes_match_committed_files_and_compile():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for unit_id, entry in manifest.items():
        script_path = PROJECT_ROOT / entry["filename"]
        assert script_path.is_file(), f"script missing for {unit_id}"
        computed_sha = _sha256(script_path)
        assert (
            computed_sha == entry["sha256"]
        ), f"hash mismatch for {unit_id}: {computed_sha} != {entry['sha256']}"
        code = script_path.read_text(encoding="utf-8")
        compile(code, str(script_path), "exec")


def test_7_generator_roundtrip_equality(tmp_path):
    sys.path.insert(0, str(KAGGLE_E1_DIR))
    try:
        from generate_execution_units import generate_all
    finally:
        if str(KAGGLE_E1_DIR) in sys.path:
            sys.path.remove(str(KAGGLE_E1_DIR))

    # Regenerate all wrappers into tmp_path
    tmp_manifest = generate_all(tmp_path)
    committed_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # Compare manifest content
    assert len(tmp_manifest) == len(committed_manifest)
    for unit_id in committed_manifest:
        assert unit_id in tmp_manifest
        assert tmp_manifest[unit_id]["sha256"] == committed_manifest[unit_id]["sha256"]

        tmp_file = (
            tmp_path / "generated" / Path(committed_manifest[unit_id]["filename"]).name
        )
        committed_file = PROJECT_ROOT / committed_manifest[unit_id]["filename"]

        assert tmp_file.is_file()
        assert committed_file.is_file()
        assert (
            tmp_file.read_bytes() == committed_file.read_bytes()
        ), f"byte mismatch for {unit_id}"


def test_8_no_env_based_routing():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    forbidden = (
        "MOTIF_EXECUTION_MODE",
        "MOTIF_STABILITY_ARM",
        "MOTIF_STABILITY_REPLICATE_ID",
    )
    for unit_id, entry in manifest.items():
        script_path = PROJECT_ROOT / entry["filename"]
        code = script_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert (
                token not in code
            ), f"env-var routing token {token} found in {unit_id}"


def test_9_no_secrets_based_routing():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    forbidden = (
        "UserSecretsClient",
        "kaggle_secrets",
    )
    for unit_id, entry in manifest.items():
        script_path = PROJECT_ROOT / entry["filename"]
        code = script_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in code, f"secret routing token {token} found in {unit_id}"


def test_10_no_public_or_private_path():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    forbidden = (
        "val.csv",
        "test.csv",
        "--public-csv",
        "--private-csv",
        "--test-csv",
    )
    for unit_id, entry in manifest.items():
        script_path = PROJECT_ROOT / entry["filename"]
        code = script_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert (
                token not in code
            ), f"forbidden test path token {token} found in {unit_id}"


def test_11_detached_source_checkout_and_clean_assertion():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for unit_id, entry in manifest.items():
        script_path = PROJECT_ROOT / entry["filename"]
        code = script_path.read_text(encoding="utf-8")
        assert (
            '"git", "checkout", "--detach", SCIENTIFIC_SHA' in code
            or "'git', 'checkout', '--detach', SCIENTIFIC_SHA" in code
        ), f"detached checkout missing in {unit_id}"
        assert (
            '"git", "status", "--porcelain"' in code
            or "'git', 'status', '--porcelain'" in code
        ), f"clean checkout assertion missing in {unit_id}"


def test_12_pytest_precedes_scientific_input_access():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for unit_id, entry in manifest.items():
        script_path = PROJECT_ROOT / entry["filename"]
        code = script_path.read_text(encoding="utf-8")
        assert "pytest" in code, f"pytest missing in {unit_id}"
        assert "sha256_file(path)" in code, f"hash check missing in {unit_id}"

        pytest_pos = code.index("pytest")
        hash_check_pos = code.index("sha256_file(path)")
        assert (
            pytest_pos < hash_check_pos
        ), f"pytest must precede hash checking and input access in {unit_id}"


def test_13_scientific_runner_invoked_directly():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for unit_id, entry in manifest.items():
        script_path = PROJECT_ROOT / entry["filename"]
        code = script_path.read_text(encoding="utf-8")
        assert (
            "pixel_relational_motif_e0.motif_train_runner" in code
        ), f"runner module missing in {unit_id}"
        assert (
            "MOTIF_SCIENTIFIC_SHA" in code
        ), f"MOTIF_SCIENTIFIC_SHA missing in {unit_id}"


def test_14_no_scientific_parameter_overrides():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    forbidden = (
        "--n-init",
        "--max-iter",
        "--jaccard-cutoff",
        "--support-threshold",
        "--node-cap",
    )
    for unit_id, entry in manifest.items():
        script_path = PROJECT_ROOT / entry["filename"]
        code = script_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert (
                token not in code
            ), f"scientific override token {token} found in {unit_id}"


def test_15_scientific_files_are_unchanged_from_scientific_sha():
    scientific_files = (
        "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/motif_qualification.py",
        "research/pixel_relational_motif_e0/src/pixel_relational_motif_e0/motif_train_runner.py",
        "research/pixel_relational_motif_e0/tests/test_motif_qualification.py",
        "research/pixel_relational_motif_e0/MOTIF_QUALIFICATION_SPARSIFICATION_PREREGISTRATION.md",
    )
    for rel_path in scientific_files:
        committed_blob = subprocess.check_output(
            ["git", "show", f"{SCIENTIFIC_SHA}:{rel_path}"],
            cwd=PROJECT_ROOT,
        )
        local_path = PROJECT_ROOT / rel_path
        assert local_path.is_file(), f"scientific file missing: {rel_path}"
        local_bytes = local_path.read_bytes().replace(b"\r\n", b"\n")
        assert local_bytes == committed_blob.replace(
            b"\r\n", b"\n"
        ), f"scientific file {rel_path} was modified compared to {SCIENTIFIC_SHA}"
