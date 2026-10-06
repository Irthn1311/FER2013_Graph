"""A clone can verify retained results and must reject a corrupt compact file."""
import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_canonical_results", ROOT / "tools" / "verify_canonical_results.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_frozen_compact_package_and_statistics_pass_without_runtime_assets():
    result = verifier.verify(ROOT)
    assert result["status"] == "PASS"
    assert result["compact_checksums_pass"] == 223
    assert result["compact_files"] == 224
    assert not result["checkpoint_weights_or_dataset_loaded"]


def test_corrupt_compact_evidence_fails_closed(tmp_path):
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copy(ROOT / "v23_config.json", tmp_path / "v23_config.json")
    shutil.copytree(ROOT / "MPG_V23_MULTI_SEED_RESULTS", tmp_path / "MPG_V23_MULTI_SEED_RESULTS")
    target = tmp_path / "MPG_V23_MULTI_SEED_RESULTS" / "aggregate_statistics.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="Compact checksum mismatch"):
        verifier.verify(tmp_path)
