"""Regression tests for Issue #86 Execution Amendment E2 replicate resolver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINALIZE_SCRIPT_PATH = (
    PROJECT_ROOT / "kaggle" / "motif_issue86" / "generated" / "motif_finalize_train.py"
)


def _load_resolver():
    spec = importlib.util.spec_from_file_location(
        "motif_finalize_train", str(FINALIZE_SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._resolve_replicates_dir


def _create_dummy_replicate_dir(
    directory: Path,
    *,
    missing: str | None = None,
    extra: str | None = None,
    omit_manifest: str | None = None,
):
    directory.mkdir(parents=True, exist_ok=True)
    for arm in ("M", "C"):
        for rep_id in range(20):
            npz_name = f"motif_stability_{arm}_rep_{rep_id:02d}.npz"
            man_name = f"motif_stability_{arm}_rep_{rep_id:02d}.manifest.json"
            if npz_name != missing:
                (directory / npz_name).write_bytes(b"dummy_npz")
            if man_name != omit_manifest and npz_name != missing:
                (directory / man_name).write_text("{}", encoding="utf-8")
    if extra:
        (directory / extra).write_bytes(b"extra")


def test_1_valid_40_replicate_dir_resolves(tmp_path):
    resolver = _load_resolver()
    rep_dir = tmp_path / "valid_replicates"
    _create_dummy_replicate_dir(rep_dir)
    resolved = resolver((rep_dir,))
    assert resolved == rep_dir


def test_2_missing_replicate_fails(tmp_path):
    resolver = _load_resolver()
    rep_dir = tmp_path / "missing_replicate"
    _create_dummy_replicate_dir(rep_dir, missing="motif_stability_M_rep_05.npz")
    with pytest.raises(
        FileNotFoundError, match="cannot resolve canonical replicate directory"
    ):
        resolver((rep_dir,))


def test_3_missing_manifest_fails(tmp_path):
    resolver = _load_resolver()
    rep_dir = tmp_path / "missing_manifest"
    _create_dummy_replicate_dir(
        rep_dir, omit_manifest="motif_stability_C_rep_10.manifest.json"
    )
    with pytest.raises(
        FileNotFoundError, match="cannot resolve canonical replicate directory"
    ):
        resolver((rep_dir,))


def test_4_extra_replicate_fails(tmp_path):
    resolver = _load_resolver()
    rep_dir = tmp_path / "extra_replicate"
    _create_dummy_replicate_dir(rep_dir, extra="motif_stability_M_rep_20.npz")
    with pytest.raises(
        FileNotFoundError, match="cannot resolve canonical replicate directory"
    ):
        resolver((rep_dir,))


def test_5_old_naming_does_not_pass(tmp_path):
    resolver = _load_resolver()
    rep_dir = tmp_path / "old_naming"
    rep_dir.mkdir()
    (rep_dir / "motif_stability_replicate_0.npz").write_bytes(b"dummy")
    with pytest.raises(
        FileNotFoundError, match="cannot resolve canonical replicate directory"
    ):
        resolver((rep_dir,))
