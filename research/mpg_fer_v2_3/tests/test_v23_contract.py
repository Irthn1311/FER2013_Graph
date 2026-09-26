from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch

from mpg_fer_v2_3.checkpoint import config_hash
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.model import MPGFER
from mpg_fer_v2_3.train import source_tree_hash


ROOT = Path(__file__).resolve().parents[3]
V22_PACKAGE = ROOT / "research" / "mpg_fer_v2_2" / "src" / "mpg_fer_v2_2"
V23_PACKAGE = ROOT / "research" / "mpg_fer_v2_3" / "src" / "mpg_fer_v2_3"
FROZEN_V22_SOURCE_SHA256 = (
    "a8dc77db29e997c4c3ab69bb862704c8a948f940a4636e1c01e0d96bab40de65"
)
BYTE_STABLE_NON_TARGET_FILES = (
    "checkpoint.py",
    "data.py",
    "ema.py",
    "evaluate.py",
    "features.py",
    "graph.py",
    "losses.py",
    "motif.py",
    "utils.py",
)


def _tree_hash(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _v22_config():
    source_root = V22_PACKAGE.parent
    sys.path.insert(0, str(source_root))
    try:
        from mpg_fer_v2_2.config import MPGConfig as MPGConfigV22
    finally:
        sys.path.remove(str(source_root))
    return MPGConfigV22()


def test_frozen_v22_source_hash_and_non_target_files_are_unchanged() -> None:
    assert _tree_hash(V22_PACKAGE) == FROZEN_V22_SOURCE_SHA256
    for name in BYTE_STABLE_NON_TARGET_FILES:
        assert (V23_PACKAGE / name).read_bytes() == (V22_PACKAGE / name).read_bytes()


def test_v23_model_summary_records_exact_identity() -> None:
    model = MPGFER()
    summary = model.model_summary(
        source_sha256="source-hash", source_git_commit="git-commit"
    )
    assert summary == {
        "version": "MPG-FER v2.3",
        "total_parameters": 2_304_528,
        "trainable_parameters": 2_304_528,
        "early_depth_intervention": {
            "mechanism": "fixed_residual_branch_scaling",
            "motif_residual_scale_schedule": [0.5, 0.5, 1.0, 1.0, 1.0],
            "target_layers": [1, 2],
            "coefficient_sweep": False,
        },
        "motif_topk_schedule": [8, 16, 16, 16, 24],
        "source_git_commit": "git-commit",
        "source_sha256": "source-hash",
    }


def test_v23_state_dict_round_trip_is_exact(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = MPGFER().eval()
    checkpoint = tmp_path / "v23-state.pt"
    torch.save(model.state_dict(), checkpoint)
    clone = MPGFER().eval()
    clone.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, clone.state_dict()[name]), name


def test_v22_and_v23_resume_identities_are_incompatible() -> None:
    v22 = _v22_config()
    v23 = MPGConfig()
    assert config_hash(v22) != config_hash(v23)
    assert _tree_hash(V22_PACKAGE) != source_tree_hash(V23_PACKAGE)
