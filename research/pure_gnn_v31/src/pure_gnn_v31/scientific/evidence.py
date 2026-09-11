"""Scientific evidence reporting and audit manifest generator for Pure-GNN v3.1."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import time

from pure_gnn_v31.scientific.config import load_scientific_config


def get_git_commit_sha(repo_root: Path) -> Optional[str]:
    """Attempts to retrieve the current git commit SHA if git is available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def generate_scientific_audit_manifest(
    package_root: Optional[str] = None,
    output_path: Optional[str] = None,
    train_sha256: Optional[str] = None,
    val_sha256: Optional[str] = None,
) -> Dict:
    """Generates an evidence manifest capturing git commit, reviewed configuration status, and protected hashes.

    Reads the actual reviewed configuration state from scientific_screen_historical_v1.yaml.
    Derives governance_status and paired_augmentation_status dynamically from config.
    Strictly records NO test SHA under any state.
    """
    if package_root is None:
        package_root = str(Path(__file__).resolve().parents[3])

    root = Path(package_root)
    repo_root = root.parents[1]

    # Load actual reviewed configuration
    cfg_file = root / "configs" / "scientific_screen_historical_v1.yaml"
    cfg_sha = hashlib.sha256(cfg_file.read_bytes()).hexdigest() if cfg_file.is_file() else "MISSING"
    config_obj = load_scientific_config(str(cfg_file))

    protected_rel_paths = [
        "src/pure_gnn_v31/model.py",
        "src/pure_gnn_v31/local_relation.py",
        "src/pure_gnn_v31/coarse_conditions.py",
        "src/pure_gnn_v31/coarsening.py",
        "src/pure_gnn_v31/graph_index.py",
        "src/pure_gnn_v31/diagnostics.py",
        "src/pure_gnn_v31/contracts.py",
        "src/pure_gnn_v31/data.py",
    ]

    hashes = {}
    for rel in protected_rel_paths:
        f = root / rel
        if f.is_file():
            hashes[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        else:
            hashes[rel] = "MISSING"

    # Derive governance_status from actual reviewed config
    if config_obj.scientific_execution_authorized and not config_obj.has_unresolved_hyperparameters:
        gov_status = "AUTHORIZED"
    else:
        gov_status = "LOCKED_DISABLED"

    # Derive paired_augmentation_status from config
    aug_spec = config_obj.hyperparameters.get("augmentation_policy", {})
    if isinstance(aug_spec, dict) and aug_spec.get("status") == "SOURCE_CONFIRMED" and aug_spec.get("value") is not None:
        paired_aug_status = "CONFIGURED"
    else:
        paired_aug_status = "BLOCKED_ON_AUGMENTATION_POLICY"

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lineage": "Pure-GNN v3.1 Scientific Line",
        "scientific_execution_authorized": config_obj.scientific_execution_authorized,
        "config_sha256": cfg_sha,
        "source_commit": get_git_commit_sha(repo_root),
        "train_sha256": train_sha256,
        "val_sha256": val_sha256,
        # Strictly NO test_sha256 field under any state
        "paired_augmentation_status": paired_aug_status,
        "protected_hashes": hashes,
        "unresolved_hyperparameters": config_obj.get_unresolved_fields(),
        "governance_status": gov_status,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest
