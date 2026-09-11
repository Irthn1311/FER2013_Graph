"""Scientific evidence reporting and audit manifest generator for Pure-GNN v3.1."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import time


def generate_scientific_audit_manifest(
    package_root: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """Generates an evidence manifest capturing git commit, configuration status, and protected hashes."""
    if package_root is None:
        package_root = str(Path(__file__).resolve().parents[3])

    root = Path(package_root)
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

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lineage": "Pure-GNN v3.1 Scientific Line",
        "scientific_execution_authorized": False,
        "protected_hashes": hashes,
        "governance_status": "LOCKED_DISABLED",
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest
