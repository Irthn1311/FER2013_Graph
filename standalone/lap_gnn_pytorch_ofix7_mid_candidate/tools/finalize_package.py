"""Create the package manifest and deterministic checksum list."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy
import sklearn
import torch
import yaml


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sha256",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return data
    text = data.decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--isolated-test-pass", action="store_true")
    args = parser.parse_args()
    root = args.package_root.resolve()
    mapping_path = root / "source_mapping.generated.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    for record in mapping:
        destination = root / record["destination"]
        record["destination_sha256"] = sha(destination)
        record["destination_lines"] = len(destination.read_text(encoding="utf-8").splitlines())
        if record["original_file"] == "d16/training/train_d16.py":
            additions = [
                "standalone policy guard rejects resume",
                "checkpoint payload adds portable provenance signatures",
            ]
            record["mechanical_changes"] = list(dict.fromkeys(record["mechanical_changes"] + additions))
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    parity = json.loads((root / "validation_assets/parity_results.json").read_text(encoding="utf-8"))
    config_hashes = {
        path.name: sha(path) for path in sorted((root / "configs").glob("*.yaml"))
    }
    contract_hashes = {
        path.name: sha(path) for path in sorted((root / "contracts").iterdir()) if path.is_file()
    }
    fixture_hashes = {
        path.name: sha(path)
        for path in sorted((root / "validation_assets/golden").iterdir())
        if path.is_file()
    }
    manifest = {
        "package_name": "lap_gnn_pytorch_ofix7_mid_candidate",
        "package_version": "0.1.0",
        "status": "baseline_candidate",
        "parent_repository_commit": "241a8872027cd284fe679533a0be95cb48e7d253",
        "baseline_lock_sha256": "d54c9162de7e4f6bda2ee37dbe735939f7542195d9d2fe6dbd5e61cd85351dc3",
        "checkpoint_policy_lock_sha256": "dfce606a69343b1a8de821ec3fc547d5700b94ff6c9a15f6b26d32e02601fc5f",
        "source_to_destination_mapping": mapping,
        "config_hashes": config_hashes,
        "contract_hashes": contract_hashes,
        "golden_fixture_hashes": fixture_hashes,
        "parameter_count": 1061192,
        "feature_schema_hash": contract_hashes["feature_schema.json"],
        "graph_schema_hash": contract_hashes["graph_batch_schema.json"],
        "tested_environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
        },
        "historical_environment": {
            "python": "3.12.12",
            "torch": "2.10.0+cu128",
            "cuda": "12.8",
            "cudnn": 91002,
        },
        "excluded_branches": [
            "D17", "D18", "D19", "S1/O1", "TensorFlow", "CNN",
            "Semantic ROI", "MGR-CNN", "notebooks", "historical outputs/checkpoints",
        ],
        "parity_results": {
            key: parity[key] for key in (
                "graph_parity_pass", "forward_parity_pass",
                "training_step_parity_pass", "metric_parity_pass",
                "checkpoint_roundtrip_pass", "sample_count",
                "optimizer_steps", "logit_max_abs", "prediction_agreement",
            )
        },
        "isolated_copy_test_pass": bool(args.isolated_test_pass),
        "known_limitations": [
            "Candidate label remains until S1/O1 policy is locked.",
            "FER2013 and precomputed priors are external.",
            "MediaPipe prior generation is optional and not used by normal training.",
            "Full standalone seed42 training was not launched in extraction.",
        ],
    }
    (root / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    excluded_names = {"CHECKSUMS.sha256", "package_manifest.json"}
    files = [
        path for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in excluded_names
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
    ]
    lines = [f"{sha(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": True, "checksummed_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
