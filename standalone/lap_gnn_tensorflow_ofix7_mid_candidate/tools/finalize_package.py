"""Finalize the scientific payload lock, package manifest and checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PYTORCH = REPO / "standalone" / "lap_gnn_pytorch_ofix7_mid_candidate"
EXECUTION_CONTRACT = (
    ROOT / "contracts" / "tensorflow_execution_contract_v2.json"
)
EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_files(include_manifest: bool) -> list[Path]:
    excluded_names = {"CHECKSUMS.sha256"}
    if not include_manifest:
        excluded_names.add("package_manifest.json")
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in excluded_names:
            continue
        if any(part in {"__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name.startswith("parity_"):
            continue
        files.append(path)
    return sorted(files)


def scientific_checksum() -> str:
    digest = hashlib.sha256()
    files = []
    for relative_root in ["src/lap_gnn_tf", "contracts", "validation_assets"]:
        files.extend(path for path in (ROOT / relative_root).rglob("*") if path.is_file())
    for path in sorted(files):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def copied_records() -> list[dict]:
    records = []
    pairs = [
        ("contracts/architecture_contract.md", "contracts/architecture_contract.md"),
        ("contracts/feature_schema.json", "contracts/feature_schema.json"),
        ("contracts/edge_schema.json", "contracts/edge_schema.json"),
        ("contracts/node_schema.json", "contracts/node_schema.json"),
        ("contracts/graph_batch_schema.json", "contracts/graph_batch_schema.json"),
        ("contracts/preprocessing_contract.json", "contracts/preprocessing_contract.json"),
        ("contracts/checkpoint_policy.json", "contracts/checkpoint_policy.json"),
        ("validation_assets/manifest.json", "validation_assets/manifest.json"),
        ("validation_assets/README.md", "validation_assets/README.md"),
    ]
    source_golden = PYTORCH / "validation_assets" / "golden"
    for source in sorted(source_golden.glob("*")):
        if source.name in {"pytorch_gradients_eval_ce.npz", "pytorch_adamw_step1_eval_ce.npz", "gradient_manifest.json"}:
            continue
        pairs.append(
            (
                f"validation_assets/golden/{source.name}",
                f"validation_assets/golden/{source.name}",
            )
        )
    for source_relative, destination_relative in pairs:
        source = PYTORCH / source_relative
        destination = ROOT / destination_relative
        records.append({
            "source": source_relative,
            "destination": destination_relative,
            "source_sha256": sha256(source),
            "destination_sha256": sha256(destination),
            "byte_identical": source.read_bytes() == destination.read_bytes(),
        })
    return records


def main() -> None:
    actual_contract_sha = sha256(EXECUTION_CONTRACT)
    if actual_contract_sha != EXECUTION_CONTRACT_SHA256:
        raise RuntimeError(
            "Execution contract drift: "
            f"{actual_contract_sha} != {EXECUTION_CONTRACT_SHA256}"
        )
    payload_checksum = scientific_checksum()
    baseline = ROOT / "configs" / "fer2013_ofix7_mid_tensorflow_baseline.yaml"
    config = yaml.safe_load(baseline.read_text(encoding="utf-8"))
    config["locked"]["package_checksum"] = payload_checksum
    baseline.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    files = package_files(include_manifest=False)
    manifest = {
        "package": "lap_gnn_tensorflow_ofix7_mid_candidate",
        "version": "0.1.0",
        "framework": "tensorflow",
        "scientific_payload_sha256": payload_checksum,
        "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
        "selected_execution_strategy": (
            "SELECT_G1_RESTRICTED_GRAPH_OPTIMIZER"
        ),
        "readiness_decision": "READY_FOR_TENSORFLOW_KAGGLE_SEED42",
        "readiness_reason": (
            "G1-A restricted Grappler passes the 2e-8 two-step gate, "
            "15 repetitions, mixed precision, and exact checkpoint continuation."
        ),
        "trainable_parameters": 1_061_192,
        "non_trainable_parameters": 0,
        "full_training_launched": False,
        "copied_source_records": copied_records(),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    manifest_path = ROOT / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in package_files(include_manifest=True)
    ]
    (ROOT / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "scientific_payload_sha256": payload_checksum,
        "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
        "readiness_decision": manifest["readiness_decision"],
        "files": len(checksum_lines),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
