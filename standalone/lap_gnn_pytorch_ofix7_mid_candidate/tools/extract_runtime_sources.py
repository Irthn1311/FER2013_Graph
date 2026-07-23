"""Mechanically extract the locked OFIX7-mid runtime from its source commit.

This tool is intentionally the only package file that reads the parent Git
repository. It is not imported by, or distributed as part of, normal runtime
execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SOURCE_COMMIT = "241a8872027cd284fe679533a0be95cb48e7d253"
SOURCE_MAP = {
    "d16/training/train_d16.py": "src/lap_gnn/training/engine.py",
    "d16/data/detail_node_features.py": "src/lap_gnn/data/detail_node_features.py",
    "d16/data/graph_builder.py": "src/lap_gnn/data/graph_builder.py",
    "d16/data/graph_cache_dataset.py": "src/lap_gnn/data/graph_cache_dataset.py",
    "d16/data/mediapipe_priors.py": "src/lap_gnn/data/mediapipe_priors.py",
    "d16/data/patch_tokenizer.py": "src/lap_gnn/data/patch_tokenizer.py",
    "d16/data/pixel_prior_dataset.py": "src/lap_gnn/data/pixel_prior_dataset.py",
    "d16/losses/hard_proto_separation.py": "src/lap_gnn/losses/hard_proto_separation.py",
    "d16/losses/main_logit_pair_margin.py": "src/lap_gnn/losses/main_logit_pair_margin.py",
    "d16/losses/pairwise_hard_relation.py": "src/lap_gnn/losses/pairwise_hard_relation.py",
    "d16/losses/part_supcon.py": "src/lap_gnn/losses/part_supcon.py",
    "d16/models/classifier.py": "src/lap_gnn/model/classifier.py",
    "d16/models/d16_model.py": "src/lap_gnn/model/d16_model.py",
    "d16/models/edge_context_gnn.py": "src/lap_gnn/model/edge_context_gnn.py",
    "d16/models/evidence_heads.py": "src/lap_gnn/model/evidence_heads.py",
    "d16/models/fallback_patch_encoder.py": "src/lap_gnn/model/fallback_patch_encoder.py",
    "d16/models/micro_motif_support_readout.py": "src/lap_gnn/model/micro_motif_support_readout.py",
    "d16/models/part_attention_readout.py": "src/lap_gnn/model/part_attention_readout.py",
    "d16/models/part_aware_gnn.py": "src/lap_gnn/model/part_aware_gnn.py",
    "d16/models/part_motif_query_readout.py": "src/lap_gnn/model/part_motif_query_readout.py",
    "d16/models/part_token_transformer_readout.py": "src/lap_gnn/model/part_token_transformer_readout.py",
    "d16/models/pixel_encoder.py": "src/lap_gnn/model/pixel_encoder.py",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(repo: Path, source: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{source}"],
        cwd=repo,
    )


def rewrite_source(source: str, text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    replacements = {
        "from d16.data.": "from lap_gnn.data.",
        "from d16.losses.": "from lap_gnn.losses.",
        "from d16.models.": "from lap_gnn.model.",
        "import d16.data.": "import lap_gnn.data.",
        "import d16.losses.": "import lap_gnn.losses.",
        "import d16.models.": "import lap_gnn.model.",
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
            changes.append(f"namespace: {old.strip()} -> {new.strip()}")

    if source == "d16/training/train_d16.py":
        block = (
            "PROJECT_ROOT = Path(__file__).resolve().parents[2]\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        )
        if block not in text:
            raise RuntimeError("Historical parent-path bootstrap block was not found")
        text = text.replace(block, "")
        changes.append("removed parent repository sys.path bootstrap")
    return text, changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    package_root = args.package_root.resolve()
    records = []
    for source, destination in SOURCE_MAP.items():
        raw = git_blob(repo, source)
        decoded = raw.decode("utf-8")
        rewritten, changes = rewrite_source(source, decoded)
        target = package_root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rewritten, encoding="utf-8", newline="\n")
        records.append(
            {
                "source_commit": SOURCE_COMMIT,
                "original_file": source,
                "original_sha256": sha256_bytes(raw),
                "original_lines": len(decoded.splitlines()),
                "destination": destination,
                "destination_sha256": sha256_bytes(target.read_bytes()),
                "destination_lines": len(rewritten.splitlines()),
                "copied_exactly": not changes,
                "mechanical_changes": changes,
            }
        )

    for relative in [
        "src/lap_gnn/__init__.py",
        "src/lap_gnn/data/__init__.py",
        "src/lap_gnn/losses/__init__.py",
        "src/lap_gnn/model/__init__.py",
        "src/lap_gnn/training/__init__.py",
    ]:
        target = package_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text('"""Locked OFIX7-mid standalone runtime."""\n', encoding="utf-8")

    mapping = package_root / "source_mapping.generated.json"
    mapping.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_commit": SOURCE_COMMIT, "files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
