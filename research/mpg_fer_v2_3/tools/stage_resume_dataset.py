"""Validate and stage one private Kaggle resume dataset for Issue #99."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ASSIGNMENTS = {"A": {0, 1}, "B": {43, 123}, "C": {3047}}
EXPECTED_SOURCE_SHA256 = (
    "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--account-alias", choices=tuple(ASSIGNMENTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.seed not in ASSIGNMENTS[args.account_alias]:
        raise ValueError("Seed/account assignment mismatch")
    if not args.owner or "/" in args.owner or any(c.isspace() for c in args.owner):
        raise ValueError("Invalid Kaggle dataset owner")

    required = (
        "resume_latest.pt",
        "resume_latest.json",
        "segment_manifest.json",
        "best_val_acc.pt",
        "best_val_acc.json",
        "execution_manifest.json",
    )
    missing = [name for name in required if not (args.dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete resume bundle: {missing}")

    resume = json.loads(
        (args.dataset_dir / "resume_latest.json").read_text(encoding="utf-8")
    )
    segment = json.loads(
        (args.dataset_dir / "segment_manifest.json").read_text(encoding="utf-8")
    )
    if segment.get("status") != "NEEDS_RESUME":
        raise RuntimeError("Previous segment is not resumable")
    checks = {
        "seed": int(json.loads(
            (args.dataset_dir / "config.json").read_text(encoding="utf-8")
        )["seed"]) == args.seed,
        "run_id": resume.get("run_id") == segment.get("run_id"),
        "epoch": int(resume.get("epoch", -1)) == int(segment.get("end_epoch", -2)),
        "source": resume.get("source_hash") == EXPECTED_SOURCE_SHA256,
        "sha256": sha256_file(args.dataset_dir / "resume_latest.pt")
        == resume.get("checkpoint_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Resume bundle identity mismatch: {checks}")

    slug = f"mpg-fer-v2-3-resume-seed-{args.seed}-seg-01"
    metadata = {
        "title": f"MPG-FER v2.3 resume seed {args.seed} segment 01",
        "id": f"{args.owner}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (args.dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "account_alias": args.account_alias,
        "seed": args.seed,
        "dataset_slug_without_owner": slug,
        "run_id": resume["run_id"],
        "resume_epoch": resume["epoch"],
        "resume_sha256": resume["checkpoint_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
