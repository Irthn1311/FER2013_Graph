"""Stage one Issue #99 kernel variant from the canonical notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    PROJECT_ROOT / "notebooks" / "MPG_FER_v2_3_MultiSeed_Kaggle_T4.ipynb"
)
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))
from mpg_fer_v2_3.checkpoint import canonical_config, config_hash  # noqa: E402
from mpg_fer_v2_3.config import MPGConfig  # noqa: E402


AUTHORIZED_ASSIGNMENTS = {
    "A": {0, 1},
    "B": {43, 123},
    "C": {3047},
}


def _replace_assignment(source: str, name: str, value: str) -> str:
    pattern = rf"(?m)^{re.escape(name)}\s*=.*$"
    updated, count = re.subn(pattern, f"{name} = {value}", source)
    if count != 1:
        raise RuntimeError(f"Expected one {name} assignment, found {count}")
    return updated


def _assert_seed_only_config_delta(seed: int) -> str:
    baseline = canonical_config(MPGConfig())
    candidate = canonical_config(MPGConfig(seed=seed))
    differences = {
        key for key in baseline
        if baseline[key] != candidate[key]
    }
    if differences != {"seed"}:
        raise RuntimeError(
            f"Expected seed-only scientific config delta, got {sorted(differences)}"
        )
    return config_hash(MPGConfig(seed=seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--account-alias", choices=tuple(AUTHORIZED_ASSIGNMENTS), required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--resume-mode", choices=("fresh", "auto", "required"), required=True)
    parser.add_argument("--dirty-worktree", action="store_true")
    parser.add_argument("--dataset", action="append", required=True)
    args = parser.parse_args()

    if args.seed not in AUTHORIZED_ASSIGNMENTS[args.account_alias]:
        raise ValueError(
            f"Seed {args.seed} is not assigned to Account {args.account_alias}"
        )
    if args.segment < 1:
        raise ValueError("segment must be positive")
    if (args.segment == 1) != (args.resume_mode == "fresh"):
        raise ValueError("Segment 1 must be fresh; later segments must not be fresh")
    if re.fullmatch(r"[0-9a-f]{40}", args.git_commit) is None:
        raise ValueError("git-commit must be an exact lowercase 40-hex commit")
    if re.fullmatch(r"[^/\s]+/[^/\s]+", args.kernel_ref) is None:
        raise ValueError("kernel-ref must use owner/slug form")
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(
            f"Canonical notebook missing: run tools/sync_multiseed_notebook.py"
        )

    expected_config_sha = _assert_seed_only_config_delta(args.seed)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    control_cell = notebook["cells"][1]
    source = "".join(control_cell["source"])
    replacements = {
        "SEED": str(args.seed),
        "ACCOUNT_ALIAS": json.dumps(args.account_alias),
        "RESUME_MODE": json.dumps(args.resume_mode),
        "SEGMENT_NUMBER": str(args.segment),
        "KERNEL_REF": json.dumps(args.kernel_ref),
        "GIT_BRANCH": json.dumps(args.git_branch),
        "GIT_COMMIT_SHA": json.dumps(args.git_commit),
        "DIRTY_WORKTREE": "True" if args.dirty_worktree else "False",
        "EXPECTED_CONFIG_SHA": json.dumps(expected_config_sha),
    }
    for name, value in replacements.items():
        source = _replace_assignment(source, name, value)
    control_cell["source"] = source.splitlines(True)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    notebook_name = "MPG_FER_v2_3_MultiSeed_Kaggle_T4.ipynb"
    notebook_path = args.output_dir / notebook_name
    notebook_path.write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )
    owner, _slug = args.kernel_ref.split("/", 1)
    metadata = {
        "id": args.kernel_ref,
        "title": (
            f"MPG-FER v2.3 seed {args.seed} official T4 "
            f"segment {args.segment:02d}"
        ),
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": args.dataset,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "kernel_ref": args.kernel_ref,
        "owner": owner,
        "account_alias": args.account_alias,
        "seed": args.seed,
        "segment": args.segment,
        "resume_mode": args.resume_mode,
        "scientific_config_delta": ["seed"],
        "config_sha256": expected_config_sha,
        "datasets": args.dataset,
        "staged_notebook_sha256": hashlib.sha256(
            notebook_path.read_bytes()
        ).hexdigest(),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
