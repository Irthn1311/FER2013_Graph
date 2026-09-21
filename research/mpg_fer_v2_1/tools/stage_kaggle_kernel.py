"""Stage a private source-locked Kaggle kernel without credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "MPG_FER_v2_1_Kaggle_T4.ipynb"


def _replace_assignment(source: str, name: str, value: str) -> str:
    pattern = rf"(?m)^{re.escape(name)}\s*=.*$"
    updated, count = re.subn(pattern, f"{name} = {value}", source)
    if count != 1:
        raise RuntimeError(f"Expected one {name} assignment, found {count}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--resume-mode", choices=("fresh", "auto", "required"), required=True)
    parser.add_argument("--dataset", action="append", required=True)
    args = parser.parse_args()
    if args.segment < 1:
        raise ValueError("segment must be positive")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    config_cell = notebook["cells"][1]
    source = "".join(config_cell["source"])
    source = _replace_assignment(source, "RESUME_MODE", json.dumps(args.resume_mode))
    source = _replace_assignment(source, "SEGMENT_NUMBER", str(args.segment))
    source = _replace_assignment(source, "KERNEL_REF", json.dumps(args.kernel_ref))
    source = _replace_assignment(source, "GIT_COMMIT_SHA", json.dumps(args.git_commit))
    config_cell["source"] = source.splitlines(True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    notebook_name = "MPG_FER_v2_1_Kaggle_T4.ipynb"
    notebook_path = args.output_dir / notebook_name
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    owner, slug = args.kernel_ref.split("/", 1)
    metadata = {
        "id": args.kernel_ref,
        "title": f"MPG-FER v2.1 official T4 segment {args.segment:02d}",
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
        "segment": args.segment,
        "resume_mode": args.resume_mode,
        "datasets": args.dataset,
        "staged_notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
