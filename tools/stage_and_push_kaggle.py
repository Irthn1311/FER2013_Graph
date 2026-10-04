"""Stage and push Kaggle kernels for cumulative ablation ladder A0..A5."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
STAGED_DIR = ROOT / "staged_kaggle"

ACCOUNTS = {
    "A0": ("irthn1311", None),
    "A1": ("maiduyen311", r"D:\Downloads\kaggle (4).json"),
    "A2": ("nuyntai", r"D:\Downloads\kaggle (5).json"),
    "A3": ("quangdangnguyen30", r"D:\Downloads\kaggle (6).json"),
    "A4": ("nadkli2704", r"D:\Downloads\kaggle (7).json"),
    "A5": ("irthn1311", None),
}


def stage_kernel(mode: str) -> Path:
    username, cred_file = ACCOUNTS[mode]
    slug = f"mpg-fer-cumabl7-opus-{mode.lower()}"
    target_dir = STAGED_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    nb_source = ROOT / "notebooks" / f"{slug}.ipynb"
    nb_target = target_dir / f"{slug}.ipynb"
    shutil.copy2(nb_source, nb_target)

    metadata = {
        "id": f"{username}/{slug}",
        "title": f"mpg fer cumabl7 opus {mode.lower()}",
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": ["doduyquynii/fer13-split"],
        "competition_sources": [],
        "kernel_sources": [],
    }

    meta_path = target_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return target_dir


def push_kernel(mode: str, target_dir: Path) -> None:
    username, cred_file = ACCOUNTS[mode]
    if cred_file:
        with open(cred_file, "r") as fp:
            creds = json.load(fp)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
    else:
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)

    api = KaggleApi()
    api.authenticate()
    print(f"Authenticated as {api.get_config_value('username')} for {mode}")

    slug = f"mpg-fer-cumabl7-opus-{mode.lower()}"
    print(f"Pushing kernel {username}/{slug} from {target_dir}...")
    api.kernels_push(str(target_dir))
    print(f"Push succeeded for {mode} ({username}/{slug})!")


def main() -> None:
    modes = ["A0", "A1", "A2", "A3", "A4", "A5"]
    for mode in modes:
        target_dir = stage_kernel(mode)
        print(f"Staged {mode} at {target_dir}")
        push_kernel(mode, target_dir)


if __name__ == "__main__":
    main()
