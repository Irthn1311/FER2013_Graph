"""Stage and push Kaggle kernel for MPG-FER Upstream Representation and Optimization Audit."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
STAGED_DIR = ROOT / "staged_kaggle" / "mpg_fer_upstream_audit"
STAGED_DIR.mkdir(parents=True, exist_ok=True)

nb_source = ROOT / "notebooks" / "mpg_fer_upstream_audit.ipynb"
nb_target = STAGED_DIR / "mpg_fer_upstream_audit.ipynb"
shutil.copy2(nb_source, nb_target)

metadata = {
    "id": "irthn1311/mpg-fer-upstream-audit-opus",
    "title": "mpg fer upstream audit opus",
    "code_file": "mpg_fer_upstream_audit.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [
        "doduyquynii/fer13-split",
        "irthn1311/mpg-fer-v2-3-seed42-official-checkpoint",
        "irthn1311/mpg-fer-npf-seed42-checkpoint",
    ],
    "competition_sources": [],
    "kernel_sources": [],
}

with open(STAGED_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Staged Upstream Audit kernel at {STAGED_DIR}")

api = KaggleApi()
api.authenticate()
print("Authenticated as:", api.get_config_value("username"))

api.kernels_push(str(STAGED_DIR))
print("Pushed Upstream Audit kernel successfully!")
