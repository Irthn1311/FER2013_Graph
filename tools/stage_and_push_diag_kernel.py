"""Stage and push Kaggle diagnostic kernel for MPG-FER Readout Diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
STAGED_DIR = ROOT / "staged_kaggle" / "mpg_fer_readout_diagnostic"
STAGED_DIR.mkdir(parents=True, exist_ok=True)

nb_source = ROOT / "notebooks" / "mpg_fer_readout_diagnostic.ipynb"
nb_target = STAGED_DIR / "mpg_fer_readout_diagnostic.ipynb"
shutil.copy2(nb_source, nb_target)

metadata = {
    "id": "irthn1311/mpg-fer-readout-diagnostic-opus",
    "title": "mpg fer readout diagnostic opus",
    "code_file": "mpg_fer_readout_diagnostic.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [
        "doduyquynii/fer13-split",
        "irthn1311/mpg-fer-v2-3-seed42-official-checkpoint",
    ],
    "competition_sources": [],
    "kernel_sources": [],
}

with open(STAGED_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Staged diagnostic kernel at {STAGED_DIR}")

api = KaggleApi()
api.authenticate()
print("Authenticated as:", api.get_config_value("username"))

api.kernels_push(str(STAGED_DIR))
print("Pushed diagnostic kernel successfully!")
