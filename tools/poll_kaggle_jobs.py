"""Poll status of cumulative ablation Kaggle jobs A0..A5."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

JOBS = [
    ("A0", "irthn1311", "mpg-fer-cumabl7-opus-a0", None),
    ("A1", "maiduyen311", "mpg-fer-cumabl7-opus-a1", r"D:\Downloads\kaggle (4).json"),
    ("A2", "nuyntai", "mpg-fer-cumabl7-opus-a2", r"D:\Downloads\kaggle (5).json"),
    ("A3", "quangdangnguyen30", "mpg-fer-cumabl7-opus-a3", r"D:\Downloads\kaggle (6).json"),
    ("A4", "nadkli2704", "mpg-fer-cumabl7-opus-a4", r"D:\Downloads\kaggle (7).json"),
    ("A5", "irthn1311", "mpg-fer-cumabl7-opus-a5", None),
]


def check_status() -> dict[str, str]:
    statuses = {}
    for mode, username, slug, cred_file in JOBS:
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
        kernel_ref = f"{username}/{slug}"
        try:
            st = api.kernels_status(kernel_ref)
            status_str = str(st.status)
            statuses[mode] = f"{kernel_ref}: {status_str}"
        except Exception as e:
            statuses[mode] = f"{kernel_ref}: ERROR ({e})"
    return statuses


def main() -> None:
    results = check_status()
    for mode, st in results.items():
        print(f"[{mode}] {st}")


if __name__ == "__main__":
    main()
