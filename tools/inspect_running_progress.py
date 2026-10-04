"""Inspect running progress of Kaggle cumulative ablation jobs A0..A5."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi

JOBS = [
    ("A0", "irthn1311", "mpg-fer-cumabl7-opus-a0", None),
    ("A1", "maiduyen311", "mpg-fer-cumabl7-opus-a1", r"D:\Downloads\kaggle (4).json"),
    ("A2", "nuyntai", "mpg-fer-cumabl7-opus-a2", r"D:\Downloads\kaggle (5).json"),
    ("A3", "quangdangnguyen30", "mpg-fer-cumabl7-opus-a3", r"D:\Downloads\kaggle (6).json"),
    ("A4", "nadkli2704", "mpg-fer-cumabl7-opus-a4", r"D:\Downloads\kaggle (7).json"),
    ("A5", "irthn1311", "mpg-fer-cumabl7-opus-a5", None),
]


def inspect_all() -> None:
    temp_base = Path(r"C:\Users\ADMIN\AppData\Local\Temp\opencode\inspect_logs")
    temp_base.mkdir(parents=True, exist_ok=True)

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
            out_dir = temp_base / mode
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{mode}] {kernel_ref}: {status_str}")
            try:
                api.kernels_output(kernel_ref, path=str(out_dir))
                for f in out_dir.glob("*.log"):
                    txt = f.read_text(encoding="utf-8", errors="replace")
                    # Search for Epoch lines
                    epoch_lines = [line for line in txt.splitlines() if "Epoch" in line]
                    if epoch_lines:
                        print(f"    Latest: {epoch_lines[-1]}")
                    else:
                        tail = "\n".join(txt.splitlines()[-5:])
                        print(f"    Tail: {tail}")
            except Exception as e:
                print(f"    Output fetch note: {e}")
        except Exception as e:
            print(f"[{mode}] {kernel_ref}: ERROR ({e})")


if __name__ == "__main__":
    inspect_all()
