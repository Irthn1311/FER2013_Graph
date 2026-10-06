"""Generate environment.json and checksums.sha256 for Composer Audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "analysis" / "mpg_fer_composer_audit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    env_info = {
        "schema_version": 1,
        "python": sys.version,
        "base_commit": "c143ba2c2eeee5ed088984461f0ad9be42d373bb",
        "branch": "research/mpg-fer-composer-audit",
        "diagnostic_namespace": "analysis/mpg_fer_composer_audit/",
        "canonical_checkpoint_sha256": "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e",
        "npf_checkpoint_sha256": "f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972",
        "val_samples": 3589,
        "train_samples": 28709,
    }
    with open(AUDIT_DIR / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    chk_lines = []
    for f in sorted(AUDIT_DIR.glob("*")):
        if f.is_file() and f.name != "checksums.sha256" and not f.name.endswith(".zip"):
            chk_lines.append(f"{sha256_file(f)}  {f.name}")
    with open(AUDIT_DIR / "checksums.sha256", "w", encoding="utf-8") as f:
        f.write("\n".join(chk_lines) + "\n")
    print("Wrote environment.json and checksums.sha256")


if __name__ == "__main__":
    main()
