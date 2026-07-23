"""Verify package checksums without parent-repository access."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    failures = []
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    print(f"verified={not failures} failures={len(failures)}")
    if failures:
        raise SystemExit(str(failures))


if __name__ == "__main__":
    main()
