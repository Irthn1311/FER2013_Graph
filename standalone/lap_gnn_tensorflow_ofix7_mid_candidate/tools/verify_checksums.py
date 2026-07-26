from __future__ import annotations

import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    checksum_file = root / "CHECKSUMS.sha256"
    failures = []
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        checked += 1
        if actual != expected:
            failures.append(relative)
    print(f"{'PASS' if not failures else 'FAIL'} checked={checked} failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

