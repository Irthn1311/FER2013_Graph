"""Verify package checksums without parent-repository access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sha256",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return data
    text = data.decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    failures = []
    checked = 0
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checked += 1
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = hashlib.sha256(canonical_bytes(path)).hexdigest() if path.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    result = {
        "verified": not failures,
        "checked_files": checked,
        "failure_count": len(failures),
        "failures": failures,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"checksum_verification={'PASS' if not failures else 'FAIL'} "
        f"checked={checked} failures={len(failures)}"
    )
    if failures:
        for failure in failures[:5]:
            print(f"  mismatch: {failure['path']}")
        if len(failures) > 5:
            print(f"  ... {len(failures) - 5} additional mismatches")
        if args.report:
            print(f"  full_report: {args.report}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
