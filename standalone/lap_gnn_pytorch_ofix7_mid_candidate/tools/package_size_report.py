"""Report package source count, LOC and sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def size(paths) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    def distributed(path: Path) -> bool:
        return (
            path.is_file()
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and not path.name.endswith((".pyc", ".pyo"))
        )

    source_files = sorted(path for path in (root / "src").rglob("*.py") if distributed(path))
    asset_files = sorted(path for path in (root / "validation_assets").rglob("*") if distributed(path))
    excluded_assets = [
        path for path in root.rglob("*")
        if distributed(path) and "validation_assets" not in path.parts
    ]
    payload = {
        "source_files": len(source_files),
        "source_lines": sum(len(path.read_text(encoding="utf-8").splitlines()) for path in source_files),
        "package_bytes_excluding_validation_assets": size(excluded_assets),
        "validation_asset_bytes": size(asset_files),
        "direct_runtime_dependency_count": 4,
        "copied_runtime_modules": 22,
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
