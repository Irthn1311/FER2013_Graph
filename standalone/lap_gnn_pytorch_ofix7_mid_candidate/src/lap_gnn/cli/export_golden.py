"""Load and verify the package's portable golden fixture manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    manifest = Path(args.package_root) / "validation_assets" / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Golden manifest not found: {manifest}")
    print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
