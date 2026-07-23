"""Reject forbidden runtime imports and personal paths."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


FORBIDDEN_MODULES = {"d16", "d17", "d18", "d19"}
FORBIDDEN_TEXT = [
    re.compile(r"D:\\SGU", re.IGNORECASE),
    re.compile(r"sys\.path\.(insert|append)\s*\("),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.package_root.resolve() / "src"
    violations = []
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_TEXT:
            if pattern.search(text):
                violations.append({"file": str(path), "type": "text", "pattern": pattern.pattern})
        tree = ast.parse(text)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                if module.split(".", 1)[0] in FORBIDDEN_MODULES:
                    violations.append({"file": str(path), "type": "import", "module": module})
    print(json.dumps({"pass": not violations, "violations": violations}, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
