from __future__ import annotations

import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "src" / "lap_gnn_tf"
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "torch" or name.startswith("torch."):
                    violations.append({"file": str(path.relative_to(root)), "module": name})
    result = {"runtime_imports_torch": bool(violations), "violations": violations}
    print(json.dumps(result, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

