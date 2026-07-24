import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    files = [path for path in root.rglob("*") if path.is_file()]
    print(json.dumps({
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "largest": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
            for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:10]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

