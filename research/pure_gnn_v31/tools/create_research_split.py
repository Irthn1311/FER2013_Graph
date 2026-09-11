"""CLI tool to generate research split manifest strictly from train.csv."""

import argparse
import json
from pathlib import Path
from pure_gnn_v31.data import create_research_split_manifest, assert_not_test_access


def main():
    parser = argparse.ArgumentParser(description="Create ResearchTrain / ResearchDev split manifest")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to official train.csv")
    parser.add_argument("--seed", type=int, default=42, help="Split random seed")
    parser.add_argument("--dev-ratio", type=float, default=0.15, help="Ratio for dev split")
    parser.add_argument("--output", type=str, required=True, help="Output manifest JSON path")
    args = parser.parse_args()

    assert_not_test_access(args.train_csv)
    manifest = create_research_split_manifest(
        train_csv_path=args.train_csv,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        output_manifest_path=args.output,
    )
    print(f"Research split created successfully:")
    print(f"  Source: {manifest['source_train_csv']}")
    print(f"  Source SHA256: {manifest['source_train_csv_sha256']}")
    print(f"  Train samples: {manifest['research_train_count']}")
    print(f"  Dev samples: {manifest['research_dev_count']}")
    print(f"  Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
