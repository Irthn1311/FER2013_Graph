import argparse
import json
from pathlib import Path

from lap_gnn_tf.conversion.export_tensorflow_weights import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--graph-batch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.state, args.graph_batch, args.output), indent=2))


if __name__ == "__main__":
    main()

