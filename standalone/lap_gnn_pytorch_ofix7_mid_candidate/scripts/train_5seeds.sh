#!/usr/bin/env bash
set -euo pipefail
for seed in 42 1009 1337 777 3407; do
  "$(dirname "$0")/train_seed.sh" "$seed" "$1" "$2" "$3" "${4:-cuda:0}" "${5:-2}"
done
