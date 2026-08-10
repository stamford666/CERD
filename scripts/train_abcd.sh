#!/usr/bin/env bash
set -euo pipefail

manifest_path=${1:?"usage: scripts/train_abcd.sh /path/to/abcd_manifest.json [device]"}
device_index=${2:-0}

for seed in 0 1 2; do
  python train.py \
    --data abcd \
    --variant dbr \
    --dataset-manifest "$manifest_path" \
    --seed "$seed" \
    --device "$device_index" \
    --evaluate-test
done
