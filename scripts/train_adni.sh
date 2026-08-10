#!/usr/bin/env bash
set -euo pipefail

data_root=${1:?"usage: scripts/train_adni.sh /path/to/adni [device] [variant]"}
device_index=${2:-0}
variant=${3:-mofe}

for seed in 0 1 2; do
  python train.py \
    --data adni \
    --variant "$variant" \
    --adni-data-root "$data_root" \
    --seed "$seed" \
    --device "$device_index" \
    --evaluate-test
done
