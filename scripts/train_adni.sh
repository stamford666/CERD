#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: scripts/train_adni.sh DATA_ROOT [DEVICE] [EVALUATE_TEST] [OUTPUT_DIR]" >&2
  exit 2
fi

data_root=$1
device_index=${2:-0}
evaluate_test=${3:-false}
output_dir=${4:-outputs/adni}

case "$evaluate_test" in
  true) test_flag=--evaluate-test ;;
  false) test_flag=--no-evaluate-test ;;
  *) echo "evaluate_test must be true or false" >&2; exit 2 ;;
esac

for seed in 0 1 2; do
  python train.py \
    --data adni \
    --variant mofe \
    --adni-data-root "$data_root" \
    --output-dir "$output_dir" \
    --no-pattern-aware-reconstruction \
    --recon-context-dropout-probability 0 \
    --recon-normalized-token-loss-weight 0 \
    --branch-confidence-mode evidence \
    --generator-output-gate \
    --no-generator-only-task-grad \
    --seed "$seed" \
    --device "$device_index" \
    "$test_flag"
done
