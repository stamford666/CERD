#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: scripts/train_abcd.sh MANIFEST [DEVICE] [EVALUATE_TEST] [OUTPUT_DIR]" >&2
  exit 2
fi

manifest_path=$1
device_index=${2:-0}
evaluate_test=${3:-false}
output_dir=${4:-outputs/abcd_adhd}

case "$evaluate_test" in
  true) test_flag=--evaluate-test ;;
  false) test_flag=--no-evaluate-test ;;
  *) echo "evaluate_test must be true or false" >&2; exit 2 ;;
esac

for seed in 0 1 2; do
  python train.py \
    --data abcd \
    --variant mofe \
    --modality SRDGNPME \
    --dataset-manifest "$manifest_path" \
    --output-dir "$output_dir" \
    --train-epochs 50 \
    --warm-up-epochs 5 \
    --batch-size 64 \
    --lr 0.0001 \
    --weight-decay 0.0 \
    --sampler-power 0.35 \
    --class-weight-power 0.15 \
    --num-layers-pred 2 \
    --no-pattern-aware-reconstruction \
    --recon-context-dropout-probability 0 \
    --recon-normalized-token-loss-weight 0 \
    --branch-confidence-mode evidence \
    --generator-output-gate \
    --no-generator-only-task-grad \
    --more-fewer-rank-loss-weight 0.1 \
    --dual-boundary-rank-loss-weight 0.0 \
    --seed "$seed" \
    --device "$device_index" \
    "$test_flag"
done
