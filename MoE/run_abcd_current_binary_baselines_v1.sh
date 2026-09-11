#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$here/.." && pwd)
python_bin=${PYTHON_BIN:-/home/shaowen/.conda/envs/flex-moe/bin/python}
runner="$here/baseline_runner.py"
evaluator="$here/evaluate_validation_checkpoint.py"
manifest="$workspace/data/abcd_adhd_current_binary_snp_v1_random_missing15/manifest.json"
root=${CAMPAIGN_ROOT:-"$here/abcd_current_binary_baselines_v1"}
models=(flex_moe i2moe moepp_corrected anymod agdic acadiff)

run_validation() {
  local gpu=$1 model=$2 seed=$3 batch dropout lr wd output result log
  case "$model" in
    flex_moe) batch=32; dropout=0.5; lr=0.0001; wd=0 ;;
    i2moe) batch=64; dropout=0.5; lr=0.0001; wd=0 ;;
    moepp_corrected) batch=128; dropout=0.5; lr=0.0001; wd=0 ;;
    anymod|agdic|acadiff) batch=128; dropout=0.3; lr=0.0003; wd=0.0001 ;;
    *) return 2 ;;
  esac
  output="$root/validation/$model"
  result="$output/abcd/${model}_seed${seed}.json"
  log="$root/logs/validation/${model}_seed${seed}.log"
  if [[ -f "$result" ]] && rg -q '"status": "validation_complete"' "$result"; then
    echo "[skip] $model seed=$seed"
    return 0
  fi
  mkdir -p "$output" "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$python_bin" -u "$runner" \
    --model "$model" --data abcd --dataset_manifest "$manifest" --modality IGCB \
    --device 0 --seed "$seed" --train_epochs 50 --warm_up_epochs 5 \
    --early_stopping_patience 10 --early_stopping_min_epochs 8 \
    --batch_size "$batch" --num_workers 2 --pin_memory true --preprocessed true \
    --initial_filling mean --use_common_ids false \
    --hidden_dim 128 --num_patches 16 --num_heads 4 --num_layers_fus 1 --num_layers_pred 1 \
    --num_experts 16 --num_routers 1 --top_k 4 --dropout "$dropout" \
    --lr "$lr" --weight_decay "$wd" --optimizer adamw --lr_scheduler constant --grad_clip 5 \
    --class_weight_power 0 --sampler_power 0 --modality_dropout_prob 0.25 \
    --checkpoint_soup_top_k 1 --validation_only true --allow_overwrite false \
    --output_dir "$output" >"$log" 2>&1
}

run_formal() {
  local gpu=$1 model=$2 seed=$3 source output log
  source="$root/validation/$model/abcd/${model}_seed${seed}.json"
  output="$root/formal/$model"
  log="$root/logs/formal/${model}_seed${seed}.log"
  mkdir -p "$output" "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$python_bin" -u "$evaluator" --source-result "$source" --output-root "$output" --device 0 \
    >"$log" 2>&1
}

worker() {
  local gpu=$1 seed=$2 lane=$3 idx model
  for idx in "${!models[@]}"; do
    if (( idx % 2 == lane )); then
      model=${models[$idx]}
      echo "[validation] $model seed=$seed gpu=$gpu"
      run_validation "$gpu" "$model" "$seed"
    fi
  done
  for idx in "${!models[@]}"; do
    if (( idx % 2 == lane )); then
      model=${models[$idx]}
      echo "[formal] $model seed=$seed gpu=$gpu"
      run_formal "$gpu" "$model" "$seed"
    fi
  done
}

mkdir -p "$root/logs"
worker 0 31 0 & pids=("$!")
worker 0 31 1 & pids+=("$!")
worker 1 32 0 & pids+=("$!")
worker 1 32 1 & pids+=("$!")
worker 2 33 0 & pids+=("$!")
worker 2 33 1 & pids+=("$!")
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
[[ "$status" -eq 0 ]] || { echo "Baseline campaign failed" >&2; exit 1; }
"$python_bin" "$here/summarize_abcd_current_binary_baselines_v1.py" --campaign-root "$root"
