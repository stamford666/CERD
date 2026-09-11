#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$here/.." && pwd)
python_bin=${PYTHON_BIN:-/home/shaowen/.conda/envs/flex-moe/bin/python}
runner="$here/baseline_runner.py"
evaluator="$here/evaluate_validation_checkpoint.py"
manifest="$workspace/data/abcd_adhd_current_binary_snp_v1_random_missing15/manifest.json"
root=${CAMPAIGN_ROOT:-"$here/abcd_current_binary_cerd_v1"}

names=(p8d030cw0 p8d035cw0 p16d030cw0 p16d035cw0 p8d030cw05 p16d030cw05 p8d030e8k2 p16d030e8k2)
patches=(8 8 16 16 8 16 8 16)
dropouts=(0.30 0.35 0.30 0.35 0.30 0.30 0.30 0.30)
class_weights=(0 0 0 0 0.5 0.5 0 0)
experts=(16 16 16 16 16 16 8 8)
topks=(4 4 4 4 4 4 2 2)

run_one() {
  local gpu=$1 seed=$2 idx=$3 name output result log
  name=${names[$idx]}
  output="$root/validation/$name"
  result="$output/abcd/our_moe_seed${seed}.json"
  log="$root/logs/validation/${name}_seed${seed}.log"
  if [[ -f "$result" ]] && rg -q '"status": "validation_complete"' "$result"; then
    echo "[skip] $name seed=$seed"
    return 0
  fi
  mkdir -p "$output" "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$python_bin" -u "$runner" \
    --model our_moe --data abcd --dataset_manifest "$manifest" --modality IGCB \
    --device 0 --seed "$seed" --train_epochs 50 --warm_up_epochs 5 \
    --early_stopping_patience 10 --early_stopping_min_epochs 8 \
    --batch_size 64 --num_workers 2 --pin_memory true --preprocessed true \
    --initial_filling mean --use_common_ids false \
    --hidden_dim 128 --num_patches "${patches[$idx]}" --num_heads 4 \
    --num_layers_fus 1 --num_layers_pred 1 --dropout "${dropouts[$idx]}" \
    --lr 0.0001 --weight_decay 0.01 --optimizer adamw --lr_scheduler constant --grad_clip 5 \
    --num_experts "${experts[$idx]}" --num_routers 1 --top_k "${topks[$idx]}" --patch_adapter_rank 4 \
    --class_weight_power "${class_weights[$idx]}" --sampler_power 0 \
    --use_generators true --generator_task_grad false --generator_only_task_grad false \
    --recon_loss_weight 0.25 --branch_aux_loss_weight 0.1 --gate_loss_weight 0.01 \
    --modality_dropout_prob 0.25 --drop_ce_loss_weight 0.1 \
    --distill_loss_weight 0.15 --distill_temperature 2 \
    --more_fewer_rank_loss_weight 0.1 --branch_confidence_mode entropy_detached \
    --vectorized_generation true --recon_targets_per_sample 4 \
    --recompute_dropped_combination false --clear_eval_gate_cache true \
    --checkpoint_soup_top_k 1 --validation_only true --allow_overwrite false \
    --output_dir "$output" >"$log" 2>&1
}

worker() {
  local gpu=$1 seed=$2 lane=$3 idx
  for idx in "${!names[@]}"; do
    if (( idx % 2 == lane )); then
      echo "[validation] ${names[$idx]} seed=$seed gpu=$gpu"
      run_one "$gpu" "$seed" "$idx"
    fi
  done
}

mkdir -p "$root/logs/validation"
worker 0 31 0 & pids=("$!")
worker 0 31 1 & pids+=("$!")
worker 1 32 0 & pids+=("$!")
worker 1 32 1 & pids+=("$!")
worker 2 33 0 & pids+=("$!")
worker 2 33 1 & pids+=("$!")
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
[[ "$status" -eq 0 ]] || { echo "Validation screen failed" >&2; exit 1; }

"$python_bin" "$here/summarize_abcd_current_binary_cerd_v1.py" --campaign-root "$root"
best=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' "$root/selection.json")

promote() {
  local gpu=$1 seed=$2 source output log
  source="$root/validation/$best/abcd/our_moe_seed${seed}.json"
  output="$root/formal"
  log="$root/logs/formal/seed${seed}.log"
  mkdir -p "$output" "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$python_bin" -u "$evaluator" --source-result "$source" --output-root "$output" --device 0 \
    >"$log" 2>&1
}
promote 0 31 & pids=("$!")
promote 1 32 & pids+=("$!")
promote 2 33 & pids+=("$!")
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
[[ "$status" -eq 0 ]] || { echo "Formal promotion failed" >&2; exit 1; }
"$python_bin" "$here/summarize_abcd_current_binary_cerd_v1.py" --campaign-root "$root"
