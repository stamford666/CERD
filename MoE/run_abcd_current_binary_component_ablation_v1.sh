#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$here/.." && pwd)
python_bin=${PYTHON_BIN:-/home/shaowen/.conda/envs/flex-moe/bin/python}
runner="$here/baseline_runner.py"
evaluator="$here/evaluate_validation_checkpoint.py"
manifest="$workspace/data/abcd_adhd_current_binary_snp_v1_random_missing15/manifest.json"
root=${CAMPAIGN_ROOT:-"$here/abcd_current_binary_component_ablation_v1"}

# Every control changes only the named component from the validation-selected
# binary CERD configuration (P=16, eight experts, top-2 routing).
arms=(dense_backbone single_expert no_multigranular_decomposition no_completion no_provenance uniform_branch_weights)

arm_args() {
  case "$1" in
    dense_backbone) printf '%s\n' --dense_backbone true ;;
    single_expert) printf '%s\n' --num_experts 1 --top_k 1 ;;
    no_multigranular_decomposition)
      printf '%s\n' \
        --joint_branch_only true \
        --branch_aux_loss_weight 0 \
        --modality_dropout_prob 0 \
        --distill_loss_weight 0 \
        --drop_ce_loss_weight 0 \
        --more_fewer_rank_loss_weight 0
      ;;
    no_completion) printf '%s\n' --disable_completion true ;;
    no_provenance) printf '%s\n' --disable_provenance true ;;
    uniform_branch_weights) printf '%s\n' --uniform_branch_weights true ;;
    *) echo "Unknown arm: $1" >&2; return 2 ;;
  esac
}

run_validation() {
  local gpu=$1 arm=$2 seed=$3 output existing log
  output="$root/$arm/validation"
  log="$root/logs/$arm/validation_seed${seed}.log"
  existing=$(find "$output/abcd" -maxdepth 1 -type f -name "our_moe_seed${seed}*.json" ! -name '*.progress.json' 2>/dev/null | head -n 1 || true)
  if [[ -n "$existing" ]] && rg -q '"status": "validation_complete"' "$existing"; then
    echo "[skip validation] arm=$arm seed=$seed"
    return 0
  fi
  mkdir -p "$output" "$(dirname "$log")"
  mapfile -t extra < <(arm_args "$arm")
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$python_bin" -u "$runner" \
    --model our_moe --data abcd --dataset_manifest "$manifest" --modality IGCB \
    --device 0 --seed "$seed" --train_epochs 50 --warm_up_epochs 5 \
    --early_stopping_patience 10 --early_stopping_min_epochs 8 \
    --batch_size 64 --num_workers 2 --pin_memory true --preprocessed true \
    --initial_filling mean --use_common_ids false \
    --hidden_dim 128 --num_patches 16 --num_heads 4 \
    --num_layers_fus 1 --num_layers_pred 1 --dropout 0.30 \
    --lr 0.0001 --weight_decay 0.01 --optimizer adamw --lr_scheduler constant --grad_clip 5 \
    --num_experts 8 --num_routers 1 --top_k 2 --patch_adapter_rank 4 \
    --class_weight_power 0 --sampler_power 0 \
    --use_generators true --generator_task_grad false --generator_only_task_grad false \
    --recon_loss_weight 0.25 --branch_aux_loss_weight 0.1 --gate_loss_weight 0.01 \
    --modality_dropout_prob 0.25 --drop_ce_loss_weight 0.1 \
    --distill_loss_weight 0.15 --distill_temperature 2 \
    --more_fewer_rank_loss_weight 0.1 --branch_confidence_mode entropy_detached \
    --vectorized_generation true --recon_targets_per_sample 4 \
    --recompute_dropped_combination false --clear_eval_gate_cache true \
    --checkpoint_soup_top_k 1 --validation_only true --allow_overwrite false \
    "${extra[@]}" --output_dir "$output" >"$log" 2>&1
}

run_formal() {
  local gpu=$1 arm=$2 seed=$3 source output existing log
  source=$(find "$root/$arm/validation/abcd" -maxdepth 1 -type f -name "our_moe_seed${seed}*.json" ! -name '*.progress.json' | head -n 1)
  output="$root/$arm/formal"
  log="$root/logs/$arm/formal_seed${seed}.log"
  existing=$(find "$output/abcd" -maxdepth 1 -type f -name "our_moe_seed${seed}*.json" 2>/dev/null | head -n 1 || true)
  if [[ -n "$existing" ]] && rg -q '"status": "complete"' "$existing"; then
    echo "[skip formal] arm=$arm seed=$seed"
    return 0
  fi
  mkdir -p "$output" "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$python_bin" -u "$evaluator" --source-result "$source" --output-root "$output" --device 0 \
    >"$log" 2>&1
}

worker() {
  local gpu=$1 seed=$2 lane=$3 idx arm
  for idx in "${!arms[@]}"; do
    if (( idx % 2 == lane )); then
      arm=${arms[$idx]}
      echo "[validation] arm=$arm seed=$seed gpu=$gpu"
      run_validation "$gpu" "$arm" "$seed"
    fi
  done
  for idx in "${!arms[@]}"; do
    if (( idx % 2 == lane )); then
      arm=${arms[$idx]}
      echo "[formal] arm=$arm seed=$seed gpu=$gpu"
      run_formal "$gpu" "$arm" "$seed"
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
[[ "$status" -eq 0 ]] || { echo "Component-ablation campaign failed" >&2; exit 1; }
"$python_bin" "$here/summarize_abcd_current_binary_component_ablation_v1.py" --campaign-root "$root"
