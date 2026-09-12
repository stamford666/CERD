#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-/home/shaowen/.conda/envs/flex-moe/bin/python}
runner="$here/baseline_runner.py"
evaluator="$here/evaluate_validation_checkpoint.py"
manifest=${DATASET_MANIFEST:-/data/mi2/shaowen/miccai2026/data/abcd_adhd_current3_snp_v2_n3000_random_missing15/manifest.json}
root=${CAMPAIGN_ROOT:-/data/mi2/shaowen/miccai2026/MoE/abcd_current3_component_suite_v1}
arms=(base plus_completion plus_provenance dense_full no_completion no_provenance no_decomposition uniform_weights single_expert)
seeds=(31 32 33)

[[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 2; }
[[ ! -e "$root/run_complete.json" ]] || { echo "Refusing completed root: $root" >&2; exit 2; }
mkdir -p "$root/logs/validation" "$root/logs/formal"

arm_args() {
  case "$1" in
    base)
      printf '%s\n' --dense_backbone true --disable_completion true \
        --disable_provenance true --joint_branch_only true \
        --recon_loss_weight 0 --branch_aux_loss_weight 0 \
        --modality_dropout_prob 0 --drop_ce_loss_weight 0 \
        --distill_loss_weight 0 --more_fewer_rank_loss_weight 0
      ;;
    plus_completion)
      printf '%s\n' --dense_backbone true --disable_provenance true \
        --joint_branch_only true --branch_aux_loss_weight 0 \
        --modality_dropout_prob 0 --drop_ce_loss_weight 0 \
        --distill_loss_weight 0 --more_fewer_rank_loss_weight 0
      ;;
    plus_provenance)
      printf '%s\n' --dense_backbone true --joint_branch_only true \
        --branch_aux_loss_weight 0 --modality_dropout_prob 0 \
        --drop_ce_loss_weight 0 --distill_loss_weight 0 \
        --more_fewer_rank_loss_weight 0
      ;;
    dense_full)
      printf '%s\n' --dense_backbone true
      ;;
    no_completion)
      printf '%s\n' --disable_completion true
      ;;
    no_provenance)
      printf '%s\n' --disable_provenance true
      ;;
    no_decomposition)
      printf '%s\n' --joint_branch_only true --branch_aux_loss_weight 0 \
        --modality_dropout_prob 0 --drop_ce_loss_weight 0 \
        --distill_loss_weight 0 --more_fewer_rank_loss_weight 0
      ;;
    uniform_weights)
      printf '%s\n' --uniform_branch_weights true
      ;;
    single_expert)
      printf '%s\n' --num_experts 1 --top_k 1
      ;;
    *) echo "Unknown arm: $1" >&2; return 2 ;;
  esac
}

run_validation() {
  local gpu=$1 seed=$2 arm=$3 output log
  output="$root/$arm/validation"
  log="$root/logs/validation/${arm}_seed${seed}.log"
  mkdir -p "$output"
  mapfile -t extra < <(arm_args "$arm")
  echo "[start] phase=validation arm=$arm seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$python_bin" -u "$runner" \
    --model our_moe --data abcd --dataset_manifest "$manifest" --modality IGCB \
    --device 0 --seed "$seed" --train_epochs 100 --warm_up_epochs 5 \
    --early_stopping_patience 0 --early_stopping_min_epochs 0 \
    --batch_size 64 --num_workers 2 --pin_memory true --preprocessed true \
    --initial_filling mean --use_common_ids false \
    --hidden_dim 128 --num_patches 8 --num_heads 4 \
    --num_layers_fus 1 --num_layers_pred 1 --dropout 0.35 \
    --lr 0.0001 --weight_decay 0.01 --optimizer adamw --lr_scheduler constant --grad_clip 5 \
    --num_experts 16 --num_routers 1 --top_k 4 --patch_adapter_rank 4 \
    --class_weight_power 0.75 --sampler_power 0 \
    --use_generators true --generator_task_grad false --generator_only_task_grad false \
    --recon_loss_weight 0.5 --branch_aux_loss_weight 0.1 --gate_loss_weight 0.01 \
    --modality_dropout_prob 0.25 --drop_ce_loss_weight 0.1 \
    --distill_loss_weight 0.15 --distill_temperature 2 \
    --more_fewer_rank_loss_weight 0.1 --branch_confidence_mode entropy_detached \
    --vectorized_generation true --recon_targets_per_sample 4 \
    --recompute_dropped_combination false --clear_eval_gate_cache true \
    --checkpoint_soup_top_k 1 --validation_only true --allow_overwrite false \
    "${extra[@]}" --output_dir "$output" >"$log" 2>&1
  echo "[done] phase=validation arm=$arm seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
}

run_formal() {
  local gpu=$1 seed=$2 arm=$3 source output log
  source=$(find "$root/$arm/validation/abcd" -maxdepth 1 -type f \
    -name "our_moe_seed${seed}*.json" ! -name '*.progress.json' | head -n 1)
  output="$root/$arm/formal"
  log="$root/logs/formal/${arm}_seed${seed}.log"
  mkdir -p "$output"
  echo "[start] phase=formal arm=$arm seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$python_bin" -u "$evaluator" --source-result "$source" \
    --output-root "$output" --device 0 >"$log" 2>&1
  echo "[done] phase=formal arm=$arm seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
}

worker() {
  local gpu=$1 seed=$2 lane=$3 phase=$4 idx arm
  for idx in "${!arms[@]}"; do
    if (( idx % 2 == lane )); then
      arm=${arms[$idx]}
      if [[ "$phase" == validation ]]; then
        run_validation "$gpu" "$seed" "$arm"
      else
        run_formal "$gpu" "$seed" "$arm"
      fi
    fi
  done
}

run_phase() {
  local phase=$1 status=0 pids=()
  worker 0 31 0 "$phase" & pids+=("$!")
  worker 0 31 1 "$phase" & pids+=("$!")
  worker 1 32 0 "$phase" & pids+=("$!")
  worker 1 32 1 "$phase" & pids+=("$!")
  worker 2 33 0 "$phase" & pids+=("$!")
  worker 2 33 1 "$phase" & pids+=("$!")
  for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
  [[ "$status" -eq 0 ]]
}

run_phase validation || { echo "Validation phase failed" >&2; exit 1; }
run_phase formal || { echo "Formal phase failed" >&2; exit 1; }

"$python_bin" - "$root" "$manifest" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
arms = (
    "base", "plus_completion", "plus_provenance", "dense_full",
    "no_completion", "no_provenance", "no_decomposition",
    "uniform_weights", "single_expert",
)
seeds = (31, 32, 33)
metrics = ("accuracy", "macro_f1", "macro_auroc")
out = {
    "schema": "abcd-current3-component-suite-v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "dataset_manifest": str(manifest),
    "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "aggregation": "arithmetic mean and sample standard deviation across seeds 31/32/33; no ensemble",
    "selection": "best validation Macro-F1 after 100 complete epochs; test evaluated once",
    "arms": {},
}
for arm in arms:
    payloads = [
        json.loads((root / arm / "formal" / "abcd" / f"our_moe_seed{seed}.json").read_text())
        for seed in seeds
    ]
    rows = []
    for seed, payload in zip(seeds, payloads):
        row = {"seed": seed, "best_epoch": int(payload["training"]["best_epoch"]), "epochs_completed": int(payload["training"]["epochs_completed"])}
        row.update({metric: 100.0 * float(payload["test"][metric]) for metric in metrics})
        rows.append(row)
    block = {
        "seed_metrics": rows,
        "aggregate": {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "sd": float(np.std([row[metric] for row in rows], ddof=1)),
            }
            for metric in metrics
        },
        "test_subsets": {},
    }
    for subset in ("complete", "missing"):
        block["test_subsets"][subset] = {
            "n": int(payloads[0]["test_subsets"][subset]["n"]),
            **{
                metric: {
                    "mean": float(np.mean([100.0 * float(p["test_subsets"][subset][metric]) for p in payloads])),
                    "sd": float(np.std([100.0 * float(p["test_subsets"][subset][metric]) for p in payloads], ddof=1)),
                }
                for metric in metrics
            },
        }
    out["arms"][arm] = block
(root / "results.json").write_text(json.dumps(out, indent=2) + "\n")
(root / "run_complete.json").write_text(json.dumps({"status": "complete", "completed_at": out["completed_at"]}, indent=2) + "\n")
for arm, block in out["arms"].items():
    print(arm, *(f'{block["aggregate"][metric]["mean"]:.2f}±{block["aggregate"][metric]["sd"]:.2f}' for metric in metrics))
PY
