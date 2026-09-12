#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-/home/shaowen/.conda/envs/flex-moe/bin/python}
root=${CAMPAIGN_ROOT:-"$here/adni_single_expert_control_v1"}
runner="$here/baseline_runner.py"
evaluator="$here/evaluate_validation_checkpoint.py"

[[ ! -e "$root/run_complete.json" ]] || { echo "Refusing completed root: $root" >&2; exit 2; }

run_seed() {
  local gpu=$1 seed=$2 source
  mkdir -p "$root/validation" "$root/formal" "$root/logs"
  echo "[start] validation seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
  (cd "$here" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
    LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
    "$python_bin" -u "$runner" \
      --model our_moe --data adni --modality IGCB --adni_image_imputation legacy_mode \
      --device 0 --seed "$seed" --train_epochs 50 --warm_up_epochs 5 \
      --early_stopping_patience 0 --early_stopping_min_epochs 15 \
      --batch_size 32 --num_workers 2 --pin_memory true --preprocessed true \
      --initial_filling mean --use_common_ids false \
      --hidden_dim 128 --num_patches 16 --num_heads 4 \
      --num_layers_fus 1 --num_layers_pred 1 --dropout 0.30 \
      --lr 0.000125 --weight_decay 0 --optimizer adamw --lr_scheduler constant --grad_clip 5 \
      --num_experts 1 --num_routers 1 --top_k 1 --patch_adapter_rank 4 \
      --class_weight_power 0 --sampler_power 0 \
      --use_generators true --generator_task_grad false --generator_only_task_grad false \
      --recon_loss_weight 1 --branch_aux_loss_weight 0.1 --gate_loss_weight 0.01 \
      --modality_dropout_prob 0.25 --drop_ce_loss_weight 0.1 \
      --distill_loss_weight 0.15 --distill_temperature 2 \
      --more_fewer_rank_loss_weight 0.1 --branch_confidence_mode evidence \
      --vectorized_generation true --recon_targets_per_sample 4 \
      --recompute_dropped_combination false --clear_eval_gate_cache true \
      --checkpoint_soup_top_k 1 --validation_only true --allow_overwrite false \
      --output_dir "$root/validation") >"$root/logs/validation_seed${seed}.log" 2>&1
  source=$(find "$root/validation/adni" -maxdepth 1 -type f \
    -name "our_moe_seed${seed}*.json" ! -name '*.progress.json' | head -n 1)
  echo "[start] formal seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
  (cd "$here" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
    LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$python_bin" -u "$evaluator" --source-result "$source" \
      --output-root "$root/formal" --device 0) >"$root/logs/formal_seed${seed}.log" 2>&1
  echo "[done] seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
}

run_seed 0 0 & pids=("$!")
run_seed 1 1 & pids+=("$!")
run_seed 2 2 & pids+=("$!")
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
[[ "$status" -eq 0 ]] || { echo "ADNI single-expert control failed" >&2; exit 1; }

"$python_bin" - "$root" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); metrics=("accuracy","macro_f1","macro_auroc")
rows=[]
for seed in (0,1,2):
    d=json.loads((root/"formal"/"adni"/f"our_moe_seed{seed}.json").read_text())
    row={"seed":seed,"best_epoch":d["training"]["best_epoch"],"epochs_completed":d["training"]["epochs_completed"]}
    row.update({m:100*float(d["test"][m]) for m in metrics}); rows.append(row)
out={"schema":"adni-single-expert-control-v1","aggregation":"three-seed arithmetic mean and sample SD; no ensemble","seed_metrics":rows,"aggregate":{m:{"mean":float(np.mean([r[m] for r in rows])),"sd":float(np.std([r[m] for r in rows],ddof=1))} for m in metrics}}
(root/"results.json").write_text(json.dumps(out,indent=2)+"\n")
(root/"run_complete.json").write_text(json.dumps({"status":"complete"},indent=2)+"\n")
print(*(f'{m}={out["aggregate"][m]["mean"]:.2f}±{out["aggregate"][m]["sd"]:.2f}' for m in metrics))
PY
