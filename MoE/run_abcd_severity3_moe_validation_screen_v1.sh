#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-/home/shaowen/.conda/envs/flex-moe/bin/python}
runner="$here/baseline_runner.py"
manifest=${DATASET_MANIFEST:-/data/mi2/shaowen/miccai2026/data/abcd_adhd_severity3_snp_v1_n3000_random_missing15/manifest.json}
root=${CAMPAIGN_ROOT:-/data/mi2/shaowen/miccai2026/MoE/abcd_severity3_moe_validation_screen_v1}
configs=(e4k2 e8k2)
seeds=(31 32 33)

[[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 2; }
mkdir -p "$root/logs"

run_one() {
  local gpu=$1 seed=$2 config=$3 experts topk output log
  case "$config" in
    e4k2) experts=4; topk=2 ;;
    e8k2) experts=8; topk=2 ;;
    *) echo "Unknown config: $config" >&2; return 2 ;;
  esac
  output="$root/$config"
  log="$root/logs/${config}_seed${seed}.log"
  mkdir -p "$output"
  if find "$output/abcd" -maxdepth 1 -type f \
      -name "our_moe_seed${seed}*.json" ! -name '*.progress.json' \
      -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] config=$config seed=$seed reason=complete"
    return 0
  fi
  echo "[start] config=$config seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$python_bin" -u "$runner" \
    --model our_moe --data abcd --dataset_manifest "$manifest" --modality IGCB \
    --device 0 --seed "$seed" --train_epochs 100 --warm_up_epochs 5 \
    --early_stopping_patience 0 --early_stopping_min_epochs 0 \
    --batch_size 64 --num_workers 1 --pin_memory true --preprocessed true \
    --initial_filling mean --use_common_ids false \
    --hidden_dim 128 --num_patches 8 --num_heads 4 \
    --num_layers_fus 1 --num_layers_pred 1 --dropout 0.35 \
    --lr 0.0001 --weight_decay 0.01 --optimizer adamw --lr_scheduler constant --grad_clip 5 \
    --num_experts "$experts" --num_routers 1 --top_k "$topk" --patch_adapter_rank 4 \
    --class_weight_power 0.75 --sampler_power 0 \
    --use_generators true --generator_task_grad false --generator_only_task_grad false \
    --recon_loss_weight 0.5 --branch_aux_loss_weight 0.1 --gate_loss_weight 0.01 \
    --modality_dropout_prob 0.25 --drop_ce_loss_weight 0.1 \
    --distill_loss_weight 0.15 --distill_temperature 2 \
    --more_fewer_rank_loss_weight 0.1 --branch_confidence_mode entropy_detached \
    --vectorized_generation true --recon_targets_per_sample 4 \
    --recompute_dropped_combination false --clear_eval_gate_cache true \
    --checkpoint_soup_top_k 1 --validation_only true --allow_overwrite true \
    --output_dir "$output" >"$log" 2>&1
  echo "[done] config=$config seed=$seed gpu=$gpu utc=$(date -u +%FT%TZ)"
}

pids=()
status=0
for index in "${!seeds[@]}"; do
  for config in "${configs[@]}"; do
    run_one "$index" "${seeds[$index]}" "$config" &
    pids+=("$!")
  done
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
[[ "$status" -eq 0 ]] || { echo "Validation screen failed" >&2; exit 1; }

"$python_bin" - "$root" "$manifest" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
configs = ("e4k2", "e8k2")
seeds = (31, 32, 33)
out = {
    "schema": "abcd-severity3-moe-validation-screen-v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "selection_scope": "validation only; formal test was not evaluated",
    "aggregation": "arithmetic mean and sample standard deviation across seeds 31/32/33",
    "configs": {},
}
for config in configs:
    rows = []
    for seed in seeds:
        path = next(
            item for item in (root / config / "abcd").glob(f"our_moe_seed{seed}*.json")
            if not item.name.endswith(".progress.json")
        )
        payload = json.loads(path.read_text())
        rows.append({
            "seed": seed,
            "epochs_completed": int(payload["training"]["epochs_completed"]),
            "best_epoch": int(payload["training"]["best_epoch"]),
            "validation_accuracy": 100.0 * float(payload["validation"]["accuracy"]),
            "validation_macro_f1": 100.0 * float(payload["validation"]["macro_f1"]),
            "validation_macro_auroc": 100.0 * float(payload["validation"]["macro_auroc"]),
        })
    out["configs"][config] = {
        "seed_metrics": rows,
        **{
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "sd": float(np.std([row[metric] for row in rows], ddof=1)),
            }
            for metric in ("validation_accuracy", "validation_macro_f1", "validation_macro_auroc")
        },
    }
(root / "results.json").write_text(json.dumps(out, indent=2) + "\n")
for config, block in out["configs"].items():
    print(config, *(f'{block[m]["mean"]:.2f}±{block[m]["sd"]:.2f}' for m in ("validation_accuracy", "validation_macro_f1", "validation_macro_auroc")))
PY
