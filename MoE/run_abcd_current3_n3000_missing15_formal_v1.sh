#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$here/.." && pwd)
python_bin=${PYTHON_BIN:-/home/shaowen/.conda/envs/flex-moe/bin/python}
runner="$here/baseline_runner.py"
evaluator="$here/evaluate_validation_checkpoint.py"
manifest=${DATASET_MANIFEST:-"$workspace/data/abcd_adhd_current3_snp_v2_n3000_random_missing15/manifest.json"}
root=${CAMPAIGN_ROOT:-"$here/abcd_current3_n3000_missing15_full_rerun_v1"}
i2moe_root=${I2MOE_OFFICIAL_ROOT:-"$workspace/I2MoE"}
class_weight_power=${CLASS_WEIGHT_POWER:-0.75}
models=(our_moe flex_moe i2moe moepp_corrected anymod agdic acadiff)
seeds=(31 32 33)

[[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 2; }
[[ -f "$i2moe_root/src/imoe/InteractionMoE.py" ]] || {
  echo "Missing official I2MoE checkout: $i2moe_root" >&2
  exit 2
}
[[ ! -e "$root/run_complete.json" ]] || {
  echo "Refusing to reuse completed campaign: $root" >&2
  exit 2
}
mkdir -p "$root/logs/validation" "$root/logs/formal"

"$python_bin" - "$root" "$manifest" "$runner" "$evaluator" "$class_weight_power" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, manifest, runner, evaluator = map(Path, sys.argv[1:5])
class_weight_power = float(sys.argv[5])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "dataset_manifest": {"path": str(manifest), "sha256": sha(manifest)},
    "implementation": {
        "runner": {"path": str(runner), "sha256": sha(runner)},
        "evaluator": {"path": str(evaluator), "sha256": sha(evaluator)},
    },
    "models": ["our_moe", "flex_moe", "i2moe", "moepp_corrected", "anymod", "agdic", "acadiff"],
    "seeds": [31, 32, 33],
    "aggregation": "arithmetic mean and sample standard deviation across three independent seeds; no ensemble",
    "training_protocol": {
        "epochs": 100,
        "early_stopping": "disabled; every model/seed completes all 100 epochs",
        "checkpoint_selection": "best validation Macro-F1; test evaluated once from the selected checkpoint",
        "parallelism": "one training process per physical GPU; one seed assigned to each GPU",
    },
    "training_only_augmentation": {
        "our_moe_modality_dropout_probability": 0.25,
        "baseline_modality_dropout_probability": 0.25,
        "class_weight_power": class_weight_power,
        "note": "fixed method configurations inherited from the validation-selected protocol; the same frozen dataset manifest and class-weight rule are used by every method",
    },
}
(root / "protocol.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

monitor_gpu() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits || true
    sleep 2
  done
}
monitor_gpu >"$root/gpu_usage.csv" 2>"$root/gpu_monitor.err" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

run_validation() {
  local gpu=$1 seed=$2 model=$3
  local batch dropout lr wd patches experts topk modality_dropout output result log
  case "$model" in
    our_moe)
      batch=64; dropout=0.35; lr=0.0001; wd=0.01
      patches=8; experts=16; topk=4; modality_dropout=0.25
      ;;
    flex_moe)
      batch=32; dropout=0.50; lr=0.0001; wd=0
      patches=16; experts=16; topk=4; modality_dropout=0.25
      ;;
    i2moe)
      batch=64; dropout=0.50; lr=0.0001; wd=0
      patches=16; experts=16; topk=4; modality_dropout=0.25
      ;;
    moepp_corrected)
      batch=128; dropout=0.50; lr=0.0001; wd=0
      patches=16; experts=16; topk=4; modality_dropout=0.25
      ;;
    anymod|agdic|acadiff)
      batch=128; dropout=0.30; lr=0.0003; wd=0.0001
      patches=16; experts=16; topk=4; modality_dropout=0.25
      ;;
    *) echo "Unknown model: $model" >&2; return 2 ;;
  esac

  output="$root/validation/$model"
  result="$output/abcd/${model}_seed${seed}.json"
  log="$root/logs/validation/${model}_seed${seed}.log"
  if [[ -e "$result" ]]; then
    echo "Refusing to reuse result: $result" >&2
    return 2
  fi
  mkdir -p "$output"
  echo "[start] phase=validation model=$model seed=$seed physical_gpu=$gpu utc=$(date -u +%FT%TZ)"

  common=(
    --model "$model" --data abcd --dataset_manifest "$manifest" --modality IGCB
    --device 0 --seed "$seed" --train_epochs 100 --warm_up_epochs 5
    --early_stopping_patience 0 --early_stopping_min_epochs 0
    --batch_size "$batch" --num_workers 2 --pin_memory true --preprocessed true
    --initial_filling mean --use_common_ids false
    --hidden_dim 128 --num_patches "$patches" --num_heads 4
    --num_layers_fus 1 --num_layers_pred 1 --dropout "$dropout"
    --lr "$lr" --weight_decay "$wd" --optimizer adamw --lr_scheduler constant --grad_clip 5
    --num_experts "$experts" --num_routers 1 --top_k "$topk"
    --class_weight_power "$class_weight_power" --sampler_power 0 --modality_dropout_prob "$modality_dropout"
    --checkpoint_soup_top_k 1 --validation_only true --allow_overwrite false
    --output_dir "$output"
  )
  if [[ "$model" == our_moe ]]; then
    common+=(
      --patch_adapter_rank 4
      --use_generators true --generator_task_grad false --generator_only_task_grad false
      --recon_loss_weight 0.5 --branch_aux_loss_weight 0.1 --gate_loss_weight 0.01
      --drop_ce_loss_weight 0.1 --distill_loss_weight 0.15 --distill_temperature 2
      --more_fewer_rank_loss_weight 0.1 --branch_confidence_mode entropy_detached
      --vectorized_generation true --recon_targets_per_sample 4
      --recompute_dropped_combination false --clear_eval_gate_cache true
    )
  fi

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  I2MOE_OFFICIAL_ROOT="$i2moe_root" \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$python_bin" -u "$runner" "${common[@]}" >"$log" 2>&1
  echo "[done] phase=validation model=$model seed=$seed physical_gpu=$gpu utc=$(date -u +%FT%TZ)"
}

run_formal() {
  local gpu=$1 seed=$2 model=$3 source output result log
  source="$root/validation/$model/abcd/${model}_seed${seed}.json"
  output="$root/formal/$model"
  result="$output/abcd/${model}_seed${seed}.json"
  log="$root/logs/formal/${model}_seed${seed}.log"
  [[ -f "$source" ]] || { echo "Missing validation result: $source" >&2; return 2; }
  [[ ! -e "$result" ]] || { echo "Refusing to reuse formal result: $result" >&2; return 2; }
  mkdir -p "$output"
  echo "[start] phase=formal model=$model seed=$seed physical_gpu=$gpu utc=$(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
  I2MOE_OFFICIAL_ROOT="$i2moe_root" \
  LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$python_bin" -u "$evaluator" --source-result "$source" \
    --output-root "$output" --device 0 >"$log" 2>&1
  echo "[done] phase=formal model=$model seed=$seed physical_gpu=$gpu utc=$(date -u +%FT%TZ)"
}

seed_worker() {
  local gpu=$1 seed=$2 model
  for model in "${models[@]}"; do
    run_validation "$gpu" "$seed" "$model"
  done
  for model in "${models[@]}"; do
    run_formal "$gpu" "$seed" "$model"
  done
}

seed_worker 0 31 & pids=("$!")
seed_worker 1 32 & pids+=("$!")
seed_worker 2 33 & pids+=("$!")
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
[[ "$status" -eq 0 ]] || { echo "Full rerun failed; inspect $root/logs." >&2; exit 1; }

"$python_bin" - "$root" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
models = ("our_moe", "flex_moe", "i2moe", "moepp_corrected", "anymod", "agdic", "acadiff")
seeds = (31, 32, 33)
metrics = ("accuracy", "macro_f1", "macro_auroc")
summary = {
    "schema": "abcd-current3-n3000-missing15-full-rerun-v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "aggregation": "arithmetic mean and sample standard deviation across seeds 31/32/33; no ensemble",
    "methods": {},
}
for model in models:
    payloads = [
        json.loads((root / "formal" / model / "abcd" / f"{model}_seed{seed}.json").read_text())
        for seed in seeds
    ]
    seed_metrics = []
    for seed, payload in zip(seeds, payloads):
        row = {"seed": seed}
        row.update({metric: 100.0 * float(payload["test"][metric]) for metric in metrics})
        seed_metrics.append(row)
    summary["methods"][model] = {
        "seed_metrics": seed_metrics,
        "aggregate": {
            metric: {
                "mean": float(np.mean([row[metric] for row in seed_metrics])),
                "sd": float(np.std([row[metric] for row in seed_metrics], ddof=1)),
            }
            for metric in metrics
        },
        "training": [
            {
                "seed": seed,
                "epochs_completed": payload.get("epochs_completed", payload.get("training", {}).get("epochs_completed")),
                "best_epoch": payload.get("best_epoch", payload.get("training", {}).get("best_epoch")),
            }
            for seed, payload in zip(seeds, payloads)
        ],
        "test_subsets": {
            subset: {
                "n": int(payloads[0]["test_subsets"][subset]["n"]),
                **{
                    metric: {
                        "mean": float(np.mean([100.0 * float(p["test_subsets"][subset][metric]) for p in payloads])),
                        "sd": float(np.std([100.0 * float(p["test_subsets"][subset][metric]) for p in payloads], ddof=1)),
                    }
                    for metric in metrics
                },
            }
            for subset in ("complete", "missing")
        },
    }
(root / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
protocol = json.loads((root / "protocol.json").read_text())
protocol["status"] = "complete"
protocol["completed_at"] = summary["completed_at"]
(root / "run_complete.json").write_text(json.dumps(protocol, indent=2) + "\n")
for model, block in summary["methods"].items():
    print(model, *(f'{block["aggregate"][metric]["mean"]:.2f}±{block["aggregate"][metric]["sd"]:.2f}' for metric in metrics))
PY
