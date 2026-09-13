#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=${CAMPAIGN_ROOT:-/data/mi2/shaowen/miccai2026/MoE/abcd_severity3_n3000_missing15_full_rerun_v1}
output=${AUDIT_OUTPUT_DIR:-/data/mi2/shaowen/miccai2026/MoE/severity3_e4k2_three_seed_modality_audit_v2}
python_bin=${PYTHON_BIN:-/home/shaowen/.conda/envs/flex-moe/bin/python}
schema=${AUDIT_SCHEMA:-cerd-three-seed-modality-audit-severity3-e4k2-v2}
output_stem=${AUDIT_OUTPUT_STEM:-cerd_three_seed_modality_audit_severity3_e4k2_v2}
figure_stem=${AUDIT_FIGURE_STEM:-cerd_two_dataset_modality_drop_severity3_e4k2_v2}

export ADNI_MODALITY_AUDIT_CHECKPOINT_ROOT="/data/mi2/shaowen/miccai2026/MoE/fair_two_dataset_tuning_20260910/runs/adni/p16_d030_lr125/checkpoints/adni"
export ADNI_MODALITY_AUDIT_REFERENCE_ROOT="/data/mi2/shaowen/miccai2026/MoE/fair_two_dataset_tuning_20260910/formal/predictions/adni"
export ABCD_MODALITY_AUDIT_CHECKPOINT_ROOT="${ABCD_MODALITY_AUDIT_CHECKPOINT_ROOT:-$root/validation/our_moe/checkpoints/abcd}"
export ABCD_MODALITY_AUDIT_REFERENCE_ROOT="${ABCD_MODALITY_AUDIT_REFERENCE_ROOT:-$root/formal/our_moe/predictions/abcd}"

LD_LIBRARY_PATH="/home/shaowen/.conda/envs/flex-moe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$python_bin" "$here/run_current_three_seed_modality_audit_v1.py" \
  --device 0 \
  --data-workdir "${ADNI_DATA_WORKDIR:-/data/mi2/shaowen/miccai2026/MoE}" \
  --output-dir "$output" \
  --schema "$schema" \
  --output-stem "$output_stem" \
  --figure-stem "$figure_stem"
