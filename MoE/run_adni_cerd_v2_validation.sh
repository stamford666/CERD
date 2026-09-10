#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 SEED PHYSICAL_GPU" >&2
    exit 2
fi
seed=$1
physical_gpu=$2
[[ "$seed" =~ ^(0|1|2)$ ]] || { echo "invalid seed" >&2; exit 2; }
[[ "$physical_gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU" >&2; exit 2; }

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python}
output_dir=${OUTPUT_ROOT:-"$here/adni_cerd_v2_validation"}
log_dir="$output_dir/logs"
mkdir -p "$output_dir" "$log_dir"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}

# The ADNI loader resolves its participant-free schema under MoE/data/adni.
cd "$here"
exec "$python_bin" baseline_runner.py \
    --model our_moe --data adni --modality IGCB \
    --adni_image_imputation legacy_mode \
    --device 0 --seed "$seed" --train_epochs 50 --warm_up_epochs 5 \
    --early_stopping_patience 0 --early_stopping_min_epochs 15 \
    --num_workers 2 --pin_memory True --preprocessed True --initial_filling mean \
    --use_common_ids False --validation_only True --allow_overwrite False \
    --batch_size 32 --hidden_dim 128 --num_patches 16 \
    --num_heads 4 --num_layers_fus 1 --num_layers_pred 1 \
    --num_experts 16 --num_routers 1 --top_k 4 --dropout 0.30 \
    --lr 0.000125 --weight_decay 0 --optimizer adamw --lr_scheduler constant \
    --grad_clip 5 --use_ema False --checkpoint_soup_top_k 1 \
    --gate_loss_weight 0.01 --recon_loss_weight 1 \
    --branch_aux_loss_weight 0.1 --modality_dropout_prob 0.25 \
    --distill_loss_weight 0.15 --distill_temperature 2 --drop_ce_loss_weight 0.1 \
    --more_fewer_rank_loss_weight 0.1 --branch_confidence_mode evidence \
    --centered_evidence_confidence False --token_attention_init -4 \
    --learn_observed_reliability False --sampler_power 0 --sampler_ramp_epochs 0 \
    --class_weight_power 0 --class_weighted_quality_branch_aux False \
    --class_conditional_fusion False --dynamic_branch_fusion False \
    --dynamic_branch_use_observed_mask False --dynamic_branch_joint_prior_boost True \
    --supervised_router_loss_weight 0 \
    --supervised_router_observed_specialists_only False \
    --prediction_observed_specialists_only False \
    --uniform_branch_weights False --joint_branch_only False \
    --clear_eval_gate_cache True --vectorized_generation True --use_generators True \
    --generator_task_grad False --generator_only_task_grad False \
    --generator_output_gate True --recon_targets_per_sample 4 \
    --recompute_dropped_combination False \
    --pattern_aware_reconstruction False \
    --recon_normalized_token_loss_weight 0 \
    --patch_adapter_rank 4 --masked_branch_tcl_loss_weight 0 \
    --output_dir "$output_dir" \
    > >(tee -a "$log_dir/seed${seed}.log") 2>&1
