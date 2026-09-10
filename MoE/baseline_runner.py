"""Unified ABCD/ADNI runner for modern multimodal baselines.

This driver deliberately owns the training, checkpoint-selection, and metric
protocol.  The architecture modules remain in their existing sibling folders,
but every model is fed by ``MoE/data.py`` and evaluated by the same code path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn

# FastMoE's dm-tree extension requires the conda C++ runtime.  Import the
# local MoE module before pandas/pyarrow can load the older system libstdc++.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models import AGMGFlexMoE, ConcatTransformerBaseline, FlexMoE, MLP

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from data import (
    ADNI_IMAGE_IMPUTATION_CHOICES,
    SampleOrderSHARecorder,
    attach_sample_order_recorder,
    create_loaders,
    load_and_preprocess_data,
    resolve_modality_dict,
)
from multimodal_data.metrics import (
    classification_auc,
    classification_average_precision,
    tune_binary_threshold,
)
from multimodal_data.abcd import (
    canonical_abcd_subject_ids,
    resolved_abcd_manifest_path,
)
from multimodal_data.training import balanced_train_loader, class_weights


MODEL_CHOICES = (
    "flex_moe",
    "i2moe",
    "moepp",
    "moepp_corrected",
    "anymod",
    "mora",
    "agdic",
    "agmd",
    "acadiff",
    "transformer_concat",
    "our_moe",
)
MODEL_DISPLAY = {
    "flex_moe": "Flex-MoE",
    "i2moe": "I2MoE (official-code adapter)",
    "moepp": "MoE++ (official-code adapter)",
    "moepp_corrected": "MoE++ (corrected official adapter)",
    "anymod": "AnyMod (reimplemented)",
    "mora": "MoRA-inspired",
    "agdic": "AGDiC-inspired",
    "agmd": "AGMD-inspired",
    "acadiff": "ACADiff-inspired",
    "transformer_concat": "Transformer (token concatenation)",
    "our_moe": "Ours (MoE)",
}

# The paper-facing comparison cohort.  Legacy adapters remain accepted by the
# runner so old artifacts stay replayable, but formal BED tables must use this
# exact shared-protocol cohort.
FORMAL_COMPARISON_MODELS = (
    "flex_moe",
    "i2moe",
    "moepp_corrected",
    "anymod",
    "agdic",
    "acadiff",
    "our_moe",
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected true or false")
    return normalized == "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--data", required=True, choices=("abcd", "adni"))
    parser.add_argument("--dataset_manifest", default=None)
    parser.add_argument("--modality", default="IGCB")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--warm_up_epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--preprocessed", type=str2bool, default=True)
    parser.add_argument("--initial_filling", default="mean")
    parser.add_argument(
        "--adni_image_imputation",
        choices=ADNI_IMAGE_IMPUTATION_CHOICES,
        default="legacy_mode",
        help=(
            "Training-split-only imputation for continuous preprocessed ADNI "
            "FreeSurfer features. legacy_mode preserves the historical "
            "initial_filling behavior exactly."
        ),
    )
    parser.add_argument("--use_common_ids", type=str2bool, default=False)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers_fus", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument(
        "--lr_scheduler", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--lr_warmup_epochs", type=int, default=0)
    parser.add_argument(
        "--exclude_bias_norm_from_weight_decay", type=str2bool, default=False
    )
    parser.add_argument("--use_ema", type=str2bool, default=False)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_epoch", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument(
        "--sam_rho",
        type=float,
        default=0.0,
        help=(
            "Global-L2 SAM radius for Ours. Zero preserves the historical "
            "single-forward optimizer path exactly."
        ),
    )
    parser.add_argument(
        "--rdrop_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Symmetric-KL weight for an independent end-to-end clean-view "
            "R-Drop pass in Ours. Zero preserves the historical forward and "
            "RNG path exactly."
        ),
    )
    parser.add_argument(
        "--masked_branch_tcl_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only correct-branch collaborative KL weight for Ours. "
            "Zero preserves the historical objective and state path exactly."
        ),
    )
    parser.add_argument(
        "--masked_branch_tcl_temperature",
        type=float,
        default=2.0,
        help="Temperature for masked correct-teacher branch KL.",
    )
    parser.add_argument(
        "--masked_branch_tcl_start_epoch",
        type=int,
        default=6,
        help=(
            "First epoch that applies masked branch TCL. Earlier epochs only "
            "collect clean-primary branch accuracy statistics."
        ),
    )
    parser.add_argument(
        "--masked_branch_tcl_class_acc_ema",
        type=float,
        default=0.9,
        help="Across-epoch EMA decay for per-class available-branch accuracy.",
    )
    parser.add_argument(
        "--trusted_branch_fusion_distill_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only correct-branch-to-final-fusion KL weight for Ours. "
            "Zero preserves the historical objective and state path exactly."
        ),
    )
    parser.add_argument(
        "--trusted_branch_fusion_temperature",
        type=float,
        default=2.0,
        help="Temperature for trusted branch-to-fusion distillation.",
    )
    parser.add_argument(
        "--trusted_branch_fusion_start_epoch",
        type=int,
        default=6,
        help=(
            "First epoch that distills trusted branches into the final fused "
            "logits; earlier epochs only collect branch accuracy statistics."
        ),
    )
    parser.add_argument(
        "--trusted_branch_fusion_class_acc_ema",
        type=float,
        default=0.9,
        help="Across-epoch EMA decay for TBFD class compensation.",
    )
    parser.add_argument(
        "--clear_eval_gate_cache",
        type=str2bool,
        default=False,
        help=(
            "Consume and discard Ours router losses after non-training forwards "
            "so validation calls cannot enter the next epoch's first objective. "
            "False preserves the historical training trajectory exactly."
        ),
    )
    parser.add_argument("--aux_loss_weight", type=float, default=0.05)
    parser.add_argument("--align_loss_weight", type=float, default=0.05)
    parser.add_argument("--anchor_loss_weight", type=float, default=0.05)
    parser.add_argument("--gate_loss_weight", type=float, default=1e-2)
    parser.add_argument(
        "--normalized_gate_loss",
        type=str2bool,
        default=False,
        help=(
            "Prevalence-weight full/missing expert-router balance terms, divide "
            "missing-pattern CE by all tokens, and average repeated modality calls."
        ),
    )
    parser.add_argument("--recon_loss_weight", type=float, default=1.0)
    parser.add_argument("--branch_aux_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--joint_head_ensemble_size",
        type=int,
        default=1,
        help=(
            "Number of parameter-efficient TabM-style members in the Ours "
            "joint diagnostic head. One constructs the historical MLP exactly."
        ),
    )
    parser.add_argument(
        "--more_tail_rank",
        type=int,
        default=0,
        help=(
            "Rank of the optional MORE-inspired low-rank residual on Ours' "
            "final fused logits. Zero constructs no adapter and preserves the "
            "historical model/RNG path exactly."
        ),
    )
    parser.add_argument(
        "--more_tail_peak_amplitude",
        type=float,
        default=0.0,
        help=(
            "Peak coefficient A in alpha=A*sin(pi*global_training_progress) "
            "for the fused-logit tail regularizer."
        ),
    )
    parser.add_argument("--modality_dropout_prob", type=float, default=0.25)
    parser.add_argument("--distill_loss_weight", type=float, default=0.15)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument(
        "--tree_teacher_distill_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only KL weight for an explicitly hashed cross-fitted "
            "tree-teacher NPZ. Zero preserves the historical objective and "
            "data-loader path exactly."
        ),
    )
    parser.add_argument(
        "--tree_teacher_temperature",
        type=float,
        default=2.0,
        help="Temperature used for cross-fitted tree-teacher KL distillation.",
    )
    parser.add_argument(
        "--tree_teacher_npz",
        default=None,
        help=(
            "Explicit cross-fitted teacher artifact. It is opened only when "
            "tree_teacher_distill_weight is positive."
        ),
    )
    parser.add_argument(
        "--tree_teacher_npz_sha256",
        default=None,
        help=(
            "Required lowercase SHA-256 digest of the enabled teacher NPZ."
        ),
    )
    parser.add_argument(
        "--external_teacher_distill_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only KL weight for a schema-v2, explicitly hashed "
            "cross-fitted external-teacher NPZ. Zero preserves the historical "
            "objective and data-loader path exactly."
        ),
    )
    parser.add_argument(
        "--external_teacher_temperature",
        type=float,
        default=2.0,
        help="Temperature used for schema-v2 external-teacher KL distillation.",
    )
    parser.add_argument(
        "--external_teacher_npz",
        default=None,
        help=(
            "Explicit schema-v2 cross-fitted teacher artifact. It is opened "
            "only when external_teacher_distill_weight is positive."
        ),
    )
    parser.add_argument(
        "--external_teacher_npz_sha256",
        default=None,
        help="Required lowercase SHA-256 digest of the enabled v2 teacher NPZ.",
    )
    parser.add_argument(
        "--external_teacher_trust_mode",
        choices=("correct_only", "all"),
        default="correct_only",
        help=(
            "Trust gate for v2 targets: correct_only retains OOF rows whose "
            "teacher argmax equals the training label; all retains every row."
        ),
    )
    parser.add_argument(
        "--external_teacher_class_compensation",
        choices=("none", "tcl_sqrt_error"),
        default="none",
        help=(
            "Optional full-OOF class compensation. tcl_sqrt_error uses "
            "normalized square-root class error weights."
        ),
    )
    parser.add_argument("--drop_ce_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--more_fewer_rank_loss_weight",
        type=float,
        default=0.0,
        help=(
            "SimMLM more-versus-fewer modality ranking weight on the existing "
            "artificial-drop view: mean relu(CE_more-CE_fewer) over rows whose "
            "observed-modality set was strictly reduced."
        ),
    )
    parser.add_argument(
        "--more_fewer_rank_loss_effective_weight", type=float, default=None,
        help=("Optional contribution weight. The control weight still fixes the "
              "reduced-view forward, mask and RNG plan; zero ablates only the loss contribution."),
    )
    parser.add_argument("--w_entropy_weight", type=float, default=0.0)
    parser.add_argument("--gen_num_layers", type=int, default=2)
    parser.add_argument("--gen_num_heads", type=int, default=4)
    parser.add_argument("--vectorized_generation", type=str2bool, default=True)
    parser.add_argument("--recon_targets_per_sample", type=int, default=4)
    parser.add_argument(
        "--pattern_aware_reconstruction",
        type=str2bool,
        default=False,
        help=(
            "Reconstruct naturally observed per-sample targets from the other "
            "observed modalities, grouped by target/context pattern."
        ),
    )
    parser.add_argument(
        "--recon_normalized_token_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of normalized token-level Smooth-L1 added to the generator "
            "reconstruction objective."
        ),
    )
    parser.add_argument(
        "--recon_context_dropout_probability",
        type=float,
        default=0.0,
        help=(
            "During pattern-aware reconstruction, independently remove "
            "observed context modalities while always retaining at least one."
        ),
    )
    parser.add_argument(
        "--recon_encoder_gradient_scale",
        type=float,
        default=1.0,
        help=(
            "Scale reconstruction-context gradients into modality encoders. "
            "One preserves the historical objective; zero trains only the "
            "generator/projector side of reconstruction. Forward values are "
            "unchanged."
        ),
    )
    parser.add_argument("--use_generators", type=str2bool, default=True)
    parser.add_argument("--generator_task_grad", type=str2bool, default=False)
    parser.add_argument(
        "--generator_only_task_grad",
        type=str2bool,
        default=False,
        help=(
            "Allow classification losses to train missing-modality generators "
            "while detaching their observed-modality context. False preserves "
            "the historical classification path exactly."
        ),
    )
    parser.add_argument(
        "--generator_output_gate",
        type=str2bool,
        default=True,
        help=(
            "Apply the post-cross-attention sigmoid output gate. False keeps "
            "the same parameters and initialization but bypasses the gate."
        ),
    )
    for flag, help_text in (
        ("dense_backbone", "Replace sparse MoE MLP calls with aligned dense MLPs."),
        ("disable_provenance", "Bypass provenance flag embeddings."),
        ("uniform_branch_weights", "Use a uniform simplex over active branches."),
        ("joint_branch_only", "Use only the joint prediction branch."),
        ("mean_pooling_only", "Bypass learned attention mixing after computing it."),
        ("disable_stochastic_context_masking", "Use leave-one-out contexts while preserving configured RNG draws."),
        ("disable_completion", "Bypass completion forwards while retaining modules."),
        ("no_output_gate", "Bypass the existing generator output gate."),
    ):
        parser.add_argument(f"--{flag}", type=str2bool, default=False, help=help_text)
    parser.add_argument(
        "--data_order_seed", type=int, default=None,
        help="Opt-in independent shuffled-DataLoader RNG seed; None preserves historical global-RNG behavior.",
    )
    parser.add_argument("--dynamic_branch_fusion", type=str2bool, default=False)
    parser.add_argument(
        "--dynamic_branch_joint_prior_boost",
        type=str2bool,
        default=True,
        help=(
            "Retain the historical log(2) joint-branch prior when dynamic "
            "routing is enabled. False is a clean-router opt-in."
        ),
    )
    parser.add_argument(
        "--dynamic_branch_use_observed_mask",
        type=str2bool,
        default=False,
        help=(
            "Feed the raw observed-modality mask, rather than the generator-"
            "expanded usable mask, to the optional dynamic gate."
        ),
    )
    parser.add_argument(
        "--prediction_observed_specialists_only",
        type=str2bool,
        default=False,
        help=(
            "Keep completion features in the joint prediction branch while "
            "restricting unimodal and pairwise prediction/reliability "
            "branches to modalities present in the original observed mask."
        ),
    )
    parser.add_argument(
        "--dynamic_branch_quality_weighted_aux",
        type=str2bool,
        default=False,
        help=(
            "Keep reliability-quality weighting for the specialist branch "
            "auxiliary while dynamic routing is enabled. False preserves the "
            "historical dynamic-router ordinary-CE branch auxiliary."
        ),
    )
    parser.add_argument(
        "--supervised_router_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only soft branch-oracle routing loss. Zero preserves the "
            "historical objective exactly."
        ),
    )
    parser.add_argument(
        "--supervised_router_temperature",
        type=float,
        default=0.25,
        help=(
            "Temperature used to turn detached per-branch true-class NLLs into "
            "the supervised router target distribution."
        ),
    )
    parser.add_argument(
        "--supervised_router_observed_specialists_only",
        type=str2bool,
        default=False,
        help=(
            "Restrict the label-supervised router teacher to the joint branch "
            "and specialist branches whose modalities are all truly observed."
        ),
    )
    parser.add_argument(
        "--supervised_contrastive_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only class-balanced supervised contrastive weight over "
            "full and modality-dropped pooled-feature views."
        ),
    )
    parser.add_argument(
        "--supervised_contrastive_temperature",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--supervised_contrastive_projection_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dual_boundary_rank_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only pairwise ranking weight on the final fused class-1 "
            "versus class-0 and class-2 logit boundaries."
        ),
    )
    parser.add_argument(
        "--dual_local_boundary_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the dual local-boundary residual (DLBR) auxiliary. "
            "A positive value also enables its bounded, zero-initialized "
            "class-1-vs-0 and class-1-vs-2 inference adapter."
        ),
    )
    parser.add_argument(
        "--presentation_axis_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the balanced IA/HI presentation-axis logistic "
            "auxiliary. A positive value also enables its bounded, "
            "zero-initialized three-class inference residual."
        ),
    )
    parser.add_argument(
        "--presentation_axis_residual_cap",
        type=float,
        default=0.5,
        help=(
            "Absolute tanh cap for each IA/HI presentation-axis residual."
        ),
    )
    parser.add_argument(
        "--missing_capacity_residual_width",
        type=int,
        default=0,
        help=(
            "Width of the optional intercept-free pooled residual MLP applied "
            "only to current-view missing rows in Ours. Zero constructs no "
            "module and preserves the historical model/RNG path exactly."
        ),
    )
    parser.add_argument(
        "--dual_boundary_rank_margin",
        type=float,
        default=0.20,
        help="Pairwise soft-margin for the class-1 dual-boundary ranking loss.",
    )
    parser.add_argument(
        "--dual_boundary_rank_10_weight",
        type=float,
        default=2.0 / 3.0,
        help=(
            "Relative weight of the class-1-versus-class-0 boundary; the "
            "class-1-versus-class-2 boundary receives one minus this value."
        ),
    )
    parser.add_argument(
        "--hard_cvar_dual_boundary_max_weight",
        type=float,
        default=0.0,
        help=(
            "Maximum training-only weight for hard-CVaR dual-boundary ranking. "
            "Zero disables the candidate exactly."
        ),
    )
    parser.add_argument(
        "--hard_cvar_dual_boundary_tail_fraction",
        type=float,
        default=0.25,
        help="Fraction of highest-loss boundary pairs retained in each CVaR tail.",
    )
    parser.add_argument(
        "--hard_cvar_dual_boundary_margin",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--hard_cvar_dual_boundary_10_weight",
        type=float,
        default=0.65,
        help="Relative class-1-vs-0 boundary weight; 1-vs-2 receives the remainder.",
    )
    parser.add_argument(
        "--hard_cvar_dual_boundary_start_epoch",
        type=int,
        default=5,
        help="One-indexed first epoch with a non-zero hard-CVaR weight.",
    )
    parser.add_argument(
        "--hard_cvar_dual_boundary_ramp_epochs",
        type=int,
        default=10,
        help="Number of active epochs used to linearly reach the maximum weight.",
    )
    parser.add_argument("--complete_joint_only", type=str2bool, default=False)
    parser.add_argument("--complete_specialist_weight", type=float, default=1.0)
    parser.add_argument(
        "--missing_family_normalized_router",
        type=str2bool,
        default=False,
        help=(
            "On Ours rows with at least one missing modality, normalize routing "
            "first within joint/unimodal/pairwise families and then across "
            "family log-mean-exp scores. False preserves the historical path."
        ),
    )
    parser.add_argument(
        "--missing_family_residual_gate",
        type=str2bool,
        default=False,
        help=(
            "Add a zero-initialized learned residual to the three missing-row "
            "family scores. Requires --missing_family_normalized_router true."
        ),
    )
    parser.add_argument(
        "--branch_confidence_mode",
        choices=("evidence", "entropy_detached", "entropy_exp_detached"),
        default="evidence",
    )
    parser.add_argument("--recompute_dropped_combination", type=str2bool, default=False)
    parser.add_argument(
        "--empirical_missing_pattern_replay",
        type=str2bool,
        default=False,
        help=(
            "ABCD/Ours training-only cardinality-matched replay of positive-"
            "support missingness patterns estimated from the current training "
            "split. False preserves the legacy dropout and RNG path exactly."
        ),
    )
    parser.add_argument("--token_attention_init", type=float, default=-4.0)
    parser.add_argument("--standard_transformer_residual", type=str2bool, default=False)
    parser.add_argument("--gated_transformer_residual", type=str2bool, default=False)
    parser.add_argument("--ordinal_fusion_weight", type=float, default=0.0)
    parser.add_argument("--ordinal_aux_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--class1_aux_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only BCE weight for an independent class-1-vs-rest "
            "auxiliary head. The auxiliary logit is never fused into predictions."
        ),
    )
    parser.add_argument(
        "--ordinal_head_type",
        choices=("proportional", "continuation"),
        default="proportional",
    )
    parser.add_argument(
        "--uncertainty_aware_ordinal_fusion",
        type=str2bool,
        default=False,
        help=(
            "Use ordinal_fusion_weight as the maximum entropy-gated blend "
            "weight for the continuation ordinal head."
        ),
    )
    parser.add_argument("--learn_observed_reliability", type=str2bool, default=False)
    parser.add_argument("--centered_evidence_confidence", type=str2bool, default=False)
    parser.add_argument("--class_conditional_fusion", type=str2bool, default=False)
    parser.add_argument("--balanced_branch_aux", type=str2bool, default=False)
    parser.add_argument(
        "--class_weighted_quality_branch_aux",
        type=str2bool,
        default=False,
        help=(
            "Apply the main CE class weights to quality-weighted specialist-branch "
            "supervision. False preserves checkpoints produced before this fix."
        ),
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--logit_adjust_tau",
        type=float,
        default=0.0,
        help=(
            "Training-only logit adjustment strength for Ours. The empirical "
            "training-split log prior is added inside CE; validation and test "
            "continue to use raw model logits."
        ),
    )
    parser.add_argument("--adapter_rank", type=int, default=16)
    parser.add_argument("--prompt_length", type=int, default=4)
    parser.add_argument("--num_query_tokens", type=int, default=8)
    parser.add_argument("--num_task_tokens", type=int, default=8)
    parser.add_argument("--num_proj_layers", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument(
        "--acadiff_mask_probability",
        type=float,
        default=0.25,
        help=(
            "Probability of synthetically holding out a naturally observed modality "
            "for ACADiff denoising supervision."
        ),
    )
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--num_routers", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--num_layers_pred", type=int, default=1)
    parser.add_argument("--missing_bank_init_std", type=float, default=0.02)
    parser.add_argument("--classifier_dropout", type=float, default=None)
    parser.add_argument("--hidden_dim_rw", type=int, default=128)
    parser.add_argument("--num_layer_rw", type=int, default=3)
    parser.add_argument("--temperature_rw", type=float, default=1.0)
    parser.add_argument("--interaction_loss_weight", type=float, default=0.3)
    parser.add_argument("--sampler_power", type=float, default=None)
    parser.add_argument(
        "--sampler_ramp_epochs",
        type=int,
        default=0,
        help=(
            "Training-only linear ramp from sampler power 0 to --sampler_power "
            "during this many epochs immediately after warm-up. Zero preserves "
            "the historical abrupt switch exactly."
        ),
    )
    parser.add_argument(
        "--checkpoint_soup_top_k",
        type=int,
        default=1,
        help=(
            "Retain the top K validation-score checkpoints and always use "
            "their equal-weight state average as the final checkpoint. One "
            "uses positive-class AUPRC for binary tasks and Macro-F1 otherwise."
        ),
    )
    parser.add_argument(
        "--checkpoint_selection_policy",
        choices=("validation_best", "final_epoch_refit"),
        default="validation_best",
        help=(
            "validation_best preserves the historical per-epoch validation "
            "selection path. final_epoch_refit is validation-only training "
            "that saves the final live state after exactly train_epochs and "
            "runs held-validation inference once, only after training ends."
        ),
    )
    parser.add_argument("--class_weight_power", type=float, default=None)
    parser.add_argument("--independent_patch_embeddings", type=str2bool, default=False)
    parser.add_argument(
        "--patch_adapter_rank",
        type=int,
        default=0,
        help=(
            "Rank of the optional patch-specific residual adapter in each "
            "tabular encoder. Zero preserves the historical shared projection."
        ),
    )
    parser.add_argument("--output_dir", default=str(HERE / "baseline_results"))
    parser.add_argument(
        "--allow_overwrite",
        type=str2bool,
        default=False,
        help="Explicitly allow replacing existing artifacts for this dataset/model/seed key.",
    )
    parser.add_argument(
        "--validation_only",
        type=str2bool,
        default=False,
        help="Train fully and save the validation-selected checkpoint without touching the test split.",
    )
    args = parser.parse_args()
    if args.generator_task_grad and args.generator_only_task_grad:
        parser.error(
            "--generator_task_grad and --generator_only_task_grad are mutually "
            "exclusive"
        )
    if args.generator_only_task_grad:
        if args.model != "our_moe":
            parser.error(
                "--generator_only_task_grad is only supported for --model our_moe"
            )
        if not args.use_generators:
            parser.error(
                "--generator_only_task_grad requires --use_generators true"
            )
    if args.empirical_missing_pattern_replay:
        if args.model != "our_moe" or args.data != "abcd":
            parser.error(
                "--empirical_missing_pattern_replay is only supported for "
                "--model our_moe --data abcd"
            )
        if not args.recompute_dropped_combination:
            parser.error(
                "--empirical_missing_pattern_replay requires "
                "--recompute_dropped_combination true"
            )
        if args.generator_only_task_grad:
            parser.error(
                "--empirical_missing_pattern_replay and "
                "--generator_only_task_grad are mutually exclusive"
            )
        if not math.isfinite(args.modality_dropout_prob) or not (
            0.0 < args.modality_dropout_prob <= 1.0
        ):
            parser.error(
                "--empirical_missing_pattern_replay requires "
                "--modality_dropout_prob in (0, 1]"
            )
        if args.sam_rho != 0.0:
            parser.error(
                "--empirical_missing_pattern_replay v1 requires --sam_rho 0"
            )
        if args.rdrop_loss_weight != 0.0:
            parser.error(
                "--empirical_missing_pattern_replay v1 requires "
                "--rdrop_loss_weight 0"
            )
        try:
            empirical_missing_pattern_replay_active_objectives(args)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    if args.patch_adapter_rank < 0:
        parser.error("--patch_adapter_rank must be non-negative")
    if args.patch_adapter_rank > 0:
        if args.model != "our_moe":
            parser.error(
                "--patch_adapter_rank is only supported for --model our_moe"
            )
        if args.independent_patch_embeddings:
            parser.error(
                "--patch_adapter_rank and --independent_patch_embeddings are "
                "mutually exclusive"
            )
    if args.standard_transformer_residual and args.gated_transformer_residual:
        parser.error(
            "--standard_transformer_residual and --gated_transformer_residual "
            "are mutually exclusive"
        )
    if (
        args.centered_evidence_confidence
        and args.branch_confidence_mode != "evidence"
    ):
        parser.error(
            "--centered_evidence_confidence only applies to evidence confidence"
        )
    if args.class_conditional_fusion and args.complete_joint_only:
        parser.error(
            "--class_conditional_fusion is incompatible with --complete_joint_only"
        )
    if args.missing_family_residual_gate and not args.missing_family_normalized_router:
        parser.error(
            "--missing_family_residual_gate requires "
            "--missing_family_normalized_router true"
        )
    if args.missing_family_normalized_router:
        if args.model != "our_moe":
            parser.error(
                "--missing_family_normalized_router is only supported for "
                "--model our_moe"
            )
        if args.dynamic_branch_fusion:
            parser.error(
                "missing-family routing v1 is incompatible with "
                "--dynamic_branch_fusion true"
            )
        if args.supervised_router_loss_weight > 0:
            parser.error(
                "missing-family routing v1 is incompatible with positive "
                "--supervised_router_loss_weight"
            )
        if args.class_conditional_fusion:
            parser.error(
                "missing-family routing v1 is incompatible with "
                "--class_conditional_fusion true"
            )
        if args.joint_head_ensemble_size != 1:
            parser.error(
                "missing-family routing v1 requires "
                "--joint_head_ensemble_size 1"
            )
    if (
        not math.isfinite(args.supervised_router_loss_weight)
        or args.supervised_router_loss_weight < 0
    ):
        parser.error(
            "--supervised_router_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.supervised_router_temperature)
        or args.supervised_router_temperature <= 0
    ):
        parser.error("--supervised_router_temperature must be finite and positive")
    if args.supervised_router_loss_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--supervised_router_loss_weight is only supported for --model our_moe"
            )
        if not args.dynamic_branch_fusion:
            parser.error(
                "--supervised_router_loss_weight requires --dynamic_branch_fusion true"
            )
    clean_dynamic_router_requested = (
        not args.dynamic_branch_joint_prior_boost
        or args.dynamic_branch_use_observed_mask
        or args.dynamic_branch_quality_weighted_aux
        or args.supervised_router_observed_specialists_only
    )
    if clean_dynamic_router_requested:
        if args.model != "our_moe":
            parser.error(
                "clean dynamic-router options are only supported for --model our_moe"
            )
        if not args.dynamic_branch_fusion:
            parser.error(
                "clean dynamic-router options require --dynamic_branch_fusion true"
            )
    if (
        args.prediction_observed_specialists_only
        and args.model != "our_moe"
    ):
        parser.error(
            "--prediction_observed_specialists_only is only supported for "
            "--model our_moe"
        )
    if args.dynamic_branch_quality_weighted_aux and args.balanced_branch_aux:
        parser.error(
            "--dynamic_branch_quality_weighted_aux and --balanced_branch_aux "
            "are mutually exclusive"
        )
    if (
        args.supervised_router_observed_specialists_only
        and args.supervised_router_loss_weight <= 0
    ):
        parser.error(
            "--supervised_router_observed_specialists_only requires positive "
            "--supervised_router_loss_weight"
        )
    if (
        not math.isfinite(args.supervised_contrastive_loss_weight)
        or args.supervised_contrastive_loss_weight < 0
    ):
        parser.error(
            "--supervised_contrastive_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.supervised_contrastive_temperature)
        or args.supervised_contrastive_temperature <= 0
    ):
        parser.error(
            "--supervised_contrastive_temperature must be finite and positive"
        )
    if args.supervised_contrastive_projection_dim <= 0:
        parser.error("--supervised_contrastive_projection_dim must be positive")
    if args.supervised_contrastive_loss_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--supervised_contrastive_loss_weight is only supported for --model our_moe"
            )
        if args.modality_dropout_prob <= 0:
            parser.error(
                "--supervised_contrastive_loss_weight requires positive modality dropout"
            )
    if (
        not math.isfinite(args.dual_boundary_rank_loss_weight)
        or args.dual_boundary_rank_loss_weight < 0
    ):
        parser.error(
            "--dual_boundary_rank_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.dual_boundary_rank_margin)
        or args.dual_boundary_rank_margin < 0
    ):
        parser.error("--dual_boundary_rank_margin must be finite and non-negative")
    if (
        not math.isfinite(args.dual_boundary_rank_10_weight)
        or not 0.0 <= args.dual_boundary_rank_10_weight <= 1.0
    ):
        parser.error("--dual_boundary_rank_10_weight must be finite and in [0, 1]")
    if args.dual_boundary_rank_loss_weight > 0 and args.model != "our_moe":
        parser.error(
            "--dual_boundary_rank_loss_weight is only supported for --model our_moe"
        )
    if args.joint_head_ensemble_size < 1:
        parser.error("--joint_head_ensemble_size must be positive")
    if args.joint_head_ensemble_size > 1:
        if args.model != "our_moe":
            parser.error(
                "--joint_head_ensemble_size greater than one is only supported "
                "for --model our_moe"
            )
        if args.num_layers_pred < 2:
            parser.error(
                "--joint_head_ensemble_size greater than one requires "
                "--num_layers_pred at least 2"
            )
        if (
            not math.isfinite(args.branch_aux_loss_weight)
            or args.branch_aux_loss_weight <= 0
        ):
            parser.error(
                "--joint_head_ensemble_size greater than one requires positive "
                "--branch_aux_loss_weight for independent member CE"
            )
    if args.more_tail_rank < 0:
        parser.error("--more_tail_rank must be non-negative")
    if (
        not math.isfinite(args.more_tail_peak_amplitude)
        or args.more_tail_peak_amplitude < 0
    ):
        parser.error(
            "--more_tail_peak_amplitude must be finite and non-negative"
        )
    if args.more_tail_rank > 0:
        if args.model != "our_moe":
            parser.error(
                "--more_tail_rank is only supported for --model our_moe"
            )
        if args.more_tail_peak_amplitude <= 0:
            parser.error(
                "positive --more_tail_rank requires positive "
                "--more_tail_peak_amplitude"
            )
    elif args.more_tail_peak_amplitude != 0:
        parser.error(
            "--more_tail_peak_amplitude must be zero when --more_tail_rank is zero"
        )
    if (
        not math.isfinite(args.more_fewer_rank_loss_weight)
        or args.more_fewer_rank_loss_weight < 0
    ):
        parser.error(
            "--more_fewer_rank_loss_weight must be finite and non-negative"
        )
    if args.more_fewer_rank_loss_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--more_fewer_rank_loss_weight is only supported for --model our_moe"
            )
        if (
            not math.isfinite(args.modality_dropout_prob)
            or not 0.0 < args.modality_dropout_prob <= 1.0
        ):
            parser.error(
                "--more_fewer_rank_loss_weight requires modality dropout in (0, 1]"
            )
    if args.more_fewer_rank_loss_effective_weight is not None and (
        not math.isfinite(args.more_fewer_rank_loss_effective_weight)
        or args.more_fewer_rank_loss_effective_weight < 0
        or args.more_fewer_rank_loss_effective_weight > args.more_fewer_rank_loss_weight
    ):
        parser.error("effective more-fewer rank weight must be in [0, control weight]")
    if (
        not math.isfinite(args.tree_teacher_distill_weight)
        or args.tree_teacher_distill_weight < 0
    ):
        parser.error(
            "--tree_teacher_distill_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.tree_teacher_temperature)
        or args.tree_teacher_temperature <= 0
    ):
        parser.error("--tree_teacher_temperature must be finite and positive")
    teacher_artifact_fields_present = (
        args.tree_teacher_npz is not None
        or args.tree_teacher_npz_sha256 is not None
    )
    if args.tree_teacher_distill_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--tree_teacher_distill_weight is only supported for --model our_moe"
            )
        if args.data != TREE_TEACHER_DATASET:
            parser.error(
                "The v1 tree-teacher artifact is restricted to ABCD"
            )
        if "".join(sorted(set(str(args.modality).upper()))) != "BCGI":
            parser.error(
                "The v1 ABCD tree teacher requires the exact IGCB modality set"
            )
        if not isinstance(args.tree_teacher_npz, str) or not args.tree_teacher_npz:
            parser.error(
                "Enabled tree-teacher distillation requires --tree_teacher_npz"
            )
        digest = args.tree_teacher_npz_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            parser.error(
                "Enabled tree-teacher distillation requires a lowercase "
                "64-character --tree_teacher_npz_sha256"
            )
    elif teacher_artifact_fields_present:
        parser.error(
            "Teacher NPZ path/hash may only be supplied when "
            "--tree_teacher_distill_weight is positive"
        )
    if (
        not math.isfinite(args.external_teacher_distill_weight)
        or args.external_teacher_distill_weight < 0
    ):
        parser.error(
            "--external_teacher_distill_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.external_teacher_temperature)
        or args.external_teacher_temperature <= 0
    ):
        parser.error("--external_teacher_temperature must be finite and positive")
    external_teacher_artifact_fields_present = (
        args.external_teacher_npz is not None
        or args.external_teacher_npz_sha256 is not None
    )
    if args.external_teacher_distill_weight > 0:
        if args.tree_teacher_distill_weight > 0:
            parser.error(
                "v1 tree-teacher and v2 external-teacher distillation are "
                "mutually exclusive"
            )
        if args.model != "our_moe":
            parser.error(
                "--external_teacher_distill_weight is only supported for "
                "--model our_moe"
            )
        if args.data != EXTERNAL_TEACHER_DATASET:
            parser.error("The v2 external-teacher artifact is restricted to ABCD")
        if "".join(sorted(set(str(args.modality).upper()))) != "BCGI":
            parser.error(
                "The v2 ABCD external teacher requires the exact IGCB modality set"
            )
        if (
            not isinstance(args.external_teacher_npz, str)
            or not args.external_teacher_npz
        ):
            parser.error(
                "Enabled external-teacher distillation requires "
                "--external_teacher_npz"
            )
        digest = args.external_teacher_npz_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            parser.error(
                "Enabled external-teacher distillation requires a lowercase "
                "64-character --external_teacher_npz_sha256"
            )
    elif external_teacher_artifact_fields_present:
        parser.error(
            "External-teacher NPZ path/hash may only be supplied when "
            "--external_teacher_distill_weight is positive"
        )
    elif (
        args.external_teacher_trust_mode != "correct_only"
        or args.external_teacher_class_compensation != "none"
    ):
        parser.error(
            "External-teacher trust/compensation modes may be changed only "
            "when --external_teacher_distill_weight is positive"
        )
    if (
        not math.isfinite(args.hard_cvar_dual_boundary_max_weight)
        or args.hard_cvar_dual_boundary_max_weight < 0
    ):
        parser.error(
            "--hard_cvar_dual_boundary_max_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.hard_cvar_dual_boundary_tail_fraction)
        or not 0.0 < args.hard_cvar_dual_boundary_tail_fraction <= 1.0
    ):
        parser.error(
            "--hard_cvar_dual_boundary_tail_fraction must be finite and in (0, 1]"
        )
    if (
        not math.isfinite(args.hard_cvar_dual_boundary_margin)
        or args.hard_cvar_dual_boundary_margin < 0
    ):
        parser.error(
            "--hard_cvar_dual_boundary_margin must be finite and non-negative"
        )
    if (
        not math.isfinite(args.hard_cvar_dual_boundary_10_weight)
        or not 0.0 <= args.hard_cvar_dual_boundary_10_weight <= 1.0
    ):
        parser.error(
            "--hard_cvar_dual_boundary_10_weight must be finite and in [0, 1]"
        )
    if args.hard_cvar_dual_boundary_start_epoch < 1:
        parser.error("--hard_cvar_dual_boundary_start_epoch must be positive")
    if args.hard_cvar_dual_boundary_ramp_epochs < 0:
        parser.error(
            "--hard_cvar_dual_boundary_ramp_epochs must be non-negative"
        )
    if args.hard_cvar_dual_boundary_max_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--hard_cvar_dual_boundary_max_weight is only supported for "
                "--model our_moe"
            )
        if args.hard_cvar_dual_boundary_start_epoch > args.train_epochs:
            parser.error(
                "--hard_cvar_dual_boundary_start_epoch must not exceed train_epochs"
            )
        if args.dual_boundary_rank_loss_weight > 0:
            parser.error(
                "legacy dual-boundary ranking and hard-CVaR dual-boundary "
                "ranking cannot be enabled together"
            )
    if args.normalized_gate_loss and args.model != "our_moe":
        parser.error("--normalized_gate_loss is only supported for --model our_moe")
    if (
        args.uncertainty_aware_ordinal_fusion
        and args.ordinal_head_type != "continuation"
    ):
        parser.error(
            "--uncertainty_aware_ordinal_fusion requires "
            "--ordinal_head_type continuation"
        )
    if (
        args.ordinal_head_type != "proportional"
        and args.ordinal_fusion_weight <= 0
        and args.ordinal_aux_loss_weight <= 0
    ):
        parser.error(
            "A non-default --ordinal_head_type requires a positive ordinal "
            "fusion or auxiliary weight"
        )
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min_lr_ratio must be in [0, 1]")
    if not 0 <= args.lr_warmup_epochs <= args.train_epochs:
        parser.error("--lr_warmup_epochs must be between 0 and train_epochs")
    if not 0 <= args.warm_up_epochs <= args.train_epochs:
        parser.error("--warm_up_epochs must be between 0 and train_epochs")
    if not 0 <= args.sampler_ramp_epochs <= args.train_epochs - args.warm_up_epochs:
        parser.error(
            "--sampler_ramp_epochs must fit between warm-up and train_epochs"
        )
    if not 1 <= args.checkpoint_soup_top_k <= args.train_epochs:
        parser.error("--checkpoint_soup_top_k must be between 1 and train_epochs")
    if args.checkpoint_selection_policy == "final_epoch_refit":
        if not args.validation_only:
            parser.error(
                "--checkpoint_selection_policy final_epoch_refit requires "
                "--validation_only true"
            )
        if args.early_stopping_patience != 0:
            parser.error(
                "--checkpoint_selection_policy final_epoch_refit requires "
                "--early_stopping_patience 0"
            )
        if args.checkpoint_soup_top_k != 1:
            parser.error(
                "--checkpoint_selection_policy final_epoch_refit requires "
                "--checkpoint_soup_top_k 1"
            )
    if args.exclude_bias_norm_from_weight_decay and args.optimizer != "adamw":
        parser.error("Selective decoupled weight decay requires --optimizer adamw")
    if not math.isfinite(args.sam_rho) or args.sam_rho < 0:
        parser.error("--sam_rho must be finite and non-negative")
    if args.sam_rho > 0 and args.model != "our_moe":
        parser.error("--sam_rho is only supported for --model our_moe")
    if not math.isfinite(args.rdrop_loss_weight) or args.rdrop_loss_weight < 0:
        parser.error("--rdrop_loss_weight must be finite and non-negative")
    if args.rdrop_loss_weight > 0 and args.model != "our_moe":
        parser.error(
            "--rdrop_loss_weight is only supported for --model our_moe"
        )
    if args.rdrop_loss_weight > 0 and args.sam_rho > 0:
        parser.error("--rdrop_loss_weight cannot be combined with --sam_rho")
    if (
        not math.isfinite(args.masked_branch_tcl_loss_weight)
        or args.masked_branch_tcl_loss_weight < 0
    ):
        parser.error(
            "--masked_branch_tcl_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.masked_branch_tcl_temperature)
        or args.masked_branch_tcl_temperature <= 0
    ):
        parser.error(
            "--masked_branch_tcl_temperature must be finite and positive"
        )
    if args.masked_branch_tcl_start_epoch < 1:
        parser.error("--masked_branch_tcl_start_epoch must be positive")
    if (
        not math.isfinite(args.masked_branch_tcl_class_acc_ema)
        or not 0.0 <= args.masked_branch_tcl_class_acc_ema < 1.0
    ):
        parser.error(
            "--masked_branch_tcl_class_acc_ema must be finite and in [0, 1)"
        )
    if args.masked_branch_tcl_loss_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--masked_branch_tcl_loss_weight is only supported for "
                "--model our_moe"
            )
        if args.masked_branch_tcl_start_epoch > args.train_epochs:
            parser.error(
                "--masked_branch_tcl_start_epoch must not exceed train_epochs "
                "when masked branch TCL is enabled"
            )
    if (
        not math.isfinite(args.trusted_branch_fusion_distill_weight)
        or args.trusted_branch_fusion_distill_weight < 0
    ):
        parser.error(
            "--trusted_branch_fusion_distill_weight must be finite and "
            "non-negative"
        )
    if (
        not math.isfinite(args.trusted_branch_fusion_temperature)
        or args.trusted_branch_fusion_temperature <= 0
    ):
        parser.error(
            "--trusted_branch_fusion_temperature must be finite and positive"
        )
    if args.trusted_branch_fusion_start_epoch < 1:
        parser.error("--trusted_branch_fusion_start_epoch must be positive")
    if (
        not math.isfinite(args.trusted_branch_fusion_class_acc_ema)
        or not 0.0 <= args.trusted_branch_fusion_class_acc_ema < 1.0
    ):
        parser.error(
            "--trusted_branch_fusion_class_acc_ema must be finite and in [0, 1)"
        )
    if args.trusted_branch_fusion_distill_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--trusted_branch_fusion_distill_weight is only supported for "
                "--model our_moe"
            )
        if args.trusted_branch_fusion_start_epoch > args.train_epochs:
            parser.error(
                "--trusted_branch_fusion_start_epoch must not exceed "
                "train_epochs when TBFD is enabled"
            )
        if args.masked_branch_tcl_loss_weight > 0:
            parser.error(
                "--trusted_branch_fusion_distill_weight and "
                "--masked_branch_tcl_loss_weight are mutually exclusive"
            )
    if args.clear_eval_gate_cache and args.model != "our_moe":
        parser.error("--clear_eval_gate_cache is only supported for --model our_moe")
    if not 0.0 < args.ema_decay < 1.0:
        parser.error("--ema_decay must be in (0, 1)")
    if not 1 <= args.ema_start_epoch <= args.train_epochs:
        parser.error("--ema_start_epoch must be between 1 and train_epochs")
    if not 0.0 <= args.acadiff_mask_probability <= 1.0:
        parser.error("--acadiff_mask_probability must be in [0, 1]")
    if not math.isfinite(args.logit_adjust_tau) or args.logit_adjust_tau < 0:
        parser.error("--logit_adjust_tau must be finite and non-negative")
    if args.logit_adjust_tau > 0 and args.model != "our_moe":
        parser.error("--logit_adjust_tau is only supported for --model our_moe")
    if (
        not math.isfinite(args.class1_aux_loss_weight)
        or args.class1_aux_loss_weight < 0
    ):
        parser.error("--class1_aux_loss_weight must be finite and non-negative")
    if args.class1_aux_loss_weight > 0 and args.model != "our_moe":
        parser.error("--class1_aux_loss_weight is only supported for --model our_moe")
    if (
        not math.isfinite(args.presentation_axis_loss_weight)
        or args.presentation_axis_loss_weight < 0
    ):
        parser.error(
            "--presentation_axis_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.presentation_axis_residual_cap)
        or args.presentation_axis_residual_cap <= 0
    ):
        parser.error(
            "--presentation_axis_residual_cap must be finite and positive"
        )
    if args.presentation_axis_loss_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--presentation_axis_loss_weight is only supported for "
                "--model our_moe"
            )
        incompatible = []
        if args.ordinal_fusion_weight > 0 or args.ordinal_aux_loss_weight > 0:
            incompatible.append("ordinal fusion/auxiliary")
        if args.class1_aux_loss_weight > 0:
            incompatible.append("class-1 auxiliary")
        if args.dual_local_boundary_loss_weight > 0:
            incompatible.append("DLBR")
        if incompatible:
            parser.error(
                "presentation-axis residual is mutually exclusive with "
                + ", ".join(incompatible)
            )
    if args.missing_capacity_residual_width < 0:
        parser.error(
            "--missing_capacity_residual_width must be a non-negative integer"
        )
    if args.missing_capacity_residual_width > 0 and args.model != "our_moe":
        parser.error(
            "--missing_capacity_residual_width is only supported for "
            "--model our_moe"
        )
    if (
        not math.isfinite(args.dual_local_boundary_loss_weight)
        or args.dual_local_boundary_loss_weight < 0
    ):
        parser.error(
            "--dual_local_boundary_loss_weight must be finite and non-negative"
        )
    if args.dual_local_boundary_loss_weight > 0:
        if args.model != "our_moe":
            parser.error(
                "--dual_local_boundary_loss_weight is only supported for "
                "--model our_moe"
            )
        incompatible = []
        if args.ordinal_fusion_weight > 0 or args.ordinal_aux_loss_weight > 0:
            incompatible.append("ordinal fusion/auxiliary")
        if args.class1_aux_loss_weight > 0:
            incompatible.append("class-1 auxiliary")
        if args.dual_boundary_rank_loss_weight > 0:
            incompatible.append("legacy dual-boundary ranking")
        if args.hard_cvar_dual_boundary_max_weight > 0:
            incompatible.append("hard-CVaR dual-boundary ranking")
        if args.more_tail_rank > 0:
            incompatible.append("MORE tail adapter")
        if args.masked_branch_tcl_loss_weight > 0:
            incompatible.append("masked branch TCL")
        if args.trusted_branch_fusion_distill_weight > 0:
            incompatible.append("TBFD")
        if args.presentation_axis_loss_weight > 0:
            incompatible.append("presentation-axis residual")
        if incompatible:
            parser.error(
                "DLBR first-revision protocol is mutually exclusive with "
                + ", ".join(incompatible)
            )
    if args.recon_targets_per_sample < 0:
        parser.error("--recon_targets_per_sample must be non-negative")
    if (
        not math.isfinite(args.recon_normalized_token_loss_weight)
        or args.recon_normalized_token_loss_weight < 0
    ):
        parser.error(
            "--recon_normalized_token_loss_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(args.recon_encoder_gradient_scale)
        or not 0.0 <= args.recon_encoder_gradient_scale <= 1.0
    ):
        parser.error(
            "--recon_encoder_gradient_scale must be finite and in [0, 1]"
        )
    if (
        not math.isfinite(args.recon_context_dropout_probability)
        or not 0.0 <= args.recon_context_dropout_probability <= 1.0
    ):
        parser.error(
            "--recon_context_dropout_probability must be finite and in [0, 1]"
        )
    if (
        args.recon_context_dropout_probability > 0
        and not args.pattern_aware_reconstruction
    ):
        parser.error(
            "--recon_context_dropout_probability requires "
            "--pattern_aware_reconstruction true"
        )
    if args.pattern_aware_reconstruction and args.recon_targets_per_sample == 0:
        parser.error(
            "--pattern_aware_reconstruction requires a positive "
            "--recon_targets_per_sample"
        )
    if (
        args.model == "our_moe"
        and (
            args.pattern_aware_reconstruction
            or args.recon_normalized_token_loss_weight > 0
        )
        and not args.use_generators
    ):
        parser.error("Reconstruction candidates require --use_generators true")
    if args.recon_encoder_gradient_scale != 1.0:
        if args.model != "our_moe":
            parser.error(
                "--recon_encoder_gradient_scale is only supported for "
                "--model our_moe"
            )
        if not args.use_generators:
            parser.error(
                "Non-default --recon_encoder_gradient_scale requires "
                "--use_generators true"
            )
    if args.adni_image_imputation != "legacy_mode":
        if args.data != "adni":
            parser.error("--adni_image_imputation is only supported for ADNI")
        if not args.preprocessed:
            parser.error(
                "Explicit ADNI image imputation requires --preprocessed true"
            )
        if "I" not in str(args.modality).upper():
            parser.error(
                "Explicit ADNI image imputation requires image modality I"
            )
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


TREE_TEACHER_NPZ_SCHEMA_VERSION = 1
TREE_TEACHER_DATASET = "abcd"
TREE_TEACHER_TASK = "psych3_target_classification_IGCB"
TREE_TEACHER_FOLD_COUNT = 5
TREE_TEACHER_NPZ_KEYS = {
    "schema_version",
    "dataset",
    "task",
    "training_ids",
    "training_oof_probabilities",
    "fold_index",
    "class_order",
    "ordered_training_id_sha256",
    "ordered_training_id_label_sha256",
    "task_binding_sha256",
    "generator_protocol_sha256",
}


class CrossFittedTreeTeacherArtifact(NamedTuple):
    """Verified teacher targets aligned to the current training-ID order."""

    probabilities: np.ndarray
    resolved_path: str
    sha256: str
    fold_count: int


EXTERNAL_TEACHER_NPZ_SCHEMA_VERSION = 2
EXTERNAL_TEACHER_DATASET = "abcd"
EXTERNAL_TEACHER_TASK = "psych3_target_classification_IGCB"
EXTERNAL_TEACHER_TEMPORAL_TASK = (
    "adhd_presentation_temporal_v2_target_classification_IGCB"
)
EXTERNAL_TEACHER_FOLD_COUNT = 5
EXTERNAL_TEACHER_LEGACY_KINDS = frozenset(
    {"native_tabm32", "catboost", "trusted_tabm32_catboost"}
)
EXTERNAL_TEACHER_TEMPORAL_KINDS = frozenset(
    {"natural_status_moepp_anymod"}
)
EXTERNAL_TEACHER_TASK_KIND_BINDINGS = {
    EXTERNAL_TEACHER_TASK: EXTERNAL_TEACHER_LEGACY_KINDS,
    EXTERNAL_TEACHER_TEMPORAL_TASK: EXTERNAL_TEACHER_TEMPORAL_KINDS,
}
EXTERNAL_TEACHER_KINDS = frozenset().union(
    *EXTERNAL_TEACHER_TASK_KIND_BINDINGS.values()
)
EXTERNAL_TEACHER_NPZ_KEYS = TREE_TEACHER_NPZ_KEYS | {"teacher_kind"}


class CrossFittedExternalTeacherArtifact(NamedTuple):
    """Verified schema-v2 OOF targets and full-training-set trust statistics."""

    probabilities: np.ndarray
    resolved_path: str
    sha256: str
    fold_count: int
    task: str
    teacher_kind: str
    generator_protocol_sha256: str
    class_recalls: np.ndarray
    class_compensation_weights: np.ndarray
    correct_coverage: float


def _strict_integer_ids(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=False)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def _length_prefixed_string_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(len(values).to_bytes(8, "little"))
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _ordered_id_label_sha256(
    ids: Sequence[str], labels: Sequence[int]
) -> str:
    if len(ids) != len(labels):
        raise ValueError("tree teacher ID/label binding lengths differ")
    digest = hashlib.sha256()
    for sample_id, label in zip(ids, labels):
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(label).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _npz_scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"tree teacher {name} must be a scalar Unicode string")
    return str(array.item())


def _require_sha256_text(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"tree teacher {name} must be lowercase SHA-256 hex")
    return value


def load_cross_fitted_tree_teacher_npz(
    path_value: str,
    expected_sha256: str,
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
    test_ids: Sequence[int],
    canonical_subject_ids: Sequence[str],
    labels: Sequence[int],
    manifest_path_value: str | os.PathLike[str],
    num_classes: int,
) -> CrossFittedTreeTeacherArtifact:
    """Load a hash-locked CatBoost OOF artifact for the exact current train IDs.

    The file is read into one immutable byte buffer, hashed, then parsed from
    that same buffer with pickle disabled.  No path is opened a second time.
    Errors report only aggregate counts/digests and never reveal subject IDs.
    """

    if not isinstance(path_value, str) or not path_value:
        raise ValueError("tree teacher path must be a non-empty string")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("tree teacher SHA-256 must be lowercase 64-character hex")
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")

    path = Path(path_value).expanduser().resolve()
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise ValueError(f"tree teacher artifact must be an existing NPZ: {path}")
    try:
        artifact_bytes = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read tree teacher NPZ: {error}") from error
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha256 != expected_sha256:
        raise ValueError(
            "tree teacher SHA-256 mismatch: "
            f"expected {expected_sha256}, found {artifact_sha256}"
        )

    try:
        with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as archive:
            if (
                len(archive.files) != len(TREE_TEACHER_NPZ_KEYS)
                or set(archive.files) != TREE_TEACHER_NPZ_KEYS
            ):
                raise ValueError(
                    "tree teacher NPZ exact-key schema mismatch "
                    f"(expected_count={len(TREE_TEACHER_NPZ_KEYS)}, "
                    f"found_count={len(archive.files)})"
                )
            schema_version = np.asarray(archive["schema_version"])
            dataset_name = _npz_scalar_string(archive["dataset"], "dataset")
            task_name = _npz_scalar_string(archive["task"], "task")
            teacher_ids = np.asarray(archive["training_ids"])
            probabilities = np.asarray(archive["training_oof_probabilities"])
            fold_indices = np.asarray(archive["fold_index"])
            class_indices = np.asarray(archive["class_order"])
            saved_id_sha256 = _require_sha256_text(
                _npz_scalar_string(
                    archive["ordered_training_id_sha256"],
                    "ordered_training_id_sha256",
                ),
                "ordered_training_id_sha256",
            )
            saved_id_label_sha256 = _require_sha256_text(
                _npz_scalar_string(
                    archive["ordered_training_id_label_sha256"],
                    "ordered_training_id_label_sha256",
                ),
                "ordered_training_id_label_sha256",
            )
            saved_task_binding_sha256 = _require_sha256_text(
                _npz_scalar_string(
                    archive["task_binding_sha256"], "task_binding_sha256"
                ),
                "task_binding_sha256",
            )
            _require_sha256_text(
                _npz_scalar_string(
                    archive["generator_protocol_sha256"],
                    "generator_protocol_sha256",
                ),
                "generator_protocol_sha256",
            )
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ValueError) and str(error).startswith("tree teacher"):
            raise
        raise ValueError(f"could not safely parse tree teacher NPZ: {error}") from error

    if (
        schema_version.shape != ()
        or schema_version.dtype != np.dtype(np.int64)
        or int(schema_version.item()) != TREE_TEACHER_NPZ_SCHEMA_VERSION
    ):
        raise ValueError("tree teacher schema_version must be scalar int64 value 1")
    if dataset_name != TREE_TEACHER_DATASET or task_name != TREE_TEACHER_TASK:
        raise ValueError("tree teacher dataset/task binding is incompatible")
    if (
        teacher_ids.ndim != 1
        or teacher_ids.dtype.kind != "U"
        or bool(np.equal(teacher_ids, "").any())
        or len(np.unique(teacher_ids)) != len(teacher_ids)
    ):
        raise ValueError(
            "tree teacher training_ids must be unique non-empty canonical Unicode IDs"
        )

    current_train_ids = _strict_integer_ids(train_ids, "current training ids")
    current_validation_ids = _strict_integer_ids(
        validation_ids, "current validation ids"
    )
    current_test_ids = _strict_integer_ids(test_ids, "current test ids")
    canonical_ids = np.asarray(canonical_subject_ids)
    all_labels = np.asarray(labels)
    if canonical_ids.ndim != 1 or canonical_ids.dtype.kind not in {"U", "S", "O"}:
        raise ValueError("canonical subject IDs must be a one-dimensional string array")
    canonical_ids = canonical_ids.astype(np.str_, copy=False)
    if all_labels.ndim != 1 or len(all_labels) != len(canonical_ids):
        raise ValueError("canonical subject IDs and labels must have equal lengths")
    all_split_indices = np.concatenate(
        (current_train_ids, current_validation_ids, current_test_ids)
    )
    if bool((all_split_indices < 0).any()) or bool(
        (all_split_indices >= len(canonical_ids)).any()
    ):
        raise ValueError("current split indices are outside canonical subject rows")
    current_training_subjects = canonical_ids[current_train_ids]
    current_validation_subjects = canonical_ids[current_validation_ids]
    current_test_subjects = canonical_ids[current_test_ids]
    forbidden_count = len(
        set(teacher_ids.tolist())
        & (
            set(current_validation_subjects.tolist())
            | set(current_test_subjects.tolist())
        )
    )
    if forbidden_count:
        raise ValueError(
            "tree teacher artifact contains validation/test IDs "
            f"(forbidden_count={forbidden_count})"
        )
    teacher_id_set = set(teacher_ids.tolist())
    train_id_set = set(current_training_subjects.tolist())
    if teacher_id_set != train_id_set:
        missing = len(train_id_set - teacher_id_set)
        extra = len(teacher_id_set - train_id_set)
        raise ValueError(
            "tree teacher IDs must exactly equal current training IDs "
            f"(missing={missing}, extra={extra})"
        )
    if not np.array_equal(teacher_ids, current_training_subjects):
        raise ValueError(
            "tree teacher IDs cover the current training split but canonical "
            "order differs"
        )

    current_training_labels = all_labels[current_train_ids].astype(np.int64, copy=False)
    recomputed_id_sha256 = _length_prefixed_string_sha256(
        current_training_subjects.tolist()
    )
    if saved_id_sha256 != recomputed_id_sha256:
        raise ValueError("tree teacher ordered training-ID SHA-256 mismatch")
    recomputed_id_label_sha256 = _ordered_id_label_sha256(
        current_training_subjects.tolist(), current_training_labels.tolist()
    )
    if saved_id_label_sha256 != recomputed_id_label_sha256:
        raise ValueError("tree teacher ordered training ID-label SHA-256 mismatch")

    manifest_path = Path(manifest_path_value).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError("ABCD manifest for tree-teacher binding does not exist")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("could not read ABCD manifest for teacher binding") from error
    try:
        task_binding = {
            "dataset": TREE_TEACHER_DATASET,
            "task": TREE_TEACHER_TASK,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "label_column": str(manifest["label"]["column"]),
            "class_values": list(manifest["label"]["class_values"]),
            "modality_codes": sorted(
                str(item["code"]).upper() for item in manifest["modalities"]
            ),
            "ordered_training_ids_sha256": recomputed_id_sha256,
        }
    except (KeyError, TypeError) as error:
        raise ValueError("ABCD manifest lacks tree-teacher task binding fields") from error
    if saved_task_binding_sha256 != _canonical_json_sha256(task_binding):
        raise ValueError("tree teacher task-binding SHA-256 mismatch")

    if (
        class_indices.ndim != 1
        or class_indices.dtype != np.dtype(np.int64)
        or not np.array_equal(
            class_indices, np.arange(num_classes, dtype=np.int64)
        )
    ):
        raise ValueError(
            "tree teacher class_indices must be the exact ordered range "
            f"[0, {num_classes})"
        )
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(teacher_ids), num_classes)
        or probabilities.dtype != np.dtype(np.float64)
    ):
        raise ValueError(
            "tree teacher probabilities must be a floating array with shape "
            f"({len(teacher_ids)}, {num_classes})"
        )
    probabilities = probabilities.astype(np.float64, copy=False)
    if not np.isfinite(probabilities).all():
        raise ValueError("tree teacher probabilities must all be finite")
    if bool((probabilities < 0.0).any()) or bool((probabilities > 1.0).any()):
        raise ValueError("tree teacher probabilities must lie in [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ValueError("every tree teacher probability row must sum to one")

    if (
        fold_indices.ndim != 1
        or fold_indices.shape != teacher_ids.shape
        or fold_indices.dtype != np.dtype(np.int64)
    ):
        raise ValueError("tree teacher fold_indices must be one integer per ID")
    unique_folds = np.unique(fold_indices)
    if (
        len(unique_folds) != TREE_TEACHER_FOLD_COUNT
        or not np.array_equal(
            unique_folds, np.arange(TREE_TEACHER_FOLD_COUNT, dtype=np.int64)
        )
    ):
        raise ValueError(
            "tree teacher fold_index must contain every zero-based family fold 0..4"
        )

    aligned = probabilities.astype(np.float32, copy=False)
    return CrossFittedTreeTeacherArtifact(
        probabilities=aligned,
        resolved_path=str(path),
        sha256=artifact_sha256,
        fold_count=len(unique_folds),
    )


def attach_cross_fitted_tree_teacher(
    train_dataset: Any,
    train_ids: Sequence[int],
    probabilities_in_train_order: np.ndarray,
) -> None:
    """Attach verified targets to the shared sorted/shuffled train dataset."""

    dataset_ids = _strict_integer_ids(train_dataset.ids, "train dataset ids")
    expected_ids = _strict_integer_ids(train_ids, "current training ids")
    if not np.array_equal(dataset_ids, expected_ids):
        raise ValueError("train dataset ID order does not match current training IDs")
    probabilities = np.asarray(probabilities_in_train_order)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(dataset_ids):
        raise ValueError("aligned tree teacher probabilities have invalid shape")
    sorted_ids = np.asarray(train_dataset.sorted_ids)
    if (
        sorted_ids.ndim != 1
        or not np.issubdtype(sorted_ids.dtype, np.integer)
        or not np.array_equal(np.sort(sorted_ids), np.arange(len(dataset_ids)))
    ):
        raise ValueError("train dataset sorted_ids is not a valid permutation")
    if getattr(train_dataset, "cross_fitted_teacher_probabilities", None) is not None:
        raise ValueError("tree teacher probabilities are already attached")
    train_dataset.cross_fitted_teacher_probabilities = np.ascontiguousarray(
        probabilities[sorted_ids], dtype=np.float32
    )


def _external_npz_scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(
            f"external teacher {name} must be a scalar Unicode string"
        )
    return str(array.item())


def _external_require_sha256_text(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"external teacher {name} must be lowercase SHA-256 hex"
        )
    return value


def external_teacher_oof_class_statistics(
    probabilities: np.ndarray,
    training_labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return full-OOF class recall, TCL-style weights, and correct coverage.

    For class ``c``, ``alpha_c`` is hard-prediction recall on every current
    training row of that class.  The compensation vector is computed once as
    ``C * sqrt(1 - alpha_c) / sum_j sqrt(1 - alpha_j)``; it is never estimated
    from a minibatch.  A uniformly perfect teacher uses the continuous
    symmetric limit (all-one weights), because the literal denominator is
    zero and no class needs relative error compensation.
    """

    probability_array = np.asarray(probabilities)
    label_array = np.asarray(training_labels)
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")
    if (
        probability_array.ndim != 2
        or probability_array.shape[1] != num_classes
        or probability_array.shape[0] != label_array.shape[0]
    ):
        raise ValueError(
            "external teacher probabilities/labels have incompatible shapes"
        )
    if label_array.ndim != 1 or not np.issubdtype(
        label_array.dtype, np.integer
    ):
        raise ValueError("external teacher training labels must be integer vector")
    labels_int64 = label_array.astype(np.int64, copy=False)
    if bool((labels_int64 < 0).any()) or bool(
        (labels_int64 >= num_classes).any()
    ):
        raise ValueError("external teacher training labels are outside class range")
    if not np.isfinite(probability_array).all():
        raise ValueError("external teacher probabilities must be finite")
    teacher_prediction = probability_array.argmax(axis=1)
    correct = teacher_prediction == labels_int64
    recalls = np.empty(num_classes, dtype=np.float64)
    for class_index in range(num_classes):
        class_rows = labels_int64 == class_index
        if not bool(class_rows.any()):
            raise ValueError(
                "external teacher class compensation requires every class in "
                "the current training split"
            )
        recalls[class_index] = float(correct[class_rows].mean())
    square_root_error = np.sqrt(1.0 - recalls)
    denominator = float(square_root_error.sum())
    if denominator == 0.0:
        compensation = np.ones(num_classes, dtype=np.float64)
    else:
        compensation = (
            float(num_classes) * square_root_error / denominator
        ).astype(np.float64, copy=False)
    return recalls, compensation, float(correct.mean())


def load_cross_fitted_external_teacher_npz(
    path_value: str,
    expected_sha256: str,
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
    test_ids: Sequence[int],
    canonical_subject_ids: Sequence[str],
    labels: Sequence[int],
    manifest_path_value: str | os.PathLike[str],
    num_classes: int,
) -> CrossFittedExternalTeacherArtifact:
    """Strictly load a schema-v2 external OOF teacher for current train IDs."""

    if not isinstance(path_value, str) or not path_value:
        raise ValueError("external teacher path must be a non-empty string")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(
            "external teacher SHA-256 must be lowercase 64-character hex"
        )
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")

    path = Path(path_value).expanduser().resolve()
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise ValueError(
            f"external teacher artifact must be an existing NPZ: {path}"
        )
    try:
        artifact_bytes = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read external teacher NPZ: {error}") from error
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha256 != expected_sha256:
        raise ValueError(
            "external teacher SHA-256 mismatch: "
            f"expected {expected_sha256}, found {artifact_sha256}"
        )

    try:
        with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as archive:
            if (
                len(archive.files) != len(EXTERNAL_TEACHER_NPZ_KEYS)
                or set(archive.files) != EXTERNAL_TEACHER_NPZ_KEYS
            ):
                raise ValueError(
                    "external teacher NPZ exact-key schema mismatch "
                    f"(expected_count={len(EXTERNAL_TEACHER_NPZ_KEYS)}, "
                    f"found_count={len(archive.files)})"
                )
            schema_version = np.asarray(archive["schema_version"])
            dataset_name = _external_npz_scalar_string(
                archive["dataset"], "dataset"
            )
            task_name = _external_npz_scalar_string(archive["task"], "task")
            teacher_kind = _external_npz_scalar_string(
                archive["teacher_kind"], "teacher_kind"
            )
            teacher_ids = np.asarray(archive["training_ids"])
            probabilities = np.asarray(archive["training_oof_probabilities"])
            fold_indices = np.asarray(archive["fold_index"])
            class_indices = np.asarray(archive["class_order"])
            saved_id_sha256 = _external_require_sha256_text(
                _external_npz_scalar_string(
                    archive["ordered_training_id_sha256"],
                    "ordered_training_id_sha256",
                ),
                "ordered_training_id_sha256",
            )
            saved_id_label_sha256 = _external_require_sha256_text(
                _external_npz_scalar_string(
                    archive["ordered_training_id_label_sha256"],
                    "ordered_training_id_label_sha256",
                ),
                "ordered_training_id_label_sha256",
            )
            saved_task_binding_sha256 = _external_require_sha256_text(
                _external_npz_scalar_string(
                    archive["task_binding_sha256"], "task_binding_sha256"
                ),
                "task_binding_sha256",
            )
            generator_protocol_sha256 = _external_require_sha256_text(
                _external_npz_scalar_string(
                    archive["generator_protocol_sha256"],
                    "generator_protocol_sha256",
                ),
                "generator_protocol_sha256",
            )
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "external teacher"
        ):
            raise
        raise ValueError(
            f"could not safely parse external teacher NPZ: {error}"
        ) from error

    if (
        schema_version.shape != ()
        or schema_version.dtype != np.dtype(np.int64)
        or int(schema_version.item()) != EXTERNAL_TEACHER_NPZ_SCHEMA_VERSION
    ):
        raise ValueError(
            "external teacher schema_version must be scalar int64 value 2"
        )
    if dataset_name != EXTERNAL_TEACHER_DATASET:
        raise ValueError("external teacher dataset binding is incompatible")
    if task_name not in EXTERNAL_TEACHER_TASK_KIND_BINDINGS:
        raise ValueError("external teacher task binding is incompatible")
    allowed_teacher_kinds = EXTERNAL_TEACHER_TASK_KIND_BINDINGS[task_name]
    if teacher_kind not in allowed_teacher_kinds:
        raise ValueError(
            "external teacher teacher_kind must be one of the kinds bound to "
            f"task={task_name!r}: "
            f"{sorted(allowed_teacher_kinds)}"
        )
    if (
        teacher_ids.ndim != 1
        or teacher_ids.dtype.kind != "U"
        or bool(np.equal(teacher_ids, "").any())
        or len(np.unique(teacher_ids)) != len(teacher_ids)
    ):
        raise ValueError(
            "external teacher training_ids must be unique non-empty canonical "
            "Unicode IDs"
        )

    current_train_ids = _strict_integer_ids(train_ids, "current training ids")
    current_validation_ids = _strict_integer_ids(
        validation_ids, "current validation ids"
    )
    current_test_ids = _strict_integer_ids(test_ids, "current test ids")
    canonical_ids = np.asarray(canonical_subject_ids)
    all_labels = np.asarray(labels)
    if canonical_ids.ndim != 1 or canonical_ids.dtype.kind not in {"U", "S", "O"}:
        raise ValueError(
            "canonical subject IDs must be a one-dimensional string array"
        )
    canonical_ids = canonical_ids.astype(np.str_, copy=False)
    if (
        all_labels.ndim != 1
        or not np.issubdtype(all_labels.dtype, np.integer)
        or len(all_labels) != len(canonical_ids)
    ):
        raise ValueError(
            "canonical subject IDs and integer labels must have equal lengths"
        )
    all_split_indices = np.concatenate(
        (current_train_ids, current_validation_ids, current_test_ids)
    )
    if bool((all_split_indices < 0).any()) or bool(
        (all_split_indices >= len(canonical_ids)).any()
    ):
        raise ValueError("current split indices are outside canonical subject rows")
    current_training_subjects = canonical_ids[current_train_ids]
    current_validation_subjects = canonical_ids[current_validation_ids]
    current_test_subjects = canonical_ids[current_test_ids]
    forbidden_count = len(
        set(teacher_ids.tolist())
        & (
            set(current_validation_subjects.tolist())
            | set(current_test_subjects.tolist())
        )
    )
    if forbidden_count:
        raise ValueError(
            "external teacher artifact contains validation/test IDs "
            f"(forbidden_count={forbidden_count})"
        )
    teacher_id_set = set(teacher_ids.tolist())
    train_id_set = set(current_training_subjects.tolist())
    if teacher_id_set != train_id_set:
        missing = len(train_id_set - teacher_id_set)
        extra = len(teacher_id_set - train_id_set)
        raise ValueError(
            "external teacher IDs must exactly equal current training IDs "
            f"(missing={missing}, extra={extra})"
        )
    if not np.array_equal(teacher_ids, current_training_subjects):
        raise ValueError(
            "external teacher IDs cover the current training split but "
            "canonical order differs"
        )

    current_training_labels = all_labels[current_train_ids].astype(
        np.int64, copy=False
    )
    if bool((current_training_labels < 0).any()) or bool(
        (current_training_labels >= num_classes).any()
    ):
        raise ValueError("current training labels are outside configured classes")
    recomputed_id_sha256 = _length_prefixed_string_sha256(
        current_training_subjects.tolist()
    )
    if saved_id_sha256 != recomputed_id_sha256:
        raise ValueError(
            "external teacher ordered training-ID SHA-256 mismatch"
        )
    recomputed_id_label_sha256 = _ordered_id_label_sha256(
        current_training_subjects.tolist(), current_training_labels.tolist()
    )
    if saved_id_label_sha256 != recomputed_id_label_sha256:
        raise ValueError(
            "external teacher ordered training ID-label SHA-256 mismatch"
        )

    manifest_path = Path(manifest_path_value).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(
            "ABCD manifest for external-teacher binding does not exist"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "could not read ABCD manifest for external-teacher binding"
        ) from error
    try:
        task_binding = {
            "dataset": EXTERNAL_TEACHER_DATASET,
            "task": task_name,
            "teacher_kind": teacher_kind,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "label_column": str(manifest["label"]["column"]),
            "class_values": list(manifest["label"]["class_values"]),
            "modality_codes": sorted(
                str(item["code"]).upper() for item in manifest["modalities"]
            ),
            "ordered_training_ids_sha256": recomputed_id_sha256,
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            "ABCD manifest lacks external-teacher task binding fields"
        ) from error
    if saved_task_binding_sha256 != _canonical_json_sha256(task_binding):
        raise ValueError("external teacher task-binding SHA-256 mismatch")

    if (
        class_indices.ndim != 1
        or class_indices.dtype != np.dtype(np.int64)
        or not np.array_equal(
            class_indices, np.arange(num_classes, dtype=np.int64)
        )
    ):
        raise ValueError(
            "external teacher class_order must be the exact ordered range "
            f"[0, {num_classes})"
        )
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(teacher_ids), num_classes)
        or probabilities.dtype != np.dtype(np.float64)
    ):
        raise ValueError(
            "external teacher probabilities must be float64 with shape "
            f"({len(teacher_ids)}, {num_classes})"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("external teacher probabilities must all be finite")
    if bool((probabilities < 0.0).any()) or bool((probabilities > 1.0).any()):
        raise ValueError("external teacher probabilities must lie in [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            "every external teacher probability row must sum to one"
        )
    if (
        fold_indices.ndim != 1
        or fold_indices.shape != teacher_ids.shape
        or fold_indices.dtype != np.dtype(np.int64)
    ):
        raise ValueError(
            "external teacher fold_index must be one int64 value per ID"
        )
    unique_folds = np.unique(fold_indices)
    if (
        len(unique_folds) != EXTERNAL_TEACHER_FOLD_COUNT
        or not np.array_equal(
            unique_folds,
            np.arange(EXTERNAL_TEACHER_FOLD_COUNT, dtype=np.int64),
        )
    ):
        raise ValueError(
            "external teacher fold_index must contain every zero-based "
            "family-disjoint OOF fold 0..4"
        )

    operational_probabilities = probabilities.astype(np.float32, copy=False)
    if not np.array_equal(
        probabilities.argmax(axis=1),
        operational_probabilities.argmax(axis=1),
    ):
        raise ValueError(
            "external teacher float32 training targets must preserve every "
            "float64 hard prediction"
        )
    recalls, compensation, correct_coverage = (
        external_teacher_oof_class_statistics(
            probabilities, current_training_labels, num_classes
        )
    )
    return CrossFittedExternalTeacherArtifact(
        probabilities=operational_probabilities,
        resolved_path=str(path),
        sha256=artifact_sha256,
        fold_count=len(unique_folds),
        task=task_name,
        teacher_kind=teacher_kind,
        generator_protocol_sha256=generator_protocol_sha256,
        class_recalls=recalls,
        class_compensation_weights=compensation,
        correct_coverage=correct_coverage,
    )


def attach_cross_fitted_external_teacher(
    train_dataset: Any,
    train_ids: Sequence[int],
    probabilities_in_train_order: np.ndarray,
) -> None:
    """Attach verified v2 targets to the shared train dataset only."""

    dataset_ids = _strict_integer_ids(train_dataset.ids, "train dataset ids")
    expected_ids = _strict_integer_ids(train_ids, "current training ids")
    if not np.array_equal(dataset_ids, expected_ids):
        raise ValueError(
            "train dataset ID order does not match current training IDs"
        )
    probabilities = np.asarray(probabilities_in_train_order)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(dataset_ids):
        raise ValueError(
            "aligned external teacher probabilities have invalid shape"
        )
    sorted_ids = np.asarray(train_dataset.sorted_ids)
    if (
        sorted_ids.ndim != 1
        or not np.issubdtype(sorted_ids.dtype, np.integer)
        or not np.array_equal(np.sort(sorted_ids), np.arange(len(dataset_ids)))
    ):
        raise ValueError("train dataset sorted_ids is not a valid permutation")
    if getattr(train_dataset, "cross_fitted_teacher_probabilities", None) is not None:
        raise ValueError("external teacher probabilities are already attached")
    train_dataset.cross_fitted_teacher_probabilities = np.ascontiguousarray(
        probabilities[sorted_ids], dtype=np.float32
    )


def training_class_prior(
    labels: np.ndarray | list[int],
    train_ids: np.ndarray | list[int],
    num_classes: int,
) -> np.ndarray:
    """Return the empirical class prior using training rows only."""
    train_labels = np.asarray(labels)[np.asarray(train_ids)]
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    return counts / max(float(counts.sum()), 1.0)


class LogitAdjustedCrossEntropy(nn.Module):
    """Training-only CE on ``logits + tau * log(training_prior)``."""

    def __init__(
        self,
        class_prior,
        tau: float = 0.0,
        label_smoothing: float = 0.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        prior = torch.as_tensor(class_prior, dtype=torch.float32)
        if prior.ndim != 1 or prior.numel() < 2:
            raise ValueError("class_prior must be a one-dimensional class vector")
        if not torch.isfinite(prior).all() or bool((prior < 0).any()):
            raise ValueError("class_prior must be finite and non-negative")
        if float(prior.sum()) <= 0:
            raise ValueError("class_prior must have positive mass")
        if not math.isfinite(tau) or tau < 0:
            raise ValueError("tau must be finite and non-negative")
        prior = prior / prior.sum().clamp_min(1e-12)
        self.register_buffer("log_prior", torch.log(prior.clamp_min(1e-12)))
        self.register_buffer("weight", weight)
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits
        if self.tau != 0:
            adjusted_logits = logits + self.tau * self.log_prior.to(logits.device)
        return torch.nn.functional.cross_entropy(
            adjusted_logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


def per_sample_cross_entropy(
    criterion: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Apply the configured CE criterion without reducing across samples.

    The runner uses either ``CrossEntropyLoss`` or its training-only
    logit-adjusted variant.  SimMLM's ranking is defined on per-sample losses,
    so calling the scalar criterion twice would erase the strict-subset
    comparison and would mishandle logit adjustment.
    """

    if logits.ndim != 2 or target.ndim != 1 or logits.shape[0] != target.shape[0]:
        raise ValueError("per-sample cross entropy expects (batch, classes) logits")
    if isinstance(criterion, LogitAdjustedCrossEntropy):
        adjusted_logits = logits
        if criterion.tau != 0:
            adjusted_logits = logits + criterion.tau * criterion.log_prior.to(
                logits.device
            )
        return torch.nn.functional.cross_entropy(
            adjusted_logits,
            target,
            weight=criterion.weight,
            reduction="none",
            label_smoothing=criterion.label_smoothing,
        )
    if isinstance(criterion, nn.CrossEntropyLoss):
        return torch.nn.functional.cross_entropy(
            logits,
            target,
            weight=criterion.weight,
            ignore_index=criterion.ignore_index,
            reduction="none",
            label_smoothing=criterion.label_smoothing,
        )
    raise TypeError(
        "Per-sample CE requires CrossEntropyLoss or LogitAdjustedCrossEntropy"
    )


def joint_member_cross_entropy(
    member_logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Mean of independently optimized TabM member classification losses."""

    if member_logits.ndim != 3:
        raise ValueError("joint member logits must have shape (batch, members, classes)")
    if labels.ndim != 1 or labels.shape[0] != member_logits.shape[0]:
        raise ValueError("labels must match the joint member batch")
    if member_logits.shape[1] <= 1:
        raise ValueError("joint member CE requires at least two members")
    return torch.stack(
        [
            criterion(member_logits[:, member_index, :], labels)
            for member_index in range(member_logits.shape[1])
        ]
    ).mean()


def more_fewer_rank_loss(
    more_logits: torch.Tensor,
    fewer_logits: torch.Tensor,
    labels: torch.Tensor,
    more_observed_mask: torch.Tensor,
    fewer_observed_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Original SimMLM modality ranking: mean relu(CE_more - CE_fewer).

    Only rows whose observed-modality set is a strict superset contribute.  No
    margin, detach, or comparison against rows that happened not to drop is
    introduced.
    """

    if more_logits.shape != fewer_logits.shape:
        raise ValueError("more/fewer logits must have identical shapes")
    if more_observed_mask.shape != fewer_observed_mask.shape:
        raise ValueError("more/fewer observed masks must have identical shapes")
    if more_observed_mask.ndim != 2 or more_observed_mask.shape[0] != labels.shape[0]:
        raise ValueError("more/fewer observed masks must match the label batch")
    more_observed_mask = more_observed_mask.bool()
    fewer_observed_mask = fewer_observed_mask.bool()
    if bool((fewer_observed_mask & ~more_observed_mask).any()):
        raise ValueError("fewer modalities must be a subset of more modalities")
    strict_subset = fewer_observed_mask.sum(dim=1) < more_observed_mask.sum(dim=1)
    if not bool(strict_subset.any()):
        return (more_logits.sum() + fewer_logits.sum()) * 0.0
    more_ce = per_sample_cross_entropy(criterion, more_logits, labels)
    fewer_ce = per_sample_cross_entropy(criterion, fewer_logits, labels)
    return torch.relu(more_ce[strict_subset] - fewer_ce[strict_subset]).mean()


class StepwiseCosineSchedule:
    """Closed-form per-update warmup + cosine schedule without resume state."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= warmup_steps <= total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps]")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_index = 0

    def factor(self, step_index: int) -> float:
        if self.warmup_steps > 0 and step_index < self.warmup_steps:
            return float(step_index + 1) / float(self.warmup_steps)
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        decay_index = max(0, step_index - self.warmup_steps)
        progress = min(1.0, decay_index / float(max(1, decay_steps - 1)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def set_current_lr(self) -> None:
        factor = self.factor(self.step_index)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def advance(self) -> None:
        self.step_index += 1


class StepwiseStateEMA:
    """Same-device EMA shadow for a model and all modality encoders."""

    def __init__(self, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.model_shadow: dict[str, torch.Tensor] | None = None
        self.encoder_shadows: dict[str, dict[str, torch.Tensor]] | None = None
        self.num_updates = 0

    @staticmethod
    def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }

    @staticmethod
    def _update_state(
        shadow: dict[str, torch.Tensor], current: dict[str, torch.Tensor], decay: float
    ) -> None:
        if shadow.keys() != current.keys():
            raise RuntimeError("EMA state keys changed during training")
        with torch.no_grad():
            for name, current_value in current.items():
                target = shadow[name]
                if (
                    target.shape != current_value.shape
                    or target.dtype != current_value.dtype
                    or target.device != current_value.device
                ):
                    raise RuntimeError(f"EMA tensor metadata changed for {name!r}")
                detached = current_value.detach()
                if torch.is_floating_point(target) or torch.is_complex(target):
                    target.mul_(decay).add_(detached, alpha=1.0 - decay)
                else:
                    target.copy_(detached)

    def update(self, model: nn.Module, encoders: dict[str, nn.Module]) -> None:
        if self.model_shadow is None:
            # The first eligible post-step update is copied exactly; random
            # initialization is never mixed into the moving average.
            self.model_shadow = self._clone_state(model)
            self.encoder_shadows = {
                name: self._clone_state(encoder) for name, encoder in encoders.items()
            }
        else:
            if self.encoder_shadows is None or self.encoder_shadows.keys() != encoders.keys():
                raise RuntimeError("EMA encoder set changed during training")
            self._update_state(self.model_shadow, model.state_dict(), self.decay)
            for name, encoder in encoders.items():
                self._update_state(
                    self.encoder_shadows[name], encoder.state_dict(), self.decay
                )
        self.num_updates += 1

    @property
    def ready(self) -> bool:
        return self.model_shadow is not None and self.encoder_shadows is not None

    @contextmanager
    def average_parameters(self, model: nn.Module, encoders: dict[str, nn.Module]):
        if not self.ready:
            raise RuntimeError("EMA has no updates")
        live_model = self._clone_state(model)
        live_encoders = {
            name: self._clone_state(encoder) for name, encoder in encoders.items()
        }
        try:
            model.load_state_dict(self.model_shadow, strict=True)
            for name, encoder in encoders.items():
                encoder.load_state_dict(self.encoder_shadows[name], strict=True)
            yield
        finally:
            model.load_state_dict(live_model, strict=True)
            for name, encoder in encoders.items():
                encoder.load_state_dict(live_encoders[name], strict=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


class ArtifactCollisionError(RuntimeError):
    """Raised before training when its artifact key is already occupied."""


def training_claim_path(args) -> Path:
    base = Path(args.output_dir).resolve()
    stem = matched_artifact_stem(args)
    return base / ".training_claims" / args.data / f"{stem}.lock"


def matched_ablation_audit_config(args) -> dict[str, Any]:
    configured=float(getattr(args,"more_fewer_rank_loss_weight",0.0))
    requested=getattr(args,"more_fewer_rank_loss_effective_weight",None)
    effective=configured if requested is None else float(requested)
    flags=("dense_backbone","disable_provenance","uniform_branch_weights",
           "joint_branch_only",
           "mean_pooling_only","disable_stochastic_context_masking",
           "disable_completion","no_output_gate")
    return {"configured_rank_weight":configured,"effective_rank_weight":effective,
      "rank_enabled":effective>0,"reduced_view_forward_retained":configured>0,
      "structural_flags":{name:bool(getattr(args,name,False)) for name in flags},
      "data_order_seed":getattr(args,"data_order_seed",None)}


def matched_artifact_stem(args) -> str:
    base=f"{args.model}_seed{args.seed}"
    audit=matched_ablation_audit_config(args)
    formal=(audit["data_order_seed"] is not None or any(audit["structural_flags"].values())
            or getattr(args,"more_fewer_rank_loss_effective_weight",None) is not None)
    if not formal: return base
    digest=hashlib.sha256(json.dumps(audit,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:12]
    return f"{base}_arm{digest}"


def acquire_training_claim(args) -> Path:
    """Reserve dataset/model/seed before CUDA/data work using O_EXCL."""

    result_path, progress_path, checkpoint_path, predictions_path = result_paths(args)
    validation_predictions_path = predictions_path.with_name(
        f"{predictions_path.stem}.validation{predictions_path.suffix}"
    )
    claim_path = training_claim_path(args)
    protected_paths = (
        result_path,
        progress_path,
        checkpoint_path,
        predictions_path,
        validation_predictions_path,
        claim_path,
    )
    existing = [path for path in protected_paths if os.path.lexists(path)]
    if existing and not args.allow_overwrite:
        raise ArtifactCollisionError(
            "Refusing to overwrite or share an occupied training key; use a new "
            f"--output_dir (existing: {', '.join(str(path) for path in existing)})"
        )

    claim_path.parent.mkdir(parents=True, exist_ok=True)
    if args.allow_overwrite and os.path.lexists(claim_path):
        # Preserve the original audit marker.  The explicit overwrite attempt
        # receives one deterministic claim, so concurrent overwrite attempts
        # still collide instead of both training.
        claim_path = claim_path.with_name(
            f"{claim_path.stem}.overwrite{claim_path.suffix}"
        )
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "training_claimed",
                    "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "pid": os.getpid(),
                    "model": args.model,
                    "dataset": args.data,
                    "seed": args.seed,
                    "output_dir": str(Path(args.output_dir).resolve()),
                    "allow_overwrite": args.allow_overwrite,
                    "matched_ablation_audit": matched_ablation_audit_config(args),
                },
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ArtifactCollisionError(
            f"Training key was claimed concurrently; refusing to start: {claim_path}"
        ) from error
    return claim_path


def external_class(relative_path: str, class_name: str):
    path = ROOT / relative_path
    module_name = "_shared_baseline_" + relative_path.replace("/", "_").replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {class_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


class FlexMoEBaseline(nn.Module):
    """Official-derived Flex-MoE backbone with its modality-combination bank."""

    def __init__(self, args, num_modalities: int, num_classes: int, full_modality_index: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.missing_bank = nn.Parameter(
            torch.randn(
                2**num_modalities - 1,
                num_modalities,
                args.num_patches,
                args.hidden_dim,
            )
            * args.missing_bank_init_std
        )
        self.backbone = FlexMoE(
            num_modalities=num_modalities,
            full_modality_index=full_modality_index,
            num_patches=args.num_patches,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers_fus,
            num_experts=args.num_experts,
            num_routers=args.num_routers,
            top_k=args.top_k,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
        self.classifier = MLP(
            input_dim=args.hidden_dim * num_modalities,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers_pred,
            dropout=args.dropout if args.classifier_dropout is None else args.classifier_dropout,
        )

    def forward(self, *tokens, observed_mask, modality_comb, return_aux=False):
        if (modality_comb < 0).any() or (modality_comb >= self.missing_bank.shape[0]).any():
            raise ValueError("Invalid modality-combination index for Flex-MoE missing bank")
        filled = []
        for modality_idx, token in enumerate(tokens):
            replacement = self.missing_bank[modality_comb, modality_idx]
            filled.append(torch.where(observed_mask[:, modality_idx, None, None], token, replacement))
        pooled = self.backbone(*filled, expert_indices=modality_comb)
        logits = self.classifier(torch.cat(pooled, dim=1))
        result = {"logits": logits}
        if return_aux:
            gate_loss = self.backbone.gate_loss()
            if not torch.is_tensor(gate_loss):
                gate_loss = logits.new_tensor(float(gate_loss))
            result["aux_loss"] = gate_loss
        return result


def build_model(args, num_modalities: int, num_classes: int, full_modality_index: int) -> nn.Module:
    missing_family_normalized_router = bool(
        getattr(args, "missing_family_normalized_router", False)
    )
    missing_family_residual_gate = bool(
        getattr(args, "missing_family_residual_gate", False)
    )
    if missing_family_residual_gate and not missing_family_normalized_router:
        raise ValueError(
            "missing_family_residual_gate requires "
            "missing_family_normalized_router"
        )
    if missing_family_normalized_router:
        if args.model != "our_moe":
            raise ValueError(
                "missing_family_normalized_router is only supported for "
                "our_moe"
            )
        incompatible = {
            "dynamic_branch_fusion": bool(
                getattr(args, "dynamic_branch_fusion", False)
            ),
            "positive supervised_router_loss_weight": float(
                getattr(args, "supervised_router_loss_weight", 0.0)
            ) > 0.0,
            "class_conditional_fusion": bool(
                getattr(args, "class_conditional_fusion", False)
            ),
            "joint_head_ensemble_size greater than one": int(
                getattr(args, "joint_head_ensemble_size", 1)
            ) != 1,
        }
        active_incompatibilities = [
            name for name, active in incompatible.items() if active
        ]
        if active_incompatibilities:
            raise ValueError(
                "missing-family routing v1 is incompatible with: "
                + ", ".join(active_incompatibilities)
            )
    common = dict(
        num_modalities=num_modalities,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    if args.model == "flex_moe":
        return FlexMoEBaseline(args, num_modalities, num_classes, full_modality_index)
    if args.model == "i2moe":
        model_class = external_class("MoE/i2moe_module.py", "I2MoEOfficialAdapter")
        return model_class(
            num_modalities=num_modalities,
            num_patches=args.num_patches,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers_fus=args.num_layers_fus,
            num_layers_pred=args.num_layers_pred,
            num_experts=args.num_experts,
            num_routers=args.num_routers,
            top_k=args.top_k,
            num_heads=args.num_heads,
            dropout=args.dropout,
            hidden_dim_rw=args.hidden_dim_rw,
            num_layer_rw=args.num_layer_rw,
            temperature_rw=args.temperature_rw,
        )
    if args.model == "moepp":
        model_class = external_class(
            "MoE/moepp_module.py", "MoEPlusPlusOfficialAdapter"
        )
        return model_class(
            num_modalities=num_modalities,
            num_patches=args.num_patches,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers_fus=args.num_layers_fus,
            num_experts=args.num_experts,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    if args.model == "moepp_corrected":
        model_class = external_class(
            "MoE/moepp_corrected_module.py", "MoEPlusPlusCorrectedAdapter"
        )
        return model_class(
            num_modalities=num_modalities,
            num_patches=args.num_patches,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers_fus=args.num_layers_fus,
            num_experts=args.num_experts,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    if args.model == "anymod":
        model_class = external_class("UnifiedAD/models.py", "UnifiedAnyModAD")
        return model_class(
            **common,
            num_query_tokens=args.num_query_tokens,
            num_task_tokens=args.num_task_tokens,
            num_proj_layers=args.num_proj_layers,
            num_fusion_layers=args.num_layers_fus,
        )
    if args.model == "mora":
        model_class = external_class("MoRA/models.py", "MoRAModel")
        return model_class(
            **common,
            num_layers=args.num_layers_fus,
            adapter_rank=args.adapter_rank,
            prompt_length=args.prompt_length,
        )
    if args.model == "agdic":
        model_class = external_class("AGDiC/models.py", "AGDiCModel")
        return model_class(**common, num_layers=args.num_layers_fus)
    if args.model == "agmd":
        model_class = external_class("AGMD/models.py", "AGMDModel")
        return model_class(**common, num_layers=args.num_layers_fus)
    if args.model == "acadiff":
        model_class = external_class("ACADiff/models.py", "ACADiffModel")
        return model_class(
            **common,
            num_layers=args.num_layers_fus,
            latent_dim=args.latent_dim,
            diffusion_steps=args.diffusion_steps,
            deterministic_eval=True,
            artificial_mask_probability=getattr(
                args, "acadiff_mask_probability", 0.25
            ),
        )
    if args.model == "transformer_concat":
        return ConcatTransformerBaseline(
            num_modalities=num_modalities,
            num_patches=args.num_patches,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers_fus,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_layers_pred=args.num_layers_pred,
        )
    if args.model == "our_moe":
        return AGMGFlexMoE(
            num_modalities=num_modalities,
            full_modality_index=full_modality_index,
            num_patches=args.num_patches,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers_fus=args.num_layers_fus,
            num_layers_pred=args.num_layers_pred,
            num_experts=args.num_experts,
            num_routers=args.num_routers,
            top_k=args.top_k,
            num_heads=args.num_heads,
            dropout=args.dropout,
            gen_num_layers=args.gen_num_layers,
            gen_num_heads=args.gen_num_heads,
            vectorized_generation=args.vectorized_generation,
            recon_targets_per_sample=args.recon_targets_per_sample,
            pattern_aware_reconstruction=getattr(
                args, "pattern_aware_reconstruction", False
            ),
            recon_normalized_token_loss_weight=getattr(
                args, "recon_normalized_token_loss_weight", 0.0
            ),
            recon_context_dropout_probability=getattr(
                args, "recon_context_dropout_probability", 0.0
            ),
            recon_encoder_gradient_scale=getattr(
                args, "recon_encoder_gradient_scale", 1.0
            ),
            use_generators=args.use_generators,
            dynamic_branch_fusion=args.dynamic_branch_fusion,
            dynamic_branch_joint_prior_boost=getattr(
                args, "dynamic_branch_joint_prior_boost", True
            ),
            dynamic_branch_use_observed_mask=getattr(
                args, "dynamic_branch_use_observed_mask", False
            ),
            prediction_observed_specialists_only=getattr(
                args, "prediction_observed_specialists_only", False
            ),
            supervised_router_observed_specialists_only=getattr(
                args, "supervised_router_observed_specialists_only", False
            ),
            complete_joint_only=args.complete_joint_only,
            complete_specialist_weight=args.complete_specialist_weight,
            missing_family_normalized_router=getattr(
                args, "missing_family_normalized_router", False
            ),
            missing_family_residual_gate=getattr(
                args, "missing_family_residual_gate", False
            ),
            branch_confidence_mode=args.branch_confidence_mode,
            token_attention_init=args.token_attention_init,
            generator_task_grad=args.generator_task_grad,
            generator_only_task_grad=getattr(
                args, "generator_only_task_grad", False
            ),
            generator_output_gate=getattr(args, "generator_output_gate", True),
            dense_backbone=getattr(args, "dense_backbone", False),
            disable_provenance=getattr(args, "disable_provenance", False),
            uniform_branch_weights=getattr(args, "uniform_branch_weights", False),
            joint_branch_only=getattr(args, "joint_branch_only", False),
            mean_pooling_only=getattr(args, "mean_pooling_only", False),
            disable_stochastic_context_masking=getattr(args, "disable_stochastic_context_masking", False),
            disable_completion=getattr(args, "disable_completion", False),
            no_output_gate=getattr(args, "no_output_gate", False),
            standard_transformer_residual=args.standard_transformer_residual,
            gated_transformer_residual=getattr(
                args, "gated_transformer_residual", False
            ),
            ordinal_fusion_weight=args.ordinal_fusion_weight,
            ordinal_aux_loss_weight=getattr(args, "ordinal_aux_loss_weight", 0.0),
            ordinal_head_type=getattr(args, "ordinal_head_type", "proportional"),
            uncertainty_aware_ordinal_fusion=getattr(
                args, "uncertainty_aware_ordinal_fusion", False
            ),
            enable_class1_aux_head=(
                getattr(args, "class1_aux_loss_weight", 0.0) > 0
            ),
            learn_observed_reliability=getattr(
                args, "learn_observed_reliability", False
            ),
            centered_evidence_confidence=getattr(
                args, "centered_evidence_confidence", False
            ),
            class_conditional_fusion=getattr(
                args, "class_conditional_fusion", False
            ),
            normalized_gate_loss=getattr(args, "normalized_gate_loss", False),
            enable_supervised_contrastive=(
                getattr(args, "supervised_contrastive_loss_weight", 0.0) > 0
            ),
            supervised_contrastive_projection_dim=getattr(
                args, "supervised_contrastive_projection_dim", 64
            ),
            joint_head_ensemble_size=getattr(
                args, "joint_head_ensemble_size", 1
            ),
            more_tail_rank=getattr(args, "more_tail_rank", 0),
            dual_local_boundary_loss_weight=getattr(
                args, "dual_local_boundary_loss_weight", 0.0
            ),
            presentation_axis_loss_weight=getattr(
                args, "presentation_axis_loss_weight", 0.0
            ),
            presentation_axis_residual_cap=getattr(
                args, "presentation_axis_residual_cap", 0.5
            ),
            missing_capacity_residual_width=getattr(
                args, "missing_capacity_residual_width", 0
            ),
        )
    raise ValueError(args.model)


def encode_batch(args, batch_samples, batch_observed, encoders, modality_dict, device):
    tokens = []
    masks = []
    for modality, samples in batch_samples.items():
        samples = samples.to(device, non_blocking=True)
        observed = batch_observed[:, modality_dict[modality]].bool()
        encoded = torch.zeros(
            samples.shape[0], args.num_patches, args.hidden_dim, device=device, dtype=torch.float32
        )
        if observed.any():
            encoded[observed] = encoders[modality](samples[observed])
        tokens.append(encoded)
        masks.append(observed)
    return tokens, torch.stack(masks, dim=1)


def objective(args, output, labels, criterion):
    ce = criterion(output["logits"], labels)
    if args.model == "i2moe":
        # Official objective: task loss + lambda * mean of the M+2
        # uniqueness/synergy/redundancy interaction losses.
        auxiliary = args.interaction_loss_weight * output["interaction_loss"]
    elif args.model in {"moepp", "moepp_corrected"}:
        # The official MoE++ baseline optimizes task loss only.  Its residual
        # router does not define a load-balancing auxiliary objective.
        auxiliary = ce.new_zeros(())
    elif args.model == "anymod":
        anchor = criterion(output["anchor_logits"], labels)
        auxiliary = args.align_loss_weight * output["align_loss"] + args.anchor_loss_weight * anchor
    else:
        raw_aux = output.get("aux_loss", ce.new_zeros(()))
        weight = args.gate_loss_weight if args.model == "flex_moe" else args.aux_loss_weight
        auxiliary = weight * raw_aux
    return ce + auxiliary, ce, auxiliary


def reconstruction_encoder_gradient_protocol(args) -> dict[str, Any]:
    """Describe the value-preserving reconstruction context gradient path."""
    scale = float(getattr(args, "recon_encoder_gradient_scale", 1.0))
    return {
        "schema": "reconstruction-encoder-gradient-v1",
        "gradient_scale": scale,
        "historical_path": scale == 1.0,
        "forward_value_change": False,
        "new_model_parameters_or_buffers": False,
        "rng_change": False,
        "scaled_path": "reconstruction context tokens entering generators only",
        "target_path_change": False,
        "generator_parameter_gradient_preserved": True,
        "reconstruction_projector_gradient_preserved": True,
        "classification_generator_gradient_control": (
            "generator_only_task_grad"
            if bool(getattr(args, "generator_only_task_grad", False))
            else "generator_task_grad"
        ),
    }


def classification_generator_gradient_protocol(args) -> dict[str, Any]:
    """Describe the classification-to-generator backward path exactly."""

    end_to_end = bool(getattr(args, "generator_task_grad", False))
    generator_only = bool(
        getattr(args, "generator_only_task_grad", False)
    )
    if end_to_end and generator_only:
        raise ValueError(
            "generator_task_grad and generator_only_task_grad are mutually "
            "exclusive"
        )
    if generator_only and not bool(getattr(args, "use_generators", True)):
        raise ValueError(
            "generator_only_task_grad requires use_generators=True"
        )
    enabled = end_to_end or generator_only
    mode = (
        "end_to_end"
        if end_to_end
        else "generator_only"
        if generator_only
        else "detached"
    )
    return {
        "schema": "classification-generator-gradient-v1",
        "mode": mode,
        "generated_output_attached": enabled,
        "generator_context_detached": generator_only,
        "generator_parameter_task_gradient": enabled,
        "context_encoder_task_gradient_via_generator": end_to_end,
        "clean_complete_view_generator_task_gradient": False,
        "forward_value_change": False,
        "new_model_parameters_or_buffers": False,
        "rng_change": False,
    }


def acadiff_objective_protocol(args) -> dict[str, Any]:
    """Auditable description of the corrected observed-only supervision path."""
    return {
        "implementation_revision": "masked-observed-denoising-v1",
        "task_loss": "cross_entropy",
        "auxiliary_weight": args.aux_loss_weight,
        "internal_auxiliary_weights": {
            "denoising": 1.0,
            "observed_reconstruction": 0.1,
            "observed_kl": 0.001,
        },
        "synthetic_target_probability": args.acadiff_mask_probability,
        "target_source": "pre-mask latent of naturally observed modalities only",
        "natural_missing_as_target": False,
        "eligible_samples": "at least two naturally observed modalities",
        "conditioning_rule": "natural_observed AND NOT synthetic_target",
        "mask_repair": (
            "for eligible samples, force one target when none are sampled and "
            "restore one conditioning modality when all are sampled"
        ),
        "classification_mask": "natural observed mask",
        "evaluation_artificial_masking": False,
        "vae_reconstruction_scope": "naturally observed modalities only",
        "deterministic_evaluation": True,
    }


def adni_image_imputation_protocol(args) -> dict[str, Any]:
    """Auditable definition of the optional ADNI image preprocessing fix."""

    strategy = getattr(args, "adni_image_imputation", "legacy_mode")
    initial_filling = getattr(args, "initial_filling", "mean")
    active = (
        getattr(args, "data", None) == "adni"
        and bool(getattr(args, "preprocessed", True))
        and "I" in str(getattr(args, "modality", "")).upper()
    )
    if not active:
        resolved_statistic = "not_applicable"
    elif strategy == "legacy_mode":
        resolved_statistic = "mode" if initial_filling == "mean" else "median"
    else:
        resolved_statistic = strategy
    return {
        "implementation_revision": "train-only-continuous-v1",
        "scope": "preprocessed ADNI FreeSurfer image features",
        "active": active,
        "configured_strategy": strategy,
        "resolved_statistic": resolved_statistic,
        "statistic_source": "training split only",
        "validation_or_test_statistics": False,
        "all_nan_column_fallback": 0.0,
    }


EMPIRICAL_MISSING_PATTERN_REPLAY_REVISION = (
    "abcd_cardinality_matched_train_patterns_v1"
)
EMPIRICAL_MISSING_PATTERN_REPLAY_DERIVED_ARGS = frozenset(
    {
        "empirical_missing_pattern_replay_pattern_counts_json",
        "empirical_missing_pattern_replay_train_id_mask_binding_sha256",
        "empirical_missing_pattern_replay_modality_order_json",
        "empirical_missing_pattern_replay_rng_seed",
    }
)
EMPIRICAL_MISSING_PATTERN_REPLAY_DROPPED_VIEW_OBJECTIVE_WEIGHTS = (
    ("drop_ce", "drop_ce_loss_weight"),
    ("distillation", "distill_loss_weight"),
    ("supervised_contrastive", "supervised_contrastive_loss_weight"),
    ("more_fewer_rank", "more_fewer_rank_loss_weight"),
)
EMPIRICAL_MISSING_PATTERN_REPLAY_ACTIVATION_RULE = (
    "drop_ce_loss_weight, distill_loss_weight, "
    "supervised_contrastive_loss_weight, and more_fewer_rank_loss_weight "
    "must each be strict numeric (non-bool), finite, and non-negative; "
    "at least one must be strictly positive"
)


def empirical_missing_pattern_replay_active_objectives(
    args: Any,
) -> dict[str, float]:
    """Validate and return the active objectives that consume the dropped view."""

    active: dict[str, float] = {}
    for objective_name, field_name in (
        EMPIRICAL_MISSING_PATTERN_REPLAY_DROPPED_VIEW_OBJECTIVE_WEIGHTS
    ):
        value = getattr(args, field_name, 0.0)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                "Enabled empirical replay requires every dropped-view "
                "objective weight to be strict numeric (non-bool), finite, "
                f"and non-negative; invalid {field_name}={value!r}"
            )
        if float(value) > 0.0:
            active[objective_name] = float(value)
    if not active:
        raise ValueError(
            "Enabled empirical replay requires at least one strictly positive "
            "dropped-view objective weight among drop_ce_loss_weight, "
            "distill_loss_weight, supervised_contrastive_loss_weight, and "
            "more_fewer_rank_loss_weight"
        )
    return active


def _empirical_pattern_text(row: Sequence[Any]) -> str:
    pattern = "".join("1" if bool(value) else "0" for value in row)
    if not pattern or "1" not in pattern:
        raise ValueError("Empirical replay patterns must retain at least one modality")
    return pattern


def empirical_missing_pattern_training_observed(
    observed: Any,
    train_ids: Sequence[int],
    data_dict: Mapping[str, Any],
    modality_dict: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Select final training masks in the exact order consumed by ``encode_batch``."""

    observed_array = np.asarray(observed)
    if observed_array.ndim != 2:
        raise ValueError("Observed-modality data must be a two-dimensional array")
    training_ids = _strict_integer_ids(train_ids, "empirical replay training ids")
    if training_ids.size == 0:
        raise ValueError("Empirical replay requires a non-empty training split")
    if bool(((training_ids < 0) | (training_ids >= observed_array.shape[0])).any()):
        raise ValueError("Empirical replay training ids are outside observed data")
    modality_order = tuple(
        name for name in data_dict if name != "modality_comb"
    )
    if not modality_order:
        raise ValueError("Empirical replay requires at least one modality")
    if set(modality_order) != set(modality_dict):
        raise ValueError(
            "Empirical replay data modalities do not match the modality mapping"
        )
    indexed_modalities: list[tuple[int, str]] = []
    for name, index in modality_dict.items():
        if type(index) is not int or isinstance(index, bool):
            raise ValueError(
                "Empirical replay modality indices must be strict integers"
            )
        indexed_modalities.append((index, name))
    indexed_modalities.sort()
    if [index for index, _name in indexed_modalities] != list(
        range(len(modality_order))
    ):
        raise ValueError(
            "Empirical replay modality indices must be contiguous from zero"
        )
    positional_order = tuple(name for _index, name in indexed_modalities)
    if modality_order != positional_order:
        raise ValueError(
            "Empirical replay data modality order must exactly match modality "
            "mapping index order"
        )
    column_indices = [modality_dict[name] for name in modality_order]
    if (
        len(set(column_indices)) != len(column_indices)
        or min(column_indices) < 0
        or max(column_indices) >= observed_array.shape[1]
    ):
        raise ValueError("Empirical replay modality columns are invalid")
    training_observed = np.stack(
        [observed_array[training_ids, column] for column in column_indices],
        axis=1,
    )
    if not np.isin(training_observed, (False, True, 0, 1)).all():
        raise ValueError("Empirical replay observed masks must be boolean")
    training_observed = training_observed.astype(bool, copy=False)
    if bool((~training_observed.any(axis=1)).any()):
        raise ValueError("Empirical replay training masks cannot be empty")
    return training_ids, training_observed, modality_order


def empirical_missing_pattern_replay_rng_seed(base_seed: int) -> int:
    """Domain-separate the independent replay stream from every global RNG."""

    if type(base_seed) is not int:
        raise TypeError("Empirical replay base seed must be an integer")
    encoded = (
        f"{EMPIRICAL_MISSING_PATTERN_REPLAY_REVISION}:seed:{base_seed}"
    ).encode("utf-8")
    # Keep the value in the signed 63-bit range accepted by every supported
    # torch.Generator implementation and safe in JSON/checkpoint metadata.
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (
        2**63 - 1
    )


def _validated_empirical_pattern_counts(
    pattern_counts: Mapping[str, Any],
) -> tuple[dict[str, int], int]:
    if not isinstance(pattern_counts, Mapping) or not pattern_counts:
        raise ValueError("Empirical replay pattern counts must be a non-empty mapping")
    keys = list(pattern_counts)
    if not all(type(key) is str for key in keys):
        raise ValueError("Empirical replay pattern keys must be strings")
    num_modalities = len(keys[0])
    if num_modalities < 1:
        raise ValueError("Empirical replay patterns cannot be empty")
    expected_patterns = {
        format(encoded, f"0{num_modalities}b")
        for encoded in range(1, 2**num_modalities)
    }
    if set(keys) != expected_patterns:
        raise ValueError(
            "Empirical replay counts must contain every non-empty pattern exactly once"
        )
    counts: dict[str, int] = {}
    for pattern in sorted(expected_patterns):
        value = pattern_counts[pattern]
        if type(value) is not int or value < 0:
            raise ValueError(
                "Empirical replay pattern counts must be non-negative integers"
            )
        counts[pattern] = value
    if sum(counts.values()) <= 0:
        raise ValueError("Empirical replay pattern counts have no positive support")
    return counts, num_modalities


def _empirical_replay_eligible_patterns(
    pattern_counts: Mapping[str, int],
    observed_pattern: str,
    cardinality: int,
) -> list[tuple[str, int]]:
    return [
        (pattern, count)
        for pattern, count in sorted(pattern_counts.items())
        if count > 0
        and pattern.count("1") == cardinality
        and all(
            target_bit == "0" or observed_bit == "1"
            for target_bit, observed_bit in zip(pattern, observed_pattern)
        )
    ]


def empirical_missing_pattern_replay_preflight(
    pattern_counts: Mapping[str, Any],
    drop_probability: float,
) -> None:
    """Prove positive empirical support for every mathematically possible trigger."""

    counts, _ = _validated_empirical_pattern_counts(pattern_counts)
    if not math.isfinite(float(drop_probability)) or not (
        0.0 < float(drop_probability) <= 1.0
    ):
        raise ValueError("Empirical replay dropout probability must be in (0, 1]")
    missing_support: list[str] = []
    for observed_pattern, observed_count in counts.items():
        if observed_count <= 0:
            continue
        observed_cardinality = observed_pattern.count("1")
        if observed_cardinality <= 1:
            # Legacy dropout always restores the sole modality, hence L == O.
            continue
        possible_cardinalities = (
            (1,)
            if float(drop_probability) == 1.0
            else tuple(range(1, observed_cardinality))
        )
        for cardinality in possible_cardinalities:
            if not _empirical_replay_eligible_patterns(
                counts, observed_pattern, cardinality
            ):
                missing_support.append(f"({observed_pattern},{cardinality})")
    if missing_support:
        raise ValueError(
            "Empirical replay has no positive-count eligible training pattern "
            "for reachable (O,k): " + ", ".join(missing_support)
        )


def empirical_missing_pattern_replay_training_audit(
    training_ids: Sequence[int],
    training_observed: Any,
    modality_order: Sequence[str],
    drop_probability: float,
    base_seed: int,
) -> dict[str, Any]:
    """Build the complete train-only replay audit without validation/test input."""

    ids = _strict_integer_ids(training_ids, "empirical replay training ids")
    masks = np.asarray(training_observed)
    order = tuple(str(name) for name in modality_order)
    if masks.ndim != 2 or masks.shape[0] != len(ids):
        raise ValueError("Empirical replay training IDs and masks must align")
    if masks.shape[1] != len(order) or not order or len(set(order)) != len(order):
        raise ValueError("Empirical replay modality order does not match masks")
    if not np.isin(masks, (False, True, 0, 1)).all():
        raise ValueError("Empirical replay training masks must be boolean")
    masks = masks.astype(bool, copy=False)
    if bool((~masks.any(axis=1)).any()):
        raise ValueError("Empirical replay training masks cannot be empty")
    pattern_counts = {
        format(encoded, f"0{masks.shape[1]}b"): 0
        for encoded in range(1, 2 ** masks.shape[1])
    }
    ordered_patterns: list[str] = []
    for row in masks:
        pattern = _empirical_pattern_text(row.tolist())
        pattern_counts[pattern] += 1
        ordered_patterns.append(pattern)
    empirical_missing_pattern_replay_preflight(
        pattern_counts, drop_probability
    )
    binding_payload = {
        "implementation_revision": "ordered_training_source_index_mask_v1",
        "modality_order": list(order),
        "rows": [
            [int(source_index), pattern]
            for source_index, pattern in zip(ids.tolist(), ordered_patterns)
        ],
    }
    return {
        "implementation_revision": EMPIRICAL_MISSING_PATTERN_REPLAY_REVISION,
        "modality_order": list(order),
        "pattern_counts": pattern_counts,
        "training_row_count": int(len(ids)),
        "train_id_mask_binding_sha256": _canonical_json_sha256(binding_payload),
        "rng_seed": empirical_missing_pattern_replay_rng_seed(base_seed),
    }


def materialize_empirical_missing_pattern_replay_args(
    args,
    observed: Any,
    train_ids: Sequence[int],
    data_dict: Mapping[str, Any],
    modality_dict: Mapping[str, int],
) -> dict[str, Any] | None:
    """Materialize enabled train-only audit fields into checkpoint arguments."""

    if not bool(getattr(args, "empirical_missing_pattern_replay", False)):
        return None
    empirical_missing_pattern_replay_active_objectives(args)
    training_ids, training_observed, modality_order = (
        empirical_missing_pattern_training_observed(
            observed, train_ids, data_dict, modality_dict
        )
    )
    audit = empirical_missing_pattern_replay_training_audit(
        training_ids,
        training_observed,
        modality_order,
        args.modality_dropout_prob,
        int(args.seed),
    )
    args.empirical_missing_pattern_replay_pattern_counts_json = json.dumps(
        audit["pattern_counts"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    args.empirical_missing_pattern_replay_train_id_mask_binding_sha256 = audit[
        "train_id_mask_binding_sha256"
    ]
    args.empirical_missing_pattern_replay_modality_order_json = json.dumps(
        audit["modality_order"],
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    args.empirical_missing_pattern_replay_rng_seed = audit["rng_seed"]
    return audit


def _empirical_missing_pattern_replay_args_audit(args) -> dict[str, Any]:
    missing = [
        name
        for name in EMPIRICAL_MISSING_PATTERN_REPLAY_DERIVED_ARGS
        if not hasattr(args, name)
    ]
    if missing:
        raise ValueError(
            "Enabled empirical replay is missing derived audit arguments: "
            f"{sorted(missing)}"
        )
    try:
        pattern_counts = json.loads(
            args.empirical_missing_pattern_replay_pattern_counts_json
        )
        modality_order = json.loads(
            args.empirical_missing_pattern_replay_modality_order_json
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Empirical replay audit JSON is invalid") from error
    counts, num_modalities = _validated_empirical_pattern_counts(pattern_counts)
    if (
        type(modality_order) is not list
        or len(modality_order) != num_modalities
        or not all(type(name) is str and name for name in modality_order)
        or len(set(modality_order)) != len(modality_order)
    ):
        raise ValueError("Empirical replay saved modality order is invalid")
    binding_sha256 = (
        args.empirical_missing_pattern_replay_train_id_mask_binding_sha256
    )
    if (
        type(binding_sha256) is not str
        or len(binding_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in binding_sha256
        )
    ):
        raise ValueError(
            "Empirical replay training ID-mask binding must be lowercase "
            "SHA-256 hex"
        )
    rng_seed = args.empirical_missing_pattern_replay_rng_seed
    if type(rng_seed) is not int or not 0 <= rng_seed < 2**63:
        raise ValueError("Empirical replay RNG seed must be a signed 63-bit integer")
    empirical_missing_pattern_replay_preflight(
        counts, args.modality_dropout_prob
    )
    return {
        "implementation_revision": EMPIRICAL_MISSING_PATTERN_REPLAY_REVISION,
        "modality_order": modality_order,
        "pattern_counts": counts,
        "training_row_count": int(sum(counts.values())),
        "train_id_mask_binding_sha256": binding_sha256,
        "rng_seed": rng_seed,
    }


def empirical_missing_pattern_replay_protocol(args) -> dict[str, Any]:
    enabled = bool(getattr(args, "empirical_missing_pattern_replay", False))
    active_objectives = (
        empirical_missing_pattern_replay_active_objectives(args)
        if enabled
        else {}
    )
    audit = _empirical_missing_pattern_replay_args_audit(args) if enabled else None
    return {
        "implementation_revision": EMPIRICAL_MISSING_PATTERN_REPLAY_REVISION,
        "enabled": enabled,
        "scope": "ABCD our_moe training artificial-drop view only",
        "statistics_source": "current job final observed masks at train_ids only",
        "validation_or_test_statistics": False,
        "legacy_shadow_rule": (
            "first call modality_dropout_mask(O,p) with the historical global RNG"
        ),
        "trigger_rule": "return O when L == O; otherwise replay at k=|L|",
        "eligibility_rule": "positive-count training pattern T subset O with |T|=k",
        "sampling_rule": "raw-count categorical; no smoothing or fallback",
        "preflight_rule": (
            "fail before training for every reachable positive-support (O,k) "
            "without an eligible T"
        ),
        "activation_rule": EMPIRICAL_MISSING_PATTERN_REPLAY_ACTIVATION_RULE,
        "active_objective_names": list(active_objectives),
        "active_objective_weights": active_objectives,
        "recompute_dropped_combination_required": enabled,
        "independent_rng": enabled,
        "independent_rng_draws": (
            "exactly one CPU categorical uniform per triggered row"
            if enabled
            else None
        ),
        "global_rng_change_beyond_legacy_shadow": False,
        "sam_supported": False,
        "rdrop_supported": False,
        "new_model_parameters_or_buffers": False,
        "inference_change": False,
        "modality_order": None if audit is None else audit["modality_order"],
        "pattern_counts": None if audit is None else audit["pattern_counts"],
        "training_row_count": (
            None if audit is None else audit["training_row_count"]
        ),
        "train_id_mask_binding_sha256": (
            None if audit is None else audit["train_id_mask_binding_sha256"]
        ),
        "rng_seed": None if audit is None else audit["rng_seed"],
    }


class EmpiricalMissingPatternReplayState:
    """Persistent independent categorical stream for one complete training job."""

    def __init__(self, pattern_counts: Mapping[str, Any], rng_seed: int):
        counts, num_modalities = _validated_empirical_pattern_counts(pattern_counts)
        if type(rng_seed) is not int or not 0 <= rng_seed < 2**63:
            raise ValueError(
                "Empirical replay RNG seed must be a signed 63-bit integer"
            )
        self.pattern_counts = counts
        self.num_modalities = num_modalities
        self.rng_seed = rng_seed
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(rng_seed)

    @classmethod
    def from_args(cls, args) -> "EmpiricalMissingPatternReplayState":
        audit = _empirical_missing_pattern_replay_args_audit(args)
        return cls(audit["pattern_counts"], audit["rng_seed"])


def empirical_missing_pattern_replay_mask(
    observed_mask: torch.Tensor,
    legacy_dropped_mask: torch.Tensor,
    state: EmpiricalMissingPatternReplayState,
) -> torch.Tensor:
    """Replace triggered legacy masks without drawing from a global RNG."""

    if not isinstance(state, EmpiricalMissingPatternReplayState):
        raise TypeError("Enabled empirical replay requires its explicit state")
    if (
        observed_mask.ndim != 2
        or legacy_dropped_mask.shape != observed_mask.shape
        or observed_mask.shape[1] != state.num_modalities
    ):
        raise ValueError("Empirical replay observed/legacy mask shapes are invalid")
    observed = observed_mask.bool()
    legacy = legacy_dropped_mask.bool()
    if bool((~observed.any(dim=1)).any()) or bool((~legacy.any(dim=1)).any()):
        raise ValueError("Empirical replay masks must retain at least one modality")
    if bool((legacy & ~observed).any()):
        raise ValueError("Legacy modality dropout cannot unmask a modality")
    replayed = observed.clone()
    for row_index in range(observed.shape[0]):
        if torch.equal(legacy[row_index], observed[row_index]):
            continue
        observed_pattern = _empirical_pattern_text(
            observed[row_index].detach().cpu().tolist()
        )
        cardinality = int(legacy[row_index].sum().item())
        if not 1 <= cardinality < observed_pattern.count("1"):
            raise ValueError(
                "Triggered empirical replay must strictly reduce cardinality"
            )
        eligible = _empirical_replay_eligible_patterns(
            state.pattern_counts, observed_pattern, cardinality
        )
        if not eligible:
            raise RuntimeError(
                "Empirical replay reached an (O,k) that failed preflight: "
                f"({observed_pattern},{cardinality})"
            )
        # One and only one independent uniform is consumed for every triggered
        # row, including the single-eligible-pattern case.
        uniform = float(torch.rand((), generator=state.generator).item())
        total_count = sum(count for _, count in eligible)
        threshold = uniform * total_count
        cumulative = 0
        selected_pattern = eligible[-1][0]
        for pattern, count in eligible:
            cumulative += count
            if threshold < cumulative:
                selected_pattern = pattern
                break
        replayed[row_index] = torch.tensor(
            [bit == "1" for bit in selected_pattern],
            dtype=torch.bool,
            device=observed.device,
        )
    if bool((replayed & ~observed).any()):
        raise RuntimeError("Empirical replay unexpectedly unmasked a modality")
    triggered = (legacy != observed).any(dim=1)
    if bool(
        (
            replayed[triggered].sum(dim=1)
            != legacy[triggered].sum(dim=1)
        ).any()
    ):
        raise RuntimeError("Empirical replay changed legacy retained cardinality")
    return replayed


def modality_dropout_mask(observed_mask: torch.Tensor, drop_probability: float) -> torch.Tensor:
    if drop_probability <= 0:
        return observed_mask
    augmented = observed_mask.clone()
    dropped = (torch.rand_like(augmented.float()) < drop_probability) & augmented
    augmented &= ~dropped
    empty_rows = ~augmented.any(dim=1)
    for row in empty_rows.nonzero(as_tuple=False).view(-1).tolist():
        available = observed_mask[row].nonzero(as_tuple=False).view(-1)
        if available.numel() > 0:
            keep = available[torch.randint(available.numel(), (1,), device=observed_mask.device)]
            augmented[row, keep] = True
    return augmented


def combination_indices_from_mask(
    observed_mask: torch.Tensor,
    modality_codes: str,
    *,
    sort_codes_before_enumeration: bool,
) -> torch.Tensor:
    """Map masks to the dataset loader's exact modality-combination ordering."""
    num_modalities = observed_mask.shape[1]
    codes = list(dict.fromkeys(str(modality_codes).upper()))
    if len(codes) != num_modalities:
        raise ValueError(
            f"Expected {num_modalities} unique modality codes, got {modality_codes!r}"
        )
    enumeration_codes = sorted(codes) if sort_codes_before_enumeration else codes
    combination_map: dict[str, int] = {}
    combination_index = 0
    from itertools import combinations

    for size in range(num_modalities, 0, -1):
        for subset in combinations(enumeration_codes, size):
            combination_map["".join(sorted(subset))] = combination_index
            combination_index += 1

    lookup = torch.full(
        (2**num_modalities,),
        -1,
        dtype=torch.long,
        device=observed_mask.device,
    )
    for bitmask in range(1, 2**num_modalities):
        key = "".join(sorted(
            codes[modality_index]
            for modality_index in range(num_modalities)
            if bitmask & (1 << modality_index)
        ))
        lookup[bitmask] = combination_map[key]
    powers = 1 << torch.arange(num_modalities, device=observed_mask.device)
    encoded = (observed_mask.long() * powers.unsqueeze(0)).sum(dim=1)
    result = lookup[encoded]
    if (result < 0).any():
        raise ValueError("At least one modality must remain after modality dropout")
    return result


def class1_auxiliary_loss(
    class1_aux_logit: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Unweighted BCE for the exact middle class against classes 0 and 2."""
    if class1_aux_logit.ndim != 1:
        raise ValueError("class1_aux_logit must have shape (batch,)")
    if labels.ndim != 1 or labels.shape[0] != class1_aux_logit.shape[0]:
        raise ValueError("labels must match the class1 auxiliary batch")
    targets = (labels == 1).to(dtype=class1_aux_logit.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        class1_aux_logit,
        targets,
    )


def label_supervised_router_loss(
    branch_logits: torch.Tensor,
    supervision_log_scores: torch.Tensor,
    branch_mask: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Match routing scores to a detached soft oracle over available branches.

    For sample ``i`` and available branch ``b``, the teacher cost is the
    branch's true-class negative log likelihood.  A temperature-softmax over
    the negated, detached costs forms the target distribution; soft CE then
    trains the router scores.  Detaching the teacher prevents the branch heads
    from reducing this auxiliary term by changing the target itself.
    """
    if branch_logits.ndim != 3:
        raise ValueError("branch_logits must have shape (batch, branches, classes)")
    if supervision_log_scores.shape != branch_logits.shape[:2]:
        raise ValueError(
            "supervision_log_scores must match branch_logits batch/branches"
        )
    if branch_mask.shape != branch_logits.shape[:2]:
        raise ValueError("branch_mask must match branch_logits batch/branches")
    if labels.ndim != 1 or labels.shape[0] != branch_logits.shape[0]:
        raise ValueError("labels must match the branch-logit batch")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    branch_mask = branch_mask.bool()
    if not bool(branch_mask.any(dim=1).all()):
        raise ValueError("every sample needs at least one available branch")

    true_class = labels[:, None, None].expand(-1, branch_logits.shape[1], 1)
    per_branch_nll = -torch.log_softmax(branch_logits, dim=-1).gather(
        dim=-1,
        index=true_class,
    ).squeeze(-1)
    finite_mask_value = -1e4
    target_scores = (-per_branch_nll.detach() / temperature).masked_fill(
        ~branch_mask,
        finite_mask_value,
    )
    target = torch.softmax(target_scores, dim=1)
    router_log_probability = torch.log_softmax(
        supervision_log_scores.masked_fill(~branch_mask, finite_mask_value),
        dim=1,
    )
    return -(target * router_log_probability).sum(dim=1).mean()


def clean_dynamic_router_protocol(args) -> dict[str, Any]:
    """Return the auditable prediction/teacher masks for dynamic routing."""

    dynamic_enabled = bool(getattr(args, "dynamic_branch_fusion", False))
    joint_prior_boost = bool(
        getattr(args, "dynamic_branch_joint_prior_boost", True)
    )
    use_observed_mask = bool(
        getattr(args, "dynamic_branch_use_observed_mask", False)
    )
    observed_specialists_only = bool(
        getattr(
            args,
            "supervised_router_observed_specialists_only",
            False,
        )
    )
    quality_weighted_aux = bool(
        getattr(args, "dynamic_branch_quality_weighted_aux", False)
    )
    prediction_observed_specialists_only = bool(
        getattr(args, "prediction_observed_specialists_only", False)
    )
    clean_gate_options_enabled = bool(
        dynamic_enabled
        and (
            not joint_prior_boost
            or use_observed_mask
            or observed_specialists_only
        )
    )
    clean_options_enabled = bool(
        clean_gate_options_enabled
        or (dynamic_enabled and quality_weighted_aux)
    )
    return {
        "schema": "clean-dynamic-router-v1",
        "dynamic_branch_fusion": dynamic_enabled,
        "clean_options_enabled": clean_options_enabled,
        "joint_branch_log2_prior_boost": joint_prior_boost,
        "dynamic_gate_mask_source": (
            "raw observed_mask"
            if use_observed_mask
            else "generator-expanded usable_mask"
        ),
        "prediction_branch_mask_source": (
            "joint from generator-expanded usable_mask; unimodal/pairwise "
            "from raw observed_mask"
            if prediction_observed_specialists_only
            else "generator-expanded usable_mask"
        ),
        "supervised_router_branch_mask_source": (
            "joint plus fully observed unimodal/pair specialists"
            if observed_specialists_only
            else "prediction-available branches"
        ),
        "dynamic_gate_cpu_construction_rng_isolated": (
            clean_gate_options_enabled
        ),
        "training_mode_dropout_rng_equivalence_claimed": False,
        "quality_weighted_branch_aux_with_dynamic": bool(
            dynamic_enabled
            and quality_weighted_aux
            and not getattr(args, "balanced_branch_aux", False)
        ),
    }


def missing_family_router_protocol(args) -> dict[str, Any]:
    """Describe the separated missing-row family-routing factors."""

    normalized = bool(
        getattr(args, "missing_family_normalized_router", False)
    )
    residual = bool(getattr(args, "missing_family_residual_gate", False))
    return {
        "schema": "missing-family-router-v1",
        "normalized_router_enabled": normalized,
        "residual_gate_enabled": residual,
        "scope": "prediction rows with at least one raw missing modality only",
        "complete_row_path": "historical branch weights and probabilities unchanged",
        "families": ["joint", "unimodal", "pairwise"],
        "within_family_rule": "softmax over active branch log scores",
        "family_base_rule": (
            "logsumexp(active branch log scores) - log(active branch count)"
        ),
        "inactive_family_rule": "zero mass",
        "branch_count_prior_removed": normalized,
        "residual_gate": {
            "requires_normalized_router": True,
            "input": "concatenated raw observed mask and modality reliability",
            "output": "three family-log-score residuals",
            "output_layer_initialization": "exact zero weight and bias",
            "cpu_construction_rng_restored": residual,
            "parameters_constructed": residual,
        },
        "factorization": {
            "parameter_free_normalization_flag": (
                "missing_family_normalized_router"
            ),
            "learned_residual_flag": "missing_family_residual_gate",
            "gate_off_ablation_prebound": True,
        },
        "v1_fail_closed_incompatibilities": [
            "dynamic_branch_fusion",
            "positive supervised_router_loss_weight",
            "class_conditional_fusion",
            "joint_head_ensemble_size greater than one",
        ],
        "checkpoint_compatibility": {
            "both_flags_false": "old state dict strict-load compatible",
            "normalization_true_gate_false": (
                "state-dict compatible diagnostic but requires matched retraining "
                "for scientific ablation"
            ),
            "residual_gate_true": "adds parameters and requires a new checkpoint",
        },
    }


def class_balanced_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-or-more-view SupCon with class-weighted anchor normalization.

    ``embeddings`` has shape ``(batch, views, dimension)``.  Every anchor
    excludes itself from the denominator; every other view with the same label
    is a positive, so a sample's paired view guarantees a positive even when it
    is the only example of its class in the minibatch.  Class weights affect
    anchors only and are normalized by their sum, matching the main weighted
    CE reduction without changing positive membership.
    """
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape (batch, views, dimension)")
    batch_size, num_views, _ = embeddings.shape
    if batch_size < 1 or num_views < 2:
        raise ValueError("supervised contrastive loss requires at least two views")
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError("labels must match the contrastive batch")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    flattened = normalized.reshape(batch_size * num_views, -1)
    anchor_labels = labels.repeat_interleave(num_views)
    similarities = flattened @ flattened.transpose(0, 1)
    similarities = similarities / temperature
    # Subtracting a detached row maximum is an exact softmax stabilization and
    # cannot create a shortcut through the scale of an embedding.
    similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(
        batch_size * num_views,
        dtype=torch.bool,
        device=embeddings.device,
    )
    positive_mask = (
        anchor_labels[:, None] == anchor_labels[None, :]
    ) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    if not bool((positive_count > 0).all()):
        raise ValueError("every contrastive anchor must have at least one positive")
    log_denominator = torch.logsumexp(
        similarities.masked_fill(self_mask, float("-inf")),
        dim=1,
    )
    log_probability = similarities - log_denominator[:, None]
    per_anchor = -(
        log_probability * positive_mask.to(log_probability.dtype)
    ).sum(dim=1) / positive_count.to(log_probability.dtype)

    if class_weights is None:
        anchor_weights = torch.ones_like(per_anchor)
    else:
        if class_weights.ndim != 1:
            raise ValueError("class_weights must be one-dimensional")
        if labels.min() < 0 or labels.max() >= class_weights.shape[0]:
            raise ValueError("labels index outside class_weights")
        anchor_weights = class_weights.to(
            device=embeddings.device,
            dtype=per_anchor.dtype,
        )[anchor_labels]
    return (per_anchor * anchor_weights).sum() / anchor_weights.sum().clamp_min(1e-8)


def dual_boundary_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.20,
    class1_vs_0_weight: float = 2.0 / 3.0,
) -> torch.Tensor:
    """Rank the middle class above each adjacent class on its own boundary.

    For boundary ``1 vs c``, every class-1 sample is paired with every class-c
    sample in the minibatch using the deployed fused score ``logit_1-logit_c``.
    A sample-independent class-logit bias cancels from each pairwise gap, so
    this auxiliary cannot be reduced by merely shifting the class-1 threshold.
    If a boundary is absent from a minibatch, the remaining active boundary is
    normalized to full scale.  With no usable pair, a differentiable exact zero
    is returned so standalone backward calls produce zero rather than no grad.
    """
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("dual-boundary ranking logits must have shape (batch, 3)")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must match the dual-boundary ranking batch")
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("dual-boundary ranking margin must be finite and non-negative")
    class1_vs_0_weight = float(class1_vs_0_weight)
    if not math.isfinite(class1_vs_0_weight) or not 0.0 <= class1_vs_0_weight <= 1.0:
        raise ValueError("class1_vs_0_weight must be finite and in [0, 1]")
    if labels.numel() > 0 and bool(((labels < 0) | (labels > 2)).any()):
        raise ValueError("dual-boundary ranking labels must be in {0, 1, 2}")

    class1_mask = labels == 1
    active_terms: list[tuple[float, torch.Tensor]] = []
    if bool(class1_mask.any()):
        for negative_class, boundary_weight in (
            (0, class1_vs_0_weight),
            (2, 1.0 - class1_vs_0_weight),
        ):
            negative_mask = labels == negative_class
            if boundary_weight <= 0 or not bool(negative_mask.any()):
                continue
            boundary_scores = logits[:, 1] - logits[:, negative_class]
            pairwise_gaps = (
                boundary_scores[class1_mask, None]
                - boundary_scores[None, negative_mask]
            )
            boundary_loss = torch.nn.functional.softplus(
                margin - pairwise_gaps
            ).mean()
            active_terms.append((boundary_weight, boundary_loss))

    if not active_terms:
        return logits.sum() * 0.0
    if len(active_terms) == 1:
        # A missing adjacent class must not attenuate the remaining usable
        # boundary, and returning it directly avoids needless round-off.
        return active_terms[0][1]
    active_weight = sum(weight for weight, _ in active_terms)
    return torch.stack(
        [weight * boundary_loss for weight, boundary_loss in active_terms]
    ).sum() / active_weight


def dual_local_boundary_residual_loss(
    base_logits: torch.Tensor,
    residuals: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Balanced local logistic loss for the two DLBR corrections.

    ``residuals[:, 0]`` corrects the class-1-vs-0 gap and
    ``residuals[:, 1]`` corrects class-1-vs-2.  The corresponding base gap is
    detached, so this auxiliary cannot directly move the categorical logits;
    it trains the residual directions and their shared pooled features.  Each
    active binary boundary gives equal mass to its two classes.  If exactly one
    boundary is available in a minibatch it is renormalized to full scale.
    """

    if base_logits.ndim != 2 or base_logits.shape[1] != 3:
        raise ValueError("DLBR base_logits must have shape (batch, 3)")
    if residuals.ndim != 2 or residuals.shape != (base_logits.shape[0], 2):
        raise ValueError("DLBR residuals must have shape (batch, 2)")
    if labels.ndim != 1 or labels.shape[0] != base_logits.shape[0]:
        raise ValueError("labels must match the DLBR batch")
    if labels.numel() > 0 and bool(((labels < 0) | (labels > 2)).any()):
        raise ValueError("DLBR labels must be in {0, 1, 2}")

    class1_mask = labels == 1
    active_terms: list[tuple[float, torch.Tensor]] = []
    if bool(class1_mask.any()):
        for residual_index, negative_class, boundary_weight in (
            (0, 0, 2.0 / 3.0),
            (1, 2, 1.0 / 3.0),
        ):
            negative_mask = labels == negative_class
            if not bool(negative_mask.any()):
                continue
            base_gap = (
                base_logits[:, 1] - base_logits[:, negative_class]
            ).detach()
            corrected_gap = base_gap + residuals[:, residual_index]
            positive_loss = torch.nn.functional.softplus(
                -corrected_gap[class1_mask]
            ).mean()
            negative_loss = torch.nn.functional.softplus(
                corrected_gap[negative_mask]
            ).mean()
            active_terms.append(
                (boundary_weight, 0.5 * (positive_loss + negative_loss))
            )

    if not active_terms:
        return residuals.sum() * 0.0
    if len(active_terms) == 1:
        return active_terms[0][1]
    active_weight = sum(weight for weight, _ in active_terms)
    return torch.stack(
        [weight * boundary_loss for weight, boundary_loss in active_terms]
    ).sum() / active_weight


def presentation_axis_balanced_logistic_loss(
    raw_axis_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Balanced IA/HI logistic supervision for ADHD presentations.

    The three nominal presentation labels encode two non-exclusive symptom
    axes: inattentive (IA) is present for labels 0 and 2, while
    hyperactive/impulsive (HI) is present for labels 1 and 2.  Within each
    axis, positive and negative groups receive equal mass when both occur in a
    minibatch.  If one side is absent, the existing side's mean is retained at
    full scale.  The two axis losses are then averaged equally.
    """

    if raw_axis_logits.ndim != 2 or raw_axis_logits.shape[1] != 2:
        raise ValueError(
            "presentation-axis raw logits must have shape (batch, 2)"
        )
    if labels.ndim != 1 or labels.shape[0] != raw_axis_logits.shape[0]:
        raise ValueError("labels must match the presentation-axis batch")
    if labels.numel() > 0 and bool(((labels < 0) | (labels > 2)).any()):
        raise ValueError("presentation-axis labels must be in {0, 1, 2}")
    if labels.numel() == 0:
        return raw_axis_logits.sum() * 0.0

    axis_positive_masks = (
        (labels == 0) | (labels == 2),
        (labels == 1) | (labels == 2),
    )
    axis_losses = []
    for axis_index, positive_mask in enumerate(axis_positive_masks):
        negative_mask = ~positive_mask
        side_losses = []
        if bool(positive_mask.any()):
            side_losses.append(
                torch.nn.functional.softplus(
                    -raw_axis_logits[positive_mask, axis_index]
                ).mean()
            )
        if bool(negative_mask.any()):
            side_losses.append(
                torch.nn.functional.softplus(
                    raw_axis_logits[negative_mask, axis_index]
                ).mean()
            )
        if len(side_losses) == 2:
            axis_losses.append(0.5 * (side_losses[0] + side_losses[1]))
        elif len(side_losses) == 1:
            axis_losses.append(side_losses[0])
        else:  # unreachable for a non-empty, valid label batch
            axis_losses.append(raw_axis_logits[:, axis_index].sum() * 0.0)
    return 0.5 * (axis_losses[0] + axis_losses[1])


def hard_cvar_dual_boundary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tail_fraction: float = 0.25,
    margin: float = 0.20,
    class1_vs_0_weight: float = 0.65,
) -> torch.Tensor:
    """CVaR over the hardest pair losses on both class-1 boundaries.

    Each active boundary forms the same pairwise softplus loss as the legacy
    dual-boundary auxiliary, then retains exactly ``ceil(q * num_pairs)`` of
    the largest losses.  Taking the tail independently prevents a large easy
    boundary from diluting the hard pairs on the other boundary.
    """

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("hard-CVaR dual-boundary logits must have shape (batch, 3)")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must match the hard-CVaR dual-boundary batch")
    tail_fraction = float(tail_fraction)
    if not math.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be finite and in (0, 1]")
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("hard-CVaR margin must be finite and non-negative")
    class1_vs_0_weight = float(class1_vs_0_weight)
    if (
        not math.isfinite(class1_vs_0_weight)
        or not 0.0 <= class1_vs_0_weight <= 1.0
    ):
        raise ValueError("class1_vs_0_weight must be finite and in [0, 1]")
    if labels.numel() > 0 and bool(((labels < 0) | (labels > 2)).any()):
        raise ValueError("hard-CVaR dual-boundary labels must be in {0, 1, 2}")

    class1_mask = labels == 1
    active_terms: list[tuple[float, torch.Tensor]] = []
    if bool(class1_mask.any()):
        for negative_class, boundary_weight in (
            (0, class1_vs_0_weight),
            (2, 1.0 - class1_vs_0_weight),
        ):
            negative_mask = labels == negative_class
            if boundary_weight <= 0 or not bool(negative_mask.any()):
                continue
            boundary_scores = logits[:, 1] - logits[:, negative_class]
            pairwise_gaps = (
                boundary_scores[class1_mask, None]
                - boundary_scores[None, negative_mask]
            )
            pair_losses = torch.nn.functional.softplus(margin - pairwise_gaps)
            flattened = pair_losses.reshape(-1)
            tail_count = max(
                1,
                int(math.ceil(tail_fraction * flattened.numel())),
            )
            if tail_count == flattened.numel():
                boundary_loss = flattened.mean()
            else:
                boundary_loss = torch.topk(
                    flattened,
                    k=tail_count,
                    largest=True,
                    sorted=False,
                ).values.mean()
            active_terms.append((boundary_weight, boundary_loss))

    if not active_terms:
        return logits.sum() * 0.0
    if len(active_terms) == 1:
        return active_terms[0][1]
    active_weight = sum(weight for weight, _ in active_terms)
    return torch.stack(
        [weight * boundary_loss for weight, boundary_loss in active_terms]
    ).sum() / active_weight


def hard_cvar_dual_boundary_weight_for_epoch(
    epoch: int,
    max_weight: float,
    start_epoch: int = 5,
    ramp_epochs: int = 10,
) -> float:
    """One-indexed linear warm-in for the hard-CVaR auxiliary.

    ``start_epoch`` is the first non-zero epoch.  With the locked defaults,
    epoch 5 uses one tenth of the maximum and epoch 14 reaches the maximum.
    A zero-length ramp enables the full weight immediately at ``start_epoch``.
    """

    if type(epoch) is not int or epoch < 1:
        raise ValueError("hard-CVaR epoch must be a positive integer")
    max_weight = float(max_weight)
    if not math.isfinite(max_weight) or max_weight < 0:
        raise ValueError("hard-CVaR max_weight must be finite and non-negative")
    if type(start_epoch) is not int or start_epoch < 1:
        raise ValueError("hard-CVaR start_epoch must be a positive integer")
    if type(ramp_epochs) is not int or ramp_epochs < 0:
        raise ValueError("hard-CVaR ramp_epochs must be a non-negative integer")
    if max_weight == 0.0 or epoch < start_epoch:
        return 0.0
    if ramp_epochs == 0:
        return max_weight
    progress = min(1.0, (epoch - start_epoch + 1) / float(ramp_epochs))
    return max_weight * progress


def cross_fitted_tree_teacher_kl_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Mean per-sample KL from fixed OOF teacher probabilities to Ours.

    Teacher probabilities are temperature-adjusted by normalized power
    ``p ** (1 / T)``.  This preserves exact zero mass without taking a raw
    logarithm, while the usual ``T**2`` factor keeps gradient scale comparable.
    """

    if student_logits.ndim != 2:
        raise ValueError("student logits must have shape (batch, classes)")
    if teacher_probabilities.shape != student_logits.shape:
        raise ValueError(
            "teacher probabilities must have the same shape as student logits"
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("tree teacher temperature must be finite and positive")
    working_logits = (
        student_logits.float()
        if student_logits.dtype in {torch.float16, torch.bfloat16}
        else student_logits
    )
    teacher = teacher_probabilities.detach().to(
        device=working_logits.device, dtype=working_logits.dtype
    )
    if not bool(torch.isfinite(teacher).all()):
        raise ValueError("teacher probabilities must be finite")
    if bool((teacher < 0).any()) or bool((teacher > 1).any()):
        raise ValueError("teacher probabilities must lie in [0, 1]")
    row_sums = teacher.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("teacher probability rows must sum to one")
    tempered_teacher = teacher.pow(1.0 / temperature)
    tempered_teacher = tempered_teacher / tempered_teacher.sum(
        dim=1, keepdim=True
    ).clamp_min(torch.finfo(tempered_teacher.dtype).tiny)
    student_log_probability = torch.nn.functional.log_softmax(
        working_logits / temperature, dim=1
    )
    per_sample = torch.nn.functional.kl_div(
        student_log_probability,
        tempered_teacher,
        reduction="none",
    ).sum(dim=1)
    return per_sample.mean() * (temperature * temperature)


def cross_fitted_external_teacher_kl_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    trust_mode: str = "correct_only",
    class_compensation: str = "none",
    class_compensation_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Trust-gated, active-weight-normalized per-sample external-teacher KL."""

    if student_logits.ndim != 2:
        raise ValueError("student logits must have shape (batch, classes)")
    if teacher_probabilities.shape != student_logits.shape:
        raise ValueError(
            "external teacher probabilities must match student logits"
        )
    if labels.ndim != 1 or labels.shape[0] != student_logits.shape[0]:
        raise ValueError("external teacher labels must match the logits batch")
    if labels.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("external teacher labels must have integer dtype")
    if labels.numel() and bool(
        ((labels < 0) | (labels >= student_logits.shape[1])).any()
    ):
        raise ValueError("external teacher labels are outside class range")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "external teacher temperature must be finite and positive"
        )
    if trust_mode not in {"correct_only", "all"}:
        raise ValueError("external teacher trust_mode must be correct_only or all")
    if class_compensation not in {"none", "tcl_sqrt_error"}:
        raise ValueError(
            "external teacher class_compensation must be none or "
            "tcl_sqrt_error"
        )

    working_logits = (
        student_logits.float()
        if student_logits.dtype in {torch.float16, torch.bfloat16}
        else student_logits
    )
    teacher = teacher_probabilities.detach().to(
        device=working_logits.device, dtype=working_logits.dtype
    )
    working_labels = labels.detach().to(
        device=working_logits.device, dtype=torch.long
    )
    if not bool(torch.isfinite(teacher).all()):
        raise ValueError("external teacher probabilities must be finite")
    if bool((teacher < 0).any()) or bool((teacher > 1).any()):
        raise ValueError("external teacher probabilities must lie in [0, 1]")
    row_sums = teacher.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("external teacher probability rows must sum to one")

    if trust_mode == "correct_only":
        active = teacher.argmax(dim=1).eq(working_labels)
    else:
        active = torch.ones_like(working_labels, dtype=torch.bool)
    if class_compensation == "none":
        sample_weights = torch.ones_like(
            working_labels, dtype=working_logits.dtype
        )
    else:
        if class_compensation_weights is None:
            raise ValueError(
                "tcl_sqrt_error requires fixed full-OOF class weights"
            )
        class_weights_tensor = torch.as_tensor(
            class_compensation_weights,
            device=working_logits.device,
            dtype=working_logits.dtype,
        ).detach()
        if (
            class_weights_tensor.ndim != 1
            or class_weights_tensor.numel() != student_logits.shape[1]
        ):
            raise ValueError(
                "external teacher class weights must contain one value per class"
            )
        if not bool(torch.isfinite(class_weights_tensor).all()) or bool(
            (class_weights_tensor < 0).any()
        ):
            raise ValueError(
                "external teacher class weights must be finite and non-negative"
            )
        expected_sum = class_weights_tensor.new_tensor(
            float(student_logits.shape[1])
        )
        if not torch.allclose(
            class_weights_tensor.sum(), expected_sum, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "external teacher class weights must sum to the class count"
            )
        sample_weights = class_weights_tensor[working_labels]
    active_weights = sample_weights * active.to(sample_weights.dtype)
    if not bool((active_weights > 0).any()):
        # Keep an exact scalar zero connected to the student graph.
        return working_logits.sum() * 0.0

    tempered_teacher = teacher.pow(1.0 / temperature)
    tempered_teacher = tempered_teacher / tempered_teacher.sum(
        dim=1, keepdim=True
    ).clamp_min(torch.finfo(tempered_teacher.dtype).tiny)
    student_log_probability = torch.nn.functional.log_softmax(
        working_logits / temperature, dim=1
    )
    per_sample = torch.nn.functional.kl_div(
        student_log_probability,
        tempered_teacher,
        reduction="none",
    ).sum(dim=1) * (temperature * temperature)
    return (per_sample * active_weights).sum() / active_weights.sum()


class MaskedBranchTCLReplayContext(NamedTuple):
    """Detached control state frozen by the clean primary forward.

    SAM reuses this exact gate, availability mask, and through-epoch-e-1
    class compensation while recomputing teacher/student distributions from
    the perturbed replay logits.
    """

    correct_teacher_mask: torch.Tensor
    active_branch_mask: torch.Tensor
    sample_compensation: torch.Tensor


def _validate_masked_branch_tcl_inputs(
    branch_logits: torch.Tensor,
    branch_mask: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if branch_logits.ndim != 3:
        raise ValueError(
            "masked branch TCL logits must have shape (batch, branches, classes)"
        )
    if branch_logits.shape[2] < 2:
        raise ValueError("masked branch TCL requires at least two classes")
    if branch_mask.shape != branch_logits.shape[:2]:
        raise ValueError(
            "masked branch TCL availability mask must match batch and branches"
        )
    if branch_mask.dtype != torch.bool:
        raise ValueError("masked branch TCL availability mask must be boolean")
    if labels.ndim != 1 or labels.shape[0] != branch_logits.shape[0]:
        raise ValueError("masked branch TCL labels must match the logits batch")
    if labels.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("masked branch TCL labels must have integer dtype")
    if labels.numel() and bool(
        ((labels < 0) | (labels >= branch_logits.shape[2])).any()
    ):
        raise ValueError("masked branch TCL labels are outside class range")


def masked_branch_tcl_replay_context(
    branch_logits: torch.Tensor,
    branch_mask: torch.Tensor,
    labels: torch.Tensor,
    class_accuracy_ema: Sequence[float] | torch.Tensor,
) -> MaskedBranchTCLReplayContext:
    """Freeze primary correct-teacher gates and sqrt-error compensation."""

    _validate_masked_branch_tcl_inputs(branch_logits, branch_mask, labels)
    class_accuracy = torch.as_tensor(
        class_accuracy_ema,
        device=branch_logits.device,
        dtype=torch.float64,
    )
    if (
        class_accuracy.ndim != 1
        or class_accuracy.numel() != branch_logits.shape[2]
    ):
        raise ValueError(
            "masked branch TCL class EMA must contain one value per class"
        )
    if not bool(torch.isfinite(class_accuracy).all()) or bool(
        ((class_accuracy < 0) | (class_accuracy > 1)).any()
    ):
        raise ValueError(
            "masked branch TCL class EMA must be finite and lie in [0, 1]"
        )
    with torch.no_grad():
        working_labels = labels.detach().to(
            device=branch_logits.device, dtype=torch.long
        )
        active_branch_mask = branch_mask.detach().clone()
        correct_teacher_mask = active_branch_mask & branch_logits.detach().argmax(
            dim=2
        ).eq(working_labels.unsqueeze(1))
        sample_compensation = torch.sqrt(
            (1.0 - class_accuracy[working_labels]).clamp_min(0.0)
        ).to(dtype=branch_logits.dtype)
    return MaskedBranchTCLReplayContext(
        correct_teacher_mask=correct_teacher_mask,
        active_branch_mask=active_branch_mask,
        sample_compensation=sample_compensation.detach(),
    )


def masked_branch_tcl_kl_loss(
    branch_logits: torch.Tensor,
    replay_context: MaskedBranchTCLReplayContext,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Correct-teacher ordered-pair KL with pair/sample normalization.

    For each sample, every correct active branch teaches every *other* active
    branch.  Ordered pairs are averaged within a sample first; sqrt-error class
    compensation is then normalized across samples with at least one pair.
    Teacher distributions and all replay control state are detached.
    """

    if branch_logits.ndim != 3:
        raise ValueError(
            "masked branch TCL logits must have shape (batch, branches, classes)"
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "masked branch TCL temperature must be finite and positive"
        )
    batch_size, branch_count, _ = branch_logits.shape
    teacher_gate = replay_context.correct_teacher_mask
    active_mask = replay_context.active_branch_mask
    sample_compensation = replay_context.sample_compensation
    if teacher_gate.shape != (batch_size, branch_count):
        raise ValueError("masked branch TCL teacher gate has the wrong shape")
    if active_mask.shape != (batch_size, branch_count):
        raise ValueError("masked branch TCL active mask has the wrong shape")
    if teacher_gate.dtype != torch.bool or active_mask.dtype != torch.bool:
        raise ValueError("masked branch TCL replay masks must be boolean")
    if sample_compensation.shape != (batch_size,):
        raise ValueError(
            "masked branch TCL sample compensation has the wrong shape"
        )

    working_logits = (
        branch_logits.float()
        if branch_logits.dtype in {torch.float16, torch.bfloat16}
        else branch_logits
    )
    teacher_gate = teacher_gate.detach().to(device=working_logits.device)
    active_mask = active_mask.detach().to(device=working_logits.device)
    sample_compensation = sample_compensation.detach().to(
        device=working_logits.device, dtype=working_logits.dtype
    )
    if bool((teacher_gate & ~active_mask).any()):
        raise ValueError(
            "masked branch TCL teacher gate must be a subset of active branches"
        )
    if not bool(torch.isfinite(sample_compensation).all()) or bool(
        (sample_compensation < 0).any()
    ):
        raise ValueError(
            "masked branch TCL sample compensation must be finite and non-negative"
        )

    not_self = ~torch.eye(
        branch_count, device=working_logits.device, dtype=torch.bool
    )
    pair_mask = (
        teacher_gate.unsqueeze(2)
        & active_mask.unsqueeze(1)
        & not_self.unsqueeze(0)
    )
    pair_counts = pair_mask.sum(dim=(1, 2))
    active_samples = pair_counts > 0
    active_weights = sample_compensation * active_samples.to(
        sample_compensation.dtype
    )
    if not bool((active_weights > 0).any()):
        # Preserve a scalar zero connected to branch logits for empty gates,
        # single-branch samples, or an all-perfect class-EMA state.
        return working_logits.sum() * 0.0

    teacher_log_probability = torch.nn.functional.log_softmax(
        working_logits / temperature, dim=2
    ).detach()
    teacher_probability = teacher_log_probability.exp()
    student_log_probability = torch.nn.functional.log_softmax(
        working_logits / temperature, dim=2
    )
    per_pair = (
        teacher_probability.unsqueeze(2)
        * (
            teacher_log_probability.unsqueeze(2)
            - student_log_probability.unsqueeze(1)
        )
    ).sum(dim=3) * (temperature * temperature)
    per_sample = (
        (per_pair * pair_mask.to(per_pair.dtype)).sum(dim=(1, 2))
        / pair_counts.clamp_min(1).to(per_pair.dtype)
    )
    return (per_sample * active_weights).sum() / active_weights.sum()


def trusted_branch_fusion_distillation_loss(
    fused_logits: torch.Tensor,
    branch_logits: torch.Tensor,
    replay_context: MaskedBranchTCLReplayContext,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Distill the mean trusted-branch distribution into final fused logits.

    The clean primary forward freezes which active branches are trusted and
    the epoch-lagged class compensation.  Teacher probabilities are recomputed
    from the supplied branch logits (including at a SAM replay point), averaged
    across the frozen trusted set, and detached.  Only final fused logits are
    students, preserving specialist diversity while training fusion/routing.
    """

    if fused_logits.ndim != 2:
        raise ValueError("TBFD fused logits must have shape (batch, classes)")
    if branch_logits.ndim != 3:
        raise ValueError(
            "TBFD branch logits must have shape (batch, branches, classes)"
        )
    if (
        branch_logits.shape[0] != fused_logits.shape[0]
        or branch_logits.shape[2] != fused_logits.shape[1]
    ):
        raise ValueError("TBFD fused and branch logits have incompatible shapes")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("TBFD temperature must be finite and positive")

    batch_size, branch_count, _ = branch_logits.shape
    teacher_gate = replay_context.correct_teacher_mask
    active_mask = replay_context.active_branch_mask
    sample_compensation = replay_context.sample_compensation
    if teacher_gate.shape != (batch_size, branch_count):
        raise ValueError("TBFD teacher gate has the wrong shape")
    if active_mask.shape != (batch_size, branch_count):
        raise ValueError("TBFD active branch mask has the wrong shape")
    if teacher_gate.dtype != torch.bool or active_mask.dtype != torch.bool:
        raise ValueError("TBFD replay masks must be boolean")
    if sample_compensation.shape != (batch_size,):
        raise ValueError("TBFD sample compensation has the wrong shape")

    working_fused_logits = (
        fused_logits.float()
        if fused_logits.dtype in {torch.float16, torch.bfloat16}
        else fused_logits
    )
    working_branch_logits = branch_logits.to(
        device=working_fused_logits.device,
        dtype=working_fused_logits.dtype,
    )
    teacher_gate = teacher_gate.detach().to(device=working_fused_logits.device)
    active_mask = active_mask.detach().to(device=working_fused_logits.device)
    sample_compensation = sample_compensation.detach().to(
        device=working_fused_logits.device,
        dtype=working_fused_logits.dtype,
    )
    if bool((teacher_gate & ~active_mask).any()):
        raise ValueError("TBFD teacher gate must be a subset of active branches")
    if not bool(torch.isfinite(sample_compensation).all()) or bool(
        (sample_compensation < 0).any()
    ):
        raise ValueError(
            "TBFD sample compensation must be finite and non-negative"
        )

    teacher_counts = teacher_gate.sum(dim=1)
    active_samples = teacher_counts > 0
    active_weights = sample_compensation * active_samples.to(
        sample_compensation.dtype
    )
    if not bool((active_weights > 0).any()):
        return working_fused_logits.sum() * 0.0

    branch_teacher_probabilities = torch.softmax(
        working_branch_logits / temperature, dim=2
    ).detach()
    trusted_teacher = (
        (
            branch_teacher_probabilities
            * teacher_gate.unsqueeze(2).to(branch_teacher_probabilities.dtype)
        ).sum(dim=1)
        / teacher_counts.clamp_min(1).unsqueeze(1).to(
            branch_teacher_probabilities.dtype
        )
    ).detach()
    fused_log_probability = torch.nn.functional.log_softmax(
        working_fused_logits / temperature, dim=1
    )
    per_sample = torch.nn.functional.kl_div(
        fused_log_probability,
        trusted_teacher,
        reduction="none",
    ).sum(dim=1) * (temperature * temperature)
    return (per_sample * active_weights).sum() / active_weights.sum()


class MaskedBranchTCLClassAccuracyEMA:
    """Epoch-lagged clean-primary available-branch class accuracy state."""

    def __init__(self, num_classes: int, decay: float):
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("masked branch TCL requires at least two classes")
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("masked branch TCL EMA decay must be in [0, 1)")
        self.num_classes = num_classes
        self.decay = decay
        self.class_accuracy = np.zeros(num_classes, dtype=np.float64)
        self.initialized = np.zeros(num_classes, dtype=np.bool_)
        self.last_updated_epoch = 0
        self.history: list[dict[str, Any]] = []
        self._active_epoch: int | None = None
        self._correct = np.zeros(num_classes, dtype=np.int64)
        self._valid = np.zeros(num_classes, dtype=np.int64)

    def begin_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 1:
            raise ValueError("masked branch TCL epoch must be a positive integer")
        if self._active_epoch is not None:
            raise RuntimeError("masked branch TCL previous epoch was not finalized")
        if epoch != self.last_updated_epoch + 1:
            raise RuntimeError(
                "masked branch TCL epochs must be finalized exactly once in order"
            )
        self._active_epoch = epoch
        self._correct.fill(0)
        self._valid.fill(0)

    def collect_clean_primary(
        self,
        branch_logits: torch.Tensor,
        branch_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if self._active_epoch is None:
            raise RuntimeError(
                "masked branch TCL statistics require an active training epoch"
            )
        _validate_masked_branch_tcl_inputs(branch_logits, branch_mask, labels)
        if branch_logits.shape[2] != self.num_classes:
            raise ValueError("masked branch TCL class count changed during training")
        with torch.no_grad():
            working_labels = labels.detach().to(
                device=branch_logits.device, dtype=torch.long
            )
            active = branch_mask.detach()
            correct = active & branch_logits.detach().argmax(dim=2).eq(
                working_labels.unsqueeze(1)
            )
            for class_index in range(self.num_classes):
                rows = working_labels.eq(class_index)
                self._correct[class_index] += int(correct[rows].sum().cpu())
                self._valid[class_index] += int(active[rows].sum().cpu())

    def replay_context(
        self,
        branch_logits: torch.Tensor,
        branch_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> MaskedBranchTCLReplayContext:
        if self._active_epoch is None:
            raise RuntimeError(
                "masked branch TCL context requires an active training epoch"
            )
        return masked_branch_tcl_replay_context(
            branch_logits,
            branch_mask,
            labels,
            self.class_accuracy,
        )

    def end_epoch(self, epoch: int) -> dict[str, Any]:
        if self._active_epoch != epoch:
            raise RuntimeError(
                "masked branch TCL epoch can only be finalized once by its owner"
            )
        observed_accuracy: list[float | None] = []
        for class_index in range(self.num_classes):
            valid = int(self._valid[class_index])
            if valid == 0:
                observed_accuracy.append(None)
                continue
            accuracy = float(self._correct[class_index]) / float(valid)
            observed_accuracy.append(accuracy)
            if self.initialized[class_index]:
                self.class_accuracy[class_index] = (
                    self.decay * self.class_accuracy[class_index]
                    + (1.0 - self.decay) * accuracy
                )
            else:
                # First observation is assigned directly, avoiding an
                # artificial zero-initialization bias.
                self.class_accuracy[class_index] = accuracy
                self.initialized[class_index] = True
        record = {
            "epoch": int(epoch),
            "branch_correct": self._correct.astype(np.int64).tolist(),
            "branch_valid": self._valid.astype(np.int64).tolist(),
            "observed_class_accuracy": observed_accuracy,
            "class_accuracy_ema": self.class_accuracy.astype(
                np.float64, copy=False
            ).tolist(),
            "initialized": self.initialized.astype(bool, copy=False).tolist(),
        }
        self.history.append(record)
        self.last_updated_epoch = epoch
        self._active_epoch = None
        return record

    def _materialize_checkpoint_args(self, args, prefix: str) -> None:
        if self._active_epoch is not None:
            raise RuntimeError(
                "cannot checkpoint branch-accuracy EMA before finalizing the epoch"
            )
        setattr(
            args,
            f"{prefix}_final_class_accuracy_ema",
            self.class_accuracy.astype(np.float64, copy=False).tolist()
        )
        setattr(
            args,
            f"{prefix}_final_class_initialized",
            self.initialized.astype(bool, copy=False).tolist()
        )
        setattr(args, f"{prefix}_epochs_updated", int(self.last_updated_epoch))
        setattr(
            args,
            f"{prefix}_ema_history_json",
            json.dumps(
                self.history,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )

    def materialize_checkpoint_args(self, args) -> None:
        self._materialize_checkpoint_args(args, "masked_branch_tcl")

    def materialize_trusted_branch_fusion_checkpoint_args(self, args) -> None:
        self._materialize_checkpoint_args(args, "trusted_branch_fusion")


def masked_branch_tcl_protocol(args) -> dict[str, Any]:
    """Return the static recipe and optional finalized EMA audit state."""

    weight = float(getattr(args, "masked_branch_tcl_loss_weight", 0.0))
    enabled = weight > 0.0
    history_json = getattr(args, "masked_branch_tcl_ema_history_json", None)
    history = None if history_json is None else json.loads(history_json)
    final_state = None
    if enabled and history is not None:
        final_state = {
            "class_accuracy_ema": getattr(
                args, "masked_branch_tcl_final_class_accuracy_ema", None
            ),
            "initialized": getattr(
                args, "masked_branch_tcl_final_class_initialized", None
            ),
            "epochs_updated": getattr(
                args, "masked_branch_tcl_epochs_updated", None
            ),
            "history": history,
        }
    return {
        "enabled": enabled,
        "loss_weight": weight,
        "temperature": float(
            getattr(args, "masked_branch_tcl_temperature", 2.0)
        ),
        "start_epoch": int(
            getattr(args, "masked_branch_tcl_start_epoch", 6)
        ),
        "class_accuracy_ema_decay": float(
            getattr(args, "masked_branch_tcl_class_acc_ema", 0.9)
        ),
        "teacher_gate": (
            "clean-primary active branch whose detached argmax equals the label"
        ),
        "ordered_pairs": (
            "each correct active teacher to every other active student branch"
        ),
        "normalization": (
            "mean active ordered-pair KL per sample, then sqrt(1-class-accuracy-EMA) "
            "weighted mean over samples with a pair"
        ),
        "class_accuracy_statistic": (
            "per-class correct active branch predictions divided by active branch "
            "predictions, collected from clean-primary training forwards only"
        ),
        "epoch_lag": (
            "epoch e loss uses EMA through epoch e-1; update once after epoch e"
        ),
        "warmup": "epochs before start_epoch collect statistics without TCL loss",
        "rdrop_secondary_loss_or_update": False,
        "sam_replay": (
            "reuse primary detached teacher gate, active mask, and class weights; "
            "recompute replay teacher/student distributions; no statistic update"
        ),
        "validation_or_test_loss_or_update": False,
        "inference_change": False,
        "paper_relationship": (
            "TCL correct-teacher gating plus sqrt-error class compensation and "
            "TKO-style collaboration adapted to dynamically available MoE branches; "
            "not a claim of complete TCL reproduction"
        ),
        "final_ema_state": final_state,
    }


def trusted_branch_fusion_distillation_protocol(args) -> dict[str, Any]:
    """Return the auditable trusted branch-to-fusion recipe and EMA state."""

    weight = float(
        getattr(args, "trusted_branch_fusion_distill_weight", 0.0)
    )
    enabled = weight > 0.0
    history_json = getattr(
        args, "trusted_branch_fusion_ema_history_json", None
    )
    history = None if history_json is None else json.loads(history_json)
    final_state = None
    if enabled and history is not None:
        final_state = {
            "class_accuracy_ema": getattr(
                args,
                "trusted_branch_fusion_final_class_accuracy_ema",
                None,
            ),
            "initialized": getattr(
                args,
                "trusted_branch_fusion_final_class_initialized",
                None,
            ),
            "epochs_updated": getattr(
                args, "trusted_branch_fusion_epochs_updated", None
            ),
            "history": history,
        }
    return {
        "enabled": enabled,
        "loss_weight": weight,
        "temperature": float(
            getattr(args, "trusted_branch_fusion_temperature", 2.0)
        ),
        "start_epoch": int(
            getattr(args, "trusted_branch_fusion_start_epoch", 6)
        ),
        "class_accuracy_ema_decay": float(
            getattr(args, "trusted_branch_fusion_class_acc_ema", 0.9)
        ),
        "teacher_gate": (
            "clean-primary active branch whose detached argmax equals the label"
        ),
        "teacher_aggregation": (
            "equal mean of detached temperature probabilities over trusted branches"
        ),
        "student": "final fused output logits only",
        "normalization": (
            "sqrt(1-class-accuracy-EMA) weighted mean over samples with at "
            "least one trusted branch"
        ),
        "class_accuracy_statistic": (
            "per-class correct active branch predictions divided by active branch "
            "predictions, collected from clean-primary training forwards only"
        ),
        "epoch_lag": (
            "epoch e loss uses EMA through epoch e-1; update once after epoch e"
        ),
        "warmup": "epochs before start_epoch collect statistics without TBFD loss",
        "rdrop_secondary_loss_or_update": False,
        "sam_replay": (
            "reuse primary detached teacher gate, active mask, and class weights; "
            "recompute detached teacher from replay branch logits and train replay "
            "fused logits; no statistic update"
        ),
        "validation_or_test_loss_or_update": False,
        "inference_change": False,
        "mutually_exclusive_with_masked_branch_tcl": True,
        "paper_relationship": (
            "TCL/TKO correct-teacher idea adapted to dynamic MoE fusion mismatch; "
            "not a claim of complete TCL reproduction"
        ),
        "final_ema_state": final_state,
    }


def more_tail_regularization_loss(
    final_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    training_class_prior,
) -> torch.Tensor:
    """MORE-inspired prior-weighted fused-logit residual magnitude.

    This is the classifier-level quantity
    ``mean_i pi[y_i] * ||f_final(x_i) - f_base(x_i)||_2^2``.  The prior is
    supplied explicitly from the current training split; this function never
    estimates it from the current batch or any evaluation labels.
    """

    if final_logits.ndim != 2 or final_logits.shape[1] < 2:
        raise ValueError(
            "MORE fused-logit regularization requires shape (batch, classes>=2)"
        )
    if base_logits.shape != final_logits.shape:
        raise ValueError("MORE base and final logits must have identical shapes")
    if labels.ndim != 1 or labels.shape[0] != final_logits.shape[0]:
        raise ValueError("MORE labels must match the logits batch")
    prior = torch.as_tensor(
        training_class_prior,
        device=final_logits.device,
        dtype=final_logits.dtype,
    )
    if prior.ndim != 1 or prior.numel() != final_logits.shape[1]:
        raise ValueError("MORE training prior must contain one value per class")
    if not bool(torch.isfinite(prior).all()) or bool((prior < 0).any()):
        raise ValueError("MORE training prior must be finite and non-negative")
    if not torch.allclose(
        prior.sum(),
        prior.new_tensor(1.0),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("MORE training prior must sum to one")
    if labels.numel() > 0 and bool(
        ((labels < 0) | (labels >= final_logits.shape[1])).any()
    ):
        raise ValueError("MORE labels are outside the configured class range")
    residual_squared_norm = (final_logits - base_logits).square().sum(dim=1)
    return (prior[labels] * residual_squared_norm).mean()


def more_tail_sine_weight(
    peak_amplitude: float,
    training_progress: float,
) -> float:
    """Return ``A*sin(pi*progress)`` for normalized global batch progress."""

    peak_amplitude = float(peak_amplitude)
    training_progress = float(training_progress)
    if not math.isfinite(peak_amplitude) or peak_amplitude < 0:
        raise ValueError("MORE peak amplitude must be finite and non-negative")
    if (
        not math.isfinite(training_progress)
        or not 0.0 <= training_progress <= 1.0
    ):
        raise ValueError("MORE training progress must be finite and in [0, 1]")
    if peak_amplitude == 0.0 or training_progress in {0.0, 1.0}:
        return 0.0
    return peak_amplitude * math.sin(math.pi * training_progress)


def dropped_view_forward_required(args, supervised_contrastive_weight: float) -> bool:
    """Whether the already-defined reduced-modality forward must execute.

    Keeping another dropped-view objective active makes the MoFe rank-weight
    ablation change only its effective scalar, not the forward/RNG schedule.
    """
    return bool(
        args.modality_dropout_prob > 0
        and (
            args.drop_ce_loss_weight > 0
            or args.distill_loss_weight > 0
            or supervised_contrastive_weight > 0
            or float(getattr(args, "more_fewer_rank_loss_weight", 0.0)) > 0
        )
    )


def more_fewer_rank_effective_weight(args) -> float:
    control = float(getattr(args, "more_fewer_rank_loss_weight", 0.0))
    effective = getattr(args, "more_fewer_rank_loss_effective_weight", None)
    return control if effective is None else float(effective)


def our_moe_objective(
    args,
    model,
    output,
    tokens,
    observed_mask,
    modality_comb,
    labels,
    criterion,
    epoch: int | None = None,
    tree_teacher_probabilities: torch.Tensor | None = None,
    external_teacher_probabilities: torch.Tensor | None = None,
    training_progress: float | None = None,
    masked_branch_tcl_context: MaskedBranchTCLReplayContext | None = None,
    trusted_branch_fusion_context: MaskedBranchTCLReplayContext | None = None,
    empirical_missing_pattern_replay_state: (
        EmpiricalMissingPatternReplayState | None
    ) = None,
):
    """Training objective used by the repository's AGMGFlexMoE method."""
    if (
        bool(getattr(model, "missing_family_normalized_router", False))
        and float(getattr(args, "supervised_router_loss_weight", 0.0)) > 0.0
    ):
        raise ValueError(
            "missing-family routing v1 cannot use the historical 11-branch "
            "supervised router objective"
        )
    if bool(getattr(args, "empirical_missing_pattern_replay", False)):
        empirical_missing_pattern_replay_active_objectives(args)
        if not isinstance(
            empirical_missing_pattern_replay_state,
            EmpiricalMissingPatternReplayState,
        ):
            raise RuntimeError(
                "Enabled empirical replay objective requires persistent "
                "explicit replay state"
            )
    task_loss = criterion(output["logits"], labels)
    gate_loss = model.gate_loss()
    if not torch.is_tensor(gate_loss):
        gate_loss = task_loss.new_tensor(float(gate_loss))
    recon_loss = output.get("recon_loss")
    recon_loss = recon_loss if recon_loss is not None else task_loss.new_zeros(())
    raw_aux = output.get("aux_loss")
    raw_aux = raw_aux if raw_aux is not None else task_loss.new_zeros(())
    entropy_loss = output.get("w_entropy")
    entropy_loss = entropy_loss if entropy_loss is not None else task_loss.new_zeros(())

    more_tail_rank = int(getattr(args, "more_tail_rank", 0))
    more_tail_effective_weight = 0.0
    more_tail_regularization = task_loss.new_zeros(())
    if more_tail_rank > 0:
        if training_progress is None:
            raise ValueError(
                "MORE fused-logit tail training requires global batch progress"
            )
        base_logits = output.get("base_logits")
        tail_logits = output.get("tail_logits")
        if base_logits is None or tail_logits is None:
            raise RuntimeError(
                "enabled MORE tail adapter returned no base/tail diagnostics"
            )
        if tail_logits.shape != output["logits"].shape:
            raise RuntimeError("MORE tail-logit diagnostics have the wrong shape")
        more_tail_regularization = more_tail_regularization_loss(
            output["logits"],
            base_logits,
            labels,
            getattr(args, "more_tail_class_prior", None),
        )
        more_tail_effective_weight = more_tail_sine_weight(
            getattr(args, "more_tail_peak_amplitude", 0.0),
            training_progress,
        )

    dual_local_boundary_weight = float(
        getattr(args, "dual_local_boundary_loss_weight", 0.0)
    )
    dual_local_boundary = task_loss.new_zeros(())
    if dual_local_boundary_weight > 0:
        dlbr_base_logits = output.get("dlbr_base_logits")
        dlbr_residuals = output.get("dlbr_residuals")
        if dlbr_base_logits is None or dlbr_residuals is None:
            raise RuntimeError(
                "enabled DLBR objective requires base logits and residuals"
            )
        if dlbr_base_logits.shape != output["logits"].shape:
            raise RuntimeError("DLBR base logits have the wrong shape")
        dual_local_boundary = dual_local_boundary_residual_loss(
            dlbr_base_logits,
            dlbr_residuals,
            labels,
        )

    presentation_axis_weight = float(
        getattr(args, "presentation_axis_loss_weight", 0.0)
    )
    presentation_axis = task_loss.new_zeros(())
    if presentation_axis_weight > 0:
        presentation_axis_base_logits = output.get(
            "presentation_axis_base_logits"
        )
        presentation_axis_raw_logits = output.get(
            "presentation_axis_raw_logits"
        )
        presentation_axis_residuals = output.get(
            "presentation_axis_residuals"
        )
        if (
            presentation_axis_base_logits is None
            or presentation_axis_raw_logits is None
            or presentation_axis_residuals is None
        ):
            raise RuntimeError(
                "enabled presentation-axis objective requires base logits, "
                "raw axis logits, and bounded residuals"
            )
        if presentation_axis_base_logits.shape != output["logits"].shape:
            raise RuntimeError(
                "presentation-axis base logits have the wrong shape"
            )
        expected_axis_shape = (output["logits"].shape[0], 2)
        if presentation_axis_raw_logits.shape != expected_axis_shape:
            raise RuntimeError(
                "presentation-axis raw logits have the wrong shape"
            )
        if presentation_axis_residuals.shape != expected_axis_shape:
            raise RuntimeError(
                "presentation-axis residuals have the wrong shape"
            )
        presentation_axis = presentation_axis_balanced_logistic_loss(
            presentation_axis_raw_logits,
            labels,
        )

    tree_teacher_weight = float(
        getattr(args, "tree_teacher_distill_weight", 0.0)
    )
    tree_teacher_distillation = task_loss.new_zeros(())
    external_teacher_weight = float(
        getattr(args, "external_teacher_distill_weight", 0.0)
    )
    if tree_teacher_weight > 0 and external_teacher_weight > 0:
        raise RuntimeError(
            "v1 tree-teacher and v2 external-teacher objectives are mutually exclusive"
        )
    if tree_teacher_weight > 0:
        if tree_teacher_probabilities is None:
            raise RuntimeError(
                "tree-teacher distillation is enabled but no targets were supplied"
            )
        tree_teacher_distillation = cross_fitted_tree_teacher_kl_loss(
            output["logits"],
            tree_teacher_probabilities,
            getattr(args, "tree_teacher_temperature", 2.0),
        )
    external_teacher_distillation = task_loss.new_zeros(())
    if external_teacher_weight > 0:
        if external_teacher_probabilities is None:
            raise RuntimeError(
                "external-teacher distillation is enabled but no targets were "
                "supplied"
            )
        external_teacher_distillation = (
            cross_fitted_external_teacher_kl_loss(
                output["logits"],
                external_teacher_probabilities,
                labels,
                getattr(args, "external_teacher_temperature", 2.0),
                getattr(args, "external_teacher_trust_mode", "correct_only"),
                getattr(args, "external_teacher_class_compensation", "none"),
                getattr(
                    args,
                    "external_teacher_class_compensation_weights",
                    None,
                ),
            )
        )

    masked_branch_tcl_weight = float(
        getattr(args, "masked_branch_tcl_loss_weight", 0.0)
    )
    trusted_branch_fusion_weight = float(
        getattr(args, "trusted_branch_fusion_distill_weight", 0.0)
    )
    if masked_branch_tcl_weight > 0 and trusted_branch_fusion_weight > 0:
        raise RuntimeError(
            "masked branch TCL and trusted branch-to-fusion distillation are "
            "mutually exclusive"
        )
    masked_branch_tcl = task_loss.new_zeros(())
    if masked_branch_tcl_weight > 0:
        if epoch is None:
            raise ValueError(
                "masked branch TCL training requires an explicit epoch"
            )
        masked_branch_tcl_start_epoch = int(
            getattr(args, "masked_branch_tcl_start_epoch", 6)
        )
        if epoch >= masked_branch_tcl_start_epoch:
            if masked_branch_tcl_context is None:
                raise RuntimeError(
                    "enabled masked branch TCL received no frozen primary context"
                )
            branch_logits_for_tcl = output.get("branch_logits")
            if branch_logits_for_tcl is None:
                raise RuntimeError(
                    "enabled masked branch TCL requires branch logits"
                )
            masked_branch_tcl = masked_branch_tcl_kl_loss(
                branch_logits_for_tcl,
                masked_branch_tcl_context,
                getattr(args, "masked_branch_tcl_temperature", 2.0),
            )
        elif masked_branch_tcl_context is not None:
            raise RuntimeError(
                "masked branch TCL context must not be supplied before start_epoch"
            )
    trusted_branch_fusion = task_loss.new_zeros(())
    if trusted_branch_fusion_weight > 0:
        if epoch is None:
            raise ValueError("TBFD training requires an explicit epoch")
        trusted_branch_fusion_start_epoch = int(
            getattr(args, "trusted_branch_fusion_start_epoch", 6)
        )
        if epoch >= trusted_branch_fusion_start_epoch:
            if trusted_branch_fusion_context is None:
                raise RuntimeError(
                    "enabled TBFD received no frozen primary context"
                )
            branch_logits_for_fusion = output.get("branch_logits")
            if branch_logits_for_fusion is None:
                raise RuntimeError("enabled TBFD requires branch logits")
            trusted_branch_fusion = trusted_branch_fusion_distillation_loss(
                output["logits"],
                branch_logits_for_fusion,
                trusted_branch_fusion_context,
                getattr(args, "trusted_branch_fusion_temperature", 2.0),
            )
        elif trusted_branch_fusion_context is not None:
            raise RuntimeError(
                "TBFD context must not be supplied before start_epoch"
            )

    class1_aux_weight = float(getattr(args, "class1_aux_loss_weight", 0.0))
    class1_aux = task_loss.new_zeros(())
    if class1_aux_weight > 0:
        class1_aux_logit = output.get("class1_aux_logit")
        if class1_aux_logit is None:
            raise RuntimeError(
                "class-1 auxiliary loss is enabled but the model returned no auxiliary logit"
            )
        class1_aux = class1_auxiliary_loss(class1_aux_logit, labels)

    supervised_router_weight = float(
        getattr(args, "supervised_router_loss_weight", 0.0)
    )
    supervised_router = task_loss.new_zeros(())
    if supervised_router_weight > 0:
        branch_logits_for_router = output.get("branch_logits")
        supervision_log_scores = output.get("supervision_log_scores")
        branch_mask_for_router = output.get("supervision_branch_mask")
        if branch_mask_for_router is None:
            if getattr(
                args, "supervised_router_observed_specialists_only", False
            ):
                raise RuntimeError(
                    "observed-specialist router supervision requires its "
                    "independent supervision branch mask"
                )
            # Backward-compatible objective calls and historical model stubs
            # use the prediction branch mask for the legacy teacher.
            branch_mask_for_router = output.get("branch_mask")
        if (
            branch_logits_for_router is None
            or supervision_log_scores is None
            or branch_mask_for_router is None
        ):
            raise RuntimeError(
                "supervised router loss requires branch logits, routing scores, and mask"
            )
        supervised_router = label_supervised_router_loss(
            branch_logits_for_router,
            supervision_log_scores,
            branch_mask_for_router,
            labels,
            getattr(args, "supervised_router_temperature", 0.25),
        )
    supervised_contrastive_weight = float(
        getattr(args, "supervised_contrastive_loss_weight", 0.0)
    )
    supervised_contrastive = task_loss.new_zeros(())
    dual_boundary_rank_weight = float(
        getattr(args, "dual_boundary_rank_loss_weight", 0.0)
    )
    dual_boundary_rank = task_loss.new_zeros(())
    if dual_boundary_rank_weight > 0:
        dual_boundary_rank = dual_boundary_rank_loss(
            output["logits"],
            labels,
            margin=getattr(args, "dual_boundary_rank_margin", 0.20),
            class1_vs_0_weight=getattr(
                args, "dual_boundary_rank_10_weight", 2.0 / 3.0
            ),
        )
    hard_cvar_max_weight = float(
        getattr(args, "hard_cvar_dual_boundary_max_weight", 0.0)
    )
    hard_cvar_effective_weight = 0.0
    hard_cvar_dual_boundary = task_loss.new_zeros(())
    if hard_cvar_max_weight > 0:
        if epoch is None:
            raise ValueError(
                "hard-CVaR dual-boundary training requires an explicit epoch"
            )
        hard_cvar_effective_weight = hard_cvar_dual_boundary_weight_for_epoch(
            epoch,
            max_weight=hard_cvar_max_weight,
            start_epoch=getattr(
                args, "hard_cvar_dual_boundary_start_epoch", 5
            ),
            ramp_epochs=getattr(
                args, "hard_cvar_dual_boundary_ramp_epochs", 10
            ),
        )
        if hard_cvar_effective_weight > 0:
            hard_cvar_dual_boundary = hard_cvar_dual_boundary_loss(
                output["logits"],
                labels,
                tail_fraction=getattr(
                    args, "hard_cvar_dual_boundary_tail_fraction", 0.25
                ),
                margin=getattr(
                    args, "hard_cvar_dual_boundary_margin", 0.20
                ),
                class1_vs_0_weight=getattr(
                    args, "hard_cvar_dual_boundary_10_weight", 0.65
                ),
            )

    ordinal_aux = task_loss.new_zeros(())
    ordinal_logits = output.get("ordinal_logits")
    if args.ordinal_aux_loss_weight > 0 and ordinal_logits is not None:
        if args.ordinal_head_type == "continuation":
            first_target = (labels >= 1).to(dtype=ordinal_logits.dtype)
            first_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                ordinal_logits[:, 0], first_target
            )
            eligible = labels >= 1
            if eligible.any():
                second_target = (labels[eligible] == 2).to(dtype=ordinal_logits.dtype)
                second_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    ordinal_logits[eligible, 1], second_target
                )
                ordinal_aux = 0.5 * (first_loss + second_loss)
            else:
                ordinal_aux = 0.5 * first_loss
        else:
            cumulative_targets = torch.stack(
                ((labels >= 1), (labels >= 2)), dim=1
            ).to(dtype=ordinal_logits.dtype)
            ordinal_aux = torch.nn.functional.binary_cross_entropy_with_logits(
                ordinal_logits, cumulative_targets
            )

    branch_aux = task_loss.new_zeros(())
    branch_logits = output.get("branch_logits")
    branch_mask = output.get("branch_mask")
    branch_quality = output.get("branch_quality")
    joint_head_ensemble_size = int(
        getattr(args, "joint_head_ensemble_size", 1)
    )
    joint_member_logits = output.get("joint_member_logits")
    if args.branch_aux_loss_weight > 0 and branch_logits is not None and branch_mask is not None:
        branch_losses = []
        for branch_index in range(branch_logits.shape[1]):
            usable = branch_mask[:, branch_index]
            if not usable.any():
                continue
            current_logits = branch_logits[usable, branch_index, :]
            current_labels = labels[usable]
            if branch_index == 0 and joint_head_ensemble_size > 1:
                if joint_member_logits is None:
                    raise RuntimeError(
                        "joint-head ensemble branch auxiliary requires member logits"
                    )
                if joint_member_logits.shape[1] != joint_head_ensemble_size:
                    raise RuntimeError(
                        "joint-head ensemble diagnostic member count mismatch"
                    )
                branch_losses.append(
                    joint_member_cross_entropy(
                        joint_member_logits[usable],
                        current_labels,
                        criterion,
                    )
                )
                continue
            if (
                args.balanced_branch_aux
                or branch_quality is None
                or (
                    args.dynamic_branch_fusion
                    and not getattr(
                        args, "dynamic_branch_quality_weighted_aux", False
                    )
                )
            ):
                branch_losses.append(criterion(current_logits, current_labels))
            else:
                per_sample = torch.nn.functional.cross_entropy(
                    current_logits,
                    current_labels,
                    weight=(
                        criterion.weight
                        if args.class_weighted_quality_branch_aux
                        else None
                    ),
                    reduction="none",
                    label_smoothing=args.label_smoothing,
                )
                quality = branch_quality[usable, branch_index].detach().clamp_min(1e-3)
                normalizer = quality
                if args.class_weighted_quality_branch_aux and criterion.weight is not None:
                    # Match CrossEntropyLoss(weight=..., reduction="mean"): the
                    # weighted numerator is normalized by target-class weights.
                    normalizer = normalizer * criterion.weight[current_labels]
                branch_losses.append(
                    (per_sample * quality).sum() / normalizer.sum().clamp_min(1e-6)
                )
        if branch_losses:
            branch_aux = torch.stack(branch_losses).mean()

    drop_ce = task_loss.new_zeros(())
    distillation = task_loss.new_zeros(())
    drop_gate = task_loss.new_zeros(())
    more_fewer_rank_weight = float(
        getattr(args, "more_fewer_rank_loss_weight", 0.0)
    )
    effective_rank_weight = more_fewer_rank_effective_weight(args)
    more_fewer_rank = task_loss.new_zeros(())
    if dropped_view_forward_required(args, supervised_contrastive_weight):
        # Always consume the exact historical dropout RNG first.  The optional
        # replay replaces only its mask value and samples from a distinct CPU
        # generator, so the global stream after this line is candidate-invariant.
        legacy_dropped_observed = modality_dropout_mask(
            observed_mask, args.modality_dropout_prob
        )
        dropped_observed = (
            empirical_missing_pattern_replay_mask(
                observed_mask,
                legacy_dropped_observed,
                empirical_missing_pattern_replay_state,
            )
            if bool(
                getattr(args, "empirical_missing_pattern_replay", False)
            )
            else legacy_dropped_observed
        )
        dropped_combination = (
            combination_indices_from_mask(
                dropped_observed,
                args.modality,
                sort_codes_before_enumeration=args.data == "abcd",
            )
            if args.recompute_dropped_combination
            else modality_comb
        )
        dropped_output = model(
            *tokens,
            observed_mask=dropped_observed,
            expert_indices=dropped_combination,
            return_importance=False,
            return_recon_loss=False,
        )
        dropped_logits = dropped_output["logits"]
        drop_gate = model.gate_loss()
        if not torch.is_tensor(drop_gate):
            drop_gate = task_loss.new_tensor(float(drop_gate))
        if args.drop_ce_loss_weight > 0:
            drop_ce = criterion(dropped_logits, labels)
        if more_fewer_rank_weight > 0:
            more_fewer_rank = more_fewer_rank_loss(
                output["logits"],
                dropped_logits,
                labels,
                observed_mask,
                dropped_observed,
                criterion,
            )
        if args.distill_loss_weight > 0:
            temperature = max(args.distill_temperature, 1e-6)
            teacher = torch.softmax(output["logits"].detach() / temperature, dim=1)
            student = torch.log_softmax(dropped_logits / temperature, dim=1)
            distillation = torch.nn.functional.kl_div(
                student, teacher, reduction="batchmean"
            ) * (temperature * temperature)
        if supervised_contrastive_weight > 0:
            full_embedding = output.get("supervised_contrastive_embedding")
            dropped_embedding = dropped_output.get(
                "supervised_contrastive_embedding"
            )
            if full_embedding is None or dropped_embedding is None:
                raise RuntimeError(
                    "supervised contrastive loss requires full and dropped embeddings"
                )
            supervised_contrastive = (
                class_balanced_supervised_contrastive_loss(
                    torch.stack((full_embedding, dropped_embedding), dim=1),
                    labels,
                    temperature=getattr(
                        args, "supervised_contrastive_temperature", 0.10
                    ),
                    class_weights=criterion.weight,
                )
            )

    auxiliary = (
        args.gate_loss_weight * gate_loss
        + args.recon_loss_weight * recon_loss
        + 0.1 * raw_aux
        + args.w_entropy_weight * entropy_loss
        + args.ordinal_aux_loss_weight * ordinal_aux
        + class1_aux_weight * class1_aux
        + args.branch_aux_loss_weight * branch_aux
        + 0.5 * args.gate_loss_weight * drop_gate
        + args.drop_ce_loss_weight * drop_ce
        + args.distill_loss_weight * distillation
    )
    if supervised_router_weight > 0:
        auxiliary = auxiliary + supervised_router_weight * supervised_router
    if supervised_contrastive_weight > 0:
        auxiliary = (
            auxiliary
            + supervised_contrastive_weight * supervised_contrastive
        )
    if dual_boundary_rank_weight > 0:
        auxiliary = auxiliary + dual_boundary_rank_weight * dual_boundary_rank
    if effective_rank_weight > 0:
        auxiliary = auxiliary + effective_rank_weight * more_fewer_rank
    if hard_cvar_effective_weight > 0:
        auxiliary = (
            auxiliary
            + hard_cvar_effective_weight * hard_cvar_dual_boundary
        )
    if tree_teacher_weight > 0:
        auxiliary = (
            auxiliary
            + tree_teacher_weight * tree_teacher_distillation
        )
    if external_teacher_weight > 0:
        auxiliary = (
            auxiliary
            + external_teacher_weight * external_teacher_distillation
        )
    if masked_branch_tcl_weight > 0:
        auxiliary = (
            auxiliary + masked_branch_tcl_weight * masked_branch_tcl
        )
    if trusted_branch_fusion_weight > 0:
        auxiliary = (
            auxiliary
            + trusted_branch_fusion_weight * trusted_branch_fusion
        )
    if more_tail_effective_weight > 0:
        auxiliary = (
            auxiliary
            + more_tail_effective_weight * more_tail_regularization
        )
    if dual_local_boundary_weight > 0:
        auxiliary = (
            auxiliary
            + dual_local_boundary_weight * dual_local_boundary
        )
    if presentation_axis_weight > 0:
        auxiliary = (
            auxiliary
            + presentation_axis_weight * presentation_axis
        )
    return task_loss + auxiliary, task_loss, auxiliary


def rdrop_symmetric_kl_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the batch-mean symmetric KL between two clean-view logits."""

    if first_logits.ndim != 2 or second_logits.ndim != 2:
        raise ValueError("R-Drop logits must both have shape (batch, classes)")
    if first_logits.shape != second_logits.shape:
        raise ValueError("R-Drop logits must have identical shapes")
    first_log_probability = torch.nn.functional.log_softmax(first_logits, dim=1)
    second_log_probability = torch.nn.functional.log_softmax(second_logits, dim=1)
    first_probability = first_log_probability.exp()
    second_probability = second_log_probability.exp()
    return 0.5 * (
        torch.nn.functional.kl_div(
            first_log_probability,
            second_probability,
            reduction="batchmean",
        )
        + torch.nn.functional.kl_div(
            second_log_probability,
            first_probability,
            reduction="batchmean",
        )
    )


def discard_rdrop_secondary_gate_loss(model) -> None:
    """Clear the secondary router cache without adding it to the objective.

    ``AddtionalNoisyGate`` accumulates its differentiable loss until
    ``model.gate_loss()`` consumes it.  The R-Drop secondary view deliberately
    excludes gate supervision, but leaving that loss cached would join the
    next minibatch to an already-backwarded graph.  Detaching the consumed
    scalar immediately avoids retaining the secondary graph.  Historical
    gates may return a numeric zero when no router loss is pending.
    """

    pending = model.gate_loss()
    if torch.is_tensor(pending):
        pending.detach()
        return
    if pending is not None and not isinstance(pending, (int, float)):
        raise TypeError(
            "R-Drop secondary gate loss must be a tensor, numeric scalar, or None"
        )


def discard_evaluation_gate_loss(model) -> None:
    """Consume router loss produced by a non-training forward without using it."""

    discard_rdrop_secondary_gate_loss(model)


def evaluation_gate_cache_protocol(args) -> dict[str, Any]:
    enabled = bool(getattr(args, "clear_eval_gate_cache", False))
    return {
        "implementation_revision": "nontraining-router-cache-discard-v1",
        "enabled": enabled,
        "scope": "our_moe validation and evaluation forwards",
        "action": "consume, detach, and discard router loss after each forward",
        "prediction_change": False,
        "historical_default": False,
    }


def joint_head_ensemble_protocol(args) -> dict[str, Any]:
    """Auditable definition of the optional TabM-style joint expert."""

    ensemble_size = int(getattr(args, "joint_head_ensemble_size", 1))
    return {
        "implementation_revision": "tabmmini-joint-probability-ensemble-v1",
        "enabled": ensemble_size > 1,
        "ensemble_size": ensemble_size,
        "shared_component": "all hidden joint-head layers",
        "member_components": "input multiplicative scale plus final linear head",
        "initialization": (
            "member0 input scale=1; other input scales~N(0,1); one legacy "
            "prediction head copied identically to every member"
        ),
        "training_member_loss": "mean_k CE(member_logits_k, label)",
        "training_member_loss_scope": (
            "primary branch-aux joint term only; parser requires a positive "
            "branch_aux_loss_weight when enabled"
        ),
        "classification_aggregation": "mean_k softmax(member_logits_k)",
        "logit_aggregation": False,
        "reliability_branch_count_change": 0,
        "joint_reliability_confidence": "mean member evidence confidence",
        "rdrop_secondary_member_auxiliary": False,
        "inference_checkpoint_ensemble": False,
    }


def low_rank_patch_adapter_protocol(args) -> dict[str, Any]:
    """Auditable definition of the optional tabular patch adaptation."""

    rank = int(getattr(args, "patch_adapter_rank", 0))
    return {
        "implementation_revision": "shared-plus-patch-low-rank-residual-v1",
        "enabled": rank > 0,
        "rank": rank,
        "formula": "shared_linear(x_p) + x_p A_p B_p / sqrt(rank)",
        "initialization": (
            "A_p Kaiming-uniform; B_p exact zero, so initialization exactly "
            "matches the historical shared tokenizer"
        ),
        "additional_parameter_scaling": (
            "num_patches * patch_size * rank + "
            "num_patches * rank * hidden_dim"
        ),
        "encoder_scope": (
            "all preprocessed tabular encoders used by ABCD and ADNI; the "
            "ADNI raw-volume 3D CNN path is unchanged"
        ),
        "selection_scope": "training and validation only",
        "labels_used_by_encoder": False,
        "test_data_used": False,
        "historical_default_rank": 0,
    }


def more_fewer_rank_protocol(args) -> dict[str, Any]:
    """Auditable SimMLM MoFe term on the existing artificial-drop view."""

    configured = float(getattr(args, "more_fewer_rank_loss_weight", 0.0))
    effective = more_fewer_rank_effective_weight(args)
    return {
        "implementation_revision": "simmlm-mofe-original-formula-v1",
        "enabled": effective > 0,
        "loss_weight": configured,
        "configured_control_weight": configured,
        "effective_contribution_weight": effective,
        "reduced_view_forward_retained": configured > 0,
        "formula": "mean_strict max(0, CE_more_i - CE_fewer_i)",
        "margin": 0.0,
        "strict_subset": (
            "fewer observed modalities are a strict subset of the same "
            "sample's more-view observed modalities"
        ),
        "view_source": "existing training-only artificial modality dropout",
        "criterion": (
            "same class weights, label smoothing, and optional training-prior "
            "logit adjustment as the configured task CE"
        ),
        "gradient_scope": "both more and fewer logits; no detach",
        "empty_subset": "differentiable exact zero",
        "inference_change": False,
    }


def dual_local_boundary_residual_protocol(args) -> dict[str, Any]:
    """Auditable fixed first-revision DLBR definition."""

    weight = float(
        getattr(args, "dual_local_boundary_loss_weight", 0.0)
    )
    return {
        "implementation_revision": "dual-local-boundary-residual-v1",
        "enabled": weight > 0,
        "loss_weight": weight,
        "scope": (
            "two independent class-1-vs-0 and class-1-vs-2 residual heads "
            "after the final reliability-fused logits; every train and "
            "inference view"
        ),
        "feature_source": "concatenated pooled multimodal features",
        "feature_normalization": (
            "LayerNorm(elementwise_affine=False)"
        ),
        "head_definition": "two independent Linear(flat_dim,1,bias=False)",
        "head_initialization": "exact zero weights",
        "residual_formula": "r_1c=0.5*tanh(head_1c(normalized_features))",
        "residual_cap": 0.5,
        "logit_delta_formula": (
            "delta0=(-2*r10+r12)/3; delta1=(r10+r12)/3; "
            "delta2=(r10-2*r12)/3"
        ),
        "delta_property": (
            "unique minimum-L2 zero-sum correction with final_gap_10="
            "base_gap_10+r10 and final_gap_12=base_gap_12+r12"
        ),
        "training_task_logits": "base_logits+delta",
        "local_loss": (
            "L_1c=0.5*mean_y1 softplus(-(stopgrad(base_gap_1c)+r_1c)) + "
            "0.5*mean_yc softplus(stopgrad(base_gap_1c)+r_1c)"
        ),
        "class1_vs_0_weight": 2.0 / 3.0,
        "class1_vs_2_weight": 1.0 / 3.0,
        "missing_boundary_rule": (
            "renormalize the remaining active boundary to full scale; "
            "differentiable zero when neither pair is present"
        ),
        "auxiliary_base_gap_gradient": "detached",
        "prediction_rule": "raw softmax argmax of DLBR-adjusted logits",
        "inference_change": weight > 0,
        "rdrop_scope": (
            "local auxiliary once on primary clean view; primary and secondary "
            "CE/KL both use DLBR-adjusted logits"
        ),
        "sam_replay": "complete DLBR-adjusted objective is recomputed",
        "fixed_first_revision_mutual_exclusions": [
            "ordinal fusion or auxiliary",
            "class-1 auxiliary",
            "legacy dual-boundary ranking",
            "hard-CVaR dual-boundary ranking",
            "MORE tail adapter",
            "masked branch TCL",
            "TBFD",
        ],
        "compatible_with": [
            "supervised contrastive learning",
            "SimMLM more-fewer ranking",
            "cross-fitted external teacher",
        ],
        "default_off_module_or_rng_change": False,
    }


def presentation_axis_residual_protocol(args) -> dict[str, Any]:
    """Auditable IA/HI presentation-axis residual definition."""

    weight = float(getattr(args, "presentation_axis_loss_weight", 0.0))
    cap = float(getattr(args, "presentation_axis_residual_cap", 0.5))
    return {
        "implementation_revision": "adhd-presentation-ia-hi-residual-v1",
        "enabled": weight > 0,
        "loss_weight": weight,
        "scope": (
            "IA/HI residual after the final reliability-fused logits; every "
            "our_moe train and inference view"
        ),
        "feature_source": "concatenated pooled multimodal token features",
        "feature_normalization": "LayerNorm(elementwise_affine=False)",
        "head_definition": "Linear(hidden_dim*num_modalities,2,bias=False)",
        "abcd_four_modality_head_definition": "Linear(4*hidden_dim,2,bias=False)",
        "head_initialization": "exact zero weights",
        "axis_order": ["IA", "HI"],
        "axis_targets": {
            "IA": "positive for labels {0,2}; negative for label {1}",
            "HI": "positive for labels {1,2}; negative for label {0}",
        },
        "raw_axis_logits": "head(normalized_features)",
        "residual_formula": "r_axis=cap*tanh(raw_axis_logit)",
        "residual_cap": cap,
        "logit_delta_formula": (
            "delta0=(rIA-2*rHI)/3; delta1=(-2*rIA+rHI)/3; "
            "delta2=(rIA+rHI)/3"
        ),
        "delta_property": (
            "zero-sum per sample; IA alone raises labels 0/2 relative to 1; "
            "HI alone raises labels 1/2 relative to 0; joint evidence raises 2"
        ),
        "training_task_logits": "base_logits+delta",
        "local_loss": (
            "per axis, balanced logistic mean: 0.5*mean_positive "
            "softplus(-raw)+0.5*mean_negative softplus(raw)"
        ),
        "axis_weighting": "equal IA/HI mean",
        "missing_side_rule": (
            "when one positive/negative side is absent, retain the existing "
            "side mean at full scale; never skip the axis"
        ),
        "prediction_rule": "raw softmax argmax of residual-adjusted logits",
        "inference_change": weight > 0,
        "mutual_exclusions": [
            "ordinal fusion or auxiliary",
            "class-1 auxiliary",
            "DLBR",
        ],
        "compatible_with": [
            "low-rank patch adapter",
            "conditional generators",
            "token-importance pooling",
        ],
        "default_off_module_or_rng_change": False,
    }


def missing_capacity_residual_protocol(args) -> dict[str, Any]:
    """Auditable current-view-missing pooled-capacity residual definition."""

    width = int(getattr(args, "missing_capacity_residual_width", 0))
    return {
        "implementation_revision": "current-view-missing-capacity-residual-v1",
        "enabled": width > 0,
        "width": width,
        "application_position": (
            "after presentation-axis final logits and before task softmax/argmax"
        ),
        "feature_source": "concatenated pooled four-modality features",
        "feature_dimension": "4*hidden_dim",
        "feature_normalization": "LayerNorm(elementwise_affine=False)",
        "head_definition": (
            "Linear(4*hidden_dim,width,bias=False)->GELU->"
            "Linear(width,3,bias=False)"
        ),
        "final_projection_initialization": "exact zero weights",
        "current_view_missing_rule": "~observed_mask.all(dim=1)",
        "clean_view_semantics": (
            "on the clean primary/validation/deployment view this equals the "
            "frozen natural/effective missingness mask"
        ),
        "artificial_drop_semantics": (
            "the existing training-only artificially dropped view also enables "
            "the residual for rows missing in that current view"
        ),
        "complete_row_rule": "delta is exact zero",
        "complete_row_head_gradient": "exact zero from complete-row task terms",
        "training_signal": (
            "all already-configured objective terms that consume final logits; "
            "this includes primary/drop CE and, when enabled, distillation, "
            "ranking, R-Drop, and SAM objective replay"
        ),
        "new_residual_specific_loss": False,
        "residual_cap": None,
        "dropout": False,
        "bias_parameters": False,
        "initial_logit_change": "exact zero for every row",
        "parameter_count_formula": "(4*hidden_dim)*width + width*3",
        "abcd_h64_width64_parameter_count": 16576,
        "abcd_h64_width64_gpu_parameter_audit": {
            "anchor_total": 795133,
            "residual_head": 16576,
            "candidate_total": 811709,
            "cpu_fmoe_fallback_parameters_excluded": True,
        },
        "construction_rng": (
            "save and restore CPU RNG around the complete optional head"
        ),
        "constraints": {
            "model": "our_moe",
            "num_classes": 3,
            "num_modalities": 4,
            "width": "non-negative integer",
        },
        "default_off_module_or_rng_change": False,
    }


def more_tail_adapter_protocol(args) -> dict[str, Any]:
    """Auditable definition of the MORE-inspired fused-logit adapter."""

    rank = int(getattr(args, "more_tail_rank", 0))
    peak_amplitude = float(
        getattr(args, "more_tail_peak_amplitude", 0.0)
    )
    class_prior = getattr(args, "more_tail_class_prior", None)
    return {
        "implementation_revision": "more-inspired-fused-logit-tail-v1",
        "enabled": rank > 0,
        "scope": (
            "classifier-level adaptation after the final reliability/ordinal "
            "fused logits; all Ours train and inference views"
        ),
        "classifier_decomposition": "W = W_g + B A",
        "logit_formula": (
            "tail_logits=B(A(flat_pooled_features)); "
            "final_logits=base_logits+tail_logits"
        ),
        "rank": rank,
        "bias": False,
        "a_initialization": "torch.nn.Linear default Kaiming-uniform initialization",
        "b_initialization": "exact zeros",
        "loss_formula": (
            "L_MORE=mean_i pi[y_i] * "
            "||final_logits_i-base_logits_i||_2^2"
        ),
        "peak_amplitude": peak_amplitude,
        "effective_weight_formula": "alpha=A*sin(pi*tau/T)",
        "progress_definition": (
            "tau is the one-indexed global optimizer-batch step and T is the "
            "planned total number of training batches across all epochs"
        ),
        "rdrop_scope": (
            "L_MORE is evaluated once on the primary clean view; both primary "
            "and secondary CE/KL predictions use the adapted final logits"
        ),
        "sam_replay": (
            "the first and perturbed passes reuse the same global batch "
            "progress and effective alpha"
        ),
        "class_prior_source": "empirical labels of the current training split only",
        "class_prior": (
            [float(value) for value in class_prior]
            if class_prior is not None
            else None
        ),
        "validation_or_test_labels_used_for_prior": False,
        "paper_relationship": (
            "MORE-inspired W_g+BA residual, prior-weighted magnitude, and sine "
            "schedule adapted to fused multiclass logits; not a claim of a "
            "complete reproduction of the paper method"
        ),
        "default_off_module_or_rng_change": False,
    }


def cross_fitted_tree_teacher_protocol(args) -> dict[str, Any]:
    """Auditable contract for external CatBoost OOF soft-target supervision."""

    weight = float(getattr(args, "tree_teacher_distill_weight", 0.0))
    enabled = weight > 0
    return {
        "implementation_revision": "exact-train-id-catboost-oof-kl-v1",
        "enabled": enabled,
        "loss_weight": weight,
        "temperature": float(getattr(args, "tree_teacher_temperature", 2.0)),
        "teacher_scope": (
            "exact current runner training IDs only; one out-of-fold probability "
            "row per training ID"
        ),
        "held_fold_label_policy": (
            "held families and labels must not be used for fitting, preprocessing, "
            "early stopping, iteration selection, calibration, or hyperparameter selection"
        ),
        "artifact_path": (
            getattr(args, "tree_teacher_npz", None) if enabled else None
        ),
        "artifact_sha256": (
            getattr(args, "tree_teacher_npz_sha256", None) if enabled else None
        ),
        "artifact_schema": {
            "version": TREE_TEACHER_NPZ_SCHEMA_VERSION,
            "dataset": TREE_TEACHER_DATASET,
            "task": TREE_TEACHER_TASK,
            "fold_count": TREE_TEACHER_FOLD_COUNT,
        },
        "required_npz_arrays": sorted(TREE_TEACHER_NPZ_KEYS),
        "integrity_checks": (
            "one immutable byte buffer for SHA-256 and allow_pickle=False load; "
            "exact keys/schema; unique canonical Unicode IDs; exact ordered "
            "training-ID coverage; zero validation/test-ID intersection; ordered "
            "ID, ID-label, and task-binding hashes; ordered classes; finite "
            "simplex probabilities; exactly five non-empty OOF folds"
        ),
        "loss": "mean per-sample KL(temperature-adjusted teacher || student) * T^2",
        "student_logits": "final fused Ours logits",
        "teacher_gradient": False,
        "rdrop_secondary_teacher_loss": False,
        "sam_replay_teacher_loss": True,
        "validation_or_test_teacher_load": False,
        "inference_dependency": False,
        "applicability_guardrail": (
            "the artifact ID set must exactly equal the current runner training "
            "universe; a full-8430 teacher cannot be sliced for family-audit "
            "inner training subsets"
        ),
        "historical_default_weight": 0.0,
    }


def cross_fitted_external_teacher_protocol(args) -> dict[str, Any]:
    """Auditable schema-v2 generic OOF teacher and trust-gated loss contract."""

    weight = float(getattr(args, "external_teacher_distill_weight", 0.0))
    enabled = weight > 0
    task = (
        getattr(args, "external_teacher_task", EXTERNAL_TEACHER_TASK)
        if enabled
        else EXTERNAL_TEACHER_TASK
    )
    if task not in EXTERNAL_TEACHER_TASK_KIND_BINDINGS:
        raise ValueError(f"Unsupported external-teacher task: {task!r}")
    allowed_teacher_kinds = EXTERNAL_TEACHER_TASK_KIND_BINDINGS[task]
    teacher_kind = getattr(args, "external_teacher_kind", None) if enabled else None
    if enabled and teacher_kind not in allowed_teacher_kinds:
        raise ValueError(
            "External-teacher task/kind binding is inconsistent: "
            f"task={task!r}, teacher_kind={teacher_kind!r}"
        )
    temporal_task = task == EXTERNAL_TEACHER_TEMPORAL_TASK
    trust_mode = getattr(args, "external_teacher_trust_mode", "correct_only")
    class_compensation = getattr(
        args, "external_teacher_class_compensation", "none"
    )
    return {
        "implementation_revision": (
            "natural-status-exact-train-id-oof-trust-kl-v1"
            if temporal_task
            else "generic-exact-train-id-oof-trust-kl-v2"
        ),
        "enabled": enabled,
        "loss_weight": weight,
        "temperature": float(
            getattr(args, "external_teacher_temperature", 2.0)
        ),
        "teacher_kind": teacher_kind,
        "allowed_teacher_kinds": sorted(allowed_teacher_kinds),
        "teacher_kind_semantics": (
            {
                "natural_status_moepp_anymod": (
                    "natural-status MoE++/AnyMod OOF orchestration with one "
                    "already-cross-fitted probability row per training ID"
                )
            }
            if temporal_task
            else {
                "native_tabm32": "one native TabM-32 family-cross-fitted OOF teacher",
                "catboost": "one CatBoost family-cross-fitted OOF teacher",
                "trusted_tabm32_catboost": (
                    "label-gated heterogeneous TabM-32/CatBoost OOF orchestration; "
                    "every selected probability remains cross-fitted for that row"
                ),
            }
        ),
        "trust_mode": trust_mode,
        "class_compensation": class_compensation,
        "artifact_path": (
            getattr(args, "external_teacher_npz", None) if enabled else None
        ),
        "artifact_sha256": (
            getattr(args, "external_teacher_npz_sha256", None)
            if enabled
            else None
        ),
        "generator_protocol_sha256": (
            getattr(
                args, "external_teacher_generator_protocol_sha256", None
            )
            if enabled
            else None
        ),
        "artifact_schema": {
            "version": EXTERNAL_TEACHER_NPZ_SCHEMA_VERSION,
            "dataset": EXTERNAL_TEACHER_DATASET,
            "task": task,
            "fold_count": EXTERNAL_TEACHER_FOLD_COUNT,
            "teacher_kind_in_task_binding_sha256": True,
        },
        "required_npz_arrays": sorted(EXTERNAL_TEACHER_NPZ_KEYS),
        "teacher_scope": (
            "exact current runner training IDs only; one family-disjoint "
            "out-of-fold probability row per training ID"
        ),
        "held_fold_label_policy": (
            "held families and labels must not be used for fitting, preprocessing, "
            "early stopping, iteration selection, calibration, or hyperparameter "
            "selection; generator_protocol_sha256 binds the external producer audit"
        ),
        "trusted_heterogeneous_label_use": (
            "natural_status_moepp_anymod producer behavior is bound by the "
            "immutable generator_protocol_sha256; this consumer neither selects "
            "a producer nor opens held-validation/test labels"
            if temporal_task
            else (
                "trusted_tabm32_catboost may use a row's training label only to "
                "orchestrate among already-produced, family-cross-fitted OOF TabM-32 "
                "and CatBoost probabilities for that same row; the label must never "
                "enter either producer fit or model selection"
            )
        ),
        "integrity_checks": (
            "one immutable byte buffer for SHA-256 and allow_pickle=False load; "
            "exact 12-key v2 schema; strict teacher_kind; unique canonical Unicode "
            "IDs; exact ordered training-ID coverage; zero validation/test-ID "
            "intersection; ordered ID, ID-label, teacher-kind-aware task-binding, "
            "and producer-protocol hashes; ordered classes; float64 finite simplex "
            "probabilities with hard-prediction-preserving float32 training cast; "
            "exactly five non-empty family OOF folds"
        ),
        "class_recalls_alpha": (
            getattr(args, "external_teacher_class_recalls", None)
            if enabled
            else None
        ),
        "class_compensation_weights": (
            getattr(
                args,
                "external_teacher_class_compensation_weights",
                None,
            )
            if enabled
            else None
        ),
        "correct_teacher_coverage": (
            getattr(args, "external_teacher_correct_coverage", None)
            if enabled
            else None
        ),
        "trust_gate_formula": (
            "g_i = 1[argmax_c p_teacher(i,c) == y_i]"
            if trust_mode == "correct_only"
            else "g_i = 1"
        ),
        "class_recall_formula": (
            "alpha_c = count_i[y_i=c and argmax p_teacher_i=c] / "
            "count_i[y_i=c], computed once on full current-training OOF rows"
        ),
        "class_weight_formula": (
            "w_c = C*sqrt(1-alpha_c)/sum_j sqrt(1-alpha_j); uniformly "
            "perfect teacher uses the symmetric all-one limit"
        ),
        "loss_formula": (
            "sum_i g_i*w_[y_i]*KL(temperature-adjusted detached teacher_i || "
            "student_i)*T^2 / sum_i g_i*w_[y_i]; w_[y_i]=1 when "
            "class_compensation=none"
        ),
        "empty_active_batch": "differentiable exact zero connected to student logits",
        "student_logits": "final fused Ours logits",
        "teacher_gradient": False,
        "rdrop_secondary_teacher_loss": False,
        "sam_replay_teacher_loss": True,
        "sam_replay_targets": "same attached OOF targets and fixed class weights",
        "validation_or_test_teacher_load": False,
        "inference_dependency": False,
        "paper_relationship": (
            "TCL-style correct-teacher trust gating and class-accuracy square-root "
            "compensation adapted to one external cross-fitted OOF teacher; this is "
            "not a claim of reproducing the original TCL multi-expert method"
        ),
        "applicability_guardrail": (
            "the artifact ID set must exactly equal the current runner training "
            "universe; an outer training-universe teacher cannot be sliced for "
            "family-audit inner subsets"
        ),
        "historical_default_weight": 0.0,
    }


def hard_cvar_dual_boundary_protocol(args) -> dict[str, Any]:
    """Auditable definition and epoch schedule for hard boundary CVaR."""

    max_weight = float(
        getattr(args, "hard_cvar_dual_boundary_max_weight", 0.0)
    )
    tail_fraction = float(
        getattr(args, "hard_cvar_dual_boundary_tail_fraction", 0.25)
    )
    margin = float(getattr(args, "hard_cvar_dual_boundary_margin", 0.20))
    class1_vs_0_weight = float(
        getattr(args, "hard_cvar_dual_boundary_10_weight", 0.65)
    )
    start_epoch = int(
        getattr(args, "hard_cvar_dual_boundary_start_epoch", 5)
    )
    ramp_epochs = int(
        getattr(args, "hard_cvar_dual_boundary_ramp_epochs", 10)
    )
    return {
        "implementation_revision": "dual-boundary-hard-cvar-top-pair-tail-v1",
        "enabled": max_weight > 0,
        "max_loss_weight": max_weight,
        "tail_fraction": tail_fraction,
        "tail_count": "ceil(tail_fraction * active_boundary_pair_count)",
        "tail_selection": "largest pairwise softplus losses independently per boundary",
        "margin": margin,
        "class1_vs_0_weight": class1_vs_0_weight,
        "class1_vs_2_weight": 1.0 - class1_vs_0_weight,
        "missing_boundary_rule": "renormalize the remaining active boundary to full scale",
        "scope": "primary full-view fused logits only",
        "start_epoch": start_epoch,
        "ramp_epochs": ramp_epochs,
        "schedule": (
            "0 before start_epoch; max_weight*min(1,(epoch-start_epoch+1)/"
            "ramp_epochs) from start_epoch; zero ramp means immediate max"
        ),
        "inference_change": False,
    }


def rdrop_training_protocol(args) -> dict[str, Any]:
    """Auditable definition of the opt-in clean-view R-Drop path."""

    weight = float(getattr(args, "rdrop_loss_weight", 0.0))
    return {
        "implementation_revision": "end-to-end-clean-view-symmetric-kl-v1",
        "enabled": weight > 0,
        "loss_weight": weight,
        "temperature": 1.0,
        "task_loss": (
            "equal mean of class-weighted CE from the primary and secondary "
            "full-view fused logits"
        ),
        "consistency_loss": "0.5*(KL(p1||p2)+KL(p2||p1))",
        "view_definition": (
            "same raw minibatch, observed-modality mask, and expert indices; "
            "independent native encoder/model stochastic realization"
        ),
        "primary_view_scope": "complete unchanged our_moe objective",
        "secondary_view_scope": "full fused logits only",
        "secondary_reconstruction": False,
        "secondary_modality_dropout": False,
        "secondary_branch_or_gate_auxiliary": False,
        "secondary_gate_cache": (
            "consume, detach, and discard immediately after the secondary "
            "forward so no router graph crosses minibatches"
        ),
        "execution_order": (
            "complete the primary objective, including its single modality-"
            "dropped pass, before the secondary clean pass"
        ),
        "sam_compatible": False,
        "inference_change": False,
        "new_model_parameters": False,
    }


class TorchRNGSnapshot(NamedTuple):
    """Exact PyTorch RNG state needed to replay one stochastic training batch."""

    cpu_state: torch.Tensor
    cuda_states: tuple[torch.Tensor, ...] | None


def sam_training_protocol(args) -> dict[str, Any]:
    """Auditable definition of the opt-in standard-SAM training path."""

    rho = float(getattr(args, "sam_rho", 0.0))
    return {
        "implementation_revision": "full-objective-global-l2-v1",
        "enabled": rho > 0,
        "rho": rho,
        "adaptive": False,
        "objective_scope": "complete unchanged our_moe objective",
        "parameter_scope": (
            "all optimizer parameters with gradients, jointly normalized "
            "across model and encoders"
        ),
        "replay_rule": (
            "same batch with restored CPU and all-CUDA RNG; re-run encoders, "
            "model, and complete objective"
        ),
        "update_rule": (
            "restore exact unperturbed parameters before one gradient clip, "
            "base optimizer step, EMA update, and scheduler advance"
        ),
        "training_log_source": "first unperturbed complete-objective pass",
        "inference_change": False,
        "new_model_parameters": False,
    }


def capture_torch_rng_state() -> TorchRNGSnapshot:
    """Capture the CPU generator and every visible CUDA generator."""

    cuda_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )
    return TorchRNGSnapshot(torch.get_rng_state().clone(), cuda_states)


def restore_torch_rng_state(snapshot: TorchRNGSnapshot) -> None:
    """Restore a state produced by :func:`capture_torch_rng_state`."""

    if not isinstance(snapshot, TorchRNGSnapshot):
        raise TypeError("snapshot must be a TorchRNGSnapshot")
    torch.set_rng_state(snapshot.cpu_state)
    if snapshot.cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore captured CUDA RNG state without CUDA")
        current_device_count = torch.cuda.device_count()
        if current_device_count != len(snapshot.cuda_states):
            raise RuntimeError(
                "CUDA device count changed between SAM RNG capture and restore: "
                f"{len(snapshot.cuda_states)} -> {current_device_count}"
            )
        torch.cuda.set_rng_state_all(list(snapshot.cuda_states))


def optimizer_unique_parameters(
    optimizer: torch.optim.Optimizer,
) -> list[torch.nn.Parameter]:
    """Return optimizer parameters once, preserving param-group order."""

    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError("Optimizer param groups must contain Parameters")
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            parameters.append(parameter)
    return parameters


@contextmanager
def global_l2_sam_perturbation(
    optimizer: torch.optim.Optimizer,
    rho: float,
    epsilon: float = 1e-12,
):
    """Temporarily apply the standard global-L2 SAM ascent perturbation.

    The context snapshots exact parameter values before perturbing them and
    restores those values in ``finally``.  Consequently a failed replay cannot
    leave the live model at the perturbed point.  Parameters without a first-
    pass gradient are left unchanged, matching standard SAM.
    """

    rho = float(rho)
    epsilon = float(epsilon)
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("SAM rho must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("SAM epsilon must be finite and positive")

    parameter_gradients: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    gradient_norms: list[torch.Tensor] = []
    for parameter in optimizer_unique_parameters(optimizer):
        if not parameter.requires_grad or parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if gradient.is_sparse:
            gradient = gradient.to_dense()
        parameter_gradients.append((parameter, gradient))
        gradient_norms.append(torch.linalg.vector_norm(gradient, ord=2))
    if not parameter_gradients:
        raise RuntimeError("SAM first pass produced no optimizer gradients")

    reference_device = gradient_norms[0].device
    reference_dtype = gradient_norms[0].dtype
    global_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                norm.to(device=reference_device, dtype=reference_dtype)
                for norm in gradient_norms
            ]
        ),
        ord=2,
    )
    if not bool(torch.isfinite(global_norm)):
        raise FloatingPointError("Non-finite global SAM gradient norm")
    scale = rho / (global_norm + epsilon)

    originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for parameter, gradient in parameter_gradients:
                original = parameter.detach().clone()
                originals.append((parameter, original))
                parameter.add_(
                    gradient
                    * scale.to(device=parameter.device, dtype=parameter.dtype)
                )
        yield global_norm.detach()
    finally:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)


def run_epoch(
    args,
    loader,
    encoders,
    modality_dict,
    model,
    criterion,
    device,
    optimizer=None,
    lr_schedule: StepwiseCosineSchedule | None = None,
    ema: StepwiseStateEMA | None = None,
    epoch: int | None = None,
    training_step_offset: int | None = None,
    total_training_steps: int | None = None,
    masked_branch_tcl_state: MaskedBranchTCLClassAccuracyEMA | None = None,
    trusted_branch_fusion_state: MaskedBranchTCLClassAccuracyEMA | None = None,
    empirical_missing_pattern_replay_state: (
        EmpiricalMissingPatternReplayState | None
    ) = None,
):
    training = optimizer is not None
    model.train(training)
    for encoder in encoders.values():
        encoder.train(training)

    losses: list[float] = []
    ce_losses: list[float] = []
    aux_losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_observed: list[np.ndarray] = []
    all_combinations: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    more_tail_enabled = int(getattr(args, "more_tail_rank", 0)) > 0
    masked_branch_tcl_enabled = training and float(
        getattr(args, "masked_branch_tcl_loss_weight", 0.0)
    ) > 0.0
    trusted_branch_fusion_enabled = training and float(
        getattr(args, "trusted_branch_fusion_distill_weight", 0.0)
    ) > 0.0
    empirical_replay_configured = bool(
        getattr(args, "empirical_missing_pattern_replay", False)
    )
    empirical_replay_enabled = training and empirical_replay_configured
    if empirical_replay_configured:
        if args.model != "our_moe" or args.data != "abcd":
            raise ValueError(
                "Empirical replay is only supported for ABCD our_moe"
            )
        if not bool(getattr(args, "recompute_dropped_combination", False)):
            raise ValueError("Empirical replay requires recomputed combinations")
        if bool(getattr(args, "generator_only_task_grad", False)):
            raise ValueError(
                "Empirical replay and generator-only task gradients are mutually exclusive"
            )
        if float(getattr(args, "sam_rho", 0.0)) != 0.0:
            raise ValueError("Empirical replay v1 does not support SAM")
        if float(getattr(args, "rdrop_loss_weight", 0.0)) != 0.0:
            raise ValueError("Empirical replay v1 does not support R-Drop")
        empirical_missing_pattern_replay_active_objectives(args)
    if empirical_replay_enabled and not isinstance(
        empirical_missing_pattern_replay_state,
        EmpiricalMissingPatternReplayState,
    ):
        raise RuntimeError(
            "Enabled empirical replay training requires persistent explicit state"
        )
    if not empirical_replay_enabled and empirical_missing_pattern_replay_state is not None:
        raise RuntimeError("Empirical replay state was supplied outside enabled training")
    empirical_replay_objective_kwargs = (
        {
            "empirical_missing_pattern_replay_state": (
                empirical_missing_pattern_replay_state
            )
        }
        if empirical_replay_enabled
        else {}
    )
    if masked_branch_tcl_enabled and trusted_branch_fusion_enabled:
        raise ValueError(
            "masked branch TCL and TBFD cannot be enabled together"
        )
    if masked_branch_tcl_enabled:
        if args.model != "our_moe":
            raise ValueError(
                "masked branch TCL is only supported for our_moe training"
            )
        if type(epoch) is not int or epoch < 1:
            raise ValueError(
                "masked branch TCL training requires a positive epoch"
            )
        if masked_branch_tcl_state is None:
            raise RuntimeError(
                "enabled masked branch TCL requires explicit epoch state"
            )
        masked_branch_tcl_state.begin_epoch(epoch)
    if trusted_branch_fusion_enabled:
        if args.model != "our_moe":
            raise ValueError("TBFD is only supported for our_moe training")
        if type(epoch) is not int or epoch < 1:
            raise ValueError("TBFD training requires a positive epoch")
        if trusted_branch_fusion_state is None:
            raise RuntimeError("enabled TBFD requires explicit epoch state")
        trusted_branch_fusion_state.begin_epoch(epoch)
    if training and more_tail_enabled:
        if (
            type(training_step_offset) is not int
            or training_step_offset < 0
            or type(total_training_steps) is not int
            or total_training_steps <= 0
        ):
            raise ValueError(
                "enabled MORE tail training requires valid global step bounds"
            )
        if training_step_offset + len(loader) > total_training_steps:
            raise ValueError("MORE tail epoch steps exceed the planned schedule")
    with context:
        for batch_index, batch in enumerate(loader):
            if len(batch) == 4:
                batch_samples, labels, modality_comb, observed = batch
                tree_teacher_probabilities = None
            elif len(batch) == 5:
                (
                    batch_samples,
                    labels,
                    modality_comb,
                    observed,
                    tree_teacher_probabilities,
                ) = batch
            else:
                raise ValueError(
                    "Loader batches must have four ordinary fields or five "
                    "fields including cross-fitted teacher probabilities"
                )
            tree_teacher_weight = float(
                getattr(args, "tree_teacher_distill_weight", 0.0)
            )
            external_teacher_weight = float(
                getattr(args, "external_teacher_distill_weight", 0.0)
            )
            if tree_teacher_weight > 0 and external_teacher_weight > 0:
                raise RuntimeError(
                    "v1 tree-teacher and v2 external-teacher training paths "
                    "are mutually exclusive"
                )
            if training and (
                tree_teacher_weight > 0 or external_teacher_weight > 0
            ):
                if tree_teacher_probabilities is None:
                    raise RuntimeError(
                        "enabled cross-fitted teacher distillation requires targets in "
                        "every training batch"
                    )
                tree_teacher_probabilities = tree_teacher_probabilities.to(
                    device, non_blocking=True
                )
            elif tree_teacher_probabilities is not None:
                raise RuntimeError(
                    "a loader supplied cross-fitted teacher targets outside "
                    "enabled training"
                )
            tree_teacher_objective_kwargs = (
                {"tree_teacher_probabilities": tree_teacher_probabilities}
                if training and tree_teacher_weight > 0
                else {}
            )
            external_teacher_objective_kwargs = (
                {
                    "external_teacher_probabilities": (
                        tree_teacher_probabilities
                    )
                }
                if training and external_teacher_weight > 0
                else {}
            )
            more_tail_objective_kwargs = {}
            if training and more_tail_enabled:
                global_training_step = training_step_offset + batch_index + 1
                more_tail_objective_kwargs = {
                    "training_progress": (
                        global_training_step / float(total_training_steps)
                    )
                }
            if training and lr_schedule is not None:
                lr_schedule.set_current_lr()
            sam_rho = float(getattr(args, "sam_rho", 0.0)) if training else 0.0
            rdrop_weight = (
                float(getattr(args, "rdrop_loss_weight", 0.0))
                if training
                else 0.0
            )
            if training and (
                not math.isfinite(rdrop_weight) or rdrop_weight < 0
            ):
                raise ValueError(
                    "R-Drop loss weight must be finite and non-negative"
                )
            if sam_rho > 0 and args.model != "our_moe":
                raise ValueError("SAM is only supported for our_moe training")
            if rdrop_weight > 0 and args.model != "our_moe":
                raise ValueError("R-Drop is only supported for our_moe training")
            if sam_rho > 0 and rdrop_weight > 0:
                raise ValueError("R-Drop and SAM cannot be enabled together")
            sam_rng_snapshot = (
                capture_torch_rng_state() if sam_rho > 0 else None
            )
            labels = labels.to(device, non_blocking=True)
            modality_comb = modality_comb.to(device, non_blocking=True)
            observed = observed.to(device, non_blocking=True)
            tokens, observed_mask = encode_batch(
                args, batch_samples, observed, encoders, modality_dict, device
            )
            forward_observed_mask = observed_mask
            if (
                training
                and args.model == "transformer_concat"
                and args.modality_dropout_prob > 0
            ):
                forward_observed_mask = modality_dropout_mask(
                    observed_mask, args.modality_dropout_prob
                )
            if args.model == "flex_moe":
                output = model(
                    *tokens,
                    observed_mask=observed_mask,
                    modality_comb=modality_comb,
                    return_aux=training,
                )
            elif args.model == "our_moe":
                if (
                    training
                    and int(getattr(args, "joint_head_ensemble_size", 1)) > 1
                ):
                    output = model(
                        *tokens,
                        observed_mask=observed_mask,
                        expert_indices=modality_comb,
                        return_importance=False,
                        return_recon_loss=True,
                        return_joint_ensemble_diagnostics=True,
                    )
                else:
                    # Preserve the exact historical call and return dictionary
                    # when the structural ensemble is disabled.
                    output = model(
                        *tokens,
                        observed_mask=observed_mask,
                        expert_indices=modality_comb,
                        return_importance=False,
                        return_recon_loss=training,
                    )
            else:
                output = model(
                    *tokens,
                    observed_mask=forward_observed_mask,
                    return_aux=training,
                )

            if (
                not training
                and args.model == "our_moe"
                and bool(getattr(args, "clear_eval_gate_cache", False))
            ):
                discard_evaluation_gate_loss(model)

            if training:
                masked_branch_tcl_objective_kwargs = {}
                trusted_branch_fusion_objective_kwargs = {}
                if masked_branch_tcl_enabled:
                    branch_logits_for_tcl = output.get("branch_logits")
                    branch_mask_for_tcl = output.get("branch_mask")
                    if (
                        branch_logits_for_tcl is None
                        or branch_mask_for_tcl is None
                    ):
                        raise RuntimeError(
                            "enabled masked branch TCL requires branch logits and mask"
                        )
                    # This is the sole statistic update site.  R-Drop secondary
                    # and SAM replay outputs never enter this collector.
                    masked_branch_tcl_state.collect_clean_primary(
                        branch_logits_for_tcl,
                        branch_mask_for_tcl,
                        labels,
                    )
                    if epoch >= int(
                        getattr(args, "masked_branch_tcl_start_epoch", 6)
                    ):
                        masked_branch_tcl_objective_kwargs = {
                            "masked_branch_tcl_context": (
                                masked_branch_tcl_state.replay_context(
                                    branch_logits_for_tcl,
                                    branch_mask_for_tcl,
                                    labels,
                                )
                            )
                        }
                if trusted_branch_fusion_enabled:
                    branch_logits_for_fusion = output.get("branch_logits")
                    branch_mask_for_fusion = output.get("branch_mask")
                    if (
                        branch_logits_for_fusion is None
                        or branch_mask_for_fusion is None
                    ):
                        raise RuntimeError(
                            "enabled TBFD requires branch logits and mask"
                        )
                    # As with masked TCL, this is clean-primary only; R-Drop
                    # secondary and SAM replay never update epoch statistics.
                    trusted_branch_fusion_state.collect_clean_primary(
                        branch_logits_for_fusion,
                        branch_mask_for_fusion,
                        labels,
                    )
                    if epoch >= int(
                        getattr(args, "trusted_branch_fusion_start_epoch", 6)
                    ):
                        trusted_branch_fusion_objective_kwargs = {
                            "trusted_branch_fusion_context": (
                                trusted_branch_fusion_state.replay_context(
                                    branch_logits_for_fusion,
                                    branch_mask_for_fusion,
                                    labels,
                                )
                            )
                        }
                optimizer.zero_grad(set_to_none=True)
                if args.model == "our_moe":
                    if (
                        float(
                            getattr(
                                args,
                                "hard_cvar_dual_boundary_max_weight",
                                0.0,
                            )
                        )
                        > 0
                    ):
                        loss, ce, auxiliary = our_moe_objective(
                            args,
                            model,
                            output,
                            tokens,
                            observed_mask,
                            modality_comb,
                            labels,
                            criterion,
                            epoch=epoch,
                            **tree_teacher_objective_kwargs,
                            **external_teacher_objective_kwargs,
                            **more_tail_objective_kwargs,
                            **masked_branch_tcl_objective_kwargs,
                            **trusted_branch_fusion_objective_kwargs,
                            **empirical_replay_objective_kwargs,
                        )
                    else:
                        if (
                            masked_branch_tcl_enabled
                            or trusted_branch_fusion_enabled
                        ):
                            loss, ce, auxiliary = our_moe_objective(
                                args,
                                model,
                                output,
                                tokens,
                                observed_mask,
                                modality_comb,
                                labels,
                                criterion,
                                epoch=epoch,
                                **tree_teacher_objective_kwargs,
                                **external_teacher_objective_kwargs,
                                **more_tail_objective_kwargs,
                                **masked_branch_tcl_objective_kwargs,
                                **trusted_branch_fusion_objective_kwargs,
                                **empirical_replay_objective_kwargs,
                            )
                        else:
                            # Keep the disabled path's exact historical call.
                            loss, ce, auxiliary = our_moe_objective(
                                args,
                                model,
                                output,
                                tokens,
                                observed_mask,
                                modality_comb,
                                labels,
                                criterion,
                                **tree_teacher_objective_kwargs,
                                **external_teacher_objective_kwargs,
                                **more_tail_objective_kwargs,
                                **empirical_replay_objective_kwargs,
                            )
                    if rdrop_weight > 0:
                        secondary_tokens, secondary_observed_mask = encode_batch(
                            args,
                            batch_samples,
                            observed,
                            encoders,
                            modality_dict,
                            device,
                        )
                        if not torch.equal(
                            secondary_observed_mask, observed_mask
                        ):
                            raise RuntimeError(
                                "R-Drop clean views produced different observed masks"
                            )
                        secondary_output = model(
                            *secondary_tokens,
                            observed_mask=secondary_observed_mask,
                            expert_indices=modality_comb,
                            return_importance=False,
                            return_recon_loss=False,
                        )
                        discard_rdrop_secondary_gate_loss(model)
                        secondary_ce = criterion(
                            secondary_output["logits"], labels
                        )
                        consistency = rdrop_symmetric_kl_loss(
                            output["logits"], secondary_output["logits"]
                        )
                        averaged_ce = 0.5 * (ce + secondary_ce)
                        weighted_consistency = rdrop_weight * consistency
                        ce = averaged_ce
                        auxiliary = auxiliary + weighted_consistency
                        loss = ce + auxiliary
                else:
                    loss, ce, auxiliary = objective(args, output, labels, criterion)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: {float(loss.detach())}")
                loss.backward()
                if sam_rho > 0:
                    if sam_rng_snapshot is None:
                        raise RuntimeError("SAM RNG snapshot was not captured")
                    with global_l2_sam_perturbation(optimizer, sam_rho):
                        optimizer.zero_grad(set_to_none=True)
                        restore_torch_rng_state(sam_rng_snapshot)
                        replay_tokens, replay_observed_mask = encode_batch(
                            args,
                            batch_samples,
                            observed,
                            encoders,
                            modality_dict,
                            device,
                        )
                        replay_output = model(
                            *replay_tokens,
                            observed_mask=replay_observed_mask,
                            expert_indices=modality_comb,
                            return_importance=False,
                            return_recon_loss=True,
                            **(
                                {"return_joint_ensemble_diagnostics": True}
                                if int(
                                    getattr(
                                        args, "joint_head_ensemble_size", 1
                                    )
                                )
                                > 1
                                else {}
                            ),
                        )
                        if (
                            float(
                                getattr(
                                    args,
                                    "hard_cvar_dual_boundary_max_weight",
                                    0.0,
                                )
                            )
                            > 0
                        ):
                            replay_loss, _, _ = our_moe_objective(
                                args,
                                model,
                                replay_output,
                                replay_tokens,
                                replay_observed_mask,
                                modality_comb,
                                labels,
                                criterion,
                                epoch=epoch,
                                **tree_teacher_objective_kwargs,
                                **external_teacher_objective_kwargs,
                                **more_tail_objective_kwargs,
                                **masked_branch_tcl_objective_kwargs,
                                **trusted_branch_fusion_objective_kwargs,
                                **empirical_replay_objective_kwargs,
                            )
                        else:
                            if (
                                masked_branch_tcl_enabled
                                or trusted_branch_fusion_enabled
                            ):
                                replay_loss, _, _ = our_moe_objective(
                                    args,
                                    model,
                                    replay_output,
                                    replay_tokens,
                                    replay_observed_mask,
                                    modality_comb,
                                    labels,
                                    criterion,
                                    epoch=epoch,
                                    **tree_teacher_objective_kwargs,
                                    **external_teacher_objective_kwargs,
                                    **more_tail_objective_kwargs,
                                    **masked_branch_tcl_objective_kwargs,
                                    **trusted_branch_fusion_objective_kwargs,
                                    **empirical_replay_objective_kwargs,
                                )
                            else:
                                replay_loss, _, _ = our_moe_objective(
                                    args,
                                    model,
                                    replay_output,
                                    replay_tokens,
                                    replay_observed_mask,
                                    modality_comb,
                                    labels,
                                    criterion,
                                    **tree_teacher_objective_kwargs,
                                    **external_teacher_objective_kwargs,
                                    **more_tail_objective_kwargs,
                                    **empirical_replay_objective_kwargs,
                                )
                        if not torch.isfinite(replay_loss):
                            raise FloatingPointError(
                                "Non-finite perturbed SAM loss: "
                                f"{float(replay_loss.detach())}"
                            )
                        replay_loss.backward()
                if args.grad_clip > 0:
                    parameters = list(model.parameters()) + [
                        parameter for encoder in encoders.values() for parameter in encoder.parameters()
                    ]
                    torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model, encoders)
                if lr_schedule is not None:
                    lr_schedule.advance()
                losses.append(float(loss.detach()))
                ce_losses.append(float(ce.detach()))
                aux_losses.append(float(auxiliary.detach()))
            else:
                logits = output["logits"]
                probabilities = torch.softmax(logits, dim=1)
                all_labels.append(labels.detach().cpu().numpy())
                all_probs.append(probabilities.detach().cpu().numpy())
                all_logits.append(logits.detach().cpu().numpy())
                all_observed.append(observed_mask.detach().cpu().numpy())
                all_combinations.append(modality_comb.detach().cpu().numpy())

    if masked_branch_tcl_enabled:
        masked_branch_tcl_state.end_epoch(epoch)
    if trusted_branch_fusion_enabled:
        trusted_branch_fusion_state.end_epoch(epoch)
    if training:
        return {
            "loss": float(np.mean(losses)),
            "ce": float(np.mean(ce_losses)),
            "aux": float(np.mean(aux_losses)),
        }
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probs)
    return {
        "labels": labels,
        "probabilities": probabilities,
        "logits": np.concatenate(all_logits),
        "predictions": probabilities.argmax(axis=1),
        "observed": np.concatenate(all_observed),
        "modality_comb": np.concatenate(all_combinations),
    }


def optional_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def metric_bundle(labels, predictions, probabilities, num_classes: int) -> dict[str, Any]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    result: dict[str, Any] = {
        "n": int(labels.shape[0]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(num_classes))).tolist(),
    }
    try:
        result["macro_auroc"] = optional_float(classification_auc(labels, probabilities))
    except ValueError:
        result["macro_auroc"] = None
    try:
        if num_classes == 2:
            positive_auprc = classification_average_precision(
                labels, probabilities
            )
            negative_auprc = average_precision_score(
                labels == 0, probabilities[:, 0]
            )
            result["positive_auprc"] = optional_float(positive_auprc)
            result["negative_auprc"] = optional_float(negative_auprc)
            result["macro_auprc"] = optional_float(
                (positive_auprc + negative_auprc) / 2.0
            )
        else:
            result["macro_auprc"] = optional_float(
                classification_average_precision(labels, probabilities)
            )
    except ValueError:
        result["macro_auprc"] = None
        if num_classes == 2:
            result["positive_auprc"] = None
            result["negative_auprc"] = None
    report = classification_report(
        labels,
        predictions,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    result["per_class"] = {
        str(index): {
            key: optional_float(report[str(index)][key])
            for key in ("precision", "recall", "f1-score", "support")
        }
        for index in range(num_classes)
    }
    if num_classes == 2:
        result["positive_f1"] = result["per_class"]["1"]["f1-score"]
        result["sensitivity"] = result["per_class"]["1"]["recall"]
        result["specificity"] = result["per_class"]["0"]["recall"]
        result["mcc"] = float(matthews_corrcoef(labels, predictions))
    return result


def binary_predictions_at_threshold(
    probabilities: np.ndarray, threshold: float
) -> np.ndarray:
    """Apply one fixed positive-class probability threshold.

    This function deliberately has no label argument, making it impossible for
    the test labels to affect the deployed decision rule.
    """

    probability_array = np.asarray(probabilities)
    if probability_array.ndim != 2 or probability_array.shape[1] != 2:
        raise ValueError(
            "Binary thresholding requires a [samples, 2] probability matrix"
        )
    if not np.isfinite(probability_array).all():
        raise ValueError("Binary probabilities must be finite")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Binary decision threshold must be finite and in [0, 1]")
    return (probability_array[:, 1] >= threshold).astype(np.int64)


def validation_binary_decision_protocol(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Choose a binary decision threshold from validation labels only."""

    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1:
        raise ValueError("Binary validation labels must be one-dimensional")
    if set(np.unique(label_array).tolist()) != {0, 1}:
        raise ValueError(
            "Binary validation threshold selection requires both classes"
        )
    threshold, validation_macro_f1 = tune_binary_threshold(
        label_array, probabilities
    )
    # Recompute through the label-free deployment function so the recorded
    # objective and the predictions applied later cannot silently diverge.
    predictions = binary_predictions_at_threshold(probabilities, threshold)
    replay_score = f1_score(
        label_array, predictions, average="macro", zero_division=0
    )
    if not math.isclose(
        float(validation_macro_f1), float(replay_score), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Validation threshold replay changed macro-F1")
    return {
        "implementation_revision": "validation_absolute_threshold_v1",
        "selection_scope": "validation labels and probabilities only",
        "positive_class": 1,
        "probability_column": 1,
        "comparison_operator": ">=",
        "threshold_selection_metric": "macro_f1",
        "threshold": float(threshold),
        "validation_macro_f1_at_threshold": float(validation_macro_f1),
        "tie_break": "largest threshold (most conservative)",
        "test_labels_used": False,
        "test_probability_distribution_used": False,
    }


def checkpoint_selection_metric(num_classes: int) -> str:
    """Use validation positive-class AUPRC for binary rare-event classification."""

    if num_classes < 2:
        raise ValueError("Classification requires at least two classes")
    return "positive_auprc" if num_classes == 2 else "macro_f1"


def subset_metrics(evaluation, num_classes: int) -> dict[str, Any]:
    complete = np.asarray(evaluation["observed"]).all(axis=1)
    output = {}
    for name, mask in (("complete", complete), ("missing", ~complete)):
        if mask.any():
            output[name] = metric_bundle(
                evaluation["labels"][mask],
                evaluation["predictions"][mask],
                evaluation["probabilities"][mask],
                num_classes,
            )
        else:
            output[name] = {"n": 0}
    return output


def evaluation_prediction_frame(
    loader,
    evaluation: Mapping[str, np.ndarray],
    split: str,
    num_classes: int,
    decision_threshold: float | None,
) -> pd.DataFrame:
    """Build an auditable row-level prediction artifact for one split."""

    source_indices = np.asarray(loader.dataset.ids)[
        np.asarray(loader.dataset.sorted_ids)
    ]
    labels = np.asarray(evaluation["labels"])
    predictions = np.asarray(evaluation["predictions"])
    raw_predictions = np.asarray(
        evaluation.get("raw_predictions", evaluation["predictions"])
    )
    if not (
        len(source_indices)
        == len(labels)
        == len(predictions)
        == len(raw_predictions)
    ):
        raise RuntimeError("Prediction rows do not align with loader source IDs")
    frame = pd.DataFrame(
        {
            "split": str(split),
            "source_index": source_indices,
            "label": labels,
            "prediction": predictions,
            "raw_argmax_prediction": raw_predictions,
            "decision_threshold": decision_threshold,
            "modality_combination_index": evaluation["modality_comb"],
            "is_complete": np.asarray(evaluation["observed"]).all(axis=1),
        }
    )
    for class_idx in range(num_classes):
        frame[f"probability_class_{class_idx}"] = evaluation["probabilities"][
            :, class_idx
        ]
        frame[f"logit_class_{class_idx}"] = evaluation["logits"][:, class_idx]
    return frame


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class ValidationCheckpointCandidate(NamedTuple):
    """One validation-ranked model/encoder snapshot; no test field exists."""

    validation_score: float
    epoch: int
    model_state: Mapping[str, torch.Tensor]
    encoder_states: Mapping[str, Mapping[str, torch.Tensor]]
    weight_source: str


def validation_checkpoint_rank_key(
    candidate: ValidationCheckpointCandidate,
) -> tuple[float, int]:
    """Higher validation score ranks first; an exact tie keeps the earlier epoch."""

    return (-float(candidate.validation_score), int(candidate.epoch))


def validation_candidate_would_enter_top_k(
    candidates: Sequence[ValidationCheckpointCandidate],
    validation_score: float,
    epoch: int,
    top_k: int,
) -> bool:
    """Decide whether a validation-only candidate merits a CPU state snapshot."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not math.isfinite(validation_score):
        raise ValueError("validation_score must be finite")
    if epoch < 1:
        raise ValueError("epoch must be one-indexed and positive")
    if len(candidates) < top_k:
        return True
    worst = max(candidates, key=validation_checkpoint_rank_key)
    return (-float(validation_score), int(epoch)) < validation_checkpoint_rank_key(
        worst
    )


def retain_top_validation_candidates(
    candidates: Sequence[ValidationCheckpointCandidate],
    candidate: ValidationCheckpointCandidate,
    top_k: int,
) -> list[ValidationCheckpointCandidate]:
    """Return the deterministic top-K list using the configured validation score."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not math.isfinite(candidate.validation_score):
        raise ValueError("validation_score must be finite")
    if candidate.epoch < 1:
        raise ValueError("epoch must be one-indexed and positive")
    if any(existing.epoch == candidate.epoch for existing in candidates):
        raise ValueError(f"duplicate validation candidate epoch: {candidate.epoch}")
    return sorted(
        [*candidates, candidate], key=validation_checkpoint_rank_key
    )[:top_k]


def average_state_dicts(
    ranked_states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Equally average floating CPU state; copy non-floating tensors from rank 1."""

    if not ranked_states:
        raise ValueError("At least one ranked state is required")
    rank_one = ranked_states[0]
    rank_one_keys = list(rank_one)
    expected_key_set = set(rank_one_keys)
    for rank, state in enumerate(ranked_states, start=1):
        if set(state) != expected_key_set:
            raise ValueError(f"State keys differ at rank {rank}")

    averaged: dict[str, torch.Tensor] = {}
    for key in rank_one_keys:
        first = rank_one[key]
        if not torch.is_tensor(first):
            raise TypeError(f"State value {key!r} is not a tensor")
        if first.device.type != "cpu":
            raise ValueError(f"State value {key!r} is not on CPU")
        values = []
        for rank, state in enumerate(ranked_states, start=1):
            value = state[key]
            if not torch.is_tensor(value):
                raise TypeError(f"State value {key!r} at rank {rank} is not a tensor")
            if value.device.type != "cpu":
                raise ValueError(f"State value {key!r} at rank {rank} is not on CPU")
            if (
                value.shape != first.shape
                or value.dtype != first.dtype
                or value.layout != first.layout
            ):
                raise ValueError(f"State tensor metadata differs for {key!r} at rank {rank}")
            values.append(value)

        if first.is_floating_point() or first.is_complex():
            accumulator_dtype = torch.complex128 if first.is_complex() else torch.float64
            accumulator = torch.zeros_like(first, dtype=accumulator_dtype)
            for value in values:
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"State value {key!r} contains non-finite data")
                accumulator.add_(value.to(dtype=accumulator_dtype))
            averaged[key] = (accumulator / float(len(values))).to(dtype=first.dtype)
        else:
            averaged[key] = first.clone()
    return averaged


def average_encoder_state_dicts(
    ranked_encoder_states: Sequence[
        Mapping[str, Mapping[str, torch.Tensor]]
    ],
) -> dict[str, dict[str, torch.Tensor]]:
    """Apply ``average_state_dicts`` independently to every encoder."""

    if not ranked_encoder_states:
        raise ValueError("At least one ranked encoder state is required")
    encoder_names = list(ranked_encoder_states[0])
    expected_names = set(encoder_names)
    for rank, states in enumerate(ranked_encoder_states, start=1):
        if set(states) != expected_names:
            raise ValueError(f"Encoder names differ at rank {rank}")
    return {
        name: average_state_dicts([states[name] for states in ranked_encoder_states])
        for name in encoder_names
    }


def checkpoint_soup_protocol(
    args,
    ranked_epochs: Sequence[int],
    ranked_validation_scores: Sequence[float | None],
    ranked_weight_sources: Sequence[str],
) -> dict[str, Any]:
    """Return the exact auditable validation-only checkpoint aggregation recipe."""

    requested_top_k = int(getattr(args, "checkpoint_soup_top_k", 1))
    selection_policy = getattr(
        args, "checkpoint_selection_policy", "validation_best"
    )
    epochs = [int(epoch) for epoch in ranked_epochs]
    weight_sources = [str(source) for source in ranked_weight_sources]
    if requested_top_k < 1:
        raise ValueError("checkpoint_soup_top_k must be positive")
    if not epochs or not (
        len(epochs) == len(ranked_validation_scores) == len(weight_sources)
    ):
        raise ValueError("Checkpoint soup ranks must be non-empty and aligned")
    if len(epochs) > requested_top_k:
        raise ValueError("Retained checkpoint count exceeds requested top-K")
    if len(set(epochs)) != len(epochs) or any(epoch < 1 for epoch in epochs):
        raise ValueError("Checkpoint soup epochs must be unique and positive")
    if selection_policy == "final_epoch_refit":
        if requested_top_k != 1 or len(epochs) != 1:
            raise ValueError(
                "final_epoch_refit requires one final live checkpoint"
            )
        if list(ranked_validation_scores) != [None]:
            raise ValueError(
                "final_epoch_refit must not carry a validation selection score"
            )
        if weight_sources != ["live"]:
            raise ValueError("final_epoch_refit must save the final live state")
        requested_epochs = int(getattr(args, "train_epochs", 0))
        if requested_epochs < 1 or epochs != [requested_epochs]:
            raise ValueError(
                "final_epoch_refit epoch must equal the requested final epoch"
            )
        return {
            "implementation_revision": "final_epoch_live_state_v1",
            "enabled": False,
            "selection_scope": None,
            "selection_metric": None,
            "tie_break": None,
            "requested_top_k": 1,
            "retained_top_k": 1,
            "ranked_epochs": epochs,
            "ranked_validation_scores": [None],
            "ranked_validation_macro_f1": None,
            "ranked_validation_macro_auprc": None,
            "ranked_validation_positive_auprc": None,
            "ranked_weight_sources": weight_sources,
            "floating_state_aggregation": None,
            "non_floating_state_aggregation": None,
            "final_candidate_rule": (
                "final live model and encoder state after exactly train_epochs"
            ),
            "post_aggregation_validation": (
                "one held-validation inference after final state capture"
            ),
            "test_used_for_selection_or_aggregation": False,
        }
    if selection_policy != "validation_best":
        raise ValueError(
            f"Unsupported checkpoint selection policy: {selection_policy!r}"
        )
    scores = [float(score) for score in ranked_validation_scores]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Checkpoint soup scores must be finite")
    if any(source not in {"live", "stepwise_ema"} for source in weight_sources):
        raise ValueError("Unsupported checkpoint candidate weight source")
    ranked = list(zip(scores, epochs))
    if ranked != sorted(ranked, key=lambda item: (-item[0], item[1])):
        raise ValueError("Checkpoint soup candidates are not validation-ranked")
    selection_metric = str(
        getattr(args, "checkpoint_selection_metric", "macro_f1")
    )
    if selection_metric not in {
        "macro_f1",
        "positive_auprc",
        "macro_auprc",  # replay compatibility for short-lived v2 artifacts
    }:
        raise ValueError("Unsupported checkpoint selection metric")
    return {
        "implementation_revision": "validation_top_k_equal_weight_v3",
        "enabled": requested_top_k > 1,
        "selection_scope": "validation only",
        "selection_metric": selection_metric,
        "tie_break": "earlier epoch",
        "requested_top_k": requested_top_k,
        "retained_top_k": len(epochs),
        "ranked_epochs": epochs,
        "ranked_validation_scores": scores,
        "ranked_validation_macro_f1": (
            scores if selection_metric == "macro_f1" else None
        ),
        "ranked_validation_macro_auprc": (
            scores if selection_metric == "macro_auprc" else None
        ),
        "ranked_validation_positive_auprc": (
            scores if selection_metric == "positive_auprc" else None
        ),
        "ranked_weight_sources": weight_sources,
        "floating_state_aggregation": "equal-weight arithmetic mean",
        "non_floating_state_aggregation": "rank-1 tensor",
        "final_candidate_rule": (
            "rank-1 validation checkpoint"
            if requested_top_k == 1
            else "always use equal-weight top-K soup; no post-hoc comparison"
        ),
        "post_aggregation_validation": "full validation replay",
        "test_used_for_selection_or_aggregation": False,
    }


def sampler_power_for_epoch(
    epoch: int,
    warm_up_epochs: int,
    target_power: float,
    ramp_epochs: int,
) -> float:
    """Return the training sampler power for a one-indexed epoch.

    Warm-up uses the historical sorted loader and therefore has effective
    sampling power zero.  With a positive ramp, the boundary immediately after
    warm-up is zero and each following epoch advances by ``target / ramp``;
    the last ramp epoch reaches the target.  A zero ramp preserves the original
    abrupt post-warm-up switch.
    """

    if epoch < 1:
        raise ValueError("epoch must be one-indexed and positive")
    if warm_up_epochs < 0:
        raise ValueError("warm_up_epochs must be non-negative")
    if ramp_epochs < 0:
        raise ValueError("ramp_epochs must be non-negative")
    if not math.isfinite(target_power) or target_power < 0:
        raise ValueError("target_power must be finite and non-negative")
    if epoch <= warm_up_epochs:
        return 0.0
    if ramp_epochs == 0:
        return float(target_power)
    progress = min(1.0, (epoch - warm_up_epochs) / float(ramp_epochs))
    return float(target_power) * progress


def training_loader_for_epoch(
    epoch: int,
    *,
    warm_up_epochs: int,
    sampler_power: float,
    sampler_ramp_epochs: int,
    train_sorted,
    train_shuffled,
    train_balanced,
    sampler_seed: int = 0,
):
    """Select/build the train loader without consulting validation or test."""

    if epoch <= warm_up_epochs:
        return train_sorted
    if sampler_ramp_epochs == 0:
        # Keep the historical object and generator state exactly when disabled.
        return train_balanced
    effective_power = sampler_power_for_epoch(
        epoch,
        warm_up_epochs,
        sampler_power,
        sampler_ramp_epochs,
    )
    if effective_power >= sampler_power:
        return train_balanced
    # A distinct but replayable stream for every ramp epoch avoids silently
    # replaying identical weighted draws after constructing a fresh loader.
    return balanced_train_loader(
        train_shuffled, effective_power, seed=int(sampler_seed) + int(epoch)
    )


def resolved_training_sampler_seed(args) -> int:
    explicit=getattr(args,"data_order_seed",None)
    return int(args.seed if explicit is None else explicit)


def sampler_curriculum_protocol(args) -> dict[str, Any]:
    """Auditable definition of the training-only sampler schedule."""

    ramp_epochs = int(getattr(args, "sampler_ramp_epochs", 0))
    return {
        "implementation_revision": "linear_train_only_v1",
        "scope": "training split only",
        "warm_up_epochs": args.warm_up_epochs,
        "warm_up_loader": "missingness-sorted training loader without replacement",
        "target_sampler_power": args.sampler_power,
        "sampler_base_seed": int(args.seed),
        "target_loader_seed": int(args.seed),
        "ramp_loader_seed": "base seed + one-indexed epoch",
        "sampler_ramp_epochs": ramp_epochs,
        "schedule_definition": (
            "historical abrupt target-power switch after warm-up"
            if ramp_epochs == 0
            else (
                "p(epoch)=target*min(1,(epoch-warm_up_epochs)/"
                "sampler_ramp_epochs) after warm-up"
            )
        ),
        "validation_or_test_sampling": False,
    }


def effective_hyperparameters(args) -> None:
    if args.lr is None:
        args.lr = (
            1e-4
            if args.model in {"flex_moe", "i2moe", "moepp", "moepp_corrected", "our_moe"}
            else 3e-4
        )
    if args.weight_decay is None:
        args.weight_decay = (
            0.0
            if args.model in {"flex_moe", "i2moe", "moepp", "moepp_corrected", "our_moe"}
            else 1e-4
        )
    if args.sampler_power is None:
        args.sampler_power = 0.5 if args.data == "abcd" else 0.0
    if args.class_weight_power is None:
        args.class_weight_power = 0.5 if args.data == "abcd" else 0.0


def validate_checkpoint_selection_runtime_contract(args) -> str:
    """Fail closed before training if checkpoint cadence is not implementable."""

    policy = getattr(args, "checkpoint_selection_policy", "validation_best")
    if policy not in {"validation_best", "final_epoch_refit"}:
        raise ValueError(f"Unsupported checkpoint selection policy: {policy!r}")
    if policy == "validation_best":
        return policy
    if getattr(args, "validation_only", None) is not True:
        raise ValueError("final_epoch_refit requires validation_only=True")
    if type(getattr(args, "early_stopping_patience", None)) is not int or (
        args.early_stopping_patience != 0
    ):
        raise ValueError("final_epoch_refit requires early_stopping_patience=0")
    if type(getattr(args, "checkpoint_soup_top_k", None)) is not int or (
        args.checkpoint_soup_top_k != 1
    ):
        raise ValueError("final_epoch_refit requires checkpoint_soup_top_k=1")
    if type(getattr(args, "train_epochs", None)) is not int or args.train_epochs < 1:
        raise ValueError("final_epoch_refit requires positive integer train_epochs")
    resume_fields = {
        "resume": (None, False, ""),
        "resume_checkpoint": (None, False, ""),
        "resume_from_checkpoint": (None, False, ""),
        "start_epoch": (None, 1),
    }
    for field_name, neutral_values in resume_fields.items():
        if hasattr(args, field_name) and getattr(args, field_name) not in neutral_values:
            raise ValueError(
                "final_epoch_refit forbids resume state; non-neutral "
                f"{field_name} was supplied"
            )
    return policy


def validation_selection_due_after_epoch(
    selection_policy: str,
    epoch: int,
    train_epochs: int,
) -> bool:
    """Return whether the held loader may be iterated after a training epoch."""

    if selection_policy not in {"validation_best", "final_epoch_refit"}:
        raise ValueError(
            f"Unsupported checkpoint selection policy: {selection_policy!r}"
        )
    if type(epoch) is not int or type(train_epochs) is not int:
        raise ValueError("epoch and train_epochs must be strict integers")
    if train_epochs < 1 or not 1 <= epoch <= train_epochs:
        raise ValueError("epoch must be in the one-indexed requested epoch range")
    return selection_policy == "validation_best"


def checkpoint_args_for_save(args, selection_policy: str) -> dict[str, Any]:
    """Serialize checkpoint args while preserving the historical default schema."""

    if selection_policy not in {"validation_best", "final_epoch_refit"}:
        raise ValueError(
            f"Unsupported checkpoint selection policy: {selection_policy!r}"
        )
    payload = {
        key: str(value) if isinstance(value, torch.device) else value
        for key, value in vars(args).items()
    }
    if selection_policy == "validation_best":
        payload.pop("checkpoint_selection_policy", None)
    else:
        if payload.get("checkpoint_selection_policy") != "final_epoch_refit":
            raise ValueError(
                "final_epoch_refit checkpoint args must save the explicit policy"
            )
    return payload


def result_paths(args) -> tuple[Path, Path, Path, Path]:
    base = Path(args.output_dir).resolve()
    stem = matched_artifact_stem(args)
    result = base / args.data / f"{stem}.json"
    progress = base / args.data / f"{stem}.progress.json"
    checkpoint = base / "checkpoints" / args.data / f"{stem}.pt"
    predictions = base / "predictions" / args.data / f"{stem}.csv"
    return result, progress, checkpoint, predictions


def train(args) -> dict[str, Any]:
    effective_hyperparameters(args)
    checkpoint_selection_policy = validate_checkpoint_selection_runtime_contract(args)
    final_epoch_refit = checkpoint_selection_policy == "final_epoch_refit"
    claim_path = acquire_training_claim(args)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full baseline run, but torch.cuda.is_available() is false")
    device = torch.device(f"cuda:{args.device}")
    args.torch_device = device
    torch.set_float32_matmul_precision("high")

    modality_dict = resolve_modality_dict(args)
    args.n_full_modalities = len(modality_dict)
    loaded = load_and_preprocess_data(args, modality_dict)
    (
        data_dict,
        encoders,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        num_classes,
        input_dims,
        transforms,
        masks,
        observed,
        full_modality_index,
    ) = loaded
    args.checkpoint_selection_metric = checkpoint_selection_metric(num_classes)
    empirical_missing_pattern_replay_audit = (
        materialize_empirical_missing_pattern_replay_args(
            args,
            observed,
            train_ids,
            data_dict,
            modality_dict,
        )
    )
    empirical_missing_pattern_replay_state = (
        EmpiricalMissingPatternReplayState.from_args(args)
        if empirical_missing_pattern_replay_audit is not None
        else None
    )
    if int(getattr(args, "more_tail_rank", 0)) > 0:
        # Materialize exactly once from current training rows.  Keeping the
        # values in checkpoint args makes the provenance replayable without
        # ever consulting validation/test labels inside the objective.
        args.more_tail_class_prior = training_class_prior(
            labels,
            train_ids,
            num_classes,
        ).astype(np.float64).tolist()
    loaders = create_loaders(
        data_dict,
        observed,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        input_dims,
        transforms,
        masks,
        args.preprocessed,
        args.use_common_ids,
        getattr(args, "data_order_seed", None),
        None,
    )
    train_sorted, train_shuffled, valid_loader, test_loader = loaders
    sample_order_recorder = (
        SampleOrderSHARecorder(args.data_order_seed, train_sorted.dataset.ids, args.train_epochs)
        if getattr(args, "data_order_seed", None) is not None else None
    )
    tree_teacher_artifact = None
    external_teacher_artifact = None
    tree_teacher_weight = float(
        getattr(args, "tree_teacher_distill_weight", 0.0)
    )
    external_teacher_weight = float(
        getattr(args, "external_teacher_distill_weight", 0.0)
    )
    if tree_teacher_weight > 0 and external_teacher_weight > 0:
        raise RuntimeError(
            "v1 tree-teacher and v2 external-teacher training are mutually exclusive"
        )
    if tree_teacher_weight > 0:
        if train_sorted.dataset is not train_shuffled.dataset:
            raise RuntimeError(
                "sorted and shuffled training loaders must share one dataset "
                "for tree-teacher target attachment"
            )
        canonical_subject_ids = canonical_abcd_subject_ids(args)
        tree_teacher_artifact = load_cross_fitted_tree_teacher_npz(
            args.tree_teacher_npz,
            args.tree_teacher_npz_sha256,
            train_ids,
            valid_ids,
            test_ids,
            canonical_subject_ids,
            labels,
            resolved_abcd_manifest_path(args),
            num_classes,
        )
        attach_cross_fitted_tree_teacher(
            train_sorted.dataset,
            train_ids,
            tree_teacher_artifact.probabilities,
        )
        # Persist the canonical resolved path and the independently recomputed
        # digest.  Formal replay validates these fields/protocol but never
        # opens the training-only teacher artifact.
        args.tree_teacher_npz = tree_teacher_artifact.resolved_path
        args.tree_teacher_npz_sha256 = tree_teacher_artifact.sha256
    if external_teacher_weight > 0:
        if train_sorted.dataset is not train_shuffled.dataset:
            raise RuntimeError(
                "sorted and shuffled training loaders must share one dataset "
                "for external-teacher target attachment"
            )
        canonical_subject_ids = canonical_abcd_subject_ids(args)
        external_teacher_artifact = load_cross_fitted_external_teacher_npz(
            args.external_teacher_npz,
            args.external_teacher_npz_sha256,
            train_ids,
            valid_ids,
            test_ids,
            canonical_subject_ids,
            labels,
            resolved_abcd_manifest_path(args),
            num_classes,
        )
        attach_cross_fitted_external_teacher(
            train_sorted.dataset,
            train_ids,
            external_teacher_artifact.probabilities,
        )
        args.external_teacher_npz = external_teacher_artifact.resolved_path
        args.external_teacher_npz_sha256 = external_teacher_artifact.sha256
        args.external_teacher_kind = external_teacher_artifact.teacher_kind
        if external_teacher_artifact.task != EXTERNAL_TEACHER_TASK:
            # Preserve the historical psych3 checkpoint-args schema exactly.
            # The opt-in temporal task must carry an explicit derived binding
            # so formal replay cannot silently reinterpret its teacher rows.
            args.external_teacher_task = external_teacher_artifact.task
        args.external_teacher_generator_protocol_sha256 = (
            external_teacher_artifact.generator_protocol_sha256
        )
        args.external_teacher_class_recalls = (
            external_teacher_artifact.class_recalls.astype(
                np.float64, copy=False
            ).tolist()
        )
        args.external_teacher_class_compensation_weights = (
            external_teacher_artifact.class_compensation_weights.astype(
                np.float64, copy=False
            ).tolist()
        )
        args.external_teacher_correct_coverage = (
            external_teacher_artifact.correct_coverage
        )
    resolved_data_order_seed = resolved_training_sampler_seed(args)
    train_balanced = balanced_train_loader(
        train_shuffled, args.sampler_power, seed=resolved_data_order_seed
    )

    model = build_model(args, len(encoders), num_classes, full_modality_index).to(device)
    for name, encoder in encoders.items():
        encoders[name] = encoder.to(device)
    parameters = list(model.parameters()) + [
        parameter for encoder in encoders.values() for parameter in encoder.parameters()
    ]
    optimizer_class = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optimizer_parameters: Any = parameters
    if args.exclude_bias_norm_from_weight_decay:
        named_parameters = [
            (f"model.{name}", parameter) for name, parameter in model.named_parameters()
        ]
        for encoder_name, encoder in encoders.items():
            named_parameters.extend(
                (f"encoder.{encoder_name}.{name}", parameter)
                for name, parameter in encoder.named_parameters()
            )
        decay_parameters = []
        no_decay_parameters = []
        for name, parameter in named_parameters:
            normalized_name = name.lower()
            should_decay = (
                parameter.ndim >= 2
                and "norm" not in normalized_name
                and "prior" not in normalized_name
                and not normalized_name.endswith("bias")
            )
            (decay_parameters if should_decay else no_decay_parameters).append(parameter)
        optimizer_parameters = [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]
    optimizer = optimizer_class(
        optimizer_parameters,
        lr=args.lr,
        weight_decay=(0.0 if args.exclude_bias_norm_from_weight_decay else args.weight_decay),
    )
    lr_schedule = None
    if args.lr_scheduler == "cosine":
        epoch_lengths = [
            len(train_sorted) if epoch <= args.warm_up_epochs else len(train_balanced)
            for epoch in range(1, args.train_epochs + 1)
        ]
        total_steps = sum(epoch_lengths)
        warmup_steps = sum(epoch_lengths[: args.lr_warmup_epochs])
        lr_schedule = StepwiseCosineSchedule(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
    ema = StepwiseStateEMA(args.ema_decay) if args.use_ema else None
    masked_branch_tcl_state = None
    if float(getattr(args, "masked_branch_tcl_loss_weight", 0.0)) > 0:
        masked_branch_tcl_state = MaskedBranchTCLClassAccuracyEMA(
            num_classes,
            getattr(args, "masked_branch_tcl_class_acc_ema", 0.9),
        )
    trusted_branch_fusion_state = None
    if float(getattr(args, "trusted_branch_fusion_distill_weight", 0.0)) > 0:
        trusted_branch_fusion_state = MaskedBranchTCLClassAccuracyEMA(
            num_classes,
            getattr(args, "trusted_branch_fusion_class_acc_ema", 0.9),
        )
    more_tail_total_training_steps = None
    more_tail_completed_training_steps = 0
    if int(getattr(args, "more_tail_rank", 0)) > 0:
        # Every post-warm-up loader samples the same training universe, so its
        # batch count equals train_balanced even when sampler power ramps.
        more_tail_total_training_steps = sum(
            len(train_sorted) if planned_epoch <= args.warm_up_epochs else len(train_balanced)
            for planned_epoch in range(1, args.train_epochs + 1)
        )
        if more_tail_total_training_steps <= 0:
            raise RuntimeError("MORE tail schedule requires at least one training batch")
    weights = (
        class_weights(labels, train_ids, args.class_weight_power, device)
        if args.class_weight_power > 0
        else None
    )
    if args.model == "our_moe" and args.logit_adjust_tau > 0:
        criterion = LogitAdjustedCrossEntropy(
            class_prior=training_class_prior(labels, train_ids, num_classes),
            tau=args.logit_adjust_tau,
            label_smoothing=args.label_smoothing,
            weight=weights,
        ).to(device)
    else:
        # Preserve the exact historical/default loss path when adjustment is off.
        criterion = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=args.label_smoothing
        )

    result_path, progress_path, checkpoint_path, predictions_path = result_paths(args)
    start_time = time.perf_counter()
    best_score = -math.inf
    best_epoch = 0
    best_model_state = None
    best_encoder_states = None
    best_valid_metrics = None
    best_weight_source = "live"
    top_validation_candidates: list[ValidationCheckpointCandidate] = []
    stale_epochs = 0
    epochs_completed = 0
    held_validation_inference_count = 0

    print(
        f"[Data] {args.data} train/val/test={len(train_ids)}/{len(valid_ids)}/{len(test_ids)} "
        f"classes={np.bincount(np.asarray(labels)[train_ids], minlength=num_classes).tolist()} "
        f"dims={input_dims}"
    )
    print(
        f"[Model] {MODEL_DISPLAY[args.model]} params={sum(p.numel() for p in parameters if p.requires_grad):,} "
        f"device={torch.cuda.get_device_name(device)}"
    )
    if tree_teacher_artifact is not None:
        print(
            "[Tree teacher] exact-training-ID OOF rows="
            f"{len(tree_teacher_artifact.probabilities)} "
            f"folds={tree_teacher_artifact.fold_count} "
            f"sha256={tree_teacher_artifact.sha256}"
        )
    if external_teacher_artifact is not None:
        print(
            "[External teacher] kind="
            f"{external_teacher_artifact.teacher_kind} "
            "exact-training-ID OOF rows="
            f"{len(external_teacher_artifact.probabilities)} "
            f"folds={external_teacher_artifact.fold_count} "
            f"correct_coverage={external_teacher_artifact.correct_coverage:.6f} "
            f"sha256={external_teacher_artifact.sha256}"
        )
    if empirical_missing_pattern_replay_audit is not None:
        print(
            "[Empirical missing-pattern replay] train_rows="
            f"{empirical_missing_pattern_replay_audit['training_row_count']} "
            "positive_patterns="
            f"{sum(count > 0 for count in empirical_missing_pattern_replay_audit['pattern_counts'].values())} "
            "binding_sha256="
            f"{empirical_missing_pattern_replay_audit['train_id_mask_binding_sha256']}"
        )

    for epoch in range(1, args.train_epochs + 1):
        epoch_start = time.perf_counter()
        if args.sampler_ramp_epochs == 0:
            # Exact historical/default path, including the persistent balanced
            # sampler generator state across post-warm-up epochs.
            train_loader = train_sorted if epoch <= args.warm_up_epochs else train_balanced
        else:
            train_loader = training_loader_for_epoch(
                epoch,
                warm_up_epochs=args.warm_up_epochs,
                sampler_power=args.sampler_power,
                sampler_ramp_epochs=args.sampler_ramp_epochs,
                train_sorted=train_sorted,
                train_shuffled=train_shuffled,
                train_balanced=train_balanced,
                sampler_seed=resolved_data_order_seed,
            )
        if sample_order_recorder is not None:
            loader_kind = "warmup_sorted" if epoch <= args.warm_up_epochs else "post_warmup_sampled"
            train_loader = attach_sample_order_recorder(train_loader, sample_order_recorder)
            sample_order_recorder.begin_epoch(epoch, loader_kind)
        train_stats = run_epoch(
            args,
            train_loader,
            encoders,
            modality_dict,
            model,
            criterion,
            device,
            optimizer,
            lr_schedule=lr_schedule,
            ema=(ema if ema is not None and epoch >= args.ema_start_epoch else None),
            epoch=epoch,
            training_step_offset=(
                more_tail_completed_training_steps
                if more_tail_total_training_steps is not None
                else None
            ),
            total_training_steps=more_tail_total_training_steps,
            masked_branch_tcl_state=masked_branch_tcl_state,
            trusted_branch_fusion_state=trusted_branch_fusion_state,
            empirical_missing_pattern_replay_state=(
                empirical_missing_pattern_replay_state
            ),
        )
        if sample_order_recorder is not None:
            sample_order_recorder.end_epoch(epoch)
        if more_tail_total_training_steps is not None:
            more_tail_completed_training_steps += len(train_loader)
        if not validation_selection_due_after_epoch(
            checkpoint_selection_policy, epoch, args.train_epochs
        ):
            # This is intentionally before every validation-selection call.
            # The held loader is not iterated at all during fitting.
            epochs_completed = epoch
            elapsed = time.perf_counter() - epoch_start
            print(
                f"[Epoch {epoch:02d}/{args.train_epochs}] "
                f"loss={train_stats['loss']:.4f} "
                f"ce={train_stats['ce']:.4f} aux={train_stats['aux']:.4f} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )
            atomic_json(
                progress_path,
                {
                    "status": "running",
                    "model": args.model,
                    "dataset": args.data,
                    "seed": args.seed,
                    "epoch": epoch,
                    "epochs_requested": args.train_epochs,
                    "matched_ablation_audit": matched_ablation_audit_config(args),
                    "sample_order_receipt": (
                        sample_order_recorder.receipt() if sample_order_recorder is not None else None
                    ),
                    "best_epoch": None,
                    "checkpoint_selection_policy": "final_epoch_refit",
                    "checkpoint_selection_metric": (
                        args.checkpoint_selection_metric
                    ),
                    "best_validation_selection_score": None,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            continue
        use_ema_validation = ema is not None and ema.ready
        validation_context = (
            ema.average_parameters(model, encoders)
            if use_ema_validation
            else nullcontext()
        )
        with validation_context:
            validation = run_epoch(
                args, valid_loader, encoders, modality_dict, model, criterion, device
            )
            valid_metrics = metric_bundle(
                validation["labels"],
                validation["predictions"],
                validation["probabilities"],
                num_classes,
            )
            score_value = valid_metrics[args.checkpoint_selection_metric]
            if score_value is None or not math.isfinite(float(score_value)):
                raise RuntimeError(
                    "Validation checkpoint metric is unavailable or non-finite: "
                    f"{args.checkpoint_selection_metric}={score_value}"
                )
            score = float(score_value)
            snapshot_model_state = None
            snapshot_encoder_states = None
            candidate_weight_source = (
                "stepwise_ema" if use_ema_validation else "live"
            )
            if (
                args.checkpoint_soup_top_k > 1
                and validation_candidate_would_enter_top_k(
                    top_validation_candidates,
                    score,
                    epoch,
                    args.checkpoint_soup_top_k,
                )
            ):
                snapshot_model_state = cpu_state(model)
                snapshot_encoder_states = {
                    name: cpu_state(encoder) for name, encoder in encoders.items()
                }
                top_validation_candidates = retain_top_validation_candidates(
                    top_validation_candidates,
                    ValidationCheckpointCandidate(
                        validation_score=score,
                        epoch=epoch,
                        model_state=snapshot_model_state,
                        encoder_states=snapshot_encoder_states,
                        weight_source=candidate_weight_source,
                    ),
                    args.checkpoint_soup_top_k,
                )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_valid_metrics = valid_metrics
                if args.checkpoint_soup_top_k > 1:
                    if snapshot_model_state is None or snapshot_encoder_states is None:
                        raise RuntimeError("New best checkpoint was not retained for soup")
                    best_model_state = snapshot_model_state
                    best_encoder_states = snapshot_encoder_states
                else:
                    # Preserve the historical top-1 snapshot path exactly.
                    best_model_state = cpu_state(model)
                    best_encoder_states = {
                        name: cpu_state(encoder) for name, encoder in encoders.items()
                    }
                best_weight_source = candidate_weight_source
                stale_epochs = 0
                marker = " BEST"
            else:
                stale_epochs += 1
                marker = ""
        epochs_completed = epoch
        elapsed = time.perf_counter() - epoch_start
        print(
            f"[Epoch {epoch:02d}/{args.train_epochs}] loss={train_stats['loss']:.4f} "
            f"ce={train_stats['ce']:.4f} aux={train_stats['aux']:.4f} "
            f"val_raw_acc={valid_metrics['accuracy']:.4f} "
            f"val_raw_macro_f1={valid_metrics['macro_f1']:.4f} "
            f"val_auc={valid_metrics['macro_auroc']:.4f} "
            f"val_pos_auprc={valid_metrics.get('positive_auprc', valid_metrics['macro_auprc']):.4f} "
            f"val_macro_auprc={valid_metrics['macro_auprc']:.4f} "
            f"seconds={elapsed:.2f}{marker}",
            flush=True,
        )
        atomic_json(
            progress_path,
            {
                "status": "running",
                "model": args.model,
                "dataset": args.data,
                "seed": args.seed,
                "epoch": epoch,
                "epochs_requested": args.train_epochs,
                "matched_ablation_audit": matched_ablation_audit_config(args),
                "sample_order_receipt": (
                    sample_order_recorder.receipt() if sample_order_recorder is not None else None
                ),
                "best_epoch": best_epoch,
                "checkpoint_selection_metric": args.checkpoint_selection_metric,
                "best_validation_selection_score": best_score,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        if (
            args.early_stopping_patience > 0
            and epoch >= args.early_stopping_min_epochs
            and stale_epochs >= args.early_stopping_patience
        ):
            print(f"[Early stopping] best epoch {best_epoch}", flush=True)
            break

    if sample_order_recorder is not None and len(sample_order_recorder.epoch_receipts) != epochs_completed:
        raise RuntimeError("sample-order receipt count does not match completed training epochs")
    if sample_order_recorder is not None and epochs_completed != sample_order_recorder.expected_epochs:
        raise RuntimeError("explicit sample-order campaign must complete exactly train_epochs")

    if masked_branch_tcl_state is not None:
        masked_branch_tcl_state.materialize_checkpoint_args(args)
    if trusted_branch_fusion_state is not None:
        trusted_branch_fusion_state.materialize_trusted_branch_fusion_checkpoint_args(
            args
        )
    if final_epoch_refit:
        if epochs_completed != args.train_epochs:
            raise RuntimeError(
                "final_epoch_refit did not complete exactly train_epochs"
            )
        best_epoch = args.train_epochs
        best_score = None
        best_model_state = cpu_state(model)
        best_encoder_states = {
            name: cpu_state(encoder) for name, encoder in encoders.items()
        }
        best_weight_source = "final_live"
        checkpoint_soup_epochs = [best_epoch]
        checkpoint_soup_scores = [None]
        checkpoint_soup_weight_sources = ["live"]
    elif best_model_state is None or best_encoder_states is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    elif args.checkpoint_soup_top_k > 1:
        if not top_validation_candidates:
            raise RuntimeError("No validation checkpoints were retained for soup")
        best_model_state = average_state_dicts(
            [candidate.model_state for candidate in top_validation_candidates]
        )
        best_encoder_states = average_encoder_state_dicts(
            [candidate.encoder_states for candidate in top_validation_candidates]
        )
        checkpoint_soup_epochs = [
            candidate.epoch for candidate in top_validation_candidates
        ]
        checkpoint_soup_scores = [
            candidate.validation_score
            for candidate in top_validation_candidates
        ]
        checkpoint_soup_weight_sources = [
            candidate.weight_source for candidate in top_validation_candidates
        ]
        best_weight_source = "validation_top_k_equal_weight_soup"
    else:
        checkpoint_soup_epochs = [best_epoch]
        checkpoint_soup_scores = [best_score]
        checkpoint_soup_weight_sources = [best_weight_source]
    model.load_state_dict(best_model_state)
    for name, encoder in encoders.items():
        encoder.load_state_dict(best_encoder_states[name])
    model.to(device).eval()
    for encoder in encoders.values():
        encoder.to(device).eval()

    held_validation_inference_count += 1
    validation = run_epoch(
        args, valid_loader, encoders, modality_dict, model, criterion, device
    )
    if final_epoch_refit and held_validation_inference_count != 1:
        raise RuntimeError(
            "final_epoch_refit must run held validation exactly once"
        )
    validation["raw_predictions"] = np.asarray(
        validation["predictions"], dtype=np.int64
    ).copy()
    if num_classes == 2:
        decision_protocol = validation_binary_decision_protocol(
            validation["labels"], validation["probabilities"]
        )
        decision_threshold = float(decision_protocol["threshold"])
        validation["predictions"] = binary_predictions_at_threshold(
            validation["probabilities"], decision_threshold
        )
    else:
        decision_threshold = None
        decision_protocol = {
            "implementation_revision": "raw_argmax_v1",
            "selection_scope": None,
            "prediction_rule": "softmax argmax",
            "test_labels_used": False,
            "test_probability_distribution_used": False,
        }
    valid_metrics = metric_bundle(
        validation["labels"],
        validation["predictions"],
        validation["probabilities"],
        num_classes,
    )
    testing = None
    test_metrics = None
    subsets = None
    if not args.validation_only:
        testing = run_epoch(args, test_loader, encoders, modality_dict, model, criterion, device)
        testing["raw_predictions"] = np.asarray(
            testing["predictions"], dtype=np.int64
        ).copy()
        if num_classes == 2:
            testing["predictions"] = binary_predictions_at_threshold(
                testing["probabilities"], decision_threshold
            )
        test_metrics = metric_bundle(
            testing["labels"], testing["predictions"], testing["probabilities"], num_classes
        )
        subsets = subset_metrics(testing, num_classes)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_args = checkpoint_args_for_save(
        args, checkpoint_selection_policy
    )
    torch.save(
        {
            "model": best_model_state,
            "encoders": best_encoder_states,
            "args": checkpoint_args,
            "matched_ablation_audit": matched_ablation_audit_config(args),
            "sample_order_receipt": (
                sample_order_recorder.receipt() if sample_order_recorder is not None else None
            ),
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "decision_protocol": decision_protocol,
        },
        checkpoint_path,
    )
    validation_predictions_path = predictions_path.with_name(
        f"{predictions_path.stem}.validation{predictions_path.suffix}"
    )
    validation_prediction_frame = evaluation_prediction_frame(
        valid_loader,
        validation,
        "validation",
        num_classes,
        decision_threshold,
    )
    validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    validation_prediction_frame.to_csv(validation_predictions_path, index=False)
    if testing is not None:
        prediction_frame = evaluation_prediction_frame(
            test_loader,
            testing,
            "test",
            num_classes,
            decision_threshold,
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(predictions_path, index=False)

    elapsed_seconds = time.perf_counter() - start_time
    result = {
        "status": "validation_complete" if args.validation_only else "complete",
        "model": args.model,
        "model_display": MODEL_DISPLAY[args.model],
        "dataset": args.data,
        "seed": args.seed,
        "matched_ablation_audit": matched_ablation_audit_config(args),
        "sample_order_receipts": (
            sample_order_recorder.receipt()
            if sample_order_recorder is not None else None
        ),
        "protocol": {
            "formal_comparison_models": list(FORMAL_COMPARISON_MODELS),
            "checkpoint_selection": (
                "final live state after exactly train_epochs; no checkpoint selection"
                if final_epoch_refit
                else {
                    "positive_auprc": "validation positive-class AUPRC",
                    "macro_auprc": "validation Macro-AUPRC",
                    "macro_f1": "validation Macro-F1",
                }[args.checkpoint_selection_metric]
            ),
            **(
                {
                    "checkpoint_selection_policy": "final_epoch_refit",
                    "held_validation_inference_count": (
                        held_validation_inference_count
                    ),
                }
                if final_epoch_refit
                else {}
            ),
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "prediction_rule": (
                "fixed validation-selected positive-class probability threshold"
                if num_classes == 2
                else "raw softmax argmax"
            ),
            "binary_decision": decision_protocol if num_classes == 2 else None,
            "metric_implementation": "shared MoE/baseline_runner.py",
            "full_training_not_smoke": True,
            "adni_image_imputation": adni_image_imputation_protocol(args),
            "sampler_curriculum": sampler_curriculum_protocol(args),
            "checkpoint_soup": checkpoint_soup_protocol(
                args,
                checkpoint_soup_epochs,
                checkpoint_soup_scores,
                checkpoint_soup_weight_sources,
            ),
            "our_moe_objective": (
                {
                    "use_generators": args.use_generators,
                    "generator_task_grad": args.generator_task_grad,
                    "generator_only_task_grad": (
                        args.generator_only_task_grad
                    ),
                    "classification_generator_gradient": (
                        classification_generator_gradient_protocol(args)
                    ),
                    "reconstruction_weight": args.recon_loss_weight,
                    "recon_encoder_gradient_scale": (
                        args.recon_encoder_gradient_scale
                    ),
                    "reconstruction_encoder_gradient": (
                        reconstruction_encoder_gradient_protocol(args)
                    ),
                    "branch_aux_weight": args.branch_aux_loss_weight,
                    "joint_head_ensemble_size": args.joint_head_ensemble_size,
                    "joint_head_ensemble": joint_head_ensemble_protocol(args),
                    "patch_adapter_rank": args.patch_adapter_rank,
                    "low_rank_patch_adapter": (
                        low_rank_patch_adapter_protocol(args)
                    ),
                    "more_tail_rank": args.more_tail_rank,
                    "more_tail_peak_amplitude": (
                        args.more_tail_peak_amplitude
                    ),
                    "more_tail_class_prior": getattr(
                        args, "more_tail_class_prior", None
                    ),
                    "more_tail_adapter": more_tail_adapter_protocol(args),
                    "normalized_gate_loss": args.normalized_gate_loss,
                    "normalized_gate_loss_definition": (
                        "mean over router modality calls of "
                        "p_full*balance_full + p_missing*balance_missing + "
                        "sum(missing-token combination CE)/all-token-count"
                    ),
                    "modality_dropout_probability": args.modality_dropout_prob,
                    "empirical_missing_pattern_replay": (
                        args.empirical_missing_pattern_replay
                    ),
                    "empirical_missing_pattern_replay_protocol": (
                        empirical_missing_pattern_replay_protocol(args)
                    ),
                    "drop_ce_weight": args.drop_ce_loss_weight,
                    "more_fewer_rank_loss_weight": (
                        args.more_fewer_rank_loss_weight
                    ),
                    "more_fewer_rank": more_fewer_rank_protocol(args),
                    "matched_ablation_structure": {
                        name: bool(getattr(args, name, False)) for name in (
                            "dense_backbone", "disable_provenance",
                            "uniform_branch_weights", "joint_branch_only",
                            "mean_pooling_only",
                            "disable_stochastic_context_masking",
                            "disable_completion", "no_output_gate"
                        )
                    },
                    "data_order_seed": getattr(args, "data_order_seed", None),
                    "distillation_weight": args.distill_loss_weight,
                    "distillation_temperature": args.distill_temperature,
                    "tree_teacher_distill_weight": (
                        args.tree_teacher_distill_weight
                    ),
                    "tree_teacher_temperature": args.tree_teacher_temperature,
                    "tree_teacher_npz": args.tree_teacher_npz,
                    "tree_teacher_npz_sha256": (
                        args.tree_teacher_npz_sha256
                    ),
                    "cross_fitted_tree_teacher": (
                        cross_fitted_tree_teacher_protocol(args)
                    ),
                    "external_teacher_distill_weight": (
                        args.external_teacher_distill_weight
                    ),
                    "external_teacher_temperature": (
                        args.external_teacher_temperature
                    ),
                    "external_teacher_npz": args.external_teacher_npz,
                    "external_teacher_npz_sha256": (
                        args.external_teacher_npz_sha256
                    ),
                    "external_teacher_trust_mode": (
                        args.external_teacher_trust_mode
                    ),
                    "external_teacher_class_compensation": (
                        args.external_teacher_class_compensation
                    ),
                    "external_teacher_kind": getattr(
                        args, "external_teacher_kind", None
                    ),
                    **(
                        {
                            "external_teacher_task": args.external_teacher_task
                        }
                        if hasattr(args, "external_teacher_task")
                        else {}
                    ),
                    "external_teacher_generator_protocol_sha256": getattr(
                        args,
                        "external_teacher_generator_protocol_sha256",
                        None,
                    ),
                    "external_teacher_class_recalls": getattr(
                        args, "external_teacher_class_recalls", None
                    ),
                    "external_teacher_class_compensation_weights": getattr(
                        args,
                        "external_teacher_class_compensation_weights",
                        None,
                    ),
                    "external_teacher_correct_coverage": getattr(
                        args, "external_teacher_correct_coverage", None
                    ),
                    "cross_fitted_external_teacher": (
                        cross_fitted_external_teacher_protocol(args)
                    ),
                    "masked_branch_tcl_loss_weight": (
                        args.masked_branch_tcl_loss_weight
                    ),
                    "masked_branch_tcl_temperature": (
                        args.masked_branch_tcl_temperature
                    ),
                    "masked_branch_tcl_start_epoch": (
                        args.masked_branch_tcl_start_epoch
                    ),
                    "masked_branch_tcl_class_acc_ema": (
                        args.masked_branch_tcl_class_acc_ema
                    ),
                    "masked_branch_tcl": masked_branch_tcl_protocol(args),
                    "trusted_branch_fusion_distill_weight": (
                        args.trusted_branch_fusion_distill_weight
                    ),
                    "trusted_branch_fusion_temperature": (
                        args.trusted_branch_fusion_temperature
                    ),
                    "trusted_branch_fusion_start_epoch": (
                        args.trusted_branch_fusion_start_epoch
                    ),
                    "trusted_branch_fusion_class_acc_ema": (
                        args.trusted_branch_fusion_class_acc_ema
                    ),
                    "trusted_branch_fusion_distillation": (
                        trusted_branch_fusion_distillation_protocol(args)
                    ),
                    "complete_joint_only": args.complete_joint_only,
                    "complete_specialist_weight": args.complete_specialist_weight,
                    "missing_family_normalized_router": (
                        args.missing_family_normalized_router
                    ),
                    "missing_family_residual_gate": (
                        args.missing_family_residual_gate
                    ),
                    "missing_family_router": (
                        missing_family_router_protocol(args)
                    ),
                    "branch_confidence_mode": args.branch_confidence_mode,
                    "centered_evidence_confidence": args.centered_evidence_confidence,
                    "learn_observed_reliability": args.learn_observed_reliability,
                    "class_conditional_fusion": args.class_conditional_fusion,
                    "ordinal_head_type": args.ordinal_head_type,
                    "ordinal_fusion_weight": args.ordinal_fusion_weight,
                    "ordinal_aux_loss_weight": args.ordinal_aux_loss_weight,
                    "class1_aux_loss_weight": args.class1_aux_loss_weight,
                    "class1_auxiliary_target": "label == 1",
                    "class1_auxiliary_pos_weight": 1.0,
                    "class1_auxiliary_inference_fusion": False,
                    "supervised_router_loss_weight": (
                        args.supervised_router_loss_weight
                    ),
                    "supervised_router_temperature": (
                        args.supervised_router_temperature
                    ),
                    "dynamic_branch_joint_prior_boost": (
                        args.dynamic_branch_joint_prior_boost
                    ),
                    "dynamic_branch_use_observed_mask": (
                        args.dynamic_branch_use_observed_mask
                    ),
                    "prediction_observed_specialists_only": (
                        args.prediction_observed_specialists_only
                    ),
                    "dynamic_branch_quality_weighted_aux": (
                        args.dynamic_branch_quality_weighted_aux
                    ),
                    "dynamic_branch_gate_mask_source": (
                        "raw observed_mask"
                        if args.dynamic_branch_use_observed_mask
                        else "generator-expanded usable_mask"
                    ),
                    "supervised_router_observed_specialists_only": (
                        args.supervised_router_observed_specialists_only
                    ),
                    "supervised_router_teacher": (
                        "softmax over detached negative per-branch true-class NLL, "
                        + (
                            "restricted to joint plus fully observed specialists"
                            if args.supervised_router_observed_specialists_only
                            else "restricted to prediction-available branches"
                        )
                    ),
                    "supervised_router_gradient_scope": (
                        "detached evidence/quality base plus differentiable dynamic "
                        "routing correction; no router-only branch-head gradient"
                    ),
                    "clean_dynamic_router": clean_dynamic_router_protocol(args),
                    "supervised_contrastive_loss_weight": (
                        args.supervised_contrastive_loss_weight
                    ),
                    "supervised_contrastive_temperature": (
                        args.supervised_contrastive_temperature
                    ),
                    "supervised_contrastive_projection_dim": (
                        args.supervised_contrastive_projection_dim
                    ),
                    "supervised_contrastive_views": (
                        "full pooled fusion feature and the existing training-only "
                        "modality-dropped pooled fusion feature"
                    ),
                    "supervised_contrastive_positive_rule": (
                        "all non-self views with the same class label; paired view "
                        "guarantees at least one positive"
                    ),
                    "supervised_contrastive_anchor_weighting": (
                        "main CE class weights normalized by the anchor-weight sum"
                    ),
                    "supervised_contrastive_inference_fusion": False,
                    "dual_boundary_rank_loss_weight": (
                        args.dual_boundary_rank_loss_weight
                    ),
                    "dual_boundary_rank_margin": args.dual_boundary_rank_margin,
                    "dual_boundary_rank_10_weight": (
                        args.dual_boundary_rank_10_weight
                    ),
                    "dual_boundary_rank_scope": "full fused logits only",
                    "dual_boundary_rank_definition": (
                        "weighted all-pairs softplus margin ranking of "
                        "(logit_1-logit_0) for label-1 versus label-0 samples "
                        "and (logit_1-logit_2) for label-1 versus label-2 samples"
                    ),
                    "dual_boundary_rank_inference_fusion": False,
                    "dual_local_boundary_loss_weight": (
                        args.dual_local_boundary_loss_weight
                    ),
                    "dual_local_boundary_residual": (
                        dual_local_boundary_residual_protocol(args)
                    ),
                    "presentation_axis_loss_weight": (
                        args.presentation_axis_loss_weight
                    ),
                    "presentation_axis_residual_cap": (
                        args.presentation_axis_residual_cap
                    ),
                    "presentation_axis_residual": (
                        presentation_axis_residual_protocol(args)
                    ),
                    "missing_capacity_residual_width": (
                        args.missing_capacity_residual_width
                    ),
                    "missing_capacity_residual": (
                        missing_capacity_residual_protocol(args)
                    ),
                    "hard_cvar_dual_boundary_max_weight": (
                        args.hard_cvar_dual_boundary_max_weight
                    ),
                    "hard_cvar_dual_boundary_tail_fraction": (
                        args.hard_cvar_dual_boundary_tail_fraction
                    ),
                    "hard_cvar_dual_boundary_margin": (
                        args.hard_cvar_dual_boundary_margin
                    ),
                    "hard_cvar_dual_boundary_10_weight": (
                        args.hard_cvar_dual_boundary_10_weight
                    ),
                    "hard_cvar_dual_boundary_start_epoch": (
                        args.hard_cvar_dual_boundary_start_epoch
                    ),
                    "hard_cvar_dual_boundary_ramp_epochs": (
                        args.hard_cvar_dual_boundary_ramp_epochs
                    ),
                    "hard_cvar_dual_boundary": (
                        hard_cvar_dual_boundary_protocol(args)
                    ),
                    "sam_rho": args.sam_rho,
                    "sam": sam_training_protocol(args),
                    "rdrop_loss_weight": args.rdrop_loss_weight,
                    "rdrop": rdrop_training_protocol(args),
                    "clear_eval_gate_cache": args.clear_eval_gate_cache,
                    "evaluation_gate_cache": evaluation_gate_cache_protocol(args),
                    "uncertainty_aware_ordinal_fusion": (
                        args.uncertainty_aware_ordinal_fusion
                    ),
                    "ordinal_auxiliary_definition": (
                        "equal-weight mean of the y>=1 boundary BCE and the "
                        "y=2|y>=1 boundary BCE"
                    ),
                    "gated_transformer_residual": args.gated_transformer_residual,
                    "class_conditional_interpretation": (
                        "scalar branch_weights remain reliability routing weights; "
                        "the bounded branch-by-class correction is predictive calibration"
                    ),
                    "optimization_stabilization": {
                        "lr_schedule": args.lr_scheduler,
                        "lr_warmup_epochs": args.lr_warmup_epochs,
                        "minimum_lr_ratio": args.min_lr_ratio,
                        "selective_weight_decay": args.exclude_bias_norm_from_weight_decay,
                        "ema_enabled": args.use_ema,
                        "ema_decay": args.ema_decay,
                        "ema_start_epoch": args.ema_start_epoch,
                        "ema_update_unit": "optimizer_step",
                        "validation_weight_source": (
                            "final_live"
                            if final_epoch_refit
                            else (
                                "stepwise_ema" if args.use_ema else "live"
                            )
                        ),
                    },
                    "class_weighted_quality_branch_aux": (
                        args.class_weighted_quality_branch_aux
                    ),
                    "logit_adjust_tau": args.logit_adjust_tau,
                    "logit_adjustment_prior_source": (
                        "empirical training-split class frequencies only"
                    ),
                    "logit_adjustment_scope": (
                        "training-only criterion calls; validation and test use "
                        "raw model logits"
                    ),
                    "recompute_dropped_combination": args.recompute_dropped_combination,
                    "vectorized_generation": args.vectorized_generation,
                    "reconstruction_targets_per_sample": args.recon_targets_per_sample,
                    "pattern_aware_reconstruction": (
                        args.pattern_aware_reconstruction
                    ),
                    "recon_normalized_token_loss_weight": (
                        args.recon_normalized_token_loss_weight
                    ),
                    "recon_context_dropout_probability": (
                        args.recon_context_dropout_probability
                    ),
                    "generator_output_gate": args.generator_output_gate,
                    "reconstruction_target_scope": (
                        "naturally observed per-sample targets with at least one "
                        "naturally observed context modality"
                        if args.pattern_aware_reconstruction
                        else "legacy batch-level complete-pattern targets"
                    ),
                }
                if args.model == "our_moe"
                else None
            ),
            "i2moe_objective": (
                {
                    "task_loss": "cross_entropy",
                    "interaction_loss": "mean of M uniqueness + synergy + redundancy losses",
                    "interaction_loss_weight": args.interaction_loss_weight,
                    "num_interaction_experts": len(encoders) + 2,
                    "fusion_backbone": "official dense Transformer (fusion_sparse=False)",
                    "fusion_layers": args.num_layers_fus,
                    "prediction_head": "official single Linear head (upstream num_layers_pred is unused)",
                    "reweighting_hidden_dim": args.hidden_dim_rw,
                    "reweighting_layers": args.num_layer_rw,
                    "reweighting_temperature": args.temperature_rw,
                    "training_forward": "official random-modality perturbation path",
                    "evaluation_forward": "official unperturbed inference path",
                }
                if args.model == "i2moe"
                else None
            ),
            "transformer_concat": (
                {
                    "fusion": "concatenate modality tokens before a standard Transformer encoder",
                    "missing_token_handling": "Transformer key-padding mask",
                    "pooling": "learned CLS token",
                    "training_modality_dropout_probability": args.modality_dropout_prob,
                    "task_loss": "cross_entropy",
                    "auxiliary_loss": None,
                    "fusion_layers": args.num_layers_fus,
                    "prediction_layers": args.num_layers_pred,
                }
                if args.model == "transformer_concat"
                else None
            ),
            "acadiff_objective": (
                acadiff_objective_protocol(args)
                if args.model == "acadiff"
                else None
            ),
            "moepp_adapter": (
                {
                    "implementation": "official-code adapter",
                    "task_loss": "cross_entropy",
                    "auxiliary_loss": None,
                    "routing": "official residual top-2 router",
                    "expert_set": "learned SwiGLU + two constant + copy + zero experts",
                    "fusion_layers": args.num_layers_fus,
                    "num_experts": args.num_experts,
                    "prediction_head": "official single Linear head",
                }
                if args.model in {"moepp", "moepp_corrected"}
                else None
            ),
            "flex_moe_adapter": (
                {
                    "implementation": "official-code-derived architecture retrained by shared runner",
                    "missing_bank": "modality-combination bank",
                    "missing_bank_init_std": args.missing_bank_init_std,
                    "classifier_dropout": (
                        args.dropout if args.classifier_dropout is None else args.classifier_dropout
                    ),
                }
                if args.model == "flex_moe"
                else None
            ),
        },
        "provenance": model.provenance() if hasattr(model, "provenance") else None,
        "data": {
            "manifest": args.dataset_manifest,
            "modality": args.modality,
            "split_sizes": {"train": len(train_ids), "validation": len(valid_ids), "test": len(test_ids)},
            "train_class_counts": np.bincount(np.asarray(labels)[train_ids], minlength=num_classes).tolist(),
            "input_dimensions": input_dims,
            "num_classes": num_classes,
            "empirical_missing_pattern_replay_training_audit": (
                empirical_missing_pattern_replay_audit
            ),
        },
        "training": {
            "epochs_requested": args.train_epochs,
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            **(
                {
                    "checkpoint_selection_policy": "final_epoch_refit",
                    "held_validation_inference_count": (
                        held_validation_inference_count
                    ),
                }
                if final_epoch_refit
                else {}
            ),
            "warm_up_epochs": args.warm_up_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "minimum_lr_ratio": args.min_lr_ratio,
            "lr_warmup_epochs": args.lr_warmup_epochs,
            "selective_weight_decay": args.exclude_bias_norm_from_weight_decay,
            "ema_enabled": args.use_ema,
            "ema_decay": args.ema_decay,
            "ema_start_epoch": args.ema_start_epoch,
            "checkpoint_weight_source": best_weight_source,
            "gradient_clip": args.grad_clip,
            "dropout": args.dropout,
            "sampler_power": args.sampler_power,
            "sampler_ramp_epochs": args.sampler_ramp_epochs,
            "checkpoint_soup_top_k": args.checkpoint_soup_top_k,
            "checkpoint_soup_retained": len(checkpoint_soup_epochs),
            "checkpoint_soup_epochs": checkpoint_soup_epochs,
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "best_validation_selection_score": best_score,
            "checkpoint_soup_validation_scores": checkpoint_soup_scores,
            # Compatibility alias consumed by the legacy strict replay
            # validator.  The adjacent explicit metric field is authoritative.
            "checkpoint_soup_validation_macro_f1": checkpoint_soup_scores,
            "checkpoint_soup_validation_macro_auprc": (
                checkpoint_soup_scores
                if args.checkpoint_selection_metric == "macro_auprc"
                else None
            ),
            "checkpoint_soup_validation_positive_auprc": (
                checkpoint_soup_scores
                if args.checkpoint_selection_metric == "positive_auprc"
                else None
            ),
            "checkpoint_soup_candidate_weight_sources": (
                checkpoint_soup_weight_sources
            ),
            "class_weight_power": args.class_weight_power,
            "acadiff_mask_probability": (
                args.acadiff_mask_probability if args.model == "acadiff" else None
            ),
            "elapsed_seconds": elapsed_seconds,
            "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
            "device": torch.cuda.get_device_name(device),
        },
        "validation": valid_metrics,
        "test": test_metrics,
        "test_subsets": subsets,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "validation_predictions": str(validation_predictions_path),
            "predictions": None if args.validation_only else str(predictions_path),
            "progress": str(progress_path),
            "training_claim": str(claim_path),
        },
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(result_path, result)
    atomic_json(
        progress_path,
        {
            "status": result["status"],
            "model": args.model,
            "dataset": args.data,
            "seed": args.seed,
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            **(
                {
                    "checkpoint_selection_policy": "final_epoch_refit",
                    "held_validation_inference_count": (
                        held_validation_inference_count
                    ),
                }
                if final_epoch_refit
                else {}
            ),
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "best_validation_selection_score": best_score,
            "checkpoint_soup_top_k": args.checkpoint_soup_top_k,
            "checkpoint_soup_retained": len(checkpoint_soup_epochs),
            "result": str(result_path),
            "updated_at": result["completed_at"],
        },
    )
    if test_metrics is None:
        print(
            f"[Validation only] accuracy={valid_metrics['accuracy']:.4f} "
            f"balanced_accuracy={valid_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={valid_metrics['macro_f1']:.4f} weighted_f1={valid_metrics['weighted_f1']:.4f} "
            f"macro_auroc={valid_metrics['macro_auroc']:.4f} "
            f"positive_auprc={valid_metrics.get('positive_auprc', valid_metrics['macro_auprc']):.4f} "
            f"macro_auprc={valid_metrics['macro_auprc']:.4f}",
            flush=True,
        )
    else:
        print(
            f"[Test] accuracy={test_metrics['accuracy']:.4f} balanced_accuracy={test_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f} weighted_f1={test_metrics['weighted_f1']:.4f} "
            f"macro_auroc={test_metrics['macro_auroc']:.4f} "
            f"positive_auprc={test_metrics.get('positive_auprc', test_metrics['macro_auprc']):.4f} "
            f"macro_auprc={test_metrics['macro_auprc']:.4f}",
            flush=True,
        )
    print(f"[Saved] {result_path}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    result_path, progress_path, _, _ = result_paths(args)
    try:
        train(args)
    except ArtifactCollisionError:
        # The collision guard exists precisely to protect the paths below; do
        # not replace a valid result/progress artifact with a failure record.
        raise
    except Exception as error:
        failure = {
            "status": "failed",
            "model": args.model,
            "model_display": MODEL_DISPLAY.get(args.model, args.model),
            "dataset": args.data,
            "seed": args.seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(result_path, failure)
        atomic_json(progress_path, failure)
        raise


if __name__ == "__main__":
    main()
