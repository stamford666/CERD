#!/usr/bin/env python3
"""Train the public CERD reference workflows on ABCD or ADNI.

The manifest-driven binary ABCD-ADHD workflow is an independent public reference
benchmark, not the source of the frozen three-class dev946 method-revision result.
Binary reference checkpoints are selected by validation Macro-AUPRC and their
decision threshold is calibrated on validation only; multiclass checkpoints are
selected by validation Macro-F1.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from cerd.ablations import (
    ABLATION_IDS,
    FULL_MODEL_ID,
    apply_matched_ablation,
    normalize_checkpoint_protocol_parameters,
)
from cerd.data import create_loaders, load_and_preprocess_data, resolve_modality_dict
from cerd.losses import (
    BranchClassAccuracyEMA,
    LogitAdjustedCrossEntropy,
    branch_auxiliary_loss,
    combination_indices_from_mask,
    dual_boundary_rank_loss,
    masked_branch_tcl_loss,
    modality_dropout_mask,
    more_fewer_rank_loss,
    self_distillation_loss,
    trusted_branch_fusion_distillation_loss,
)
from cerd.metrics import (
    binary_predictions_at_threshold,
    metric_bundle,
    tune_binary_threshold,
)
from cerd.model import AGMGFlexMoE
from cerd.sampling import DATA_ORDER_RNG, balanced_train_loader, class_weights


VARIANTS = ("auto", "core", "dbr", "mofe", "unified", "mofe_tcl", "mofe_tbfd")
PUBLIC_CONFIGURATION_SCHEMA = "cerd-public-training-config-v1"
_FOLD_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")

# Only non-path, protocol-relevant values may enter the public run JSON.  An
# allowlist prevents a future CLI path from being published accidentally.
PUBLIC_PROTOCOL_PARAMETER_NAMES = (
    "ablation_id",
    "ablation_order_sha256",
    "configuration_sha256",
    "data_order_seed",
    "data_order_rng",
    "seed",
    "fold_id",
    "split_receipt_sha256",
    "data",
    "variant",
    "modality",
    "evaluate_test",
    "train_epochs",
    "warm_up_epochs",
    "batch_size",
    "num_workers",
    "pin_memory",
    "lr",
    "weight_decay",
    "grad_clip",
    "sampler_power",
    "class_weight_power",
    "hidden_dim",
    "num_patches",
    "num_heads",
    "num_layers_fus",
    "num_layers_pred",
    "num_experts",
    "num_routers",
    "top_k",
    "dropout",
    "gen_num_layers",
    "gen_num_heads",
    "pattern_aware_reconstruction",
    "recon_context_dropout_probability",
    "recon_normalized_token_loss_weight",
    "branch_confidence_mode",
    "generator_output_gate",
    "generator_only_task_grad",
    "generator_task_grad",
    "use_generators",
    "vectorized_generation",
    "recon_targets_per_sample",
    "use_sparse_moe_backbone",
    "use_provenance_embeddings",
    "use_reliability_branch_weights",
    "use_attentive_token_pooling",
    "use_stochastic_reconstruction_context",
    "use_latent_completion",
    "use_more_fewer_objective",
    "gate_loss_weight",
    "recon_loss_weight",
    "branch_aux_loss_weight",
    "modality_dropout_prob",
    "drop_ce_loss_weight",
    "distill_loss_weight",
    "distill_temperature",
    "dual_boundary_rank_loss_weight",
    "dual_boundary_rank_margin",
    "dual_boundary_rank_10_weight",
    "more_fewer_rank_loss_weight",
    "effective_more_fewer_rank_loss_weight",
    "branch_distill_loss_weight",
    "branch_distill_temperature",
    "branch_distill_start_epoch",
    "branch_accuracy_ema_decay",
    "adni_image_imputation",
    "label_smoothing",
    "logit_adjust_tau",
    "preprocessed",
    "initial_filling",
    "use_common_ids",
)


def public_protocol_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """Select reproducibility metadata without publishing local paths."""

    selected = {
        name: getattr(args, name)
        for name in PUBLIC_PROTOCOL_PARAMETER_NAMES
        if hasattr(args, name)
    }
    if "ablation_id" not in selected:
        return selected
    return normalize_checkpoint_protocol_parameters(selected)


def canonical_configuration_sha256(args: argparse.Namespace) -> str:
    """Digest the sanitized, normalized configuration used by one public run."""

    parameters = public_protocol_parameters(args)
    parameters.pop("configuration_sha256", None)
    parameters.setdefault(
        "data_order_seed",
        int(getattr(args, "data_order_seed", getattr(args, "seed", 0))),
    )
    payload = {
        "schema": PUBLIC_CONFIGURATION_SCHEMA,
        "parameters": parameters,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_fold_id(value: Any) -> str | None:
    if value is None:
        return None
    fold_id = str(value)
    if _FOLD_ID_PATTERN.fullmatch(fold_id) is None:
        raise ValueError(
            "fold_id must contain 1-32 ASCII letters, digits, underscores, or hyphens"
        )
    return fold_id


def normalize_optional_sha256(value: Any, field: str) -> str | None:
    if value is None:
        return None
    digest = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, choices=("abcd", "adni"))
    parser.add_argument("--variant", default="auto", choices=VARIANTS)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--adni-data-root", default="data/adni")
    parser.add_argument(
        "--modality",
        default=None,
        help="modality codes (default: SRDGNPME for ABCD, IGCB for ADNI)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fold-id",
        default=None,
        help="safe fold label for fold-based runs (for example 0 or fold_0)",
    )
    parser.add_argument(
        "--split-receipt-sha256",
        default=None,
        help="safe digest binding a controlled fold/split receipt",
    )
    parser.add_argument(
        "--data-order-seed",
        type=int,
        default=None,
        help="independent DataLoader/sampler seed (default: --seed)",
    )
    parser.add_argument(
        "--ablation-id",
        choices=(FULL_MODEL_ID, *ABLATION_IDS),
        default=FULL_MODEL_ID,
        help="one canonical matched control; full leaves the base method intact",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--warm-up-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--sampler-power", type=float, default=None)
    parser.add_argument("--class-weight-power", type=float, default=None)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-patches", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers-fus", type=int, default=1)
    parser.add_argument("--num-layers-pred", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--num-routers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gen-num-layers", type=int, default=2)
    parser.add_argument("--gen-num-heads", type=int, default=4)
    parser.add_argument(
        "--pattern-aware-reconstruction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--recon-context-dropout-probability", type=float, default=0.0
    )
    parser.add_argument(
        "--recon-normalized-token-loss-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--branch-confidence-mode",
        choices=("evidence", "entropy_detached", "entropy_exp_detached"),
        default="evidence",
    )
    parser.add_argument(
        "--generator-output-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--generator-only-task-grad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allow classification loss to update generators only (opt in)",
    )

    parser.add_argument("--gate-loss-weight", type=float, default=0.01)
    parser.add_argument("--recon-loss-weight", type=float, default=1.0)
    parser.add_argument("--branch-aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.25)
    parser.add_argument("--drop-ce-loss-weight", type=float, default=0.1)
    parser.add_argument("--distill-loss-weight", type=float, default=0.15)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--dual-boundary-rank-loss-weight", type=float, default=None)
    parser.add_argument("--dual-boundary-rank-margin", type=float, default=0.2)
    parser.add_argument("--dual-boundary-rank-10-weight", type=float, default=2.0 / 3.0)
    parser.add_argument("--more-fewer-rank-loss-weight", type=float, default=None)
    parser.add_argument("--branch-distill-loss-weight", type=float, default=None)
    parser.add_argument("--branch-distill-temperature", type=float, default=2.0)
    parser.add_argument("--branch-distill-start-epoch", type=int, default=6)
    parser.add_argument("--branch-accuracy-ema-decay", type=float, default=0.9)

    parser.add_argument("--adni-image-imputation", choices=("mean", "median"), default="mean")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--logit-adjust-tau", type=float, default=0.0)
    return parser.parse_args()


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    method_defaults = {
        "pattern_aware_reconstruction": False,
        "recon_context_dropout_probability": 0.0,
        "recon_normalized_token_loss_weight": 0.0,
        "branch_confidence_mode": "evidence",
        "generator_output_gate": True,
        "generator_only_task_grad": False,
    }
    for name, value in method_defaults.items():
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, value)
    if not hasattr(args, "data_order_seed") or args.data_order_seed is None:
        args.data_order_seed = int(getattr(args, "seed", 0))
    args.data_order_rng = DATA_ORDER_RNG
    args.fold_id = normalize_fold_id(getattr(args, "fold_id", None))
    args.split_receipt_sha256 = normalize_optional_sha256(
        getattr(args, "split_receipt_sha256", None),
        "split_receipt_sha256",
    )
    if args.fold_id is not None and args.split_receipt_sha256 is None:
        raise ValueError("fold_id requires --split-receipt-sha256")
    if getattr(args, "modality", None) is None:
        args.modality = "SRDGNPME" if args.data == "abcd" else "IGCB"
    if args.variant == "auto":
        # DBR encodes the two boundaries around the middle class and is only
        # defined for a three-class endpoint.  MoFe is valid for both the
        # independent public binary ABCD reference and multiclass endpoints.
        args.variant = "mofe"
    args.batch_size = args.batch_size or (64 if args.data == "abcd" else 32)
    args.weight_decay = (
        args.weight_decay if args.weight_decay is not None else (0.0 if args.data == "abcd" else 0.01)
    )
    args.sampler_power = (
        args.sampler_power if args.sampler_power is not None else (0.35 if args.data == "abcd" else 0.0)
    )
    args.class_weight_power = (
        args.class_weight_power if args.class_weight_power is not None else (0.15 if args.data == "abcd" else 0.0)
    )
    args.num_layers_pred = args.num_layers_pred or (2 if args.data == "abcd" else 1)
    default_dbr = 0.02 if args.variant in {"dbr", "unified"} else 0.0
    default_mofe = 0.1 if args.variant in {"mofe", "unified", "mofe_tcl", "mofe_tbfd"} else 0.0
    default_branch_distill = 0.05 if args.variant in {"mofe_tcl", "mofe_tbfd"} else 0.0
    if args.dual_boundary_rank_loss_weight is None:
        args.dual_boundary_rank_loss_weight = default_dbr
    if args.more_fewer_rank_loss_weight is None:
        args.more_fewer_rank_loss_weight = default_mofe
    if args.branch_distill_loss_weight is None:
        args.branch_distill_loss_weight = default_branch_distill

    # Compatibility attributes consumed by the shared data adapter.
    args.dataset_manifest = args.dataset_manifest
    args.adni_data_root = args.adni_data_root
    args.preprocessed = True
    args.initial_filling = "mean"
    args.use_common_ids = False
    args = apply_matched_ablation(args)
    args.effective_more_fewer_rank_loss_weight = (
        float(args.more_fewer_rank_loss_weight)
        if args.use_more_fewer_objective
        else 0.0
    )
    return args


def validate_objective_compatibility(args: argparse.Namespace, num_classes: int) -> None:
    """Reject objectives whose mathematical assumptions do not match the target."""

    context_dropout = float(args.recon_context_dropout_probability)
    if not math.isfinite(context_dropout) or not 0.0 <= context_dropout <= 1.0:
        raise ValueError(
            "recon_context_dropout_probability must be finite and in [0, 1]"
        )
    if context_dropout > 0 and not args.pattern_aware_reconstruction:
        raise ValueError(
            "positive reconstruction context dropout requires pattern-aware reconstruction"
        )
    token_weight = float(args.recon_normalized_token_loss_weight)
    if not math.isfinite(token_weight) or token_weight < 0:
        raise ValueError(
            "recon_normalized_token_loss_weight must be finite and non-negative"
        )

    if num_classes < 2:
        raise ValueError("CERD classification requires at least two classes")
    if args.dual_boundary_rank_loss_weight > 0 and num_classes != 3:
        raise ValueError(
            "dual-boundary ranking requires exactly three classes; use "
            "--variant mofe/core for a binary target or set "
            "--dual-boundary-rank-loss-weight 0"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def run_stem(args: argparse.Namespace) -> str:
    """Bind run identity to arm, fold, sample-order seed, and configuration."""

    ablation_id = getattr(args, "ablation_id", FULL_MODEL_ID)
    fold_id = normalize_fold_id(getattr(args, "fold_id", None))
    fold_suffix = "" if fold_id is None else f"_fold{fold_id}"
    data_order_seed = int(getattr(args, "data_order_seed", args.seed))
    computed_configuration_sha256 = canonical_configuration_sha256(args)
    recorded_configuration_sha256 = getattr(args, "configuration_sha256", None)
    if (
        recorded_configuration_sha256 is not None
        and recorded_configuration_sha256 != computed_configuration_sha256
    ):
        raise ValueError("configuration_sha256 does not match the resolved configuration")
    configuration_sha256 = computed_configuration_sha256
    return (
        f"{args.data}_{args.variant}_{ablation_id}{fold_suffix}_seed{args.seed}"
        f"_order{data_order_seed}_cfg{configuration_sha256}"
    )


def existing_exact_run(
    checkpoint_path: Path,
    result_path: Path,
    expected_parameters: dict[str, Any],
) -> dict[str, Any] | None:
    """Return an exact prior run or fail closed on partial/mismatched output."""

    for path in (checkpoint_path, result_path):
        if path.is_symlink():
            raise FileExistsError("refusing symlinked public run output")
        if path.exists() and not path.is_file():
            raise FileExistsError("refusing non-file public run output")
    checkpoint_exists = checkpoint_path.exists()
    result_exists = result_path.exists()
    if not checkpoint_exists and not result_exists:
        return None
    if checkpoint_exists != result_exists:
        raise FileExistsError(
            "refusing partial public run output; checkpoint and receipt must both exist"
        )

    expected_digest = expected_parameters.get("configuration_sha256")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FileExistsError("refusing unreadable existing public run receipt") from error
    if not isinstance(result, dict):
        raise FileExistsError("refusing malformed existing public run receipt")
    result_protocol = result.get("protocol")
    if not isinstance(result_protocol, dict):
        raise FileExistsError("refusing malformed existing public run receipt")
    result_parameters = result_protocol.get("parameters")
    if (
        result.get("configuration_sha256") != expected_digest
        or result.get("schema") != "cerd-run-v2"
        or result.get("status") != "complete"
        or result.get("dataset") != expected_parameters.get("data")
        or result.get("variant") != expected_parameters.get("variant")
        or result.get("ablation_id") != expected_parameters.get("ablation_id")
        or result.get("fold_id") != expected_parameters.get("fold_id")
        or result.get("seed") != expected_parameters.get("seed")
        or result.get("data_order_seed")
        != expected_parameters.get("data_order_seed")
        or result.get("checkpoint") != checkpoint_path.name
        or result_protocol.get("configuration_schema")
        != PUBLIC_CONFIGURATION_SCHEMA
        or result_parameters != expected_parameters
    ):
        raise FileExistsError(
            "refusing existing public run output with a different configuration"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise FileExistsError("refusing unreadable existing public checkpoint") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("args") != expected_parameters
    ):
        raise FileExistsError(
            "refusing existing public checkpoint with a different configuration"
        )
    checkpoint_sha256 = result.get("checkpoint_sha256")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_sha256)) is None
        or sha256_file(checkpoint_path) != checkpoint_sha256
    ):
        raise FileExistsError("refusing existing public checkpoint with a bad digest")
    return result


def create_run_claim(claim_path: Path, configuration_sha256: str) -> None:
    """Atomically reserve a stem so concurrent jobs cannot both start training."""

    if claim_path.is_symlink() or claim_path.exists():
        raise FileExistsError("public run stem is already claimed")
    try:
        with claim_path.open("x", encoding="ascii") as claim_file:
            claim_file.write(f"{configuration_sha256}\n")
    except FileExistsError as error:
        raise FileExistsError("public run stem is already claimed") from error


def publish_exclusive(temp_path: Path, final_path: Path) -> None:
    """Publish a flushed same-directory temporary file without overwriting."""

    if final_path.is_symlink() or final_path.exists():
        raise FileExistsError("refusing to overwrite an existing public run output")
    try:
        os.link(temp_path, final_path, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite an existing public run output"
        ) from error
    temp_path.unlink()


def encode_batch(
    samples: dict[str, torch.Tensor],
    observed: torch.Tensor,
    encoders: dict[str, torch.nn.Module],
    modality_dict: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> list[torch.Tensor]:
    tokens: list[torch.Tensor] = []
    for modality, modality_index in sorted(modality_dict.items(), key=lambda item: item[1]):
        values = samples[modality].to(device, non_blocking=True)
        active = observed[:, modality_index]
        encoded = torch.zeros(
            values.shape[0], args.num_patches, args.hidden_dim, device=device
        )
        if bool(active.any()):
            encoded[active] = encoders[modality](values[active])
        tokens.append(encoded)
    return tokens


def forward_batch(
    model: AGMGFlexMoE,
    encoders: dict[str, torch.nn.Module],
    modality_dict: dict[str, int],
    batch: tuple[Any, ...],
    args: argparse.Namespace,
    device: torch.device,
    *,
    reconstruction: bool,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    samples, labels, combinations, observed = batch[:4]
    labels = labels.to(device, non_blocking=True)
    combinations = combinations.to(device, non_blocking=True)
    observed = observed.to(device, non_blocking=True).bool()
    tokens = encode_batch(samples, observed, encoders, modality_dict, args, device)
    output = model(
        *tokens,
        observed_mask=observed,
        expert_indices=combinations,
        return_importance=False,
        return_recon_loss=reconstruction,
    )
    return output, tokens, labels, combinations, observed


def train_epoch(
    model: AGMGFlexMoE,
    encoders: dict[str, torch.nn.Module],
    modality_dict: dict[str, int],
    loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    branch_ema: BranchClassAccuracyEMA | None,
) -> dict[str, float]:
    model.train()
    for encoder in encoders.values():
        encoder.train()
    totals: dict[str, list[float]] = {}

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        output, tokens, labels, combinations, observed = forward_batch(
            model, encoders, modality_dict, batch, args, device, reconstruction=True
        )
        task = criterion(output["logits"], labels)
        gate = model.gate_loss()
        if not torch.is_tensor(gate):
            gate = task.new_tensor(float(gate))
        recon = output["recon_loss"] if output["recon_loss"] is not None else task.new_zeros(())
        branch_aux = branch_auxiliary_loss(
            output["branch_logits"],
            output["branch_mask"],
            output.get("branch_quality"),
            labels,
            criterion,
        )
        dbr = task.new_zeros(())
        if args.dual_boundary_rank_loss_weight > 0:
            dbr = dual_boundary_rank_loss(
                output["logits"],
                labels,
                margin=args.dual_boundary_rank_margin,
                class1_vs_0_weight=args.dual_boundary_rank_10_weight,
            )

        branch_distill = task.new_zeros(())
        if branch_ema is not None:
            context = branch_ema.context(
                output["branch_logits"], output["branch_mask"], labels
            )
            branch_ema.collect(output["branch_logits"], output["branch_mask"], labels)
            if epoch >= args.branch_distill_start_epoch:
                if args.variant == "mofe_tcl":
                    branch_distill = masked_branch_tcl_loss(
                        output["branch_logits"], context, args.branch_distill_temperature
                    )
                elif args.variant == "mofe_tbfd":
                    branch_distill = trusted_branch_fusion_distillation_loss(
                        output["logits"],
                        output["branch_logits"],
                        context,
                        args.branch_distill_temperature,
                    )

        dropped_observed = modality_dropout_mask(observed, args.modality_dropout_prob)
        dropped_combinations = combination_indices_from_mask(
            dropped_observed,
            args.modality,
            # The manifest-driven ABCD adapter canonicalizes modality codes
            # before enumerating combinations.  The legacy ADNI adapter keeps
            # the requested code order during enumeration.
            sort_codes_before_enumeration=args.data == "abcd",
        )
        dropped = model(
            *tokens,
            observed_mask=dropped_observed,
            expert_indices=dropped_combinations,
            return_importance=False,
            return_recon_loss=False,
        )
        drop_gate = model.gate_loss()
        if not torch.is_tensor(drop_gate):
            drop_gate = task.new_tensor(float(drop_gate))
        drop_ce = criterion(dropped["logits"], labels)
        distill = self_distillation_loss(
            output["logits"], dropped["logits"], args.distill_temperature
        )
        raw_mofe = task.new_zeros(())
        if args.more_fewer_rank_loss_weight > 0:
            raw_mofe = more_fewer_rank_loss(
                output["logits"],
                dropped["logits"],
                labels,
                observed,
                dropped_observed,
                criterion,
            )
        mofe = raw_mofe if args.use_more_fewer_objective else task.new_zeros(())

        loss = (
            task
            + args.gate_loss_weight * gate
            + args.recon_loss_weight * recon
            + args.branch_aux_loss_weight * branch_aux
            + 0.5 * args.gate_loss_weight * drop_gate
            + args.drop_ce_loss_weight * drop_ce
            + args.distill_loss_weight * distill
            + args.dual_boundary_rank_loss_weight * dbr
            + args.effective_more_fewer_rank_loss_weight * raw_mofe
            + args.branch_distill_loss_weight * branch_distill
        )
        loss.backward()
        parameters = list(model.parameters()) + [
            parameter for encoder in encoders.values() for parameter in encoder.parameters()
        ]
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()

        values = {
            "loss": loss,
            "task": task,
            "gate": gate,
            "recon": recon,
            "branch_aux": branch_aux,
            "drop_ce": drop_ce,
            "distill": distill,
            "dbr": dbr,
            "mofe": mofe,
            "branch_distill": branch_distill,
        }
        for name, value in values.items():
            totals.setdefault(name, []).append(float(value.detach()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


@torch.no_grad()
def evaluate(
    model: AGMGFlexMoE,
    encoders: dict[str, torch.nn.Module],
    modality_dict: dict[str, int],
    loader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    for encoder in encoders.values():
        encoder.eval()
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    for batch in loader:
        output, _, labels, _, _ = forward_batch(
            model, encoders, modality_dict, batch, args, device, reconstruction=False
        )
        # Consume and clear router diagnostics so no graph/cache crosses batches.
        gate = model.gate_loss()
        if torch.is_tensor(gate):
            gate.detach()
        probabilities_all.append(torch.softmax(output["logits"], dim=1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    return metric_bundle(labels, probabilities), labels, probabilities


def train(args: argparse.Namespace) -> dict[str, Any]:
    args = resolve_defaults(args)
    seed_everything(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    args.torch_device = device
    modality_dict = resolve_modality_dict(args)
    args.n_full_modalities = len(modality_dict)
    # These modules are constructed for full CERD and every aligned bypass,
    # including no_completion; the latter changes only execution.
    args.use_generators = True
    args.vectorized_generation = True
    args.recon_targets_per_sample = len(modality_dict)
    args.generator_task_grad = False
    args.configuration_sha256 = canonical_configuration_sha256(args)
    expected_parameters = public_protocol_parameters(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = run_stem(args)
    checkpoint_path = output_dir / f"{stem}.pt"
    result_path = output_dir / f"{stem}.json"
    claim_path = output_dir / f".{stem}.claim"
    checkpoint_temp_path = output_dir / f".{stem}.pt.tmp"
    result_temp_path = output_dir / f".{stem}.json.tmp"
    existing_result = existing_exact_run(
        checkpoint_path,
        result_path,
        expected_parameters,
    )
    if existing_result is not None:
        print(
            json.dumps(
                {
                    "result": result_path.name,
                    "status": "exact configuration already complete",
                },
                indent=2,
            )
        )
        return existing_result
    for temp_path in (checkpoint_temp_path, result_temp_path):
        if temp_path.is_symlink() or temp_path.exists():
            raise FileExistsError("refusing stale public run temporary output")
    create_run_claim(claim_path, args.configuration_sha256)
    (
        data_dict,
        encoders,
        labels,
        train_ids,
        validation_ids,
        test_ids,
        num_classes,
        input_dims,
        transforms,
        masks,
        observed,
        full_modality_index,
    ) = load_and_preprocess_data(args, modality_dict)
    validate_objective_compatibility(args, num_classes)
    train_sorted, train_shuffled, validation_loader, test_loader = create_loaders(
        data_dict,
        observed,
        labels,
        train_ids,
        validation_ids,
        test_ids,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        input_dims,
        transforms,
        masks,
        args.preprocessed,
        args.use_common_ids,
        args.data_order_seed,
    )
    train_balanced = balanced_train_loader(
        train_shuffled,
        args.sampler_power,
        args.data_order_seed,
    )

    model = AGMGFlexMoE(
        num_modalities=len(modality_dict),
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
        pattern_aware_reconstruction=args.pattern_aware_reconstruction,
        recon_normalized_token_loss_weight=(
            args.recon_normalized_token_loss_weight
        ),
        recon_context_dropout_probability=(
            args.recon_context_dropout_probability
        ),
        vectorized_generation=args.vectorized_generation,
        recon_targets_per_sample=args.recon_targets_per_sample,
        use_generators=args.use_generators,
        dynamic_branch_fusion=False,
        branch_confidence_mode=args.branch_confidence_mode,
        generator_task_grad=args.generator_task_grad,
        generator_only_task_grad=args.generator_only_task_grad,
        generator_output_gate=args.generator_output_gate,
        use_sparse_moe_backbone=args.use_sparse_moe_backbone,
        use_provenance_embeddings=args.use_provenance_embeddings,
        use_reliability_branch_weights=args.use_reliability_branch_weights,
        use_attentive_token_pooling=args.use_attentive_token_pooling,
        use_stochastic_reconstruction_context=(
            args.use_stochastic_reconstruction_context
        ),
        use_latent_completion=args.use_latent_completion,
    ).to(device)

    parameters = list(model.parameters()) + [
        parameter for encoder in encoders.values() for parameter in encoder.parameters()
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    train_labels = np.asarray(labels)[np.asarray(train_ids)]
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    prior = counts / counts.sum()
    weights = (
        class_weights(labels, train_ids, args.class_weight_power, device)
        if args.class_weight_power > 0
        else None
    )
    criterion = LogitAdjustedCrossEntropy(
        prior,
        tau=args.logit_adjust_tau,
        label_smoothing=args.label_smoothing,
        weight=weights,
    ).to(device)
    branch_ema = (
        BranchClassAccuracyEMA(num_classes, args.branch_accuracy_ema_decay)
        if args.variant in {"mofe_tcl", "mofe_tbfd"}
        else None
    )

    selection_name = "macro_auprc" if num_classes == 2 else "macro_f1"
    best_selection_score = -1.0
    best_epoch = 0
    best_model: dict[str, torch.Tensor] | None = None
    best_encoders: dict[str, dict[str, torch.Tensor]] | None = None
    best_validation: dict[str, float] | None = None
    started = time.perf_counter()

    for epoch_index in trange(
        args.train_epochs,
        desc=f"{args.data}/{args.variant}/{args.ablation_id}/seed{args.seed}",
    ):
        epoch = epoch_index + 1
        loader = train_sorted if epoch <= args.warm_up_epochs else train_balanced
        losses = train_epoch(
            model,
            encoders,
            modality_dict,
            loader,
            criterion,
            optimizer,
            args,
            device,
            epoch,
            branch_ema,
        )
        if branch_ema is not None:
            branch_ema.update()
        validation, _, _ = evaluate(
            model, encoders, modality_dict, validation_loader, args, device
        )
        selection_score = float(validation[selection_name])
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch
            best_validation = copy.deepcopy(validation)
            best_model = cpu_state(model)
            best_encoders = {name: cpu_state(module) for name, module in encoders.items()}
        print(
            f"epoch={epoch:02d} loss={losses['loss']:.4f} "
            f"val_acc={float(validation['accuracy']):.4f} "
            f"val_macro_f1={float(validation['macro_f1']):.4f} "
            f"val_auprc={float(validation['macro_auprc']):.4f}"
        )

    if best_model is None or best_encoders is None or best_validation is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_model, strict=True)
    for name, encoder in encoders.items():
        encoder.load_state_dict(best_encoders[name], strict=True)
    replay_validation, validation_labels, validation_probabilities = evaluate(
        model, encoders, modality_dict, validation_loader, args, device
    )
    if abs(float(replay_validation[selection_name]) - best_selection_score) > 1e-12:
        raise RuntimeError("best-checkpoint validation replay mismatch")

    binary_decision = None
    if num_classes == 2:
        decision_threshold, validation_decision_f1 = tune_binary_threshold(
            validation_labels,
            validation_probabilities,
        )
        binary_decision = {
            "threshold": decision_threshold,
            "selection_metric": "validation macro-F1",
            "selection_score": validation_decision_f1,
        }

    result: dict[str, Any] = {
        "schema": "cerd-run-v2",
        "status": "complete",
        "dataset": args.data,
        "variant": args.variant,
        "ablation_id": args.ablation_id,
        "seed": args.seed,
        "fold_id": args.fold_id,
        "data_order_seed": args.data_order_seed,
        "configuration_sha256": args.configuration_sha256,
        "protocol": {
            "configuration_schema": PUBLIC_CONFIGURATION_SCHEMA,
            "checkpoint_selection": f"validation {selection_name}",
            "prediction_rule": (
                "validation-selected probability threshold"
                if num_classes == 2
                else "raw softmax argmax"
            ),
            "epochs": args.train_epochs,
            "test_used_for_selection": False,
            "parameters": expected_parameters,
        },
        "best_epoch": best_epoch,
        "validation": replay_validation,
        "binary_decision": binary_decision,
        "test": None,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint_path.name,
    }
    if args.evaluate_test:
        test, test_labels, test_probabilities = evaluate(
            model, encoders, modality_dict, test_loader, args, device
        )
        test_predictions = test_probabilities.argmax(axis=1)
        if num_classes == 2:
            if binary_decision is None:
                raise RuntimeError("binary decision threshold was not initialized")
            test_predictions = binary_predictions_at_threshold(
                test_probabilities,
                float(binary_decision["threshold"]),
            )
            test = metric_bundle(test_labels, test_probabilities, test_predictions)
        result["test"] = test
    raced_result = existing_exact_run(
        checkpoint_path,
        result_path,
        expected_parameters,
    )
    if raced_result is not None:
        claim_path.unlink()
        return raced_result
    with checkpoint_temp_path.open("xb") as checkpoint_file:
        torch.save(
            {
                "model": best_model,
                "encoders": best_encoders,
                # Keep replay metadata while excluding manifests, roots,
                # devices, and output paths.
                "args": public_protocol_parameters(args),
                "best_epoch": best_epoch,
                "validation": replay_validation,
            },
            checkpoint_file,
        )
        checkpoint_file.flush()
        os.fsync(checkpoint_file.fileno())
    result["checkpoint_sha256"] = sha256_file(checkpoint_temp_path)
    publish_exclusive(checkpoint_temp_path, checkpoint_path)
    with result_temp_path.open("x", encoding="utf-8") as result_file:
        json.dump(
            result,
            result_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        result_file.write("\n")
        result_file.flush()
        os.fsync(result_file.fileno())
    publish_exclusive(result_temp_path, result_path)
    claim_path.unlink()
    print(
        json.dumps(
            {
                "result": result_path.name,
                "validation": replay_validation,
                "test": result["test"],
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    train(parse_args())
