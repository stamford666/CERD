#!/usr/bin/env python3
"""Train the CERD main method on ABCD or ADNI.

The runner contains no baseline adapters. Binary ABCD-ADHD checkpoints are selected
by validation AUPRC and their decision threshold is calibrated on validation
only; multiclass checkpoints are selected by validation Macro-F1.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

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
from cerd.sampling import balanced_train_loader, class_weights


VARIANTS = ("auto", "core", "dbr", "mofe", "unified", "mofe_tcl", "mofe_tbfd")


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
    if getattr(args, "modality", None) is None:
        args.modality = "SRDGNPME" if args.data == "abcd" else "IGCB"
    if args.variant == "auto":
        # DBR encodes the two boundaries around the middle class and is only
        # defined for a three-class endpoint.  MoFe is valid for both the new
        # binary ABCD endpoint and the three-class ADNI endpoint.
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
        mofe = task.new_zeros(())
        if args.more_fewer_rank_loss_weight > 0:
            mofe = more_fewer_rank_loss(
                output["logits"],
                dropped["logits"],
                labels,
                observed,
                dropped_observed,
                criterion,
            )

        loss = (
            task
            + args.gate_loss_weight * gate
            + args.recon_loss_weight * recon
            + args.branch_aux_loss_weight * branch_aux
            + 0.5 * args.gate_loss_weight * drop_gate
            + args.drop_ce_loss_weight * drop_ce
            + args.distill_loss_weight * distill
            + args.dual_boundary_rank_loss_weight * dbr
            + args.more_fewer_rank_loss_weight * mofe
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
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
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
    )
    train_balanced = balanced_train_loader(train_shuffled, args.sampler_power, args.seed)

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
        vectorized_generation=True,
        recon_targets_per_sample=len(modality_dict),
        use_generators=True,
        dynamic_branch_fusion=False,
        branch_confidence_mode=args.branch_confidence_mode,
        generator_task_grad=False,
        generator_only_task_grad=args.generator_only_task_grad,
        generator_output_gate=args.generator_output_gate,
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

    selection_name = "positive_auprc" if num_classes == 2 else "macro_f1"
    best_selection_score = -1.0
    best_epoch = 0
    best_model: dict[str, torch.Tensor] | None = None
    best_encoders: dict[str, dict[str, torch.Tensor]] | None = None
    best_validation: dict[str, float | int] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch_index in trange(args.train_epochs, desc=f"{args.data}/{args.variant}/seed{args.seed}"):
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
        ema_values = branch_ema.update() if branch_ema is not None else None
        validation, _, _ = evaluate(
            model, encoders, modality_dict, validation_loader, args, device
        )
        history.append(
            {"epoch": epoch, "losses": losses, "validation": validation, "class_accuracy_ema": ema_values}
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
        validation_predictions = binary_predictions_at_threshold(
            validation_probabilities,
            decision_threshold,
        )
        binary_decision = {
            "threshold": decision_threshold,
            "selection_metric": "validation macro-F1",
            "selection_score": validation_decision_f1,
            "validation_metrics": metric_bundle(
                validation_labels,
                validation_probabilities,
                validation_predictions,
            ),
        }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.data}_{args.variant}_seed{args.seed}"
    checkpoint_path = output_dir / f"{stem}.pt"
    torch.save(
        {
            "model": best_model,
            "encoders": best_encoders,
            "args": vars(args),
            "best_epoch": best_epoch,
            "validation": replay_validation,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        output_dir / f"{stem}_validation.npz",
        labels=validation_labels,
        probabilities=validation_probabilities,
    )

    result: dict[str, Any] = {
        "schema": "cerd-run-v1",
        "dataset": args.data,
        "variant": args.variant,
        "seed": args.seed,
        "protocol": {
            "checkpoint_selection": f"validation {selection_name}",
            "prediction_rule": (
                "validation-selected probability threshold"
                if num_classes == 2
                else "raw softmax argmax"
            ),
            "epochs": args.train_epochs,
            "test_used_for_selection": False,
        },
        "best_epoch": best_epoch,
        "validation": replay_validation,
        "binary_decision": binary_decision,
        "test": None,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint_path.name,
        "history": history,
        "config": {key: value for key, value in vars(args).items() if key != "torch_device"},
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
        np.savez_compressed(
            output_dir / f"{stem}_test.npz",
            labels=test_labels,
            probabilities=test_probabilities,
            predictions=test_predictions,
        )
        if args.data == "abcd" and num_classes == 2:
            np.savez_compressed(
                output_dir / f"cerd_seed{args.seed}.npz",
                method=np.asarray("CERD"),
                seed=np.asarray(args.seed, dtype=np.int64),
                experiment_tag=np.asarray("abcd_adhd"),
                validation_labels=validation_labels,
                validation_probabilities=validation_probabilities,
                test_labels=test_labels,
                test_probabilities=test_probabilities,
                test_predictions=test_predictions,
                decision_threshold=np.asarray(
                    binary_decision["threshold"], dtype=np.float64
                ),
            )
    result_path = output_dir / f"{stem}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"result": str(result_path), "validation": replay_validation, "test": result["test"]}, indent=2))
    return result


if __name__ == "__main__":
    train(parse_args())
