"""Safely unlock test evaluation for a validation-only baseline checkpoint.

The source artifact must have ``status=validation_complete``.  This program
reconstructs the exact data pipeline, encoders, and model from the checkpoint,
strictly reloads every state tensor, and replays validation first.  Test is
evaluated only if the complete saved validation metric tree is reproduced
within a deliberately small tolerance.  Source artifacts are never modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from baseline_runner import (  # noqa: E402
    EMPIRICAL_MISSING_PATTERN_REPLAY_DERIVED_ARGS,
    EXTERNAL_TEACHER_TASK,
    EXTERNAL_TEACHER_TASK_KIND_BINDINGS,
    EXTERNAL_TEACHER_TEMPORAL_TASK,
    MODEL_CHOICES,
    MODEL_DISPLAY,
    acadiff_objective_protocol,
    adni_image_imputation_protocol,
    build_model,
    classification_generator_gradient_protocol,
    checkpoint_soup_protocol,
    clean_dynamic_router_protocol,
    cross_fitted_external_teacher_protocol,
    cross_fitted_tree_teacher_protocol,
    dual_local_boundary_residual_protocol,
    empirical_missing_pattern_replay_active_objectives,
    empirical_missing_pattern_replay_protocol,
    empirical_missing_pattern_replay_training_audit,
    empirical_missing_pattern_training_observed,
    evaluation_gate_cache_protocol,
    hard_cvar_dual_boundary_protocol,
    joint_head_ensemble_protocol,
    low_rank_patch_adapter_protocol,
    masked_branch_tcl_protocol,
    metric_bundle,
    missing_capacity_residual_protocol,
    missing_family_router_protocol,
    more_fewer_rank_protocol,
    more_tail_adapter_protocol,
    presentation_axis_residual_protocol,
    reconstruction_encoder_gradient_protocol,
    rdrop_training_protocol,
    result_paths,
    run_epoch,
    sam_training_protocol,
    sampler_curriculum_protocol,
    seed_everything,
    subset_metrics,
    training_class_prior,
    trusted_branch_fusion_distillation_protocol,
)
from data import (  # noqa: E402
    create_loaders,
    load_and_preprocess_data,
    resolve_modality_dict,
)


DEFAULT_METRIC_ATOL = 1e-10
MAX_METRIC_ATOL = 1e-8
RUNTIME_ONLY_CHECKPOINT_ARGS = {
    "allow_overwrite",
    "device",
    "output_dir",
    "seed",
    "torch_device",
}

LEGACY_CHECKPOINT_SCHEMA_V1 = "legacy-core-v1"
PRODUCTION_CHECKPOINT_SCHEMA_V2 = "production-validation-metadata-v2"
AUDITED_PRODUCTION_CHECKPOINT_SCHEMA_V3 = (
    "audited-production-validation-metadata-v3"
)
LEGACY_CHECKPOINT_KEYS = frozenset(
    {"model", "encoders", "args", "best_epoch"}
)
PRODUCTION_CHECKPOINT_KEYS = frozenset(
    {
        *LEGACY_CHECKPOINT_KEYS,
        "best_validation_selection_score",
        "checkpoint_selection_metric",
        "decision_protocol",
    }
)
AUDITED_PRODUCTION_CHECKPOINT_KEYS = frozenset(
    {
        *PRODUCTION_CHECKPOINT_KEYS,
        "matched_ablation_audit",
        "sample_order_receipt",
    }
)

LEGACY_SOURCE_PROTOCOL_V1 = "legacy-raw-argmax-v1"
PRODUCTION_SOURCE_PROTOCOL_V2 = "selection-metadata-v2"
RAW_ARGMAX_DECISION_PROTOCOL_V1 = {
    "implementation_revision": "raw_argmax_v1",
    "selection_scope": None,
    "prediction_rule": "softmax argmax",
    "test_labels_used": False,
    "test_probability_distribution_used": False,
}


class IntegrityError(RuntimeError):
    """Raised before test evaluation when the validation artifact is unsafe."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-result",
        required=True,
        type=Path,
        help="Path to a baseline_runner JSON with status=validation_complete.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Separate formal result root; source and existing files are never overwritten.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ordinal used for deterministic validation replay and test evaluation.",
    )
    parser.add_argument(
        "--metric-atol",
        type=float,
        default=DEFAULT_METRIC_ATOL,
        help=f"Absolute validation metric tolerance (maximum {MAX_METRIC_ATOL:g}).",
    )
    parser.add_argument(
        "--expected-configuration-sha256",
        default=None,
        help="Optional locked configuration digest; mismatch aborts before test evaluation.",
    )
    return parser.parse_args()


def _reject_json_constant(value: str) -> None:
    raise IntegrityError(f"Non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"JSON file does not exist: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"Could not read strict JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise IntegrityError(f"Expected a JSON object in {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise IntegrityError(
            f"{name} mismatch: found {actual!r} ({type(actual).__name__}), "
            f"expected {expected!r} ({type(expected).__name__})"
        )


def require_numeric_equal(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        require_equal(name, actual, expected)
        return
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise IntegrityError(f"{name} must be numeric; found {actual!r}")
    if not math.isfinite(float(actual)) or float(actual) != float(expected):
        raise IntegrityError(f"{name} mismatch: found {actual!r}, expected {expected!r}")


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{name} must be a mapping")
    return value


def resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_configuration_args(args_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a seed/runtime-independent experiment configuration."""

    normalized = {
        key: value
        for key, value in args_payload.items()
        if key not in RUNTIME_ONLY_CHECKPOINT_ARGS
    }
    # These values are derived audit state bound to the current fold/seed, not
    # a seed-independent design choice. The material opt-in boolean remains in
    # the fingerprint while counts, ordered ID-mask binding, order, and private
    # RNG seed are verified separately and excluded here.
    for field_name in EMPIRICAL_MISSING_PATTERN_REPLAY_DERIVED_ARGS:
        normalized.pop(field_name, None)
    # This parser-level default is material only for ACADiff.  Excluding it
    # from every other model preserves their pre-extension fingerprints, while
    # retaining it for ACADiff distinguishes corrected observed-only training
    # from historical checkpoints whose argument is absent.
    if normalized.get("model") != "acadiff":
        normalized.pop("acadiff_mask_probability", None)
    # Historical artifacts predate this opt-in flag and therefore have the
    # exact False behavior.  Materialize it to keep their digest comparable.
    normalized.setdefault("class_weighted_quality_branch_aux", False)
    # These later opt-in extensions are exact no-ops at their defaults.  Omit
    # neutral values from the v1 fingerprint so an old checkpoint with absent
    # fields and a new checkpoint with explicit defaults remain the same
    # experiment, while every non-default structural candidate stays distinct.
    neutral_extensions = {
        "ordinal_aux_loss_weight": 0.0,
        "ordinal_head_type": "proportional",
        "uncertainty_aware_ordinal_fusion": False,
        "learn_observed_reliability": False,
        "centered_evidence_confidence": False,
        "class_conditional_fusion": False,
        "gated_transformer_residual": False,
        "pattern_aware_reconstruction": False,
        "recon_normalized_token_loss_weight": 0.0,
        "recon_context_dropout_probability": 0.0,
        "recon_encoder_gradient_scale": 1.0,
        "logit_adjust_tau": 0.0,
        "class1_aux_loss_weight": 0.0,
        "exclude_bias_norm_from_weight_decay": False,
        "sampler_ramp_epochs": 0,
        "checkpoint_soup_top_k": 1,
        "checkpoint_selection_policy": "validation_best",
        "adni_image_imputation": "legacy_mode",
        "normalized_gate_loss": False,
        "sam_rho": 0.0,
        "rdrop_loss_weight": 0.0,
        "joint_head_ensemble_size": 1,
        "patch_adapter_rank": 0,
        "more_fewer_rank_loss_weight": 0.0,
        "more_tail_rank": 0,
        "more_tail_peak_amplitude": 0.0,
        "clear_eval_gate_cache": False,
        "dual_local_boundary_loss_weight": 0.0,
        "presentation_axis_loss_weight": 0.0,
        "presentation_axis_residual_cap": 0.5,
        "missing_capacity_residual_width": 0,
        "missing_family_normalized_router": False,
        "missing_family_residual_gate": False,
        "dynamic_branch_joint_prior_boost": True,
        "dynamic_branch_use_observed_mask": False,
        "prediction_observed_specialists_only": False,
        "dynamic_branch_quality_weighted_aux": False,
        "supervised_router_observed_specialists_only": False,
        "generator_only_task_grad": False,
        "generator_output_gate": True,
        "empirical_missing_pattern_replay": False,
    }
    for key, neutral_value in neutral_extensions.items():
        if normalized.get(key) == neutral_value:
            normalized.pop(key)
    if normalized.get("more_tail_rank", 0) == 0:
        # A rank-zero checkpoint constructs no adapter and never materializes a
        # training prior.  Remove the whole family for historical fingerprints.
        normalized.pop("more_tail_rank", None)
        normalized.pop("more_tail_peak_amplitude", None)
        normalized.pop("more_tail_class_prior", None)
    if normalized.get("supervised_router_loss_weight", 0.0) == 0.0:
        # The router temperature has no material effect while its loss is off.
        # Remove both fields so parser-extended checkpoints keep historical
        # configuration fingerprints exactly.
        normalized.pop("supervised_router_loss_weight", None)
        normalized.pop("supervised_router_temperature", None)
    if normalized.get("supervised_contrastive_loss_weight", 0.0) == 0.0:
        # Temperature and projection width are immaterial when no projection
        # head is instantiated and no contrastive objective is evaluated.
        normalized.pop("supervised_contrastive_loss_weight", None)
        normalized.pop("supervised_contrastive_temperature", None)
        normalized.pop("supervised_contrastive_projection_dim", None)
    if normalized.get("dual_boundary_rank_loss_weight", 0.0) == 0.0:
        # Margin and boundary mixture are immaterial while the training-only
        # ranking auxiliary is disabled.
        normalized.pop("dual_boundary_rank_loss_weight", None)
        normalized.pop("dual_boundary_rank_margin", None)
        normalized.pop("dual_boundary_rank_10_weight", None)
    if normalized.get("hard_cvar_dual_boundary_max_weight", 0.0) == 0.0:
        # Tail definition and epoch schedule are immaterial while hard-CVaR is
        # disabled; remove the whole family for old-checkpoint fingerprinting.
        normalized.pop("hard_cvar_dual_boundary_max_weight", None)
        normalized.pop("hard_cvar_dual_boundary_tail_fraction", None)
        normalized.pop("hard_cvar_dual_boundary_margin", None)
        normalized.pop("hard_cvar_dual_boundary_10_weight", None)
        normalized.pop("hard_cvar_dual_boundary_start_epoch", None)
        normalized.pop("hard_cvar_dual_boundary_ramp_epochs", None)
    if normalized.get("tree_teacher_distill_weight", 0.0) == 0.0:
        # No artifact is opened and no soft-target term is evaluated while the
        # candidate is disabled.  Remove the whole family so explicit parser
        # defaults retain historical checkpoint fingerprints.
        normalized.pop("tree_teacher_distill_weight", None)
        normalized.pop("tree_teacher_temperature", None)
        normalized.pop("tree_teacher_npz", None)
        normalized.pop("tree_teacher_npz_sha256", None)
    if normalized.get("external_teacher_distill_weight", 0.0) == 0.0:
        # The v2 artifact, trust gate, and derived full-OOF statistics are all
        # immaterial while disabled.  Remove the complete family so old and
        # explicit-default checkpoints retain the same fingerprint.
        for field_name in (
            "external_teacher_distill_weight",
            "external_teacher_temperature",
            "external_teacher_npz",
            "external_teacher_npz_sha256",
            "external_teacher_trust_mode",
            "external_teacher_class_compensation",
            "external_teacher_kind",
            "external_teacher_task",
            "external_teacher_generator_protocol_sha256",
            "external_teacher_class_recalls",
            "external_teacher_class_compensation_weights",
            "external_teacher_correct_coverage",
        ):
            normalized.pop(field_name, None)
    # Final branch-accuracy EMA values are training audit state, never a
    # seed-independent configuration dimension.
    for field_name in (
        "masked_branch_tcl_final_class_accuracy_ema",
        "masked_branch_tcl_final_class_initialized",
        "masked_branch_tcl_epochs_updated",
        "masked_branch_tcl_ema_history_json",
    ):
        normalized.pop(field_name, None)
    if normalized.get("masked_branch_tcl_loss_weight", 0.0) == 0.0:
        # No state is constructed, no statistics are collected, and no loss is
        # evaluated while disabled.  Remove the complete family for historical
        # checkpoint fingerprints.
        for field_name in (
            "masked_branch_tcl_loss_weight",
            "masked_branch_tcl_temperature",
            "masked_branch_tcl_start_epoch",
            "masked_branch_tcl_class_acc_ema",
        ):
            normalized.pop(field_name, None)
    for field_name in (
        "trusted_branch_fusion_final_class_accuracy_ema",
        "trusted_branch_fusion_final_class_initialized",
        "trusted_branch_fusion_epochs_updated",
        "trusted_branch_fusion_ema_history_json",
    ):
        normalized.pop(field_name, None)
    if normalized.get("trusted_branch_fusion_distill_weight", 0.0) == 0.0:
        for field_name in (
            "trusted_branch_fusion_distill_weight",
            "trusted_branch_fusion_temperature",
            "trusted_branch_fusion_start_epoch",
            "trusted_branch_fusion_class_acc_ema",
        ):
            normalized.pop(field_name, None)
    if normalized.get("lr_scheduler", "constant") == "constant":
        normalized.pop("lr_scheduler", None)
        normalized.pop("lr_warmup_epochs", None)
        normalized.pop("min_lr_ratio", None)
    if not normalized.get("use_ema", False):
        normalized.pop("use_ema", None)
        normalized.pop("ema_decay", None)
        normalized.pop("ema_start_epoch", None)
    return normalized


_MORE_FEWER_COMMON_PROTOCOL_KEYS = {
    "implementation_revision", "enabled", "formula", "margin",
    "strict_subset", "view_source", "criterion", "gradient_scope",
    "empty_subset", "inference_change",
}
_MORE_FEWER_LEGACY_PROTOCOL_KEYS = _MORE_FEWER_COMMON_PROTOCOL_KEYS | {"loss_weight"}
_MORE_FEWER_CURRENT_PROTOCOL_KEYS = _MORE_FEWER_COMMON_PROTOCOL_KEYS | {
    "loss_weight", "configured_control_weight", "effective_contribution_weight",
    "reduced_view_forward_retained",
}


def canonical_more_fewer_rank_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the exact, known pre-effective-weight protocol schema."""
    protocol = dict(payload)
    keys = set(protocol)
    if keys == _MORE_FEWER_CURRENT_PROTOCOL_KEYS:
        return protocol
    if keys != _MORE_FEWER_LEGACY_PROTOCOL_KEYS:
        return protocol
    weight = protocol["loss_weight"]
    if (
        type(weight) not in {int, float}
        or not math.isfinite(float(weight))
        or float(weight) < 0.0
        or protocol.get("enabled") is not (float(weight) > 0.0)
    ):
        raise IntegrityError("Invalid legacy more-fewer rank protocol")
    weight = float(weight)
    protocol["configured_control_weight"] = weight
    protocol["effective_contribution_weight"] = weight
    protocol["reduced_view_forward_retained"] = weight > 0.0
    return protocol


def resolve_replay_manifest(value: Any) -> str | None:
    """Resolve a saved manifest using the legacy runner's possible CWDs."""

    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        require(path.is_file(), f"Dataset manifest does not exist: {path}")
        return str(path.resolve())
    candidates = {
        (base / path).resolve()
        for base in (Path.cwd(), HERE, HERE.parent)
        if (base / path).is_file()
    }
    require(
        len(candidates) == 1,
        f"Could not resolve saved relative dataset manifest {value!r} uniquely; "
        f"existing candidates={sorted(str(candidate) for candidate in candidates)}",
    )
    return str(next(iter(candidates)))


def resolve_replay_output_root(value: Any, source_path: Path) -> str:
    """Canonicalize a saved output root against its exact source location."""

    require(type(value) is str and value, "checkpoint.args.output_dir must be a path string")
    expected = source_path.parent.parent.resolve()
    path = Path(value).expanduser()
    candidates = (
        {path.resolve()}
        if path.is_absolute()
        else {(base / path).resolve() for base in (Path.cwd(), HERE, HERE.parent)}
    )
    require(
        expected in candidates,
        f"Could not resolve saved output_dir {value!r} to source root {expected}; "
        f"candidates={sorted(str(candidate) for candidate in candidates)}",
    )
    return str(expected)


def require_unchanged_digest(name: str, path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise IntegrityError(
            f"{name} changed during formal evaluation: "
            f"expected sha256={expected}, found sha256={actual}"
        )


def source_protocol_schema_version(source: Mapping[str, Any]) -> str:
    """Identify one exact result-protocol generation without fuzzy fallback."""

    protocol = require_mapping(source.get("protocol"), "source.protocol")
    has_selection_metric = "checkpoint_selection_metric" in protocol
    has_binary_decision = "binary_decision" in protocol
    if not has_selection_metric and not has_binary_decision:
        require_equal(
            "source.protocol.checkpoint_selection",
            protocol.get("checkpoint_selection"),
            "validation macro-F1",
        )
        require_equal(
            "source.protocol.prediction_rule",
            protocol.get("prediction_rule"),
            "raw softmax argmax; no validation/test calibration",
        )
        return LEGACY_SOURCE_PROTOCOL_V1

    require(
        has_selection_metric and has_binary_decision,
        "Source protocol mixes legacy and production selection fields",
    )
    selection_metric = protocol.get("checkpoint_selection_metric")
    require(
        type(selection_metric) is str
        and selection_metric in {"macro_f1", "macro_auprc", "positive_auprc"},
        "source.protocol.checkpoint_selection_metric is invalid",
    )
    selection_policy = protocol.get(
        "checkpoint_selection_policy", "validation_best"
    )
    require(
        selection_policy in {"validation_best", "final_epoch_refit"},
        "source.protocol.checkpoint_selection_policy is invalid",
    )
    expected_selection = (
        "final live state after exactly train_epochs; no checkpoint selection"
        if selection_policy == "final_epoch_refit"
        else {
            "macro_f1": "validation Macro-F1",
            "macro_auprc": "validation Macro-AUPRC",
            "positive_auprc": "validation positive-class AUPRC",
        }[selection_metric]
    )
    require_equal(
        "source.protocol.checkpoint_selection",
        protocol.get("checkpoint_selection"),
        expected_selection,
    )
    if selection_metric == "positive_auprc":
        require_equal(
            "source.protocol.prediction_rule",
            protocol.get("prediction_rule"),
            "fixed validation-selected positive-class probability threshold",
        )
        require_mapping(
            protocol.get("binary_decision"), "source.protocol.binary_decision"
        )
    else:
        require_equal(
            "source.protocol.prediction_rule",
            protocol.get("prediction_rule"),
            "raw softmax argmax",
        )
        require_equal(
            "source.protocol.binary_decision",
            protocol.get("binary_decision"),
            None,
        )
    return PRODUCTION_SOURCE_PROTOCOL_V2


def checkpoint_schema_for_source(source: Mapping[str, Any]) -> str:
    """Bind the checkpoint container generation to the source generation.

    This compatibility decision covers only the checkpoint key/selection-
    metadata schema.  Independently versioned replay protocols (for example,
    checkpoint soup) remain subject to their own exact validators below.
    """

    source_version = source_protocol_schema_version(source)
    base_schema = {
        LEGACY_SOURCE_PROTOCOL_V1: LEGACY_CHECKPOINT_SCHEMA_V1,
        PRODUCTION_SOURCE_PROTOCOL_V2: PRODUCTION_CHECKPOINT_SCHEMA_V2,
    }[source_version]
    has_ablation_audit = "matched_ablation_audit" in source
    has_order_receipt = "sample_order_receipts" in source
    require(
        has_ablation_audit == has_order_receipt,
        "Source mixes audited and unaudited checkpoint metadata fields",
    )
    if has_ablation_audit:
        require_equal(
            "audited source protocol generation",
            source_version,
            PRODUCTION_SOURCE_PROTOCOL_V2,
        )
        return AUDITED_PRODUCTION_CHECKPOINT_SCHEMA_V3
    return base_schema


def validate_source_schema(source: Mapping[str, Any], source_path: Path) -> None:
    require_equal("source.status", source.get("status"), "validation_complete")
    require(source.get("model") in MODEL_CHOICES, f"Unknown source model: {source.get('model')!r}")
    require(source.get("dataset") in {"abcd", "adni"}, "Source dataset must be abcd or adni")
    require(type(source.get("seed")) is int, "source.seed must be an integer")
    require_equal(
        "source.model_display",
        source.get("model_display"),
        MODEL_DISPLAY[source["model"]],
    )
    require(source.get("test") is None, "Validation-only source unexpectedly contains test metrics")
    require(source.get("test_subsets") is None, "Validation-only source unexpectedly contains test subsets")

    protocol = require_mapping(source.get("protocol"), "source.protocol")
    source_protocol_schema_version(source)
    require_equal("source.protocol.full_training_not_smoke", protocol.get("full_training_not_smoke"), True)
    require_mapping(source.get("data"), "source.data")
    require_mapping(source.get("training"), "source.training")
    require_mapping(source.get("validation"), "source.validation")
    artifacts = require_mapping(source.get("artifacts"), "source.artifacts")
    require(artifacts.get("checkpoint") is not None, "Source checkpoint artifact is missing")
    require(artifacts.get("progress") is not None, "Source progress artifact is missing")
    require(artifacts.get("predictions") is None, "Validation-only source must not have predictions")
    require(source_path.is_file(), f"Source result is not a regular file: {source_path}")


def checkpoint_schema_version(checkpoint: Mapping[str, Any]) -> str:
    """Return the exact checkpoint generation; partial hybrids are forbidden."""

    found_keys = frozenset(checkpoint)
    if found_keys == AUDITED_PRODUCTION_CHECKPOINT_KEYS:
        return AUDITED_PRODUCTION_CHECKPOINT_SCHEMA_V3
    if found_keys == PRODUCTION_CHECKPOINT_KEYS:
        return PRODUCTION_CHECKPOINT_SCHEMA_V2
    if found_keys == LEGACY_CHECKPOINT_KEYS:
        return LEGACY_CHECKPOINT_SCHEMA_V1
    found_display = sorted((repr(key) for key in found_keys))
    raise IntegrityError(
        "Checkpoint keys do not match an exact supported schema: "
        f"found={found_display}; "
        f"{AUDITED_PRODUCTION_CHECKPOINT_SCHEMA_V3}="
        f"{sorted(AUDITED_PRODUCTION_CHECKPOINT_KEYS)}; "
        f"{PRODUCTION_CHECKPOINT_SCHEMA_V2}="
        f"{sorted(PRODUCTION_CHECKPOINT_KEYS)}; "
        f"{LEGACY_CHECKPOINT_SCHEMA_V1}={sorted(LEGACY_CHECKPOINT_KEYS)}"
    )


def _validate_flat_scalar_mapping(value: Any, name: str) -> Mapping[str, Any]:
    mapping = require_mapping(value, name)
    for key, item in mapping.items():
        require(type(key) is str, f"Every {name} key must be a string")
        require(
            item is None or type(item) in {bool, int, float, str},
            f"{name}.{key} has forbidden type {type(item).__name__}",
        )
        if type(item) is float:
            require(math.isfinite(item), f"{name}.{key} is non-finite")
    return mapping


def load_checkpoint_strict(
    path: Path,
    *,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise IntegrityError(f"Safe checkpoint load failed for {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise IntegrityError("Checkpoint root must be a dictionary")
    schema_version = checkpoint_schema_version(checkpoint)
    if expected_schema is not None:
        require(
            expected_schema
            in {
                LEGACY_CHECKPOINT_SCHEMA_V1,
                PRODUCTION_CHECKPOINT_SCHEMA_V2,
                AUDITED_PRODUCTION_CHECKPOINT_SCHEMA_V3,
            },
            f"Unknown expected checkpoint schema: {expected_schema!r}",
        )
        require_equal(
            "checkpoint schema selected by source protocol",
            schema_version,
            expected_schema,
        )
    require_mapping(checkpoint["model"], "checkpoint.model")
    require_mapping(checkpoint["encoders"], "checkpoint.encoders")
    args = require_mapping(checkpoint["args"], "checkpoint.args")
    finite_numeric_list_args = {
        "more_tail_class_prior",
        "external_teacher_class_recalls",
        "external_teacher_class_compensation_weights",
        "masked_branch_tcl_final_class_accuracy_ema",
        "trusted_branch_fusion_final_class_accuracy_ema",
    }
    boolean_list_args = {
        "masked_branch_tcl_final_class_initialized",
        "trusted_branch_fusion_final_class_initialized",
    }
    for key, value in args.items():
        require(type(key) is str, "Every checkpoint argument name must be a string")
        if key in finite_numeric_list_args and type(value) is list:
            require(
                all(
                    type(item) in {int, float}
                    and not isinstance(item, bool)
                    and math.isfinite(float(item))
                    for item in value
                ),
                f"checkpoint.args.{key} must be a finite numeric list",
            )
        elif key in boolean_list_args and type(value) is list:
            require(
                all(type(item) is bool for item in value),
                f"checkpoint.args.{key} must be a boolean list",
            )
        else:
            require(
                value is None or type(value) in {bool, int, float, str},
                f"checkpoint.args.{key} has forbidden type {type(value).__name__}",
            )
        if type(value) is float:
            require(math.isfinite(value), f"checkpoint.args.{key} is non-finite")
    require(
        type(checkpoint["best_epoch"]) is int and checkpoint["best_epoch"] >= 1,
        "checkpoint.best_epoch must be a positive integer",
    )
    if schema_version in {
        PRODUCTION_CHECKPOINT_SCHEMA_V2,
        AUDITED_PRODUCTION_CHECKPOINT_SCHEMA_V3,
    }:
        selection_score = checkpoint["best_validation_selection_score"]
        selection_policy = args.get(
            "checkpoint_selection_policy", "validation_best"
        )
        require(
            selection_policy in {"validation_best", "final_epoch_refit"},
            "checkpoint.args.checkpoint_selection_policy is invalid",
        )
        if selection_policy == "final_epoch_refit":
            require_equal(
                "checkpoint.best_validation_selection_score",
                selection_score,
                None,
            )
        else:
            require(
                type(selection_score) in {int, float}
                and not isinstance(selection_score, bool)
                and math.isfinite(float(selection_score)),
                "checkpoint.best_validation_selection_score must be finite numeric",
            )
        require(
            type(checkpoint["checkpoint_selection_metric"]) is str
            and bool(checkpoint["checkpoint_selection_metric"]),
            "checkpoint.checkpoint_selection_metric must be a non-empty string",
        )
        _validate_flat_scalar_mapping(
            checkpoint["decision_protocol"], "checkpoint.decision_protocol"
        )
        if schema_version == AUDITED_PRODUCTION_CHECKPOINT_SCHEMA_V3:
            require_mapping(
                checkpoint["matched_ablation_audit"],
                "checkpoint.matched_ablation_audit",
            )
            receipt = checkpoint["sample_order_receipt"]
            require(
                receipt is None or isinstance(receipt, Mapping),
                "checkpoint.sample_order_receipt must be null or a mapping",
            )
            # Reject non-JSON or non-finite audit payloads before comparing
            # them with the signed validation result.
            try:
                sha256_json(checkpoint["matched_ablation_audit"])
                sha256_json(receipt)
            except (TypeError, ValueError) as error:
                raise IntegrityError(
                    "Checkpoint audit metadata must be finite JSON"
                ) from error
    return checkpoint


def _source_class_count(source: Mapping[str, Any]) -> int:
    data = require_mapping(source.get("data"), "source.data")
    train_class_counts = data.get("train_class_counts")
    require(
        type(train_class_counts) is list
        and len(train_class_counts) in {2, 3}
        and all(type(count) is int and count >= 0 for count in train_class_counts)
        and sum(train_class_counts) > 0,
        "source.data.train_class_counts must contain two or three non-negative "
        "integer class counts with positive total support",
    )
    return len(train_class_counts)


EXTERNAL_TEACHER_DERIVED_CHECKPOINT_FIELDS = frozenset(
    {
        "external_teacher_kind",
        "external_teacher_generator_protocol_sha256",
        "external_teacher_class_recalls",
        "external_teacher_class_compensation_weights",
        "external_teacher_correct_coverage",
    }
)


def validate_external_teacher_task_kind_binding(
    args: argparse.Namespace,
    original_arg_names: set[str],
    enabled: bool,
) -> str | None:
    """Validate the exact legacy-or-temporal derived checkpoint schema."""

    task_field = {"external_teacher_task"}
    all_derived_fields = set(EXTERNAL_TEACHER_DERIVED_CHECKPOINT_FIELDS) | task_field
    saved_fields = all_derived_fields & original_arg_names
    if not enabled:
        require(
            not saved_fields,
            "Disabled external teacher must not save derived OOF fields",
        )
        return None
    if "external_teacher_task" in original_arg_names:
        require(
            saved_fields == all_derived_fields,
            "Temporal external-teacher checkpoint must save task plus all "
            "five derived OOF fields",
        )
        require_equal(
            "checkpoint external teacher task",
            getattr(args, "external_teacher_task", None),
            EXTERNAL_TEACHER_TEMPORAL_TASK,
        )
        task = args.external_teacher_task
    else:
        require(
            saved_fields == set(EXTERNAL_TEACHER_DERIVED_CHECKPOINT_FIELDS),
            "Legacy external-teacher checkpoint must save exactly the "
            "historical five derived OOF fields",
        )
        task = EXTERNAL_TEACHER_TASK
    allowed_kinds = EXTERNAL_TEACHER_TASK_KIND_BINDINGS.get(task)
    require(
        allowed_kinds is not None
        and getattr(args, "external_teacher_kind", None) in allowed_kinds,
        "checkpoint external task/teacher_kind binding is unsupported",
    )
    return task


def validate_checkpoint_metadata_against_source(
    checkpoint: Mapping[str, Any],
    source: Mapping[str, Any],
    args_payload: Mapping[str, Any],
) -> str:
    """Close checkpoint selection metadata against its exact source generation."""

    checkpoint_version = checkpoint_schema_version(checkpoint)
    source_version = source_protocol_schema_version(source)
    training = require_mapping(source.get("training"), "source.training")
    selection_policy = args_payload.get(
        "checkpoint_selection_policy", "validation_best"
    )
    require(
        selection_policy in {"validation_best", "final_epoch_refit"},
        "checkpoint.args.checkpoint_selection_policy is invalid",
    )
    require_equal(
        "checkpoint.best_epoch/source.training.best_epoch",
        checkpoint["best_epoch"],
        training.get("best_epoch"),
    )

    if checkpoint_version == LEGACY_CHECKPOINT_SCHEMA_V1:
        # This certifies the legacy checkpoint container and its source
        # selection metadata only.  It deliberately does not waive any later
        # replay-protocol revision checks.
        require_equal(
            "legacy checkpoint source protocol",
            source_version,
            LEGACY_SOURCE_PROTOCOL_V1,
        )
        require_equal(
            "legacy checkpoint selection policy",
            selection_policy,
            "validation_best",
        )
        for field_name in (
            "best_validation_selection_score",
            "checkpoint_selection_metric",
        ):
            require(
                field_name not in training,
                f"Legacy source.training must not contain {field_name}",
            )
        require(
            "checkpoint_selection_metric" not in args_payload,
            "Legacy checkpoint args must not contain checkpoint_selection_metric",
        )
        return checkpoint_version

    if checkpoint_version == AUDITED_PRODUCTION_CHECKPOINT_SCHEMA_V3:
        require_equal(
            "checkpoint.matched_ablation_audit/source.matched_ablation_audit",
            checkpoint["matched_ablation_audit"],
            source.get("matched_ablation_audit"),
        )
        require_equal(
            "checkpoint.sample_order_receipt/source.sample_order_receipts",
            checkpoint["sample_order_receipt"],
            source.get("sample_order_receipts"),
        )

    require_equal(
        "production checkpoint source protocol",
        source_version,
        PRODUCTION_SOURCE_PROTOCOL_V2,
    )
    for field_name in (
        "best_validation_selection_score",
        "checkpoint_selection_metric",
    ):
        require(
            field_name in training,
            f"Production source.training is missing {field_name}",
        )
    protocol = require_mapping(source.get("protocol"), "source.protocol")
    protocol_policy = protocol.get(
        "checkpoint_selection_policy", "validation_best"
    )
    training_policy = training.get(
        "checkpoint_selection_policy", "validation_best"
    )
    require_equal(
        "checkpoint/source.protocol checkpoint selection policy",
        selection_policy,
        protocol_policy,
    )
    require_equal(
        "checkpoint/source.training checkpoint selection policy",
        selection_policy,
        training_policy,
    )
    if selection_policy == "final_epoch_refit":
        require(
            "checkpoint_selection_policy" in args_payload
            and "checkpoint_selection_policy" in protocol
            and "checkpoint_selection_policy" in training,
            "final_epoch_refit must be explicit in checkpoint, protocol, and training metadata",
        )
        require_equal(
            "checkpoint.best_validation_selection_score/source.training",
            checkpoint["best_validation_selection_score"],
            training["best_validation_selection_score"],
        )
        require_equal(
            "final_epoch_refit best validation selection score",
            checkpoint["best_validation_selection_score"],
            None,
        )
        require_equal(
            "final_epoch_refit protocol held-validation inference count",
            protocol.get("held_validation_inference_count"),
            1,
        )
        require_equal(
            "final_epoch_refit training held-validation inference count",
            training.get("held_validation_inference_count"),
            1,
        )
    else:
        require_numeric_equal(
            "checkpoint.best_validation_selection_score/source.training",
            checkpoint["best_validation_selection_score"],
            training["best_validation_selection_score"],
        )
    require_equal(
        "checkpoint.checkpoint_selection_metric/source.training",
        checkpoint["checkpoint_selection_metric"],
        training["checkpoint_selection_metric"],
    )
    require_equal(
        "checkpoint.checkpoint_selection_metric/source.protocol",
        checkpoint["checkpoint_selection_metric"],
        protocol["checkpoint_selection_metric"],
    )
    require(
        "checkpoint_selection_metric" in args_payload,
        "Production checkpoint args are missing checkpoint_selection_metric",
    )
    require_equal(
        "checkpoint.checkpoint_selection_metric/checkpoint.args",
        checkpoint["checkpoint_selection_metric"],
        args_payload["checkpoint_selection_metric"],
    )

    class_count = _source_class_count(source)
    if class_count == 3:
        require_equal(
            "three-class checkpoint selection metric",
            checkpoint["checkpoint_selection_metric"],
            "macro_f1",
        )
        require_equal(
            "three-class checkpoint.decision_protocol",
            dict(
                require_mapping(
                    checkpoint["decision_protocol"],
                    "checkpoint.decision_protocol",
                )
            ),
            RAW_ARGMAX_DECISION_PROTOCOL_V1,
        )
        require_equal(
            "three-class source.protocol.binary_decision",
            protocol.get("binary_decision"),
            None,
        )
    else:
        require_equal(
            "binary checkpoint selection metric",
            checkpoint["checkpoint_selection_metric"],
            "positive_auprc",
        )
        require_equal(
            "binary checkpoint/source decision protocol",
            dict(
                require_mapping(
                    checkpoint["decision_protocol"],
                    "checkpoint.decision_protocol",
                )
            ),
            dict(
                require_mapping(
                    protocol.get("binary_decision"),
                    "source.protocol.binary_decision",
                )
            ),
        )
    return checkpoint_version


def validate_checkpoint_soup_replay_schema(
    args: argparse.Namespace,
    soup_argument_was_saved: bool,
    protocol: Mapping[str, Any],
    training: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    """Strictly validate the validation-only top-K aggregation audit trail."""

    if not soup_argument_was_saved:
        # Historical top-1 artifacts predate the extension and retain their
        # original result schema and checkpoint bytes.
        return
    requested_top_k = getattr(args, "checkpoint_soup_top_k", None)
    require(
        type(requested_top_k) is int and requested_top_k >= 1,
        "checkpoint.args.checkpoint_soup_top_k must be a positive integer",
    )
    require_equal(
        "source.training.checkpoint_soup_top_k",
        training.get("checkpoint_soup_top_k"),
        requested_top_k,
    )
    epochs_completed = training.get("epochs_completed")
    require(type(epochs_completed) is int, "epochs_completed must be an integer")
    expected_retained = min(requested_top_k, epochs_completed)
    require_equal(
        "source.training.checkpoint_soup_retained",
        training.get("checkpoint_soup_retained"),
        expected_retained,
    )
    epochs = training.get("checkpoint_soup_epochs")
    scores = training.get("checkpoint_soup_validation_macro_f1")
    weight_sources = training.get("checkpoint_soup_candidate_weight_sources")
    require(type(epochs) is list, "checkpoint_soup_epochs must be a list")
    require(
        type(scores) is list,
        "checkpoint_soup_validation_macro_f1 must be a list",
    )
    require(
        type(weight_sources) is list,
        "checkpoint_soup_candidate_weight_sources must be a list",
    )
    require(
        len(epochs) == len(scores) == len(weight_sources) == expected_retained,
        "Checkpoint soup rank lists have inconsistent lengths",
    )
    require(
        all(type(epoch) is int and 1 <= epoch <= epochs_completed for epoch in epochs),
        "Checkpoint soup epochs must be valid one-indexed training epochs",
    )
    require(
        len(set(epochs)) == len(epochs),
        "Checkpoint soup epochs must be unique",
    )
    selection_policy = getattr(
        args, "checkpoint_selection_policy", "validation_best"
    )
    if selection_policy == "final_epoch_refit":
        require_equal("final_epoch_refit requested top-K", requested_top_k, 1)
        require_equal(
            "final_epoch_refit validation_only",
            getattr(args, "validation_only", None),
            True,
        )
        require_equal(
            "final_epoch_refit early_stopping_patience",
            getattr(args, "early_stopping_patience", None),
            0,
        )
        require_equal(
            "final_epoch_refit epochs_completed",
            epochs_completed,
            getattr(args, "train_epochs", None),
        )
        require_equal(
            "final_epoch_refit best/final epoch",
            checkpoint["best_epoch"],
            getattr(args, "train_epochs", None),
        )
        require_equal("final_epoch_refit retained epochs", epochs, [epochs_completed])
        require_equal("final_epoch_refit selection scores", scores, [None])
        require_equal(
            "final_epoch_refit canonical selection scores",
            training.get("checkpoint_soup_validation_scores"),
            [None],
        )
        require_equal(
            "final_epoch_refit candidate weight sources",
            weight_sources,
            ["live"],
        )
        require_equal(
            "final_epoch_refit checkpoint weight source",
            training.get("checkpoint_weight_source"),
            "final_live",
        )
        selection_metric = getattr(args, "checkpoint_selection_metric", None)
        require_equal(
            "final_epoch_refit Macro-AUPRC score audit",
            training.get("checkpoint_soup_validation_macro_auprc"),
            [None] if selection_metric == "macro_auprc" else None,
        )
        require_equal(
            "final_epoch_refit positive-AUPRC score audit",
            training.get("checkpoint_soup_validation_positive_auprc"),
            [None] if selection_metric == "positive_auprc" else None,
        )
        saved_protocol = require_mapping(
            protocol.get("checkpoint_soup"), "source.protocol.checkpoint_soup"
        )
        require_equal(
            "source.protocol.checkpoint_soup",
            dict(saved_protocol),
            checkpoint_soup_protocol(args, epochs, scores, weight_sources),
        )
        return
    require_equal(
        "checkpoint selection policy", selection_policy, "validation_best"
    )
    require(
        all(
            type(score) is float and math.isfinite(score)
            for score in scores
        ),
        "Checkpoint soup validation Macro-F1 scores must be finite floats",
    )
    require(
        list(zip(scores, epochs))
        == sorted(zip(scores, epochs), key=lambda item: (-item[0], item[1])),
        "Checkpoint soup candidates are not ranked by validation Macro-F1",
    )
    require_equal("checkpoint.best_epoch", checkpoint["best_epoch"], epochs[0])
    expected_weight_sources = [
        (
            "stepwise_ema"
            if bool(getattr(args, "use_ema", False))
            and epoch >= int(getattr(args, "ema_start_epoch", 1))
            else "live"
        )
        for epoch in epochs
    ]
    require_equal(
        "source.training.checkpoint_soup_candidate_weight_sources",
        weight_sources,
        expected_weight_sources,
    )
    expected_final_weight_source = (
        "validation_top_k_equal_weight_soup"
        if requested_top_k > 1
        else weight_sources[0]
    )
    require_equal(
        "source.training.checkpoint_weight_source",
        training.get("checkpoint_weight_source"),
        expected_final_weight_source,
    )
    saved_protocol = require_mapping(
        protocol.get("checkpoint_soup"), "source.protocol.checkpoint_soup"
    )
    require_equal(
        "source.protocol.checkpoint_soup",
        dict(saved_protocol),
        checkpoint_soup_protocol(args, epochs, scores, weight_sources),
    )


def validate_empirical_missing_pattern_replay_checkpoint_contract(
    args: argparse.Namespace,
    original_arg_names: set[str],
    objective: Mapping[str, Any],
    source_data: Mapping[str, Any],
) -> None:
    """Fail closed on the saved opt-in schema before any data/test iteration."""

    enabled = getattr(args, "empirical_missing_pattern_replay", False)
    require(
        type(enabled) is bool,
        "checkpoint.args.empirical_missing_pattern_replay must be boolean",
    )
    saved_derived = (
        set(EMPIRICAL_MISSING_PATTERN_REPLAY_DERIVED_ARGS)
        & original_arg_names
    )
    if enabled:
        require_equal("empirical replay checkpoint model", args.model, "our_moe")
        require_equal("empirical replay checkpoint dataset", args.data, "abcd")
        require_equal(
            "empirical replay recomputed combination",
            args.recompute_dropped_combination,
            True,
        )
        require_equal(
            "empirical replay generator-only gradient exclusion",
            args.generator_only_task_grad,
            False,
        )
        require_numeric_equal("empirical replay SAM exclusion", args.sam_rho, 0.0)
        require_numeric_equal(
            "empirical replay R-Drop exclusion", args.rdrop_loss_weight, 0.0
        )
        require(
            type(args.modality_dropout_prob) in {int, float}
            and not isinstance(args.modality_dropout_prob, bool)
            and math.isfinite(float(args.modality_dropout_prob))
            and 0.0 < float(args.modality_dropout_prob) <= 1.0,
            "enabled empirical replay requires modality dropout in (0, 1]",
        )
        try:
            empirical_missing_pattern_replay_active_objectives(args)
        except (TypeError, ValueError) as error:
            raise IntegrityError(
                "Enabled empirical replay dropped-view objective activation "
                f"is invalid: {error}"
            ) from error
        require(
            saved_derived
            == set(EMPIRICAL_MISSING_PATTERN_REPLAY_DERIVED_ARGS),
            "Enabled empirical replay checkpoint must save all four derived "
            "audit arguments",
        )
    else:
        require(
            not saved_derived,
            "Disabled empirical replay checkpoint must not save derived audit state",
        )

    saved_protocol = objective.get(
        "empirical_missing_pattern_replay_protocol"
    )
    argument_was_saved = "empirical_missing_pattern_replay" in original_arg_names
    try:
        expected_protocol = empirical_missing_pattern_replay_protocol(args)
    except (TypeError, ValueError, RuntimeError) as error:
        raise IntegrityError(
            f"Checkpoint empirical replay audit arguments are invalid: {error}"
        ) from error
    if argument_was_saved:
        require_equal(
            "our_moe_objective.empirical_missing_pattern_replay_protocol",
            dict(
                require_mapping(
                    saved_protocol,
                    "our_moe_objective.empirical_missing_pattern_replay_protocol",
                )
            ),
            expected_protocol,
        )
    else:
        require(
            saved_protocol is None,
            "Historical checkpoint without empirical replay argument cannot "
            "contain its protocol",
        )

    saved_data_audit = source_data.get(
        "empirical_missing_pattern_replay_training_audit"
    )
    if enabled:
        protocol = expected_protocol
        expected_audit = {
            "implementation_revision": protocol["implementation_revision"],
            "modality_order": protocol["modality_order"],
            "pattern_counts": protocol["pattern_counts"],
            "training_row_count": protocol["training_row_count"],
            "train_id_mask_binding_sha256": protocol[
                "train_id_mask_binding_sha256"
            ],
            "rng_seed": protocol["rng_seed"],
        }
        require_equal(
            "source.data empirical replay training audit",
            dict(
                require_mapping(
                    saved_data_audit,
                    "source.data.empirical_missing_pattern_replay_training_audit",
                )
            ),
            expected_audit,
        )
        canonical_counts_json = json.dumps(
            protocol["pattern_counts"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        require_equal(
            "checkpoint empirical replay canonical pattern-count JSON",
            args.empirical_missing_pattern_replay_pattern_counts_json,
            canonical_counts_json,
        )
        canonical_order_json = json.dumps(
            protocol["modality_order"],
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        require_equal(
            "checkpoint empirical replay canonical modality-order JSON",
            args.empirical_missing_pattern_replay_modality_order_json,
            canonical_order_json,
        )
    else:
        require_equal(
            "disabled source.data empirical replay training audit",
            saved_data_audit,
            None,
        )


def validate_empirical_missing_pattern_replay_current_data(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    observed: Any,
    train_ids: Sequence[int],
    data_dict: Mapping[str, Any],
    modality_dict: Mapping[str, int],
) -> None:
    """Recompute train-only counts/binding from the currently loaded manifest."""

    if not bool(getattr(args, "empirical_missing_pattern_replay", False)):
        return
    try:
        current_ids, current_observed, current_order = (
            empirical_missing_pattern_training_observed(
                observed, train_ids, data_dict, modality_dict
            )
        )
        current_audit = empirical_missing_pattern_replay_training_audit(
            current_ids,
            current_observed,
            current_order,
            args.modality_dropout_prob,
            args.seed,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise IntegrityError(
            f"Current empirical replay training-mask audit is invalid: {error}"
        ) from error
    saved_audit = require_mapping(
        require_mapping(source.get("data"), "source.data").get(
            "empirical_missing_pattern_replay_training_audit"
        ),
        "source.data.empirical_missing_pattern_replay_training_audit",
    )
    require_equal(
        "current training masks / saved empirical replay audit",
        current_audit,
        dict(saved_audit),
    )
    require_equal(
        "current empirical replay pattern-count JSON",
        json.dumps(
            current_audit["pattern_counts"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        args.empirical_missing_pattern_replay_pattern_counts_json,
    )
    require_equal(
        "current empirical replay modality-order JSON",
        json.dumps(
            current_audit["modality_order"],
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        args.empirical_missing_pattern_replay_modality_order_json,
    )
    require_equal(
        "current empirical replay training ID-mask binding",
        current_audit["train_id_mask_binding_sha256"],
        args.empirical_missing_pattern_replay_train_id_mask_binding_sha256,
    )
    require_equal(
        "current empirical replay independent RNG seed",
        current_audit["rng_seed"],
        args.empirical_missing_pattern_replay_rng_seed,
    )


def validate_args_against_source(
    args_payload: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    source: Mapping[str, Any],
    source_path: Path,
    checkpoint_path: Path,
) -> argparse.Namespace:
    original_arg_names = set(args_payload)
    args = argparse.Namespace(**dict(args_payload))
    required_args = (
        "model",
        "data",
        "seed",
        "output_dir",
        "validation_only",
        "device",
        "torch_device",
        "dataset_manifest",
        "modality",
        "batch_size",
        "num_workers",
        "pin_memory",
        "preprocessed",
        "initial_filling",
        "use_common_ids",
        "hidden_dim",
        "num_patches",
        "num_heads",
        "num_layers_fus",
        "dropout",
        "independent_patch_embeddings",
    )
    missing = [name for name in required_args if not hasattr(args, name)]
    require(not missing, f"Checkpoint is missing required arguments: {missing}")
    require_equal("checkpoint.args.model", args.model, source["model"])
    require_equal("checkpoint.args.data", args.data, source["dataset"])
    require_equal("checkpoint.args.seed", args.seed, source["seed"])
    require_equal("checkpoint.args.validation_only", args.validation_only, True)
    require_equal("checkpoint.args.torch_device", args.torch_device, f"cuda:{args.device}")
    args.output_dir = resolve_replay_output_root(args.output_dir, source_path)

    expected_result, _, expected_checkpoint, _ = result_paths(args)
    require(
        expected_result.resolve() == source_path,
        f"Source result path is inconsistent with checkpoint args: expected {expected_result}",
    )
    require(
        expected_checkpoint.resolve() == checkpoint_path,
        f"Checkpoint path is inconsistent with checkpoint args: expected {expected_checkpoint}",
    )

    data = require_mapping(source["data"], "source.data")
    require_equal("source.data.manifest", data.get("manifest"), args.dataset_manifest)
    require_equal("source.data.modality", data.get("modality"), args.modality)

    sampler_ramp_was_saved = "sampler_ramp_epochs" in original_arg_names
    if not hasattr(args, "sampler_ramp_epochs"):
        # Checkpoints produced before the curriculum extension used the exact
        # historical abrupt switch represented by zero.
        args.sampler_ramp_epochs = 0
    checkpoint_soup_was_saved = "checkpoint_soup_top_k" in original_arg_names
    if not hasattr(args, "checkpoint_soup_top_k"):
        # Historical runs selected and saved exactly one best validation epoch.
        args.checkpoint_soup_top_k = 1
    checkpoint_selection_policy_was_saved = (
        "checkpoint_selection_policy" in original_arg_names
    )
    if not hasattr(args, "checkpoint_selection_policy"):
        # Absence is the exact historical per-epoch validation-best cadence.
        args.checkpoint_selection_policy = "validation_best"
    require(
        args.checkpoint_selection_policy
        in {"validation_best", "final_epoch_refit"},
        "checkpoint.args.checkpoint_selection_policy is invalid",
    )
    adni_image_imputation_was_saved = (
        "adni_image_imputation" in original_arg_names
    )
    if not hasattr(args, "adni_image_imputation"):
        # Historical checkpoints used the first training-only pandas mode when
        # initial_filling was "mean", represented exactly by legacy_mode.
        args.adni_image_imputation = "legacy_mode"

    training = require_mapping(source["training"], "source.training")
    validate_checkpoint_metadata_against_source(
        checkpoint,
        source,
        args_payload,
    )
    training_args = {
        "epochs_requested": "train_epochs",
        "warm_up_epochs": "warm_up_epochs",
        "early_stopping_patience": "early_stopping_patience",
        "batch_size": "batch_size",
        "learning_rate": "lr",
        "weight_decay": "weight_decay",
        "optimizer": "optimizer",
        "gradient_clip": "grad_clip",
        "dropout": "dropout",
        "sampler_power": "sampler_power",
        "sampler_ramp_epochs": "sampler_ramp_epochs",
        "checkpoint_soup_top_k": "checkpoint_soup_top_k",
        "class_weight_power": "class_weight_power",
    }
    for result_key, args_key in training_args.items():
        require(hasattr(args, args_key), f"Checkpoint is missing argument {args_key!r}")
        result_value = training.get(result_key)
        if result_key == "sampler_ramp_epochs" and not sampler_ramp_was_saved:
            # The source result and checkpoint schema both predate this neutral
            # field, so materialize their canonical historical value.
            result_value = 0
        if result_key == "checkpoint_soup_top_k" and not checkpoint_soup_was_saved:
            result_value = 1
        require_numeric_equal(
            f"source.training.{result_key}", result_value, getattr(args, args_key)
        )
    require(type(training.get("epochs_completed")) is int, "epochs_completed must be an integer")
    require(
        checkpoint["best_epoch"] <= training["epochs_completed"] <= args.train_epochs,
        "best_epoch/epochs_completed/epochs_requested ordering is invalid",
    )
    if args.checkpoint_selection_policy == "final_epoch_refit":
        require(
            checkpoint_selection_policy_was_saved,
            "final_epoch_refit must be explicit in checkpoint args",
        )
        require(
            checkpoint_soup_was_saved,
            "final_epoch_refit must save checkpoint_soup_top_k and its final-state audit",
        )
        require(
            not (
                {"resume", "resume_checkpoint", "resume_from_checkpoint", "start_epoch"}
                & original_arg_names
            ),
            "final_epoch_refit checkpoint must not contain resume state",
        )
        require_equal("final_epoch_refit validation_only", args.validation_only, True)
        require_equal(
            "final_epoch_refit early_stopping_patience",
            args.early_stopping_patience,
            0,
        )
        require_equal(
            "final_epoch_refit checkpoint_soup_top_k",
            args.checkpoint_soup_top_k,
            1,
        )
        require_equal(
            "final_epoch_refit epochs_completed",
            training["epochs_completed"],
            args.train_epochs,
        )
        require_equal(
            "final_epoch_refit best_epoch",
            checkpoint["best_epoch"],
            args.train_epochs,
        )

    protocol = require_mapping(source["protocol"], "source.protocol")
    validate_checkpoint_soup_replay_schema(
        args,
        checkpoint_soup_was_saved,
        protocol,
        training,
        checkpoint,
    )
    if adni_image_imputation_was_saved:
        imputation_protocol = require_mapping(
            protocol.get("adni_image_imputation"), "adni_image_imputation"
        )
        require_equal(
            "source.protocol.adni_image_imputation",
            dict(imputation_protocol),
            adni_image_imputation_protocol(args),
        )
    if sampler_ramp_was_saved:
        sampler_protocol = require_mapping(
            protocol.get("sampler_curriculum"), "sampler_curriculum"
        )
        require_equal(
            "source.protocol.sampler_curriculum",
            dict(sampler_protocol),
            sampler_curriculum_protocol(args),
        )
    if args.model != "our_moe":
        neutral_our_moe_extensions = {
            "joint_head_ensemble_size": 1,
            "patch_adapter_rank": 0,
            "more_fewer_rank_loss_weight": 0.0,
            "more_tail_rank": 0,
            "more_tail_peak_amplitude": 0.0,
            "hard_cvar_dual_boundary_max_weight": 0.0,
            "clear_eval_gate_cache": False,
            "tree_teacher_distill_weight": 0.0,
            "tree_teacher_npz": None,
            "tree_teacher_npz_sha256": None,
            "external_teacher_distill_weight": 0.0,
            "external_teacher_npz": None,
            "external_teacher_npz_sha256": None,
            "masked_branch_tcl_loss_weight": 0.0,
            "trusted_branch_fusion_distill_weight": 0.0,
            "dual_local_boundary_loss_weight": 0.0,
            "presentation_axis_loss_weight": 0.0,
            "presentation_axis_residual_cap": 0.5,
            "missing_capacity_residual_width": 0,
            "missing_family_normalized_router": False,
            "missing_family_residual_gate": False,
            "dynamic_branch_joint_prior_boost": True,
            "dynamic_branch_use_observed_mask": False,
            "prediction_observed_specialists_only": False,
            "dynamic_branch_quality_weighted_aux": False,
            "supervised_router_observed_specialists_only": False,
            "recon_encoder_gradient_scale": 1.0,
            "generator_only_task_grad": False,
            "recon_context_dropout_probability": 0.0,
            "generator_output_gate": True,
            "empirical_missing_pattern_replay": False,
        }
        for field_name, neutral_value in neutral_our_moe_extensions.items():
            if hasattr(args, field_name):
                require_equal(
                    f"checkpoint.args.{field_name}",
                    getattr(args, field_name),
                    neutral_value,
                )
        require(
            not (
                set(EMPIRICAL_MISSING_PATTERN_REPLAY_DERIVED_ARGS)
                & original_arg_names
            ),
            "Non-our_moe checkpoint cannot contain empirical replay audit state",
        )
    if args.model == "our_moe":
        objective = require_mapping(protocol.get("our_moe_objective"), "our_moe_objective")
        backward_compatible_defaults = {
            "use_generators": True,
            "generator_task_grad": False,
            "generator_only_task_grad": False,
            "empirical_missing_pattern_replay": False,
            "class_weighted_quality_branch_aux": False,
            "centered_evidence_confidence": False,
            "learn_observed_reliability": False,
            "class_conditional_fusion": False,
            "ordinal_head_type": "proportional",
            "ordinal_fusion_weight": 0.0,
            "ordinal_aux_loss_weight": 0.0,
            "uncertainty_aware_ordinal_fusion": False,
            "gated_transformer_residual": False,
            "pattern_aware_reconstruction": False,
            "recon_normalized_token_loss_weight": 0.0,
            "recon_context_dropout_probability": 0.0,
            "recon_encoder_gradient_scale": 1.0,
            "generator_output_gate": True,
            "logit_adjust_tau": 0.0,
            "class1_aux_loss_weight": 0.0,
            "supervised_router_loss_weight": 0.0,
            "supervised_router_temperature": 0.25,
            "dynamic_branch_joint_prior_boost": True,
            "dynamic_branch_use_observed_mask": False,
            "prediction_observed_specialists_only": False,
            "dynamic_branch_quality_weighted_aux": False,
            "supervised_router_observed_specialists_only": False,
            "normalized_gate_loss": False,
            "supervised_contrastive_loss_weight": 0.0,
            "supervised_contrastive_temperature": 0.10,
            "supervised_contrastive_projection_dim": 64,
            "dual_boundary_rank_loss_weight": 0.0,
            "dual_boundary_rank_margin": 0.20,
            "dual_boundary_rank_10_weight": 2.0 / 3.0,
            "sam_rho": 0.0,
            "rdrop_loss_weight": 0.0,
            "joint_head_ensemble_size": 1,
            "patch_adapter_rank": 0,
            "more_fewer_rank_loss_weight": 0.0,
            "more_tail_rank": 0,
            "more_tail_peak_amplitude": 0.0,
            "more_tail_class_prior": None,
            "clear_eval_gate_cache": False,
            "hard_cvar_dual_boundary_max_weight": 0.0,
            "hard_cvar_dual_boundary_tail_fraction": 0.25,
            "hard_cvar_dual_boundary_margin": 0.20,
            "hard_cvar_dual_boundary_10_weight": 0.65,
            "hard_cvar_dual_boundary_start_epoch": 5,
            "hard_cvar_dual_boundary_ramp_epochs": 10,
            "tree_teacher_distill_weight": 0.0,
            "tree_teacher_temperature": 2.0,
            "tree_teacher_npz": None,
            "tree_teacher_npz_sha256": None,
            "external_teacher_distill_weight": 0.0,
            "external_teacher_temperature": 2.0,
            "external_teacher_npz": None,
            "external_teacher_npz_sha256": None,
            "external_teacher_trust_mode": "correct_only",
            "external_teacher_class_compensation": "none",
            "external_teacher_kind": None,
            "external_teacher_generator_protocol_sha256": None,
            "external_teacher_class_recalls": None,
            "external_teacher_class_compensation_weights": None,
            "external_teacher_correct_coverage": None,
            "masked_branch_tcl_loss_weight": 0.0,
            "masked_branch_tcl_temperature": 2.0,
            "masked_branch_tcl_start_epoch": 6,
            "masked_branch_tcl_class_acc_ema": 0.9,
            "masked_branch_tcl_final_class_accuracy_ema": None,
            "masked_branch_tcl_final_class_initialized": None,
            "masked_branch_tcl_epochs_updated": None,
            "masked_branch_tcl_ema_history_json": None,
            "trusted_branch_fusion_distill_weight": 0.0,
            "trusted_branch_fusion_temperature": 2.0,
            "trusted_branch_fusion_start_epoch": 6,
            "trusted_branch_fusion_class_acc_ema": 0.9,
            "trusted_branch_fusion_final_class_accuracy_ema": None,
            "trusted_branch_fusion_final_class_initialized": None,
            "trusted_branch_fusion_epochs_updated": None,
            "trusted_branch_fusion_ema_history_json": None,
            "dual_local_boundary_loss_weight": 0.0,
            "presentation_axis_loss_weight": 0.0,
            "presentation_axis_residual_cap": 0.5,
            "missing_capacity_residual_width": 0,
            "missing_family_normalized_router": False,
            "missing_family_residual_gate": False,
        }
        for arg_name, default_value in backward_compatible_defaults.items():
            if not hasattr(args, arg_name):
                setattr(args, arg_name, default_value)
        require(
            type(args.dynamic_branch_fusion) is bool,
            "checkpoint.args.dynamic_branch_fusion must be boolean",
        )
        require(
            type(args.missing_family_normalized_router) is bool,
            "checkpoint.args.missing_family_normalized_router must be boolean",
        )
        require(
            type(args.missing_family_residual_gate) is bool,
            "checkpoint.args.missing_family_residual_gate must be boolean",
        )
        require(
            not args.missing_family_residual_gate
            or args.missing_family_normalized_router,
            "checkpoint missing-family residual gate requires normalized routing",
        )
        if args.missing_family_normalized_router:
            require(
                not args.dynamic_branch_fusion
                and float(args.supervised_router_loss_weight) == 0.0
                and not args.class_conditional_fusion
                and int(args.joint_head_ensemble_size) == 1,
                "checkpoint missing-family routing violates its v1 "
                "fail-closed compatibility restrictions",
            )
        require(
            type(args.generator_task_grad) is bool,
            "checkpoint.args.generator_task_grad must be boolean",
        )
        require(
            type(args.generator_only_task_grad) is bool,
            "checkpoint.args.generator_only_task_grad must be boolean",
        )
        require(
            not (
                args.generator_task_grad
                and args.generator_only_task_grad
            ),
            "Checkpoint cannot enable generator_task_grad and "
            "generator_only_task_grad together",
        )
        require(
            not args.generator_only_task_grad or bool(args.use_generators),
            "generator_only_task_grad checkpoint requires generators",
        )
        validate_empirical_missing_pattern_replay_checkpoint_contract(
            args,
            original_arg_names,
            objective,
            data,
        )
        require(
            type(args.recon_encoder_gradient_scale) in {int, float}
            and not isinstance(args.recon_encoder_gradient_scale, bool)
            and math.isfinite(float(args.recon_encoder_gradient_scale))
            and 0.0 <= float(args.recon_encoder_gradient_scale) <= 1.0,
            "checkpoint.args.recon_encoder_gradient_scale must be finite and "
            "in [0, 1]",
        )
        require(
            float(args.recon_encoder_gradient_scale) == 1.0
            or bool(args.use_generators),
            "Non-default reconstruction encoder-gradient scale requires "
            "generators",
        )
        require(
            type(args.supervised_router_loss_weight) in {int, float}
            and not isinstance(args.supervised_router_loss_weight, bool)
            and math.isfinite(float(args.supervised_router_loss_weight))
            and float(args.supervised_router_loss_weight) >= 0.0,
            "checkpoint.args.supervised_router_loss_weight must be finite and "
            "non-negative",
        )
        require(
            type(args.supervised_router_temperature) in {int, float}
            and not isinstance(args.supervised_router_temperature, bool)
            and math.isfinite(float(args.supervised_router_temperature))
            and float(args.supervised_router_temperature) > 0.0,
            "checkpoint.args.supervised_router_temperature must be finite and "
            "positive",
        )
        require(
            float(args.supervised_router_loss_weight) == 0.0
            or args.dynamic_branch_fusion,
            "Positive supervised router loss requires dynamic branch fusion",
        )
        clean_dynamic_router_fields = {
            "dynamic_branch_joint_prior_boost",
            "dynamic_branch_use_observed_mask",
            "dynamic_branch_quality_weighted_aux",
            "supervised_router_observed_specialists_only",
        }
        saved_clean_dynamic_router_fields = (
            clean_dynamic_router_fields & original_arg_names
        )
        require(
            not saved_clean_dynamic_router_fields
            or saved_clean_dynamic_router_fields
            == clean_dynamic_router_fields,
            "Checkpoint clean dynamic-router schema must contain either zero "
            "or all four fields",
        )
        for field_name in clean_dynamic_router_fields:
            require(
                type(getattr(args, field_name)) is bool,
                f"checkpoint.args.{field_name} must be boolean",
            )
        require(
            type(args.prediction_observed_specialists_only) is bool,
            "checkpoint.args.prediction_observed_specialists_only must be boolean",
        )
        clean_dynamic_router_enabled = (
            not args.dynamic_branch_joint_prior_boost
            or args.dynamic_branch_use_observed_mask
            or args.dynamic_branch_quality_weighted_aux
            or args.supervised_router_observed_specialists_only
        )
        require(
            not clean_dynamic_router_enabled
            or bool(args.dynamic_branch_fusion),
            "Clean dynamic-router checkpoint options require dynamic branch fusion",
        )
        require(
            not args.supervised_router_observed_specialists_only
            or float(args.supervised_router_loss_weight) > 0.0,
            "Observed-specialist router supervision requires positive router loss",
        )
        require(
            not args.dynamic_branch_quality_weighted_aux
            or not bool(args.balanced_branch_aux),
            "Dynamic quality-weighted branch auxiliary is incompatible with "
            "balanced branch auxiliary",
        )
        require(
            type(args.patch_adapter_rank) is int
            and args.patch_adapter_rank >= 0,
            "checkpoint.args.patch_adapter_rank must be a non-negative integer",
        )
        require(
            args.patch_adapter_rank == 0
            or not bool(args.independent_patch_embeddings),
            "Checkpoint cannot enable low-rank and independent patch encoders together",
        )
        require(
            type(args.rdrop_loss_weight) in {int, float}
            and not isinstance(args.rdrop_loss_weight, bool)
            and math.isfinite(float(args.rdrop_loss_weight))
            and float(args.rdrop_loss_weight) >= 0.0,
            "checkpoint.args.rdrop_loss_weight must be finite and non-negative",
        )
        require(
            not (
                float(args.rdrop_loss_weight) > 0.0
                and float(args.sam_rho) > 0.0
            ),
            "Checkpoint cannot enable R-Drop and SAM together",
        )
        require(
            type(args.dual_local_boundary_loss_weight) in {int, float}
            and not isinstance(args.dual_local_boundary_loss_weight, bool)
            and math.isfinite(float(args.dual_local_boundary_loss_weight))
            and float(args.dual_local_boundary_loss_weight) >= 0.0,
            "checkpoint.args.dual_local_boundary_loss_weight must be finite "
            "and non-negative",
        )
        if float(args.dual_local_boundary_loss_weight) > 0.0:
            require(
                float(args.ordinal_fusion_weight) == 0.0
                and float(args.ordinal_aux_loss_weight) == 0.0
                and float(args.class1_aux_loss_weight) == 0.0
                and float(args.dual_boundary_rank_loss_weight) == 0.0
                and float(args.hard_cvar_dual_boundary_max_weight) == 0.0
                and int(args.more_tail_rank) == 0
                and float(args.masked_branch_tcl_loss_weight) == 0.0
                and float(args.trusted_branch_fusion_distill_weight) == 0.0
                and float(args.presentation_axis_loss_weight) == 0.0,
                "Enabled DLBR checkpoint violates its fixed first-revision "
                "mutual-exclusion protocol",
            )
        require(
            type(args.presentation_axis_loss_weight) in {int, float}
            and not isinstance(args.presentation_axis_loss_weight, bool)
            and math.isfinite(float(args.presentation_axis_loss_weight))
            and float(args.presentation_axis_loss_weight) >= 0.0,
            "checkpoint.args.presentation_axis_loss_weight must be finite "
            "and non-negative",
        )
        require(
            type(args.presentation_axis_residual_cap) in {int, float}
            and not isinstance(args.presentation_axis_residual_cap, bool)
            and math.isfinite(float(args.presentation_axis_residual_cap))
            and float(args.presentation_axis_residual_cap) > 0.0,
            "checkpoint.args.presentation_axis_residual_cap must be finite "
            "and positive",
        )
        if float(args.presentation_axis_loss_weight) > 0.0:
            require(
                float(args.ordinal_fusion_weight) == 0.0
                and float(args.ordinal_aux_loss_weight) == 0.0
                and float(args.class1_aux_loss_weight) == 0.0
                and float(args.dual_local_boundary_loss_weight) == 0.0,
                "Enabled presentation-axis checkpoint violates its "
                "mutual-exclusion protocol",
            )
        require(
            type(args.missing_capacity_residual_width) is int
            and args.missing_capacity_residual_width >= 0,
            "checkpoint.args.missing_capacity_residual_width must be a "
            "non-negative integer",
        )
        if args.missing_capacity_residual_width > 0:
            require_equal(
                "source.data.num_classes for missing-capacity residual",
                data.get("num_classes"),
                3,
            )
            input_dimensions = require_mapping(
                data.get("input_dimensions"), "source.data.input_dimensions"
            )
            require(
                len(input_dimensions) == 4,
                "Enabled missing-capacity residual requires exactly four "
                "modalities",
            )
        require(
            type(args.joint_head_ensemble_size) is int
            and args.joint_head_ensemble_size >= 1,
            "checkpoint.args.joint_head_ensemble_size must be a positive integer",
        )
        require(
            args.joint_head_ensemble_size == 1
            or (
                type(args.num_layers_pred) is int
                and args.num_layers_pred >= 2
            ),
            "Joint-head ensemble checkpoints require num_layers_pred >= 2",
        )
        require(
            args.joint_head_ensemble_size == 1
            or (
                type(args.branch_aux_loss_weight) in {int, float}
                and not isinstance(args.branch_aux_loss_weight, bool)
                and math.isfinite(float(args.branch_aux_loss_weight))
                and float(args.branch_aux_loss_weight) > 0.0
            ),
            "Joint-head ensemble checkpoints require positive branch auxiliary weight",
        )
        require(
            type(args.clear_eval_gate_cache) is bool,
            "checkpoint.args.clear_eval_gate_cache must be boolean",
        )
        require(
            type(args.more_fewer_rank_loss_weight) in {int, float}
            and not isinstance(args.more_fewer_rank_loss_weight, bool)
            and math.isfinite(float(args.more_fewer_rank_loss_weight))
            and float(args.more_fewer_rank_loss_weight) >= 0.0,
            "checkpoint.args.more_fewer_rank_loss_weight must be finite and non-negative",
        )
        require(
            float(args.more_fewer_rank_loss_weight) == 0.0
            or (
                type(args.modality_dropout_prob) in {int, float}
                and not isinstance(args.modality_dropout_prob, bool)
                and math.isfinite(float(args.modality_dropout_prob))
                and 0.0 < float(args.modality_dropout_prob) <= 1.0
            ),
            "More/fewer ranking requires modality dropout in (0, 1]",
        )
        require(
            type(args.more_tail_rank) is int and args.more_tail_rank >= 0,
            "checkpoint.args.more_tail_rank must be a non-negative integer",
        )
        require(
            type(args.more_tail_peak_amplitude) in {int, float}
            and not isinstance(args.more_tail_peak_amplitude, bool)
            and math.isfinite(float(args.more_tail_peak_amplitude))
            and float(args.more_tail_peak_amplitude) >= 0.0,
            "checkpoint.args.more_tail_peak_amplitude must be finite and non-negative",
        )
        require(
            (
                args.more_tail_rank > 0
                and float(args.more_tail_peak_amplitude) > 0.0
                and isinstance(args.more_tail_class_prior, (list, tuple))
                and len(args.more_tail_class_prior) >= 2
                and all(
                    type(value) in {int, float}
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                    for value in args.more_tail_class_prior
                )
                and math.isclose(
                    sum(float(value) for value in args.more_tail_class_prior),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            )
            or (
                args.more_tail_rank == 0
                and float(args.more_tail_peak_amplitude) == 0.0
                and args.more_tail_class_prior is None
            ),
            "checkpoint MORE tail rank/amplitude/prior enablement is inconsistent",
        )
        more_tail_base_schema_fields = {
            "more_tail_rank",
            "more_tail_peak_amplitude",
        }
        saved_more_tail_base_fields = (
            more_tail_base_schema_fields & original_arg_names
        )
        require(
            not saved_more_tail_base_fields
            or saved_more_tail_base_fields == more_tail_base_schema_fields,
            "Checkpoint MORE tail schema must contain both rank and amplitude",
        )
        require(
            args.more_tail_rank == 0
            or "more_tail_class_prior" in original_arg_names,
            "Enabled MORE tail checkpoint must save its training class prior",
        )
        require(
            type(args.tree_teacher_distill_weight) in {int, float}
            and not isinstance(args.tree_teacher_distill_weight, bool)
            and math.isfinite(float(args.tree_teacher_distill_weight))
            and float(args.tree_teacher_distill_weight) >= 0.0,
            "checkpoint.args.tree_teacher_distill_weight must be finite and non-negative",
        )
        require(
            type(args.tree_teacher_temperature) in {int, float}
            and not isinstance(args.tree_teacher_temperature, bool)
            and math.isfinite(float(args.tree_teacher_temperature))
            and float(args.tree_teacher_temperature) > 0.0,
            "checkpoint.args.tree_teacher_temperature must be finite and positive",
        )
        tree_teacher_enabled = float(args.tree_teacher_distill_weight) > 0.0
        require(
            (
                tree_teacher_enabled
                and type(args.tree_teacher_npz) is str
                and bool(args.tree_teacher_npz)
                and type(args.tree_teacher_npz_sha256) is str
                and len(args.tree_teacher_npz_sha256) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in args.tree_teacher_npz_sha256
                )
            )
            or (
                not tree_teacher_enabled
                and args.tree_teacher_npz is None
                and args.tree_teacher_npz_sha256 is None
            ),
            "checkpoint tree-teacher path/hash enablement is inconsistent",
        )
        require(
            not tree_teacher_enabled
            or (
                args.data == "abcd"
                and "".join(sorted(set(str(args.modality).upper()))) == "BCGI"
            ),
            "enabled v1 tree teacher requires the ABCD IGCB task",
        )
        tree_teacher_schema_fields = {
            "tree_teacher_distill_weight",
            "tree_teacher_temperature",
            "tree_teacher_npz",
            "tree_teacher_npz_sha256",
        }
        saved_tree_teacher_fields = (
            tree_teacher_schema_fields & original_arg_names
        )
        require(
            not saved_tree_teacher_fields
            or saved_tree_teacher_fields == tree_teacher_schema_fields,
            "Checkpoint tree-teacher schema must contain either zero or all four fields",
        )
        require(
            type(args.external_teacher_distill_weight) in {int, float}
            and not isinstance(args.external_teacher_distill_weight, bool)
            and math.isfinite(float(args.external_teacher_distill_weight))
            and float(args.external_teacher_distill_weight) >= 0.0,
            "checkpoint.args.external_teacher_distill_weight must be finite "
            "and non-negative",
        )
        require(
            type(args.external_teacher_temperature) in {int, float}
            and not isinstance(args.external_teacher_temperature, bool)
            and math.isfinite(float(args.external_teacher_temperature))
            and float(args.external_teacher_temperature) > 0.0,
            "checkpoint.args.external_teacher_temperature must be finite and positive",
        )
        require(
            args.external_teacher_trust_mode in {"correct_only", "all"},
            "checkpoint external-teacher trust mode is unsupported",
        )
        require(
            args.external_teacher_class_compensation
            in {"none", "tcl_sqrt_error"},
            "checkpoint external-teacher class compensation is unsupported",
        )
        external_teacher_enabled = (
            float(args.external_teacher_distill_weight) > 0.0
        )
        require(
            not (tree_teacher_enabled and external_teacher_enabled),
            "Checkpoint cannot enable v1 tree and v2 external teachers together",
        )
        require(
            (
                external_teacher_enabled
                and type(args.external_teacher_npz) is str
                and bool(args.external_teacher_npz)
                and type(args.external_teacher_npz_sha256) is str
                and len(args.external_teacher_npz_sha256) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in args.external_teacher_npz_sha256
                )
            )
            or (
                not external_teacher_enabled
                and args.external_teacher_npz is None
                and args.external_teacher_npz_sha256 is None
                and args.external_teacher_trust_mode == "correct_only"
                and args.external_teacher_class_compensation == "none"
            ),
            "checkpoint external-teacher enablement/path/mode fields are inconsistent",
        )
        require(
            not external_teacher_enabled
            or (
                args.data == "abcd"
                and "".join(sorted(set(str(args.modality).upper()))) == "BCGI"
            ),
            "enabled v2 external teacher requires the ABCD IGCB task",
        )
        raw_external_teacher_schema_fields = {
            "external_teacher_distill_weight",
            "external_teacher_temperature",
            "external_teacher_npz",
            "external_teacher_npz_sha256",
            "external_teacher_trust_mode",
            "external_teacher_class_compensation",
        }
        saved_raw_external_teacher_fields = (
            raw_external_teacher_schema_fields & original_arg_names
        )
        require(
            not saved_raw_external_teacher_fields
            or saved_raw_external_teacher_fields
            == raw_external_teacher_schema_fields,
            "Checkpoint external-teacher raw schema must contain either zero "
            "or all six fields",
        )
        validate_external_teacher_task_kind_binding(
            args,
            original_arg_names,
            external_teacher_enabled,
        )
        if external_teacher_enabled:
            generator_digest = args.external_teacher_generator_protocol_sha256
            require(
                type(generator_digest) is str
                and len(generator_digest) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in generator_digest
                ),
                "checkpoint external producer protocol digest is invalid",
            )
            recalls = args.external_teacher_class_recalls
            compensation_weights = (
                args.external_teacher_class_compensation_weights
            )
            require(
                type(recalls) is list
                and len(recalls) == 3
                and all(
                    type(value) in {int, float}
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in recalls
                ),
                "checkpoint external class recalls must be three finite values in [0, 1]",
            )
            require(
                type(compensation_weights) is list
                and len(compensation_weights) == 3
                and all(
                    type(value) in {int, float}
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                    for value in compensation_weights
                )
                and math.isclose(
                    sum(float(value) for value in compensation_weights),
                    3.0,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ),
                "checkpoint external class compensation weights must be "
                "non-negative and sum to three",
            )
            coverage = args.external_teacher_correct_coverage
            require(
                type(coverage) in {int, float}
                and not isinstance(coverage, bool)
                and math.isfinite(float(coverage))
                and 0.0 <= float(coverage) <= 1.0,
                "checkpoint external correct coverage must be finite and in [0, 1]",
            )
        else:
            require(
                args.external_teacher_kind is None
                and args.external_teacher_generator_protocol_sha256 is None
                and args.external_teacher_class_recalls is None
                and args.external_teacher_class_compensation_weights is None
                and args.external_teacher_correct_coverage is None,
                "Disabled external teacher must not save derived OOF fields",
            )
        require(
            type(args.masked_branch_tcl_loss_weight) in {int, float}
            and not isinstance(args.masked_branch_tcl_loss_weight, bool)
            and math.isfinite(float(args.masked_branch_tcl_loss_weight))
            and float(args.masked_branch_tcl_loss_weight) >= 0.0,
            "checkpoint masked branch TCL weight must be finite and non-negative",
        )
        require(
            type(args.masked_branch_tcl_temperature) in {int, float}
            and not isinstance(args.masked_branch_tcl_temperature, bool)
            and math.isfinite(float(args.masked_branch_tcl_temperature))
            and float(args.masked_branch_tcl_temperature) > 0.0,
            "checkpoint masked branch TCL temperature must be finite and positive",
        )
        require(
            type(args.masked_branch_tcl_start_epoch) is int
            and args.masked_branch_tcl_start_epoch >= 1,
            "checkpoint masked branch TCL start epoch must be positive",
        )
        require(
            type(args.masked_branch_tcl_class_acc_ema) in {int, float}
            and not isinstance(args.masked_branch_tcl_class_acc_ema, bool)
            and math.isfinite(float(args.masked_branch_tcl_class_acc_ema))
            and 0.0 <= float(args.masked_branch_tcl_class_acc_ema) < 1.0,
            "checkpoint masked branch TCL EMA decay must be in [0, 1)",
        )
        masked_branch_tcl_enabled = (
            float(args.masked_branch_tcl_loss_weight) > 0.0
        )
        require(
            not masked_branch_tcl_enabled
            or args.masked_branch_tcl_start_epoch <= args.train_epochs,
            "enabled masked branch TCL start epoch exceeds training epochs",
        )
        raw_masked_branch_tcl_fields = {
            "masked_branch_tcl_loss_weight",
            "masked_branch_tcl_temperature",
            "masked_branch_tcl_start_epoch",
            "masked_branch_tcl_class_acc_ema",
        }
        saved_raw_masked_branch_tcl_fields = (
            raw_masked_branch_tcl_fields & original_arg_names
        )
        require(
            not saved_raw_masked_branch_tcl_fields
            or saved_raw_masked_branch_tcl_fields
            == raw_masked_branch_tcl_fields,
            "Checkpoint masked branch TCL raw schema must contain zero or all fields",
        )
        derived_masked_branch_tcl_fields = {
            "masked_branch_tcl_final_class_accuracy_ema",
            "masked_branch_tcl_final_class_initialized",
            "masked_branch_tcl_epochs_updated",
            "masked_branch_tcl_ema_history_json",
        }
        saved_derived_masked_branch_tcl_fields = (
            derived_masked_branch_tcl_fields & original_arg_names
        )
        if masked_branch_tcl_enabled:
            require(
                saved_derived_masked_branch_tcl_fields
                == derived_masked_branch_tcl_fields,
                "Enabled masked branch TCL checkpoint must save finalized EMA state",
            )
            class_count = len(
                require_mapping(source["data"], "source.data").get(
                    "train_class_counts", []
                )
            )
            final_accuracy = args.masked_branch_tcl_final_class_accuracy_ema
            final_initialized = args.masked_branch_tcl_final_class_initialized
            require(
                class_count >= 2
                and type(final_accuracy) is list
                and len(final_accuracy) == class_count
                and all(
                    type(value) in {int, float}
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in final_accuracy
                ),
                "checkpoint masked branch TCL final class EMA is invalid",
            )
            require(
                type(final_initialized) is list
                and len(final_initialized) == class_count
                and all(type(value) is bool for value in final_initialized),
                "checkpoint masked branch TCL initialized flags are invalid",
            )
            require(
                type(args.masked_branch_tcl_epochs_updated) is int
                and args.masked_branch_tcl_epochs_updated
                == training["epochs_completed"],
                "checkpoint masked branch TCL state must cover every completed epoch",
            )
            try:
                masked_branch_tcl_history = json.loads(
                    args.masked_branch_tcl_ema_history_json
                )
            except (TypeError, ValueError) as error:
                raise IntegrityError(
                    "checkpoint masked branch TCL EMA history is not valid JSON"
                ) from error
            require(
                type(masked_branch_tcl_history) is list
                and len(masked_branch_tcl_history)
                == args.masked_branch_tcl_epochs_updated,
                "checkpoint masked branch TCL history length is invalid",
            )
            for expected_epoch, record in enumerate(
                masked_branch_tcl_history, start=1
            ):
                record = require_mapping(
                    record,
                    f"masked branch TCL EMA history epoch {expected_epoch}",
                )
                require_equal(
                    "masked branch TCL EMA history epoch",
                    record.get("epoch"),
                    expected_epoch,
                )
                correct = record.get("branch_correct")
                valid = record.get("branch_valid")
                observed_accuracy = record.get("observed_class_accuracy")
                record_accuracy = record.get("class_accuracy_ema")
                record_initialized = record.get("initialized")
                require(
                    type(correct) is list
                    and type(valid) is list
                    and len(correct) == len(valid) == class_count
                    and all(type(value) is int and value >= 0 for value in correct)
                    and all(type(value) is int and value >= 0 for value in valid)
                    and all(c <= v for c, v in zip(correct, valid)),
                    "masked branch TCL history counts are invalid",
                )
                require(
                    type(observed_accuracy) is list
                    and len(observed_accuracy) == class_count
                    and all(
                        value is None
                        or (
                            type(value) in {int, float}
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                            and 0.0 <= float(value) <= 1.0
                        )
                        for value in observed_accuracy
                    ),
                    "masked branch TCL observed class accuracy is invalid",
                )
                require(
                    type(record_accuracy) is list
                    and len(record_accuracy) == class_count
                    and all(
                        type(value) in {int, float}
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and 0.0 <= float(value) <= 1.0
                        for value in record_accuracy
                    )
                    and type(record_initialized) is list
                    and len(record_initialized) == class_count
                    and all(type(value) is bool for value in record_initialized),
                    "masked branch TCL history EMA state is invalid",
                )
            require_equal(
                "checkpoint masked branch TCL final EMA/history closure",
                masked_branch_tcl_history[-1]["class_accuracy_ema"],
                final_accuracy,
            )
            require_equal(
                "checkpoint masked branch TCL final flags/history closure",
                masked_branch_tcl_history[-1]["initialized"],
                final_initialized,
            )
        else:
            require(
                not saved_derived_masked_branch_tcl_fields
                and args.masked_branch_tcl_final_class_accuracy_ema is None
                and args.masked_branch_tcl_final_class_initialized is None
                and args.masked_branch_tcl_epochs_updated is None
                and args.masked_branch_tcl_ema_history_json is None,
                "Disabled masked branch TCL must not save EMA state",
            )
        require(
            type(args.trusted_branch_fusion_distill_weight) in {int, float}
            and not isinstance(args.trusted_branch_fusion_distill_weight, bool)
            and math.isfinite(float(args.trusted_branch_fusion_distill_weight))
            and float(args.trusted_branch_fusion_distill_weight) >= 0.0,
            "checkpoint TBFD weight must be finite and non-negative",
        )
        require(
            type(args.trusted_branch_fusion_temperature) in {int, float}
            and not isinstance(args.trusted_branch_fusion_temperature, bool)
            and math.isfinite(float(args.trusted_branch_fusion_temperature))
            and float(args.trusted_branch_fusion_temperature) > 0.0,
            "checkpoint TBFD temperature must be finite and positive",
        )
        require(
            type(args.trusted_branch_fusion_start_epoch) is int
            and args.trusted_branch_fusion_start_epoch >= 1,
            "checkpoint TBFD start epoch must be positive",
        )
        require(
            type(args.trusted_branch_fusion_class_acc_ema) in {int, float}
            and not isinstance(args.trusted_branch_fusion_class_acc_ema, bool)
            and math.isfinite(float(args.trusted_branch_fusion_class_acc_ema))
            and 0.0 <= float(args.trusted_branch_fusion_class_acc_ema) < 1.0,
            "checkpoint TBFD EMA decay must be in [0, 1)",
        )
        trusted_branch_fusion_enabled = (
            float(args.trusted_branch_fusion_distill_weight) > 0.0
        )
        require(
            not (
                trusted_branch_fusion_enabled
                and masked_branch_tcl_enabled
            ),
            "Checkpoint cannot enable masked branch TCL and TBFD together",
        )
        require(
            not trusted_branch_fusion_enabled
            or args.trusted_branch_fusion_start_epoch <= args.train_epochs,
            "enabled TBFD start epoch exceeds training epochs",
        )
        raw_trusted_branch_fusion_fields = {
            "trusted_branch_fusion_distill_weight",
            "trusted_branch_fusion_temperature",
            "trusted_branch_fusion_start_epoch",
            "trusted_branch_fusion_class_acc_ema",
        }
        saved_raw_trusted_branch_fusion_fields = (
            raw_trusted_branch_fusion_fields & original_arg_names
        )
        require(
            not saved_raw_trusted_branch_fusion_fields
            or saved_raw_trusted_branch_fusion_fields
            == raw_trusted_branch_fusion_fields,
            "Checkpoint TBFD raw schema must contain zero or all fields",
        )
        derived_trusted_branch_fusion_fields = {
            "trusted_branch_fusion_final_class_accuracy_ema",
            "trusted_branch_fusion_final_class_initialized",
            "trusted_branch_fusion_epochs_updated",
            "trusted_branch_fusion_ema_history_json",
        }
        saved_derived_trusted_branch_fusion_fields = (
            derived_trusted_branch_fusion_fields & original_arg_names
        )
        if trusted_branch_fusion_enabled:
            require(
                saved_derived_trusted_branch_fusion_fields
                == derived_trusted_branch_fusion_fields,
                "Enabled TBFD checkpoint must save finalized EMA state",
            )
            class_count = len(
                require_mapping(source["data"], "source.data").get(
                    "train_class_counts", []
                )
            )
            final_accuracy = (
                args.trusted_branch_fusion_final_class_accuracy_ema
            )
            final_initialized = (
                args.trusted_branch_fusion_final_class_initialized
            )
            require(
                class_count >= 2
                and type(final_accuracy) is list
                and len(final_accuracy) == class_count
                and all(
                    type(value) in {int, float}
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in final_accuracy
                ),
                "checkpoint TBFD final class EMA is invalid",
            )
            require(
                type(final_initialized) is list
                and len(final_initialized) == class_count
                and all(type(value) is bool for value in final_initialized),
                "checkpoint TBFD initialized flags are invalid",
            )
            require(
                type(args.trusted_branch_fusion_epochs_updated) is int
                and args.trusted_branch_fusion_epochs_updated
                == training["epochs_completed"],
                "checkpoint TBFD state must cover every completed epoch",
            )
            try:
                trusted_branch_fusion_history = json.loads(
                    args.trusted_branch_fusion_ema_history_json
                )
            except (TypeError, ValueError) as error:
                raise IntegrityError(
                    "checkpoint TBFD EMA history is not valid JSON"
                ) from error
            require(
                type(trusted_branch_fusion_history) is list
                and len(trusted_branch_fusion_history)
                == args.trusted_branch_fusion_epochs_updated,
                "checkpoint TBFD history length is invalid",
            )
            for expected_epoch, record in enumerate(
                trusted_branch_fusion_history, start=1
            ):
                record = require_mapping(
                    record, f"TBFD EMA history epoch {expected_epoch}"
                )
                require_equal(
                    "TBFD EMA history epoch",
                    record.get("epoch"),
                    expected_epoch,
                )
                correct = record.get("branch_correct")
                valid = record.get("branch_valid")
                record_accuracy = record.get("class_accuracy_ema")
                record_initialized = record.get("initialized")
                require(
                    type(correct) is list
                    and type(valid) is list
                    and len(correct) == len(valid) == class_count
                    and all(type(value) is int and value >= 0 for value in correct)
                    and all(type(value) is int and value >= 0 for value in valid)
                    and all(c <= v for c, v in zip(correct, valid)),
                    "TBFD history counts are invalid",
                )
                require(
                    type(record_accuracy) is list
                    and len(record_accuracy) == class_count
                    and all(
                        type(value) in {int, float}
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and 0.0 <= float(value) <= 1.0
                        for value in record_accuracy
                    )
                    and type(record_initialized) is list
                    and len(record_initialized) == class_count
                    and all(type(value) is bool for value in record_initialized),
                    "TBFD history EMA state is invalid",
                )
            require_equal(
                "checkpoint TBFD final EMA/history closure",
                trusted_branch_fusion_history[-1]["class_accuracy_ema"],
                final_accuracy,
            )
            require_equal(
                "checkpoint TBFD final flags/history closure",
                trusted_branch_fusion_history[-1]["initialized"],
                final_initialized,
            )
        else:
            require(
                not saved_derived_trusted_branch_fusion_fields
                and args.trusted_branch_fusion_final_class_accuracy_ema is None
                and args.trusted_branch_fusion_final_class_initialized is None
                and args.trusted_branch_fusion_epochs_updated is None
                and args.trusted_branch_fusion_ema_history_json is None,
                "Disabled TBFD must not save EMA state",
            )
        for field_name in (
            "hard_cvar_dual_boundary_max_weight",
            "hard_cvar_dual_boundary_margin",
        ):
            value = getattr(args, field_name)
            require(
                type(value) in {int, float}
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0,
                f"checkpoint.args.{field_name} must be finite and non-negative",
            )
        require(
            type(args.hard_cvar_dual_boundary_tail_fraction) in {int, float}
            and not isinstance(args.hard_cvar_dual_boundary_tail_fraction, bool)
            and math.isfinite(float(args.hard_cvar_dual_boundary_tail_fraction))
            and 0.0 < float(args.hard_cvar_dual_boundary_tail_fraction) <= 1.0,
            "checkpoint hard-CVaR tail fraction must be in (0, 1]",
        )
        require(
            type(args.hard_cvar_dual_boundary_10_weight) in {int, float}
            and not isinstance(args.hard_cvar_dual_boundary_10_weight, bool)
            and math.isfinite(float(args.hard_cvar_dual_boundary_10_weight))
            and 0.0 <= float(args.hard_cvar_dual_boundary_10_weight) <= 1.0,
            "checkpoint hard-CVaR class-1-vs-0 weight must be in [0, 1]",
        )
        require(
            type(args.hard_cvar_dual_boundary_start_epoch) is int
            and args.hard_cvar_dual_boundary_start_epoch >= 1,
            "checkpoint hard-CVaR start epoch must be a positive integer",
        )
        require(
            type(args.hard_cvar_dual_boundary_ramp_epochs) is int
            and args.hard_cvar_dual_boundary_ramp_epochs >= 0,
            "checkpoint hard-CVaR ramp epochs must be a non-negative integer",
        )
        require(
            float(args.hard_cvar_dual_boundary_max_weight) == 0.0
            or args.hard_cvar_dual_boundary_start_epoch <= args.train_epochs,
            "Enabled hard-CVaR start epoch must not exceed train_epochs",
        )
        require(
            not (
                float(args.hard_cvar_dual_boundary_max_weight) > 0.0
                and float(args.dual_boundary_rank_loss_weight) > 0.0
            ),
            "Legacy and hard-CVaR dual-boundary objectives are mutually exclusive",
        )
        hard_cvar_schema_fields = {
            "hard_cvar_dual_boundary_max_weight",
            "hard_cvar_dual_boundary_tail_fraction",
            "hard_cvar_dual_boundary_margin",
            "hard_cvar_dual_boundary_10_weight",
            "hard_cvar_dual_boundary_start_epoch",
            "hard_cvar_dual_boundary_ramp_epochs",
        }
        saved_hard_cvar_fields = hard_cvar_schema_fields & original_arg_names
        require(
            not saved_hard_cvar_fields
            or saved_hard_cvar_fields == hard_cvar_schema_fields,
            "Checkpoint hard-CVaR schema must contain either zero or all six fields",
        )
        objective_args = {
            "use_generators": "use_generators",
            "generator_task_grad": "generator_task_grad",
            "generator_only_task_grad": "generator_only_task_grad",
            "empirical_missing_pattern_replay": (
                "empirical_missing_pattern_replay"
            ),
            "reconstruction_weight": "recon_loss_weight",
            "branch_aux_weight": "branch_aux_loss_weight",
            "modality_dropout_probability": "modality_dropout_prob",
            "drop_ce_weight": "drop_ce_loss_weight",
            "distillation_weight": "distill_loss_weight",
            "distillation_temperature": "distill_temperature",
            "complete_joint_only": "complete_joint_only",
            "complete_specialist_weight": "complete_specialist_weight",
            "missing_family_normalized_router": (
                "missing_family_normalized_router"
            ),
            "missing_family_residual_gate": "missing_family_residual_gate",
            "branch_confidence_mode": "branch_confidence_mode",
            "centered_evidence_confidence": "centered_evidence_confidence",
            "learn_observed_reliability": "learn_observed_reliability",
            "class_conditional_fusion": "class_conditional_fusion",
            "ordinal_head_type": "ordinal_head_type",
            "ordinal_fusion_weight": "ordinal_fusion_weight",
            "ordinal_aux_loss_weight": "ordinal_aux_loss_weight",
            "uncertainty_aware_ordinal_fusion": (
                "uncertainty_aware_ordinal_fusion"
            ),
            "gated_transformer_residual": "gated_transformer_residual",
            "recompute_dropped_combination": "recompute_dropped_combination",
            "vectorized_generation": "vectorized_generation",
            "reconstruction_targets_per_sample": "recon_targets_per_sample",
            "pattern_aware_reconstruction": "pattern_aware_reconstruction",
            "recon_normalized_token_loss_weight": (
                "recon_normalized_token_loss_weight"
            ),
            "recon_context_dropout_probability": (
                "recon_context_dropout_probability"
            ),
            "recon_encoder_gradient_scale": (
                "recon_encoder_gradient_scale"
            ),
            "generator_output_gate": "generator_output_gate",
            "logit_adjust_tau": "logit_adjust_tau",
            "class1_aux_loss_weight": "class1_aux_loss_weight",
            "supervised_router_loss_weight": "supervised_router_loss_weight",
            "supervised_router_temperature": "supervised_router_temperature",
            "dynamic_branch_joint_prior_boost": (
                "dynamic_branch_joint_prior_boost"
            ),
            "dynamic_branch_use_observed_mask": (
                "dynamic_branch_use_observed_mask"
            ),
            "prediction_observed_specialists_only": (
                "prediction_observed_specialists_only"
            ),
            "dynamic_branch_quality_weighted_aux": (
                "dynamic_branch_quality_weighted_aux"
            ),
            "supervised_router_observed_specialists_only": (
                "supervised_router_observed_specialists_only"
            ),
            "normalized_gate_loss": "normalized_gate_loss",
            "supervised_contrastive_loss_weight": (
                "supervised_contrastive_loss_weight"
            ),
            "supervised_contrastive_temperature": (
                "supervised_contrastive_temperature"
            ),
            "supervised_contrastive_projection_dim": (
                "supervised_contrastive_projection_dim"
            ),
            "dual_boundary_rank_loss_weight": (
                "dual_boundary_rank_loss_weight"
            ),
            "dual_boundary_rank_margin": "dual_boundary_rank_margin",
            "dual_boundary_rank_10_weight": "dual_boundary_rank_10_weight",
            "dual_local_boundary_loss_weight": (
                "dual_local_boundary_loss_weight"
            ),
            "presentation_axis_loss_weight": (
                "presentation_axis_loss_weight"
            ),
            "presentation_axis_residual_cap": (
                "presentation_axis_residual_cap"
            ),
            "missing_capacity_residual_width": (
                "missing_capacity_residual_width"
            ),
            "sam_rho": "sam_rho",
            "rdrop_loss_weight": "rdrop_loss_weight",
            "joint_head_ensemble_size": "joint_head_ensemble_size",
            "patch_adapter_rank": "patch_adapter_rank",
            "more_fewer_rank_loss_weight": "more_fewer_rank_loss_weight",
            "more_tail_rank": "more_tail_rank",
            "more_tail_peak_amplitude": "more_tail_peak_amplitude",
            "more_tail_class_prior": "more_tail_class_prior",
            "tree_teacher_distill_weight": "tree_teacher_distill_weight",
            "tree_teacher_temperature": "tree_teacher_temperature",
            "tree_teacher_npz": "tree_teacher_npz",
            "tree_teacher_npz_sha256": "tree_teacher_npz_sha256",
            "external_teacher_distill_weight": (
                "external_teacher_distill_weight"
            ),
            "external_teacher_temperature": "external_teacher_temperature",
            "external_teacher_npz": "external_teacher_npz",
            "external_teacher_npz_sha256": "external_teacher_npz_sha256",
            "external_teacher_trust_mode": "external_teacher_trust_mode",
            "external_teacher_class_compensation": (
                "external_teacher_class_compensation"
            ),
            "external_teacher_kind": "external_teacher_kind",
            "external_teacher_generator_protocol_sha256": (
                "external_teacher_generator_protocol_sha256"
            ),
            "external_teacher_class_recalls": (
                "external_teacher_class_recalls"
            ),
            "external_teacher_class_compensation_weights": (
                "external_teacher_class_compensation_weights"
            ),
            "external_teacher_correct_coverage": (
                "external_teacher_correct_coverage"
            ),
            "masked_branch_tcl_loss_weight": (
                "masked_branch_tcl_loss_weight"
            ),
            "masked_branch_tcl_temperature": (
                "masked_branch_tcl_temperature"
            ),
            "masked_branch_tcl_start_epoch": (
                "masked_branch_tcl_start_epoch"
            ),
            "masked_branch_tcl_class_acc_ema": (
                "masked_branch_tcl_class_acc_ema"
            ),
            "trusted_branch_fusion_distill_weight": (
                "trusted_branch_fusion_distill_weight"
            ),
            "trusted_branch_fusion_temperature": (
                "trusted_branch_fusion_temperature"
            ),
            "trusted_branch_fusion_start_epoch": (
                "trusted_branch_fusion_start_epoch"
            ),
            "trusted_branch_fusion_class_acc_ema": (
                "trusted_branch_fusion_class_acc_ema"
            ),
            "clear_eval_gate_cache": "clear_eval_gate_cache",
            "hard_cvar_dual_boundary_max_weight": (
                "hard_cvar_dual_boundary_max_weight"
            ),
            "hard_cvar_dual_boundary_tail_fraction": (
                "hard_cvar_dual_boundary_tail_fraction"
            ),
            "hard_cvar_dual_boundary_margin": (
                "hard_cvar_dual_boundary_margin"
            ),
            "hard_cvar_dual_boundary_10_weight": (
                "hard_cvar_dual_boundary_10_weight"
            ),
            "hard_cvar_dual_boundary_start_epoch": (
                "hard_cvar_dual_boundary_start_epoch"
            ),
            "hard_cvar_dual_boundary_ramp_epochs": (
                "hard_cvar_dual_boundary_ramp_epochs"
            ),
            "class_weighted_quality_branch_aux": "class_weighted_quality_branch_aux",
        }
        for result_key, args_key in objective_args.items():
            require(hasattr(args, args_key), f"Checkpoint is missing argument {args_key!r}")
            result_value = objective.get(result_key)
            if result_key in backward_compatible_defaults:
                legacy_extension_keys = {
                    "centered_evidence_confidence",
                    "learn_observed_reliability",
                    "class_conditional_fusion",
                    "ordinal_head_type",
                    "ordinal_fusion_weight",
                    "ordinal_aux_loss_weight",
                    "uncertainty_aware_ordinal_fusion",
                }
                introduced_with = (
                    result_key
                    if result_key == "uncertainty_aware_ordinal_fusion"
                    else (
                        "ordinal_aux_loss_weight"
                        if result_key in legacy_extension_keys
                        else result_key
                    )
                )
                if result_key not in objective and introduced_with not in original_arg_names:
                    # The field did not exist in this checkpoint schema; its
                    # materialized neutral value is canonical for replay.
                    result_value = getattr(args, args_key)
            require_numeric_equal(
                f"our_moe_objective.{result_key}",
                result_value,
                getattr(args, args_key),
            )
        if external_teacher_enabled and (
            getattr(args, "external_teacher_task", EXTERNAL_TEACHER_TASK)
            == EXTERNAL_TEACHER_TEMPORAL_TASK
        ):
            require_equal(
                "our_moe_objective.external_teacher_task",
                objective.get("external_teacher_task"),
                args.external_teacher_task,
            )
        else:
            require(
                "external_teacher_task" not in objective,
                "Only the temporal external-teacher schema may record an "
                "external_teacher_task objective field",
            )
        if saved_clean_dynamic_router_fields:
            require_equal(
                "our_moe_objective.dynamic_branch_gate_mask_source",
                objective.get("dynamic_branch_gate_mask_source"),
                (
                    "raw observed_mask"
                    if args.dynamic_branch_use_observed_mask
                    else "generator-expanded usable_mask"
                ),
            )
        saved_sam_protocol = objective.get("sam")
        if saved_sam_protocol is not None:
            require_equal(
                "our_moe_objective.sam",
                dict(require_mapping(saved_sam_protocol, "our_moe_objective.sam")),
                sam_training_protocol(args),
            )
        elif "sam_rho" in original_arg_names:
            raise IntegrityError(
                "Checkpoint schema contains sam_rho but source protocol omits SAM"
            )
        saved_rdrop_protocol = objective.get("rdrop")
        if saved_rdrop_protocol is not None:
            require_equal(
                "our_moe_objective.rdrop",
                dict(
                    require_mapping(
                        saved_rdrop_protocol, "our_moe_objective.rdrop"
                    )
                ),
                rdrop_training_protocol(args),
            )
        elif "rdrop_loss_weight" in original_arg_names:
            raise IntegrityError(
                "Checkpoint schema contains rdrop_loss_weight but source "
                "protocol omits R-Drop"
            )
        extension_protocols = (
            (
                "generator_only_task_grad",
                "classification_generator_gradient",
                classification_generator_gradient_protocol,
            ),
            (
                "recon_encoder_gradient_scale",
                "reconstruction_encoder_gradient",
                reconstruction_encoder_gradient_protocol,
            ),
            (
                "dynamic_branch_joint_prior_boost",
                "clean_dynamic_router",
                clean_dynamic_router_protocol,
            ),
            (
                "patch_adapter_rank",
                "low_rank_patch_adapter",
                low_rank_patch_adapter_protocol,
            ),
            (
                "joint_head_ensemble_size",
                "joint_head_ensemble",
                joint_head_ensemble_protocol,
            ),
            (
                "more_fewer_rank_loss_weight",
                "more_fewer_rank",
                more_fewer_rank_protocol,
            ),
            (
                "more_tail_rank",
                "more_tail_adapter",
                more_tail_adapter_protocol,
            ),
            (
                "dual_local_boundary_loss_weight",
                "dual_local_boundary_residual",
                dual_local_boundary_residual_protocol,
            ),
            (
                "presentation_axis_loss_weight",
                "presentation_axis_residual",
                presentation_axis_residual_protocol,
            ),
            (
                "missing_capacity_residual_width",
                "missing_capacity_residual",
                missing_capacity_residual_protocol,
            ),
            (
                "missing_family_normalized_router",
                "missing_family_router",
                missing_family_router_protocol,
            ),
            (
                "hard_cvar_dual_boundary_max_weight",
                "hard_cvar_dual_boundary",
                hard_cvar_dual_boundary_protocol,
            ),
            (
                "clear_eval_gate_cache",
                "evaluation_gate_cache",
                evaluation_gate_cache_protocol,
            ),
            (
                "tree_teacher_distill_weight",
                "cross_fitted_tree_teacher",
                cross_fitted_tree_teacher_protocol,
            ),
            (
                "external_teacher_distill_weight",
                "cross_fitted_external_teacher",
                cross_fitted_external_teacher_protocol,
            ),
            (
                "masked_branch_tcl_loss_weight",
                "masked_branch_tcl",
                masked_branch_tcl_protocol,
            ),
            (
                "trusted_branch_fusion_distill_weight",
                "trusted_branch_fusion_distillation",
                trusted_branch_fusion_distillation_protocol,
            ),
        )
        for argument_name, protocol_name, protocol_factory in extension_protocols:
            saved_extension_protocol = objective.get(protocol_name)
            if saved_extension_protocol is not None:
                saved_protocol_mapping = dict(
                    require_mapping(
                        saved_extension_protocol,
                        f"our_moe_objective.{protocol_name}",
                    )
                )
                if protocol_name == "more_fewer_rank":
                    saved_protocol_mapping = canonical_more_fewer_rank_protocol(
                        saved_protocol_mapping
                    )
                require_equal(
                    f"our_moe_objective.{protocol_name}",
                    saved_protocol_mapping,
                    protocol_factory(args),
                )
            elif argument_name in original_arg_names:
                raise IntegrityError(
                    f"Checkpoint schema contains {argument_name} but source "
                    f"protocol omits {protocol_name}"
                )
    elif args.model == "flex_moe":
        adapter = require_mapping(protocol.get("flex_moe_adapter"), "flex_moe_adapter")
        require_numeric_equal(
            "flex_moe_adapter.missing_bank_init_std",
            adapter.get("missing_bank_init_std"),
            args.missing_bank_init_std,
        )
        expected_classifier_dropout = (
            args.dropout if args.classifier_dropout is None else args.classifier_dropout
        )
        require_numeric_equal(
            "flex_moe_adapter.classifier_dropout",
            adapter.get("classifier_dropout"),
            expected_classifier_dropout,
        )
    elif args.model == "i2moe":
        objective = require_mapping(protocol.get("i2moe_objective"), "i2moe_objective")
        for result_key, args_key in {
            "interaction_loss_weight": "interaction_loss_weight",
            "fusion_layers": "num_layers_fus",
            "reweighting_hidden_dim": "hidden_dim_rw",
            "reweighting_layers": "num_layer_rw",
            "reweighting_temperature": "temperature_rw",
        }.items():
            require_numeric_equal(
                f"i2moe_objective.{result_key}", objective.get(result_key), getattr(args, args_key)
            )
    elif args.model == "acadiff" and "acadiff_mask_probability" in original_arg_names:
        objective = require_mapping(
            protocol.get("acadiff_objective"), "acadiff_objective"
        )
        require_equal(
            "acadiff_objective",
            dict(objective),
            acadiff_objective_protocol(args),
        )
        require_numeric_equal(
            "source.training.acadiff_mask_probability",
            training.get("acadiff_mask_probability"),
            args.acadiff_mask_probability,
        )
    elif args.model in {"moepp", "moepp_corrected"}:
        adapter = require_mapping(protocol.get("moepp_adapter"), "moepp_adapter")
        require_numeric_equal("moepp_adapter.fusion_layers", adapter.get("fusion_layers"), args.num_layers_fus)
        require_numeric_equal("moepp_adapter.num_experts", adapter.get("num_experts"), args.num_experts)
    return args


def validate_progress(source: Mapping[str, Any], source_path: Path) -> Path:
    progress_path = resolved(source["artifacts"]["progress"])
    progress = load_json_strict(progress_path)
    for key, expected in {
        "status": "validation_complete",
        "model": source["model"],
        "dataset": source["dataset"],
        "seed": source["seed"],
        "best_epoch": source["training"]["best_epoch"],
        "epochs_completed": source["training"]["epochs_completed"],
    }.items():
        require_equal(f"progress.{key}", progress.get(key), expected)
    training = require_mapping(source["training"], "source.training")
    selection_policy = training.get(
        "checkpoint_selection_policy", "validation_best"
    )
    if selection_policy == "final_epoch_refit":
        require_equal(
            "progress.checkpoint_selection_policy",
            progress.get("checkpoint_selection_policy"),
            "final_epoch_refit",
        )
        require_equal(
            "progress.best_validation_selection_score",
            progress.get("best_validation_selection_score"),
            None,
        )
        require_equal(
            "progress.held_validation_inference_count",
            progress.get("held_validation_inference_count"),
            1,
        )
        require_equal(
            "progress.checkpoint_selection_metric",
            progress.get("checkpoint_selection_metric"),
            training.get("checkpoint_selection_metric"),
        )
    if "checkpoint_soup_top_k" in training:
        for key in ("checkpoint_soup_top_k", "checkpoint_soup_retained"):
            require_equal(
                f"progress.{key}", progress.get(key), training.get(key)
            )
    require(
        resolved(progress.get("result", "")) == source_path,
        "Progress artifact points to a different source result",
    )
    return progress_path


def validate_state_dict(
    name: str,
    saved_state: Mapping[str, Any],
    module: nn.Module,
) -> int:
    expected_state = module.state_dict()
    if set(saved_state) != set(expected_state):
        missing = sorted(set(expected_state) - set(saved_state))
        unexpected = sorted(set(saved_state) - set(expected_state))
        raise IntegrityError(
            f"{name} state keys mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    total = 0
    for key, expected in expected_state.items():
        value = saved_state[key]
        if not torch.is_tensor(value):
            raise IntegrityError(f"{name}.{key} is not a tensor")
        if value.device.type != "cpu":
            raise IntegrityError(f"{name}.{key} was not safely loaded onto CPU")
        if tuple(value.shape) != tuple(expected.shape):
            raise IntegrityError(
                f"{name}.{key} shape mismatch: {tuple(value.shape)} != {tuple(expected.shape)}"
            )
        if value.dtype != expected.dtype:
            raise IntegrityError(
                f"{name}.{key} dtype mismatch: {value.dtype} != {expected.dtype}"
            )
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all().item():
                raise IntegrityError(f"{name}.{key} contains non-finite values")
        total += value.numel()
    try:
        module.load_state_dict(saved_state, strict=True)
    except RuntimeError as error:
        raise IntegrityError(f"Strict load failed for {name}: {error}") from error
    return total


def assert_metric_tree_close(saved: Any, replayed: Any, path: str, atol: float) -> None:
    if isinstance(saved, Mapping):
        if not isinstance(replayed, Mapping):
            raise IntegrityError(f"{path} type mismatch")
        if set(saved) != set(replayed):
            raise IntegrityError(
                f"{path} keys mismatch: saved={sorted(saved)}, replayed={sorted(replayed)}"
            )
        for key in sorted(saved):
            assert_metric_tree_close(saved[key], replayed[key], f"{path}.{key}", atol)
        return
    if isinstance(saved, list):
        if not isinstance(replayed, list) or len(saved) != len(replayed):
            raise IntegrityError(f"{path} list shape mismatch")
        for index, (saved_item, replayed_item) in enumerate(zip(saved, replayed)):
            assert_metric_tree_close(saved_item, replayed_item, f"{path}[{index}]", atol)
        return
    if saved is None:
        if replayed is not None:
            raise IntegrityError(f"{path} mismatch: saved None, replayed {replayed!r}")
        return
    if isinstance(saved, bool):
        require_equal(path, replayed, saved)
        return
    if isinstance(saved, int):
        require_equal(path, replayed, saved)
        return
    if isinstance(saved, float):
        if not isinstance(replayed, (int, float)) or isinstance(replayed, bool):
            raise IntegrityError(f"{path} type mismatch: replayed {type(replayed).__name__}")
        if not math.isfinite(saved) or not math.isfinite(float(replayed)):
            raise IntegrityError(f"{path} contains a non-finite value")
        if abs(saved - float(replayed)) > atol:
            raise IntegrityError(
                f"{path} mismatch: saved={saved:.17g}, replayed={float(replayed):.17g}, "
                f"absolute_delta={abs(saved - float(replayed)):.3g}, atol={atol:.3g}"
            )
        return
    require_equal(path, replayed, saved)


def validate_splits(
    source: Mapping[str, Any],
    labels: np.ndarray,
    train_ids: Sequence[int],
    valid_ids: Sequence[int],
    test_ids: Sequence[int],
    num_classes: int,
    input_dims: Mapping[str, int],
) -> dict[str, str]:
    split_map = {"train": train_ids, "validation": valid_ids, "test": test_ids}
    source_data = source["data"]
    require_equal(
        "source.data.split_sizes",
        source_data.get("split_sizes"),
        {name: len(ids) for name, ids in split_map.items()},
    )
    require_equal("source.data.num_classes", source_data.get("num_classes"), num_classes)
    require_equal("source.data.input_dimensions", source_data.get("input_dimensions"), dict(input_dims))

    arrays: dict[str, np.ndarray] = {}
    for name, ids in split_map.items():
        array = np.asarray(ids)
        require(array.ndim == 1, f"{name} split IDs must be one-dimensional")
        require(np.issubdtype(array.dtype, np.integer), f"{name} split IDs must be integer indices")
        array = array.astype(np.int64, copy=False)
        require(len(np.unique(array)) == len(array), f"{name} split contains duplicate IDs")
        require(((array >= 0) & (array < len(labels))).all(), f"{name} split contains invalid IDs")
        arrays[name] = array
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        require(
            len(np.intersect1d(arrays[left], arrays[right])) == 0,
            f"{left} and {right} splits overlap",
        )

    train_counts = np.bincount(np.asarray(labels)[arrays["train"]], minlength=num_classes).tolist()
    require_equal("source.data.train_class_counts", source_data.get("train_class_counts"), train_counts)
    fingerprints: dict[str, str] = {}
    labels_array = np.asarray(labels, dtype=np.int64)
    for name, ids in arrays.items():
        digest = hashlib.sha256()
        digest.update(ids.astype("<i8", copy=False).tobytes())
        digest.update(labels_array[ids].astype("<i8", copy=False).tobytes())
        fingerprints[name] = digest.hexdigest()
    return fingerprints


def validate_evaluation_alignment(
    split_name: str,
    evaluation: Mapping[str, np.ndarray],
    loader: Any,
    labels: np.ndarray,
    observed: np.ndarray,
    data_dict: Mapping[str, Any],
    modality_dict: Mapping[str, int],
    num_classes: int,
) -> np.ndarray:
    source_indices = np.asarray(loader.dataset.ids, dtype=np.int64)[
        np.asarray(loader.dataset.sorted_ids, dtype=np.int64)
    ]
    n = len(source_indices)
    expected_labels = np.asarray(labels)[source_indices]
    modality_names = [name for name in data_dict if name != "modality_comb"]
    require(
        set(modality_names) == set(modality_dict),
        f"{split_name} data/modalities do not match the resolved modality mapping",
    )
    # ``encode_batch`` stacks masks in the data dictionary's modality order.
    # Select those exact columns instead of assuming every run used all four
    # modalities or that a subset mapping is contiguous.
    expected_observed = np.stack(
        [np.asarray(observed)[source_indices, modality_dict[name]] for name in modality_names],
        axis=1,
    )
    expected_combinations = np.asarray(data_dict["modality_comb"])[source_indices]
    require(np.array_equal(evaluation["labels"], expected_labels), f"{split_name} labels/order mismatch")
    require(
        np.array_equal(evaluation["observed"], expected_observed),
        f"{split_name} observed-modality mask/order mismatch",
    )
    require(
        np.array_equal(evaluation["modality_comb"], expected_combinations),
        f"{split_name} modality-combination/order mismatch",
    )
    probabilities = np.asarray(evaluation["probabilities"])
    logits = np.asarray(evaluation["logits"])
    predictions = np.asarray(evaluation["predictions"])
    require(probabilities.shape == (n, num_classes), f"{split_name} probability shape mismatch")
    require(logits.shape == (n, num_classes), f"{split_name} logit shape mismatch")
    require(predictions.shape == (n,), f"{split_name} prediction shape mismatch")
    require(np.isfinite(probabilities).all(), f"{split_name} probabilities are non-finite")
    require(np.isfinite(logits).all(), f"{split_name} logits are non-finite")
    require(
        np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6),
        f"{split_name} probabilities do not sum to one",
    )
    require(
        np.array_equal(predictions, probabilities.argmax(axis=1)),
        f"{split_name} prediction rule is not raw probability argmax",
    )
    return source_indices


def create_prediction_frame(
    testing: Mapping[str, np.ndarray], source_indices: np.ndarray, num_classes: int
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "source_index": source_indices,
            "label": testing["labels"],
            "prediction": testing["predictions"],
            "modality_combination_index": testing["modality_comb"],
            "is_complete": testing["observed"].all(axis=1),
        }
    )
    for class_index in range(num_classes):
        frame[f"probability_class_{class_index}"] = testing["probabilities"][:, class_index]
        frame[f"logit_class_{class_index}"] = testing["logits"][:, class_index]
    return frame


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def publish_without_overwrite(
    result_path: Path,
    result: Mapping[str, Any],
    predictions_path: Path,
    predictions: pd.DataFrame,
) -> None:
    """Atomically publish both artifacts using hard links with no-clobber semantics."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(result_path) or os.path.lexists(predictions_path):
        raise IntegrityError(
            f"Refusing to overwrite formal artifacts: {result_path} or {predictions_path}"
        )
    result_temp = _temporary_sibling(result_path)
    predictions_temp = _temporary_sibling(predictions_path)
    prediction_published = False
    try:
        with predictions_temp.open("x", encoding="utf-8", newline="") as handle:
            predictions.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        with result_temp.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # link(2) fails with FileExistsError instead of replacing a destination.
        os.link(predictions_temp, predictions_path)
        prediction_published = True
        os.link(result_temp, result_path)
    except Exception:
        if prediction_published:
            predictions_path.unlink(missing_ok=True)
        raise
    finally:
        predictions_temp.unlink(missing_ok=True)
        result_temp.unlink(missing_ok=True)


def acquire_test_evaluation_claim(path: Path, payload: Mapping[str, Any]) -> None:
    """Reserve a single test evaluation using O_EXCL no-clobber semantics.

    A claim is acquired only after validation replay passes and before the
    first test-loader iteration.  It deliberately remains after failures so an
    interrupted test is inspected instead of silently repeated.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise IntegrityError(
            "A formal test-evaluation claim already exists; refusing to "
            f"evaluate test again: {path}"
        ) from error


def evaluate(args_cli: argparse.Namespace) -> tuple[Path, Path]:
    require(
        math.isfinite(args_cli.metric_atol)
        and 0.0 <= args_cli.metric_atol <= MAX_METRIC_ATOL,
        f"metric-atol must be between 0 and {MAX_METRIC_ATOL:g}",
    )
    if args_cli.expected_configuration_sha256 is not None:
        digest = args_cli.expected_configuration_sha256
        require(
            type(digest) is str
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "expected-configuration-sha256 must be a lowercase 64-character hex digest",
        )
    source_path = resolved(args_cli.source_result)
    formal_root = resolved(args_cli.output_root)
    require(source_path.is_file(), f"JSON file does not exist: {source_path}")
    source_digest_before = sha256_file(source_path)
    source = load_json_strict(source_path)
    source_digest = sha256_file(source_path)
    require_equal("source result sha256 before/after load", source_digest, source_digest_before)
    validate_source_schema(source, source_path)

    checkpoint_path = resolved(source["artifacts"]["checkpoint"])
    require(checkpoint_path.is_file(), f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint_digest_before = sha256_file(checkpoint_path)
    checkpoint = load_checkpoint_strict(
        checkpoint_path,
        expected_schema=checkpoint_schema_for_source(source),
    )
    checkpoint_digest = sha256_file(checkpoint_path)
    require_equal(
        "checkpoint sha256 before/after load", checkpoint_digest, checkpoint_digest_before
    )
    replay_args = validate_args_against_source(
        checkpoint["args"], checkpoint, source, source_path, checkpoint_path
    )
    progress_candidate = resolved(source["artifacts"]["progress"])
    require(progress_candidate.is_file(), f"Progress JSON does not exist: {progress_candidate}")
    progress_digest_before = sha256_file(progress_candidate)
    progress_path = validate_progress(source, source_path)
    progress_digest = sha256_file(progress_path)
    require_equal(
        "progress sha256 before/after load", progress_digest, progress_digest_before
    )
    replay_args.dataset_manifest = resolve_replay_manifest(replay_args.dataset_manifest)

    source_root = resolved(replay_args.output_dir)
    require(formal_root != source_root, "Formal output root must differ from the source output root")
    stem = f"{source['model']}_seed{source['seed']}"
    result_path = formal_root / source["dataset"] / f"{stem}.json"
    predictions_path = formal_root / "predictions" / source["dataset"] / f"{stem}.csv"
    claim_path = formal_root / ".evaluation_claims" / source["dataset"] / f"{stem}.lock"
    protected = {source_path, checkpoint_path, progress_path}
    require(result_path.resolve() not in protected, "Formal result resolves to a protected source artifact")
    require(predictions_path.resolve() not in protected, "Formal predictions resolve to a protected source artifact")
    require(claim_path.resolve() not in protected, "Formal claim resolves to a protected source artifact")
    require(not os.path.lexists(result_path), f"Formal result already exists: {result_path}")
    require(not os.path.lexists(predictions_path), f"Formal predictions already exist: {predictions_path}")
    require(not os.path.lexists(claim_path), f"Formal test-evaluation claim already exists: {claim_path}")

    if not torch.cuda.is_available():
        raise IntegrityError("CUDA is required for checkpoint replay")
    require(0 <= args_cli.device < torch.cuda.device_count(), f"Invalid CUDA device: {args_cli.device}")
    replay_device = torch.device(f"cuda:{args_cli.device}")
    replay_device_name = torch.cuda.get_device_name(replay_device)
    require_equal("CUDA device model", replay_device_name, source["training"].get("device"))

    # Preserve a digest of the unmodified training arguments before replacing
    # the runtime-only CUDA ordinal for this replay process.
    checkpoint_args_digest = sha256_json(checkpoint["args"])
    configuration_digest = sha256_json(normalized_configuration_args(vars(replay_args)))
    if args_cli.expected_configuration_sha256 is not None:
        require_equal(
            "locked configuration sha256",
            configuration_digest,
            args_cli.expected_configuration_sha256,
        )
    seed_everything(replay_args.seed)
    torch.set_float32_matmul_precision("high")
    replay_args.device = args_cli.device
    replay_args.torch_device = replay_device

    # The legacy ADNI loader uses paths such as ``./data/adni`` relative to the
    # MoE checkout.  Resolve CLI paths first, then replay from that exact root.
    os.chdir(HERE)

    modality_dict = resolve_modality_dict(replay_args)
    if hasattr(replay_args, "n_full_modalities"):
        require_equal(
            "checkpoint.args.n_full_modalities",
            replay_args.n_full_modalities,
            len(modality_dict),
        )
    replay_args.n_full_modalities = len(modality_dict)
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
    ) = load_and_preprocess_data(replay_args, modality_dict)
    validate_empirical_missing_pattern_replay_current_data(
        replay_args,
        source,
        observed,
        train_ids,
        data_dict,
        modality_dict,
    )
    if int(getattr(replay_args, "more_tail_rank", 0)) > 0:
        expected_more_tail_prior = training_class_prior(
            labels,
            train_ids,
            num_classes,
        ).astype(np.float64).tolist()
        require_equal(
            "checkpoint.args.more_tail_class_prior from current training labels",
            replay_args.more_tail_class_prior,
            expected_more_tail_prior,
        )
    if float(getattr(replay_args, "external_teacher_distill_weight", 0.0)) > 0:
        training_labels = np.asarray(labels)[np.asarray(train_ids)].astype(
            np.int64, copy=False
        )
        class_support = np.bincount(
            training_labels, minlength=num_classes
        ).astype(np.float64)
        require(
            num_classes == 3 and bool((class_support > 0).all()),
            "enabled external teacher requires all three classes in current training labels",
        )
        saved_recalls = np.asarray(
            replay_args.external_teacher_class_recalls, dtype=np.float64
        )
        expected_coverage = float(
            np.dot(class_support / class_support.sum(), saved_recalls)
        )
        require(
            math.isclose(
                float(replay_args.external_teacher_correct_coverage),
                expected_coverage,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "checkpoint external correct coverage is inconsistent with current "
            "training-label supports and saved class recalls",
        )
        square_root_error = np.sqrt(1.0 - saved_recalls)
        denominator = float(square_root_error.sum())
        expected_compensation = (
            np.ones(num_classes, dtype=np.float64)
            if denominator == 0.0
            else num_classes * square_root_error / denominator
        )
        require(
            np.allclose(
                np.asarray(
                    replay_args.external_teacher_class_compensation_weights,
                    dtype=np.float64,
                ),
                expected_compensation,
                rtol=0.0,
                atol=1e-12,
            ),
            "checkpoint external class compensation weights do not match the "
            "saved full-OOF recalls",
        )
    _, _, valid_loader, test_loader = create_loaders(
        data_dict,
        observed,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        replay_args.batch_size,
        replay_args.num_workers,
        replay_args.pin_memory,
        input_dims,
        transforms,
        masks,
        replay_args.preprocessed,
        replay_args.use_common_ids,
    )
    split_fingerprints = validate_splits(
        source,
        np.asarray(labels),
        train_ids,
        valid_ids,
        test_ids,
        num_classes,
        input_dims,
    )

    model = build_model(
        replay_args, len(encoders), num_classes, full_modality_index
    ).to(replay_device)
    expected_encoder_names = set(encoders)
    saved_encoders = require_mapping(checkpoint["encoders"], "checkpoint.encoders")
    require(
        set(saved_encoders) == expected_encoder_names,
        f"Encoder names mismatch: saved={sorted(saved_encoders)}, expected={sorted(expected_encoder_names)}",
    )
    state_tensor_elements = validate_state_dict("model", checkpoint["model"], model)
    for name, encoder in encoders.items():
        state_tensor_elements += validate_state_dict(
            f"encoder[{name}]", saved_encoders[name], encoder
        )
        encoders[name] = encoder.to(replay_device).eval()
    model = model.to(replay_device).eval()

    parameters = list(model.parameters()) + [
        parameter for encoder in encoders.values() for parameter in encoder.parameters()
    ]
    trainable_parameters = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    require_equal(
        "source.training.trainable_parameters",
        source["training"].get("trainable_parameters"),
        trainable_parameters,
    )
    current_provenance = model.provenance() if hasattr(model, "provenance") else None
    require_equal("source.provenance", source.get("provenance"), current_provenance)
    criterion = nn.CrossEntropyLoss()

    # HARD VALIDATION GATE.  No test loader iteration and no formal filesystem
    # write occurs before every check below succeeds.
    validation = run_epoch(
        replay_args,
        valid_loader,
        encoders,
        modality_dict,
        model,
        criterion,
        replay_device,
    )
    validate_evaluation_alignment(
        "validation",
        validation,
        valid_loader,
        np.asarray(labels),
        np.asarray(observed),
        data_dict,
        modality_dict,
        num_classes,
    )
    replayed_validation_metrics = metric_bundle(
        validation["labels"],
        validation["predictions"],
        validation["probabilities"],
        num_classes,
    )
    assert_metric_tree_close(
        source["validation"],
        replayed_validation_metrics,
        "validation",
        args_cli.metric_atol,
    )

    print(
        "[Validation gate passed] "
        f"model={source['model']} data={source['dataset']} seed={source['seed']} "
        f"macro_f1={replayed_validation_metrics['macro_f1']:.6f}",
        flush=True,
    )
    # Close the concurrent-promoter race before the first test-loader
    # iteration.  Result no-clobber alone is too late: two processes could
    # otherwise both evaluate test and only collide while publishing.
    require_unchanged_digest("source result", source_path, source_digest)
    require_unchanged_digest("checkpoint", checkpoint_path, checkpoint_digest)
    require_unchanged_digest("progress", progress_path, progress_digest)
    acquire_test_evaluation_claim(
        claim_path,
        {
            "status": "test_evaluation_claimed",
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
            "model": source["model"],
            "dataset": source["dataset"],
            "seed": source["seed"],
            "source_result": str(source_path),
            "source_result_sha256": source_digest,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest,
            "configuration_sha256": configuration_digest,
        },
    )
    require_unchanged_digest("source result", source_path, source_digest)
    require_unchanged_digest("checkpoint", checkpoint_path, checkpoint_digest)
    require_unchanged_digest("progress", progress_path, progress_digest)
    test_started = time.perf_counter()
    testing = run_epoch(
        replay_args,
        test_loader,
        encoders,
        modality_dict,
        model,
        criterion,
        replay_device,
    )
    test_source_indices = validate_evaluation_alignment(
        "test",
        testing,
        test_loader,
        np.asarray(labels),
        np.asarray(observed),
        data_dict,
        modality_dict,
        num_classes,
    )
    test_metrics = metric_bundle(
        testing["labels"], testing["predictions"], testing["probabilities"], num_classes
    )
    test_subsets = subset_metrics(testing, num_classes)
    test_elapsed_seconds = time.perf_counter() - test_started
    prediction_frame = create_prediction_frame(testing, test_source_indices, num_classes)
    require_unchanged_digest("source result", source_path, source_digest)
    require_unchanged_digest("checkpoint", checkpoint_path, checkpoint_digest)
    require_unchanged_digest("progress", progress_path, progress_digest)

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    formal_result = copy.deepcopy(source)
    formal_result["status"] = "complete"
    formal_result["validation"] = replayed_validation_metrics
    formal_result["test"] = test_metrics
    formal_result["test_subsets"] = test_subsets
    formal_result["completed_at"] = completed_at
    formal_result["artifacts"] = {
        "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_path.resolve()),
        "progress": None,
        "source_validation_result": str(source_path),
        "source_validation_progress": str(progress_path),
        "source_training_claim": source["artifacts"].get("training_claim"),
        "test_evaluation_claim": str(claim_path.resolve()),
    }
    formal_result["formal_evaluation"] = {
        "method": "strict validation replay before one-time test evaluation",
        "source_status": "validation_complete",
        "validation_metric_atol": args_cli.metric_atol,
        "validation_gate_passed": True,
        "source_result_sha256": source_digest,
        "source_validation_progress_sha256": progress_digest,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_args_sha256": checkpoint_args_digest,
        "configuration_sha256": configuration_digest,
        "configuration_fingerprint_version": 1,
        "split_id_label_sha256": split_fingerprints,
        "strict_state_load": True,
        "state_tensor_elements_checked": state_tensor_elements,
        "replay_device": replay_device_name,
        "test_elapsed_seconds": test_elapsed_seconds,
        "evaluator": str(Path(__file__).resolve()),
    }
    publish_without_overwrite(
        result_path.resolve(), formal_result, predictions_path.resolve(), prediction_frame
    )
    print(
        f"[Test] accuracy={test_metrics['accuracy']:.4f} "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"macro_auroc={test_metrics['macro_auroc']:.4f}",
        flush=True,
    )
    print(f"[Saved] {result_path.resolve()}", flush=True)
    print(f"[Saved] {predictions_path.resolve()}", flush=True)
    return result_path.resolve(), predictions_path.resolve()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
