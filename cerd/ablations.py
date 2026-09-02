"""Canonical controls for the eight pre-specified matched ablations.

The public training entry point applies one control profile on top of the
otherwise matched training configuration.  Keeping this mapping in one small
module prevents CLI, checkpoint metadata, documentation, and tests from
silently assigning different meanings to an ablation ID.
"""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from typing import Any, Mapping


FULL_MODEL_ID = "full"
ABLATION_IDS = (
    "dense_backbone",
    "no_provenance",
    "uniform_branch_weights",
    "mean_pooling",
    "no_stochastic_context",
    "no_completion",
    "no_mofe",
    "no_output_gate",
)

# These values describe the unablated control path.  Each matched arm below
# changes exactly one boolean; numerical hyperparameters remain in the base
# dataset configuration and are not smuggled into this mapping.
FULL_CONTROL_PROFILE = {
    "use_sparse_moe_backbone": True,
    "use_provenance_embeddings": True,
    "use_reliability_branch_weights": True,
    "use_attentive_token_pooling": True,
    "use_stochastic_reconstruction_context": True,
    "use_latent_completion": True,
    "use_more_fewer_objective": True,
    "generator_output_gate": True,
}

ABLATION_CONTROL = {
    "dense_backbone": "use_sparse_moe_backbone",
    "no_provenance": "use_provenance_embeddings",
    "uniform_branch_weights": "use_reliability_branch_weights",
    "mean_pooling": "use_attentive_token_pooling",
    "no_stochastic_context": "use_stochastic_reconstruction_context",
    "no_completion": "use_latent_completion",
    "no_mofe": "use_more_fewer_objective",
    "no_output_gate": "generator_output_gate",
}


def _ordered_ids_bytes() -> bytes:
    return json.dumps(
        ABLATION_IDS,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


ABLATION_ORDER_SHA256 = hashlib.sha256(_ordered_ids_bytes()).hexdigest()


def canonical_control_profile(ablation_id: str) -> dict[str, bool]:
    """Return the full boolean profile for one public ablation ID."""

    if ablation_id == FULL_MODEL_ID:
        return dict(FULL_CONTROL_PROFILE)
    if ablation_id not in ABLATION_CONTROL:
        raise ValueError(
            f"unknown ablation_id {ablation_id!r}; expected {FULL_MODEL_ID!r} "
            f"or one of {ABLATION_IDS!r}"
        )
    profile = dict(FULL_CONTROL_PROFILE)
    profile[ABLATION_CONTROL[ablation_id]] = False
    return profile


def apply_matched_ablation(args: Namespace) -> Namespace:
    """Resolve a canonical ablation profile onto a training namespace.

    ``full`` is deliberately neutral for already-present legacy method flags:
    it fills only missing controls.  A named matched arm is strict and writes
    all eight controls, preventing an accidental compound ablation.
    """

    ablation_id = getattr(args, "ablation_id", FULL_MODEL_ID)
    if ablation_id is None:
        ablation_id = FULL_MODEL_ID
    profile = canonical_control_profile(str(ablation_id))
    args.ablation_id = str(ablation_id)
    args.ablation_order_sha256 = ABLATION_ORDER_SHA256
    for name, value in profile.items():
        if args.ablation_id == FULL_MODEL_ID and hasattr(args, name):
            continue
        setattr(args, name, value)
    return args


def normalize_checkpoint_protocol_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill ablation metadata for replay without changing legacy values.

    Historical checkpoints have no ``ablation_id``.  They normalize to the
    neutral ``full`` identifier and retain any method flags they did record.
    Explicit named arms fail closed if their stored control profile has been
    altered, so a checkpoint cannot be mislabeled as a matched ablation.
    """

    normalized = dict(parameters)
    explicit_id = "ablation_id" in normalized
    ablation_id = str(normalized.get("ablation_id", FULL_MODEL_ID))
    profile = canonical_control_profile(ablation_id)
    normalized["ablation_id"] = ablation_id

    recorded_hash = normalized.get("ablation_order_sha256")
    if recorded_hash is not None and recorded_hash != ABLATION_ORDER_SHA256:
        raise ValueError(
            "checkpoint ablation order hash does not match public v1 order"
        )
    normalized["ablation_order_sha256"] = ABLATION_ORDER_SHA256

    for name, expected in profile.items():
        if (
            explicit_id
            and ablation_id != FULL_MODEL_ID
            and name in normalized
            and normalized[name] is not expected
        ):
            raise ValueError(
                f"checkpoint {ablation_id!r} has non-canonical control {name!r}"
            )
        normalized.setdefault(name, expected)

    configured_mofe_weight = normalized.get("more_fewer_rank_loss_weight")
    if ablation_id == "no_mofe":
        expected_effective_weight = 0.0
    elif configured_mofe_weight is not None:
        expected_effective_weight = float(configured_mofe_weight)
    else:
        expected_effective_weight = None
    if expected_effective_weight is not None:
        recorded_effective_weight = normalized.get(
            "effective_more_fewer_rank_loss_weight"
        )
        if (
            explicit_id
            and ablation_id != FULL_MODEL_ID
            and recorded_effective_weight is not None
            and float(recorded_effective_weight) != expected_effective_weight
        ):
            raise ValueError(
                f"checkpoint {ablation_id!r} has a non-canonical effective "
                "more/fewer-modality weight"
            )
        normalized.setdefault(
            "effective_more_fewer_rank_loss_weight",
            expected_effective_weight,
        )
    return normalized
