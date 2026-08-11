from argparse import Namespace

import pytest

from train import resolve_defaults, validate_objective_compatibility


def arguments(*, data="abcd", variant="auto", modality=None):
    return Namespace(
        data=data,
        variant=variant,
        modality=modality,
        batch_size=None,
        weight_decay=None,
        sampler_power=None,
        class_weight_power=None,
        num_layers_pred=None,
        dual_boundary_rank_loss_weight=None,
        more_fewer_rank_loss_weight=None,
        branch_distill_loss_weight=None,
        dataset_manifest=None,
        adni_data_root="data/adni",
    )


def test_auto_configuration_is_binary_safe_mofe():
    args = resolve_defaults(arguments())
    assert args.variant == "mofe"
    assert args.modality == "SRDGNPME"
    assert args.batch_size == 64
    assert args.dual_boundary_rank_loss_weight == 0.0
    assert args.more_fewer_rank_loss_weight == 0.1
    validate_objective_compatibility(args, num_classes=2)


def test_binary_target_rejects_explicit_dual_boundary_objective():
    args = resolve_defaults(arguments(variant="dbr"))
    with pytest.raises(ValueError, match="requires exactly three classes"):
        validate_objective_compatibility(args, num_classes=2)


def test_adni_uses_legacy_four_modality_default():
    args = resolve_defaults(arguments(data="adni"))
    assert args.modality == "IGCB"


@pytest.mark.parametrize("data", ["abcd", "adni"])
def test_explicit_modality_selection_is_preserved(data):
    args = resolve_defaults(arguments(data=data, modality="GN"))
    assert args.modality == "GN"
