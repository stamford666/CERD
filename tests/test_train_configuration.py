from argparse import Namespace

import pytest

from train import (
    public_protocol_parameters,
    resolve_defaults,
    validate_objective_compatibility,
)


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


def test_auto_configuration_is_binary_reference_safe_mofe():
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


def test_public_protocol_parameters_exclude_local_paths():
    args = Namespace(
        modality="IGCB",
        lr=1e-4,
        dataset_manifest="/private/abcd/manifest.json",
        adni_data_root="/private/adni",
        output_dir="/private/output",
        torch_device="cuda:0",
        device=0,
    )
    parameters = public_protocol_parameters(args)
    assert parameters == {"modality": "IGCB", "lr": 1e-4}
    assert not any(
        token in key for key in parameters for token in ("path", "root", "dir", "device")
    )
