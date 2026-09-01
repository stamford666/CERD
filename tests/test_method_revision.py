import inspect
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cerd.model import (
    AGMGFlexMoE,
    ConditionalGenerator,
    ReliabilityBranchFusion,
    sample_observed_reconstruction_groups,
)
from train import parse_args, resolve_defaults, validate_objective_compatibility


def _assignments(groups):
    result = {}
    for (target, context), rows in groups.items():
        for row in rows.tolist():
            result.setdefault(row, []).append((target, context))
    return result


@pytest.mark.parametrize("drop_probability", [0.1, 0.25, 1.0])
def test_stochastic_context_is_nonempty_proper_observed_subset(drop_probability):
    observed = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]],
        dtype=torch.bool,
    )
    torch.manual_seed(7)
    assignments = _assignments(
        sample_observed_reconstruction_groups(
            observed,
            targets_per_sample=4,
            context_dropout_probability=drop_probability,
        )
    )
    assert set(assignments) == {0, 1}
    for row, tasks in assignments.items():
        observed_set = set(observed[row].nonzero(as_tuple=False).flatten().tolist())
        contexts = {context for _, context in tasks}
        assert len(contexts) == 1
        context = set(next(iter(contexts)))
        targets = {target for target, _ in tasks}
        assert context
        assert context < observed_set
        assert targets == observed_set - context


def test_zero_context_dropout_preserves_groups_and_rng():
    observed = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool
    )
    torch.manual_seed(19)
    default_groups = sample_observed_reconstruction_groups(observed, 1)
    default_next = torch.rand(3)
    torch.manual_seed(19)
    explicit_groups = sample_observed_reconstruction_groups(
        observed, 1, context_dropout_probability=0.0
    )
    explicit_next = torch.rand(3)
    assert default_groups.keys() == explicit_groups.keys()
    for key in default_groups:
        assert torch.equal(default_groups[key], explicit_groups[key])
    assert torch.equal(default_next, explicit_next)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_context_dropout_rejects_invalid_probability(value):
    with pytest.raises(ValueError, match="finite and in"):
        sample_observed_reconstruction_groups(
            torch.ones(1, 2, dtype=torch.bool), 2, value
        )


def test_generator_gate_bypass_preserves_parameters_and_initialization():
    torch.manual_seed(23)
    gated = ConditionalGenerator(
        hidden_dim=8,
        num_patches=2,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_output_gate=True,
    )
    torch.manual_seed(23)
    ungated = ConditionalGenerator(
        hidden_dim=8,
        num_patches=2,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_output_gate=False,
    )
    assert gated.state_dict().keys() == ungated.state_dict().keys()
    for key in gated.state_dict():
        assert torch.equal(gated.state_dict()[key], ungated.state_dict()[key])
    context = torch.randn(3, 4, 8)
    gated.eval()
    ungated.eval()
    assert not torch.equal(gated(context), ungated(context))


def test_entropy_confidence_is_invariant_to_common_logit_shift():
    torch.manual_seed(29)
    fusion = ReliabilityBranchFusion(
        hidden_dim=8,
        num_modalities=2,
        output_dim=3,
        num_layers_pred=1,
        dropout=0.0,
        confidence_mode="entropy_detached",
    ).eval()
    features = [torch.randn(4, 8), torch.randn(4, 8)]
    usable = torch.ones(4, 2, dtype=torch.bool)
    reliability = torch.ones(4, 2)
    before = fusion(features, usable, reliability)[2]
    heads = [fusion.joint_head, *fusion.unimodal_heads, *fusion.pair_heads]
    with torch.no_grad():
        for head in heads:
            head.network[-1].bias.add_(10.0)
    after = fusion(features, usable, reliability)[2]
    assert torch.allclose(before, after, atol=1e-6, rtol=0.0)


def test_exp_entropy_confidence_matches_detached_exp_negative_entropy():
    torch.manual_seed(31)
    fusion = ReliabilityBranchFusion(
        hidden_dim=8,
        num_modalities=2,
        output_dim=3,
        num_layers_pred=1,
        dropout=0.0,
        confidence_mode="entropy_exp_detached",
    ).eval()
    features = [torch.randn(4, 8), torch.randn(4, 8)]
    usable = torch.ones(4, 2, dtype=torch.bool)
    reliability = torch.ones(4, 2)
    _, branch_logits, branch_weights, branch_mask, *_ = fusion(
        features, usable, reliability
    )
    branch_probabilities = branch_logits.softmax(dim=-1)
    entropy = -(
        branch_probabilities.clamp_min(1e-8)
        * branch_probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1)
    expected = torch.exp(-entropy).detach()
    expected = expected.masked_fill(~branch_mask, 0.0)
    expected = expected / expected.sum(dim=1, keepdim=True)
    assert torch.allclose(branch_weights, expected, atol=1e-6, rtol=0.0)
    confidence_gradient = torch.autograd.grad(
        branch_weights[0, 0],
        branch_logits,
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert confidence_gradient is None


def test_generator_only_task_grad_constraints():
    kwargs = dict(
        num_modalities=2,
        full_modality_index=0,
        num_patches=2,
        hidden_dim=8,
        output_dim=3,
        num_layers_fus=1,
        num_layers_pred=1,
        num_experts=2,
        num_routers=1,
        top_k=1,
        num_heads=2,
        dropout=0.0,
        gen_num_layers=1,
        gen_num_heads=2,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        AGMGFlexMoE(
            **kwargs,
            generator_task_grad=True,
            generator_only_task_grad=True,
        )
    with pytest.raises(ValueError, match="requires use_generators"):
        AGMGFlexMoE(
            **kwargs,
            use_generators=False,
            generator_only_task_grad=True,
        )
    with pytest.raises(ValueError, match="requires pattern-aware"):
        AGMGFlexMoE(
            **kwargs,
            pattern_aware_reconstruction=False,
            recon_context_dropout_probability=0.25,
        )


def test_generator_only_task_gradient_reaches_generator_with_detached_context():
    torch.manual_seed(37)
    model = AGMGFlexMoE(
        num_modalities=2,
        full_modality_index=0,
        num_patches=2,
        hidden_dim=8,
        output_dim=3,
        num_layers_fus=1,
        num_layers_pred=1,
        num_experts=2,
        num_routers=1,
        top_k=1,
        num_heads=2,
        dropout=0.0,
        gen_num_layers=1,
        gen_num_heads=2,
        vectorized_generation=True,
        generator_only_task_grad=True,
    ).eval()
    generator_context_requires_grad = []
    hook = model.generators[1].register_forward_pre_hook(
        lambda _module, inputs: generator_context_requires_grad.append(
            inputs[0].requires_grad
        )
    )
    observed_tokens = torch.randn(2, 2, 8, requires_grad=True)
    missing_tokens = torch.randn(2, 2, 8, requires_grad=True)
    output = model(
        observed_tokens,
        missing_tokens,
        observed_mask=torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
        expert_indices=torch.zeros(1, dtype=torch.long),
        return_recon_loss=False,
    )
    output["logits"].square().sum().backward()
    hook.remove()
    assert generator_context_requires_grad == [False]
    generator_gradients = [
        parameter.grad
        for parameter in model.generators[1].parameters()
        if parameter.grad is not None
    ]
    assert generator_gradients
    assert sum(gradient.abs().sum() for gradient in generator_gradients) > 0
    assert missing_tokens.grad is not None
    assert torch.count_nonzero(missing_tokens.grad) == 0


def test_normalized_token_smooth_l1_formula_and_detached_target():
    torch.manual_seed(41)
    token_weight = 0.2
    model = AGMGFlexMoE(
        num_modalities=2,
        full_modality_index=0,
        num_patches=2,
        hidden_dim=8,
        output_dim=3,
        num_layers_fus=1,
        num_layers_pred=1,
        num_experts=2,
        num_routers=1,
        top_k=1,
        num_heads=2,
        dropout=0.0,
        gen_num_layers=1,
        gen_num_heads=2,
        recon_normalized_token_loss_weight=token_weight,
    )
    prediction = torch.randn(3, 2, 8, requires_grad=True)
    target = torch.randn(3, 2, 8, requires_grad=True)

    actual = model._recon_loss_disc_batch(0, prediction, target)
    projected_prediction = model.recon_projectors[0](prediction.mean(dim=1))
    projected_target = model.recon_projectors[0](target.mean(dim=1)).detach()
    pooled_loss = 1.0 - F.cosine_similarity(
        projected_prediction,
        projected_target,
        dim=-1,
    )
    normalized_prediction = F.layer_norm(prediction, (prediction.shape[-1],))
    normalized_target = F.layer_norm(
        target.detach(),
        (target.shape[-1],),
    )
    token_loss = F.smooth_l1_loss(
        normalized_prediction,
        normalized_target,
        reduction="none",
    ).mean(dim=(1, 2))
    expected = pooled_loss + token_weight * token_loss

    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)
    actual.sum().backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) > 0
    assert target.grad is None


def test_resolve_defaults_preserves_legacy_method_by_dataset():
    base = dict(
        variant="core",
        modality=None,
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
    adni = resolve_defaults(type("Args", (), {**base, "data": "adni"})())
    abcd = resolve_defaults(type("Args", (), {**base, "data": "abcd"})())
    for args in (adni, abcd):
        assert args.pattern_aware_reconstruction is False
        assert args.recon_context_dropout_probability == 0.0
        assert args.recon_normalized_token_loss_weight == 0.0
        assert args.branch_confidence_mode == "evidence"
        assert args.generator_output_gate is True
        assert args.generator_only_task_grad is False
        validate_objective_compatibility(args, num_classes=3)


@pytest.mark.parametrize("data", ["abcd", "adni"])
def test_cli_defaults_preserve_legacy_method(monkeypatch, data):
    monkeypatch.setattr(sys, "argv", ["train.py", "--data", data])
    args = resolve_defaults(parse_args())
    assert args.pattern_aware_reconstruction is False
    assert args.recon_context_dropout_probability == 0.0
    assert args.recon_normalized_token_loss_weight == 0.0
    assert args.branch_confidence_mode == "evidence"
    assert args.generator_output_gate is True
    assert args.generator_only_task_grad is False


@pytest.mark.parametrize(
    ("data", "generator_only_flag", "expected_generator_only"),
    [
        ("abcd", "--no-generator-only-task-grad", False),
        ("adni", "--generator-only-task-grad", True),
    ],
)
def test_revision_method_is_explicit_cli_opt_in(
    monkeypatch, data, generator_only_flag, expected_generator_only
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            data,
            "--pattern-aware-reconstruction",
            "--recon-context-dropout-probability",
            "0.25",
            "--recon-normalized-token-loss-weight",
            "0.05",
            "--branch-confidence-mode",
            "entropy_detached",
            "--generator-output-gate",
            generator_only_flag,
        ],
    )
    args = resolve_defaults(parse_args())
    assert args.pattern_aware_reconstruction is True
    assert args.recon_context_dropout_probability == 0.25
    assert args.recon_normalized_token_loss_weight == 0.05
    assert args.branch_confidence_mode == "entropy_detached"
    assert args.generator_output_gate is True
    assert args.generator_only_task_grad is expected_generator_only
    validate_objective_compatibility(args, num_classes=3)


def test_historical_training_scripts_pin_legacy_method_flags():
    required = {
        "--no-pattern-aware-reconstruction",
        "--recon-context-dropout-probability 0",
        "--recon-normalized-token-loss-weight 0",
        "--branch-confidence-mode evidence",
        "--generator-output-gate",
        "--no-generator-only-task-grad",
    }
    for script in (Path("scripts/train_abcd.sh"), Path("scripts/train_adni.sh")):
        contents = script.read_text()
        assert all(flag in contents for flag in required)


def test_historical_positional_constructor_mapping_is_unchanged():
    historical_names = [
        "num_modalities",
        "full_modality_index",
        "num_patches",
        "hidden_dim",
        "output_dim",
        "num_layers_fus",
        "num_layers_pred",
        "num_experts",
        "num_routers",
        "top_k",
        "num_heads",
        "dropout",
        "gen_num_layers",
        "gen_num_heads",
        "recon_use_token_mse",
        "recon_token_mse_weight",
        "pattern_aware_reconstruction",
        "recon_normalized_token_loss_weight",
        "vectorized_generation",
        "recon_targets_per_sample",
        "use_generators",
        "dynamic_branch_fusion",
        "complete_joint_only",
        "complete_specialist_weight",
        "branch_confidence_mode",
        "token_attention_init",
        "generator_task_grad",
        "standard_transformer_residual",
        "gated_transformer_residual",
        "ordinal_fusion_weight",
        "ordinal_aux_loss_weight",
        "ordinal_head_type",
        "uncertainty_aware_ordinal_fusion",
        "enable_class1_aux_head",
        "learn_observed_reliability",
        "centered_evidence_confidence",
        "class_conditional_fusion",
        "normalized_gate_loss",
        "enable_supervised_contrastive",
        "supervised_contrastive_projection_dim",
        "unique_dim",
        "ortho_reg_weight",
        "shared_align_weight",
        "more_tail_rank",
        "dual_local_boundary_loss_weight",
    ]
    signature = inspect.signature(AGMGFlexMoE.__init__)
    positional_names = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional_names == ["self", *historical_names]
    for name in (
        "recon_context_dropout_probability",
        "generator_only_task_grad",
        "generator_output_gate",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    required_values = {
        "num_modalities": 2,
        "full_modality_index": 0,
        "num_patches": 2,
        "hidden_dim": 8,
        "output_dim": 3,
        "num_layers_fus": 1,
        "num_layers_pred": 1,
        "num_experts": 2,
        "num_routers": 1,
        "top_k": 1,
    }
    positional_values = [
        required_values.get(name, signature.parameters[name].default)
        for name in historical_names
    ]
    model = AGMGFlexMoE(*positional_values)
    assert model.recon_context_dropout_probability == 0.0
    assert model.generator_only_task_grad is False
    assert model.generator_output_gate is True
    with pytest.raises(TypeError):
        AGMGFlexMoE(*positional_values, 0.25)


def test_positive_context_dropout_requires_pattern_aware_reconstruction():
    args = type(
        "Args",
        (),
        {
            "recon_context_dropout_probability": 0.25,
            "pattern_aware_reconstruction": False,
            "recon_normalized_token_loss_weight": 0.05,
            "dual_boundary_rank_loss_weight": 0.0,
        },
    )()
    with pytest.raises(ValueError, match="requires pattern-aware"):
        validate_objective_compatibility(args, num_classes=3)
