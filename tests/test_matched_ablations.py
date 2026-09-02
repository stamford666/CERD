import ast
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import train as train_module
from cerd.ablations import (
    ABLATION_CONTROL,
    ABLATION_IDS,
    ABLATION_ORDER_SHA256,
    FULL_CONTROL_PROFILE,
    canonical_control_profile,
    normalize_checkpoint_protocol_parameters,
)
from cerd.data import create_loaders
from cerd.model import (
    AGMGFlexMoE,
    ReliabilityBranchFusion,
    sample_observed_reconstruction_groups,
)
from cerd.sampling import DATA_ORDER_RNG, balanced_train_loader, data_order_generator
from train import (
    canonical_configuration_sha256,
    create_run_claim,
    existing_exact_run,
    normalize_fold_id,
    parse_args,
    publish_exclusive,
    public_protocol_parameters,
    resolve_defaults,
    run_stem,
    sha256_file,
    train_epoch,
)


EXPECTED_IDS = (
    "dense_backbone",
    "no_provenance",
    "uniform_branch_weights",
    "mean_pooling",
    "no_stochastic_context",
    "no_completion",
    "no_mofe",
    "no_output_gate",
)


def _model(profile=None, *, num_layers_fus=1, include_controls=True):
    controls = dict(FULL_CONTROL_PROFILE if profile is None else profile)
    controls.pop("use_more_fewer_objective")
    if not include_controls:
        controls = {}
    return AGMGFlexMoE(
        num_modalities=2,
        full_modality_index=0,
        num_patches=2,
        hidden_dim=8,
        output_dim=3,
        num_layers_fus=num_layers_fus,
        num_layers_pred=1,
        num_experts=2,
        num_routers=1,
        top_k=1,
        num_heads=2,
        dropout=0.0,
        gen_num_layers=1,
        gen_num_heads=2,
        pattern_aware_reconstruction=True,
        recon_context_dropout_probability=0.25,
        vectorized_generation=True,
        recon_targets_per_sample=2,
        **controls,
    )


def _state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _row_assignments(groups):
    assignments = {}
    for (target, context), rows in groups.items():
        for row in rows.tolist():
            assignments.setdefault(row, {})[target] = context
    return assignments


def test_registry_config_and_order_digest_are_exact():
    config = json.loads(
        Path("configs/matched_ablations_v1.json").read_text(encoding="utf-8")
    )
    encoded = json.dumps(
        EXPECTED_IDS,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    assert ABLATION_IDS == EXPECTED_IDS
    assert hashlib.sha256(encoded).hexdigest() == ABLATION_ORDER_SHA256
    assert config["order_sha256"] == ABLATION_ORDER_SHA256
    assert config["data_order_rng"] == DATA_ORDER_RNG
    assert config["epoch_order_hash_closure"] is False
    assert config["output_identity"] == [
        "fold_id",
        "split_receipt_sha256",
        "seed",
        "data_order_seed",
        "configuration_sha256",
    ]
    assert config["aligned_completion_module_construction"] is True
    assert config["no_mofe_effective_weight"] == 0.0
    assert config["full_control_profile"] == FULL_CONTROL_PROFILE
    assert [row["id"] for row in config["ablations"]] == list(EXPECTED_IDS)
    assert {
        row["id"]: row["control"] for row in config["ablations"]
    } == ABLATION_CONTROL
    assert all(row["value"] is False for row in config["ablations"])
    renderer_tree = ast.parse(
        Path("scripts/render_release_results.py").read_text(encoding="utf-8")
    )
    renderer_rows = next(
        ast.literal_eval(node.value)
        for node in renderer_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_ABLATIONS"
            for target in node.targets
        )
    )
    assert tuple(identifier for identifier, _label in renderer_rows) == EXPECTED_IDS


@pytest.mark.parametrize("ablation_id", EXPECTED_IDS)
def test_each_profile_has_exactly_one_named_control_diff(ablation_id):
    profile = canonical_control_profile(ablation_id)
    differences = {
        name
        for name, full_value in FULL_CONTROL_PROFILE.items()
        if profile[name] != full_value
    }
    assert differences == {ABLATION_CONTROL[ablation_id]}
    assert profile[ABLATION_CONTROL[ablation_id]] is False


@pytest.mark.parametrize("ablation_id", EXPECTED_IDS)
def test_cli_resolves_each_canonical_profile(monkeypatch, ablation_id):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--data",
            "adni",
            "--ablation-id",
            ablation_id,
            "--data-order-seed",
            "73",
        ],
    )
    args = resolve_defaults(parse_args())
    assert args.ablation_id == ablation_id
    assert args.data_order_seed == 73
    assert args.ablation_order_sha256 == ABLATION_ORDER_SHA256
    assert {
        name: getattr(args, name) for name in FULL_CONTROL_PROFILE
    } == canonical_control_profile(ablation_id)
    protocol = public_protocol_parameters(args)
    assert protocol["ablation_id"] == ablation_id
    assert protocol["data_order_seed"] == 73
    assert protocol["ablation_order_sha256"] == ABLATION_ORDER_SHA256
    if ablation_id == "no_mofe":
        assert args.more_fewer_rank_loss_weight == 0.1
        assert args.effective_more_fewer_rank_loss_weight == 0.0
        assert protocol["effective_more_fewer_rank_loss_weight"] == 0.0


def test_checkpoint_protocol_normalizes_legacy_and_rejects_tampered_arm():
    legacy = normalize_checkpoint_protocol_parameters(
        {"generator_output_gate": False, "hidden_dim": 128}
    )
    assert legacy["ablation_id"] == "full"
    assert legacy["generator_output_gate"] is False
    assert legacy["use_sparse_moe_backbone"] is True
    assert legacy["ablation_order_sha256"] == ABLATION_ORDER_SHA256

    named = normalize_checkpoint_protocol_parameters(
        {
            "ablation_id": "no_provenance",
            "more_fewer_rank_loss_weight": 0.1,
            **canonical_control_profile("no_provenance"),
        }
    )
    assert named["use_provenance_embeddings"] is False
    assert named["effective_more_fewer_rank_loss_weight"] == 0.1
    with pytest.raises(ValueError, match="non-canonical control"):
        normalize_checkpoint_protocol_parameters(
            {
                "ablation_id": "no_provenance",
                "use_provenance_embeddings": True,
            }
        )
    with pytest.raises(ValueError, match="order hash"):
        normalize_checkpoint_protocol_parameters(
            {"ablation_id": "full", "ablation_order_sha256": "0" * 64}
        )
    with pytest.raises(ValueError, match="effective more/fewer"):
        normalize_checkpoint_protocol_parameters(
            {
                "ablation_id": "no_mofe",
                "effective_more_fewer_rank_loss_weight": 0.1,
            }
        )


def test_all_arm_stems_are_digest_bound_and_do_not_collide():
    base = dict(
        data="adni",
        variant="mofe",
        seed=7,
        data_order_seed=17,
        fold_id=None,
        split_receipt_sha256=None,
    )
    stems = [run_stem(Namespace(**base, ablation_id="full"))]
    stems.extend(
        run_stem(Namespace(**base, ablation_id=ablation_id))
        for ablation_id in EXPECTED_IDS
    )
    assert len(stems) == len(set(stems)) == 9
    assert stems[0].startswith("adni_mofe_full_seed7_order17_cfg")
    assert len(stems[0].rsplit("_cfg", 1)[1]) == 64


def test_stem_binds_fold_order_and_recomputed_configuration_digest():
    args = Namespace(
        data="abcd",
        variant="mofe",
        seed=9,
        data_order_seed=109,
        fold_id="2",
        split_receipt_sha256="a" * 64,
        ablation_id="mean_pooling",
    )
    digest = canonical_configuration_sha256(args)
    args.configuration_sha256 = digest
    stem = run_stem(args)
    assert "_mean_pooling_fold2_seed9_order109_" in stem
    assert stem.endswith(f"_cfg{digest}")
    args.configuration_sha256 = "b" * 64
    with pytest.raises(ValueError, match="does not match"):
        run_stem(args)
    for invalid in ("", "../fold", "fold.1", "x" * 33):
        with pytest.raises(ValueError, match="fold_id"):
            normalize_fold_id(invalid)


def test_fold_cli_requires_an_explicit_controlled_split_receipt(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--data", "adni", "--fold-id", "0"]
    )
    with pytest.raises(ValueError, match="split-receipt"):
        resolve_defaults(parse_args())

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--data",
            "adni",
            "--fold-id",
            "0",
            "--split-receipt-sha256",
            "a" * 64,
        ],
    )
    args = resolve_defaults(parse_args())
    assert args.fold_id == "0"
    assert args.split_receipt_sha256 == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("data", "adni"),
        ("variant", "core"),
        ("seed", 10),
        ("data_order_seed", 110),
        ("fold_id", "3"),
        ("split_receipt_sha256", "b" * 64),
        ("ablation_id", "no_mofe"),
        ("hidden_dim", 32),
    ),
)
def test_each_run_identity_dimension_changes_configuration_digest(field, value):
    base = dict(
        data="abcd",
        variant="mofe",
        seed=9,
        data_order_seed=109,
        fold_id="2",
        split_receipt_sha256="a" * 64,
        ablation_id="mean_pooling",
        hidden_dim=16,
    )
    changed = dict(base)
    changed[field] = value
    assert canonical_configuration_sha256(Namespace(**base)) != (
        canonical_configuration_sha256(Namespace(**changed))
    )


def _existing_run_fixture(tmp_path, *, stem="run"):
    args = Namespace(
        data="adni",
        variant="mofe",
        seed=7,
        data_order_seed=17,
        fold_id="0",
        split_receipt_sha256="c" * 64,
        ablation_id="no_mofe",
        more_fewer_rank_loss_weight=0.1,
        effective_more_fewer_rank_loss_weight=0.0,
    )
    args.configuration_sha256 = canonical_configuration_sha256(args)
    parameters = public_protocol_parameters(args)
    checkpoint_path = tmp_path / f"{stem}.pt"
    result_path = tmp_path / f"{stem}.json"
    torch.save({"args": parameters, "model": {}}, checkpoint_path)
    result = {
        "schema": "cerd-run-v2",
        "status": "complete",
        "dataset": "adni",
        "variant": "mofe",
        "ablation_id": "no_mofe",
        "seed": 7,
        "fold_id": "0",
        "data_order_seed": 17,
        "configuration_sha256": args.configuration_sha256,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "protocol": {
            "configuration_schema": train_module.PUBLIC_CONFIGURATION_SCHEMA,
            "parameters": parameters,
        },
    }
    result_path.write_text(
        json.dumps(result, allow_nan=False),
        encoding="utf-8",
    )
    return checkpoint_path, result_path, parameters, result


def test_existing_output_allows_only_digest_bound_exact_reentry(tmp_path):
    checkpoint_path, result_path, parameters, result = _existing_run_fixture(tmp_path)
    assert existing_exact_run(checkpoint_path, result_path, parameters) == result

    result["fold_id"] = "1"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different configuration"):
        existing_exact_run(checkpoint_path, result_path, parameters)


def test_existing_output_rejects_partial_tampered_and_symlinked_state(tmp_path):
    checkpoint_path, result_path, parameters, _ = _existing_run_fixture(
        tmp_path, stem="tampered"
    )
    with checkpoint_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(FileExistsError, match="bad digest"):
        existing_exact_run(checkpoint_path, result_path, parameters)

    partial_checkpoint = tmp_path / "partial.pt"
    torch.save({"args": parameters}, partial_checkpoint)
    with pytest.raises(FileExistsError, match="partial"):
        existing_exact_run(partial_checkpoint, tmp_path / "partial.json", parameters)

    symlink_checkpoint = tmp_path / "linked.pt"
    symlink_checkpoint.symlink_to(partial_checkpoint)
    with pytest.raises(FileExistsError, match="symlinked"):
        existing_exact_run(symlink_checkpoint, tmp_path / "linked.json", parameters)


def test_atomic_claim_refuses_a_second_job_for_the_same_stem(tmp_path):
    claim = tmp_path / ".run.claim"
    create_run_claim(claim, "d" * 64)
    assert claim.read_text(encoding="ascii") == f"{'d' * 64}\n"
    with pytest.raises(FileExistsError, match="already claimed"):
        create_run_claim(claim, "d" * 64)


def test_exclusive_publish_is_atomic_and_never_overwrites(tmp_path):
    temporary = tmp_path / ".run.json.tmp"
    final = tmp_path / "run.json"
    temporary.write_text("first", encoding="utf-8")
    publish_exclusive(temporary, final)
    assert final.read_text(encoding="utf-8") == "first"
    assert not temporary.exists()

    temporary.write_text("second", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_exclusive(temporary, final)
    assert final.read_text(encoding="utf-8") == "first"
    assert temporary.read_text(encoding="utf-8") == "second"


@pytest.mark.parametrize(
    "ablation_id",
    [identifier for identifier in EXPECTED_IDS if identifier != "dense_backbone"],
)
def test_all_nonbackbone_arms_retain_module_tree_and_initialization(ablation_id):
    torch.manual_seed(101)
    full = _model(canonical_control_profile("full"))
    torch.manual_seed(101)
    ablated = _model(canonical_control_profile(ablation_id))
    full_state = _state(full)
    ablated_state = _state(ablated)
    assert full_state.keys() == ablated_state.keys()
    assert all(torch.equal(full_state[name], ablated_state[name]) for name in full_state)
    assert any(layer.mlp_sparse for layer in ablated.backbone.layers)


def test_dense_is_the_only_profile_without_sparse_moe_layers():
    for ablation_id in ("full", *EXPECTED_IDS):
        torch.manual_seed(103)
        model = _model(canonical_control_profile(ablation_id))
        sparse_layers = [layer.mlp_sparse for layer in model.backbone.layers]
        if ablation_id == "dense_backbone":
            assert sparse_layers and not any(sparse_layers)
            assert model.gate_loss() == 0.0
        else:
            assert any(sparse_layers)


def test_dense_restores_reference_rng_and_changes_only_exclusive_ffn_keys():
    torch.manual_seed(104)
    full = _model(canonical_control_profile("full"), num_layers_fus=3)
    full_tail = torch.rand(8)
    torch.manual_seed(104)
    dense = _model(
        canonical_control_profile("dense_backbone"), num_layers_fus=3
    )
    dense_tail = torch.rand(8)

    full_state = _state(full)
    dense_state = _state(dense)
    common_keys = full_state.keys() & dense_state.keys()
    full_only = full_state.keys() - dense_state.keys()
    dense_only = dense_state.keys() - full_state.keys()
    assert common_keys
    assert all(torch.equal(full_state[key], dense_state[key]) for key in common_keys)
    assert full_only and dense_only
    assert all(
        key.startswith("backbone.layers.") and ".mlp." in key
        for key in full_only | dense_only
    )
    assert {
        key.split(".")[2]
        for key in full_only | dense_only
        if key.startswith("backbone.layers.")
    } == {"0", "2"}
    assert not any(
        ".all_gates." in key or ".experts." in key for key in dense_state
    )
    assert torch.equal(full_tail, dense_tail)

    post_backbone_prefixes = (
        "generators.",
        "flag_embeds.",
        "recon_projectors.",
        "token_poolers.",
        "reliability_scorers.",
        "branch_fusion.",
    )
    assert any(key.startswith(post_backbone_prefixes) for key in common_keys)


def test_explicit_full_controls_preserve_omitted_default_state_and_rng():
    torch.manual_seed(105)
    historical = _model(include_controls=False, num_layers_fus=3)
    historical_tail = torch.rand(8)
    torch.manual_seed(105)
    explicit = _model(canonical_control_profile("full"), num_layers_fus=3)
    explicit_tail = torch.rand(8)

    historical_state = _state(historical)
    explicit_state = _state(explicit)
    assert historical_state.keys() == explicit_state.keys()
    assert all(
        torch.equal(historical_state[key], explicit_state[key])
        for key in historical_state
    )
    assert torch.equal(historical_tail, explicit_tail)


def test_no_provenance_bypasses_only_the_constructed_embeddings():
    torch.manual_seed(107)
    full = _model(canonical_control_profile("full")).eval()
    torch.manual_seed(107)
    ablated = _model(canonical_control_profile("no_provenance")).eval()
    with torch.no_grad():
        for model in (full, ablated):
            for embedding in model.flag_embeds:
                embedding.emb.weight[0].fill_(1.0)
                embedding.emb.weight[1].fill_(2.0)

    captured = {}

    def capture(name):
        return lambda _module, inputs: captured.setdefault(
            name, tuple(value.detach().clone() for value in inputs)
        )

    full_hook = full.backbone.register_forward_pre_hook(capture("full"))
    ablated_hook = ablated.backbone.register_forward_pre_hook(capture("ablated"))
    tokens = (torch.randn(2, 2, 8), torch.randn(2, 2, 8))
    observed = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
    kwargs = dict(
        observed_mask=observed,
        expert_indices=torch.zeros(2, dtype=torch.long),
        return_recon_loss=False,
    )
    full(*tokens, **kwargs)
    ablated(*tokens, **kwargs)
    full_hook.remove()
    ablated_hook.remove()

    assert len(full.flag_embeds) == len(ablated.flag_embeds) == 2
    assert torch.allclose(captured["full"][0], tokens[0] + 1.0)
    assert torch.equal(captured["ablated"][0], tokens[0])


def test_uniform_arm_uses_exact_valid_branch_distribution_with_same_modules():
    kwargs = dict(
        hidden_dim=8,
        num_modalities=3,
        output_dim=3,
        num_layers_pred=1,
        dropout=0.0,
        confidence_mode="entropy_detached",
    )
    torch.manual_seed(109)
    full = ReliabilityBranchFusion(**kwargs).eval()
    torch.manual_seed(109)
    uniform = ReliabilityBranchFusion(
        **kwargs, uniform_valid_branch_weights=True
    ).eval()
    assert all(
        torch.equal(full.state_dict()[name], uniform.state_dict()[name])
        for name in full.state_dict()
    )
    features = [torch.randn(2, 8) for _ in range(3)]
    usable = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    reliability = usable.float()
    _, _, weights, mask, _, log_scores, _ = uniform(
        features, usable, reliability
    )
    expected = mask.float() / mask.float().sum(dim=1, keepdim=True)
    assert torch.equal(weights, expected)
    assert torch.equal(log_scores[mask], torch.zeros_like(log_scores[mask]))
    assert torch.equal(
        log_scores[~mask], torch.full_like(log_scores[~mask], -1e4)
    )


def test_mean_pooling_executes_poolers_but_returns_uniform_token_mass():
    torch.manual_seed(113)
    model = _model(canonical_control_profile("mean_pooling")).eval()
    calls = [0, 0]
    hooks = [
        pooler.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, pooler in enumerate(model.token_poolers)
    ]
    tokens = [torch.randn(2, 2, 8) for _ in range(2)]
    output = model(
        *tokens,
        observed_mask=torch.ones(2, 2, dtype=torch.bool),
        expert_indices=torch.zeros(2, dtype=torch.long),
        return_recon_loss=False,
    )
    for hook in hooks:
        hook.remove()
    assert calls == [1, 1]
    assert torch.equal(
        output["token_importance"],
        torch.full_like(output["token_importance"], 0.5),
    )


def test_no_stochastic_context_preserves_targets_and_rng_and_only_expands_context():
    fixture = json.loads(
        Path("tests/fixtures/stochastic_context_golden_v1.json").read_text(
            encoding="utf-8"
        )
    )
    observed = torch.tensor(fixture["observed_mask"], dtype=torch.bool)
    torch.manual_seed(fixture["seed"])
    stochastic = sample_observed_reconstruction_groups(
        observed,
        targets_per_sample=fixture["targets_per_sample"],
        context_dropout_probability=fixture["context_dropout_probability"],
    )
    stochastic_next = torch.rand(5)
    torch.manual_seed(fixture["seed"])
    expanded = sample_observed_reconstruction_groups(
        observed,
        targets_per_sample=fixture["targets_per_sample"],
        context_dropout_probability=fixture["context_dropout_probability"],
        expand_context_to_all_observed=True,
    )
    expanded_next = torch.rand(5)

    stochastic_rows = _row_assignments(stochastic)
    expanded_rows = _row_assignments(expanded)
    assert stochastic_rows.keys() == expanded_rows.keys()
    for row in stochastic_rows:
        assert stochastic_rows[row].keys() == expanded_rows[row].keys()
        observed_set = set(observed[row].nonzero(as_tuple=False).flatten().tolist())
        for target, context in expanded_rows[row].items():
            assert set(context) == observed_set - {target}
    assert torch.equal(stochastic_next, expanded_next)

    def serialized(groups):
        return [
            {"target": target, "context": list(context), "rows": rows.tolist()}
            for (target, context), rows in sorted(groups.items())
        ]

    assert serialized(stochastic) == fixture["stochastic"]
    assert serialized(expanded) == fixture["expanded"]
    assert np.allclose(
        stochastic_next.numpy(),
        np.asarray(fixture["rng_tail"]),
        atol=5e-10,
        rtol=0.0,
    )


def test_no_completion_keeps_modules_but_calls_neither_generator_nor_projector():
    torch.manual_seed(131)
    model = _model(canonical_control_profile("no_completion")).eval()
    generator_calls = [0 for _ in model.generators]
    projector_calls = [0 for _ in model.recon_projectors]
    hooks = []
    for index, generator in enumerate(model.generators):
        hooks.append(
            generator.register_forward_hook(
                lambda _module, _inputs, _output, index=index: generator_calls.__setitem__(
                    index, generator_calls[index] + 1
                )
            )
        )
    for index, projector in enumerate(model.recon_projectors):
        hooks.append(
            projector.register_forward_hook(
                lambda _module, _inputs, _output, index=index: projector_calls.__setitem__(
                    index, projector_calls[index] + 1
                )
            )
        )
    output = model(
        torch.randn(2, 2, 8),
        torch.randn(2, 2, 8),
        observed_mask=torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
        expert_indices=torch.zeros(2, dtype=torch.long),
        return_recon_loss=True,
    )
    for hook in hooks:
        hook.remove()
    assert len(model.generators) == len(model.recon_projectors) == 2
    assert generator_calls == projector_calls == [0, 0]
    assert not output["generated_mask"].any()
    assert output["recon_loss"].item() == 0.0
    assert any(layer.mlp_sparse for layer in model.backbone.layers)


class _MoFeBehaviorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.2))
        self.rank_probe = torch.nn.Parameter(torch.tensor(0.5))
        self.forward_masks = []
        self.gate_calls = 0

    def forward(self, *tokens, observed_mask, **_kwargs):
        self.forward_masks.append(observed_mask.detach().clone())
        batch_size = observed_mask.shape[0]
        logits = torch.stack(
            (
                self.shared.expand(batch_size),
                (-self.shared).expand(batch_size),
                (0.5 * self.shared).expand(batch_size),
            ),
            dim=1,
        )
        return {
            "logits": logits,
            "branch_logits": logits.unsqueeze(1),
            "branch_mask": torch.ones(
                batch_size, 1, dtype=torch.bool, device=logits.device
            ),
            "branch_quality": torch.ones(batch_size, 1, device=logits.device),
            "recon_loss": logits.new_zeros(()),
        }

    def gate_loss(self):
        self.gate_calls += 1
        return self.shared * 0.0


def test_no_mofe_actual_behavior_keeps_views_mask_and_rng_aligned(monkeypatch):
    active_model = {"value": None}
    rank_calls = []
    dropped_masks = []
    original_dropout = train_module.modality_dropout_mask

    def fake_forward_batch(
        model,
        _encoders,
        _modality_dict,
        batch,
        _args,
        _device,
        *,
        reconstruction,
    ):
        tokens, labels, combinations, observed = batch
        output = model(
            *tokens,
            observed_mask=observed,
            expert_indices=combinations,
            return_recon_loss=reconstruction,
        )
        return output, list(tokens), labels, combinations, observed

    def recorded_dropout(observed, probability):
        dropped = original_dropout(observed, probability)
        dropped_masks.append(dropped.detach().clone())
        return dropped

    def rank_loss(*_args, **_kwargs):
        rank_calls.append(active_model["value"])
        return active_model["value"].rank_probe.square()

    monkeypatch.setattr(train_module, "forward_batch", fake_forward_batch)
    monkeypatch.setattr(train_module, "modality_dropout_mask", recorded_dropout)
    monkeypatch.setattr(train_module, "more_fewer_rank_loss", rank_loss)

    batch_size = 4
    batch = (
        tuple(torch.zeros(batch_size, 1, 1) for _ in range(3)),
        torch.tensor([0, 1, 2, 0]),
        torch.zeros(batch_size, dtype=torch.long),
        torch.ones(batch_size, 3, dtype=torch.bool),
    )
    base_args = dict(
        dual_boundary_rank_loss_weight=0.0,
        dual_boundary_rank_margin=0.2,
        dual_boundary_rank_10_weight=2.0 / 3.0,
        branch_distill_start_epoch=99,
        variant="mofe",
        modality_dropout_prob=0.5,
        modality="IGC",
        data="adni",
        distill_temperature=2.0,
        more_fewer_rank_loss_weight=0.6,
        gate_loss_weight=0.0,
        recon_loss_weight=0.0,
        branch_aux_loss_weight=0.0,
        drop_ce_loss_weight=0.1,
        distill_loss_weight=0.1,
        branch_distill_loss_weight=0.0,
        grad_clip=100.0,
    )

    def run(use_mofe):
        model = _MoFeBehaviorModel()
        active_model["value"] = model
        args = Namespace(
            **base_args,
            use_more_fewer_objective=use_mofe,
            effective_more_fewer_rank_loss_weight=(0.6 if use_mofe else 0.0),
        )
        torch.manual_seed(317)
        before_masks = len(dropped_masks)
        losses = train_epoch(
            model,
            {},
            {"I": 0, "G": 1, "C": 2},
            [batch],
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            args,
            torch.device("cpu"),
            epoch=1,
            branch_ema=None,
        )
        return {
            "losses": losses,
            "model": model,
            "drop_mask": dropped_masks[before_masks].clone(),
            "rng": torch.random.get_rng_state().clone(),
            "shared_grad": model.shared.grad.detach().clone(),
            "rank_grad": model.rank_probe.grad.detach().clone(),
        }

    full = run(True)
    without_mofe = run(False)
    assert len(rank_calls) == 2
    assert full["model"].gate_calls == without_mofe["model"].gate_calls == 2
    assert len(full["model"].forward_masks) == len(without_mofe["model"].forward_masks) == 2
    assert torch.equal(full["drop_mask"], without_mofe["drop_mask"])
    assert all(
        torch.equal(full_mask, no_mofe_mask)
        for full_mask, no_mofe_mask in zip(
            full["model"].forward_masks,
            without_mofe["model"].forward_masks,
        )
    )
    assert torch.equal(full["rng"], without_mofe["rng"])
    assert torch.equal(full["shared_grad"], without_mofe["shared_grad"])
    assert full["rank_grad"].abs().item() > 0
    assert without_mofe["rank_grad"].item() == 0.0
    assert full["losses"]["mofe"] > 0
    assert without_mofe["losses"]["mofe"] == 0.0
    for name in full["losses"].keys() - {"loss", "mofe"}:
        assert full["losses"][name] == without_mofe["losses"][name]


def _shuffle_order_after_model_init(ablation_id):
    torch.manual_seed(137)
    _model(canonical_control_profile(ablation_id))
    loader = DataLoader(
        torch.arange(37),
        batch_size=7,
        shuffle=True,
        generator=data_order_generator(211, stream=1),
    )
    return torch.cat(list(loader)).numpy()


def test_dedicated_dataloader_rng_decouples_order_from_model_initialization():
    sparse_order = _shuffle_order_after_model_init("full")
    dense_order = _shuffle_order_after_model_init("dense_backbone")
    assert np.array_equal(sparse_order, dense_order)
    digest = hashlib.sha256(sparse_order.astype("<i8").tobytes()).hexdigest()
    assert len(digest) == 64


def _public_loader_order(global_model_width):
    torch.manual_seed(138)
    torch.nn.Linear(global_model_width, global_model_width)
    count = 30
    data = {
        "feature": np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        "modality_comb": np.zeros(count, dtype=np.int64),
    }
    observed = np.ones((count, 1), dtype=bool)
    labels = np.arange(count, dtype=np.int64)
    _, shuffled, _, _ = create_loaders(
        data,
        observed,
        labels,
        np.arange(24),
        np.arange(24, 27),
        np.arange(27, 30),
        6,
        0,
        False,
        {"feature": 2},
        {},
        {},
        True,
        False,
        219,
    )
    return torch.cat([batch[1] for batch in shuffled]).numpy()


def test_public_loader_wires_the_dedicated_order_seed():
    assert np.array_equal(_public_loader_order(3), _public_loader_order(19))


class _IndexedDataset(Dataset):
    def __init__(self):
        self.label_new = np.asarray([0, 1, 1, 0, 1, 0, 1, 1] * 4)

    def __len__(self):
        return len(self.label_new)

    def __getitem__(self, index):
        return index


def _balanced_order(global_draws):
    torch.manual_seed(139)
    torch.rand(global_draws)
    base = DataLoader(_IndexedDataset(), batch_size=8, shuffle=False)
    balanced = balanced_train_loader(base, power=0.5, seed=223)
    return torch.cat(list(balanced)).numpy()


def test_weighted_sampler_rng_is_independent_of_global_model_rng():
    first = _balanced_order(1)
    second = _balanced_order(10_000)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, np.arange(first.size))
