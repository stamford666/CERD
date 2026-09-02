import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_release_results.py"
SPEC = importlib.util.spec_from_file_location("render_release_results", SCRIPT)
RENDERER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RENDERER)

EXPECTED_ABLATIONS = [
    ("dense_backbone", "Dense FFN instead of sparse MoE"),
    ("no_provenance", "Without observed/generated provenance"),
    ("uniform_branch_weights", "Uniform instead of reliability-aware branch weights"),
    ("mean_pooling", "Mean instead of gated-attention pooling"),
    (
        "no_stochastic_context",
        "Without stochastic observed-subset context masking",
    ),
    ("no_completion", "Without latent completion"),
    ("no_mofe", "Without more/fewer-modality objective"),
    ("no_output_gate", "Without generator output gate"),
]


def _metrics(value):
    return {
        "accuracy": value + 0.02,
        "balanced_accuracy": value + 0.01,
        "macro_f1": value,
        "weighted_f1": value + 0.03,
        "macro_auroc": value + 0.12,
        "macro_auprc": value - 0.01,
    }


def _dataset_payload(dataset_key):
    is_adni = dataset_key == "adni"
    contract = RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]
    subjects = contract["subjects"]
    complete_subjects = subjects // 2
    comparator = "Synthetic ADNI comparator" if is_adni else "Synthetic ABCD comparator"
    p_value = 0.01 if is_adni else 0.03
    adjusted_p = 0.02 if is_adni else 0.03
    modalities = list(contract["modality_names"])
    ours_metrics = _metrics(0.58)
    comparator_metrics = _metrics(0.56)
    return {
        "display_name": contract["display_name"],
        "task": contract["task"],
        "evaluation": {
            "design": contract["design"],
            "subjects": subjects,
            "folds": contract["folds"],
            "split": contract["split"],
            "evidence_scope": contract["evidence_scope"],
            "partition_reused": contract["partition_reused"],
            "selection_independent": contract["selection_independent"],
        },
        "primary_results": [
            {
                "method": "CERD (ours)",
                "role": "ours",
                "configuration_id": f"synthetic {dataset_key} cerd configuration",
                "configuration_sha256": "a" * 64,
                "execution_receipt_sha256": "b" * 64,
                "metrics": ours_metrics,
            },
            {
                "method": comparator,
                "role": "comparator",
                "configuration_id": f"synthetic {dataset_key} comparator configuration",
                "configuration_sha256": "c" * 64,
                "execution_receipt_sha256": "d" * 64,
                "metrics": comparator_metrics,
            },
        ],
        "statistical_comparisons": [
            {
                "ours": "CERD (ours)",
                "comparator": comparator,
                "metric": "macro_f1",
                "paired": True,
                "delta": 0.02,
                "bootstrap_lower_bound": 0.01,
                "bootstrap_confidence_level": 0.95,
                "p_value": p_value,
                "adjusted_p_value": adjusted_p,
                "alpha": 0.05,
                "test": "paired swap test",
                "alternative": "greater",
                "resampling_unit": "subject" if is_adni else "family",
                "n_units": contract["resampling_units"],
                "swap_draws": 50_000,
                "swap_rng_seed": 20_260_905,
                "bootstrap_draws": 20_000,
                "bootstrap_rng_seed": 20_260_906,
                "multiplicity_adjustment": (
                    "Holm across ADNI and ABCD primary Macro-F1 comparisons"
                ),
                "analysis_receipt_sha256": "e" * 64,
                "confirmatory_support": False,
            }
        ],
        "ablations": [
            {
                "id": ablation_id,
                "label": label,
                "configuration_id": (
                    f"synthetic {dataset_key} {ablation_id.replace('_', '-')} configuration"
                ),
                "configuration_sha256": f"{index + 16:064x}",
                "execution_receipt_sha256": f"{index + 32:064x}",
                "metrics": copy.deepcopy(comparator_metrics),
            }
            for index, (ablation_id, label) in enumerate(EXPECTED_ABLATIONS)
        ],
        "interpretability": {
            "source": "aggregate out-of-fold checkpoint replay",
            "condition_design": "natural_disjoint",
            "aggregation_unit": RENDERER.INTERPRETABILITY_AGGREGATION_UNIT,
            "aggregation_method": RENDERER.INTERPRETABILITY_AGGREGATION_METHOD,
            "aggregation_receipt_sha256": "f" * 64,
            "modality_names": modalities,
            "conditions": {
                "complete": {
                    "definition": RENDERER.INTERPRETABILITY_DEFINITIONS["complete"],
                    "subjects": complete_subjects,
                    "decision_allocation": [0.1, 0.2, 0.3, 0.4],
                    "branch_mass": {
                        "joint": 0.2,
                        "unimodal": 0.3,
                        "pairwise": 0.5,
                    },
                },
                "incomplete": {
                    "definition": RENDERER.INTERPRETABILITY_DEFINITIONS["incomplete"],
                    "subjects": subjects - complete_subjects,
                    "decision_allocation": [0.2, 0.2, 0.3, 0.3],
                    "branch_mass": {
                        "joint": 0.3,
                        "unimodal": 0.3,
                        "pairwise": 0.4,
                    },
                },
            },
        },
    }


def synthetic_final_payload():
    return {
        "schema": "cerd-final-public-results-v3",
        "status": "final",
        "generated_at": "2026-09-01T12:00:00Z",
        "source_manifest_sha256": "9" * 64,
        "metric_unit": "fraction",
        "metric_order": list(RENDERER.METRICS),
        "claim_boundary": RENDERER.ADAPTIVE_DEVELOPMENT_CLAIM,
        "datasets": {
            "adni": _dataset_payload("adni"),
            "abcd": _dataset_payload("abcd"),
        },
    }


def test_complete_aggregate_renders_only_the_public_contract():
    payload = synthetic_final_payload()
    RENDERER.validate_payload(payload)
    markdown = RENDERER.render_markdown(payload)
    assert "Release status: FINAL" in markdown
    assert "Macro-AUPRC" in markdown
    assert "Synthetic ADNI comparator" in markdown
    assert "Holm adjusted p" in markdown
    assert "50000 / 20260905" in markdown
    assert "20000 / 20260906" in markdown
    assert "Pre-specified matched ablations" in markdown
    assert "descriptive ablation effects" in markdown
    assert "natural_disjoint" in markdown
    assert "Confirmatory support" in markdown
    assert "ABCD protected temporal internal holdout 850" in markdown
    assert "NOT AVAILABLE" not in markdown
    assert "configuration_sha256" not in markdown
    assert "synthetic adni cerd configuration" not in markdown
    assert (
        "| Dataset | Method | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | "
        "Macro-AUROC | Macro-AUPRC |"
    ) in markdown
    assert "<svg" in RENDERER.render_common6_svg(payload)
    ablation_svg = RENDERER.render_ablation_svg(payload)
    assert "Pre-specified matched ablations" in ablation_svg
    assert "-2.00" in ablation_svg
    assert "not causal importance" in RENDERER.render_decision_allocation_svg(payload)


def test_final_rejects_non_common_metric():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["primary_results"][0]["metrics"]["mcc"] = 0.5
    with pytest.raises(ValueError, match="keys must be exactly"):
        RENDERER.validate_payload(payload)


def test_final_requires_exactly_two_ordered_primary_rows():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["primary_results"].append(
        copy.deepcopy(payload["datasets"]["adni"]["primary_results"][1])
    )
    payload["datasets"]["adni"]["primary_results"][2]["method"] = "Extra comparator"
    with pytest.raises(ValueError, match="exactly two rows"):
        RENDERER.validate_payload(payload)

    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["primary_results"].reverse()
    with pytest.raises(ValueError, match="ordered exactly"):
        RENDERER.validate_payload(payload)


def test_final_requires_exactly_one_primary_macro_f1_comparison():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["statistical_comparisons"].append(
        copy.deepcopy(payload["datasets"]["adni"]["statistical_comparisons"][0])
    )
    with pytest.raises(ValueError, match="exactly one paired"):
        RENDERER.validate_payload(payload)

    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["statistical_comparisons"][0]["metric"] = "accuracy"
    with pytest.raises(ValueError, match="primary metric 'macro_f1'"):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize(
    ("dataset_key", "field", "value", "message"),
    [
        ("adni", "paired", False, "must be true"),
        ("adni", "resampling_unit", "family", "must be 'subject'"),
        ("abcd", "resampling_unit", "subject", "must be 'family'"),
        ("adni", "swap_draws", 49_999, "must equal 50000"),
        ("adni", "swap_rng_seed", 1, "must equal 20260905"),
        ("adni", "bootstrap_draws", 19_999, "must equal 20000"),
        ("adni", "bootstrap_rng_seed", 1, "must equal 20260906"),
        ("adni", "alternative", "two-sided", "must be 'greater'"),
        ("adni", "alpha", 0.10, "must equal 0.05"),
        ("adni", "bootstrap_confidence_level", 0.90, "must equal 0.95"),
        ("adni", "n_units", 1479, "frozen ADNI subject count 1480"),
        ("abcd", "n_units", 921, "frozen ABCD family count 922"),
    ],
)
def test_final_rejects_unfrozen_statistical_protocol(
    dataset_key, field, value, message
):
    payload = synthetic_final_payload()
    payload["datasets"][dataset_key]["statistical_comparisons"][0][field] = value
    with pytest.raises(ValueError, match=message):
        RENDERER.validate_payload(payload)


def test_final_recomputes_delta_from_public_macro_f1():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["statistical_comparisons"][0]["delta"] = 0.03
    with pytest.raises(ValueError, match="must equal ours macro_f1"):
        RENDERER.validate_payload(payload)


def test_percentile_bootstrap_bound_is_not_forced_below_point_estimate():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["statistical_comparisons"][0][
        "bootstrap_lower_bound"
    ] = 0.025
    RENDERER.validate_payload(payload)


def test_final_recomputes_holm_across_both_dataset_comparisons():
    payload = synthetic_final_payload()
    payload["datasets"]["abcd"]["statistical_comparisons"][0][
        "adjusted_p_value"
    ] = 0.04
    with pytest.raises(ValueError, match="jointly recomputed two-comparison Holm"):
        RENDERER.validate_payload(payload)


def test_significance_is_derived_not_supplied():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["statistical_comparisons"][0]["significant"] = True
    with pytest.raises(ValueError, match="keys must be exactly"):
        RENDERER.validate_payload(payload)


def test_reused_development_forces_confirmatory_support_false():
    payload = synthetic_final_payload()
    comparison = payload["datasets"]["adni"]["statistical_comparisons"][0]
    comparison["confirmatory_support"] = True
    with pytest.raises(ValueError, match="reused development must be false"):
        RENDERER.validate_payload(payload)
    comparison["confirmatory_support"] = False
    RENDERER.validate_payload(payload)


@pytest.mark.parametrize(
    ("dataset_key", "field", "value"),
    [
        ("adni", "subjects", 1479),
        ("adni", "folds", 4),
        ("adni", "evidence_scope", "locked_evaluation"),
        ("adni", "partition_reused", False),
        ("adni", "selection_independent", True),
        ("abcd", "subjects", 945),
        ("abcd", "folds", 4),
    ],
)
def test_final_locks_reused_development_evaluation_metadata(
    dataset_key, field, value
):
    payload = synthetic_final_payload()
    payload["datasets"][dataset_key]["evaluation"][field] = value
    with pytest.raises(ValueError, match="reused"):
        RENDERER.validate_payload(payload)


def test_final_requires_frozen_adaptive_claim_wording():
    payload = synthetic_final_payload()
    payload["claim_boundary"] = "Descriptive development result."
    with pytest.raises(ValueError, match="frozen adaptive reused-development"):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize(
    ("dataset_key", "field", "value"),
    [
        ("adni", "display_name", "ADNI cohort"),
        ("adni", "task", "binary diagnosis classification"),
        ("abcd", "display_name", "ABCD dev946"),
        ("abcd", "task", "binary ADHD classification"),
    ],
)
def test_final_locks_dataset_identity_and_three_class_tasks(
    dataset_key, field, value
):
    payload = synthetic_final_payload()
    payload["datasets"][dataset_key][field] = value
    with pytest.raises(ValueError, match="frozen public"):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("configuration_sha256", "A" * 64),
        ("execution_receipt_sha256", "abc"),
    ],
)
def test_final_requires_primary_configuration_and_receipt_digests(field, value):
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["primary_results"][0][field] = value
    with pytest.raises(ValueError, match="lowercase 64-character SHA-256"):
        RENDERER.validate_payload(payload)


def test_primary_configuration_id_cannot_publish_a_receipt_path():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["primary_results"][0][
        "configuration_id"
    ] = "/private/configuration.json"
    with pytest.raises(ValueError, match="absolute local path"):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize("field", ["configuration_id", "configuration_sha256"])
def test_final_distinguishes_ours_and_comparator_configurations(field):
    payload = synthetic_final_payload()
    rows = payload["datasets"]["adni"]["primary_results"]
    rows[1][field] = rows[0][field]
    with pytest.raises(ValueError, match="must distinguish ours from comparator"):
        RENDERER.validate_payload(payload)


def test_final_requires_paired_analysis_receipt_digest():
    payload = synthetic_final_payload()
    payload["datasets"]["abcd"]["statistical_comparisons"][0][
        "analysis_receipt_sha256"
    ] = "bad"
    with pytest.raises(ValueError, match="lowercase 64-character SHA-256"):
        RENDERER.validate_payload(payload)


def test_final_requires_exact_ordered_ablation_ids_and_labels():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["ablations"].reverse()
    with pytest.raises(ValueError, match="exact ordered"):
        RENDERER.validate_payload(payload)

    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["ablations"][0]["label"] = "Dense control"
    with pytest.raises(ValueError, match="canonical labels"):
        RENDERER.validate_payload(payload)


def test_natural_disjoint_counts_match_evaluation_and_protect_small_cells():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["interpretability"]["conditions"]["complete"][
        "subjects"
    ] = 11
    with pytest.raises(ValueError, match="must sum to evaluation.subjects"):
        RENDERER.validate_payload(payload)

    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["interpretability"]["conditions"]["complete"][
        "subjects"
    ] = 9
    with pytest.raises(ValueError, match="at least 10"):
        RENDERER.validate_payload(payload)


def test_interpretability_uses_controlled_design_and_public_source():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["interpretability"][
        "condition_design"
    ] = "paired_conditions"
    with pytest.raises(ValueError, match="condition_design"):
        RENDERER.validate_payload(payload)

    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["interpretability"]["source"] = "internal campaign replay"
    with pytest.raises(ValueError, match="must be one of"):
        RENDERER.validate_payload(payload)

    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["interpretability"][
        "source"
    ] = "aggregate locked-evaluation checkpoint replay"
    with pytest.raises(ValueError, match="must match the declared evaluation"):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aggregation_unit", "family"),
        ("aggregation_method", "weighted row mean"),
    ],
)
def test_interpretability_locks_subject_level_average_semantics(field, value):
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["interpretability"][field] = value
    with pytest.raises(ValueError, match="subject-level aggregation"):
        RENDERER.validate_payload(payload)


def test_interpretability_locks_natural_condition_definitions():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["interpretability"]["conditions"]["complete"][
        "definition"
    ] = "complete after synthetic masking"
    with pytest.raises(ValueError, match="natural complete/incomplete"):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize("dataset_key", ["adni", "abcd"])
def test_interpretability_locks_dataset_specific_modality_names(dataset_key):
    payload = synthetic_final_payload()
    modalities = payload["datasets"][dataset_key]["interpretability"][
        "modality_names"
    ]
    modalities[0], modalities[1] = modalities[1], modalities[0]
    with pytest.raises(ValueError, match="dataset-specific modality names"):
        RENDERER.validate_payload(payload)


def test_final_rejects_probability_rounding_drift():
    payload = synthetic_final_payload()
    payload["datasets"]["abcd"]["interpretability"]["conditions"]["complete"][
        "decision_allocation"
    ] = [0.1, 0.2, 0.3, 0.399]
    with pytest.raises(ValueError, match="1e-06"):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("2026-09-01 12:00:00+00:00", "RFC3339"),
        ("not a timestamp", "RFC3339"),
    ],
)
def test_final_requires_timezone_aware_rfc3339(value, message):
    payload = synthetic_final_payload()
    payload["generated_at"] = value
    with pytest.raises(ValueError, match=message):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize("token", ["TBD", "TODO", "placeholder", "pending"])
def test_final_rejects_placeholder_tokens(token):
    payload = synthetic_final_payload()
    payload["claim_boundary"] = f"Aggregate is {token} publication review."
    with pytest.raises(ValueError, match="placeholder text"):
        RENDERER.validate_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "/protected/example.bin", "absolute local path"),
        ("split", "runs/private/example.bin", "relative internal path"),
        ("design", "paired | broken table", "Markdown control"),
        ("design", "replay for participant-123", "participant-like identifier"),
    ],
)
def test_final_rejects_private_paths_and_markdown_control(field, value, message):
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["evaluation"][field] = value
    with pytest.raises(ValueError, match=message):
        RENDERER.validate_payload(payload)


def test_schema_rejects_participant_level_fields():
    payload = synthetic_final_payload()
    payload["datasets"]["adni"]["participant_ids"] = ["participant-1"]
    with pytest.raises(ValueError, match="keys must be exactly"):
        RENDERER.validate_payload(payload)


def test_readme_marker_replacement_is_bounded():
    original = "before\n<!-- FINAL_RESULTS_START -->old<!-- FINAL_RESULTS_END -->\nafter\n"
    block = "<!-- FINAL_RESULTS_START -->new<!-- FINAL_RESULTS_END -->"
    assert RENDERER.replace_generated_block(original, block) == "before\n" + block + "\nafter\n"


def test_public_results_tree_has_only_allowlisted_aggregate_files():
    allowed = {
        "results/README.md",
        "results/final_results.json",
        "figures/README.md",
        "figures/common6.svg",
        "figures/ablations.svg",
        "figures/decision_allocation.svg",
    }
    local_files = {
        path.relative_to(ROOT).as_posix()
        for directory in (ROOT / "results", ROOT / "figures")
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert local_files <= allowed
    for relative_path in local_files:
        assert not (ROOT / relative_path).is_symlink()

    tracked = subprocess.run(
        ["git", "ls-files", "results", "figures"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_existing = {
        name for name in tracked if (ROOT / name).exists() or (ROOT / name).is_symlink()
    }
    assert tracked_existing <= allowed


def test_checked_in_final_artifact_is_aggregate_only_if_present():
    final_path = ROOT / "results" / "final_results.json"
    if not final_path.exists():
        return
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    RENDERER.validate_payload(payload)
    assert payload["status"] == "final"
    serialized = final_path.read_text(encoding="utf-8")
    assert RENDERER.PLACEHOLDER_TEXT.search(serialized) is None


def test_renderer_rejects_duplicate_json_keys(tmp_path):
    payload = json.dumps(synthetic_final_payload(), sort_keys=True)
    duplicate = '{"hidden_private_rows":[1],"hidden_private_rows":null,"payload":' + payload + "}"
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="unique-key UTF-8 JSON"):
        RENDERER.load_unique_json(path)


def test_release_workflow_never_skips_missing_final_artifact():
    workflow = (ROOT / ".github/workflows/release-results.yml").read_text(
        encoding="utf-8"
    )
    gate = workflow.split(
        "- name: Require approved, complete, synchronized public results", 1
    )[1]
    assert "hashFiles('results/final_results.json')" not in gate
    assert "--check --require-final" in gate
