from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cghc_release_builder", ROOT / "scripts" / "build_cghc_release.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def test_metric_map_uses_common_six() -> None:
    source = {name: 0.5 for name in BUILDER.METRICS}
    mapped = BUILDER.metric_map(source)
    assert tuple(mapped) == tuple(BUILDER.SNAKE[name] for name in BUILDER.METRICS)
    assert set(mapped.values()) == {0.5}


def test_holm_inference_record_is_concise() -> None:
    source = {
        "randomization": {
            "observed_macro_f1_delta": 0.1,
            "one_sided_p": 0.01,
            "draws": 50_000,
            "clusters": 10,
            "subjects": 12,
        },
        "bootstrap": {"lower_one_sided_95": 0.02, "draws": 20_000},
        "holm_adjusted_p": 0.03,
        "holm_significant_0.05": True,
        "bootstrap_lower_positive": True,
    }
    record = BUILDER.inference_record(source, "macro_f1")
    assert record["delta"] == 0.1
    assert record["holm_significant_0.05"] is True
    assert record["bootstrap_lower_positive"] is True


def test_svg_renderers_mark_noncausal_boundary() -> None:
    payload = {
        "metric_order": [BUILDER.SNAKE[name] for name in BUILDER.METRICS],
        "datasets": {},
    }
    for dataset in BUILDER.DATASETS:
        cohorts = (
            ("cognitively_normal", "mild_cognitive_impairment", "dementia")
            if dataset == "adni"
            else ("no_diagnosis", "one_domain", "multiple_domains")
        )
        rows = []
        for cohort in cohorts:
            for component in ("a", "b", "c", "d"):
                rows.append({"cohort": cohort, "component": component, "mean": 0.25})
        payload["datasets"][dataset] = {
            "candidate": {name: 0.7 for name in payload["metric_order"]},
            "baselines": {
                "base": {
                    "display_name": "Base",
                    "metrics": {name: 0.6 for name in payload["metric_order"]},
                }
            },
            "interpretability": {
                "decision_allocation_by_disease_stratum": rows,
                "disease_strata": [
                    {
                        "cohort": cohort,
                        "class_index": index,
                        "display_name": cohort.replace("_", " ").title(),
                    }
                    for index, cohort in enumerate(cohorts)
                ],
            },
        }
    assert "CGHC-v1 common-six" in BUILDER.render_common6_svg(payload)
    interpretation = BUILDER.render_interpretability_svg(payload)
    assert "non-causal" in interpretation
    assert "Disease-burden strata" in interpretation


def test_markdown_reports_boundary_and_common_six() -> None:
    payload = {
        "metric_order": [BUILDER.SNAKE[name] for name in BUILDER.METRICS],
        "datasets": {},
    }
    for dataset in BUILDER.DATASETS:
        metrics = {name: 0.7 for name in payload["metric_order"]}
        inference = {
            "delta": 0.05,
            "one_sided_p": 0.01,
            "holm_adjusted_p": 0.02,
            "bootstrap_lower_one_sided_95": 0.01,
            "holm_significant_0.05": True,
            "bootstrap_lower_positive": True,
        }
        payload["datasets"][dataset] = {
            "candidate": metrics,
            "baselines": {
                name: {
                    "display_name": BUILDER.DISPLAY[name],
                    "metrics": {key: 0.6 for key in metrics},
                    "macro_f1_inference": inference,
                    "accuracy_inference": inference,
                }
                for name in BUILDER.BASELINES
            },
            "component_ablations": {
                "full_minus_ablation": {
                    "no_completion": {
                        "metrics": {key: 0.01 for key in metrics}
                    }
                }
            },
        }
    markdown = BUILDER.render_markdown(payload)
    assert "retrospective fixed-candidate" in markdown
    assert "ADNI | CGHC-v1" in markdown
    assert "ABCD Accuracy and ADNI Macro-F1" in markdown
    assert "not disease causes" in markdown


def test_current_private_summary_uses_candidate_metrics_field() -> None:
    summary_path = (
        ROOT.parent
        / "MoE"
        / "conditional_generative_heterogeneous_consensus_v1"
        / "summary.json"
    )
    if not summary_path.is_file():
        return
    summary = BUILDER.load_json(summary_path)
    for dataset in BUILDER.DATASETS:
        result = summary["results"][dataset]
        assert "candidate_metrics" in result
        assert "candidate" not in result
        mapped = BUILDER.metric_map(result["candidate_metrics"])
        assert tuple(mapped) == tuple(BUILDER.SNAKE[name] for name in BUILDER.METRICS)


def test_current_private_receipt_is_complete() -> None:
    receipt_path = (
        ROOT.parent
        / "MoE"
        / "conditional_generative_heterogeneous_consensus_v1"
        / "FINAL_RECEIPT.json"
    )
    if not receipt_path.is_file():
        return
    receipt = BUILDER.load_json(receipt_path)
    assert receipt["status"] == "complete"


def test_build_payload_matches_current_private_structures(monkeypatch) -> None:
    source_root = ROOT.parent / "MoE"
    summary_path = (
        source_root
        / "conditional_generative_heterogeneous_consensus_v1"
        / "summary.json"
    )
    if not summary_path.is_file():
        return
    original_load = BUILDER.load_json

    def fake_interpretability(dataset: str) -> dict:
        components = (
            ("image", "genomic", "clinical", "biospecimen")
            if dataset == "adni"
            else ("imaging", "genetic", "cognition_health", "behavior_environment")
        )
        cohorts = (
            ("cognitively_normal", "mild_cognitive_impairment", "dementia")
            if dataset == "adni"
            else ("no_diagnosis", "one_domain", "multiple_domains")
        )
        records = []
        for cohort in cohorts:
            for component in components:
                records.append(
                    {
                        "dataset": dataset,
                        "section": "decision_allocation",
                        "cohort": cohort,
                        "component": component,
                        "mean": 0.25,
                        "ci95_lower": 0.2,
                        "ci95_upper": 0.3,
                        "n": 10,
                        "clusters": 10,
                    }
                )
        return {
            "status": "PASS",
            "scope": {"validation_only": True},
            "semantics": {"causal_claim": False},
            "disease_strata": [
                {
                    "cohort": cohort,
                    "class_index": index,
                    "display_name": cohort.replace("_", " ").title(),
                }
                for index, cohort in enumerate(cohorts)
            ],
            "disease_allocation_contrast": (
                "dementia_minus_cognitively_normal"
                if dataset == "adni"
                else "multiple_domains_minus_no_diagnosis"
            ),
            "aggregate_records": records,
            "disease_stratum_allocation_contrasts": [
                {
                    "dataset": dataset,
                    "contrast": "multiple_domains_minus_no_diagnosis",
                    "component": component,
                    "mean_difference": 0.0,
                    "causal_effect": False,
                }
                for component in components
            ],
            "replay_audits": [],
            "uncertainty": {"method": "percentile cluster bootstrap"},
        }

    def fake_load(path: Path) -> dict:
        if path.name == "component_interpretability.json":
            return fake_interpretability(path.parent.name)
        if path.name == "FINAL_RECEIPT.json" and "interpretability_v1" in path.parts:
            return {
                "status": "PASS",
                "artifacts": {"json": {"sha256": "0" * 64}},
            }
        return original_load(path)

    monkeypatch.setattr(BUILDER, "load_json", fake_load)
    monkeypatch.setattr(BUILDER, "sha256_file", lambda _path: "0" * 64)
    payload, paths = BUILDER.build_payload(source_root)
    assert payload["method"]["conditional_generation_required"] is True
    assert payload["method"]["moe_retained"] is True
    assert payload["method"]["final_consensus_is_moe"] is False
    assert set(payload["datasets"]) == {"adni", "abcd"}
    assert set(paths) == {
        "summary",
        "receipt",
        "accuracy",
        "ablation",
        "abcd_interpretability",
        "abcd_interpretability_receipt",
        "adni_interpretability",
        "adni_interpretability_receipt",
    }
    for dataset in BUILDER.DATASETS:
        result = payload["datasets"][dataset]
        assert result["candidate_higher_than_every_baseline_on_all_six"] is True
        assert set(result["baselines"]) == set(BUILDER.BASELINES)
        assert len(result["interpretability"]["decision_allocation_by_disease_stratum"]) == 12
