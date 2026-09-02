#!/usr/bin/env python3
"""Build the aggregate-only public CGHC-v1 result bundle from private receipts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "Accuracy",
    "Balanced Accuracy",
    "Macro-F1",
    "Weighted-F1",
    "Macro-AUROC",
    "Macro-AUPRC",
)
SNAKE = {
    "Accuracy": "accuracy",
    "Balanced Accuracy": "balanced_accuracy",
    "Macro-F1": "macro_f1",
    "Weighted-F1": "weighted_f1",
    "Macro-AUROC": "macro_auroc",
    "Macro-AUPRC": "macro_auprc",
}
DATASETS = ("adni", "abcd")
BASELINES = (
    "flex_moe",
    "i2moe",
    "moepp_corrected",
    "anymod",
    "agdic",
    "acadiff",
)
DISPLAY = {
    "flex_moe": "Flex-MoE",
    "i2moe": "I2MoE",
    "moepp_corrected": "MoE++ (corrected)",
    "anymod": "AnyMod",
    "agdic": "AGDiC",
    "acadiff": "ACADiff",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def atomic_create(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def metric_map(source: dict[str, Any], *, bounded: bool = True) -> dict[str, float]:
    require(set(source) == set(METRICS), "common-six metric schema changed")
    result = {SNAKE[name]: float(source[name]) for name in METRICS}
    if bounded:
        require(all(0.0 <= value <= 1.0 for value in result.values()), "metric out of range")
    return result


def inference_record(source: dict[str, Any], endpoint: str) -> dict[str, Any]:
    randomization = source["randomization"]
    bootstrap = source["bootstrap"]
    observed_key = f"observed_{endpoint}_delta"
    return {
        "delta": float(randomization[observed_key]),
        "one_sided_p": float(randomization["one_sided_p"]),
        "holm_adjusted_p": float(source["holm_adjusted_p"]),
        "holm_significant_0.05": bool(source["holm_significant_0.05"]),
        "bootstrap_lower_one_sided_95": float(bootstrap["lower_one_sided_95"]),
        "bootstrap_lower_positive": bool(source["bootstrap_lower_positive"]),
        "randomization_draws": int(randomization["draws"]),
        "bootstrap_draws": int(bootstrap["draws"]),
        "clusters": int(randomization["clusters"]),
        "subjects": int(randomization["subjects"]),
    }


def interpretation_summary(source: dict[str, Any]) -> dict[str, Any]:
    require(source.get("status") == "PASS", "interpretability replay did not pass")
    require(source["semantics"]["causal_claim"] is False, "causal claim must be false")
    strata = source["disease_strata"]
    require(
        isinstance(strata, list)
        and len(strata) == 3
        and [row["class_index"] for row in strata] == [0, 1, 2],
        "disease-stratum schema changed",
    )
    cohorts = {row["cohort"] for row in strata}
    allocation = [
        row
        for row in source["aggregate_records"]
        if row["section"] == "decision_allocation"
        and row["cohort"] in cohorts
    ]
    require(len(allocation) == 12, "expected three disease strata by four modalities")
    return {
        "scope": source["scope"],
        "semantics": source["semantics"],
        "disease_strata": strata,
        "disease_allocation_contrast": source["disease_allocation_contrast"],
        "decision_allocation_by_disease_stratum": allocation,
        "multiple_domains_minus_no_diagnosis": source[
            "disease_stratum_allocation_contrasts"
        ],
        "replay_audits": source["replay_audits"],
        "uncertainty": source["uncertainty"],
    }


def build_payload(source_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {
        "summary": source_root
        / "conditional_generative_heterogeneous_consensus_v1"
        / "summary.json",
        "receipt": source_root
        / "conditional_generative_heterogeneous_consensus_v1"
        / "FINAL_RECEIPT.json",
        "accuracy": source_root
        / "conditional_generative_heterogeneous_consensus_v1"
        / "secondary_accuracy_significance.json",
        "ablation": source_root
        / "formal_matched_ablation_common6_v3"
        / "analysis_v1"
        / "summary.json",
        "abcd_interpretability": source_root
        / "conditional_generative_heterogeneous_consensus_v1"
        / "interpretability_v1"
        / "abcd"
        / "component_interpretability.json",
        "abcd_interpretability_receipt": source_root
        / "conditional_generative_heterogeneous_consensus_v1"
        / "interpretability_v1"
        / "abcd"
        / "FINAL_RECEIPT.json",
        "adni_interpretability": source_root
        / "conditional_generative_heterogeneous_consensus_v1"
        / "interpretability_v1"
        / "adni"
        / "component_interpretability.json",
        "adni_interpretability_receipt": source_root
        / "conditional_generative_heterogeneous_consensus_v1"
        / "interpretability_v1"
        / "adni"
        / "FINAL_RECEIPT.json",
    }
    loaded = {name: load_json(path) for name, path in paths.items()}
    summary = loaded["summary"]
    receipt = loaded["receipt"]
    accuracy = loaded["accuracy"]
    ablation = loaded["ablation"]
    require(summary.get("method") == "CGHC-v1", "unexpected method")
    require(
        summary["claim_boundary"]["retrospective_fixed_candidate_paired_evidence"] is True,
        "retrospective boundary missing",
    )
    require(receipt.get("status") == "complete", "candidate receipt is incomplete")
    require(accuracy.get("candidate_changed") is False, "accuracy analysis changed candidate")
    for dataset in DATASETS:
        replay_receipt = loaded[f"{dataset}_interpretability_receipt"]
        require(replay_receipt.get("status") == "PASS", "interpretability receipt did not pass")
        require(
            replay_receipt["artifacts"]["json"]["sha256"]
            == sha256_file(paths[f"{dataset}_interpretability"]),
            "interpretability JSON does not match its receipt",
        )
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        result = summary["results"][dataset]
        require(result["candidate_higher_than_every_baseline_on_all_six"] is True, "all-six gate failed")
        baselines: dict[str, Any] = {}
        for baseline in BASELINES:
            row = result["baselines"][baseline]
            require(row["candidate_strictly_higher_all_six"] is True, "baseline all-six gate failed")
            baselines[baseline] = {
                "display_name": DISPLAY[baseline],
                "metrics": metric_map(row["metrics"]),
                "candidate_minus_baseline": metric_map(row["delta_candidate_minus_baseline"]),
                "macro_f1_inference": inference_record(
                    summary["inference"][dataset][baseline], "macro_f1"
                ),
                "accuracy_inference": inference_record(
                    accuracy["results"][dataset][baseline], "accuracy"
                ),
            }
        ablation_result = ablation["results"][dataset]
        arms = {
            arm: metric_map(record["ensemble_metrics"])
            for arm, record in ablation_result["arms"].items()
        }
        deltas = {
            arm: {
                "metrics": metric_map(record["delta_full_minus_arm"], bounded=False),
                "full_strictly_higher_all_six": bool(
                    record["full_strictly_higher_all_six"]
                ),
            }
            for arm, record in ablation_result["full_minus_ablation"].items()
        }
        datasets[dataset] = {
            "display_name": dataset.upper(),
            "subjects": int(result["participants"]),
            "clusters": int(result["clusters"]),
            "candidate": metric_map(result["candidate_metrics"]),
            "candidate_higher_than_every_baseline_on_all_six": True,
            "baselines": baselines,
            "component_ablation_boundary": (
                "separate five-seed, five-fold development-OOF campaign; describes "
                "the conditional-generative component and is not a matched ablation "
                "of the final heterogeneous consensus"
            ),
            "component_ablations": {
                "participants": int(ablation_result["participants"]),
                "seeds": list(ablation_result["seeds"]),
                "arms": arms,
                "full_minus_ablation": deltas,
            },
            "interpretability": interpretation_summary(
                loaded[f"{dataset}_interpretability"]
            ),
        }
    payload = {
        "schema": "cghc-public-results-v1",
        "status": "retrospective_fixed_candidate_paired_evidence",
        "claim_boundary": {
            "pristine_new_confirmatory_test": False,
            "retrospective_fixed_candidate_paired_evidence": True,
            "test_targets_used_to_fit_or_select_decision_bias": False,
            "causal_modality_or_disease_etiology_claim": False,
        },
        "method": {
            "name": "CGHC-v1",
            "expanded_name": "Conditional-Generative Heterogeneous Consensus",
            "conditional_generation_required": True,
            "moe_retained": True,
            "final_consensus_is_moe": False,
            "abcd_members": "three Rank-MoE, three CatBoost, and three TabM-32 models",
            "adni_members": "four conditional-generative MoE variants with three seeds each",
        },
        "metric_order": [SNAKE[name] for name in METRICS],
        "metric_unit": "fraction",
        "datasets": datasets,
        "source_receipts": {
            name: {"sha256": sha256_file(path)} for name, path in paths.items()
        },
    }
    return payload, paths


def render_common6_svg(payload: dict[str, Any]) -> str:
    width, height = 1120, 560
    colors = {"adni": "#31688e", "abcd": "#35b779"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="560" y="32" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="700">CGHC-v1 common-six results</text>',
        '<text x="560" y="54" text-anchor="middle" font-family="sans-serif" font-size="13">Candidate and strongest baseline per metric; retrospective fixed-candidate evidence</text>',
    ]
    panel_width = 520
    for panel, dataset in enumerate(DATASETS):
        x0 = 45 + panel * 555
        candidate = payload["datasets"][dataset]["candidate"]
        baselines = payload["datasets"][dataset]["baselines"]
        lines.append(
            f'<text x="{x0 + panel_width / 2:.1f}" y="86" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">{dataset.upper()}</text>'
        )
        for index, metric in enumerate(payload["metric_order"]):
            y = 116 + index * 66
            strongest_name, strongest = max(
                ((row["display_name"], row["metrics"][metric]) for row in baselines.values()),
                key=lambda item: item[1],
            )
            cand = candidate[metric]
            scale = 430.0
            label = metric.replace("_", " ").title()
            lines.extend(
                [
                    f'<text x="{x0}" y="{y}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
                    f'<rect x="{x0}" y="{y + 8}" width="{strongest * scale:.2f}" height="14" fill="#b9c2cc"/>',
                    f'<rect x="{x0}" y="{y + 26}" width="{cand * scale:.2f}" height="14" fill="{colors[dataset]}"/>',
                    f'<text x="{x0 + strongest * scale + 5:.2f}" y="{y + 20}" font-family="sans-serif" font-size="10">{strongest * 100:.2f} {html.escape(strongest_name)}</text>',
                    f'<text x="{x0 + cand * scale + 5:.2f}" y="{y + 38}" font-family="sans-serif" font-size="10" font-weight="700">{cand * 100:.2f} CGHC</text>',
                ]
            )
    lines.append(
        '<text x="560" y="545" text-anchor="middle" font-family="sans-serif" font-size="11">Values are percentages in labels; bar length uses fractional metric values.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_interpretability_svg(payload: dict[str, Any]) -> str:
    width, height = 1120, 520
    colors = ("#3b528b", "#21918c", "#5ec962")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="560" y="30" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="700">Conditional-generative component decision allocation</text>',
        '<text x="560" y="52" text-anchor="middle" font-family="sans-serif" font-size="13">Disease-burden strata; cluster-bootstrap aggregates; descriptive and non-causal</text>',
    ]
    for panel, dataset in enumerate(DATASETS):
        x0 = 55 + panel * 555
        rows = payload["datasets"][dataset]["interpretability"][
            "decision_allocation_by_disease_stratum"
        ]
        strata = payload["datasets"][dataset]["interpretability"]["disease_strata"]
        modalities = sorted({row["component"] for row in rows})
        lines.append(
            f'<text x="{x0 + 235}" y="84" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">{dataset.upper()}</text>'
        )
        for modality_index, modality in enumerate(modalities):
            base_x = x0 + modality_index * 112
            lines.append(
                f'<text x="{base_x + 46}" y="454" text-anchor="middle" font-family="sans-serif" font-size="10">{html.escape(modality.replace("_", " ").title())}</text>'
            )
            for cohort_index, stratum in enumerate(strata):
                cohort = stratum["cohort"]
                row = next(
                    item
                    for item in rows
                    if item["component"] == modality and item["cohort"] == cohort
                )
                bar_height = float(row["mean"]) * 310.0
                x = base_x + cohort_index * 28
                y = 425 - bar_height
                lines.append(
                    f'<rect x="{x}" y="{y:.2f}" width="22" height="{bar_height:.2f}" fill="{colors[cohort_index]}"/>'
                )
        lines.append(f'<line x1="{x0}" y1="425" x2="{x0 + 440}" y2="425" stroke="#333"/>')
    for panel, dataset in enumerate(DATASETS):
        strata = payload["datasets"][dataset]["interpretability"]["disease_strata"]
        x0 = 70 + panel * 555
        for index, stratum in enumerate(strata):
            x = x0 + index * 155
            lines.append(f'<rect x="{x}" y="480" width="14" height="14" fill="{colors[index]}"/>')
            lines.append(
                f'<text x="{x + 20}" y="492" font-family="sans-serif" font-size="10">{html.escape(stratum["display_name"])}</text>'
            )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CGHC-v1 aggregate results",
        "",
        "> **Evidence boundary:** retrospective fixed-candidate paired evidence. "
        "This is not a pristine new confirmatory test, and no modality result is "
        "a causal or disease-etiology estimate.",
        "",
        "CGHC-v1 retains conditional generation and sparse-MoE neural members, "
        "while its final cross-model hierarchical median consensus is not an MoE.",
        "",
        "## Common-six results",
        "",
        "Values are percentages. Every CGHC-v1 value is strictly higher than every "
        "listed three-seed baseline ensemble on the same metric and dataset.",
        "",
        "| Dataset | Method | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        result = payload["datasets"][dataset]
        ordered = [("CGHC-v1", result["candidate"])] + [
            (result["baselines"][name]["display_name"], result["baselines"][name]["metrics"])
            for name in BASELINES
        ]
        for method, metrics in ordered:
            values = " | ".join(percent(metrics[name]) for name in payload["metric_order"])
            lines.append(f"| {dataset.upper()} | {method} | {values} |")
    lines.extend(
        [
            "",
            "- [Common-six performance figure](../figures/cghc_common6.svg)",
            "",
            "## Paired evidence against Flex-MoE",
            "",
            "Holm adjustment is across all six comparators within each dataset and "
            "endpoint. Positive lower bounds are one-sided 95% cluster-bootstrap bounds.",
            "",
            "| Dataset | Endpoint | Difference (pp) | Raw p | Holm p | Lower bound (pp) | Holm significant and lower positive |",
            "|---|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for dataset in DATASETS:
        flex = payload["datasets"][dataset]["baselines"]["flex_moe"]
        for label, key in (("Macro-F1", "macro_f1_inference"), ("Accuracy", "accuracy_inference")):
            row = flex[key]
            supported = row["holm_significant_0.05"] and row["bootstrap_lower_positive"]
            lines.append(
                f"| {dataset.upper()} | {label} | {row['delta'] * 100:+.2f} | "
                f"{row['one_sided_p']:.5f} | {row['holm_adjusted_p']:.5f} | "
                f"{row['bootstrap_lower_one_sided_95'] * 100:+.2f} | "
                f"{'yes' if supported else 'no'} |"
            )
    lines.extend(
        [
            "",
            "The supported dataset-specific statements are ABCD Accuracy and ADNI "
            "Macro-F1. Other rows remain descriptive even when their point difference "
            "is positive.",
            "",
            "## Conditional-generation component controls",
            "",
            "These controls come from a separate exact-500 five-seed, five-fold "
            "development-OOF campaign. They describe the conditional-generative "
            "component and are not matched ablations of the final heterogeneous consensus.",
            "",
            "| Dataset | Full minus no completion: Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        delta = payload["datasets"][dataset]["component_ablations"][
            "full_minus_ablation"
        ]["no_completion"]["metrics"]
        values = " | ".join(f"{delta[name] * 100:+.2f}" for name in payload["metric_order"])
        lines.append(f"| {dataset.upper()} | {values} |")
    lines.extend(
        [
            "",
            "Completion improves ABCD hard classification and five of six metrics "
            "in this matched campaign; its ABCD Macro-AUPRC difference is essentially "
            "zero. On ADNI it improves the hard classification metrics, including "
            "Macro-F1, but not both ranking metrics. The evidence therefore supports "
            "a task-dependent benefit rather than an unconditional all-metric claim.",
            "",
            "## Modality–disease association",
            "",
            "The public interpretation replays frozen validation checkpoints and "
            "reports participant-level member-averaged decision allocation with "
            "cluster-bootstrap intervals. ADNI compares dementia with cognitively "
            "normal; ABCD compares multiple diagnosis domains with no diagnosis. "
            "These are fitted-model associations, not disease causes or causal effects.",
            "",
            "- [Modality allocation by disease stratum](../figures/cghc_modality_association.svg)",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cghc_v1.json")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, _paths = build_payload(args.source_root.resolve(strict=True))
    output = args.output.resolve()
    figure_dir = args.figure_dir.resolve()
    common6 = figure_dir / "cghc_common6.svg"
    interpretation = figure_dir / "cghc_modality_association.svg"
    markdown = output.with_suffix(".md")
    receipt = output.with_name("cghc_v1_receipt.json")
    for path in (output, markdown, common6, interpretation, receipt):
        require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    atomic_create(output, canonical_json(payload))
    atomic_create(markdown, render_markdown(payload))
    atomic_create(common6, render_common6_svg(payload))
    atomic_create(interpretation, render_interpretability_svg(payload))
    receipt_payload = {
        "schema": "cghc-public-release-receipt-v1",
        "status": "PASS",
        "artifacts": {
            path.name: sha256_file(path)
            for path in (output, markdown, common6, interpretation)
        },
        "builder_sha256": sha256_file(Path(__file__)),
    }
    atomic_create(receipt, canonical_json(receipt_payload))
    print(canonical_json(receipt_payload), end="")


if __name__ == "__main__":
    main()
