#!/usr/bin/env python3
"""Fail-closed requirement audit for the requested CGHC-v1 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


METRICS = (
    "Accuracy",
    "Balanced Accuracy",
    "Macro-F1",
    "Weighted-F1",
    "Macro-AUROC",
    "Macro-AUPRC",
)
DATASETS = ("adni", "abcd")
BASELINES = (
    "flex_moe",
    "i2moe",
    "moepp_corrected",
    "anymod",
    "agdic",
    "acadiff",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("r", encoding="utf-8", errors="strict") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def requirement(
    records: list[dict[str, Any]],
    identifier: str,
    description: str,
    check: Callable[[], dict[str, Any] | None],
) -> None:
    try:
        evidence = check() or {}
        records.append(
            {
                "id": identifier,
                "description": description,
                "status": "PASS",
                "evidence": evidence,
            }
        )
    except Exception as error:
        records.append(
            {
                "id": identifier,
                "description": description,
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
            }
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(source_root: Path, release_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    release_root = release_root.resolve(strict=True)
    cghc = source_root / "conditional_generative_heterogeneous_consensus_v1"
    records: list[dict[str, Any]] = []

    def method_boundary() -> dict[str, Any]:
        freeze = load_json(cghc / "freeze.json")
        candidate = freeze["candidate"]
        require(candidate["conditional_generation_required"] is True, "conditional generation is not required")
        require(candidate["final_fusion_is_moe"] is False, "final fusion is still marked as MoE")
        method_text = (release_root / "docs" / "METHOD.md").read_text(encoding="utf-8")
        require("sparse MoE remains" in method_text, "MoE retention is not documented")
        require("final consensus is deliberately a" in method_text, "non-MoE final consensus is not documented")
        return {
            "freeze_sha256": sha256_file(cghc / "freeze.json"),
            "conditional_generation_required": True,
            "moe_retained_in_neural_path": True,
            "final_fusion_is_moe": False,
        }

    requirement(
        records,
        "method_boundary",
        "Conditional generation and MoE neural members are retained while final consensus is not an MoE.",
        method_boundary,
    )

    def all_six_superiority() -> dict[str, Any]:
        summary = load_json(cghc / "summary.json")
        require(summary["status"] == "retrospective_fixed_candidate_paired_evaluation_complete", "summary incomplete")
        for dataset in DATASETS:
            result = summary["results"][dataset]
            require(result["candidate_higher_than_every_baseline_on_all_six"] is True, f"{dataset} all-six gate failed")
            require(set(result["baselines"]) == set(BASELINES), f"{dataset} baseline suite changed")
            for baseline in BASELINES:
                row = result["baselines"][baseline]
                require(row["candidate_strictly_higher_all_six"] is True, f"{dataset}/{baseline} not all-six higher")
                require(set(row["delta_candidate_minus_baseline"]) == set(METRICS), "metric set changed")
                require(all(float(value) > 0.0 for value in row["delta_candidate_minus_baseline"].values()), f"{dataset}/{baseline} has non-positive delta")
        return {
            "summary_sha256": sha256_file(cghc / "summary.json"),
            "datasets": list(DATASETS),
            "baselines_per_dataset": len(BASELINES),
            "metrics": list(METRICS),
        }

    requirement(
        records,
        "all_six_superiority",
        "Both datasets are strictly above every baseline ensemble on all six aligned metrics.",
        all_six_superiority,
    )

    def paired_significance() -> dict[str, Any]:
        summary = load_json(cghc / "summary.json")
        accuracy = load_json(cghc / "secondary_accuracy_significance.json")
        adni = summary["inference"]["adni"]["flex_moe"]
        abcd = accuracy["results"]["abcd"]["flex_moe"]
        for name, result in (("ADNI Macro-F1", adni), ("ABCD Accuracy", abcd)):
            require(result["holm_significant_0.05"] is True, f"{name} is not Holm significant")
            require(result["bootstrap_lower_positive"] is True, f"{name} bootstrap lower bound is not positive")
        return {
            "adni_endpoint": "Macro-F1",
            "adni_holm_p": float(adni["holm_adjusted_p"]),
            "abcd_endpoint": "Accuracy",
            "abcd_holm_p": float(abcd["holm_adjusted_p"]),
            "claim_boundary": "retrospective fixed-candidate paired evidence",
        }

    requirement(
        records,
        "paired_significance",
        "At least one disclosed endpoint per dataset significantly exceeds Flex-MoE after Holm with a positive bootstrap bound.",
        paired_significance,
    )

    def conditional_generation_ablation() -> dict[str, Any]:
        ablation = load_json(
            source_root
            / "formal_matched_ablation_common6_v3"
            / "analysis_v1"
            / "summary.json"
        )
        deltas: dict[str, float] = {}
        for dataset in DATASETS:
            result = ablation["results"][dataset]
            require("no_completion" in result["arms"], f"{dataset} no-completion arm missing")
            delta = result["full_minus_ablation"]["no_completion"]["delta_full_minus_arm"]
            require(float(delta["Macro-F1"]) > 0.0, f"{dataset} completion does not improve Macro-F1")
            deltas[dataset] = float(delta["Macro-F1"])
        return {
            "campaign": "exact-500 separate matched component campaign",
            "macro_f1_full_minus_no_completion": deltas,
        }

    requirement(
        records,
        "conditional_generation_ablation",
        "A matched no-completion control exists on both datasets and full improves Macro-F1.",
        conditional_generation_ablation,
    )

    def interpretability(dataset: str) -> dict[str, Any]:
        root = cghc / "interpretability_v1" / dataset
        aggregate_path = root / "component_interpretability.json"
        receipt_path = root / "FINAL_RECEIPT.json"
        aggregate = load_json(aggregate_path)
        receipt = load_json(receipt_path)
        require(aggregate["status"] == "PASS", "aggregate did not pass")
        require(receipt["status"] == "PASS", "receipt did not pass")
        require(receipt["artifacts"]["json"]["sha256"] == sha256_file(aggregate_path), "aggregate hash differs from receipt")
        require(aggregate["semantics"]["causal_claim"] is False, "interpretation asserts causality")
        require(len(aggregate["disease_strata"]) == 3, "three disease strata missing")
        require(len(aggregate["disease_stratum_allocation_contrasts"]) == 4, "four modality contrasts missing")
        require(all(row["causal_effect"] is False for row in aggregate["disease_stratum_allocation_contrasts"]), "contrast asserts causal effect")
        require(all(row["status"] == "PASS" for row in aggregate["replay_audits"]), "checkpoint replay audit failed")
        return {
            "aggregate_sha256": sha256_file(aggregate_path),
            "receipt_sha256": sha256_file(receipt_path),
            "neural_members": int(aggregate["scope"]["neural_member_count"]),
            "disease_strata": aggregate["disease_strata"],
            "causal_claim": False,
        }

    for dataset in DATASETS:
        requirement(
            records,
            f"{dataset}_interpretability",
            f"{dataset.upper()} frozen conditional-generative members have receipt-bound modality–disease association output.",
            lambda dataset=dataset: interpretability(dataset),
        )

    def public_bundle() -> dict[str, Any]:
        result = release_root / "results" / "cghc_v1.json"
        markdown = release_root / "results" / "cghc_v1.md"
        receipt_path = release_root / "results" / "cghc_v1_receipt.json"
        figures = (
            release_root / "figures" / "cghc_common6.svg",
            release_root / "figures" / "cghc_modality_association.svg",
        )
        receipt = load_json(receipt_path)
        require(receipt["status"] == "PASS", "public receipt did not pass")
        for path in (result, markdown, *figures):
            require(receipt["artifacts"][path.name] == sha256_file(path), f"public hash mismatch: {path.name}")
        payload = load_json(result)
        require(payload["claim_boundary"]["causal_modality_or_disease_etiology_claim"] is False, "public bundle asserts causal etiology")
        require(payload["method"]["conditional_generation_required"] is True, "public bundle omits conditional generation")
        require(payload["method"]["moe_retained"] is True, "public bundle omits MoE retention")
        require(payload["method"]["final_consensus_is_moe"] is False, "public final consensus incorrectly marked MoE")
        readme = (release_root / "README.md").read_text(encoding="utf-8")
        require("results/cghc_v1.md" in readme, "README does not link current CGHC result page")
        return {
            "result_sha256": sha256_file(result),
            "markdown_sha256": sha256_file(markdown),
            "receipt_sha256": sha256_file(receipt_path),
            "figures": {path.name: sha256_file(path) for path in figures},
        }

    requirement(
        records,
        "public_bundle",
        "The aggregate-only JSON, synchronized result page, figures, hashes, and README link are complete.",
        public_bundle,
    )

    def main_branch() -> dict[str, Any]:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=release_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        require(branch == "main", f"release branch is {branch!r}, not main")
        return {"branch": branch}

    requirement(
        records,
        "main_branch",
        "Release work is assembled on main before commit and push.",
        main_branch,
    )

    passed = sum(record["status"] == "PASS" for record in records)
    return {
        "schema": "cghc-goal-completion-audit-v1",
        "status": "PASS" if passed == len(records) else "INCOMPLETE",
        "requirements_passed": passed,
        "requirements_total": len(records),
        "requirements": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.source_root, args.release_root)
    print(canonical_json(result), end="")
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
