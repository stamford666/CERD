#!/usr/bin/env python3
"""Validate one aggregate result artifact and render the public tables/figures.

Only aggregate values are accepted. Predictions, labels, identifiers,
checkpoints, and split files are outside the renderer input contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "final_results.json"
DEFAULT_README = ROOT / "README.md"
DEFAULT_FIGURES = ROOT / "figures"

START_MARKER = "<!-- FINAL_RESULTS_START -->"
END_MARKER = "<!-- FINAL_RESULTS_END -->"

DATASET_ORDER = ("adni", "abcd")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_auroc",
    "macro_auprc",
)
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "BalAcc",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "macro_auroc": "Macro-AUROC",
    "macro_auprc": "Macro-AUPRC",
}
REQUIRED_ABLATIONS = (
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
)
BRANCH_GROUPS = ("joint", "unimodal", "pairwise")
MISSING = "NOT AVAILABLE"


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_unique_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object while rejecting duplicate keys at any depth."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("input must be unique-key UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    return payload
ADAPTIVE_DEVELOPMENT_CLAIM = (
    "Adaptive same cohort development evidence only; both scored cohorts were "
    "reused for model and configuration selection, so any Holm adjusted "
    "difference is descriptive and confirmatory support is false."
)
ADNI_CAMPAIGN_BOUNDARY = (
    "ADNI validation-318, test-318, and unassigned-910 are outside this campaign: "
    "not selected into any arm, not iterated over, not scored, and not included "
    "in any fitted statistic."
)
ABCD_CAMPAIGN_BOUNDARY = (
    "ABCD protected temporal internal holdout 850 is outside this campaign: "
    "not selected into any arm, not iterated over, not scored, and not included "
    "in any fitted statistic."
)
DATASET_PUBLIC_CONTRACT = {
    "adni": {
        "display_name": "ADNI",
        "task": "three-class diagnosis classification",
        "design": "five-fold subject-level out-of-fold ensemble evaluation",
        "subjects": 1480,
        "folds": 5,
        "split": "training1480 out-of-fold development cohort",
        "evidence_scope": "development_cv",
        "partition_reused": True,
        "selection_independent": False,
        "resampling_units": 1480,
        "modality_names": ("image", "genomic", "clinical", "biospecimen"),
    },
    "abcd": {
        "display_name": "ABCD",
        "task": (
            "strict three-class ADHD presentation at ses-01A from ses-00A "
            "IGCB features"
        ),
        "design": "five-fold family-disjoint out-of-fold ensemble evaluation",
        "subjects": 946,
        "folds": 5,
        "split": "dev946 family-disjoint out-of-fold development cohort",
        "evidence_scope": "development_cv",
        "partition_reused": True,
        "selection_independent": False,
        "resampling_units": 922,
        "modality_names": (
            "imaging",
            "genetic",
            "cognition and health",
            "behavior and environment",
        ),
    },
}
EVIDENCE_SCOPES = {
    "development_cv",
    "reused_validation",
    "locked_evaluation",
    "locked_internal_holdout",
    "external_test",
}
SUPERIORITY_ELIGIBLE_SCOPES = {
    "locked_evaluation",
    "locked_internal_holdout",
    "external_test",
}
RESAMPLING_UNIT = {"adni": "subject", "abcd": "family"}
SWAP_DRAWS = 50_000
SWAP_RNG_SEED = 20_260_905
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_RNG_SEED = 20_260_906
ALPHA = 0.05
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
PAIRED_TEST = "paired swap test"
MULTIPLICITY_ADJUSTMENT = "Holm across ADNI and ABCD primary Macro-F1 comparisons"
INTERPRETABILITY_CONDITION_DESIGNS = {
    "natural_disjoint",
}
INTERPRETABILITY_AGGREGATION_UNIT = "subject"
INTERPRETABILITY_AGGREGATION_METHOD = (
    "arithmetic mean of per-subject normalized vectors"
)
INTERPRETABILITY_DEFINITIONS = {
    "complete": "naturally complete four-modality inputs",
    "incomplete": "naturally incomplete inputs",
}
PUBLIC_INTERPRETABILITY_SOURCES = {
    "aggregate out-of-fold checkpoint replay",
    "aggregate locked-evaluation checkpoint replay",
    "aggregate external-evaluation checkpoint replay",
}
INTERPRETABILITY_SOURCE_BY_SCOPE = {
    "development_cv": "aggregate out-of-fold checkpoint replay",
    "reused_validation": "aggregate out-of-fold checkpoint replay",
    "locked_evaluation": "aggregate locked-evaluation checkpoint replay",
    "locked_internal_holdout": "aggregate locked-evaluation checkpoint replay",
    "external_test": "aggregate external-evaluation checkpoint replay",
}
MIN_PUBLIC_CELL = 10
PROBABILITY_SUM_TOLERANCE = 1e-6
MARKDOWN_SPECIAL = re.compile(r"[\r\n|<>`\[\]*_#\\]")
ABSOLUTE_LOCAL_PATH = re.compile(
    r"(?:^|[\s=:,(])(?:~[/\\]|/(?!/)[^\s|]+|[A-Za-z]:[/\\][^\s|]+|file://)",
    re.IGNORECASE,
)
RELATIVE_INTERNAL_PATH = re.compile(
    r"(?:^|[\s=:,(])(?:\.{1,2}/|(?:runs?|outputs?|logs?|checkpoints?|saves?|private|tmp|home)/)[^\s|]*",
    re.IGNORECASE,
)
PARTICIPANT_LIKE_IDENTIFIER = re.compile(
    r"\b(?:participant|subject|rid)[-_]?\d+\b",
    re.IGNORECASE,
)
PLACEHOLDER_TEXT = re.compile(
    r"\b(?:NOT\s+AVAILABLE|TBD|TODO|PLACEHOLDER|PENDING)\b",
    re.IGNORECASE,
)
RFC3339_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_string(value: Any, path: str, *, final: bool) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    if final and PLACEHOLDER_TEXT.search(value):
        _fail(path, "must not contain placeholder text in a final artifact")


def _require_rfc3339(value: Any, path: str) -> None:
    _require_public_text(value, path, final=True)
    if not RFC3339_TIMESTAMP.fullmatch(value):
        _fail(path, "must be an RFC3339 timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(path, "must be an RFC3339 timestamp with a timezone")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(path, "must be an RFC3339 timestamp with a timezone")


def _require_sha256(value: Any, path: str) -> None:
    if not isinstance(value, str) or SHA256_HEX.fullmatch(value) is None:
        _fail(path, "must be a lowercase 64-character SHA-256 digest")


def _require_public_text(value: Any, path: str, *, final: bool) -> None:
    """Require plain public text that cannot break Markdown or expose a path."""

    _require_string(value, path, final=final)
    if ABSOLUTE_LOCAL_PATH.search(value):
        _fail(path, "must not contain an absolute local path")
    if RELATIVE_INTERNAL_PATH.search(value):
        _fail(path, "must not contain a relative internal path")
    if PARTICIPANT_LIKE_IDENTIFIER.search(value):
        _fail(path, "must not contain a participant-like identifier")
    if MARKDOWN_SPECIAL.search(value):
        _fail(path, "must not contain Markdown control characters")


def _validate_metrics(value: Any, path: str, *, final: bool) -> None:
    if value is None and not final:
        return
    if not isinstance(value, dict):
        _fail(path, "must be an object containing exactly the common-six metrics")
    if set(value) != set(METRICS):
        _fail(path, f"keys must be exactly {list(METRICS)}")
    for metric in METRICS:
        score = value[metric]
        if not _is_number(score) or not 0.0 <= float(score) <= 1.0:
            _fail(f"{path}.{metric}", "must be a finite fraction in [0, 1]")


def _validate_probability_vector(value: Any, size: int, path: str, *, final: bool) -> None:
    if value is None and not final:
        return
    if not isinstance(value, list) or len(value) != size:
        _fail(path, f"must contain {size} aggregate fractions")
    if any(not _is_number(item) or not 0.0 <= float(item) <= 1.0 for item in value):
        _fail(path, "all values must be finite fractions in [0, 1]")
    if not math.isclose(
        sum(float(item) for item in value),
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE,
    ):
        _fail(
            path,
            f"values must sum to one within absolute tolerance {PROBABILITY_SUM_TOLERANCE}",
        )


def _validate_branch_mass(value: Any, path: str, *, final: bool) -> None:
    if value is None and not final:
        return
    if not isinstance(value, dict) or set(value) != set(BRANCH_GROUPS):
        _fail(path, f"keys must be exactly {list(BRANCH_GROUPS)}")
    vals = [value[key] for key in BRANCH_GROUPS]
    _validate_probability_vector(vals, len(BRANCH_GROUPS), path, final=True)


def _comparison_significant(comparison: dict[str, Any]) -> bool:
    return float(comparison["adjusted_p_value"]) < ALPHA


def _holm_adjusted_p_values(raw_p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in the original order."""

    count = len(raw_p_values)
    ordered = sorted(range(count), key=lambda index: (raw_p_values[index], index))
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, original_index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * raw_p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def _validate_comparison(
    comparison: Any,
    path: str,
    *,
    dataset_key: str,
    final: bool,
    primary: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> None:
    required = {
        "ours",
        "comparator",
        "metric",
        "paired",
        "delta",
        "bootstrap_lower_bound",
        "bootstrap_confidence_level",
        "p_value",
        "adjusted_p_value",
        "alpha",
        "test",
        "alternative",
        "resampling_unit",
        "n_units",
        "swap_draws",
        "swap_rng_seed",
        "bootstrap_draws",
        "bootstrap_rng_seed",
        "multiplicity_adjustment",
        "analysis_receipt_sha256",
        "confirmatory_support",
    }
    if not isinstance(comparison, dict) or set(comparison) != required:
        _fail(path, f"keys must be exactly {sorted(required)}")
    if not final:
        return

    for field in (
        "ours",
        "comparator",
        "test",
        "alternative",
        "resampling_unit",
        "multiplicity_adjustment",
    ):
        _require_public_text(comparison[field], f"{path}.{field}", final=True)
    _require_sha256(
        comparison["analysis_receipt_sha256"],
        f"{path}.analysis_receipt_sha256",
    )
    ours, comparator = primary
    if comparison["ours"] != ours["method"]:
        _fail(f"{path}.ours", "must name the sole primary row whose role is 'ours'")
    if comparison["comparator"] != comparator["method"]:
        _fail(
            f"{path}.comparator",
            "must name the sole primary row whose role is 'comparator'",
        )
    if comparison["metric"] != "macro_f1":
        _fail(f"{path}.metric", "must be the pre-specified primary metric 'macro_f1'")
    if comparison["paired"] is not True:
        _fail(f"{path}.paired", "must be true")
    if comparison["test"] != PAIRED_TEST:
        _fail(f"{path}.test", f"must be {PAIRED_TEST!r}")
    if comparison["alternative"] != "greater":
        _fail(f"{path}.alternative", "must be 'greater'")
    if comparison["resampling_unit"] != RESAMPLING_UNIT[dataset_key]:
        _fail(
            f"{path}.resampling_unit",
            f"must be {RESAMPLING_UNIT[dataset_key]!r} for {dataset_key}",
        )
    if comparison["multiplicity_adjustment"] != MULTIPLICITY_ADJUSTMENT:
        _fail(
            f"{path}.multiplicity_adjustment",
            f"must be {MULTIPLICITY_ADJUSTMENT!r}",
        )

    delta = comparison["delta"]
    if not _is_number(delta) or not -1.0 <= float(delta) <= 1.0:
        _fail(f"{path}.delta", "must be a finite fraction in [-1, 1]")
    expected_delta = float(ours["metrics"]["macro_f1"]) - float(
        comparator["metrics"]["macro_f1"]
    )
    if not math.isclose(float(delta), expected_delta, rel_tol=0.0, abs_tol=1e-12):
        _fail(
            f"{path}.delta",
            "must equal ours macro_f1 minus comparator macro_f1",
        )

    lower_bound = comparison["bootstrap_lower_bound"]
    if not _is_number(lower_bound) or not -1.0 <= float(lower_bound) <= 1.0:
        _fail(
            f"{path}.bootstrap_lower_bound",
            "must be a finite fraction in [-1, 1]",
        )
    for field in ("p_value", "adjusted_p_value"):
        value = comparison[field]
        if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
            _fail(f"{path}.{field}", "must be a finite fraction in [0, 1]")

    exact_numeric_fields = {
        "alpha": ALPHA,
        "bootstrap_confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
    }
    for field, expected in exact_numeric_fields.items():
        value = comparison[field]
        if not _is_number(value) or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            _fail(f"{path}.{field}", f"must equal {expected}")

    exact_integer_fields = {
        "swap_draws": SWAP_DRAWS,
        "swap_rng_seed": SWAP_RNG_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_rng_seed": BOOTSTRAP_RNG_SEED,
    }
    for field, expected in exact_integer_fields.items():
        if comparison[field] != expected or isinstance(comparison[field], bool):
            _fail(f"{path}.{field}", f"must equal {expected}")

    n_units = comparison["n_units"]
    if not isinstance(n_units, int) or isinstance(n_units, bool) or n_units <= 0:
        _fail(f"{path}.n_units", "must be a positive integer")
    expected_units = DATASET_PUBLIC_CONTRACT[dataset_key]["resampling_units"]
    if n_units != expected_units:
        unit = RESAMPLING_UNIT[dataset_key]
        _fail(
            f"{path}.n_units",
            f"must equal the frozen {dataset_key.upper()} {unit} count {expected_units}",
        )
    if not isinstance(comparison["confirmatory_support"], bool):
        _fail(f"{path}.confirmatory_support", "must be boolean")


def _validate_joint_statistics(payload: dict[str, Any]) -> None:
    comparisons = [
        payload["datasets"][dataset_key]["statistical_comparisons"][0]
        for dataset_key in DATASET_ORDER
    ]
    expected_adjusted = _holm_adjusted_p_values(
        [float(comparison["p_value"]) for comparison in comparisons]
    )
    for dataset_key, comparison, expected_p in zip(
        DATASET_ORDER, comparisons, expected_adjusted
    ):
        path = f"datasets.{dataset_key}.statistical_comparisons[0]"
        if not math.isclose(
            float(comparison["adjusted_p_value"]),
            expected_p,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(
                f"{path}.adjusted_p_value",
                "must equal the jointly recomputed two-comparison Holm value",
            )

        evaluation = payload["datasets"][dataset_key]["evaluation"]
        expected_support = (
            _comparison_significant(comparison)
            and float(comparison["delta"]) > 0.0
            and float(comparison["bootstrap_lower_bound"]) > 0.0
            and evaluation["evidence_scope"] in SUPERIORITY_ELIGIBLE_SCOPES
            and evaluation["partition_reused"] is False
            and evaluation["selection_independent"] is True
        )
        if comparison["confirmatory_support"] != expected_support:
            _fail(
                f"{path}.confirmatory_support",
                "must match the locked, independent Holm/paired-bootstrap rule; reused development must be false",
            )


def validate_payload(payload: Any) -> None:
    """Validate the public aggregate schema and final-release completeness."""

    if not isinstance(payload, dict):
        _fail("root", "must be a JSON object")
    required_root = {
        "schema",
        "status",
        "generated_at",
        "source_manifest_sha256",
        "metric_unit",
        "metric_order",
        "claim_boundary",
        "datasets",
    }
    if set(payload) != required_root:
        _fail("root", f"keys must be exactly {sorted(required_root)}")
    if payload["schema"] != "cerd-final-public-results-v3":
        _fail("schema", "unsupported schema")
    if payload["status"] not in {"not_available", "final"}:
        _fail("status", "must be 'not_available' or 'final'")
    final = payload["status"] == "final"
    if payload["metric_unit"] != "fraction":
        _fail("metric_unit", "must be 'fraction'")
    if payload["metric_order"] != list(METRICS):
        _fail("metric_order", f"must be exactly {list(METRICS)}")
    _require_public_text(payload["claim_boundary"], "claim_boundary", final=final)
    if final:
        _require_sha256(payload["source_manifest_sha256"], "source_manifest_sha256")
        if payload["claim_boundary"] != ADAPTIVE_DEVELOPMENT_CLAIM:
            _fail(
                "claim_boundary",
                "must use the frozen adaptive reused-development claim wording",
            )
        _require_rfc3339(payload["generated_at"], "generated_at")
    else:
        if payload["generated_at"] is not None:
            _fail("generated_at", "must be null while status is not_available")
        if payload["source_manifest_sha256"] is not None:
            _fail(
                "source_manifest_sha256",
                "must be null while status is not_available",
            )

    datasets = payload["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != set(DATASET_ORDER):
        _fail("datasets", f"must contain exactly {list(DATASET_ORDER)}")

    for dataset_key in DATASET_ORDER:
        dataset = datasets[dataset_key]
        path = f"datasets.{dataset_key}"
        public_contract = DATASET_PUBLIC_CONTRACT[dataset_key]
        required_dataset = {
            "display_name",
            "task",
            "evaluation",
            "primary_results",
            "statistical_comparisons",
            "ablations",
            "interpretability",
        }
        if not isinstance(dataset, dict) or set(dataset) != required_dataset:
            _fail(path, f"keys must be exactly {sorted(required_dataset)}")
        _require_public_text(dataset["display_name"], f"{path}.display_name", final=final)
        _require_public_text(dataset["task"], f"{path}.task", final=final)
        if final:
            for field in ("display_name", "task"):
                if dataset[field] != public_contract[field]:
                    _fail(
                        f"{path}.{field}",
                        f"must equal the frozen public {dataset_key.upper()} {field}",
                    )

        evaluation = dataset["evaluation"]
        required_evaluation = {
            "design",
            "subjects",
            "folds",
            "split",
            "evidence_scope",
            "partition_reused",
            "selection_independent",
        }
        if not isinstance(evaluation, dict) or set(evaluation) != required_evaluation:
            _fail(
                f"{path}.evaluation",
                f"keys must be exactly {sorted(required_evaluation)}",
            )
        _require_public_text(evaluation["design"], f"{path}.evaluation.design", final=final)
        _require_public_text(evaluation["split"], f"{path}.evaluation.split", final=final)
        if final:
            if evaluation["evidence_scope"] not in EVIDENCE_SCOPES:
                _fail(
                    f"{path}.evaluation.evidence_scope",
                    f"must be one of {sorted(EVIDENCE_SCOPES)}",
                )
            for field in ("subjects", "folds"):
                value = evaluation[field]
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    _fail(f"{path}.evaluation.{field}", "must be a positive integer")
            for field in ("partition_reused", "selection_independent"):
                if not isinstance(evaluation[field], bool):
                    _fail(f"{path}.evaluation.{field}", "must be boolean")
            if evaluation["partition_reused"] and evaluation["selection_independent"]:
                _fail(
                    f"{path}.evaluation",
                    "a reused partition cannot be selection-independent",
                )
            if (
                evaluation["evidence_scope"] == "reused_validation"
                and evaluation["partition_reused"] is not True
            ):
                _fail(
                    f"{path}.evaluation.partition_reused",
                    "must be true when evidence_scope is reused_validation",
                )
            expected_evaluation = {
                field: public_contract[field]
                for field in (
                    "design",
                    "subjects",
                    "folds",
                    "split",
                    "evidence_scope",
                    "partition_reused",
                    "selection_independent",
                )
            }
            for field, expected in expected_evaluation.items():
                if evaluation[field] != expected:
                    _fail(
                        f"{path}.evaluation.{field}",
                        f"must equal the frozen reused-development value {expected!r}",
                    )
        else:
            if evaluation["evidence_scope"] != MISSING:
                _fail(
                    f"{path}.evaluation.evidence_scope",
                    f"must be {MISSING!r} before finalization",
                )
            if any(
                evaluation[field] is not None
                for field in ("subjects", "folds", "partition_reused", "selection_independent")
            ):
                _fail(
                    f"{path}.evaluation",
                    "subjects, folds, partition_reused, and selection_independent must be null before finalization",
                )

        primary = dataset["primary_results"]
        if not isinstance(primary, list) or len(primary) != 2:
            _fail(
                f"{path}.primary_results",
                "must contain exactly two rows ordered as ours, comparator",
            )
        methods: list[str] = []
        roles: list[str] = []
        for index, row in enumerate(primary):
            row_path = f"{path}.primary_results[{index}]"
            required_primary = {
                "method",
                "role",
                "configuration_id",
                "configuration_sha256",
                "execution_receipt_sha256",
                "metrics",
            }
            if not isinstance(row, dict) or set(row) != required_primary:
                _fail(
                    row_path,
                    f"keys must be exactly {sorted(required_primary)}",
                )
            _require_public_text(row["method"], f"{row_path}.method", final=final)
            _require_public_text(
                row["configuration_id"],
                f"{row_path}.configuration_id",
                final=final,
            )
            if final:
                _require_sha256(
                    row["configuration_sha256"],
                    f"{row_path}.configuration_sha256",
                )
                _require_sha256(
                    row["execution_receipt_sha256"],
                    f"{row_path}.execution_receipt_sha256",
                )
            if row["role"] not in {"ours", "comparator"}:
                _fail(f"{row_path}.role", "must be 'ours' or 'comparator'")
            methods.append(row["method"])
            roles.append(row["role"])
            _validate_metrics(row["metrics"], f"{row_path}.metrics", final=final)
        if len(methods) != len(set(methods)):
            _fail(f"{path}.primary_results", "method names must be unique")
        configuration_ids = [row["configuration_id"] for row in primary]
        configuration_hashes = [row["configuration_sha256"] for row in primary]
        if len(configuration_ids) != len(set(configuration_ids)):
            _fail(
                f"{path}.primary_results",
                "configuration ids must distinguish ours from comparator",
            )
        if final and len(configuration_hashes) != len(set(configuration_hashes)):
            _fail(
                f"{path}.primary_results",
                "configuration digests must distinguish ours from comparator",
            )
        if roles != ["ours", "comparator"]:
            _fail(
                f"{path}.primary_results",
                "roles must be ordered exactly as ours, comparator",
            )

        comparisons = dataset["statistical_comparisons"]
        if not isinstance(comparisons, list) or len(comparisons) != 1:
            _fail(
                f"{path}.statistical_comparisons",
                "must contain exactly one paired primary Macro-F1 comparison",
            )
        _validate_comparison(
            comparisons[0],
            f"{path}.statistical_comparisons[0]",
            dataset_key=dataset_key,
            final=final,
            primary=primary,
            evaluation=evaluation,
        )

        ablations = dataset["ablations"]
        if not isinstance(ablations, list):
            _fail(f"{path}.ablations", "must be a list")
        id_labels: list[tuple[str, str]] = []
        ablation_configuration_ids: list[str] = []
        ablation_configuration_hashes: list[str] = []
        for index, row in enumerate(ablations):
            row_path = f"{path}.ablations[{index}]"
            required_ablation = {
                "id",
                "label",
                "configuration_id",
                "configuration_sha256",
                "execution_receipt_sha256",
                "metrics",
            }
            if not isinstance(row, dict) or set(row) != required_ablation:
                _fail(
                    row_path,
                    f"keys must be exactly {sorted(required_ablation)}",
                )
            _require_string(row["id"], f"{row_path}.id", final=False)
            _require_public_text(row["label"], f"{row_path}.label", final=final)
            _require_public_text(
                row["configuration_id"],
                f"{row_path}.configuration_id",
                final=final,
            )
            if final:
                _require_sha256(
                    row["configuration_sha256"],
                    f"{row_path}.configuration_sha256",
                )
                _require_sha256(
                    row["execution_receipt_sha256"],
                    f"{row_path}.execution_receipt_sha256",
                )
            ablation_configuration_ids.append(row["configuration_id"])
            ablation_configuration_hashes.append(row["configuration_sha256"])
            id_labels.append((row["id"], row["label"]))
            _validate_metrics(row["metrics"], f"{row_path}.metrics", final=final)
        if id_labels != list(REQUIRED_ABLATIONS):
            _fail(
                f"{path}.ablations",
                "must contain the exact ordered pre-specified ids and canonical labels",
            )
        all_configuration_ids = configuration_ids + ablation_configuration_ids
        if len(all_configuration_ids) != len(set(all_configuration_ids)):
            _fail(
                f"{path}.ablations",
                "all primary and ablation configuration ids must be unique",
            )
        if final:
            all_configuration_hashes = (
                configuration_hashes + ablation_configuration_hashes
            )
            if len(all_configuration_hashes) != len(set(all_configuration_hashes)):
                _fail(
                    f"{path}.ablations",
                    "all primary and ablation configuration digests must be unique",
                )

        interpretability = dataset["interpretability"]
        required_interpretability = {
            "source",
            "condition_design",
            "aggregation_unit",
            "aggregation_method",
            "aggregation_receipt_sha256",
            "modality_names",
            "conditions",
        }
        if not isinstance(interpretability, dict) or set(interpretability) != required_interpretability:
            _fail(f"{path}.interpretability", f"keys must be exactly {sorted(required_interpretability)}")
        _require_public_text(
            interpretability["source"],
            f"{path}.interpretability.source",
            final=final,
        )
        if final:
            _require_sha256(
                interpretability["aggregation_receipt_sha256"],
                f"{path}.interpretability.aggregation_receipt_sha256",
            )
        if final and interpretability["source"] not in PUBLIC_INTERPRETABILITY_SOURCES:
            _fail(
                f"{path}.interpretability.source",
                f"must be one of {sorted(PUBLIC_INTERPRETABILITY_SOURCES)}",
            )
        if final and interpretability["source"] != INTERPRETABILITY_SOURCE_BY_SCOPE[
            evaluation["evidence_scope"]
        ]:
            _fail(
                f"{path}.interpretability.source",
                "must match the declared evaluation evidence scope",
            )
        condition_design = interpretability["condition_design"]
        if final and condition_design not in INTERPRETABILITY_CONDITION_DESIGNS:
            _fail(
                f"{path}.interpretability.condition_design",
                f"must be one of {sorted(INTERPRETABILITY_CONDITION_DESIGNS)}",
            )
        for field, expected in (
            ("aggregation_unit", INTERPRETABILITY_AGGREGATION_UNIT),
            ("aggregation_method", INTERPRETABILITY_AGGREGATION_METHOD),
        ):
            _require_public_text(
                interpretability[field],
                f"{path}.interpretability.{field}",
                final=final,
            )
            if final and interpretability[field] != expected:
                _fail(
                    f"{path}.interpretability.{field}",
                    f"must equal the frozen subject-level aggregation value {expected!r}",
                )
        modalities = interpretability["modality_names"]
        if not isinstance(modalities, list) or len(modalities) != 4:
            _fail(f"{path}.interpretability.modality_names", "must contain exactly four names")
        for index, name in enumerate(modalities):
            _require_public_text(name, f"{path}.interpretability.modality_names[{index}]", final=final)
        if final and modalities != list(public_contract["modality_names"]):
            _fail(
                f"{path}.interpretability.modality_names",
                "must equal the frozen dataset-specific modality names and order",
            )
        conditions = interpretability["conditions"]
        if not isinstance(conditions, dict) or set(conditions) != {"complete", "incomplete"}:
            _fail(f"{path}.interpretability.conditions", "must contain complete and incomplete")
        for condition_name in ("complete", "incomplete"):
            condition = conditions[condition_name]
            condition_path = f"{path}.interpretability.conditions.{condition_name}"
            required_condition = {
                "definition",
                "subjects",
                "decision_allocation",
                "branch_mass",
            }
            if not isinstance(condition, dict) or set(condition) != required_condition:
                _fail(
                    condition_path,
                    f"keys must be exactly {sorted(required_condition)}",
                )
            _require_public_text(
                condition["definition"],
                f"{condition_path}.definition",
                final=final,
            )
            if final and condition["definition"] != INTERPRETABILITY_DEFINITIONS[
                condition_name
            ]:
                _fail(
                    f"{condition_path}.definition",
                    "must equal the frozen natural complete/incomplete definition",
                )
            if final:
                subjects = condition["subjects"]
                if (
                    not isinstance(subjects, int)
                    or isinstance(subjects, bool)
                    or subjects < MIN_PUBLIC_CELL
                ):
                    _fail(
                        f"{condition_path}.subjects",
                        f"must be an integer at least {MIN_PUBLIC_CELL}",
                    )
            elif condition["subjects"] is not None:
                _fail(f"{condition_path}.subjects", "must be null before finalization")
            _validate_probability_vector(
                condition["decision_allocation"],
                len(modalities),
                f"{condition_path}.decision_allocation",
                final=final,
            )
            _validate_branch_mass(condition["branch_mass"], f"{condition_path}.branch_mass", final=final)

        if final:
            public_subjects = sum(
                conditions[name]["subjects"] for name in ("complete", "incomplete")
            )
            if public_subjects != evaluation["subjects"]:
                _fail(
                    f"{path}.interpretability.conditions",
                    "natural_disjoint condition counts must sum to evaluation.subjects",
                )

    if final:
        _validate_joint_statistics(payload)


def _metric(value: Any) -> str:
    return MISSING if value is None else f"{100.0 * float(value):.2f}"


def _pvalue(value: Any) -> str:
    if value is None:
        return MISSING
    value = float(value)
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _comparison_cell(value: Any) -> str:
    return MISSING if value is None else ("yes" if value else "no")


def _plain(value: Any) -> str:
    return MISSING if value is None else str(value)


def _scope(value: Any) -> str:
    if value in (None, MISSING):
        return MISSING
    return str(value).replace("_", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the generated README block."""

    final = payload["status"] == "final"
    if not final:
        return "\n".join(
            [
                START_MARKER,
                "> **Release status: NOT AVAILABLE.** " + payload["claim_boundary"],
                "",
                "Aggregate tables and figures are generated only from an approved final artifact.",
                "",
                END_MARKER,
            ]
        )
    status = "FINAL" if final else MISSING
    lines = [
        START_MARKER,
        f"> **Release status: {status}.** {payload['claim_boundary']}",
        "",
        "All reported predictive results use exactly six metrics. Values are percentages; no binary-only or task-specific metric is mixed into either dataset.",
        "",
        "### Evaluation boundary",
        "",
        "| Dataset | Evidence scope | Design | Split | n | Folds | Partition reused | Selection independent |",
        "|---|---|---|---|---:|---:|:---:|:---:|",
    ]
    for dataset_key in DATASET_ORDER:
        dataset = payload["datasets"][dataset_key]
        evaluation = dataset["evaluation"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset["display_name"],
                    _scope(evaluation.get("evidence_scope")),
                    _plain(evaluation.get("design")),
                    _plain(evaluation.get("split")),
                    _plain(evaluation.get("subjects")),
                    _plain(evaluation.get("folds")),
                    _comparison_cell(evaluation.get("partition_reused")),
                    _comparison_cell(evaluation.get("selection_independent")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            f"ADNI campaign boundary: {ADNI_CAMPAIGN_BOUNDARY}",
            "",
            f"ABCD campaign boundary: {ABCD_CAMPAIGN_BOUNDARY}",
        ]
    )

    lines.extend(
        [
        "",
        "### Final common-six results",
        "",
        "| Dataset | Method | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_key in DATASET_ORDER:
        dataset = payload["datasets"][dataset_key]
        for row in dataset["primary_results"]:
            metrics = row["metrics"] or {metric: None for metric in METRICS}
            rendered = " | ".join(_metric(metrics[metric]) for metric in METRICS)
            lines.append(f"| {dataset['display_name']} | {row['method']} | {rendered} |")

    lines.extend(
        [
            "",
            "### Statistical comparisons",
            "",
            "Each dataset contributes exactly one paired, one-sided Macro-F1 comparison. Swap-test p-values are Holm-adjusted jointly across ADNI and ABCD; significance is derived from adjusted p < 0.05. Confirmatory support additionally requires a positive 95% bootstrap lower bound on a selection-independent, non-reused locked evaluation scope.",
            "",
            "| Dataset | Paired comparison | Metric | Difference (pp) | Bootstrap lower (pp) / level | Swap p | Holm adjusted p | Alpha | Significant difference | Confirmatory support | Test / unit / n | Swap draws / RNG | Bootstrap draws / RNG |",
            "|---|---|---|---:|---:|---:|---:|---:|:---:|:---:|---|---|---|",
        ]
    )
    for dataset_key in DATASET_ORDER:
        dataset = payload["datasets"][dataset_key]
        comparisons = dataset["statistical_comparisons"]
        if not comparisons:
            lines.append(
                f"| {dataset['display_name']} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} |"
            )
            continue
        for comparison in comparisons:
            delta = comparison.get("delta")
            delta_text = MISSING if delta is None else f"{100.0 * float(delta):+.2f}"
            lower = comparison.get("bootstrap_lower_bound")
            lower_text = MISSING if lower is None else f"{100.0 * float(lower):+.2f}"
            confidence_level = comparison.get("bootstrap_confidence_level")
            if lower_text != MISSING and confidence_level is not None:
                lower_text += f" / {100.0 * float(confidence_level):.1f}%"
            metric = comparison.get("metric")
            metric_text = METRIC_LABELS.get(metric, MISSING)
            test = comparison.get("test")
            unit = comparison.get("resampling_unit")
            n_units = comparison.get("n_units")
            test_text = (
                MISSING
                if not test or not unit or n_units is None
                else f"{test} / {unit} / n={n_units}"
            )
            swap_text = (
                f"{comparison.get('swap_draws')} / {comparison.get('swap_rng_seed')}"
            )
            bootstrap_text = (
                f"{comparison.get('bootstrap_draws')} / "
                f"{comparison.get('bootstrap_rng_seed')}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset["display_name"],
                        f"{comparison.get('ours', MISSING)} − {comparison.get('comparator', MISSING)}",
                        metric_text,
                        delta_text,
                        lower_text,
                        _pvalue(comparison.get("p_value")),
                        _pvalue(comparison.get("adjusted_p_value")),
                        _pvalue(comparison.get("alpha")),
                        _comparison_cell(_comparison_significant(comparison)),
                        _comparison_cell(comparison.get("confirmatory_support")),
                        test_text,
                        swap_text,
                        bootstrap_text,
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "### Pre-specified matched ablations",
            "",
            "Each row is one pre-specified matched configuration relative to full CERD. Differences are descriptive ablation effects and do not by themselves establish causal necessity.",
            "",
            "| Dataset | Ablation | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_key in DATASET_ORDER:
        dataset = payload["datasets"][dataset_key]
        for row in dataset["ablations"]:
            metrics = row["metrics"] or {metric: None for metric in METRICS}
            rendered = " | ".join(_metric(metrics[metric]) for metric in METRICS)
            lines.append(f"| {dataset['display_name']} | {row['label']} | {rendered} |")

    lines.extend(
        [
            "",
            "### Aggregate interpretability",
            "",
            "The explanation artifact contrasts complete and incomplete inputs using modality-level decision allocation and grouped joint/unimodal/pairwise branch mass. These are descriptive routing summaries, not causal feature importance. No participant-level explanation data are released.",
            "",
            "| Dataset | Condition design | Complete condition | Incomplete condition | Aggregate source |",
            "|---|---|---|---|---|",
        ]
    )
    for dataset_key in DATASET_ORDER:
        dataset = payload["datasets"][dataset_key]
        interpretability = dataset["interpretability"]
        conditions = interpretability["conditions"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset["display_name"],
                    _plain(interpretability.get("condition_design")),
                    _plain(conditions["complete"].get("definition")),
                    _plain(conditions["incomplete"].get("definition")),
                    _plain(interpretability.get("source")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "- [Common-six performance](figures/common6.svg)",
            "- [Ablation effects](figures/ablations.svg)",
            "- [Decision allocation and branch mass](figures/decision_allocation.svg)",
            "",
            END_MARKER,
        ]
    )
    return "\n".join(lines)


def replace_generated_block(readme: str, block: str) -> str:
    if readme.count(START_MARKER) != 1 or readme.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one generated result marker pair")
    prefix, remainder = readme.split(START_MARKER, 1)
    _, suffix = remainder.split(END_MARKER, 1)
    return prefix + block + suffix


def _heat_color(value: float) -> str:
    # A fixed 0-100% scale avoids changing visual meaning between campaigns.
    start = (239, 246, 255)
    end = (30, 94, 168)
    return "#{:02x}{:02x}{:02x}".format(
        *(round(a + (b - a) * value) for a, b in zip(start, end))
    )


def render_common6_svg(payload: dict[str, Any]) -> str:
    if payload["status"] != "final":
        raise ValueError("figures require an approved final result artifact")
    rows: list[tuple[str, str, dict[str, float]]] = []
    for dataset_key in DATASET_ORDER:
        dataset = payload["datasets"][dataset_key]
        for result in dataset["primary_results"]:
            rows.append((dataset["display_name"], result["method"], result["metrics"]))

    left = 310
    top = 120
    cell_width = 140
    row_height = 52
    width = left + len(METRICS) * cell_width + 35
    height = top + len(rows) * row_height + 80
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">ADNI and ABCD common-six performance</title>',
        '  <desc id="desc">Aggregate percentage metrics by dataset and method; darker cells indicate larger values on a fixed zero-to-one scale.</desc>',
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        '  <text x="28" y="42" font-family="sans-serif" font-size="27" font-weight="700" fill="#20242a">ADNI and ABCD common-six performance</text>',
        '  <text x="28" y="72" font-family="sans-serif" font-size="15" fill="#555b63">Values are percentages; color uses a fixed 0–100% scale.</text>',
    ]
    for column, metric in enumerate(METRICS):
        x = left + column * cell_width + cell_width / 2
        items.append(
            f'  <text x="{x:.1f}" y="104" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="600" fill="#30353b">{html.escape(METRIC_LABELS[metric])}</text>'
        )
    previous_dataset = None
    for row_index, (dataset, method, metrics) in enumerate(rows):
        y = top + row_index * row_height
        fill = "#f7f8fa" if row_index % 2 == 0 else "#ffffff"
        items.append(f'  <rect x="20" y="{y}" width="{width - 40}" height="{row_height}" fill="{fill}"/>')
        dataset_text = dataset if dataset != previous_dataset else ""
        previous_dataset = dataset
        items.append(f'  <text x="30" y="{y + 32}" font-family="sans-serif" font-size="15" font-weight="700" fill="#30353b">{html.escape(dataset_text)}</text>')
        items.append(f'  <text x="105" y="{y + 32}" font-family="sans-serif" font-size="14" fill="#30353b">{html.escape(method)}</text>')
        for column, metric in enumerate(METRICS):
            value = float(metrics[metric])
            x = left + column * cell_width + 8
            items.append(f'  <rect x="{x}" y="{y + 7}" width="{cell_width - 16}" height="{row_height - 14}" rx="5" fill="{_heat_color(value)}"/>')
            text_color = "#ffffff" if value >= 0.55 else "#20242a"
            items.append(f'  <text x="{x + (cell_width - 16) / 2:.1f}" y="{y + 33}" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="600" fill="{text_color}">{100.0 * value:.2f}</text>')
    items.append('</svg>')
    return "\n".join(items) + "\n"


def _ours_metrics(dataset: dict[str, Any]) -> dict[str, float]:
    return next(row["metrics"] for row in dataset["primary_results"] if row["role"] == "ours")


def render_ablation_svg(payload: dict[str, Any]) -> str:
    if payload["status"] != "final":
        raise ValueError("figures require an approved final result artifact")
    rows: list[tuple[str, str, float]] = []
    for dataset_key in DATASET_ORDER:
        dataset = payload["datasets"][dataset_key]
        reference = float(_ours_metrics(dataset)["macro_f1"])
        for ablation in dataset["ablations"]:
            delta = float(ablation["metrics"]["macro_f1"]) - reference
            rows.append((dataset["display_name"], ablation["label"], delta))

    left = 470
    right = 120
    width = 1280
    top = 115
    row_height = 40
    height = top + len(rows) * row_height + 75
    chart_width = width - left - right
    bound = max(0.01, max(abs(delta) for _, _, delta in rows)) * 1.15
    zero = left + chart_width / 2
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">Pre-specified matched ablation effects</title>',
        '  <desc id="desc">Descriptive Macro-F1 percentage-point change for each pre-specified matched ablation relative to the full CERD configuration.</desc>',
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        '  <text x="28" y="40" font-family="sans-serif" font-size="27" font-weight="700" fill="#20242a">Pre-specified matched ablations</text>',
        '  <text x="28" y="70" font-family="sans-serif" font-size="15" fill="#555b63">Macro-F1 change from full CERD (percentage points)</text>',
        f'  <line x1="{zero:.1f}" y1="95" x2="{zero:.1f}" y2="{height - 45}" stroke="#4c525a" stroke-width="1.5"/>',
    ]
    previous_dataset = None
    for index, (dataset, label, delta) in enumerate(rows):
        y = top + index * row_height
        if index % 2 == 0:
            items.append(f'  <rect x="20" y="{y - 7}" width="{width - 40}" height="{row_height}" fill="#f7f8fa"/>')
        dataset_text = dataset if dataset != previous_dataset else ""
        previous_dataset = dataset
        items.append(f'  <text x="30" y="{y + 17}" font-family="sans-serif" font-size="14" font-weight="700" fill="#30353b">{html.escape(dataset_text)}</text>')
        items.append(f'  <text x="105" y="{y + 17}" font-family="sans-serif" font-size="13" fill="#30353b">{html.escape(label)}</text>')
        extent = abs(delta) / bound * (chart_width / 2)
        x = zero if delta >= 0 else zero - extent
        color = "#2e7d32" if delta >= 0 else "#b3261e"
        items.append(f'  <rect x="{x:.1f}" y="{y + 1}" width="{max(extent, 1):.1f}" height="20" rx="2" fill="{color}"/>')
        label_x = zero + extent + 8 if delta >= 0 else zero - extent - 8
        anchor = "start" if delta >= 0 else "end"
        items.append(f'  <text x="{label_x:.1f}" y="{y + 17}" text-anchor="{anchor}" font-family="sans-serif" font-size="13" fill="#30353b">{100.0 * delta:+.2f}</text>')
    items.append('</svg>')
    return "\n".join(items) + "\n"


def _stacked_bar(
    x: float,
    y: float,
    width: float,
    height: float,
    values: list[float],
    colors: list[str],
) -> list[str]:
    items: list[str] = []
    offset = x
    for value, color in zip(values, colors):
        segment = width * float(value)
        items.append(f'  <rect x="{offset:.1f}" y="{y:.1f}" width="{segment:.1f}" height="{height:.1f}" fill="{color}"/>')
        if segment >= 42:
            items.append(f'  <text x="{offset + segment / 2:.1f}" y="{y + height / 2 + 5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#ffffff">{100.0 * float(value):.1f}</text>')
        offset += segment
    return items


def render_decision_allocation_svg(payload: dict[str, Any]) -> str:
    if payload["status"] != "final":
        raise ValueError("figures require an approved final result artifact")
    width = 1320
    height = 760
    panel_width = 610
    modality_colors = ["#2f6fbb", "#4c9f70", "#e59f3a", "#9a65b7"]
    branch_colors = ["#355c9a", "#d17a22", "#6b8e23"]
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">Aggregate decision allocation and branch mass</title>',
        '  <desc id="desc">Two-by-two comparison of complete and incomplete input routing summaries for ADNI and ABCD.</desc>',
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        '  <text x="28" y="40" font-family="sans-serif" font-size="27" font-weight="700" fill="#20242a">Aggregate decision allocation</text>',
        '  <text x="28" y="70" font-family="sans-serif" font-size="15" fill="#555b63">Descriptive routing mass; not causal importance. Values are percentages.</text>',
    ]
    for column, dataset_key in enumerate(DATASET_ORDER):
        dataset = payload["datasets"][dataset_key]
        interp = dataset["interpretability"]
        panel_x = 35 + column * 650
        items.append(f'  <rect x="{panel_x}" y="95" width="{panel_width}" height="620" rx="10" fill="#f8f9fb" stroke="#c4c9d0"/>')
        items.append(f'  <text x="{panel_x + 20}" y="128" font-family="sans-serif" font-size="21" font-weight="700" fill="#20242a">{html.escape(dataset["display_name"])}</text>')
        items.append(f'  <text x="{panel_x + 20}" y="157" font-family="sans-serif" font-size="16" font-weight="600" fill="#30353b">Modality decision allocation</text>')
        for index, condition_name in enumerate(("complete", "incomplete")):
            condition = interp["conditions"][condition_name]
            y = 185 + index * 66
            label = f"{condition_name.capitalize()} (n={condition['subjects']})"
            items.append(f'  <text x="{panel_x + 20}" y="{y + 18}" font-family="sans-serif" font-size="13" fill="#30353b">{html.escape(label)}</text>')
            items.extend(_stacked_bar(panel_x + 165, y, 410, 28, condition["decision_allocation"], modality_colors))
        legend_y = 326
        for index, (name, color) in enumerate(zip(interp["modality_names"], modality_colors)):
            x = panel_x + 20 + (index % 2) * 290
            y = legend_y + (index // 2) * 28
            items.append(f'  <rect x="{x}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
            items.append(f'  <text x="{x + 21}" y="{y}" font-family="sans-serif" font-size="13" fill="#30353b">{html.escape(name)}</text>')

        items.append(f'  <text x="{panel_x + 20}" y="425" font-family="sans-serif" font-size="16" font-weight="600" fill="#30353b">Grouped branch mixture mass</text>')
        for index, condition_name in enumerate(("complete", "incomplete")):
            condition = interp["conditions"][condition_name]
            values = [condition["branch_mass"][key] for key in BRANCH_GROUPS]
            y = 452 + index * 66
            label = condition_name.capitalize()
            items.append(f'  <text x="{panel_x + 20}" y="{y + 18}" font-family="sans-serif" font-size="13" fill="#30353b">{html.escape(label)}</text>')
            items.extend(_stacked_bar(panel_x + 165, y, 410, 28, values, branch_colors))
        for index, (name, color) in enumerate(zip(BRANCH_GROUPS, branch_colors)):
            x = panel_x + 20 + index * 185
            y = 603
            items.append(f'  <rect x="{x}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
            items.append(f'  <text x="{x + 21}" y="{y}" font-family="sans-serif" font-size="13" fill="#30353b">{html.escape(name)}</text>')
        items.append(f'  <text x="{panel_x + 20}" y="680" font-family="sans-serif" font-size="12" fill="#656b73">Source: {html.escape(interp["source"])}</text>')
    items.append('</svg>')
    return "\n".join(items) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="aggregate JSON artifact")
    parser.add_argument("--readme", type=Path, default=DEFAULT_README, help="README containing result markers")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES, help="SVG output directory")
    parser.add_argument("--check", action="store_true", help="validate and fail if generated files are stale")
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="reject a placeholder artifact; use this in the public-release gate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = load_unique_json(args.input)
        validate_payload(payload)
        if args.require_final and payload["status"] != "final":
            raise ValueError("--require-final rejects status other than 'final'")
        readme_text = args.readme.read_text(encoding="utf-8")
        block = render_markdown(payload)
        outputs = {args.readme: replace_generated_block(readme_text, block)}
        figure_paths = tuple(
            args.figures_dir / name
            for name in ("common6.svg", "ablations.svg", "decision_allocation.svg")
        )
        if payload["status"] == "final":
            outputs.update(
                {
                    figure_paths[0]: render_common6_svg(payload),
                    figure_paths[1]: render_ablation_svg(payload),
                    figure_paths[2]: render_decision_allocation_svg(payload),
                }
            )
        if args.check:
            stale = [path for path, expected in outputs.items() if not path.exists() or path.read_text(encoding="utf-8") != expected]
            if payload["status"] != "final":
                stale.extend(path for path in figure_paths if path.exists())
            if stale:
                raise ValueError("stale generated files: " + ", ".join(str(path) for path in stale))
        else:
            if payload["status"] == "final":
                args.figures_dir.mkdir(parents=True, exist_ok=True)
            for path, content in outputs.items():
                path.write_text(content, encoding="utf-8")
        print(f"validated {args.input} ({payload['status']})")
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
