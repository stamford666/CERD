#!/usr/bin/env python3
"""Validate an ABCD manifest before launching expensive baseline jobs.

The generic adapter validation intentionally supports many ABCD tasks. The
ADHD status3 cohort has a stronger, release-facing contract, so this entrypoint
also performs a read-only audit of the materialized files whenever
``provenance.task == "adhd_status3"``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from multimodal_data import (
    resolve_abcd_modalities,
    validate_abcd_manifest as validate_manifest_schema,
)


STATUS3_TASK = "adhd_status3"
SPLIT_NAMES = ("training", "validation", "testing")
FAMILY_SOURCE = "genetic/gn_y_genrel.parquet"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix in {".tsv", ".txt"} else ","
        frame = pd.read_csv(path, sep=separator, low_memory=False)
        if columns is not None:
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            frame = frame[columns]
        return frame
    raise ValueError(f"Unsupported table format: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_ids(values: pd.Series | list[Any], context: str) -> list[str]:
    series = pd.Series(values, dtype="object")
    if series.isna().any():
        raise ValueError(f"{context} contains null participant IDs")
    normalized = series.astype(str)
    if normalized.str.strip().eq("").any():
        raise ValueError(f"{context} contains empty participant IDs")
    return normalized.tolist()


def _duplicates(values: list[str]) -> list[str]:
    series = pd.Series(values, dtype="string")
    return sorted(series[series.duplicated(keep=False)].unique().tolist())


def _preview(values: set[str] | list[str], limit: int = 10) -> str:
    return ", ".join(sorted(values)[:limit])


def _parse_keep_mask(values: pd.Series, context: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{context} contains missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        if not values.isin([0, 1]).all():
            raise ValueError(f"{context} must contain only boolean or 0/1 values")
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "0", "1"]).all():
        raise ValueError(f"{context} must contain only boolean or 0/1 values")
    return normalized.isin(["true", "1"])


def _validate_source_hashes(manifest: dict[str, Any]) -> tuple[Path, int]:
    provenance = manifest.get("provenance", {})
    root_value = provenance.get("abcd_root")
    if not root_value:
        raise ValueError("status3 provenance.abcd_root is required")
    abcd_root = Path(str(root_value)).expanduser().resolve()
    hashes = provenance.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("status3 provenance.source_sha256 must be a non-empty mapping")

    target_sources = provenance.get("source_target_tables", [])
    if not isinstance(target_sources, list) or not target_sources:
        raise ValueError("status3 provenance.source_target_tables must be non-empty")
    required = {str(source) for source in target_sources} | {FAMILY_SOURCE}
    required |= {
        str(Path(source).with_suffix(".json"))
        for source in target_sources
        if Path(source).suffix.lower() == ".parquet"
    }
    missing_hashes = sorted(required - set(hashes))
    if missing_hashes:
        raise ValueError(
            "status3 provenance is missing required source hashes: "
            + ", ".join(missing_hashes)
        )

    for relative_path, expected_value in sorted(hashes.items()):
        expected = str(expected_value).lower()
        if not SHA256_PATTERN.fullmatch(expected):
            raise ValueError(f"Invalid SHA-256 value for source {relative_path!r}")
        source_path = _resolve(abcd_root, relative_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Hashed ABCD source is missing: {source_path}")
        actual = _sha256_file(source_path)
        if actual != expected:
            raise ValueError(
                f"Source SHA-256 mismatch for {relative_path}: "
                f"manifest={expected}, actual={actual}"
            )
    return abcd_root, len(hashes)


def _validate_labels_and_splits(
    manifest: dict[str, Any], manifest_base: Path
) -> tuple[set[str], dict[str, set[str]], dict[str, int]]:
    id_column = manifest["id_column"]
    label_spec = manifest["label"]
    label_path = _resolve(manifest_base, label_spec["path"])
    labels = _read_table(label_path)
    label_column = label_spec["column"]
    missing_columns = sorted({id_column, label_column} - set(labels.columns))
    if missing_columns:
        raise ValueError(f"Label table is missing columns: {missing_columns}")

    label_ids = _normalized_ids(labels[id_column], "label table")
    duplicate_label_ids = _duplicates(label_ids)
    if duplicate_label_ids:
        raise ValueError(
            "Label table contains duplicate participant IDs: "
            + _preview(duplicate_label_ids)
        )
    label_id_set = set(label_ids)
    if not label_id_set:
        raise ValueError("Label table is empty")

    class_values = label_spec.get("class_values")
    if class_values != [0, 1, 2]:
        raise ValueError(
            f"status3 label.class_values must be [0, 1, 2], got {class_values!r}"
        )
    numeric_labels = pd.to_numeric(labels[label_column], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin(class_values).all():
        invalid = labels.loc[
            numeric_labels.isna() | ~numeric_labels.isin(class_values), label_column
        ]
        raise ValueError(
            "Label table contains values outside [0, 1, 2]: "
            + _preview(set(invalid.astype(str)))
        )
    actual_counts = {
        str(int(key)): int(value)
        for key, value in numeric_labels.value_counts().sort_index().items()
    }
    if set(actual_counts) != {"0", "1", "2"}:
        raise ValueError(f"status3 label table must contain all three classes: {actual_counts}")

    endpoint = manifest.get("provenance", {}).get("adhd_endpoint", {})
    declared_counts = endpoint.get("class_counts")
    if declared_counts is not None:
        normalized_declared = {
            str(key): int(value) for key, value in declared_counts.items()
        }
        if normalized_declared != actual_counts:
            raise ValueError(
                "adhd_endpoint.class_counts does not match the label table: "
                f"declared={normalized_declared}, actual={actual_counts}"
            )
    declared_total = endpoint.get("eligible_participants")
    if declared_total is not None and int(declared_total) != len(label_ids):
        raise ValueError(
            "adhd_endpoint.eligible_participants does not match the label table: "
            f"declared={declared_total}, actual={len(label_ids)}"
        )

    split_path = _resolve(manifest_base, manifest["splits"])
    with split_path.open(encoding="utf-8") as handle:
        split_payload = json.load(handle)
    if not isinstance(split_payload, dict):
        raise ValueError("Split file must contain a JSON object")
    missing_splits = set(SPLIT_NAMES) - set(split_payload)
    extra_splits = set(split_payload) - set(SPLIT_NAMES)
    if missing_splits or extra_splits:
        raise ValueError(
            "Split file must contain exactly training/validation/testing; "
            f"missing={sorted(missing_splits)}, extra={sorted(extra_splits)}"
        )

    split_sets: dict[str, set[str]] = {}
    split_sizes: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        raw_ids = split_payload[split_name]
        if not isinstance(raw_ids, list):
            raise ValueError(f"{split_name} split must be a JSON list")
        ids = _normalized_ids(raw_ids, f"{split_name} split")
        duplicate_ids = _duplicates(ids)
        if duplicate_ids:
            raise ValueError(
                f"{split_name} split contains duplicate participant IDs: "
                + _preview(duplicate_ids)
            )
        split_sets[split_name] = set(ids)
        split_sizes[split_name] = len(ids)

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(
                    f"Participant IDs overlap between {left} and {right}: "
                    + _preview(overlap)
                )
    assigned_ids = set().union(*(split_sets[name] for name in SPLIT_NAMES))
    unknown_ids = assigned_ids - label_id_set
    if unknown_ids:
        raise ValueError(
            "Split file contains IDs absent from the label table: "
            + _preview(unknown_ids)
        )
    unassigned_ids = label_id_set - assigned_ids
    if unassigned_ids:
        raise ValueError(
            "Label table contains IDs absent from all splits: "
            + _preview(unassigned_ids)
        )
    return label_id_set, split_sets, split_sizes


def _validate_family_isolation(
    abcd_root: Path,
    id_column: str,
    label_ids: set[str],
    split_sets: dict[str, set[str]],
) -> int:
    family_path = _resolve(abcd_root, FAMILY_SOURCE)
    family_column = "gn_y_genrel_id__fam"
    family = _read_table(family_path, columns=[id_column, family_column])
    family_ids = _normalized_ids(family[id_column], "family source")
    duplicate_family_ids = _duplicates(family_ids)
    if duplicate_family_ids:
        raise ValueError(
            "Family source contains duplicate participant IDs: "
            + _preview(duplicate_family_ids)
        )
    family = family.assign(**{id_column: family_ids}).set_index(id_column)
    family_by_participant: dict[str, str] = {}
    for participant_id in label_ids:
        if participant_id not in family.index or pd.isna(
            family.at[participant_id, family_column]
        ):
            # This reproduces the builder's documented split behavior.
            family_by_participant[participant_id] = f"missing_{participant_id}"
        else:
            family_by_participant[participant_id] = str(
                family.at[participant_id, family_column]
            )

    family_sets = {
        split_name: {
            family_by_participant[participant_id]
            for participant_id in split_sets[split_name]
        }
        for split_name in SPLIT_NAMES
    }
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = family_sets[left] & family_sets[right]
            if overlap:
                raise ValueError(
                    f"Genetic families overlap between {left} and {right}: "
                    + _preview(overlap)
                )
    return len(set(family_by_participant.values()))


def _forbidden_predictor_columns(manifest: dict[str, Any]) -> set[str]:
    provenance = manifest.get("provenance", {})
    endpoint = provenance.get("adhd_endpoint", {})
    forbidden = {
        manifest["label"]["column"],
        "presentation_episode",
        "inattentive_symptom_count",
        "hyperactive_impulsive_symptom_count",
        "current_inattentive_symptom_count",
        "current_hyperactive_impulsive_symptom_count",
        *provenance.get("excluded_predictor_feature_columns", []),
        *endpoint.get("status_columns", []),
        *endpoint.get("full_diagnosis_columns", []),
        *endpoint.get("present_inattentive_columns", []),
        *endpoint.get("present_hyperactive_impulsive_columns", []),
        *endpoint.get("past_inattentive_columns", []),
        *endpoint.get("past_hyperactive_impulsive_columns", []),
        *endpoint.get("current_inattentive_columns", []),
        *endpoint.get("current_hyperactive_impulsive_columns", []),
    }
    for modality in manifest["modalities"]:
        forbidden.update(modality.get("excluded_feature_columns", []))
    return {str(column) for column in forbidden}


def _validate_materialized_modalities(
    manifest: dict[str, Any], manifest_base: Path, label_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    id_column = manifest["id_column"]
    forbidden = _forbidden_predictor_columns(manifest)
    availability: dict[str, set[str]] = {}
    row_counts: dict[str, int] = {}
    for modality in manifest["modalities"]:
        code = str(modality["code"]).upper()
        path = _resolve(manifest_base, modality["path"])
        frame = _read_table(path)
        modality_id_column = modality.get("id_column", id_column)
        if modality_id_column not in frame.columns:
            raise ValueError(
                f"Materialized modality {code} has no ID column {modality_id_column!r}"
            )
        if frame.columns.duplicated().any():
            duplicated_columns = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(
                f"Materialized modality {code} has duplicate columns: "
                + _preview(duplicated_columns)
            )
        modality_ids = _normalized_ids(
            frame[modality_id_column], f"materialized modality {code}"
        )
        duplicate_ids = _duplicates(modality_ids)
        if duplicate_ids:
            raise ValueError(
                f"Materialized modality {code} contains duplicate participant IDs: "
                + _preview(duplicate_ids)
            )
        modality_id_set = set(modality_ids)
        unknown_ids = modality_id_set - label_ids
        if unknown_ids:
            raise ValueError(
                f"Materialized modality {code} contains unknown participant IDs: "
                + _preview(unknown_ids)
            )
        predictor_columns = set(map(str, frame.columns)) - {str(modality_id_column)}
        if not predictor_columns:
            raise ValueError(f"Materialized modality {code} has no predictor columns")
        leaked = predictor_columns & forbidden
        leaked |= {
            column
            for column in predictor_columns
            if column.lower().startswith("mh_p_ksads__adhd")
        }
        if leaked:
            raise ValueError(
                f"Materialized modality {code} contains forbidden ADHD/target predictors: "
                + _preview(leaked)
            )
        availability[code] = modality_id_set
        row_counts[code] = len(modality_ids)

    no_materialized_modality = {
        participant_id
        for participant_id in label_ids
        if not any(participant_id in ids for ids in availability.values())
    }
    if no_materialized_modality:
        raise ValueError(
            "Labeled participants have no materialized modality: "
            + _preview(no_materialized_modality)
        )
    return availability, row_counts


def _validate_missingness(
    manifest: dict[str, Any],
    manifest_base: Path,
    label_ids: set[str],
    availability: dict[str, set[str]],
) -> int:
    missingness_spec = manifest.get("missingness")
    if not missingness_spec:
        return min(
            sum(participant_id in ids for ids in availability.values())
            for participant_id in label_ids
        )

    id_column = manifest["id_column"]
    path = _resolve(manifest_base, missingness_spec["path"])
    missingness = _read_table(path)
    if id_column not in missingness.columns:
        raise ValueError(f"Missingness table has no ID column {id_column!r}")
    missingness_ids = _normalized_ids(missingness[id_column], "missingness table")
    duplicate_ids = _duplicates(missingness_ids)
    if duplicate_ids:
        raise ValueError(
            "Missingness table contains duplicate participant IDs: "
            + _preview(duplicate_ids)
        )
    missingness_id_set = set(missingness_ids)
    unknown_ids = missingness_id_set - label_ids
    if unknown_ids:
        raise ValueError(
            "Missingness table contains unknown participant IDs: "
            + _preview(unknown_ids)
        )
    absent_ids = label_ids - missingness_id_set
    if absent_ids:
        raise ValueError(
            "Missingness table omits labeled participant IDs: " + _preview(absent_ids)
        )

    missingness = missingness.assign(**{id_column: missingness_ids}).set_index(id_column)
    column_map = missingness_spec.get("columns", {})
    keep_masks: dict[str, pd.Series] = {}
    for code in availability:
        column = column_map.get(code, code)
        if column not in missingness.columns:
            raise ValueError(f"Missingness table has no keep-mask column for modality {code}")
        keep_masks[code] = _parse_keep_mask(
            missingness[column], f"Missingness keep-mask column {column!r}"
        )

    mask_only_counts = sum(mask.astype(int) for mask in keep_masks.values())
    zero_keep_ids = set(mask_only_counts.index[mask_only_counts.eq(0)])
    if zero_keep_ids:
        raise ValueError(
            "Missingness mask leaves zero kept modalities for participants: "
            + _preview(zero_keep_ids)
        )

    effective_counts = pd.Series(0, index=missingness.index, dtype="int64")
    for code, keep_mask in keep_masks.items():
        physically_available = pd.Series(
            missingness.index.isin(availability[code]), index=missingness.index
        )
        effective_counts += (keep_mask & physically_available).astype(int)
    zero_effective_ids = set(effective_counts.index[effective_counts.eq(0)])
    if zero_effective_ids:
        raise ValueError(
            "Missingness plus physical coverage leaves zero observed modalities for "
            "participants: "
            + _preview(zero_effective_ids)
        )
    return int(effective_counts.min())


def strict_validate_status3_manifest(
    dataset_manifest: str | Path, modality: str = "IGCB"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and strictly audit one materialized ADHD status3 manifest.

    No file is created or modified. The returned report contains compact counts
    suitable for logs and tests.
    """

    manifest_path = Path(dataset_manifest).expanduser().resolve()
    namespace = SimpleNamespace(dataset_manifest=str(manifest_path), modality=modality)
    manifest = validate_manifest_schema(namespace)
    if manifest.get("provenance", {}).get("task") != STATUS3_TASK:
        raise ValueError(
            "strict status3 validation requires provenance.task='adhd_status3'"
        )
    # Resolve the requested modality codes as part of the same launch contract.
    resolve_abcd_modalities(namespace)
    manifest_base = manifest_path.parent
    abcd_root, hash_count = _validate_source_hashes(manifest)
    label_ids, split_sets, split_sizes = _validate_labels_and_splits(
        manifest, manifest_base
    )
    family_count = _validate_family_isolation(
        abcd_root, manifest["id_column"], label_ids, split_sets
    )
    availability, modality_rows = _validate_materialized_modalities(
        manifest, manifest_base, label_ids
    )
    minimum_observed = _validate_missingness(
        manifest, manifest_base, label_ids, availability
    )
    report = {
        "participants": len(label_ids),
        "split_sizes": split_sizes,
        "families": family_count,
        "source_hashes_verified": hash_count,
        "modality_rows": modality_rows,
        "minimum_observed_modalities": minimum_observed,
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--modality", default="IGNP")
    args = parser.parse_args()
    namespace = SimpleNamespace(
        dataset_manifest=args.dataset_manifest, modality=args.modality
    )
    manifest = validate_manifest_schema(namespace)
    selected = resolve_abcd_modalities(namespace)
    report = None
    if manifest.get("provenance", {}).get("task") == STATUS3_TASK:
        manifest, report = strict_validate_status3_manifest(
            args.dataset_manifest, args.modality
        )
    print(f"dataset={manifest.get('dataset', 'abcd')}")
    print(f"selected_modalities={selected}")
    print(f"label={manifest['label']['path']}:{manifest['label']['column']}")
    print(f"splits={manifest['splits']}")
    if report is not None:
        print(f"status3_strict_audit={json.dumps(report, sort_keys=True)}")
    print("manifest_status=ok")


if __name__ == "__main__":
    main()
