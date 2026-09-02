#!/usr/bin/env python3
"""Build one aggregate public result from explicit private OOF inputs.

This command never discovers campaigns. It accepts one exact manifest, verifies
the hashes of two row-aligned NPZ inputs, computes the frozen aggregate protocol,
passes the result through the public renderer validator, and only then performs
an atomic write. It does not render README tables or figures.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cerd.metrics import metric_bundle  # noqa: E402


RENDERER_PATH = ROOT / "scripts" / "render_release_results.py"
RENDERER_SPEC = importlib.util.spec_from_file_location(
    "cerd_release_renderer",
    RENDERER_PATH,
)
if RENDERER_SPEC is None or RENDERER_SPEC.loader is None:
    raise RuntimeError("could not load the public result validator")
RENDERER = importlib.util.module_from_spec(RENDERER_SPEC)
RENDERER_SPEC.loader.exec_module(RENDERER)

PRIVATE_MANIFEST_SCHEMA = "cerd-private-final-builder-v1"
PRIVATE_OOF_SCHEMA = "cerd-private-row-aligned-oof-v1"
PRIVATE_EXECUTION_RECEIPT_SCHEMA = "cerd-private-execution-receipt-v1"
PRIVATE_ANALYSIS_RECEIPT_SCHEMA = "cerd-private-analysis-receipt-v1"
PRIVATE_AGGREGATION_RECEIPT_SCHEMA = "cerd-private-aggregation-receipt-v1"
CLASS_COLUMNS = (0, 1, 2)
PRIMARY_INPUT_IDS = ("full", "comparator")
SEED_ENSEMBLE_RULE = "equal-weight arithmetic mean in manifest seed order"
INTERPRETABILITY_AGGREGATION_UNIT = RENDERER.INTERPRETABILITY_AGGREGATION_UNIT
INTERPRETABILITY_AGGREGATION_METHOD = RENDERER.INTERPRETABILITY_AGGREGATION_METHOD
INTERPRETABILITY_DEFINITIONS = RENDERER.INTERPRETABILITY_DEFINITIONS
PROBABILITY_TOLERANCE = 1e-6
MAX_PUBLIC_BYTES = 200_000
MAX_BOUND_FILE_BYTES = 10_000_000
MAX_PRIVATE_NPZ_BYTES = 512_000_000
MAX_PRIVATE_NPZ_UNCOMPRESSED_BYTES = 1_000_000_000
STATISTIC_CHUNK_SIZE = 256
BOOTSTRAP_QUANTILE = 0.05
BOOTSTRAP_QUANTILE_METHOD = "linear"

PRIVATE_ROOT_KEYS = {"schema", "generated_at", "datasets"}
PRIVATE_DATASET_KEYS = {
    "contract",
    "input_npz",
    "input_sha256",
    "class_columns",
    "seed_ids",
    "primary",
    "ablations",
    "analysis_receipt_path",
    "analysis_receipt_sha256",
    "interpretability",
}
PRIVATE_CONTRACT_KEYS = {
    "subjects",
    "folds",
    "resampling_unit",
    "resampling_units",
}
PRIVATE_PRIMARY_KEYS = {
    "input_id",
    "method",
    "role",
    "configuration_id",
    "configuration_path",
    "configuration_sha256",
    "execution_receipt_path",
    "execution_receipt_sha256",
}
PRIVATE_ABLATION_KEYS = {
    "id",
    "configuration_id",
    "configuration_path",
    "configuration_sha256",
    "execution_receipt_path",
    "execution_receipt_sha256",
}
PRIVATE_INTERPRETABILITY_KEYS = {
    "aggregation_unit",
    "aggregation_method",
    "aggregation_receipt_path",
    "aggregation_receipt_sha256",
}
PRIVATE_CONDITION_KEYS = {"subjects", "decision_allocation", "branch_mass"}
PRIVATE_EXECUTION_RECEIPT_KEYS = {
    "schema",
    "dataset",
    "arm_id",
    "configuration_sha256",
    "oof_input_sha256",
    "probability_array",
    "seed_ids",
    "class_columns",
    "seed_ensemble",
    "subjects",
    "folds",
    "resampling_unit",
    "resampling_units",
    "status",
}
PRIVATE_ANALYSIS_RECEIPT_KEYS = {
    "schema",
    "dataset",
    "oof_input_sha256",
    "class_columns",
    "seed_ids",
    "seed_ensemble",
    "resampling_unit",
    "n_units",
    "comparison",
}
PRIVATE_ANALYSIS_COMPARISON_KEYS = {
    "metric",
    "ours_input_id",
    "comparator_input_id",
    "alternative",
    "swap_draws",
    "swap_rng_seed",
    "bootstrap_draws",
    "bootstrap_rng_seed",
    "bootstrap_quantile",
    "bootstrap_quantile_method",
    "observed_delta",
    "p_value",
    "bootstrap_lower_bound",
}
PRIVATE_AGGREGATION_RECEIPT_KEYS = {
    "schema",
    "dataset",
    "oof_input_sha256",
    "aggregation_unit",
    "aggregation_method",
    "condition_design",
    "subjects",
    "conditions",
}
PRIVATE_ARRAY_KEYS = {
    "format",
    "dataset",
    "row_ids",
    "cluster_ids",
    "fold_ids",
    "labels",
    "seed_ids",
    "class_columns",
}
FORBIDDEN_PUBLIC_KEYS = {
    "format",
    "dataset",
    "row_ids",
    "cluster_ids",
    "fold_ids",
    "labels",
    "probabilities",
    "seed_ids",
    "class_columns",
    "input_npz",
    "input_sha256",
    "input_id",
    "configuration_path",
    "execution_receipt_path",
    "analysis_receipt_path",
    "aggregation_receipt_path",
}


@dataclass(frozen=True)
class PrivateBuildResult:
    """Private computation result awaiting the public validation phase."""

    payload: dict[str, Any]
    private_identifiers: frozenset[str]
    private_paths: frozenset[str]


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _require_exact_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        _fail(path, f"keys must be exactly {sorted(expected)}")
    return value


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or RENDERER.SHA256_HEX.fullmatch(value) is None:
        _fail(path, "must be a lowercase 64-character SHA-256 digest")
    return value


def _require_int_list(value: Any, path: str, *, minimum_length: int) -> list[int]:
    if not isinstance(value, list) or len(value) < minimum_length:
        _fail(path, f"must be a list containing at least {minimum_length} integers")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        _fail(path, "must contain integers only")
    if len(value) != len(set(value)):
        _fail(path, "must not contain duplicate integers")
    return value


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_private_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError("private manifest could not be read") from error
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("private manifest is not unique-key UTF-8 JSON") from error
    return payload, raw


def _decode_private_json(raw: bytes, path: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: must be unique-key UTF-8 JSON") from error
    if not isinstance(payload, dict):
        _fail(path, "must be a JSON object")
    return payload


def _resolve_private_input(manifest_path: Path, value: Any, path: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be one explicit private file path")
    private_path = Path(value).expanduser()
    if not private_path.is_absolute():
        private_path = manifest_path.parent / private_path
    return private_path


def _read_and_verify_file(
    manifest_path: Path,
    path_value: Any,
    expected_sha256: str,
    path: str,
) -> tuple[bytes, Path]:
    private_path = _resolve_private_input(manifest_path, path_value, path)
    try:
        if private_path.stat().st_size > MAX_BOUND_FILE_BYTES:
            _fail(path, "private provenance file exceeds the size ceiling")
        raw = private_path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}: private file could not be read") from error
    if not raw:
        _fail(path, "private file must not be empty")
    if len(raw) > MAX_BOUND_FILE_BYTES:
        _fail(path, "private provenance file exceeds the size ceiling")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail(path.replace("_path", "_sha256"), "does not match the exact file bytes")
    return raw, private_path


def _read_hashed_npz(
    path: Path,
    expected_sha256: str,
    expected_keys: set[str],
    dataset_path: str,
) -> dict[str, np.ndarray]:
    try:
        if path.stat().st_size > MAX_PRIVATE_NPZ_BYTES:
            _fail(f"{dataset_path}.input_npz", "private NPZ exceeds the size ceiling")
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"{dataset_path}.input_npz: private input could not be read"
        ) from error
    if not raw:
        _fail(f"{dataset_path}.input_npz", "private NPZ must not be empty")
    if len(raw) > MAX_PRIVATE_NPZ_BYTES:
        _fail(f"{dataset_path}.input_npz", "private NPZ exceeds the size ceiling")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail(f"{dataset_path}.input_sha256", "does not match the exact NPZ bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zip_archive:
            expected_members = {f"{name}.npy" for name in expected_keys}
            members = zip_archive.infolist()
            member_names = [member.filename for member in members]
            if (
                len(member_names) != len(expected_members)
                or set(member_names) != expected_members
            ):
                _fail(
                    f"{dataset_path}.input_npz",
                    f"archive members must be exactly {sorted(expected_members)}",
                )
            if (
                sum(member.file_size for member in members)
                > MAX_PRIVATE_NPZ_UNCOMPRESSED_BYTES
            ):
                _fail(
                    f"{dataset_path}.input_npz",
                    "private NPZ uncompressed arrays exceed the size ceiling",
                )
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if (
                len(archive.files) != len(expected_keys)
                or set(archive.files) != expected_keys
            ):
                _fail(
                    f"{dataset_path}.input_npz",
                    f"array keys must be exactly {sorted(expected_keys)}",
                )
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"{dataset_path}.input_npz: invalid non-pickled NPZ"
        ) from error
    for name, array in arrays.items():
        if array.dtype.hasobject:
            _fail(f"{dataset_path}.input_npz.{name}", "object arrays are forbidden")
    return arrays


def _validate_identifier_vector(
    value: np.ndarray,
    expected_rows: int,
    path: str,
) -> np.ndarray:
    if value.ndim != 1 or value.shape[0] != expected_rows:
        _fail(path, f"must have shape [{expected_rows}]")
    if value.dtype.kind not in "iuUS":
        _fail(path, "must contain integer or string identifiers")
    if value.dtype.kind in "US" and any(
        not str(item).strip() for item in value.tolist()
    ):
        _fail(path, "must not contain empty identifiers")
    if np.unique(value).size != expected_rows:
        _fail(path, "must contain one unique identifier per public subject row")
    return value


def _validate_private_arrays(
    dataset_key: str,
    arrays: dict[str, np.ndarray],
    seed_ids: list[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    frozenset[str],
]:
    contract = RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]
    dataset_path = f"datasets.{dataset_key}"
    subjects = int(contract["subjects"])
    folds = int(contract["folds"])
    expected_units = int(contract["resampling_units"])

    for field, expected in (("format", PRIVATE_OOF_SCHEMA), ("dataset", dataset_key)):
        value = arrays[field]
        if value.ndim != 0 or value.dtype.kind not in "US" or value.item() != expected:
            _fail(
                f"{dataset_path}.input_npz.{field}",
                f"must be the scalar string {expected!r}",
            )

    row_ids = _validate_identifier_vector(
        arrays["row_ids"],
        subjects,
        f"{dataset_path}.input_npz.row_ids",
    )
    cluster_ids = arrays["cluster_ids"]
    if cluster_ids.ndim != 1 or cluster_ids.shape[0] != subjects:
        _fail(
            f"{dataset_path}.input_npz.cluster_ids",
            f"must have shape [{subjects}]",
        )
    if cluster_ids.dtype.kind not in "iuUS":
        _fail(
            f"{dataset_path}.input_npz.cluster_ids",
            "must contain integer or string cluster identifiers",
        )
    if cluster_ids.dtype.kind in "US" and any(
        not str(item).strip() for item in cluster_ids.tolist()
    ):
        _fail(
            f"{dataset_path}.input_npz.cluster_ids",
            "must not contain empty identifiers",
        )
    if np.unique(cluster_ids).size != expected_units:
        _fail(
            f"{dataset_path}.input_npz.cluster_ids",
            f"must contain exactly {expected_units} unique resampling units",
        )

    labels = arrays["labels"]
    if labels.ndim != 1 or labels.shape[0] != subjects or labels.dtype.kind not in "iu":
        _fail(
            f"{dataset_path}.input_npz.labels",
            f"must be an integer vector with shape [{subjects}]",
        )
    labels = labels.astype(np.int64, copy=False)
    if set(np.unique(labels).tolist()) != set(CLASS_COLUMNS):
        _fail(
            f"{dataset_path}.input_npz.labels",
            f"must contain every fixed class exactly within {list(CLASS_COLUMNS)}",
        )

    fold_ids = arrays["fold_ids"]
    if (
        fold_ids.ndim != 1
        or fold_ids.shape[0] != subjects
        or fold_ids.dtype.kind not in "iu"
    ):
        _fail(
            f"{dataset_path}.input_npz.fold_ids",
            f"must be an integer vector with shape [{subjects}]",
        )
    fold_ids = fold_ids.astype(np.int64, copy=False)
    if set(np.unique(fold_ids).tolist()) != set(range(folds)):
        _fail(
            f"{dataset_path}.input_npz.fold_ids",
            f"must contain exactly the five OOF fold ids {list(range(folds))}",
        )
    _, cluster_inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_fold_min = np.full(expected_units, folds, dtype=np.int64)
    cluster_fold_max = np.full(expected_units, -1, dtype=np.int64)
    np.minimum.at(cluster_fold_min, cluster_inverse, fold_ids)
    np.maximum.at(cluster_fold_max, cluster_inverse, fold_ids)
    if not np.array_equal(cluster_fold_min, cluster_fold_max):
        _fail(
            f"{dataset_path}.input_npz",
            "every resampling unit must belong to exactly one OOF fold",
        )

    class_columns = arrays["class_columns"]
    if (
        class_columns.ndim != 1
        or class_columns.dtype.kind not in "iu"
        or class_columns.tolist() != list(CLASS_COLUMNS)
    ):
        _fail(
            f"{dataset_path}.input_npz.class_columns",
            f"must equal the fixed class columns {list(CLASS_COLUMNS)}",
        )
    array_seed_ids = arrays["seed_ids"]
    if (
        array_seed_ids.ndim != 1
        or array_seed_ids.dtype.kind not in "iu"
        or array_seed_ids.tolist() != seed_ids
    ):
        _fail(
            f"{dataset_path}.input_npz.seed_ids",
            "must exactly match the manifest seed order",
        )

    ensembles: dict[str, np.ndarray] = {}
    probability_keys = sorted(set(arrays) - PRIVATE_ARRAY_KEYS)
    for probability_key in probability_keys:
        probabilities = arrays[probability_key]
        expected_shape = (len(seed_ids), subjects, len(CLASS_COLUMNS))
        if probabilities.shape != expected_shape:
            _fail(
                f"{dataset_path}.input_npz.{probability_key}",
                f"must have shape {list(expected_shape)}",
            )
        if (
            probabilities.dtype.kind not in "fiu"
            or not np.isfinite(probabilities).all()
        ):
            _fail(
                f"{dataset_path}.input_npz.{probability_key}",
                "must contain finite numeric probabilities",
            )
        probabilities = probabilities.astype(np.float64, copy=False)
        if (probabilities < 0.0).any() or (probabilities > 1.0).any():
            _fail(
                f"{dataset_path}.input_npz.{probability_key}",
                "probabilities must lie in [0, 1]",
            )
        if not np.allclose(
            probabilities.sum(axis=2),
            1.0,
            rtol=0.0,
            atol=PROBABILITY_TOLERANCE,
        ):
            _fail(
                f"{dataset_path}.input_npz.{probability_key}",
                "every seed probability row must sum to one",
            )
        ensemble = probabilities.mean(axis=0, dtype=np.float64)
        if not np.allclose(
            ensemble.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=PROBABILITY_TOLERANCE,
        ):
            _fail(
                f"{dataset_path}.input_npz.{probability_key}",
                "equal-weight seed ensemble rows must sum to one",
            )
        ensembles[probability_key.removeprefix("probabilities_")] = ensemble

    private_identifiers = frozenset(
        [str(item) for item in row_ids.tolist()]
        + [str(item) for item in cluster_ids.tolist()]
    )
    return labels, fold_ids, cluster_ids, ensembles, private_identifiers


def _cluster_confusions(
    labels: np.ndarray,
    predictions: np.ndarray,
    cluster_ids: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    _, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_count = int(inverse.max()) + 1
    result = np.zeros((cluster_count, num_classes, num_classes), dtype=np.int64)
    flat_cells = labels * num_classes + predictions
    np.add.at(result.reshape(cluster_count, -1), (inverse, flat_cells), 1)
    return result


def _macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    confusions = np.asarray(confusions)
    true_positive = np.diagonal(confusions, axis1=-2, axis2=-1).astype(np.float64)
    actual = confusions.sum(axis=-1, dtype=np.float64)
    predicted = confusions.sum(axis=-2, dtype=np.float64)
    denominator = actual + predicted
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator > 0,
    )
    return class_f1.mean(axis=-1)


def _paired_cluster_swap(
    labels: np.ndarray,
    ours_probabilities: np.ndarray,
    comparator_probabilities: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    draws: int,
    seed: int,
    chunk_size: int = STATISTIC_CHUNK_SIZE,
) -> tuple[float, float]:
    """Return observed Macro-F1 difference and one-sided cluster-swap p-value."""

    if draws <= 0 or chunk_size <= 0:
        raise ValueError("swap draws and chunk size must be positive")
    num_classes = ours_probabilities.shape[1]
    ours_clusters = _cluster_confusions(
        labels,
        ours_probabilities.argmax(axis=1),
        cluster_ids,
        num_classes,
    )
    comparator_clusters = _cluster_confusions(
        labels,
        comparator_probabilities.argmax(axis=1),
        cluster_ids,
        num_classes,
    )
    ours_base = ours_clusters.sum(axis=0)
    comparator_base = comparator_clusters.sum(axis=0)
    observed = float(
        _macro_f1_from_confusions(ours_base)
        - _macro_f1_from_confusions(comparator_base)
    )
    cluster_shift = (comparator_clusters - ours_clusters).reshape(
        ours_clusters.shape[0],
        -1,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    exceedances = 0
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws - start)
        swaps = rng.integers(
            0,
            2,
            size=(count, ours_clusters.shape[0]),
            dtype=np.int64,
        )
        shifts = swaps @ cluster_shift
        ours_null = ours_base.reshape(1, -1) + shifts
        comparator_null = comparator_base.reshape(1, -1) - shifts
        null_differences = _macro_f1_from_confusions(
            ours_null.reshape(count, num_classes, num_classes)
        ) - _macro_f1_from_confusions(
            comparator_null.reshape(count, num_classes, num_classes)
        )
        exceedances += int(np.count_nonzero(null_differences >= observed))
    p_value = (exceedances + 1.0) / (draws + 1.0)
    return observed, float(p_value)


def _paired_cluster_bootstrap_lower(
    labels: np.ndarray,
    ours_probabilities: np.ndarray,
    comparator_probabilities: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    draws: int,
    seed: int,
    chunk_size: int = STATISTIC_CHUNK_SIZE,
) -> float:
    """Return the fixed one-sided 95% paired cluster-bootstrap lower bound."""

    if draws <= 0 or chunk_size <= 0:
        raise ValueError("bootstrap draws and chunk size must be positive")
    num_classes = ours_probabilities.shape[1]
    ours_clusters = _cluster_confusions(
        labels,
        ours_probabilities.argmax(axis=1),
        cluster_ids,
        num_classes,
    )
    comparator_clusters = _cluster_confusions(
        labels,
        comparator_probabilities.argmax(axis=1),
        cluster_ids,
        num_classes,
    )
    cluster_count = ours_clusters.shape[0]
    differences = np.empty(draws, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws - start)
        sampled = rng.integers(
            0,
            cluster_count,
            size=(count, cluster_count),
            dtype=np.int64,
        )
        ours_sample = ours_clusters[sampled].sum(axis=1)
        comparator_sample = comparator_clusters[sampled].sum(axis=1)
        differences[start : start + count] = _macro_f1_from_confusions(
            ours_sample
        ) - _macro_f1_from_confusions(comparator_sample)
    return float(
        np.quantile(
            differences,
            BOOTSTRAP_QUANTILE,
            method=BOOTSTRAP_QUANTILE_METHOD,
        )
    )


def _metric_bundle(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = metric_bundle(labels, probabilities)
    if tuple(metrics) != tuple(RENDERER.METRICS):
        raise RuntimeError("metric implementation did not return exactly common6")
    return {name: float(metrics[name]) for name in RENDERER.METRICS}


def _validate_execution_receipt(
    receipt: dict[str, Any],
    *,
    dataset_key: str,
    arm_id: str,
    configuration_sha256: str,
    input_sha256: str,
    seed_ids: list[int],
    path: str,
) -> None:
    _require_exact_keys(receipt, PRIVATE_EXECUTION_RECEIPT_KEYS, path)
    contract = RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]
    expected = {
        "schema": PRIVATE_EXECUTION_RECEIPT_SCHEMA,
        "dataset": dataset_key,
        "arm_id": arm_id,
        "configuration_sha256": configuration_sha256,
        "oof_input_sha256": input_sha256,
        "probability_array": f"probabilities_{arm_id}",
        "seed_ids": seed_ids,
        "class_columns": list(CLASS_COLUMNS),
        "seed_ensemble": SEED_ENSEMBLE_RULE,
        "subjects": contract["subjects"],
        "folds": contract["folds"],
        "resampling_unit": RENDERER.RESAMPLING_UNIT[dataset_key],
        "resampling_units": contract["resampling_units"],
        "status": "complete",
    }
    for field, expected_value in expected.items():
        if receipt[field] != expected_value:
            _fail(
                f"{path}.{field}", f"must equal the frozen binding {expected_value!r}"
            )


def _validate_analysis_receipt_metadata(
    receipt: dict[str, Any],
    *,
    dataset_key: str,
    input_sha256: str,
    seed_ids: list[int],
    path: str,
) -> None:
    _require_exact_keys(receipt, PRIVATE_ANALYSIS_RECEIPT_KEYS, path)
    expected = {
        "schema": PRIVATE_ANALYSIS_RECEIPT_SCHEMA,
        "dataset": dataset_key,
        "oof_input_sha256": input_sha256,
        "class_columns": list(CLASS_COLUMNS),
        "seed_ids": seed_ids,
        "seed_ensemble": SEED_ENSEMBLE_RULE,
        "resampling_unit": RENDERER.RESAMPLING_UNIT[dataset_key],
        "n_units": RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]["resampling_units"],
    }
    for field, expected_value in expected.items():
        if receipt[field] != expected_value:
            _fail(
                f"{path}.{field}", f"must equal the frozen binding {expected_value!r}"
            )

    comparison = _require_exact_keys(
        receipt["comparison"],
        PRIVATE_ANALYSIS_COMPARISON_KEYS,
        f"{path}.comparison",
    )
    expected_comparison = {
        "metric": "macro_f1",
        "ours_input_id": "full",
        "comparator_input_id": "comparator",
        "alternative": "greater",
        "swap_draws": RENDERER.SWAP_DRAWS,
        "swap_rng_seed": RENDERER.SWAP_RNG_SEED,
        "bootstrap_draws": RENDERER.BOOTSTRAP_DRAWS,
        "bootstrap_rng_seed": RENDERER.BOOTSTRAP_RNG_SEED,
        "bootstrap_quantile": BOOTSTRAP_QUANTILE,
        "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
    }
    for field, expected_value in expected_comparison.items():
        if comparison[field] != expected_value:
            _fail(
                f"{path}.comparison.{field}",
                f"must equal the frozen analysis value {expected_value!r}",
            )
    for field in ("observed_delta", "p_value", "bootstrap_lower_bound"):
        if not RENDERER._is_number(comparison[field]):
            _fail(f"{path}.comparison.{field}", "must be a finite number")


def _validate_analysis_receipt_results(
    receipt: dict[str, Any],
    *,
    observed_delta: float,
    p_value: float,
    bootstrap_lower_bound: float,
    path: str,
) -> None:
    expected = {
        "observed_delta": observed_delta,
        "p_value": p_value,
        "bootstrap_lower_bound": bootstrap_lower_bound,
    }
    comparison = receipt["comparison"]
    for field, expected_value in expected.items():
        if not math.isclose(
            float(comparison[field]),
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(
                f"{path}.comparison.{field}",
                "does not match the builder-recomputed analysis",
            )


def _validate_aggregation_receipt(
    receipt: dict[str, Any],
    *,
    dataset_key: str,
    input_sha256: str,
    path: str,
) -> dict[str, Any]:
    _require_exact_keys(receipt, PRIVATE_AGGREGATION_RECEIPT_KEYS, path)
    subjects = RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]["subjects"]
    expected = {
        "schema": PRIVATE_AGGREGATION_RECEIPT_SCHEMA,
        "dataset": dataset_key,
        "oof_input_sha256": input_sha256,
        "aggregation_unit": INTERPRETABILITY_AGGREGATION_UNIT,
        "aggregation_method": INTERPRETABILITY_AGGREGATION_METHOD,
        "condition_design": "natural_disjoint",
        "subjects": subjects,
    }
    for field, expected_value in expected.items():
        if receipt[field] != expected_value:
            _fail(
                f"{path}.{field}", f"must equal the frozen binding {expected_value!r}"
            )
    conditions = receipt["conditions"]
    if not isinstance(conditions, dict) or set(conditions) != {
        "complete",
        "incomplete",
    }:
        _fail(f"{path}.conditions", "must contain exactly complete and incomplete")
    for condition_name in ("complete", "incomplete"):
        condition_path = f"{path}.conditions.{condition_name}"
        condition = _require_exact_keys(
            conditions[condition_name],
            PRIVATE_CONDITION_KEYS,
            condition_path,
        )
        count = condition["subjects"]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < RENDERER.MIN_PUBLIC_CELL
        ):
            _fail(
                f"{condition_path}.subjects",
                f"must be an integer at least {RENDERER.MIN_PUBLIC_CELL}",
            )
        RENDERER._validate_probability_vector(
            condition["decision_allocation"],
            4,
            f"{condition_path}.decision_allocation",
            final=True,
        )
        RENDERER._validate_branch_mass(
            condition["branch_mass"],
            f"{condition_path}.branch_mass",
            final=True,
        )
    if (
        sum(conditions[name]["subjects"] for name in ("complete", "incomplete"))
        != subjects
    ):
        _fail(f"{path}.conditions", "subject counts must sum to the frozen cohort")
    return copy.deepcopy(conditions)


def _validate_manifest_dataset(
    dataset_key: str,
    value: Any,
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    frozenset[str],
    frozenset[str],
]:
    dataset_path = f"datasets.{dataset_key}"
    dataset = _require_exact_keys(value, PRIVATE_DATASET_KEYS, dataset_path)
    public_contract = RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]

    contract = _require_exact_keys(
        dataset["contract"],
        PRIVATE_CONTRACT_KEYS,
        f"{dataset_path}.contract",
    )
    expected_contract = {
        "subjects": public_contract["subjects"],
        "folds": public_contract["folds"],
        "resampling_unit": RENDERER.RESAMPLING_UNIT[dataset_key],
        "resampling_units": public_contract["resampling_units"],
    }
    if contract != expected_contract:
        _fail(
            f"{dataset_path}.contract",
            f"must equal the frozen public contract {expected_contract}",
        )
    if dataset["class_columns"] != list(CLASS_COLUMNS):
        _fail(
            f"{dataset_path}.class_columns",
            f"must equal the fixed class columns {list(CLASS_COLUMNS)}",
        )
    seed_ids = _require_int_list(
        dataset["seed_ids"],
        f"{dataset_path}.seed_ids",
        minimum_length=2,
    )
    input_sha256 = _require_sha256(
        dataset["input_sha256"],
        f"{dataset_path}.input_sha256",
    )
    private_paths: set[str] = set()

    def verify_private_file(
        owner: dict[str, Any],
        path_field: str,
        sha_field: str,
        logical_path: str,
    ) -> bytes:
        expected_sha256 = _require_sha256(
            owner[sha_field],
            f"{logical_path}.{sha_field}",
        )
        raw, verified_path = _read_and_verify_file(
            manifest_path,
            owner[path_field],
            expected_sha256,
            f"{logical_path}.{path_field}",
        )
        private_paths.update(
            {
                str(owner[path_field]),
                str(verified_path),
                str(verified_path.resolve()),
                verified_path.name,
            }
        )
        return raw

    primary = dataset["primary"]
    if not isinstance(primary, list) or len(primary) != 2:
        _fail(f"{dataset_path}.primary", "must contain exactly full and comparator")
    if [
        row.get("input_id") if isinstance(row, dict) else None for row in primary
    ] != list(PRIMARY_INPUT_IDS):
        _fail(
            f"{dataset_path}.primary",
            f"input ids must be ordered exactly as {list(PRIMARY_INPUT_IDS)}",
        )
    if [row.get("role") if isinstance(row, dict) else None for row in primary] != [
        "ours",
        "comparator",
    ]:
        _fail(
            f"{dataset_path}.primary",
            "roles must be ordered exactly as ours and comparator",
        )
    for index, row in enumerate(primary):
        row_path = f"{dataset_path}.primary[{index}]"
        _require_exact_keys(row, PRIVATE_PRIMARY_KEYS, row_path)
        RENDERER._require_public_text(row["method"], f"{row_path}.method", final=True)
        RENDERER._require_public_text(
            row["configuration_id"],
            f"{row_path}.configuration_id",
            final=True,
        )
        verify_private_file(
            row,
            "configuration_path",
            "configuration_sha256",
            row_path,
        )
        execution_raw = verify_private_file(
            row,
            "execution_receipt_path",
            "execution_receipt_sha256",
            row_path,
        )
        _validate_execution_receipt(
            _decode_private_json(execution_raw, f"{row_path}.execution_receipt"),
            dataset_key=dataset_key,
            arm_id=row["input_id"],
            configuration_sha256=row["configuration_sha256"],
            input_sha256=input_sha256,
            seed_ids=seed_ids,
            path=f"{row_path}.execution_receipt",
        )

    ablations = dataset["ablations"]
    if not isinstance(ablations, list):
        _fail(f"{dataset_path}.ablations", "must be a list")
    expected_ablation_ids = [row[0] for row in RENDERER.REQUIRED_ABLATIONS]
    if [row.get("id") if isinstance(row, dict) else None for row in ablations] != (
        expected_ablation_ids
    ):
        _fail(
            f"{dataset_path}.ablations",
            "must contain the exact ordered pre-specified ablation ids",
        )
    for index, row in enumerate(ablations):
        row_path = f"{dataset_path}.ablations[{index}]"
        _require_exact_keys(
            row,
            PRIVATE_ABLATION_KEYS,
            row_path,
        )
        RENDERER._require_public_text(
            row["configuration_id"],
            f"{row_path}.configuration_id",
            final=True,
        )
        verify_private_file(
            row,
            "configuration_path",
            "configuration_sha256",
            row_path,
        )
        execution_raw = verify_private_file(
            row,
            "execution_receipt_path",
            "execution_receipt_sha256",
            row_path,
        )
        _validate_execution_receipt(
            _decode_private_json(execution_raw, f"{row_path}.execution_receipt"),
            dataset_key=dataset_key,
            arm_id=row["id"],
            configuration_sha256=row["configuration_sha256"],
            input_sha256=input_sha256,
            seed_ids=seed_ids,
            path=f"{row_path}.execution_receipt",
        )

    if primary[0]["method"] == primary[1]["method"]:
        _fail(f"{dataset_path}.primary", "public method names must be unique")
    all_rows = [*primary, *ablations]
    for field in ("configuration_id", "configuration_sha256"):
        values = [row[field] for row in all_rows]
        if len(values) != len(set(values)):
            _fail(
                f"{dataset_path}.{field}",
                "must be unique across primary and ablation arms",
            )

    analysis_raw = verify_private_file(
        dataset,
        "analysis_receipt_path",
        "analysis_receipt_sha256",
        dataset_path,
    )
    analysis_receipt = _decode_private_json(
        analysis_raw,
        f"{dataset_path}.analysis_receipt",
    )
    _validate_analysis_receipt_metadata(
        analysis_receipt,
        dataset_key=dataset_key,
        input_sha256=input_sha256,
        seed_ids=seed_ids,
        path=f"{dataset_path}.analysis_receipt",
    )
    interpretability = _require_exact_keys(
        dataset["interpretability"],
        PRIVATE_INTERPRETABILITY_KEYS,
        f"{dataset_path}.interpretability",
    )
    if interpretability["aggregation_unit"] != INTERPRETABILITY_AGGREGATION_UNIT:
        _fail(
            f"{dataset_path}.interpretability.aggregation_unit",
            f"must equal {INTERPRETABILITY_AGGREGATION_UNIT!r}",
        )
    if interpretability["aggregation_method"] != INTERPRETABILITY_AGGREGATION_METHOD:
        _fail(
            f"{dataset_path}.interpretability.aggregation_method",
            f"must equal {INTERPRETABILITY_AGGREGATION_METHOD!r}",
        )
    aggregation_raw = verify_private_file(
        interpretability,
        "aggregation_receipt_path",
        "aggregation_receipt_sha256",
        f"{dataset_path}.interpretability",
    )
    aggregation_receipt = _decode_private_json(
        aggregation_raw,
        f"{dataset_path}.interpretability.aggregation_receipt",
    )
    conditions = _validate_aggregation_receipt(
        aggregation_receipt,
        dataset_key=dataset_key,
        input_sha256=input_sha256,
        path=f"{dataset_path}.interpretability.aggregation_receipt",
    )

    private_path = _resolve_private_input(
        manifest_path,
        dataset["input_npz"],
        f"{dataset_path}.input_npz",
    )
    private_paths.update(
        {
            str(dataset["input_npz"]),
            str(private_path),
            str(private_path.resolve()),
            private_path.name,
        }
    )
    input_ids = list(PRIMARY_INPUT_IDS) + expected_ablation_ids
    expected_array_keys = PRIVATE_ARRAY_KEYS | {
        f"probabilities_{input_id}" for input_id in input_ids
    }
    arrays = _read_hashed_npz(
        private_path,
        input_sha256,
        expected_array_keys,
        dataset_path,
    )
    labels, _, cluster_ids, ensembles, private_identifiers = _validate_private_arrays(
        dataset_key,
        arrays,
        seed_ids,
    )
    if set(ensembles) != set(input_ids):
        _fail(
            f"{dataset_path}.input_npz",
            "must contain one probability tensor for every frozen public arm",
        )

    validated_dataset = dict(dataset)
    validated_dataset["interpretability"] = {
        **interpretability,
        "conditions": conditions,
    }
    validated_dataset["_analysis_receipt"] = analysis_receipt
    return (
        validated_dataset,
        {
            "labels": labels,
            "cluster_ids": cluster_ids,
            **ensembles,
        },
        private_identifiers,
        frozenset(private_paths),
    )


def _public_interpretability(
    dataset_key: str,
    private_interpretability: dict[str, Any],
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition_name in ("complete", "incomplete"):
        source = private_interpretability["conditions"][condition_name]
        conditions[condition_name] = {
            "definition": INTERPRETABILITY_DEFINITIONS[condition_name],
            "subjects": source["subjects"],
            "decision_allocation": source["decision_allocation"],
            "branch_mass": source["branch_mass"],
        }
    return {
        "source": "aggregate out-of-fold checkpoint replay",
        "condition_design": "natural_disjoint",
        "aggregation_unit": INTERPRETABILITY_AGGREGATION_UNIT,
        "aggregation_method": INTERPRETABILITY_AGGREGATION_METHOD,
        "aggregation_receipt_sha256": private_interpretability[
            "aggregation_receipt_sha256"
        ],
        "modality_names": list(
            RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]["modality_names"]
        ),
        "conditions": conditions,
    }


def _compute_public_dataset(
    dataset_key: str,
    private_dataset: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = arrays["labels"]
    cluster_ids = arrays["cluster_ids"]
    primary_metadata = private_dataset["primary"]
    ours_probabilities = arrays["full"]
    comparator_probabilities = arrays["comparator"]
    ours_metrics = _metric_bundle(labels, ours_probabilities)
    comparator_metrics = _metric_bundle(labels, comparator_probabilities)

    observed_delta, raw_p_value = _paired_cluster_swap(
        labels,
        ours_probabilities,
        comparator_probabilities,
        cluster_ids,
        draws=RENDERER.SWAP_DRAWS,
        seed=RENDERER.SWAP_RNG_SEED,
    )
    public_delta = ours_metrics["macro_f1"] - comparator_metrics["macro_f1"]
    if not math.isclose(observed_delta, public_delta, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("paired statistic and public Macro-F1 difference diverged")
    bootstrap_lower = _paired_cluster_bootstrap_lower(
        labels,
        ours_probabilities,
        comparator_probabilities,
        cluster_ids,
        draws=RENDERER.BOOTSTRAP_DRAWS,
        seed=RENDERER.BOOTSTRAP_RNG_SEED,
    )
    _validate_analysis_receipt_results(
        private_dataset["_analysis_receipt"],
        observed_delta=observed_delta,
        p_value=raw_p_value,
        bootstrap_lower_bound=bootstrap_lower,
        path=f"datasets.{dataset_key}.analysis_receipt",
    )

    primary_results = []
    for metadata, metrics in zip(
        primary_metadata,
        (ours_metrics, comparator_metrics),
    ):
        primary_results.append(
            {
                "method": metadata["method"],
                "role": metadata["role"],
                "configuration_id": metadata["configuration_id"],
                "configuration_sha256": metadata["configuration_sha256"],
                "execution_receipt_sha256": metadata["execution_receipt_sha256"],
                "metrics": metrics,
            }
        )

    ablations = []
    label_by_id = dict(RENDERER.REQUIRED_ABLATIONS)
    for metadata in private_dataset["ablations"]:
        ablation_id = metadata["id"]
        ablations.append(
            {
                "id": ablation_id,
                "label": label_by_id[ablation_id],
                "configuration_id": metadata["configuration_id"],
                "configuration_sha256": metadata["configuration_sha256"],
                "execution_receipt_sha256": metadata["execution_receipt_sha256"],
                "metrics": _metric_bundle(labels, arrays[ablation_id]),
            }
        )

    contract = RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]
    comparison = {
        "ours": primary_results[0]["method"],
        "comparator": primary_results[1]["method"],
        "metric": "macro_f1",
        "paired": True,
        "delta": public_delta,
        "bootstrap_lower_bound": bootstrap_lower,
        "bootstrap_confidence_level": RENDERER.BOOTSTRAP_CONFIDENCE_LEVEL,
        "p_value": raw_p_value,
        "adjusted_p_value": None,
        "alpha": RENDERER.ALPHA,
        "test": RENDERER.PAIRED_TEST,
        "alternative": "greater",
        "resampling_unit": RENDERER.RESAMPLING_UNIT[dataset_key],
        "n_units": contract["resampling_units"],
        "swap_draws": RENDERER.SWAP_DRAWS,
        "swap_rng_seed": RENDERER.SWAP_RNG_SEED,
        "bootstrap_draws": RENDERER.BOOTSTRAP_DRAWS,
        "bootstrap_rng_seed": RENDERER.BOOTSTRAP_RNG_SEED,
        "multiplicity_adjustment": RENDERER.MULTIPLICITY_ADJUSTMENT,
        "analysis_receipt_sha256": private_dataset["analysis_receipt_sha256"],
        "confirmatory_support": False,
    }
    public_dataset = {
        "display_name": contract["display_name"],
        "task": contract["task"],
        "evaluation": {
            field: contract[field]
            for field in (
                "design",
                "subjects",
                "folds",
                "split",
                "evidence_scope",
                "partition_reused",
                "selection_independent",
            )
        },
        "primary_results": primary_results,
        "statistical_comparisons": [comparison],
        "ablations": ablations,
        "interpretability": _public_interpretability(
            dataset_key,
            private_dataset["interpretability"],
        ),
    }
    return public_dataset, comparison


def compute_private_result(manifest_path: Path) -> PrivateBuildResult:
    """Phase one: verify private inputs and compute an in-memory aggregate."""

    manifest_path = Path(manifest_path).expanduser()
    manifest, manifest_bytes = _read_private_manifest(manifest_path)
    _require_exact_keys(manifest, PRIVATE_ROOT_KEYS, "root")
    if manifest["schema"] != PRIVATE_MANIFEST_SCHEMA:
        _fail("schema", f"must equal {PRIVATE_MANIFEST_SCHEMA!r}")
    RENDERER._require_rfc3339(manifest["generated_at"], "generated_at")
    datasets = manifest["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != set(RENDERER.DATASET_ORDER):
        _fail("datasets", f"must contain exactly {list(RENDERER.DATASET_ORDER)}")

    public_datasets: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    private_identifiers: set[str] = set()
    private_paths = {
        str(manifest_path),
        str(manifest_path.resolve()),
        manifest_path.name,
    }
    for dataset_key in RENDERER.DATASET_ORDER:
        private_dataset, arrays, identifiers, dataset_private_paths = (
            _validate_manifest_dataset(
                dataset_key,
                datasets[dataset_key],
                manifest_path,
            )
        )
        public_dataset, comparison = _compute_public_dataset(
            dataset_key,
            private_dataset,
            arrays,
        )
        public_datasets[dataset_key] = public_dataset
        comparisons.append(comparison)
        private_identifiers.update(identifiers)
        private_paths.update(dataset_private_paths)

    adjusted = RENDERER._holm_adjusted_p_values(
        [float(comparison["p_value"]) for comparison in comparisons]
    )
    for comparison, adjusted_p_value in zip(comparisons, adjusted):
        comparison["adjusted_p_value"] = adjusted_p_value

    payload = {
        "schema": "cerd-final-public-results-v3",
        "status": "final",
        "generated_at": manifest["generated_at"],
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "metric_unit": "fraction",
        "metric_order": list(RENDERER.METRICS),
        "claim_boundary": RENDERER.ADAPTIVE_DEVELOPMENT_CLAIM,
        "datasets": public_datasets,
    }
    return PrivateBuildResult(
        payload=payload,
        private_identifiers=frozenset(private_identifiers),
        private_paths=frozenset(private_paths),
    )


def _walk_public(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk_public(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk_public(item)


def validate_public_result(result: PrivateBuildResult) -> bytes:
    """Phase two: enforce public schema/privacy and return canonical JSON bytes."""

    RENDERER.validate_payload(result.payload)
    public_strings: set[str] = set()
    for key, value in _walk_public(result.payload):
        if key in FORBIDDEN_PUBLIC_KEYS or (
            isinstance(key, str) and key.endswith("_path")
        ):
            _fail("public payload", f"forbidden private field {key!r}")
        if isinstance(value, str):
            public_strings.add(value)
    leaked_identifiers = public_strings & set(result.private_identifiers)
    if leaked_identifiers:
        _fail("public payload", "contains a private row or cluster identifier")

    serialized = (
        json.dumps(
            result.payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(serialized) > MAX_PUBLIC_BYTES:
        _fail("public payload", "exceeds the aggregate-only size ceiling")
    serialized_text = serialized.decode("utf-8")
    if any(private_path in serialized_text for private_path in result.private_paths):
        _fail("public payload", "contains a private input path")
    return serialized


def atomic_write_public_result(
    output_path: Path,
    serialized: bytes,
    *,
    replace: bool = False,
) -> None:
    """Atomically replace one explicitly named public JSON output."""

    output_path = Path(output_path).expanduser()
    if output_path.name != "final_results.json":
        raise ValueError("public output must be named final_results.json")
    parent = output_path.parent
    if not parent.is_dir():
        raise ValueError("public output parent directory must already exist")
    if Path(os.path.realpath(parent)) != Path(os.path.abspath(parent)):
        raise ValueError("public output parent must not traverse a symbolic link")
    if output_path.is_symlink():
        raise ValueError("public output must not be a symbolic link")
    if output_path.exists() and not output_path.is_file():
        raise ValueError("existing public output must be a regular file")
    if output_path.exists() and not replace:
        raise ValueError("public output already exists; pass --replace to replace it")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        observed = temporary_path.read_bytes()
        if observed != serialized:
            raise RuntimeError("temporary public artifact failed byte verification")
        reparsed = json.loads(
            observed.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
        RENDERER.validate_payload(reparsed)
        if replace:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as error:
                raise ValueError(
                    "public output already exists; pass --replace to replace it"
                ) from error
            temporary_path.unlink()
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_and_write(
    manifest_path: Path,
    output_path: Path,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    result = compute_private_result(manifest_path)
    serialized = validate_public_result(result)
    output_path = Path(output_path).expanduser()
    output_markers = {
        str(output_path),
        str(output_path.absolute()),
        str(output_path.resolve()),
        output_path.name,
    }
    if output_markers & set(result.private_paths):
        raise ValueError("public output must not overwrite a private input")
    atomic_write_public_result(output_path, serialized, replace=replace)
    return result.payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="explicit private manifest; no campaign discovery is performed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="explicit aggregate JSON output; README and figures are untouched",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing regular aggregate only after full validation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        expected_output = (ROOT / "results" / "final_results.json").resolve()
        if args.output.expanduser().resolve() != expected_output:
            raise ValueError(
                "command-line output must be the repository results/final_results.json"
            )
        build_and_write(args.manifest, args.output, replace=args.replace)
        print(f"wrote validated aggregate artifact {args.output.name}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
