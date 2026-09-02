import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_final_results as BUILDER  # noqa: E402


ABLATION_IDS = [row[0] for row in BUILDER.RENDERER.REQUIRED_ABLATIONS]
ARM_IDS = ["full", "comparator", *ABLATION_IDS]
SEED_IDS = [101, 202]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probability_tensor(labels: np.ndarray, error_stride: int) -> np.ndarray:
    rows = np.arange(labels.size)
    predictions = labels.copy()
    errors = rows % error_stride == 0
    predictions[errors] = (predictions[errors] + 1) % 3
    tensors = []
    for confidence in (0.82, 0.76):
        probabilities = np.full(
            (labels.size, 3),
            (1.0 - confidence) / 2.0,
            dtype=np.float64,
        )
        probabilities[rows, predictions] = confidence
        tensors.append(probabilities)
    return np.stack(tensors)


def _private_arrays(dataset_key: str) -> dict[str, np.ndarray]:
    subjects = BUILDER.RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]["subjects"]
    labels = np.arange(subjects, dtype=np.int64) % 3
    row_ids = np.asarray(
        [f"row-{dataset_key}-{index:04d}" for index in range(subjects)]
    )
    if dataset_key == "adni":
        cluster_ids = np.asarray(
            [f"subject-cluster-{index:04d}" for index in range(subjects)]
        )
        fold_ids = np.arange(subjects, dtype=np.int64) % 5
    else:
        family_ids = np.asarray([f"family-cluster-{index:04d}" for index in range(922)])
        cluster_ids = np.concatenate((family_ids, family_ids[:24]))
        family_folds = np.arange(922, dtype=np.int64) % 5
        fold_ids = np.concatenate((family_folds, family_folds[:24]))

    arrays = {
        "format": np.asarray(BUILDER.PRIVATE_OOF_SCHEMA),
        "dataset": np.asarray(dataset_key),
        "row_ids": row_ids,
        "cluster_ids": cluster_ids,
        "fold_ids": fold_ids,
        "labels": labels,
        "seed_ids": np.asarray(SEED_IDS, dtype=np.int64),
        "class_columns": np.asarray(BUILDER.CLASS_COLUMNS, dtype=np.int64),
    }
    strides = {"full": 19, "comparator": 7}
    strides.update({arm_id: 8 + index for index, arm_id in enumerate(ABLATION_IDS)})
    for arm_id in ARM_IDS:
        arrays[f"probabilities_{arm_id}"] = _probability_tensor(
            labels,
            strides[arm_id],
        )
    return arrays


def _write_bound_file(tmp_path: Path, name: str, content: str) -> tuple[str, str]:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path.name, _sha256(path)


def _row_metadata(
    tmp_path: Path,
    dataset_key: str,
    arm_id: str,
    input_sha256: str,
    *,
    primary: bool,
) -> dict[str, object]:
    stem = f"{dataset_key}-{arm_id}"
    config_path, config_sha = _write_bound_file(
        tmp_path,
        f"{stem}.config.json",
        json.dumps({"dataset": dataset_key, "arm": arm_id}, sort_keys=True),
    )
    receipt_path, receipt_sha = _write_bound_file(
        tmp_path,
        f"{stem}.execution.json",
        json.dumps(
            {
                "schema": BUILDER.PRIVATE_EXECUTION_RECEIPT_SCHEMA,
                "dataset": dataset_key,
                "arm_id": arm_id,
                "configuration_sha256": config_sha,
                "oof_input_sha256": input_sha256,
                "probability_array": f"probabilities_{arm_id}",
                "seed_ids": SEED_IDS,
                "class_columns": list(BUILDER.CLASS_COLUMNS),
                "seed_ensemble": BUILDER.SEED_ENSEMBLE_RULE,
                "subjects": BUILDER.RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key][
                    "subjects"
                ],
                "folds": 5,
                "resampling_unit": BUILDER.RENDERER.RESAMPLING_UNIT[dataset_key],
                "resampling_units": BUILDER.RENDERER.DATASET_PUBLIC_CONTRACT[
                    dataset_key
                ]["resampling_units"],
                "status": "complete",
            },
            sort_keys=True,
        ),
    )
    row = {
        "configuration_id": (
            f"synthetic {dataset_key} {arm_id.replace('_', '-')} configuration"
        ),
        "configuration_path": config_path,
        "configuration_sha256": config_sha,
        "execution_receipt_path": receipt_path,
        "execution_receipt_sha256": receipt_sha,
    }
    if primary:
        row.update(
            {
                "input_id": arm_id,
                "method": (
                    "CERD synthetic"
                    if arm_id == "full"
                    else f"Synthetic {dataset_key.upper()} comparator"
                ),
                "role": "ours" if arm_id == "full" else "comparator",
            }
        )
    else:
        row["id"] = arm_id
    return row


def _make_private_fixture(tmp_path: Path):
    manifest = {
        "schema": BUILDER.PRIVATE_MANIFEST_SCHEMA,
        "generated_at": "2026-09-01T12:00:00Z",
        "datasets": {},
    }
    arrays_by_dataset = {}
    for dataset_key in BUILDER.RENDERER.DATASET_ORDER:
        contract = BUILDER.RENDERER.DATASET_PUBLIC_CONTRACT[dataset_key]
        arrays = _private_arrays(dataset_key)
        arrays_by_dataset[dataset_key] = arrays
        input_path = tmp_path / f"{dataset_key}-oof.npz"
        np.savez_compressed(input_path, **arrays)
        input_sha256 = _sha256(input_path)
        ours_metrics = BUILDER._metric_bundle(
            arrays["labels"],
            arrays["probabilities_full"].mean(axis=0),
        )
        comparator_metrics = BUILDER._metric_bundle(
            arrays["labels"],
            arrays["probabilities_comparator"].mean(axis=0),
        )
        observed_delta = ours_metrics["macro_f1"] - comparator_metrics["macro_f1"]
        analysis_path, analysis_sha = _write_bound_file(
            tmp_path,
            f"{dataset_key}-analysis.json",
            json.dumps(
                {
                    "schema": BUILDER.PRIVATE_ANALYSIS_RECEIPT_SCHEMA,
                    "dataset": dataset_key,
                    "oof_input_sha256": input_sha256,
                    "class_columns": list(BUILDER.CLASS_COLUMNS),
                    "seed_ids": SEED_IDS,
                    "seed_ensemble": BUILDER.SEED_ENSEMBLE_RULE,
                    "resampling_unit": BUILDER.RENDERER.RESAMPLING_UNIT[dataset_key],
                    "n_units": contract["resampling_units"],
                    "comparison": {
                        "metric": "macro_f1",
                        "ours_input_id": "full",
                        "comparator_input_id": "comparator",
                        "alternative": "greater",
                        "swap_draws": 50_000,
                        "swap_rng_seed": 20_260_905,
                        "bootstrap_draws": 20_000,
                        "bootstrap_rng_seed": 20_260_906,
                        "bootstrap_quantile": 0.05,
                        "bootstrap_quantile_method": "linear",
                        "observed_delta": observed_delta,
                        "p_value": 0.01 if dataset_key == "adni" else 0.03,
                        "bootstrap_lower_bound": observed_delta - 0.01,
                    },
                },
                sort_keys=True,
            ),
        )
        complete = contract["subjects"] // 2
        aggregate_conditions = {
            "complete": {
                "subjects": complete,
                "decision_allocation": [0.1, 0.2, 0.3, 0.4],
                "branch_mass": {
                    "joint": 0.2,
                    "unimodal": 0.3,
                    "pairwise": 0.5,
                },
            },
            "incomplete": {
                "subjects": contract["subjects"] - complete,
                "decision_allocation": [0.2, 0.2, 0.3, 0.3],
                "branch_mass": {
                    "joint": 0.3,
                    "unimodal": 0.3,
                    "pairwise": 0.4,
                },
            },
        }
        aggregation_path, aggregation_sha = _write_bound_file(
            tmp_path,
            f"{dataset_key}-aggregation.json",
            json.dumps(
                {
                    "schema": BUILDER.PRIVATE_AGGREGATION_RECEIPT_SCHEMA,
                    "dataset": dataset_key,
                    "oof_input_sha256": input_sha256,
                    "aggregation_unit": BUILDER.INTERPRETABILITY_AGGREGATION_UNIT,
                    "aggregation_method": BUILDER.INTERPRETABILITY_AGGREGATION_METHOD,
                    "condition_design": "natural_disjoint",
                    "subjects": contract["subjects"],
                    "conditions": aggregate_conditions,
                },
                sort_keys=True,
            ),
        )
        manifest["datasets"][dataset_key] = {
            "contract": {
                "subjects": contract["subjects"],
                "folds": contract["folds"],
                "resampling_unit": BUILDER.RENDERER.RESAMPLING_UNIT[dataset_key],
                "resampling_units": contract["resampling_units"],
            },
            "input_npz": input_path.name,
            "input_sha256": input_sha256,
            "class_columns": list(BUILDER.CLASS_COLUMNS),
            "seed_ids": list(SEED_IDS),
            "primary": [
                _row_metadata(
                    tmp_path,
                    dataset_key,
                    arm_id,
                    input_sha256,
                    primary=True,
                )
                for arm_id in BUILDER.PRIMARY_INPUT_IDS
            ],
            "ablations": [
                _row_metadata(
                    tmp_path,
                    dataset_key,
                    arm_id,
                    input_sha256,
                    primary=False,
                )
                for arm_id in ABLATION_IDS
            ],
            "analysis_receipt_path": analysis_path,
            "analysis_receipt_sha256": analysis_sha,
            "interpretability": {
                "aggregation_unit": BUILDER.INTERPRETABILITY_AGGREGATION_UNIT,
                "aggregation_method": BUILDER.INTERPRETABILITY_AGGREGATION_METHOD,
                "aggregation_receipt_path": aggregation_path,
                "aggregation_receipt_sha256": aggregation_sha,
            },
        }
    manifest_path = tmp_path / "private-manifest.json"
    _write_manifest(manifest_path, manifest)
    return manifest_path, manifest, arrays_by_dataset


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _persist_inputs(
    tmp_path: Path,
    manifest_path: Path,
    manifest: dict[str, object],
    arrays_by_dataset: dict[str, dict[str, np.ndarray]],
) -> None:
    for dataset_key, arrays in arrays_by_dataset.items():
        dataset = manifest["datasets"][dataset_key]
        input_path = tmp_path / dataset["input_npz"]
        np.savez_compressed(input_path, **arrays)
        input_sha256 = _sha256(input_path)
        dataset["input_sha256"] = input_sha256
        for row in [*dataset["primary"], *dataset["ablations"]]:
            receipt_path = tmp_path / row["execution_receipt_path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["oof_input_sha256"] = input_sha256
            _write_manifest(receipt_path, receipt)
            row["execution_receipt_sha256"] = _sha256(receipt_path)
        analysis_path = tmp_path / dataset["analysis_receipt_path"]
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis["oof_input_sha256"] = input_sha256
        _write_manifest(analysis_path, analysis)
        dataset["analysis_receipt_sha256"] = _sha256(analysis_path)
        aggregation = dataset["interpretability"]
        aggregation_path = tmp_path / aggregation["aggregation_receipt_path"]
        aggregation_receipt = json.loads(aggregation_path.read_text(encoding="utf-8"))
        aggregation_receipt["oof_input_sha256"] = input_sha256
        _write_manifest(aggregation_path, aggregation_receipt)
        aggregation["aggregation_receipt_sha256"] = _sha256(aggregation_path)
    _write_manifest(manifest_path, manifest)


def _install_fast_statistics(monkeypatch):
    calls = []

    def macro_f1(labels, probabilities):
        return BUILDER._metric_bundle(labels, probabilities)["macro_f1"]

    def fake_swap(
        labels,
        ours_probabilities,
        comparator_probabilities,
        cluster_ids,
        *,
        draws,
        seed,
        chunk_size=BUILDER.STATISTIC_CHUNK_SIZE,
    ):
        units = np.unique(cluster_ids).size
        calls.append(("swap", units, draws, seed))
        delta = macro_f1(labels, ours_probabilities) - macro_f1(
            labels,
            comparator_probabilities,
        )
        return delta, 0.01 if units == 1480 else 0.03

    def fake_bootstrap(
        labels,
        ours_probabilities,
        comparator_probabilities,
        cluster_ids,
        *,
        draws,
        seed,
        chunk_size=BUILDER.STATISTIC_CHUNK_SIZE,
    ):
        units = np.unique(cluster_ids).size
        calls.append(("bootstrap", units, draws, seed))
        delta = macro_f1(labels, ours_probabilities) - macro_f1(
            labels,
            comparator_probabilities,
        )
        return delta - 0.01

    monkeypatch.setattr(BUILDER, "_paired_cluster_swap", fake_swap)
    monkeypatch.setattr(BUILDER, "_paired_cluster_bootstrap_lower", fake_bootstrap)
    return calls


def test_two_phase_builder_computes_only_the_frozen_public_aggregate(
    tmp_path,
    monkeypatch,
):
    manifest_path, manifest, arrays = _make_private_fixture(tmp_path)
    calls = _install_fast_statistics(monkeypatch)

    result = BUILDER.compute_private_result(manifest_path)
    assert not (tmp_path / "final_results.json").exists()
    serialized = BUILDER.validate_public_result(result)
    payload = json.loads(serialized)
    BUILDER.RENDERER.validate_payload(payload)

    assert payload["source_manifest_sha256"] == _sha256(manifest_path)
    assert payload["claim_boundary"] == BUILDER.RENDERER.ADAPTIVE_DEVELOPMENT_CLAIM
    assert payload["metric_order"] == list(BUILDER.RENDERER.METRICS)
    assert calls == [
        ("swap", 1480, 50_000, 20_260_905),
        ("bootstrap", 1480, 20_000, 20_260_906),
        ("swap", 922, 50_000, 20_260_905),
        ("bootstrap", 922, 20_000, 20_260_906),
    ]

    for dataset_key in BUILDER.RENDERER.DATASET_ORDER:
        dataset = payload["datasets"][dataset_key]
        assert [row["id"] for row in dataset["ablations"]] == ABLATION_IDS
        for row in [*dataset["primary_results"], *dataset["ablations"]]:
            assert set(row["metrics"]) == set(BUILDER.RENDERER.METRICS)
        expected_full = BUILDER._metric_bundle(
            arrays[dataset_key]["labels"],
            arrays[dataset_key]["probabilities_full"].mean(axis=0),
        )
        assert dataset["primary_results"][0]["metrics"] == expected_full
        comparison = dataset["statistical_comparisons"][0]
        assert comparison["confirmatory_support"] is False
        assert (
            comparison["analysis_receipt_sha256"]
            == manifest["datasets"][dataset_key]["analysis_receipt_sha256"]
        )
        interpretation = dataset["interpretability"]
        assert interpretation["aggregation_unit"] == "subject"
        assert interpretation["aggregation_method"] == (
            "arithmetic mean of per-subject normalized vectors"
        )
    assert payload["datasets"]["adni"]["statistical_comparisons"][0][
        "adjusted_p_value"
    ] == pytest.approx(0.02)
    assert payload["datasets"]["abcd"]["statistical_comparisons"][0][
        "adjusted_p_value"
    ] == pytest.approx(0.03)

    public_text = serialized.decode("utf-8")
    for forbidden in (
        "row-adni-0000",
        "family-cluster-0000",
        "private-manifest.json",
        "adni-oof.npz",
        "probabilities_full",
        "fold_ids",
    ):
        assert forbidden not in public_text


def test_real_cluster_statistics_are_deterministic_and_use_fixed_class_macro_f1():
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    cluster_ids = np.asarray(["a", "a", "b", "c", "c", "d", "e", "e", "f"])
    ours = _probability_tensor(labels, 8).mean(axis=0)
    comparator = _probability_tensor(labels, 4).mean(axis=0)

    first_swap = BUILDER._paired_cluster_swap(
        labels,
        ours,
        comparator,
        cluster_ids,
        draws=257,
        seed=20_260_905,
        chunk_size=17,
    )
    second_swap = BUILDER._paired_cluster_swap(
        labels,
        ours,
        comparator,
        cluster_ids,
        draws=257,
        seed=20_260_905,
        chunk_size=31,
    )
    assert first_swap == second_swap
    assert 0.0 < first_swap[1] <= 1.0
    permutation = np.asarray([8, 0, 4, 2, 7, 1, 5, 3, 6])
    permuted_swap = BUILDER._paired_cluster_swap(
        labels[permutation],
        ours[permutation],
        comparator[permutation],
        cluster_ids[permutation],
        draws=257,
        seed=20_260_905,
        chunk_size=17,
    )
    assert permuted_swap == first_swap

    first_bootstrap = BUILDER._paired_cluster_bootstrap_lower(
        labels,
        ours,
        comparator,
        cluster_ids,
        draws=257,
        seed=20_260_906,
        chunk_size=17,
    )
    second_bootstrap = BUILDER._paired_cluster_bootstrap_lower(
        labels,
        ours,
        comparator,
        cluster_ids,
        draws=257,
        seed=20_260_906,
        chunk_size=31,
    )
    assert first_bootstrap == second_bootstrap
    permuted_bootstrap = BUILDER._paired_cluster_bootstrap_lower(
        labels[permutation],
        ours[permutation],
        comparator[permutation],
        cluster_ids[permutation],
        draws=257,
        seed=20_260_906,
        chunk_size=17,
    )
    assert permuted_bootstrap == first_bootstrap

    _, tie_p = BUILDER._paired_cluster_swap(
        labels,
        ours,
        ours,
        cluster_ids,
        draws=37,
        seed=20_260_905,
    )
    assert tie_p == 1.0

    # A bootstrap sample may omit class 2; the helper still averages over all
    # three fixed confusion-matrix columns and assigns the absent class F1 zero.
    missing_class_confusion = np.asarray(
        [[[3, 0, 0], [0, 2, 0], [0, 0, 0]]],
        dtype=np.int64,
    )
    assert BUILDER._macro_f1_from_confusions(missing_class_confusion)[
        0
    ] == pytest.approx(2.0 / 3.0)


def test_bootstrap_definition_is_pooled_cluster_resampling_and_linear_quantile():
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    clusters = np.asarray(
        ["family-a", "family-a", "family-b", "family-c", "family-c", "family-d"]
    )
    ours = _probability_tensor(labels, 5).mean(axis=0)
    comparator = _probability_tensor(labels, 3).mean(axis=0)
    draws = 41
    seed = 73

    ours_clusters = BUILDER._cluster_confusions(
        labels,
        ours.argmax(axis=1),
        clusters,
        3,
    )
    comparator_clusters = BUILDER._cluster_confusions(
        labels,
        comparator.argmax(axis=1),
        clusters,
        3,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, ours_clusters.shape[0], size=(draws, ours_clusters.shape[0])
    )
    expected_differences = BUILDER._macro_f1_from_confusions(
        ours_clusters[sampled].sum(axis=1)
    ) - BUILDER._macro_f1_from_confusions(comparator_clusters[sampled].sum(axis=1))
    expected = float(np.quantile(expected_differences, 0.05, method="linear"))
    observed = BUILDER._paired_cluster_bootstrap_lower(
        labels,
        ours,
        comparator,
        clusters,
        draws=draws,
        seed=seed,
        chunk_size=draws,
    )
    assert observed == expected
    assert ours_clusters[0].sum() == 2
    assert ours_clusters[2].sum() == 2


def test_manifest_and_oof_tampering_fail_closed(tmp_path, monkeypatch):
    manifest_path, baseline_manifest, baseline_arrays = _make_private_fixture(tmp_path)
    _install_fast_statistics(monkeypatch)

    cases = []

    def wrong_subjects(manifest, arrays):
        manifest["datasets"]["adni"]["contract"]["subjects"] = 1479

    cases.append((wrong_subjects, "frozen public contract", False))

    def wrong_ablation_order(manifest, arrays):
        manifest["datasets"]["adni"]["ablations"].reverse()

    cases.append((wrong_ablation_order, "exact ordered", False))

    def wrong_aggregation(manifest, arrays):
        manifest["datasets"]["adni"]["interpretability"][
            "aggregation_method"
        ] = "row weighted average"

    cases.append((wrong_aggregation, "arithmetic mean", False))

    def raw_interpretability_rows(manifest, arrays):
        manifest["datasets"]["adni"]["interpretability"]["conditions"] = {}

    cases.append((raw_interpretability_rows, "keys must be exactly", False))

    def wrong_class_columns(manifest, arrays):
        arrays["adni"]["class_columns"] = np.asarray([0, 2, 1])

    cases.append((wrong_class_columns, "fixed class columns", False))

    def wrong_seed_order(manifest, arrays):
        arrays["adni"]["seed_ids"] = np.asarray(SEED_IDS[::-1])

    cases.append((wrong_seed_order, "manifest seed order", False))

    def invalid_probability_sum(manifest, arrays):
        arrays["adni"]["probabilities_full"][0, 0, 0] = 0.5

    cases.append((invalid_probability_sum, "sum to one", False))

    def family_crosses_fold(manifest, arrays):
        arrays["abcd"]["fold_ids"][922] = 1

    cases.append((family_crosses_fold, "exactly one OOF fold", False))

    def duplicate_row(manifest, arrays):
        arrays["adni"]["row_ids"][1] = arrays["adni"]["row_ids"][0]

    cases.append((duplicate_row, "one unique identifier", False))

    def wrong_cluster_count(manifest, arrays):
        arrays["adni"]["cluster_ids"][-1] = arrays["adni"]["cluster_ids"][0]

    cases.append((wrong_cluster_count, "1480 unique resampling units", False))

    def wrong_dataset_marker(manifest, arrays):
        arrays["adni"]["dataset"] = np.asarray("abcd")

    cases.append((wrong_dataset_marker, "scalar string 'adni'", False))

    def nonfinite_probability(manifest, arrays):
        arrays["adni"]["probabilities_full"][0, 0, 0] = np.nan

    cases.append((nonfinite_probability, "finite numeric probabilities", False))

    def object_identifier_array(manifest, arrays):
        arrays["adni"]["row_ids"] = arrays["adni"]["row_ids"].astype(object)

    cases.append((object_identifier_array, "Object arrays cannot be loaded", False))

    def extra_array(manifest, arrays):
        arrays["adni"]["unexpected"] = np.asarray([1])

    cases.append((extra_array, "archive members must be exactly", False))

    def stale_input_digest(manifest, arrays):
        manifest["datasets"]["adni"]["input_sha256"] = "0" * 64

    cases.append((stale_input_digest, "frozen binding", True))

    for mutate, message, mutate_after_persist in cases:
        manifest = copy.deepcopy(baseline_manifest)
        arrays = {
            dataset_key: {
                key: np.array(value, copy=True) for key, value in dataset_arrays.items()
            }
            for dataset_key, dataset_arrays in baseline_arrays.items()
        }
        if not mutate_after_persist:
            mutate(manifest, arrays)
        _persist_inputs(tmp_path, manifest_path, manifest, arrays)
        if mutate_after_persist:
            mutate(manifest, arrays)
            _write_manifest(manifest_path, manifest)
        with pytest.raises(ValueError, match=message):
            BUILDER.compute_private_result(manifest_path)


@pytest.mark.parametrize(
    "path_selector",
    [
        lambda dataset: dataset["primary"][0]["configuration_path"],
        lambda dataset: dataset["primary"][0]["execution_receipt_path"],
        lambda dataset: dataset["analysis_receipt_path"],
        lambda dataset: dataset["interpretability"]["aggregation_receipt_path"],
    ],
)
def test_every_provenance_digest_is_checked_against_exact_bytes(
    tmp_path,
    monkeypatch,
    path_selector,
):
    manifest_path, manifest, _ = _make_private_fixture(tmp_path)
    _install_fast_statistics(monkeypatch)
    private_path = tmp_path / path_selector(manifest["datasets"]["adni"])
    private_path.write_bytes(private_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="does not match the exact file bytes"):
        BUILDER.compute_private_result(manifest_path)


def test_oof_digest_is_checked_against_the_exact_analyzed_bytes(tmp_path, monkeypatch):
    manifest_path, manifest, _ = _make_private_fixture(tmp_path)
    _install_fast_statistics(monkeypatch)
    input_path = tmp_path / manifest["datasets"]["adni"]["input_npz"]
    input_path.write_bytes(input_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="does not match the exact NPZ bytes"):
        BUILDER.compute_private_result(manifest_path)


@pytest.mark.parametrize("commitment_change", ["row_membership", "seed_composition"])
def test_approved_receipt_commits_exact_row_set_and_seed_composition(
    tmp_path,
    monkeypatch,
    commitment_change,
):
    manifest_path, manifest, arrays = _make_private_fixture(tmp_path)
    _install_fast_statistics(monkeypatch)
    if commitment_change == "row_membership":
        arrays["adni"]["row_ids"][0] = "different-but-still-unique-row"
    else:
        arrays["adni"]["seed_ids"] = np.asarray([303, 404], dtype=np.int64)
        manifest["datasets"]["adni"]["seed_ids"] = [303, 404]
    input_path = tmp_path / manifest["datasets"]["adni"]["input_npz"]
    np.savez_compressed(input_path, **arrays["adni"])
    manifest["datasets"]["adni"]["input_sha256"] = _sha256(input_path)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="frozen binding"):
        BUILDER.compute_private_result(manifest_path)


@pytest.mark.parametrize(
    ("constant", "message"),
    [
        ("MAX_BOUND_FILE_BYTES", "provenance file exceeds"),
        ("MAX_PRIVATE_NPZ_BYTES", "private NPZ exceeds"),
        ("MAX_PRIVATE_NPZ_UNCOMPRESSED_BYTES", "uncompressed arrays exceed"),
    ],
)
def test_private_input_size_ceilings_fail_before_analysis(
    tmp_path,
    monkeypatch,
    constant,
    message,
):
    manifest_path, _, _ = _make_private_fixture(tmp_path)
    monkeypatch.setattr(BUILDER, constant, 1)
    with pytest.raises(ValueError, match=message):
        BUILDER.compute_private_result(manifest_path)


@pytest.mark.parametrize(
    ("receipt_kind", "message"),
    [
        ("execution", "frozen binding"),
        ("analysis", "builder-recomputed analysis"),
        ("aggregation", "frozen binding"),
    ],
)
def test_receipt_contents_are_semantically_bound_not_just_hashed(
    tmp_path,
    monkeypatch,
    receipt_kind,
    message,
):
    manifest_path, manifest, _ = _make_private_fixture(tmp_path)
    _install_fast_statistics(monkeypatch)
    dataset = manifest["datasets"]["adni"]
    if receipt_kind == "execution":
        owner = dataset["primary"][0]
        path_field = "execution_receipt_path"
        sha_field = "execution_receipt_sha256"
        receipt_path = tmp_path / owner[path_field]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["configuration_sha256"] = "0" * 64
    elif receipt_kind == "analysis":
        owner = dataset
        path_field = "analysis_receipt_path"
        sha_field = "analysis_receipt_sha256"
        receipt_path = tmp_path / owner[path_field]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["comparison"]["observed_delta"] += 0.1
    else:
        owner = dataset["interpretability"]
        path_field = "aggregation_receipt_path"
        sha_field = "aggregation_receipt_sha256"
        receipt_path = tmp_path / owner[path_field]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["oof_input_sha256"] = "0" * 64
    _write_manifest(receipt_path, receipt)
    owner[sha_field] = _sha256(receipt_path)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        BUILDER.compute_private_result(manifest_path)


def test_public_privacy_phase_rejects_injected_private_identifier_and_path(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _make_private_fixture(tmp_path)
    _install_fast_statistics(monkeypatch)
    result = BUILDER.compute_private_result(manifest_path)

    identifier_payload = copy.deepcopy(result.payload)
    identifier_payload["datasets"]["adni"]["primary_results"][0][
        "configuration_id"
    ] = "row-adni-0000"
    identifier_result = BUILDER.PrivateBuildResult(
        payload=identifier_payload,
        private_identifiers=result.private_identifiers,
        private_paths=result.private_paths,
    )
    with pytest.raises(ValueError, match="private row or cluster identifier"):
        BUILDER.validate_public_result(identifier_result)

    path_payload = copy.deepcopy(result.payload)
    path_payload["datasets"]["adni"]["primary_results"][0][
        "configuration_id"
    ] = "adni-oof.npz"
    path_result = BUILDER.PrivateBuildResult(
        payload=path_payload,
        private_identifiers=result.private_identifiers,
        private_paths=result.private_paths,
    )
    with pytest.raises(ValueError, match="private input path"):
        BUILDER.validate_public_result(path_result)


def test_atomic_output_is_deterministic_no_clobber_and_renderer_independent(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _make_private_fixture(tmp_path)
    _install_fast_statistics(monkeypatch)
    output_path = tmp_path / "final_results.json"

    BUILDER.build_and_write(manifest_path, output_path)
    first = output_path.read_bytes()
    assert json.loads(first)["status"] == "final"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob("*.svg")) == []

    with pytest.raises(ValueError, match="already exists"):
        BUILDER.build_and_write(manifest_path, output_path)
    assert output_path.read_bytes() == first

    BUILDER.build_and_write(manifest_path, output_path, replace=True)
    assert output_path.read_bytes() == first

    with pytest.raises((ValueError, json.JSONDecodeError)):
        BUILDER.atomic_write_public_result(output_path, b"not json\n", replace=True)
    assert output_path.read_bytes() == first
    assert list(tmp_path.glob(".*.tmp")) == []

    def fail_replace(source, destination):
        raise OSError("injected pre-replace failure")

    monkeypatch.setattr(BUILDER.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected pre-replace failure"):
        BUILDER.atomic_write_public_result(output_path, first, replace=True)
    assert output_path.read_bytes() == first
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_writer_rejects_symlink_and_noncanonical_name(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _make_private_fixture(tmp_path)
    _install_fast_statistics(monkeypatch)
    result = BUILDER.compute_private_result(manifest_path)
    serialized = BUILDER.validate_public_result(result)
    target = tmp_path / "target.json"
    target.write_text("private", encoding="utf-8")
    output = tmp_path / "final_results.json"
    output.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        BUILDER.atomic_write_public_result(output, serialized, replace=True)
    assert target.read_text(encoding="utf-8") == "private"
    with pytest.raises(ValueError, match="must be named final_results.json"):
        BUILDER.atomic_write_public_result(tmp_path / "other.json", serialized)

    output.unlink()
    output.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        BUILDER.atomic_write_public_result(output, serialized, replace=True)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="parent must not traverse"):
        BUILDER.atomic_write_public_result(
            linked_parent / "final_results.json",
            serialized,
        )


def test_duplicate_manifest_keys_and_duplicate_npz_members_are_rejected(tmp_path):
    manifest_path = tmp_path / "duplicate.json"
    manifest_path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unique-key"):
        BUILDER._read_private_manifest(manifest_path)

    npy = io.BytesIO()
    np.save(npy, np.asarray([1], dtype=np.int64), allow_pickle=False)
    duplicate_npz = tmp_path / "duplicate.npz"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate_npz, "w") as archive:
            archive.writestr("a.npy", npy.getvalue())
            archive.writestr("a.npy", npy.getvalue())
    with pytest.raises(ValueError, match="archive members must be exactly"):
        BUILDER._read_hashed_npz(
            duplicate_npz,
            _sha256(duplicate_npz),
            {"a"},
            "datasets.synthetic",
        )


def test_builder_cli_has_no_implicit_private_input_or_output_defaults():
    with pytest.raises(SystemExit):
        BUILDER.parse_args([])


def test_command_line_builder_is_scoped_to_repository_final_artifact(tmp_path):
    code = BUILDER.main(
        [
            "--manifest",
            str(tmp_path / "does-not-need-to-exist.json"),
            "--output",
            str(tmp_path / "final_results.json"),
        ]
    )
    assert code == 2
