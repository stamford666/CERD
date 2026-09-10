#!/usr/bin/env python3
"""Refine the frozen ABCD ADHD-presentation dataset without changing labels/splits.

The v5 representation corrects the structural MRI definition, adds compact
task-performance summaries, and replaces direct high-dimensional SNP dosage
input with within-gene PCA fitted on the training partition only.  It retains
the v4 cohort, family-disjoint split, four modality groups, and fixed 15%
missingness assignment exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from prepare_abcd_multimodal import table_at_session


SESSION = "ses-00A"
T1_INTENSITY_PREFIX = "mr_y_smri__t1__gm__dsk"
STRUCTURAL_TABLES = {
    "cortical_volume": "imaging/mr_y_smri__vol__dsk.parquet",
    "cortical_thickness": "imaging/mr_y_smri__thk__dsk.parquet",
    "subcortical_volume": "imaging/mr_y_smri__vol__aseg.parquet",
}
ICV_COLUMN = "mr_y_smri__vol__aseg__icv_sum"
TASK_FEATURES = {
    "imaging/mr_y_tfmri__sst__beh.parquet": [
        "mr_y_tfmri__sst__beh__go__crct_prop",
        "mr_y_tfmri__sst__beh__go__crct_rt",
        "mr_y_tfmri__sst__beh__stop__crct_prop",
        "mr_y_tfmri__sst__beh__stop__incrct_rt",
        "mr_y_tfmri__sst__beh__ssrt_mean",
        "mr_y_tfmri__sst__beh__ssrt_intgr",
    ],
    "imaging/mr_y_tfmri__nback__beh.parquet": [
        "mr_y_tfmri__nback__beh__0b_acc",
        "mr_y_tfmri__nback__beh__0b_rt",
        "mr_y_tfmri__nback__beh__2b_acc",
        "mr_y_tfmri__nback__beh__2b_rt",
        "mr_y_tfmri__nback__beh_acc",
        "mr_y_tfmri__nback__beh_rt",
    ],
    "imaging/mr_y_tfmri__mid__beh.parquet": [
        "mr_y_tfmri__mid__beh__earn_sum",
        "mr_y_tfmri__mid__beh__trial__loss__crct_prop",
        "mr_y_tfmri__mid__beh__trial__loss__crct_rt",
        "mr_y_tfmri__mid__beh__trial__neut__crct_prop",
        "mr_y_tfmri__mid__beh__trial__neut__crct_rt",
        "mr_y_tfmri__mid__beh__trial__rwrd__crct_prop",
        "mr_y_tfmri__mid__beh__trial__rwrd__crct_rt",
    ],
}
ETIOLOGIC_CONTEXT_FEATURES = {
    "general/ab_g_stc.parquet": [
        "ab_g_stc__cohort_sex",
    ],
    "physical_health/ph_p_dhx.parquet": [
        "ph_p_dhx_birthweight",
        "ph_p_dhx__birth_001",
        "ph_p_dhx__birth_001__01",
        "ph_p_dhx_008",
        "ph_p_dhx__med_009",
        "ph_p_dhx__med_010",
        "ph_p_dhx__alc_001a",
        "ph_p_dhx__alc_001b",
        "ph_p_dhx__nic_001a",
        "ph_p_dhx__nic_001b",
        "ph_p_dhx__mj_001a",
        "ph_p_dhx__mj_001b",
        "ph_p_dhx__vit_001",
    ],
    "mental_health/mh_p_famhx.parquet": [
        "mh_p_famhx__alc_001",
        "mh_p_famhx__dep_001",
        "mh_p_famhx__doc_001",
        "mh_p_famhx__drg_001",
        "mh_p_famhx__halluc_001",
        "mh_p_famhx__hosp_001",
        "mh_p_famhx__mania_001",
        "mh_p_famhx__nerve_001",
        "mh_p_famhx__suic_001",
        "mh_p_famhx__troub_001",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "participant_id"]


def read_baseline_table(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    frame = table_at_session(path, SESSION)
    if frame["participant_id"].duplicated().any():
        raise ValueError(f"Duplicate participant IDs in {path}")
    return frame


def build_corrected_imaging(
    source_dir: Path,
    abcd_root: Path,
    cohort_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    old = pd.read_parquet(source_dir / "imaging.parquet")
    old["participant_id"] = old["participant_id"].astype(str)
    retained = [
        column
        for column in numeric_feature_columns(old)
        if not column.startswith(T1_INTENSITY_PREFIX)
    ]
    removed = [
        column
        for column in numeric_feature_columns(old)
        if column.startswith(T1_INTENSITY_PREFIX)
    ]
    if len(removed) != 71:
        raise ValueError(f"Expected 71 T1-intensity features, found {len(removed)}")
    output = old[["participant_id", *retained]].copy()

    cortical_volume = read_baseline_table(
        abcd_root, STRUCTURAL_TABLES["cortical_volume"]
    )
    cortical_thickness = read_baseline_table(
        abcd_root, STRUCTURAL_TABLES["cortical_thickness"]
    )
    subcortical = read_baseline_table(
        abcd_root, STRUCTURAL_TABLES["subcortical_volume"]
    )
    cortical_volume_columns = [
        column
        for column in numeric_feature_columns(cortical_volume)
        if column.endswith(("__lh_sum", "__rh_sum"))
        and column
        not in {
            "mr_y_smri__vol__dsk__lh_sum",
            "mr_y_smri__vol__dsk__rh_sum",
        }
    ]
    thickness_columns = [
        column
        for column in numeric_feature_columns(cortical_thickness)
        if column.endswith(("__lh_mean", "__rh_mean"))
        and column
        not in {
            "mr_y_smri__thk__dsk__lh_mean",
            "mr_y_smri__thk__dsk__rh_mean",
        }
    ]
    subcortical_columns = [
        column
        for column in numeric_feature_columns(subcortical)
        if column.endswith(("__lh_sum", "__rh_sum"))
    ]
    expected = {
        "cortical volume": (len(cortical_volume_columns), 68),
        "cortical thickness": (len(thickness_columns), 68),
        "subcortical volume": (len(subcortical_columns), 26),
    }
    for name, (actual, target) in expected.items():
        if actual != target:
            raise ValueError(f"Expected {target} {name} columns, found {actual}")
    if ICV_COLUMN not in subcortical:
        raise ValueError(f"Missing ICV column: {ICV_COLUMN}")

    icv = subcortical.set_index("participant_id")[ICV_COLUMN]
    cortical_volume = cortical_volume.set_index("participant_id")
    subcortical = subcortical.set_index("participant_id")
    volume_ratios = cortical_volume[cortical_volume_columns].div(icv, axis=0)
    volume_ratios.columns = [f"{column}__icv_ratio" for column in volume_ratios]
    subcortical_ratios = subcortical[subcortical_columns].div(icv, axis=0)
    subcortical_ratios.columns = [
        f"{column}__icv_ratio" for column in subcortical_ratios
    ]
    structural = pd.concat(
        [
            volume_ratios,
            cortical_thickness.set_index("participant_id")[thickness_columns],
            subcortical_ratios,
            icv.rename(ICV_COLUMN),
        ],
        axis=1,
    ).reset_index()
    structural = structural[structural["participant_id"].isin(cohort_ids)]
    output = output.merge(structural, on="participant_id", how="outer")
    output = output[output["participant_id"].isin(cohort_ids)]
    output = output.dropna(subset=numeric_feature_columns(output), how="all")
    output = output.sort_values("participant_id", kind="stable").reset_index(drop=True)

    receipt = {
        "definition": (
            "baseline rs-fMRI + SST/n-back/MID task-fMRI + correct structural "
            "MRI measures + DTI FA"
        ),
        "correction": (
            "removed regional T1 signal-intensity columns previously mislabeled "
            "as gray-matter volume"
        ),
        "retained_functional_and_dti_features": len(retained),
        "removed_t1_intensity_features": len(removed),
        "added_features": {
            "Desikan cortical volume divided by ICV": len(cortical_volume_columns),
            "Desikan cortical thickness": len(thickness_columns),
            "bilateral subcortical volume divided by ICV": len(subcortical_columns),
            "intracranial volume": 1,
        },
        "normalization": (
            "regional volume/ICV ratios computed before downstream training-only "
            "median imputation and z-score scaling"
        ),
        "sources": [
            {
                "table": relative,
                "sha256": sha256_file(abcd_root / relative),
            }
            for relative in STRUCTURAL_TABLES.values()
        ],
        "feature_count": len(numeric_feature_columns(output)),
        "participants": len(output),
    }
    return output, receipt


def build_task_augmented_cognition(
    source_dir: Path,
    abcd_root: Path,
    cohort_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = pd.read_parquet(source_dir / "cognition_health.parquet")
    output["participant_id"] = output["participant_id"].astype(str)
    source_receipts = []
    for relative, requested in TASK_FEATURES.items():
        frame = read_baseline_table(abcd_root, relative)
        missing = sorted(set(requested) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing requested task features in {relative}: {missing}")
        selected = frame[["participant_id", *requested]].copy()
        selected = selected[selected["participant_id"].isin(cohort_ids)]
        output = output.merge(selected, on="participant_id", how="outer")
        source_receipts.append(
            {
                "table": relative,
                "sha256": sha256_file(abcd_root / relative),
                "features": requested,
                "cohort_rows_with_any_feature": int(
                    selected[requested].notna().any(axis=1).sum()
                ),
            }
        )
    output = output[output["participant_id"].isin(cohort_ids)]
    output = output.dropna(subset=numeric_feature_columns(output), how="all")
    output = output.sort_values("participant_id", kind="stable").reset_index(drop=True)
    receipt = {
        "definition": (
            "v4 cognition/health measures plus compact task-level performance "
            "summaries of inhibitory control, working memory, and reward processing"
        ),
        "original_feature_count": 90,
        "added_task_feature_count": sum(len(value) for value in TASK_FEATURES.values()),
        "task_feature_selection_uses_labels": False,
        "sources": source_receipts,
        "feature_count": len(numeric_feature_columns(output)),
        "participants": len(output),
    }
    return output, receipt


def build_augmented_behavior(
    source_dir: Path,
    abcd_root: Path,
    cohort_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = pd.read_parquet(source_dir / "behavior_environment.parquet")
    output["participant_id"] = output["participant_id"].astype(str)
    source_receipts = []
    for relative, requested in ETIOLOGIC_CONTEXT_FEATURES.items():
        frame = read_baseline_table(abcd_root, relative)
        missing = sorted(set(requested) - set(frame.columns))
        if missing:
            raise ValueError(
                f"Missing requested etiologic-context features in {relative}: {missing}"
            )
        selected = frame[["participant_id", *requested]].copy()
        selected = selected[selected["participant_id"].isin(cohort_ids)]
        duplicate = sorted(set(output.columns) & set(requested))
        if duplicate:
            raise ValueError(f"Etiologic-context features already present: {duplicate}")
        output = output.merge(selected, on="participant_id", how="outer")
        source_receipts.append(
            {
                "table": relative,
                "sha256": sha256_file(abcd_root / relative),
                "features": requested,
                "cohort_rows_with_any_feature": int(
                    selected[requested].notna().any(axis=1).sum()
                ),
            }
        )
    output = output[output["participant_id"].isin(cohort_ids)]
    output = output.dropna(subset=numeric_feature_columns(output), how="all")
    output = output.sort_values("participant_id", kind="stable").reset_index(drop=True)
    receipt = {
        "definition": (
            "v4 behavior/environment measures plus sex, developmental/perinatal "
            "history, and broad familial psychopathology history"
        ),
        "rationale": (
            "prespecified ADHD-relevant etiologic context; no child ADHD symptom, "
            "diagnosis, medication, or CBCL attention score"
        ),
        "original_feature_count": 221,
        "added_feature_count": sum(
            len(value) for value in ETIOLOGIC_CONTEXT_FEATURES.values()
        ),
        "feature_selection_uses_labels": False,
        "sources": source_receipts,
        "feature_count": len(numeric_feature_columns(output)),
        "participants": len(output),
    }
    return output, receipt


def load_gene_map(bim_path: Path, ranges_path: Path) -> dict[str, str]:
    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["chromosome", "variant", "cm", "position", "a1", "a2"],
        dtype={"chromosome": str, "variant": str},
    )
    ranges = pd.read_csv(
        ranges_path,
        sep=r"\s+",
        header=None,
        names=["chromosome", "start", "end", "gene"],
        dtype={"chromosome": str, "gene": str},
    )
    ranges["chromosome"] = ranges["chromosome"].str.removeprefix("chr")
    ranges["chromosome"] = ranges["chromosome"].replace({"X": "23"})
    mapping: dict[str, str] = {}
    for row in ranges.itertuples(index=False):
        members = bim[
            bim["chromosome"].eq(row.chromosome)
            & bim["position"].between(row.start, row.end)
        ]["variant"]
        for variant in members:
            if variant in mapping and mapping[variant] != row.gene:
                raise ValueError(f"Variant {variant} belongs to overlapping gene windows")
            mapping[variant] = row.gene
    return mapping


def build_gene_pca(
    source_dir: Path,
    missingness_path: Path,
    bim_path: Path,
    ranges_path: Path,
    components_per_gene: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_parquet(source_dir / "genetic.parquet")
    raw["participant_id"] = raw["participant_id"].astype(str)
    raw = raw.set_index("participant_id")
    splits = json.loads((source_dir / "splits.json").read_text())
    train_ids = set(map(str, splits["training"]))
    missingness = pd.read_parquet(missingness_path)
    missingness["participant_id"] = missingness["participant_id"].astype(str)
    if "G" not in missingness:
        raise ValueError("Missingness table has no G keep-mask column")
    observed_training_ids = set(
        missingness.loc[missingness["G"].astype(bool), "participant_id"]
    ) & train_ids & set(raw.index)
    if not observed_training_ids:
        raise ValueError("No observed training genotypes available for gene PCA")

    variant_to_gene = load_gene_map(bim_path, ranges_path)
    gene_columns: dict[str, list[str]] = {}
    unmapped = []
    for column in raw.columns:
        variant = column.rsplit("_", 1)[0]
        gene = variant_to_gene.get(variant)
        if gene is None:
            unmapped.append(column)
        else:
            gene_columns.setdefault(gene, []).append(column)
    if unmapped:
        raise ValueError(f"Unmapped retained SNP dosage columns: {unmapped[:10]}")

    transformed = []
    gene_receipts: dict[str, object] = {}
    fit_ids = sorted(observed_training_ids)
    for gene in sorted(gene_columns):
        columns = gene_columns[gene]
        train = raw.loc[fit_ids, columns].apply(pd.to_numeric, errors="coerce")
        medians = train.median(axis=0)
        usable = medians[medians.notna()].index.tolist()
        if not usable:
            raise ValueError(f"No usable training SNPs for {gene}")
        train_array = train[usable].fillna(medians[usable]).to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(train_array)
        train_scaled = scaler.transform(train_array)
        n_components = min(components_per_gene, len(usable), len(fit_ids))
        pca = PCA(n_components=n_components, svd_solver="full").fit(train_scaled)
        all_array = (
            raw[usable]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(medians[usable])
            .to_numpy(dtype=np.float64)
        )
        scores = pca.transform(scaler.transform(all_array)).astype(np.float32)
        names = [f"snp_{gene}__pc{index:02d}" for index in range(1, n_components + 1)]
        transformed.append(pd.DataFrame(scores, index=raw.index, columns=names))
        gene_receipts[gene] = {
            "input_retained_snps": len(columns),
            "usable_training_snps": len(usable),
            "components": n_components,
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
        }
    output = pd.concat(transformed, axis=1).reset_index()
    output = output.sort_values("participant_id", kind="stable").reset_index(drop=True)
    receipt = {
        "definition": (
            "within-gene principal components derived exclusively from directly "
            "observed SNP dosages in 14 prespecified candidate-gene windows"
        ),
        "external_prs_features": 0,
        "ancestry_pc_features": 0,
        "label_supervision_in_representation": False,
        "fit_scope": "training participants with G observed under the fixed keep mask",
        "fit_participants": len(fit_ids),
        "components_per_gene_cap": components_per_gene,
        "input_snp_features": len(raw.columns),
        "output_features": len(numeric_feature_columns(output)),
        "gene_details": gene_receipts,
        "bim_sha256": sha256_file(bim_path),
        "gene_ranges_sha256": sha256_file(ranges_path),
    }
    return output, receipt


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/abcd_adhd_presentation3_snp_v4"),
    )
    parser.add_argument(
        "--source-missing-dir",
        type=Path,
        default=Path("data/abcd_adhd_presentation3_snp_v4_random_missing15"),
    )
    parser.add_argument(
        "--abcd-root", type=Path, required=True
    )
    parser.add_argument(
        "--genotype-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/abcd_adhd_presentation3_refined_v5_random_missing15"),
    )
    parser.add_argument("--components-per-gene", type=int, default=8)
    parser.add_argument(
        "--raw-snp-dosages",
        action="store_true",
        help="retain the v4 training-QC/LD-pruned additive SNP dosage columns",
    )
    parser.add_argument(
        "--exclude-task-behavior",
        action="store_true",
        help="retain the original v4 cognition/health modality without task summaries",
    )
    parser.add_argument(
        "--add-etiologic-context",
        action="store_true",
        help=(
            "add prespecified sex, perinatal/developmental, and family-history "
            "variables to B"
        ),
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    source_missing_dir = args.source_missing_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(source_dir / "labels.parquet")
    labels["participant_id"] = labels["participant_id"].astype(str)
    cohort_ids = set(labels["participant_id"])
    missingness_source = source_missing_dir / "missingness.parquet"
    bim_path = args.genotype_root / "target_gene_variants.bim"
    ranges_path = args.genotype_root / "gene_coordinates_auto.bed"

    imaging, imaging_receipt = build_corrected_imaging(
        source_dir, args.abcd_root, cohort_ids
    )
    if args.exclude_task_behavior:
        cognition = pd.read_parquet(source_dir / "cognition_health.parquet")
        cognition_receipt = {
            "definition": "unchanged v4 cognition/health modality",
            "original_feature_count": 90,
            "added_task_feature_count": 0,
        }
    else:
        cognition, cognition_receipt = build_task_augmented_cognition(
            source_dir, args.abcd_root, cohort_ids
        )
    if args.raw_snp_dosages:
        genetic = pd.read_parquet(source_dir / "genetic.parquet")
        genetic_receipt = {
            "definition": (
                "v4 direct additive SNP dosages after training-only variant QC "
                "and LD pruning"
            ),
            "external_prs_features": 0,
            "ancestry_pc_features": 0,
            "label_supervision_in_representation": False,
            "input_snp_features": len(numeric_feature_columns(genetic)),
            "output_features": len(numeric_feature_columns(genetic)),
            "source_sha256": sha256_file(source_dir / "genetic.parquet"),
        }
    else:
        genetic, genetic_receipt = build_gene_pca(
            source_dir,
            missingness_source,
            bim_path,
            ranges_path,
            args.components_per_gene,
        )
    if args.add_etiologic_context:
        behavior, behavior_receipt = build_augmented_behavior(
            source_dir, args.abcd_root, cohort_ids
        )
    else:
        behavior = pd.read_parquet(source_dir / "behavior_environment.parquet")
        behavior_receipt = {
            "definition": "unchanged v4 behavior/environment modality",
            "original_feature_count": 221,
            "added_feature_count": 0,
        }

    imaging.to_parquet(output_dir / "imaging.parquet", index=False)
    genetic.to_parquet(output_dir / "genetic.parquet", index=False)
    cognition.to_parquet(output_dir / "cognition_health.parquet", index=False)
    behavior.to_parquet(output_dir / "behavior_environment.parquet", index=False)
    for name in [
        "labels.parquet",
        "splits.json",
    ]:
        copy_file(source_dir / name, output_dir / name)
    copy_file(missingness_source, output_dir / "missingness.parquet")
    copy_file(
        source_missing_dir / "missingness_summary.json",
        output_dir / "missingness_summary.json",
    )

    frames = {
        "I": imaging,
        "G": genetic,
        "C": cognition,
        "B": behavior,
    }
    names = {
        "I": "imaging",
        "G": "genetic",
        "C": "cognition_health",
        "B": "behavior_environment",
    }
    paths = {
        "I": "imaging.parquet",
        "G": "genetic.parquet",
        "C": "cognition_health.parquet",
        "B": "behavior_environment.parquet",
    }
    modalities = []
    for code in "IGCB":
        frame = frames[code]
        frame["participant_id"] = frame["participant_id"].astype(str)
        features = numeric_feature_columns(frame)
        modalities.append(
            {
                "code": code,
                "name": names[code],
                "path": paths[code],
                "max_missing": 0.8,
                "min_variance": 1e-8,
                "feature_columns": features,
            }
        )

    old_manifest = json.loads((source_dir / "manifest.json").read_text())
    old_missing_manifest = json.loads(
        (source_missing_dir / "manifest.json").read_text()
    )
    manifest = {
        "dataset": "abcd",
        "id_column": "participant_id",
        "label": {
            "path": "labels.parquet",
            "column": "target",
            "class_values": [0, 1, 2],
        },
        "splits": "splits.json",
        "modalities": modalities,
        "provenance": {
            **old_manifest.get("provenance", {}),
            "representation_version": "refined_v5",
            "cohort_and_split_frozen_from": str(source_dir),
            "data_refinement_uses_testing_labels": False,
            "modality_grouping": {
                "I": (
                    "rs-fMRI + SST/n-back/MID task-fMRI + cortical volume/ICV + "
                    "cortical thickness + subcortical volume/ICV + ICV + DTI FA"
                ),
                "G": "training-fitted within-gene PCA of direct SNP dosages only",
                "C": (
                    "cognition, health, and SST/n-back/MID task-performance summaries"
                ),
                "B": (
                    "non-ADHD psychopathology, impulsivity/reward, family/SES/"
                    "neighborhood, and physical environment"
                ),
            },
            "imaging_receipt": imaging_receipt,
            "genetic_receipt": genetic_receipt,
            "cognition_receipt": cognition_receipt,
            "behavior_receipt": behavior_receipt,
        },
        "missingness": old_missing_manifest["missingness"] | {
            "path": "missingness.parquet"
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    missingness = pd.read_parquet(output_dir / "missingness.parquet")
    if set(missingness["participant_id"].astype(str)) != cohort_ids:
        raise ValueError("Fixed missingness participant IDs differ from frozen cohort")
    if (~missingness[list("IGCB")].astype(bool)).all(axis=1).any():
        raise ValueError("Fixed missingness table contains an all-missing participant")
    splits = json.loads((output_dir / "splits.json").read_text())
    summary = {
        "participants": len(labels),
        "class_counts": {
            str(key): int(value)
            for key, value in labels["target"].value_counts().sort_index().items()
        },
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "raw_dimensions": {
            names[code]: len(numeric_feature_columns(frames[code])) for code in "IGCB"
        },
        "coverage": {
            names[code]: len(frames[code]) for code in "IGCB"
        },
        "fixed_missingness_sha256": sha256_file(output_dir / "missingness.parquet"),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "representation_receipt.json").write_text(
        json.dumps(
            {
                "imaging": imaging_receipt,
                "genetic": genetic_receipt,
                "cognition_health": cognition_receipt,
                "behavior_environment": behavior_receipt,
                "output_sha256": {
                    path: sha256_file(output_dir / path)
                    for path in paths.values()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
