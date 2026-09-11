#!/usr/bin/env python3
"""Shared ABCD imaging, genetic, and environment feature-building functions.

Variant eligibility and LD pruning are fitted on the fixed training split.  No
label is read while selecting genetic or environmental predictors.  The
downstream manifest loader remains responsible for training-only median
imputation and z-score standardization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


ENVIRONMENT_FEATURES = {
    "external_linked_data/le_l_adi.parquet": {
        "le_l_adi__addr1__national_prcnt": "Area Deprivation Index national percentile",
    },
    "external_linked_data/le_l_coi.parquet": {
        "le_l_coi__addr1__coi__total__national_zscore": "overall Child Opportunity Index z-score",
        "le_l_coi__addr1__ed__total__national_zscore": "education Child Opportunity Index z-score",
        "le_l_coi__addr1__he__total__national_zscore": "health/environment Child Opportunity Index z-score",
        "le_l_coi__addr1__se__total__national_zscore": "social/economic Child Opportunity Index z-score",
    },
    "external_linked_data/le_l_leadrisk.parquet": {
        "le_l_leadrisk__addr1_idx": "census-tract lead-risk index",
    },
    "external_linked_data/le_l_noise.parquet": {
        "le_l_noise__addr1__24hmean__total_soundlvl": "day-night average sound level",
    },
    "external_linked_data/le_l_pm25.parquet": {
        "le_l_pm25__addr1__pm25_mean__2016": "2016 annual mean PM2.5",
    },
    "external_linked_data/le_l_no2.parquet": {
        "le_l_no2__addr1__no2_mean__2016": "2016 annual mean NO2",
    },
    "external_linked_data/le_l_nata.parquet": {
        "le_l_nata__addr1__resphaz_idx": "National Air Toxics respiratory-hazard index",
    },
    "external_linked_data/le_l_roadprox.parquet": {
        "le_l_roadprox__addr1_m": "distance to a major road",
    },
    "external_linked_data/le_l_parks.parquet": {
        "le_l_parks__addr1__parks_prop": "census-tract park-land proportion",
    },
}


FUNCTIONAL_IMAGING_TABLES = {
    "imaging/mr_y_rsfmri__corr__gpnet.parquet": (
        "resting-state Gordon network-to-network connectivity"
    ),
    "imaging/mr_y_rsfmri__corr__gpnet__aseg.parquet": (
        "resting-state Gordon network-to-subcortical connectivity"
    ),
    "imaging/mr_y_rsfmri__var__dsk.parquet": (
        "resting-state Desikan cortical temporal variance"
    ),
    "imaging/mr_y_rsfmri__var__aseg.parquet": (
        "resting-state subcortical temporal variance"
    ),
    "imaging/mr_y_tfmri__sst__csvcg__dsk.parquet": (
        "SST correct-stop versus correct-go cortical beta"
    ),
    "imaging/mr_y_tfmri__nback__2bv0b__dsk.parquet": (
        "n-back 2-back versus 0-back cortical beta"
    ),
    "imaging/mr_y_tfmri__mid__arvn__dsk.parquet": (
        "MID reward anticipation versus neutral cortical beta"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def copy_base_dataset(base: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "labels.parquet",
        "splits.json",
        "imaging.parquet",
        "cognition_health.parquet",
        "behavior_environment.parquet",
        "manifest.json",
        "summary.json",
    ):
        shutil.copy2(base / name, output / name)


def read_gene_ranges(path: Path) -> list[tuple[int, int, int, str]]:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chromosome", "start", "end", "gene"],
    )
    frame["chromosome"] = (
        frame["chromosome"].astype(str).str.removeprefix("chr").replace({"X": "23"}).astype(int)
    )
    return list(frame.itertuples(index=False, name=None))


def build_genetic_modality(
    *,
    cohort_ids: set[str],
    train_ids: set[str],
    abcd_root: Path,
    genotype_prefix: Path,
    gene_ranges_path: Path,
    plink: Path,
    output: Path,
    geno: float,
    maf: float,
    ld_window: int,
    ld_step: int,
    ld_r2: float,
) -> tuple[pd.DataFrame, dict]:
    qc_dir = output / "genetic_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    fam = pd.read_csv(
        genotype_prefix.with_suffix(".fam"),
        sep=r"\s+",
        header=None,
        names=["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"],
        dtype={"FID": str, "IID": str},
    )
    train_keep = fam[fam["IID"].isin(train_ids)][["FID", "IID"]]
    if train_keep.empty:
        raise ValueError("No training participant matches the genotype FAM file")
    keep_path = qc_dir / "training.keep"
    train_keep.to_csv(keep_path, sep="\t", header=False, index=False)

    prune_prefix = qc_dir / "training_qc_ld"
    run(
        [
            str(plink),
            "--bfile", str(genotype_prefix),
            "--keep", str(keep_path),
            "--geno", str(geno),
            "--maf", str(maf),
            "--indep-pairwise", str(ld_window), str(ld_step), str(ld_r2),
            "--out", str(prune_prefix),
        ]
    )
    selected_variants = [
        line.strip()
        for line in prune_prefix.with_suffix(".prune.in").read_text().splitlines()
        if line.strip()
    ]
    if not selected_variants:
        raise ValueError("Training-only genotype QC retained no variant")

    recode_prefix = qc_dir / "candidate_gene_dosages"
    run(
        [
            str(plink),
            "--bfile", str(genotype_prefix),
            "--extract", str(prune_prefix.with_suffix(".prune.in")),
            "--recode", "A",
            "--out", str(recode_prefix),
        ]
    )
    dosage = pd.read_csv(recode_prefix.with_suffix(".raw"), sep=r"\s+", dtype={"IID": str})
    dosage["IID"] = dosage["IID"].astype(str)
    dosage = dosage[dosage["IID"].isin(cohort_ids)].copy()
    dosage_columns = dosage.columns[6:].tolist()
    genetic = dosage[["IID", *dosage_columns]].rename(columns={"IID": "participant_id"})

    pc_path = abcd_root / "genetic/gn_y_popstruct.parquet"
    pcs = pd.read_parquet(pc_path)
    if "session_id" in pcs:
        pcs = pcs[pcs["session_id"].astype(str).eq("ses-00A")]
    pc_columns = [column for column in pcs if column.startswith("gn_y_popstruct_pc__")]
    pcs = pcs[["participant_id", *pc_columns]].drop_duplicates("participant_id")
    pcs["participant_id"] = pcs["participant_id"].astype(str)
    genetic = genetic.merge(pcs, on="participant_id", how="left", validate="one_to_one")
    genetic.to_parquet(output / "genetic.parquet", index=False)

    bim = pd.read_csv(
        genotype_prefix.with_suffix(".bim"),
        sep=r"\s+",
        header=None,
        names=["chromosome", "variant", "cm", "position", "a1", "a2"],
    )
    selected_bim = bim[bim["variant"].astype(str).isin(selected_variants)].copy()
    gene_ranges = read_gene_ranges(gene_ranges_path)
    gene_counts: Counter[str] = Counter()
    unmapped: list[str] = []
    for row in selected_bim.itertuples(index=False):
        genes = [
            gene
            for chromosome, start, end, gene in gene_ranges
            if int(row.chromosome) == chromosome and start <= int(row.position) <= end
        ]
        if genes:
            gene_counts.update(genes)
        else:
            unmapped.append(str(row.variant))

    receipt = {
        "definition": "direct additive dosages in 14 prespecified ADHD candidate-gene windows plus ancestry PCs",
        "candidate_genes": [item[3] for item in gene_ranges],
        "gene_windows": "gene coordinates extended by 20 kb in the supplied hg19 range file",
        "variant_selection_uses_labels": False,
        "training_genotyped_participants": int(len(train_keep)),
        "cohort_genotyped_participants": int(len(genetic)),
        "cohort_genotype_coverage": float(len(genetic) / len(cohort_ids)),
        "input_variants": int(len(bim)),
        "retained_variants": int(len(selected_variants)),
        "dosage_features": int(len(dosage_columns)),
        "ancestry_pc_features": int(len(pc_columns)),
        "training_only_qc": {
            "variant_missingness_max": geno,
            "minor_allele_frequency_min": maf,
            "ld_pruning": f"--indep-pairwise {ld_window} {ld_step} {ld_r2}",
        },
        "gene_variant_counts_after_qc_and_ld_pruning": dict(sorted(gene_counts.items())),
        "unmapped_retained_variants": unmapped,
        "inputs": {
            "bed_sha256": sha256_file(genotype_prefix.with_suffix(".bed")),
            "bim_sha256": sha256_file(genotype_prefix.with_suffix(".bim")),
            "fam_sha256": sha256_file(genotype_prefix.with_suffix(".fam")),
            "gene_ranges_sha256": sha256_file(gene_ranges_path),
            "population_structure_sha256": sha256_file(pc_path),
        },
        "output_sha256": sha256_file(output / "genetic.parquet"),
        "downstream_preprocessing": "training-only median imputation and z-score scaling in the manifest loader",
    }
    return genetic, receipt


def build_environment_modality(
    *, base: Path, abcd_root: Path, cohort_ids: set[str], output: Path
) -> tuple[pd.DataFrame, dict]:
    behavior = pd.read_parquet(base / "behavior_environment.parquet")
    behavior["participant_id"] = behavior["participant_id"].astype(str)
    source_receipts = []
    selected_columns: list[str] = []
    for relative, definitions in ENVIRONMENT_FEATURES.items():
        path = abcd_root / relative
        frame = pd.read_parquet(path)
        if "session_id" in frame:
            frame = frame[frame["session_id"].astype(str).eq("ses-00A")]
        missing = sorted(set(definitions) - set(frame.columns))
        if missing:
            raise ValueError(f"{relative} is missing selected columns: {missing}")
        frame = frame[["participant_id", *definitions]].copy()
        frame["participant_id"] = frame["participant_id"].astype(str)
        frame = frame[frame["participant_id"].isin(cohort_ids)]
        if frame["participant_id"].duplicated().any():
            raise ValueError(f"Duplicate baseline participant in {relative}")
        behavior = behavior.merge(frame, on="participant_id", how="left", validate="one_to_one")
        selected_columns.extend(definitions)
        source_receipts.append(
            {
                "table": relative,
                "sha256": sha256_file(path),
                "features": definitions,
                "cohort_rows_with_any_selected_exposure": int(
                    frame[list(definitions)].notna().any(axis=1).sum()
                ),
            }
        )
    behavior.to_parquet(output / "behavior_environment.parquet", index=False)
    receipt = {
        "selection_uses_labels": False,
        "address": "baseline primary residential address (addr1) only",
        "rationale": (
            "prespecified low-dimensional summaries spanning socioeconomic opportunity, "
            "air/noise/lead exposure, traffic proximity, and green space"
        ),
        "added_features": selected_columns,
        "added_feature_definitions": {
            column: description
            for definitions in ENVIRONMENT_FEATURES.values()
            for column, description in definitions.items()
        },
        "sources": source_receipts,
        "output_sha256": sha256_file(output / "behavior_environment.parquet"),
        "downstream_preprocessing": "training-only median imputation and z-score scaling in the manifest loader",
    }
    return behavior, receipt


def functional_columns(relative: str, frame: pd.DataFrame) -> list[str]:
    candidates = [
        column
        for column in frame.columns
        if column not in {"participant_id", "session_id"}
        and not column.endswith("_dtt")
    ]
    if "tfmri" in relative:
        # Each task table contains beta and t-statistic for the same 68 Desikan
        # regions.  Retain beta estimates only to avoid duplicating one GLM
        # contrast in two numerical forms.
        return [column for column in candidates if column.endswith("_beta")]
    if relative.endswith("corr__gpnet.parquet"):
        # The network-by-network table stores both A-B and B-A. Keep one
        # triangle (including within-network connectivity) exactly once.
        selected = []
        for column in candidates:
            tail = column.split("gpnet__", 1)[1]
            source, target = tail.split("__", 1)
            target = target.removesuffix("_mean")
            if source <= target:
                selected.append(column)
        return selected
    return candidates


def build_functional_imaging_modality(
    *, abcd_root: Path, cohort_ids: set[str], output: Path
) -> tuple[pd.DataFrame, dict]:
    imaging: pd.DataFrame | None = None
    sources = []
    for relative, definition in FUNCTIONAL_IMAGING_TABLES.items():
        path = abcd_root / relative
        frame = pd.read_parquet(path)
        if "session_id" in frame:
            frame = frame[frame["session_id"].astype(str).eq("ses-00A")]
        columns = functional_columns(relative, frame)
        frame = frame[["participant_id", *columns]].copy()
        frame["participant_id"] = frame["participant_id"].astype(str)
        frame = frame[frame["participant_id"].isin(cohort_ids)]
        if frame["participant_id"].duplicated().any():
            raise ValueError(f"Duplicate baseline participant in {relative}")
        imaging = (
            frame
            if imaging is None
            else imaging.merge(frame, on="participant_id", how="outer", validate="one_to_one")
        )
        sources.append(
            {
                "table": relative,
                "definition": definition,
                "sha256": sha256_file(path),
                "features": len(columns),
                "cohort_rows": int(len(frame)),
            }
        )
    assert imaging is not None
    imaging.to_parquet(output / "imaging.parquet", index=False)
    receipt = {
        "definition": "baseline resting-state and task fMRI summaries only",
        "structural_mri_included": False,
        "diffusion_mri_included": False,
        "resting_state": (
            "unique Gordon network-to-network correlations plus Gordon "
            "network-to-subcortical correlations and cortical/subcortical "
            "regional temporal variance"
        ),
        "task_fmri": (
            "all-run regional beta estimates for SST correct stop vs correct go, "
            "n-back 2-back vs 0-back, and MID reward anticipation vs neutral"
        ),
        "task_t_statistics_included": False,
        "sources": sources,
        "participants_with_any_functional_imaging": int(len(imaging)),
        "features": int(imaging.shape[1] - 1),
        "output_sha256": sha256_file(output / "imaging.parquet"),
        "downstream_preprocessing": "training-only median imputation and z-score scaling in the manifest loader",
    }
    return imaging, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--abcd-root", type=Path, required=True)
    parser.add_argument("--genotype-prefix", type=Path, required=True)
    parser.add_argument("--gene-ranges", type=Path, required=True)
    parser.add_argument(
        "--preprocessing-keep-mask",
        type=Path,
        help=(
            "Optional fixed missingness table. When supplied, participants whose "
            "G flag is false are excluded before fitting genetic QC and LD pruning."
        ),
    )
    parser.add_argument("--plink", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geno", type=float, default=0.02)
    parser.add_argument("--maf", type=float, default=0.01)
    parser.add_argument("--ld-window", type=int, default=50)
    parser.add_argument("--ld-step", type=int, default=5)
    parser.add_argument("--ld-r2", type=float, default=0.5)
    args = parser.parse_args()

    base = args.base_dataset.resolve()
    abcd_root = args.abcd_root.resolve()
    output = args.output_dir.resolve()
    copy_base_dataset(base, output)
    labels = pd.read_parquet(output / "labels.parquet")
    cohort_ids = set(labels["participant_id"].astype(str))
    splits = json.loads((output / "splits.json").read_text())
    train_ids = set(map(str, splits["training"]))
    if args.preprocessing_keep_mask is not None:
        missingness = pd.read_parquet(args.preprocessing_keep_mask.resolve())
        if not {"participant_id", "G"}.issubset(missingness.columns):
            raise ValueError("Preprocessing keep mask must contain participant_id and G")
        g_observed = set(
            missingness.loc[missingness["G"].astype(bool), "participant_id"].astype(str)
        )
        train_ids &= g_observed

    genetic, genetic_receipt = build_genetic_modality(
        cohort_ids=cohort_ids,
        train_ids=train_ids,
        abcd_root=abcd_root,
        genotype_prefix=args.genotype_prefix.resolve(),
        gene_ranges_path=args.gene_ranges.resolve(),
        plink=args.plink.resolve(),
        output=output,
        geno=args.geno,
        maf=args.maf,
        ld_window=args.ld_window,
        ld_step=args.ld_step,
        ld_r2=args.ld_r2,
    )
    behavior, environment_receipt = build_environment_modality(
        base=base,
        abcd_root=abcd_root,
        cohort_ids=cohort_ids,
        output=output,
    )
    imaging, imaging_receipt = build_functional_imaging_modality(
        abcd_root=abcd_root,
        cohort_ids=cohort_ids,
        output=output,
    )

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for spec in manifest["modalities"]:
        if spec["code"] == "I":
            spec["path"] = "imaging.parquet"
            spec["feature_columns"] = [
                column for column in imaging.columns if column != "participant_id"
            ]
            spec["source_tables"] = list(FUNCTIONAL_IMAGING_TABLES)
        elif spec["code"] == "G":
            spec["path"] = "genetic.parquet"
            spec["feature_columns"] = [
                column for column in genetic.columns if column != "participant_id"
            ]
            spec["source_tables"] = [
                str(args.genotype_prefix.resolve()) + ".{bed,bim,fam}",
                "genetic/gn_y_popstruct.parquet",
            ]
        elif spec["code"] == "B":
            spec["path"] = "behavior_environment.parquet"
            spec["feature_columns"] = [
                column for column in behavior.columns if column != "participant_id"
            ]
            spec["source_tables"] = [
                *spec["source_tables"],
                *ENVIRONMENT_FEATURES,
            ]
    manifest["provenance"]["abcd_root"] = str(abcd_root)
    manifest["provenance"]["modality_grouping"]["I"] = (
        "resting-state functional connectivity and regional temporal variance "
        "+ SST/n-back/MID task-fMRI contrasts"
    )
    manifest["provenance"]["modality_grouping"]["G"] = (
        "additive dosages in 14 prespecified ADHD candidate-gene windows plus ancestry PCs"
    )
    manifest["provenance"]["modality_grouping"]["B"] = (
        "non-K-SADS mental health + family/neighborhood context + linked physical environment"
    )
    manifest["provenance"]["genetic_processing"] = genetic_receipt
    manifest["provenance"]["genetic_processing"]["preprocessing_keep_mask"] = (
        str(args.preprocessing_keep_mask.resolve())
        if args.preprocessing_keep_mask is not None
        else None
    )
    manifest["provenance"]["environment_processing"] = environment_receipt
    manifest["provenance"]["functional_imaging_processing"] = imaging_receipt
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["coverage"]["imaging"] = int(len(imaging))
    summary["raw_dimensions"]["imaging"] = int(imaging.shape[1] - 1)
    summary["coverage"]["genetic"] = int(len(genetic))
    summary["raw_dimensions"]["genetic"] = int(genetic.shape[1] - 1)
    summary["raw_dimensions"]["behavior_environment"] = int(behavior.shape[1] - 1)
    summary["genetic_processing"] = genetic_receipt
    summary["environment_processing"] = environment_receipt
    summary["functional_imaging_processing"] = imaging_receipt
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    (output / "genetic_processing_receipt.json").write_text(
        json.dumps(genetic_receipt, indent=2) + "\n"
    )
    (output / "environment_processing_receipt.json").write_text(
        json.dumps(environment_receipt, indent=2) + "\n"
    )
    (output / "functional_imaging_processing_receipt.json").write_text(
        json.dumps(imaging_receipt, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str(output),
        "participants": len(cohort_ids),
        "functional_imaging_rows": len(imaging),
        "functional_imaging_features": imaging.shape[1] - 1,
        "genetic_rows": len(genetic),
        "genetic_features": genetic.shape[1] - 1,
        "retained_variants": genetic_receipt["retained_variants"],
        "behavior_environment_features": behavior.shape[1] - 1,
        "added_environment_features": len(environment_receipt["added_features"]),
    }, indent=2))


if __name__ == "__main__":
    main()
