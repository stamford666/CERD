#!/usr/bin/env python3
"""Build leakage-controlled multimodal ABCD classification datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_abcd_ed import build_bed_course_labels, build_labels, split_ids


SOURCES = {
    "S": {
        "name": "structural_mri",
        "tables": ["imaging/mr_y_smri__t1__gm__dsk.parquet"],
    },
    "R": {
        "name": "resting_fmri",
        "tables": ["imaging/mr_y_rsfmri__var__dsk.parquet"],
    },
    "D": {
        "name": "diffusion_mri",
        "tables": ["imaging/mr_y_dti__is__fa__wm__dsk.parquet"],
    },
    "G": {
        "name": "genetic",
        "tables": ["genetic/gn_y_popstruct.parquet"],
    },
    "N": {
        "name": "neurocognition",
        "tables": [
            "neurocognition/nc_y_nihtb.parquet",
            "neurocognition/nc_y_lmt.parquet",
            "neurocognition/nc_y_ravlt.parquet",
        ],
    },
    "P": {
        "name": "physical_health",
        "tables": [
            "physical_health/ph_y_anthr.parquet",
            "physical_health/ph_p_pds.parquet",
            "physical_health/ph_y_pds.parquet",
            "physical_health/ph_y_bld.parquet",
        ],
    },
    "M": {
        "name": "mental_health",
        # The K-SADS ED target table and all direct ED instruments are excluded.
        "tables": [
            "mental_health/mh_p_cbcl.parquet",
            "mental_health/mh_y_bisbas.parquet",
            "mental_health/mh_y_upps.parquet",
            "mental_health/mh_y_bpm.parquet",
        ],
    },
    "E": {
        "name": "environment",
        "tables": [
            "general/ab_p_demo.parquet",
            "friends_family_community/fc_p_fes.parquet",
            "friends_family_community/fc_p_nsc.parquet",
            "social_development/sdev_p_nbh.parquet",
            "social_development/sdev_p_apq.parquet",
        ],
    },
}

# Four aggregated modalities with the same I/G/C/B interface used by ADNI.
# The grouping keeps all ABCD predictor tables while making the modality count
# and missing-genomics experiment directly comparable to the original method.
ADNI4_SOURCES = {
    "I": {
        "name": "imaging",
        "tables": [
            *SOURCES["S"]["tables"],
            *SOURCES["R"]["tables"],
            *SOURCES["D"]["tables"],
        ],
    },
    "G": {
        "name": "genetic",
        "tables": SOURCES["G"]["tables"],
    },
    "C": {
        "name": "cognition_health",
        "tables": [
            *SOURCES["N"]["tables"],
            *SOURCES["P"]["tables"],
        ],
    },
    "B": {
        "name": "behavior_environment",
        "tables": [
            *SOURCES["M"]["tables"],
            *SOURCES["E"]["tables"],
        ],
    },
}

# Diagnostic domains used for the ADNI-like three-class ABCD task. The target
# is ordinal transdiagnostic burden (0 domains, 1 domain, >=2 domains), not an
# attempt to relabel children with adult CN/MCI/AD diagnoses.
DIAGNOSIS_DOMAINS = (
    "adhd", "agor", "bpd", "cond", "dep", "dmdd", "ed", "gad",
    "ocd", "odd", "panic", "phobia", "psych", "ptsd", "sepanx", "socanx",
)

COGNITION_TARGET_TABLE = "neurocognition/nc_y_nihtb.parquet"
COGNITION_TARGET_COLUMN = "nc_y_nihtb__comp__tot__agecor_score"
ADHD_TARGET_TABLE = "mental_health/mh_p_ksads__adhd.parquet"
ADHD_TARGET_COLUMNS = (
    "mh_p_ksads__adhd__pres_dx",
    "mh_p_ksads__adhd__past_dx",
)
ADHD_CONTROL_EXCLUSION_COLUMNS = (
    "mh_p_ksads__adhd__partrem_dx",
    "mh_p_ksads__adhd__unspec_dx",
    "mh_p_ksads__adhd__fullrem_dx",
    "mh_p_ksads__adhd__oth_dx",
)
ADHD_VALID_DIAGNOSIS_VALUES = (0, 1, 888)

# The aligned three-status endpoint uses the current K-SADS symptom domains.
# Four symptoms in either nine-item domain is a commonly used research
# boundary for elevated/subthreshold childhood ADHD.  It is not relabeled as a
# clinical diagnosis: released present/past/remission diagnosis fields remain
# authoritative for the status branches below.
ADHD_INTERMEDIATE_SYMPTOM_THRESHOLD = 4
ADHD_CURRENT_INATTENTIVE_COLUMNS = (
    "mh_p_ksads__adhd__mistake__pres_sx",
    "mh_p_ksads__adhd__sustattn__grdschl__pres_sx",
    "mh_p_ksads__adhd__notlisten__pres_sx",
    "mh_p_ksads__adhd__flwinstr__pres_sx",
    "mh_p_ksads__adhd__orgtask__pres_sx",
    "mh_p_ksads__adhd__avoid__task__pres_sx",
    "mh_p_ksads__adhd__loses__pres_sx",
    "mh_p_ksads__adhd__distract__grdschl__pres_sx",
    "mh_p_ksads__adhd__forget__pres_sx",
)
ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS = (
    "mh_p_ksads__adhd__fidget__pres_sx",
    "mh_p_ksads__adhd__seat__grdschl__pres_sx",
    "mh_p_ksads__adhd__hypractv__pres_sx",
    "mh_p_ksads__adhd__quiet__pres_sx",
    "mh_p_ksads__adhd__motor__pres_sx",
    "mh_p_ksads__adhd__talkexcess__pres_sx",
    "mh_p_ksads__adhd__blurt__pres_sx",
    "mh_p_ksads__adhd__wait__pres_sx",
    "mh_p_ksads__adhd__interrupt__pres_sx",
)

# Optional presentation-only endpoint support.  The main aligned experiment
# below uses the current-status endpoint, but keeping these explicit aliases
# makes the clinically standard three-presentation derivation reproducible.
ADHD_PRESENTATION_THRESHOLD = 6
ADHD_PRESENT_INATTENTIVE_COLUMNS = ADHD_CURRENT_INATTENTIVE_COLUMNS
ADHD_PRESENT_HYPERACTIVE_IMPULSIVE_COLUMNS = (
    ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS
)
ADHD_PAST_INATTENTIVE_COLUMNS = (
    "mh_p_ksads__adhd__mistake__past_sx",
    "mh_p_ksads__adhd__sustattn__1schlyr__past_sx",
    "mh_p_ksads__adhd__notlisten__past_sx",
    "mh_p_ksads__adhd__flwinstr__past_sx",
    "mh_p_ksads__adhd__orgtask__past_sx",
    "mh_p_ksads__adhd__avoid__task__past_sx",
    "mh_p_ksads__adhd__loses__past_sx",
    "mh_p_ksads__adhd__distract__1schlyr__past_sx",
    "mh_p_ksads__adhd__forget__past_sx",
)
ADHD_PAST_HYPERACTIVE_IMPULSIVE_COLUMNS = (
    "mh_p_ksads__adhd__fidget__past_sx",
    "mh_p_ksads__adhd__seat__1schlyr__past_sx",
    "mh_p_ksads__adhd__hypractv__past_sx",
    "mh_p_ksads__adhd__quiet__past_sx",
    "mh_p_ksads__adhd__motor__past_sx",
    "mh_p_ksads__adhd__talkexcess__past_sx",
    "mh_p_ksads__adhd__blurt__past_sx",
    "mh_p_ksads__adhd__wait__past_sx",
    "mh_p_ksads__adhd__interrupt__past_sx",
)

# Fixed ADNI experimental-cohort counts (CN/MCI/AD after the repository's
# 1/2/3-to-0/1/2 mapping).  Only the class-0/class-1 ratio is used to
# deterministically subsample the large ABCD low-status pool.
ADNI_REFERENCE_CLASS_COUNTS = (926, 746, 444)
# These tables contain either the endpoint itself or variables that can reveal
# an ADHD diagnosis/treatment directly.  Most are not in the current predictor
# allowlist; keeping the explicit blacklist in the generated manifest prevents
# an apparently harmless future source expansion from introducing leakage.
ADHD_SOURCE_TABLE_BLACKLIST = {
    ADHD_TARGET_TABLE,
    "physical_health/ph_p_meds.parquet",
    "physical_health/ph_y_meds.parquet",
    "general/ab_p_screen.parquet",
    "friends_family_community/fc_p_sag.parquet",
    "mental_health/mh_y_ysr.parquet",
    "mental_health/mh_p_abcl.parquet",
    "mental_health/mh_p_asr.parquet",
    "mental_health/mh_y_bpm.parquet",
}

NONRESPONSE_WORDS = (
    "not administered", "not asked", "decline", "don't know", "do not know",
    "missing", "refused", "skip",
)

# Remove CBCL items that directly describe eating/weight symptoms.  Every CBCL
# composite is also removed because totals and syndrome scores algebraically
# reintroduce these item values after the raw columns are excluded.
# Anthropometrics remain legitimate baseline predictors rather than diagnostic
# questionnaire proxies; a no-anthropometry sensitivity analysis is reported
# separately.
BED_DIRECT_FEATURE_EXCLUSIONS = {
    "mh_p_cbcl__othpr_004",       # Overeating
    "mh_p_cbcl__othpr_005",       # Overweight
    "mh_p_cbcl__othpr__dep_001", # Doesn't eat well
    "mh_p_cbcl__som__somat_007", # Vomiting / throwing up
}
BED_CBCL_COMPOSITE_SUFFIXES = ("_sum", "_tscore", "_nm")
ADHD_DIRECT_CBCL_COLUMNS = {
    "mh_p_cbcl__aggr__adhd_001",  # Unusually loud
    "mh_p_cbcl__attn_001",        # Acts too young
    "mh_p_cbcl__attn_002",        # Confused / in a fog
    "mh_p_cbcl__attn_003",        # Daydreams
    "mh_p_cbcl__attn_004",        # Poor school work
    "mh_p_cbcl__attn_005",        # Stares blankly
    "mh_p_cbcl__attn__adhd_001",  # Fails to finish things
    "mh_p_cbcl__attn__adhd_002",  # Cannot concentrate
    "mh_p_cbcl__attn__adhd_003",  # Restless / hyperactive
    "mh_p_cbcl__attn__adhd_004",  # Impulsive
    "mh_p_cbcl__attn__adhd_005",  # Inattentive / distracted
    "mh_p_cbcl__othpr__adhd_001", # Talks too much
}
BED_TABLES_WITHOUT_BASELINE_ROWS = {
    "physical_health/ph_y_bld.parquet",
    "mental_health/mh_y_bpm.parquet",
    "social_development/sdev_p_nbh.parquet",
    "social_development/sdev_p_apq.parquet",
}


def bed_feature_exclusions(root: Path) -> set[str]:
    """Resolve every direct or algebraically contaminated CBCL predictor."""

    cbcl_path = root / "mental_health/mh_p_cbcl.parquet"
    cbcl_columns = pd.read_parquet(cbcl_path).columns
    composites = {
        column
        for column in cbcl_columns
        if column.startswith("mh_p_cbcl")
        and column.endswith(BED_CBCL_COMPOSITE_SUFFIXES)
    }
    return BED_DIRECT_FEATURE_EXCLUSIONS | composites


def adhd_feature_exclusions(root: Path) -> set[str]:
    """Remove direct CBCL ADHD/attention proxies and contaminated scores.

    Individual non-ADHD CBCL items remain available, but every CBCL aggregate
    is removed because syndrome, DSM-oriented, broadband, and total scores can
    algebraically reintroduce excluded ADHD items.
    """

    cbcl_path = root / "mental_health/mh_p_cbcl.parquet"
    cbcl_columns = set(pd.read_parquet(cbcl_path).columns)
    missing_direct = sorted(ADHD_DIRECT_CBCL_COLUMNS - cbcl_columns)
    if missing_direct:
        raise ValueError(f"Direct ADHD CBCL columns missing: {missing_direct}")
    composites = {
        column
        for column in cbcl_columns
        if column.startswith("mh_p_cbcl")
        and column.endswith(BED_CBCL_COMPOSITE_SUFFIXES)
    }
    return ADHD_DIRECT_CBCL_COLUMNS | composites


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for provenance records."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adni_reference_details(reference_dir: Path) -> dict[str, object]:
    """Audit the fixed ADNI cohort whose class prior is used for matching."""

    label_path = (reference_dir / "label.csv").resolve()
    split_path = (reference_dir / "PTID_splits.json").resolve()
    if not label_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            f"ADNI reference files are missing under {reference_dir}"
        )
    labels = pd.read_csv(label_path)
    splits = json.loads(split_path.read_text())
    cohort_ids = {
        str(participant_id)
        for split_ids_value in splits.values()
        for participant_id in split_ids_value
    }
    cohort = labels[labels["PTID"].astype(str).isin(cohort_ids)].copy()
    diagnosis = pd.to_numeric(cohort["DIAGNOSIS"], errors="coerce")
    if len(cohort) != len(cohort_ids) or diagnosis.isna().any():
        raise ValueError("ADNI label/split reference is incomplete")
    counts = tuple(
        int(diagnosis.eq(class_value).sum()) for class_value in (1, 2, 3)
    )
    if counts != ADNI_REFERENCE_CLASS_COUNTS:
        raise ValueError(
            f"ADNI reference counts changed: {counts} != "
            f"{ADNI_REFERENCE_CLASS_COUNTS}"
        )
    return {
        "role_mapping": {
            "0": "CN",
            "1": "MCI",
            "2": "AD",
        },
        "class_counts": list(counts),
        "class_proportions": [count / sum(counts) for count in counts],
        "label_path": str(label_path),
        "label_sha256": sha256_file(label_path),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
    }


def build_adhd_labels(
    root: Path,
    session: str = "ses-00A",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build strict parent K-SADS full-ADHD present-or-past labels.

    Cases require one of the two full ADHD diagnosis fields.  Participants with
    partial-remission or unspecified ADHD are excluded from controls rather
    than mislabeled as ADHD-negative.  Full-remission and other-specified fields
    are also checked when populated (they are entirely unavailable at baseline
    in the audited local release).  A branch skip (888) is criterion-negative;
    555, NaN, or any other code in either full-diagnosis field is ineligible.
    """

    path = root / ADHD_TARGET_TABLE
    frame = pd.read_parquet(path)
    required_columns = set(ADHD_TARGET_COLUMNS) | set(ADHD_CONTROL_EXCLUSION_COLUMNS)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"ADHD diagnosis columns missing from K-SADS table: {missing_columns}")
    at_session = frame[frame["session_id"].astype(str).eq(session)].copy()
    at_session["participant_id"] = at_session["participant_id"].astype(str)
    if at_session["participant_id"].duplicated().any():
        duplicates = int(at_session["participant_id"].duplicated(keep=False).sum())
        raise ValueError(f"ADHD target has {duplicates} duplicate participant rows at {session}")

    full_values = at_session[list(ADHD_TARGET_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    full_status_valid = full_values.isin(ADHD_VALID_DIAGNOSIS_VALUES).all(axis=1)
    positive = full_values.eq(1).any(axis=1)

    variant_values = at_session[list(ADHD_CONTROL_EXCLUSION_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    available_variant_columns = [
        column
        for column in ADHD_CONTROL_EXCLUSION_COLUMNS
        if variant_values[column].isin(ADHD_VALID_DIAGNOSIS_VALUES).any()
    ]
    unavailable_variant_columns = [
        column
        for column in ADHD_CONTROL_EXCLUSION_COLUMNS
        if column not in available_variant_columns
    ]
    available_variants = variant_values[available_variant_columns]
    variant_status_valid = available_variants.isin(
        ADHD_VALID_DIAGNOSIS_VALUES
    ).all(axis=1)
    variant_positive = available_variants.eq(1).any(axis=1)

    strict_cases = full_status_valid & positive
    clean_controls = (
        full_status_valid
        & ~positive
        & variant_status_valid
        & ~variant_positive
    )
    eligible = strict_cases | clean_controls
    labels = pd.DataFrame(
        {
            "participant_id": at_session.loc[eligible, "participant_id"],
            "target": strict_cases.loc[eligible].astype(int).to_numpy(),
        }
    )
    excluded_variant_counts = {
        column: int((full_status_valid & ~positive & variant_values[column].eq(1)).sum())
        for column in available_variant_columns
    }
    details: dict[str, object] = {
        "endpoint_type": (
            "baseline strict full ADHD case versus assessed available-field-negative "
            "control classification"
        ),
        "clinical_diagnosis": False,
        "informant": "parent",
        "session": session,
        "episode_status": "present or past",
        "positive_columns": list(ADHD_TARGET_COLUMNS),
        "valid_diagnosis_values": list(ADHD_VALID_DIAGNOSIS_VALUES),
        "control_exclusion_columns": list(ADHD_CONTROL_EXCLUSION_COLUMNS),
        "available_control_exclusion_columns": available_variant_columns,
        "unavailable_control_exclusion_columns": unavailable_variant_columns,
        "branch_skipped_888": "valid criterion-negative branch skip",
        "not_administered_555": (
            "ineligible; both full-diagnosis fields must be one of 0, 1, or 888"
        ),
        "missing_or_unknown_full_status": (
            "ineligible; both full-diagnosis fields must be one of 0, 1, or 888"
        ),
        "raw_session_participants": int(len(at_session)),
        "valid_full_status_participants": int(full_status_valid.sum()),
        "invalid_full_status_excluded": int((~full_status_valid).sum()),
        "strict_full_adhd_positive": int(strict_cases.sum()),
        "clean_adhd_negative": int(clean_controls.sum()),
        "control_contamination_excluded": int(
            (full_status_valid & ~positive & variant_positive).sum()
        ),
        "invalid_control_variant_status_excluded": int(
            (
                full_status_valid
                & ~positive
                & ~variant_positive
                & ~variant_status_valid
            ).sum()
        ),
        "strict_cases_with_control_variant_flag": int(
            (strict_cases & variant_positive).sum()
        ),
        "control_exclusion_counts_by_column": excluded_variant_counts,
        "branch_skipped_888_both_full_fields": int(
            full_values.eq(888).all(axis=1).sum()
        ),
        "full_status_555_counts_by_column": {
            column: int(full_values[column].eq(555).sum())
            for column in ADHD_TARGET_COLUMNS
        },
        "full_status_missing_counts_by_column": {
            column: int(full_values[column].isna().sum())
            for column in ADHD_TARGET_COLUMNS
        },
        "eligible_participants": int(len(labels)),
    }
    return labels, details


def build_adhd_presentation_labels(
    root: Path,
    session: str = "ses-00A",
    strict_symptoms: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Derive the three DSM-style ADHD presentations from parent K-SADS.

    A participant must first have a released full-ADHD diagnosis (present or
    past).  Current symptoms define the presentation whenever a present
    diagnosis is available; otherwise the past-symptom fields define it.  The
    primary strict endpoint accepts only explicit 0/1 values in all 18
    symptoms of the selected episode.  ``strict_symptoms=False`` retains a
    predeclared sensitivity variant in which branch skip (888) is interpreted
    as criterion-negative.  Not-administered, missing, or unknown values in the
    selected episode are always excluded.  Cases that do not meet either
    six-symptom domain threshold are left unresolved and excluded instead of
    being forced into a class.
    """

    path = root / ADHD_TARGET_TABLE
    frame = pd.read_parquet(path)
    symptom_columns = (
        *ADHD_PRESENT_INATTENTIVE_COLUMNS,
        *ADHD_PRESENT_HYPERACTIVE_IMPULSIVE_COLUMNS,
        *ADHD_PAST_INATTENTIVE_COLUMNS,
        *ADHD_PAST_HYPERACTIVE_IMPULSIVE_COLUMNS,
    )
    required_columns = set(ADHD_TARGET_COLUMNS) | set(symptom_columns)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "ADHD presentation columns missing from K-SADS table: "
            f"{missing_columns}"
        )

    at_session = frame[frame["session_id"].astype(str).eq(session)].copy()
    at_session["participant_id"] = at_session["participant_id"].astype(str)
    if at_session["participant_id"].duplicated().any():
        duplicates = int(at_session["participant_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"ADHD presentation target has {duplicates} duplicate participant rows "
            f"at {session}"
        )

    diagnosis = at_session[list(ADHD_TARGET_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    diagnosis_valid = diagnosis.isin(ADHD_VALID_DIAGNOSIS_VALUES).all(axis=1)
    present_diagnosis = diagnosis[ADHD_TARGET_COLUMNS[0]].eq(1)
    past_diagnosis = diagnosis[ADHD_TARGET_COLUMNS[1]].eq(1)
    full_case = diagnosis_valid & (present_diagnosis | past_diagnosis)

    def numeric_symptoms(columns: tuple[str, ...]) -> pd.DataFrame:
        return at_session[list(columns)].apply(pd.to_numeric, errors="coerce")

    present_inattentive = numeric_symptoms(ADHD_PRESENT_INATTENTIVE_COLUMNS)
    present_hyperactive = numeric_symptoms(
        ADHD_PRESENT_HYPERACTIVE_IMPULSIVE_COLUMNS
    )
    past_inattentive = numeric_symptoms(ADHD_PAST_INATTENTIVE_COLUMNS)
    past_hyperactive = numeric_symptoms(ADHD_PAST_HYPERACTIVE_IMPULSIVE_COLUMNS)

    diagnosis_valid_values = set(ADHD_VALID_DIAGNOSIS_VALUES)
    symptom_valid_values = (
        {0, 1} if strict_symptoms else diagnosis_valid_values
    )
    present_symptoms = pd.concat(
        [present_inattentive, present_hyperactive], axis=1
    )
    past_symptoms = pd.concat(
        [past_inattentive, past_hyperactive], axis=1
    )
    selected_symptoms = pd.DataFrame(
        np.where(
            present_diagnosis.to_numpy()[:, None],
            present_symptoms.to_numpy(),
            past_symptoms.to_numpy(),
        ),
        index=at_session.index,
    )
    selected_symptoms_valid = selected_symptoms.isin(
        symptom_valid_values
    ).all(axis=1)

    present_inattentive_count = present_inattentive.eq(1).sum(axis=1)
    present_hyperactive_count = present_hyperactive.eq(1).sum(axis=1)
    past_inattentive_count = past_inattentive.eq(1).sum(axis=1)
    past_hyperactive_count = past_hyperactive.eq(1).sum(axis=1)
    inattentive_count = present_inattentive_count.where(
        present_diagnosis, past_inattentive_count
    )
    hyperactive_count = present_hyperactive_count.where(
        present_diagnosis, past_hyperactive_count
    )

    threshold = ADHD_PRESENTATION_THRESHOLD
    inattentive_only = (inattentive_count >= threshold) & (
        hyperactive_count < threshold
    )
    hyperactive_only = (inattentive_count < threshold) & (
        hyperactive_count >= threshold
    )
    combined = (inattentive_count >= threshold) & (
        hyperactive_count >= threshold
    )
    resolved = inattentive_only | hyperactive_only | combined
    eligible = full_case & selected_symptoms_valid & resolved
    target = np.select(
        [inattentive_only, hyperactive_only, combined],
        [0, 1, 2],
        default=-1,
    ).astype(np.int64)
    episode = np.where(present_diagnosis, "present", "past")

    labels = pd.DataFrame(
        {
            "participant_id": at_session.loc[eligible, "participant_id"],
            "target": target[eligible.to_numpy()],
            "presentation_episode": episode[eligible.to_numpy()],
            "inattentive_symptom_count": inattentive_count.loc[eligible].to_numpy(
                dtype=np.int8
            ),
            "hyperactive_impulsive_symptom_count": hyperactive_count.loc[
                eligible
            ].to_numpy(dtype=np.int8),
        }
    ).reset_index(drop=True)

    class_counts = labels["target"].value_counts().sort_index()
    unresolved = full_case & selected_symptoms_valid & ~resolved
    invalid_selected_symptoms = full_case & ~selected_symptoms_valid
    selected_any_888 = selected_symptoms.eq(888).any(axis=1)
    selected_any_555 = selected_symptoms.eq(555).any(axis=1)
    selected_any_missing = selected_symptoms.isna().any(axis=1)
    selected_any_unknown = ~(
        selected_symptoms.isin(diagnosis_valid_values | {555})
        | selected_symptoms.isna()
    ).all(axis=1)
    details: dict[str, object] = {
        "endpoint_type": "baseline parent K-SADS full-ADHD three-presentation classification",
        "clinical_diagnosis": False,
        "informant": "parent",
        "session": session,
        "full_diagnosis_columns": list(ADHD_TARGET_COLUMNS),
        "episode_rule": (
            "use present symptoms for a present full-ADHD diagnosis; otherwise use "
            "past symptoms for a past-only full-ADHD diagnosis"
        ),
        "presentation_variant": (
            "strict_selected_episode_01_primary"
            if strict_symptoms
            else "selected_episode_888_negative_sensitivity"
        ),
        "strict_selected_episode_symptoms": bool(strict_symptoms),
        "scoring_assumption": (
            "all 18 selected-episode symptoms must be explicit 0/1; any 888 is "
            "excluded"
            if strict_symptoms
            else "selected-episode 888 is retained and scored criterion-negative"
        ),
        "presentation_threshold": threshold,
        "threshold_population": "children younger than 17 years",
        "class_mapping": {
            "0": "predominantly inattentive presentation",
            "1": "predominantly hyperactive/impulsive presentation",
            "2": "combined presentation",
        },
        "present_inattentive_columns": list(ADHD_PRESENT_INATTENTIVE_COLUMNS),
        "present_hyperactive_impulsive_columns": list(
            ADHD_PRESENT_HYPERACTIVE_IMPULSIVE_COLUMNS
        ),
        "past_inattentive_columns": list(ADHD_PAST_INATTENTIVE_COLUMNS),
        "past_hyperactive_impulsive_columns": list(
            ADHD_PAST_HYPERACTIVE_IMPULSIVE_COLUMNS
        ),
        "valid_full_diagnosis_values": list(ADHD_VALID_DIAGNOSIS_VALUES),
        "valid_symptom_values": sorted(symptom_valid_values),
        "branch_skipped_888": (
            "excluded in the strict primary endpoint"
            if strict_symptoms
            else "valid criterion-negative value in the sensitivity endpoint"
        ),
        "not_administered_555": (
            "excluded only when present in the selected presentation episode; "
            "the unselected episode is not consulted"
        ),
        "missing_or_unknown_selected_symptom": (
            "excluded when present in the selected presentation episode"
        ),
        "raw_session_participants": int(len(at_session)),
        "invalid_full_diagnosis_status_excluded": int((~diagnosis_valid).sum()),
        "full_diagnosis_555_counts_by_column": {
            column: int(diagnosis[column].eq(555).sum())
            for column in ADHD_TARGET_COLUMNS
        },
        "full_diagnosis_missing_counts_by_column": {
            column: int(diagnosis[column].isna().sum())
            for column in ADHD_TARGET_COLUMNS
        },
        "valid_full_adhd_cases": int(full_case.sum()),
        "present_full_adhd_cases": int((full_case & present_diagnosis).sum()),
        "past_only_full_adhd_cases": int(
            (full_case & ~present_diagnosis & past_diagnosis).sum()
        ),
        "selected_episode_any_888_cases": int(
            (full_case & selected_any_888).sum()
        ),
        "selected_episode_888_cases_excluded": int(
            (full_case & selected_any_888 & ~selected_symptoms_valid).sum()
        ),
        "selected_episode_any_555_cases": int(
            (full_case & selected_any_555).sum()
        ),
        "selected_episode_any_missing_cases": int(
            (full_case & selected_any_missing).sum()
        ),
        "selected_episode_any_unknown_cases": int(
            (full_case & selected_any_unknown).sum()
        ),
        "eligible_present_episode": int((eligible & present_diagnosis).sum()),
        "eligible_past_episode": int((eligible & ~present_diagnosis).sum()),
        "invalid_selected_symptom_cases_excluded": int(
            invalid_selected_symptoms.sum()
        ),
        "unresolved_below_threshold_cases_excluded": int(unresolved.sum()),
        "eligible_participants": int(len(labels)),
        "class_counts": {
            str(class_id): int(class_counts.get(class_id, 0))
            for class_id in (0, 1, 2)
        },
    }
    return labels, details


def build_adhd_status3_labels(
    root: Path,
    session: str = "ses-00A",
    cohort_seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build an ADNI-aligned three-status ADHD research endpoint.

    These labels play the same statistical roles as CN/MCI/AD without
    borrowing adult-neurodegeneration names: low-symptom comparator,
    intermediate ADHD spectrum/history/remission, and current full ADHD.
    The oversized low-status pool is sampled to the fixed ADNI class-0/class-1
    ratio so ordinary accuracy has a comparable chance level.
    """

    path = root / ADHD_TARGET_TABLE
    frame = pd.read_parquet(path)
    status_columns = (
        *ADHD_TARGET_COLUMNS,
        ADHD_CONTROL_EXCLUSION_COLUMNS[0],
        ADHD_CONTROL_EXCLUSION_COLUMNS[1],
    )
    symptom_columns = (
        *ADHD_CURRENT_INATTENTIVE_COLUMNS,
        *ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS,
    )
    required_columns = set(status_columns) | set(symptom_columns)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"ADHD status3 columns missing from K-SADS table: {missing_columns}"
        )

    at_session = frame[frame["session_id"].astype(str).eq(session)].copy()
    at_session["participant_id"] = at_session["participant_id"].astype(str)
    if at_session["participant_id"].duplicated().any():
        duplicates = int(at_session["participant_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"ADHD status3 target has {duplicates} duplicate participant rows at "
            f"{session}"
        )

    status = at_session[list(status_columns)].apply(pd.to_numeric, errors="coerce")
    symptoms = at_session[list(symptom_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_values = set(ADHD_VALID_DIAGNOSIS_VALUES)
    valid = status.isin(valid_values).all(axis=1) & symptoms.isin(
        valid_values
    ).all(axis=1)

    present_full = status[ADHD_TARGET_COLUMNS[0]].eq(1)
    past_full = status[ADHD_TARGET_COLUMNS[1]].eq(1)
    partial_remission = status[ADHD_CONTROL_EXCLUSION_COLUMNS[0]].eq(1)
    unspecified = status[ADHD_CONTROL_EXCLUSION_COLUMNS[1]].eq(1)
    history_or_variant = past_full | partial_remission | unspecified

    inattentive_count = symptoms[
        list(ADHD_CURRENT_INATTENTIVE_COLUMNS)
    ].eq(1).sum(axis=1)
    hyperactive_count = symptoms[
        list(ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS)
    ].eq(1).sum(axis=1)
    max_domain_count = pd.concat(
        [inattentive_count, hyperactive_count], axis=1
    ).max(axis=1)
    elevated_current_symptoms = (
        max_domain_count >= ADHD_INTERMEDIATE_SYMPTOM_THRESHOLD
    )

    current_full = valid & present_full
    intermediate = (
        valid
        & ~present_full
        & (history_or_variant | elevated_current_symptoms)
    )
    low_pool = (
        valid
        & ~present_full
        & ~history_or_variant
        & ~elevated_current_symptoms
    )
    reference_low, reference_intermediate, _ = ADNI_REFERENCE_CLASS_COUNTS
    target_low_count = round(
        int(intermediate.sum()) * reference_low / reference_intermediate
    )
    if int(low_pool.sum()) < target_low_count:
        raise ValueError(
            "ADHD status3 low-status pool is too small for ADNI-ratio matching"
        )

    low_ids = at_session.loc[low_pool, "participant_id"].tolist()
    low_ids = sorted(
        low_ids,
        key=lambda participant_id: (
            hashlib.sha256(
                (
                    f"{cohort_seed}|adhd_status3_low_match|{participant_id}"
                ).encode("utf-8")
            ).hexdigest(),
            participant_id,
        ),
    )[:target_low_count]
    selected_low = at_session["participant_id"].isin(low_ids)
    included = selected_low | intermediate | current_full
    target = np.select(
        [selected_low, intermediate, current_full],
        [0, 1, 2],
        default=-1,
    ).astype(np.int64)

    labels = pd.DataFrame(
        {
            "participant_id": at_session.loc[included, "participant_id"],
            "target": target[included.to_numpy()],
            "current_inattentive_symptom_count": inattentive_count.loc[
                included
            ].to_numpy(dtype=np.int8),
            "current_hyperactive_impulsive_symptom_count": hyperactive_count.loc[
                included
            ].to_numpy(dtype=np.int8),
        }
    ).sort_values("participant_id", kind="stable").reset_index(drop=True)

    class_counts = labels["target"].value_counts().sort_index()
    details: dict[str, object] = {
        "endpoint_type": "baseline parent K-SADS aligned three-status ADHD classification",
        "clinical_diagnosis": False,
        "informant": "parent",
        "session": session,
        "alignment_scope": (
            "statistical status roles analogous to CN/MCI/AD; no ABCD class is "
            "named or interpreted as an ADNI diagnosis"
        ),
        "class_mapping": {
            "0": "low-symptom/no-positive-status comparator",
            "1": "intermediate ADHD spectrum/history/remission",
            "2": "current full ADHD",
        },
        "intermediate_rule": (
            "no current full ADHD and at least one of: past full ADHD, partial "
            "remission, unspecified ADHD, or >=4 current symptoms in either "
            "nine-item domain"
        ),
        "low_status_rule": (
            "no present/past/partial-remission/unspecified ADHD and <=3 current "
            "symptoms in each nine-item domain"
        ),
        "intermediate_symptom_threshold": ADHD_INTERMEDIATE_SYMPTOM_THRESHOLD,
        "current_inattentive_columns": list(ADHD_CURRENT_INATTENTIVE_COLUMNS),
        "current_hyperactive_impulsive_columns": list(
            ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS
        ),
        "status_columns": list(status_columns),
        "valid_values": list(ADHD_VALID_DIAGNOSIS_VALUES),
        "branch_skipped_888": (
            "structural branch skip interpreted as criterion-negative; an "
            "exclude-all-status-888 sensitivity analysis is required"
        ),
        "not_administered_555": "excluded",
        "raw_session_participants": int(len(at_session)),
        "valid_status_and_symptom_participants": int(valid.sum()),
        "source_low_status_pool": int(low_pool.sum()),
        "sampled_low_status": int(selected_low.sum()),
        "intermediate_total": int(intermediate.sum()),
        "intermediate_with_history_or_variant": int(
            (intermediate & history_or_variant).sum()
        ),
        "intermediate_symptom_only": int(
            (intermediate & ~history_or_variant).sum()
        ),
        "current_full_adhd": int(current_full.sum()),
        "all_status_fields_888_source": int(status.eq(888).all(axis=1).sum()),
        "all_status_fields_888_selected": int(
            (status.eq(888).all(axis=1) & included).sum()
        ),
        "sampling": {
            "cohort_seed": int(cohort_seed),
            "sampling_unit": "participant_id",
            "sampling_stage": "before_family_disjoint_split",
            "hash_algorithm": "sha256",
            "hash_input_template": (
                "{cohort_seed}|adhd_status3_low_match|{participant_id}"
            ),
            "hash_tie_break": "participant_id lexical order",
            "encoding": "utf-8",
            "target": "match the fixed ADNI class-0/class-1 ratio",
            "target_formula": "round(n_class1 * 926 / 746)",
            "target_low_count": int(target_low_count),
            "adni_reference_class_counts": list(ADNI_REFERENCE_CLASS_COUNTS),
            "pre_sampling_class_counts": [
                int(low_pool.sum()), int(intermediate.sum()), int(current_full.sum())
            ],
            "post_sampling_class_counts": [
                int(selected_low.sum()), int(intermediate.sum()), int(current_full.sum())
            ],
            "source_low_status_discarded": int(low_pool.sum()) - target_low_count,
            "generalizability": (
                "ratio-matched case-control research cohort; not natural ABCD "
                "prevalence"
            ),
        },
        "eligible_participants": int(len(labels)),
        "class_counts": {
            str(class_id): int(class_counts.get(class_id, 0))
            for class_id in (0, 1, 2)
        },
    }
    return labels, details


def nonresponse_values(metadata: dict, column: str) -> set[str]:
    levels = metadata.get(column, {}).get("Levels", {})
    return {
        str(value) for value, description in levels.items()
        if any(word in str(description).lower() for word in NONRESPONSE_WORDS)
    }


def table_at_session(
    path: Path,
    session: str,
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "session_id" in frame:
        frame = frame[frame["session_id"].astype(str) == session]
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    columns = {"participant_id": frame["participant_id"].astype(str)}
    excluded_columns = excluded_columns or set()
    for column in frame.columns:
        if column in {"participant_id", "session_id"} or column.endswith("_dtt"):
            continue
        if column in excluded_columns:
            continue
        values = frame[column]
        invalid = nonresponse_values(metadata, column)
        as_string = values.astype(str)
        if invalid:
            values = values.mask(as_string.isin(invalid))
        numeric = pd.to_numeric(values.astype(str), errors="coerce")
        if numeric.notna().any():
            columns[column] = numeric.astype(np.float32)
    output = pd.DataFrame(columns)
    feature_cols = [column for column in output if column != "participant_id"]
    output = output.dropna(subset=feature_cols, how="all")
    return output.drop_duplicates("participant_id", keep="first")


def merge_modality(
    root: Path,
    tables: list[str],
    session: str,
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    merged = None
    for relative in tables:
        table = table_at_session(root / relative, session, excluded_columns)
        merged = table if merged is None else merged.merge(table, on="participant_id", how="outer")
    if merged is None:
        raise RuntimeError("A modality must contain at least one source table")
    return merged


def build_diagnostic_burden_labels(
    root: Path,
    session: str,
    high_burden_start: int = 2,
    low_burden_end: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Build ordinal lifetime K-SADS diagnosis-domain burden labels.

    ``high_burden_start=2`` preserves the original 0 / 1 / 2+ definition.
    Larger values create a broader, more stable middle burden stage.
    """
    if low_burden_end < 0 or high_burden_start <= low_burden_end + 1:
        raise ValueError("Burden bins require 0 <= low_end < high_start - 1")
    merged = None
    source_tables = []
    for domain in DIAGNOSIS_DOMAINS:
        relative = f"mental_health/mh_p_ksads__{domain}.parquet"
        path = root / relative
        frame = pd.read_parquet(path)
        frame = frame[frame["session_id"].astype(str) == session].drop_duplicates("participant_id")
        metadata = json.loads(path.with_suffix(".json").read_text())
        diagnosis_columns = [
            column for column, spec in metadata.items()
            if column in frame
            and "Diagnosis:" in spec.get("Description", "")
            and (
                " - Present " in spec.get("Description", "")
                or " - Past " in spec.get("Description", "")
            )
        ]
        if not diagnosis_columns:
            raise ValueError(f"No present/past diagnosis columns found in {relative}")
        values = frame[diagnosis_columns].astype(str)
        administered = (values != "555").any(axis=1)
        indicator = (values == "1").any(axis=1).astype(np.int8)
        domain_frame = pd.DataFrame({
            "participant_id": frame.loc[administered, "participant_id"].astype(str),
            domain: indicator.loc[administered].to_numpy(),
        })
        merged = domain_frame if merged is None else merged.merge(domain_frame, on="participant_id", how="inner")
        source_tables.append(relative)

    diagnosis_count = merged[list(DIAGNOSIS_DOMAINS)].sum(axis=1).astype(np.int16)
    target = np.where(
        diagnosis_count <= low_burden_end,
        0,
        np.where(diagnosis_count < high_burden_start, 1, 2),
    ).astype(np.int64)
    labels = pd.DataFrame({
        "participant_id": merged["participant_id"].astype(str),
        "target": target,
        "diagnosis_count": diagnosis_count,
    })
    return labels, source_tables


def build_cognitive_status_labels(root: Path, session: str) -> pd.DataFrame:
    """Bin the ABCD NIH Toolbox age-corrected total cognition standard score."""
    frame = pd.read_parquet(root / COGNITION_TARGET_TABLE)
    frame = frame[frame["session_id"].astype(str) == session].drop_duplicates("participant_id")
    score = pd.to_numeric(frame[COGNITION_TARGET_COLUMN], errors="coerce")
    keep = score.notna()
    score = score.loc[keep]
    target = np.where(score < 85.0, 0, np.where(score <= 115.0, 1, 2)).astype(np.int64)
    return pd.DataFrame({
        "participant_id": frame.loc[keep, "participant_id"].astype(str),
        "target": target,
        "cognition_standard_score": score.to_numpy(dtype=np.float32),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abcd-root", required=True)
    parser.add_argument("--output-dir", default="./data/abcd_ed6")
    parser.add_argument("--session", default="ses-00A")
    parser.add_argument("--target", choices=["present_any", "lifetime_any"], default="lifetime_any")
    parser.add_argument(
        "--task",
        choices=[
            "ed", "bed_2y", "adhd", "adhd_presentation", "adhd_status3",
            "diagnostic_burden", "cognitive_status",
        ],
        default="ed",
        help=(
            "ed: legacy any-eating-disorder target; bed_2y: symptom-derived probable full BED "
            "through year 2; adhd: baseline parent K-SADS full ADHD present/past; "
            "adhd_presentation: three full-ADHD symptom presentations; "
            "adhd_status3: low/intermediate/current-full ADHD status aligned to ADNI roles; "
            "diagnostic_burden: 0/1/2+ lifetime diagnosis domains; "
            "cognitive_status: NIH Toolbox total cognition <85 / 85-115 / >115"
        ),
    )
    parser.add_argument(
        "--modality-preset",
        choices=["expanded", "adni4"],
        default="expanded",
        help="expanded: separate ABCD tables into up to eight modalities; adni4: aggregate as I/G/C/B",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--cohort-seed",
        type=int,
        default=2026,
        help=(
            "Independent deterministic seed used only for adhd_status3 "
            "case-control cohort sampling."
        ),
    )
    parser.add_argument(
        "--adhd-presentation-strict-symptoms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For adhd_presentation, require explicit 0/1 values in all 18 "
            "symptoms of the selected episode. The default strict endpoint is "
            "the primary analysis; --no-adhd-presentation-strict-symptoms "
            "retains 888 as criterion-negative for sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--adni-reference-dir",
        default=str(Path(__file__).resolve().parent / "MoE/data/adni"),
        help=(
            "Directory containing the fixed ADNI label.csv and "
            "PTID_splits.json used to audit the status3 prior match."
        ),
    )
    parser.add_argument(
        "--bed-outcome-sessions",
        default="ses-00A,ses-01A,ses-02A",
        help="Comma-separated K-SADS visits contributing to the BED endpoint.",
    )
    parser.add_argument(
        "--bed-required-session",
        default="ses-02A",
        help="Follow-up visit required for inclusion in the BED endpoint.",
    )
    parser.add_argument(
        "--bed-require-all-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require an administered parent ED module at every BED outcome visit.",
    )
    parser.add_argument(
        "--burden-high-start",
        type=int,
        default=2,
        help="For diagnostic_burden, bins are 0 / 1..N-1 / N+ (default N=2 preserves 0/1/2+).",
    )
    parser.add_argument(
        "--burden-low-end",
        type=int,
        default=0,
        help="For diagnostic_burden, class 0 contains counts <= this value.",
    )
    args = parser.parse_args()
    root = Path(args.abcd_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    adhd_task = args.task in {"adhd", "adhd_presentation", "adhd_status3"}

    excluded_modality_codes = set()
    bed_details: dict[str, object] = {}
    adhd_details: dict[str, object] = {}
    if args.task == "cognitive_status":
        labels = build_cognitive_status_labels(root, args.session)
        target_tables = [COGNITION_TARGET_TABLE]
        class_values = [0, 1, 2]
        target_definition = "NIH Toolbox age-corrected total cognition: <85 / 85-115 / >115"
        class_meanings = {"0": "low cognition", "1": "average cognition", "2": "high cognition"}
        leakage_control = "NIH Toolbox target table and the entire neurocognition modality excluded from features"
        # NIH Toolbox and its component tasks mathematically determine the target.
        # Exclude the entire neurocognition modality for a conservative leakage-free task.
        excluded_modality_codes.add("N")
    elif args.task == "diagnostic_burden":
        labels, target_tables = build_diagnostic_burden_labels(
            root, args.session, args.burden_high_start, args.burden_low_end
        )
        class_values = [0, 1, 2]
        target_definition = (
            "lifetime K-SADS diagnosis domains: "
            f"0 to {args.burden_low_end} / "
            f"{args.burden_low_end + 1} to {args.burden_high_start - 1} / "
            f"{args.burden_high_start} or more"
        )
        class_meanings = {
            "0": f"zero to {args.burden_low_end} diagnosis domains (low burden)",
            "1": (
                f"{args.burden_low_end + 1} to {args.burden_high_start - 1} "
                "diagnosis domains (moderate burden)"
            ),
            "2": f"{args.burden_high_start} or more diagnosis domains",
        }
        leakage_control = (
            "all direct K-SADS target tables excluded from features; non-K-SADS mental-health "
            "instruments remain predictors"
        )
    elif args.task == "adhd_status3":
        labels, adhd_details = build_adhd_status3_labels(
            root, args.session, args.cohort_seed
        )
        adhd_details["adni_reference"] = adni_reference_details(
            Path(args.adni_reference_dir).resolve()
        )
        target_tables = [ADHD_TARGET_TABLE]
        class_values = [0, 1, 2]
        target_definition = (
            "baseline parent K-SADS ADHD status: low-symptom comparator / "
            "intermediate spectrum-history-remission / current full ADHD; "
            "class-0 sampling matches the fixed ADNI class-0/class-1 ratio"
        )
        class_meanings = adhd_details["class_mapping"]
        leakage_control = (
            "parent K-SADS ADHD table, direct CBCL ADHD/attention items, every CBCL "
            "aggregate score, and ADHD medication variables excluded"
        )
    elif args.task == "adhd_presentation":
        labels, adhd_details = build_adhd_presentation_labels(
            root,
            args.session,
            strict_symptoms=args.adhd_presentation_strict_symptoms,
        )
        target_tables = [ADHD_TARGET_TABLE]
        class_values = [0, 1, 2]
        target_definition = (
            "full-ADHD predominantly inattentive / predominantly "
            "hyperactive-impulsive / combined presentation"
        )
        class_meanings = adhd_details["class_mapping"]
        leakage_control = (
            "parent K-SADS ADHD table, direct CBCL ADHD/attention items, every CBCL "
            "aggregate score, and ADHD medication variables excluded"
        )
    elif args.task == "adhd":
        labels, adhd_details = build_adhd_labels(root, args.session)
        target_tables = [ADHD_TARGET_TABLE]
        class_values = [0, 1]
        target_definition = (
            "baseline parent K-SADS full attention-deficit/hyperactivity disorder "
            "present or past versus assessed controls negative on the available full, "
            "partial-remission, and unspecified fields; algorithmic research "
            "classification, not a clinical diagnosis"
        )
        class_meanings = {
            "0": (
                "assessed control negative on the available full, "
                "partial-remission, and unspecified ADHD fields"
            ),
            "1": "full ADHD present or past",
        }
        leakage_control = (
            "parent K-SADS ADHD table, direct CBCL ADHD/attention items, every CBCL "
            "aggregate score, and ADHD medication variables excluded"
        )
    elif args.task == "bed_2y":
        outcome_sessions = tuple(
            session.strip() for session in args.bed_outcome_sessions.split(",") if session.strip()
        )
        labels, bed_details = build_bed_course_labels(
            root,
            outcome_sessions=outcome_sessions,
            required_session=args.bed_required_session,
            require_all_sessions=args.bed_require_all_sessions,
        )
        target_tables = ["mental_health/mh_p_ksads__ed.parquet"]
        class_values = [0, 1]
        target_definition = (
            "cumulative parent K-SADS symptom-derived probable full binge-eating "
            "disorder phenotype (present or past) observed at any administered "
            f"outcome visit in {list(outcome_sessions)}; not incident onset or a "
            "clinical diagnosis"
        )
        class_meanings = {
            "0": "assessed BED-negative at every required outcome visit",
            "1": "meets symptom-derived probable full BED criteria at one or more outcome visits",
        }
        leakage_control = (
            "all K-SADS eating-disorder variables and direct eating-disorder instruments "
            "excluded; predictors are restricted to the baseline visit"
        )
    else:
        labels = build_labels(root, args.session, args.target)
        target_tables = ["mental_health/mh_p_ksads__ed.parquet"]
        class_values = [0, 1]
        target_definition = args.target
        class_meanings = {"0": "no eating-disorder diagnosis", "1": "eating-disorder diagnosis"}
        leakage_control = "K-SADS ED target table and direct eating-disorder instruments excluded from features"
    if args.task == "adhd_presentation":
        diagnostic_columns = [
            column
            for column in labels.columns
            if column not in {"participant_id", "target"}
        ]
        adhd_details["materialized_label_columns"] = [
            "participant_id", "target"
        ]
        adhd_details["participant_level_diagnostic_fields_materialized"] = False
        adhd_details["builder_diagnostic_fields_not_materialized"] = (
            diagnostic_columns
        )
        labels = labels[["participant_id", "target"]].copy()
    labels.to_parquet(output / "labels.parquet", index=False)
    splits = split_ids(labels, root, args.seed)
    (output / "splits.json").write_text(json.dumps(splits, indent=2))

    if args.task == "cognitive_status" and args.modality_preset == "adni4":
        raise ValueError(
            "The adni4 C modality contains the NIH Toolbox target table. "
            "Use --modality-preset expanded for the leakage-free cognitive task."
        )

    source_preset = ADNI4_SOURCES if args.modality_preset == "adni4" else SOURCES
    preset_source_tables = {
        table for spec in source_preset.values() for table in spec["tables"]
    }
    excluded_predictor_source_tables = (
        sorted(preset_source_tables & ADHD_SOURCE_TABLE_BLACKLIST)
        if adhd_task
        else []
    )
    modality_specs = []
    coverage = {}
    dimensions = {}
    if args.task == "bed_2y":
        excluded_feature_columns = bed_feature_exclusions(root)
    elif adhd_task:
        excluded_feature_columns = adhd_feature_exclusions(root)
    else:
        excluded_feature_columns = set()
    for code, spec in source_preset.items():
        if code in excluded_modality_codes:
            continue
        source_tables = [
            table
            for table in spec["tables"]
            if not (
                args.task == "bed_2y"
                and table in BED_TABLES_WITHOUT_BASELINE_ROWS
            )
            and not (
                adhd_task
                and table in ADHD_SOURCE_TABLE_BLACKLIST
            )
        ]
        table = merge_modality(
            root,
            source_tables,
            args.session,
            excluded_columns=excluded_feature_columns,
        )
        table = table[table["participant_id"].isin(labels["participant_id"])]
        path = output / f"{spec['name']}.parquet"
        table.to_parquet(path, index=False)
        coverage[spec["name"]] = int(len(table))
        dimensions[spec["name"]] = int(table.shape[1] - 1)
        modality_specs.append({
            "code": code,
            "name": spec["name"],
            "path": path.name,
            "max_missing": 0.80,
            "min_variance": 1e-8,
            "source_tables": source_tables,
            "excluded_feature_columns": (
                sorted(excluded_feature_columns)
                if "mental_health/mh_p_cbcl.parquet" in spec["tables"]
                else []
            ),
        })

    manifest = {
        "dataset": "abcd",
        "id_column": "participant_id",
        "label": {"path": "labels.parquet", "column": "target", "class_values": class_values},
        "splits": "splits.json",
        "modalities": modality_specs,
        "provenance": {
            "abcd_root": str(root),
            "session": args.session,
            "task": args.task,
            "modality_preset": args.modality_preset,
            "modality_grouping": {
                "I": "structural MRI + resting fMRI + diffusion MRI",
                "G": "genetic population structure",
                "C": "neurocognition + physical health",
                "B": "non-K-SADS mental health + environment",
            } if args.modality_preset == "adni4" else {},
            "target_definition": target_definition,
            "class_meanings": class_meanings,
            "excluded_modality_codes": sorted(excluded_modality_codes),
            "split_seed": args.seed,
            "cohort_seed": (
                args.cohort_seed if args.task == "adhd_status3" else None
            ),
            "split_strategy": "stratified_group_by_genetic_family",
            "source_target_tables": target_tables,
            "source_sha256": (
                {
                    ADHD_TARGET_TABLE: sha256_file(root / ADHD_TARGET_TABLE),
                    str(Path(ADHD_TARGET_TABLE).with_suffix(".json")): sha256_file(
                        (root / ADHD_TARGET_TABLE).with_suffix(".json")
                    ),
                    "genetic/gn_y_genrel.parquet": sha256_file(
                        root / "genetic/gn_y_genrel.parquet"
                    ),
                }
                if adhd_task
                else {}
            ),
            "source_table_blacklist": (
                sorted(ADHD_SOURCE_TABLE_BLACKLIST) if adhd_task else []
            ),
            "excluded_predictor_source_tables": excluded_predictor_source_tables,
            "excluded_predictor_feature_columns": sorted(excluded_feature_columns),
            "excluded_predictor_feature_count": int(len(excluded_feature_columns)),
            "excluded_source_tables_without_baseline_rows": (
                sorted(BED_TABLES_WITHOUT_BASELINE_ROWS)
                if args.task == "bed_2y"
                else []
            ),
            "target_domains": list(DIAGNOSIS_DOMAINS) if args.task == "diagnostic_burden" else [],
            "bed_endpoint": bed_details,
            "adhd_endpoint": adhd_details,
            "burden_high_start": args.burden_high_start if args.task == "diagnostic_burden" else None,
            "burden_low_end": args.burden_low_end if args.task == "diagnostic_burden" else None,
            "ordinal_target": args.task in {
                "adhd_status3", "diagnostic_burden", "cognitive_status"
            },
            "leakage_control": leakage_control,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    y = labels.set_index("participant_id")["target"]
    summary = {
        "participants": int(len(labels)),
        "class_counts": {
            str(k): int(v) for k, v in labels["target"].value_counts().sort_index().items()
        },
        "class_proportions": {
            str(k): float(v / len(labels))
            for k, v in labels["target"].value_counts().sort_index().items()
        },
        "majority_accuracy": float(labels["target"].value_counts(normalize=True).max()),
        "coverage": coverage,
        "raw_dimensions": dimensions,
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "class_by_split": {
            key: {str(k): int(v) for k, v in y.loc[ids].value_counts().sort_index().items()}
            for key, ids in splits.items()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"manifest={output / 'manifest.json'}")


if __name__ == "__main__":
    main()
