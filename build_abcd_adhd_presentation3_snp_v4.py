#!/usr/bin/env python3
"""Build the frozen ABCD ADHD clinical-presentation three-class dataset.

The endpoint is baseline parent K-SADS clinical presentation:
0 = low-symptom non-ADHD comparator;
1 = full ADHD with predominantly inattentive presentation;
2 = full ADHD with a hyperactive/impulsive component.

Predictors comprise rs/task fMRI + T1 + DTI, directly observed SNP dosages,
and a construct-level set of cognition/health and behavior/environment
measures. Direct ADHD instruments and diagnostic derivatives are not used as
predictors.
Family-disjoint train/validation/test splitting precedes genetic QC and scaling
is left to the downstream manifest loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from abcd_feature_building import (
    ENVIRONMENT_FEATURES,
    build_functional_imaging_modality,
    build_genetic_modality,
    sha256_file,
)
from prepare_abcd_ed import split_ids
from prepare_abcd_multimodal import (
    ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS,
    ADHD_CURRENT_INATTENTIVE_COLUMNS,
    ADHD_PAST_HYPERACTIVE_IMPULSIVE_COLUMNS,
    ADHD_PAST_INATTENTIVE_COLUMNS,
    ADHD_PRESENTATION_THRESHOLD,
    ADHD_TARGET_TABLE,
    ADHD_VALID_DIAGNOSIS_VALUES,
    table_at_session,
)


T1_DTI_TABLES = {
    "imaging/mr_y_smri__t1__gm__dsk.parquet": (
        "baseline Desikan cortical T1 gray-matter signal intensity"
    ),
    "imaging/mr_y_dti__is__fa__wm__dsk.parquet": (
        "baseline Desikan-aligned white-matter fractional anisotropy"
    ),
}

COGNITION_HEALTH_FEATURES = {
    "neurocognition/nc_y_nihtb.parquet": {
        "nc_y_nihtb__crdst__agecorr_score": "DCCS cognitive flexibility, age-corrected",
        "nc_y_nihtb__flnkr__agecor_score": "Flanker inhibitory control/attention, age-corrected",
        "nc_y_nihtb__lswmt__agecor_score": "List Sorting working memory, age-corrected",
        "nc_y_nihtb__picsq__agecor_score": "Picture Sequence memory, age-corrected",
        "nc_y_nihtb__pttcp__agecor_score": "Pattern Comparison processing speed, age-corrected",
    },
    "neurocognition/nc_y_wisc.parquet": {
        "nc_y_wisc__scaled_score": "WISC-V Matrix Reasoning scaled score",
    },
    "neurocognition/nc_y_lmt.parquet": {
        "nc_y_lmt__crct_acc": "Little Man visuospatial accuracy",
        "nc_y_lmt_effncy": "Little Man visuospatial efficiency",
    },
    "neurocognition/nc_y_ravlt.parquet": {
        "nc_y_ravlt__trial5__crct_count": "RAVLT trial-V learning",
        "nc_y_ravlt__trial6__sd__crct_count": "RAVLT short-delay recall",
        "nc_y_ravlt__trial7__ld__crct_count": "RAVLT long-delay recall",
    },
    "physical_health/ph_y_pa.parquet": {
        "ph_y_pa_001": "days with at least 60 minutes physical activity",
        "ph_y_pa_002": "days with muscle-strengthening activity",
        "ph_y_pa_003": "school physical-education days",
    },
}

SLEEP_DOMAINS = {
    "sleep_arousal_mean": ("ph_p_sds__da_", "disorders of arousal"),
    "sleep_initiation_maintenance_mean": (
        "ph_p_sds__dims_",
        "disorders of initiating and maintaining sleep",
    ),
    "sleep_excessive_somnolence_mean": (
        "ph_p_sds__does_",
        "disorders of excessive somnolence",
    ),
    "sleep_hyperhidrosis_mean": ("ph_p_sds__hyphy_", "sleep hyperhidrosis"),
    "sleep_breathing_mean": ("ph_p_sds__sbd_", "sleep-breathing disorders"),
    "sleep_wake_transition_mean": (
        "ph_p_sds__swtd_",
        "sleep-wake transition disorders",
    ),
}

BEHAVIOR_ENVIRONMENT_FEATURES = {
    "mental_health/mh_p_cbcl.parquet": {
        # Retain non-ADHD syndrome/DSM domains that capture clinically common
        # comorbidity and impairment. ADHD, attention-problem, broadband, total,
        # and sluggish-cognitive-tempo scores are deliberately absent.
        "mh_p_cbcl__dsm__anx_tscore": "CBCL DSM-oriented anxiety",
        "mh_p_cbcl__dsm__cond_tscore": "CBCL DSM-oriented conduct problems",
        "mh_p_cbcl__dsm__dep_tscore": "CBCL DSM-oriented depressive problems",
        "mh_p_cbcl__dsm__opp_tscore": "CBCL DSM-oriented oppositional problems",
        "mh_p_cbcl__dsm__somat_tscore": "CBCL DSM-oriented somatic complaints",
        "mh_p_cbcl__ocd_tscore": "CBCL obsessive-compulsive problems",
        "mh_p_cbcl__strs_tscore": "CBCL stress problems",
        "mh_p_cbcl__synd__aggr_tscore": "CBCL aggressive behavior",
        "mh_p_cbcl__synd__anxdep_tscore": "CBCL anxious/depressed",
        "mh_p_cbcl__synd__rule_tscore": "CBCL rule-breaking behavior",
        "mh_p_cbcl__synd__soc_tscore": "CBCL social problems",
        "mh_p_cbcl__synd__som_tscore": "CBCL somatic complaints",
        "mh_p_cbcl__synd__tho_tscore": "CBCL thought problems",
        "mh_p_cbcl__synd__wthdep_tscore": "CBCL withdrawn/depressed",
    },
    "mental_health/mh_y_bisbas.parquet": {
        "mh_y_bisbas__bas__dr_sum": "BAS drive",
        "mh_y_bisbas__bas__fs_sum": "BAS fun seeking",
        "mh_y_bisbas__bas__rr_sum": "BAS reward responsiveness",
        "mh_y_bisbas__bis_sum": "behavioral inhibition",
    },
    "mental_health/mh_y_upps.parquet": {
        "mh_y_upps__nurg_sum": "negative urgency",
        "mh_y_upps__pers_sum": "lack of perseverance",
        "mh_y_upps__plan_sum": "lack of planning",
        "mh_y_upps__purg_sum": "positive urgency",
        "mh_y_upps__sens_sum": "sensation seeking",
    },
    "friends_family_community/fc_p_fes.parquet": {
        "fc_p_fes__confl_mean": "family conflict",
    },
    "friends_family_community/fc_y_pm.parquet": {
        "fc_y_pm_mean": "youth-reported parental monitoring",
    },
    "friends_family_community/fc_p_nsc.parquet": {
        "fc_p_nsc__ns_mean": "parent-reported neighborhood safety",
    },
    "general/ab_p_demo.parquet": {
        "ab_p_demo__income__hhold_001": "household income bracket",
        "ab_p_demo__edu__slf_001": "responding caregiver education",
        "ab_p_demo__edu__prtnr_001": "caregiver partner education",
    },
}

PRENATAL_EXPOSURES = {
    "external_linked_data/le_l_prenatal.parquet": {
        "le_l_prenatal__addr1__no2_mean": "prenatal residential NO2",
        "le_l_prenatal__addr1__o3_mean": "prenatal residential ozone",
        "le_l_prenatal__addr1__pm25_mean": "prenatal residential PM2.5",
    }
}

COGNITION_HEALTH_TABLES = (
    "neurocognition/nc_y_nihtb.parquet",
    "neurocognition/nc_y_wisc.parquet",
    "neurocognition/nc_y_lmt.parquet",
    "neurocognition/nc_y_ravlt.parquet",
    "physical_health/ph_p_sds.parquet",
    "physical_health/ph_y_pa.parquet",
)

BEHAVIOR_ENVIRONMENT_TABLES = (
    "mental_health/mh_p_cbcl.parquet",
    "mental_health/mh_y_bisbas.parquet",
    "mental_health/mh_y_upps.parquet",
    "friends_family_community/fc_p_fes.parquet",
    "friends_family_community/fc_y_pm.parquet",
    "friends_family_community/fc_p_nsc.parquet",
    "general/ab_p_demo.parquet",
)

SAFE_CBCL_TSCORES = {
    "mh_p_cbcl__dsm__anx_tscore",
    "mh_p_cbcl__dsm__cond_tscore",
    "mh_p_cbcl__dsm__dep_tscore",
    "mh_p_cbcl__dsm__opp_tscore",
    "mh_p_cbcl__dsm__somat_tscore",
    "mh_p_cbcl__ocd_tscore",
    "mh_p_cbcl__strs_tscore",
    "mh_p_cbcl__synd__aggr_tscore",
    "mh_p_cbcl__synd__anxdep_tscore",
    "mh_p_cbcl__synd__int_tscore",
    "mh_p_cbcl__synd__rule_tscore",
    "mh_p_cbcl__synd__soc_tscore",
    "mh_p_cbcl__synd__som_tscore",
    "mh_p_cbcl__synd__tho_tscore",
    "mh_p_cbcl__synd__wthdep_tscore",
}

DEMOGRAPHIC_CONTEXT_COLUMNS = {
    "ab_p_demo__child__time_001",
    "ab_p_demo__child__time_001__01",
    "ab_p_demo__income__hhold_001",
    "ab_p_demo__income__slf_001",
    "ab_p_demo__income__prtnr_001",
    "ab_p_demo__edu__slf_001",
    "ab_p_demo__edu__prtnr_001",
    "ab_p_demo__marital__slf_001",
    "ab_p_demo__insur_001",
    "ab_p_demo__insur_002",
    "ab_p_demo__insur_004",
    *(f"ab_p_demo__exp__fam_{index:03d}" for index in range(1, 8)),
}



def stable_ids(ids: list[str], *, seed: int, namespace: str, n: int) -> list[str]:
    return sorted(
        ids,
        key=lambda participant_id: (
            hashlib.sha256(
                f"{seed}|{namespace}|{participant_id}".encode("utf-8")
            ).hexdigest(),
            participant_id,
        ),
    )[:n]


def build_labels(
    root: Path,
    session: str,
    cohort_seed: int,
    control_to_remitted_ratio: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = root / ADHD_TARGET_TABLE
    frame = pd.read_parquet(path)
    frame = frame[frame["session_id"].astype(str).eq(session)].copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if frame["participant_id"].duplicated().any():
        raise ValueError("Duplicate participant in baseline ADHD target table")

    present = "mh_p_ksads__adhd__pres_dx"
    past = "mh_p_ksads__adhd__past_dx"
    partial = "mh_p_ksads__adhd__partrem_dx"
    unspecified = "mh_p_ksads__adhd__unspec_dx"
    status_columns = [present, past, partial, unspecified]
    symptom_columns = [
        *ADHD_CURRENT_INATTENTIVE_COLUMNS,
        *ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS,
    ]
    required = set(status_columns) | set(symptom_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing ADHD endpoint columns: {missing}")

    values = frame[[*status_columns, *symptom_columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = values.isin({0, 1, 888}).all(axis=1)
    inattentive = values[list(ADHD_CURRENT_INATTENTIVE_COLUMNS)].eq(1).sum(axis=1)
    hyperactive = values[list(ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS)].eq(1).sum(axis=1)
    max_domain = pd.concat([inattentive, hyperactive], axis=1).max(axis=1)

    clean_pool = (
        valid
        & values[present].ne(1)
        & values[past].ne(1)
        & values[partial].ne(1)
        & values[unspecified].ne(1)
        & max_domain.le(3)
    )
    past_or_remitted = (
        valid
        & values[present].ne(1)
        & (values[past].eq(1) | values[partial].eq(1))
        & values[unspecified].ne(1)
    )
    current = valid & values[present].eq(1) & max_domain.ge(6)

    class1_count = int(past_or_remitted.sum())
    control_count = int(round(class1_count * control_to_remitted_ratio))
    clean_ids = stable_ids(
        frame.loc[clean_pool, "participant_id"].tolist(),
        seed=cohort_seed,
        namespace="adhd_course3_clean_control",
        n=control_count,
    )
    if len(clean_ids) != control_count:
        raise ValueError("Insufficient clean controls for requested sampling ratio")
    selected_clean = frame["participant_id"].isin(clean_ids)
    included = selected_clean | past_or_remitted | current
    target = np.select(
        [selected_clean, past_or_remitted, current], [0, 1, 2], default=-1
    ).astype(np.int64)
    labels = pd.DataFrame(
        {
            "participant_id": frame.loc[included, "participant_id"].to_numpy(),
            "target": target[included.to_numpy()],
        }
    ).sort_values("participant_id", kind="stable").reset_index(drop=True)
    counts = labels["target"].value_counts().sort_index()
    details = {
        "name": "baseline parent K-SADS ADHD clinical-course status",
        "session": session,
        "informant": "parent",
        "class_mapping": {
            "0": "no current/past/partial-remission/unspecified ADHD and <=3 current symptoms in each domain",
            "1": "past full ADHD or partial remission, without current full or unspecified ADHD",
            "2": "current full ADHD and >=6 current symptoms in at least one nine-item domain",
        },
        "available_status_columns": status_columns,
        "unavailable_empty_status_columns": [
            "mh_p_ksads__adhd__fullrem_dx",
            "mh_p_ksads__adhd__oth_dx",
        ],
        "current_inattentive_columns": list(ADHD_CURRENT_INATTENTIVE_COLUMNS),
        "current_hyperactive_impulsive_columns": list(
            ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS
        ),
        "valid_codes": [0, 1, 888],
        "not_administered_code": 555,
        "raw_baseline_participants": int(len(frame)),
        "valid_participants": int(valid.sum()),
        "clean_control_pool": int(clean_pool.sum()),
        "sampled_clean_controls": int(selected_clean.sum()),
        "past_full_only": int((past_or_remitted & values[past].eq(1)).sum()),
        "partial_remission_only": int(
            (past_or_remitted & values[partial].eq(1)).sum()
        ),
        "current_full_before_symptom_consistency": int(
            (valid & values[present].eq(1)).sum()
        ),
        "current_full_included": int(current.sum()),
        "excluded_unspecified": int((valid & values[unspecified].eq(1)).sum()),
        "excluded_symptom_only_intermediate": int(
            (
                valid
                & values[present].ne(1)
                & values[past].ne(1)
                & values[partial].ne(1)
                & values[unspecified].ne(1)
                & max_domain.ge(4)
            ).sum()
        ),
        "control_sampling": {
            "seed": cohort_seed,
            "method": "SHA256-stable random sample before family-disjoint splitting",
            "target": "retain a prespecified multiple of the observed past/remission class",
            "control_to_remitted_ratio": control_to_remitted_ratio,
        },
        "class_counts": {str(k): int(counts.get(k, 0)) for k in (0, 1, 2)},
        "materialized_label_columns": ["participant_id", "target"],
    }
    return labels, details


def build_presentation_labels(
    root: Path,
    session: str,
    cohort_seed: int,
    control_to_inattentive_ratio: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a clinically interpretable ADHD presentation endpoint."""

    path = root / ADHD_TARGET_TABLE
    frame = pd.read_parquet(path)
    frame = frame[frame["session_id"].astype(str).eq(session)].copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if frame["participant_id"].duplicated().any():
        raise ValueError("Duplicate participant in baseline ADHD target table")

    present = "mh_p_ksads__adhd__pres_dx"
    past = "mh_p_ksads__adhd__past_dx"
    partial = "mh_p_ksads__adhd__partrem_dx"
    unspecified = "mh_p_ksads__adhd__unspec_dx"
    status_columns = [present, past, partial, unspecified]
    present_symptom_columns = [
        *ADHD_CURRENT_INATTENTIVE_COLUMNS,
        *ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS,
    ]
    past_symptom_columns = [
        *ADHD_PAST_INATTENTIVE_COLUMNS,
        *ADHD_PAST_HYPERACTIVE_IMPULSIVE_COLUMNS,
    ]
    required = set(status_columns + present_symptom_columns + past_symptom_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing ADHD endpoint columns: {missing}")

    status = frame[status_columns].apply(pd.to_numeric, errors="coerce")
    present_symptoms = frame[present_symptom_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    past_symptoms = frame[past_symptom_columns].apply(pd.to_numeric, errors="coerce")
    diagnosis_valid = status.isin(set(ADHD_VALID_DIAGNOSIS_VALUES)).all(axis=1)
    present_diagnosis = status[present].eq(1)
    past_diagnosis = status[past].eq(1)
    full_case = diagnosis_valid & (present_diagnosis | past_diagnosis)

    selected_symptoms = pd.DataFrame(
        np.where(
            present_diagnosis.to_numpy()[:, None],
            present_symptoms.to_numpy(),
            past_symptoms.to_numpy(),
        ),
        index=frame.index,
    )
    selected_symptoms_valid = selected_symptoms.isin({0, 1}).all(axis=1)
    present_inattentive = present_symptoms[
        list(ADHD_CURRENT_INATTENTIVE_COLUMNS)
    ].eq(1).sum(axis=1)
    present_hyperactive = present_symptoms[
        list(ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS)
    ].eq(1).sum(axis=1)
    past_inattentive = past_symptoms[
        list(ADHD_PAST_INATTENTIVE_COLUMNS)
    ].eq(1).sum(axis=1)
    past_hyperactive = past_symptoms[
        list(ADHD_PAST_HYPERACTIVE_IMPULSIVE_COLUMNS)
    ].eq(1).sum(axis=1)
    inattentive_count = present_inattentive.where(
        present_diagnosis, past_inattentive
    )
    hyperactive_count = present_hyperactive.where(
        present_diagnosis, past_hyperactive
    )
    threshold = ADHD_PRESENTATION_THRESHOLD
    inattentive_presentation = (
        full_case
        & selected_symptoms_valid
        & inattentive_count.ge(threshold)
        & hyperactive_count.lt(threshold)
    )
    hyperactive_involved = (
        full_case & selected_symptoms_valid & hyperactive_count.ge(threshold)
    )

    current_values_valid = present_symptoms.isin({0, 1, 888}).all(axis=1)
    current_inattentive_count = present_symptoms[
        list(ADHD_CURRENT_INATTENTIVE_COLUMNS)
    ].eq(1).sum(axis=1)
    current_hyperactive_count = present_symptoms[
        list(ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS)
    ].eq(1).sum(axis=1)
    clean_pool = (
        diagnosis_valid
        & current_values_valid
        & status.ne(1).all(axis=1)
        & current_inattentive_count.le(3)
        & current_hyperactive_count.le(3)
    )
    control_count = int(
        round(int(inattentive_presentation.sum()) * control_to_inattentive_ratio)
    )
    clean_ids = stable_ids(
        frame.loc[clean_pool, "participant_id"].tolist(),
        seed=cohort_seed,
        namespace="adhd_presentation3_clean_control",
        n=control_count,
    )
    if len(clean_ids) != control_count:
        raise ValueError("Insufficient clean controls for requested sampling ratio")
    selected_clean = frame["participant_id"].isin(clean_ids)
    included = selected_clean | inattentive_presentation | hyperactive_involved
    target = np.select(
        [selected_clean, inattentive_presentation, hyperactive_involved],
        [0, 1, 2],
        default=-1,
    ).astype(np.int64)
    labels = pd.DataFrame(
        {
            "participant_id": frame.loc[included, "participant_id"].to_numpy(),
            "target": target[included.to_numpy()],
        }
    ).sort_values("participant_id", kind="stable").reset_index(drop=True)
    counts = labels["target"].value_counts().sort_index()
    details = {
        "name": "baseline parent K-SADS ADHD clinical-presentation status",
        "endpoint_type": "low-symptom comparator versus two full-ADHD presentation groups",
        "session": session,
        "informant": "parent",
        "class_mapping": {
            "0": "no present/past/partial-remission/unspecified ADHD and <=3 current symptoms in each domain",
            "1": "full ADHD with predominantly inattentive presentation in the selected episode",
            "2": "full ADHD with a hyperactive/impulsive component in the selected episode (hyperactive/impulsive or combined presentation)",
        },
        "episode_rule": "use present symptoms for present full ADHD; otherwise use past symptoms for past-only full ADHD",
        "presentation_threshold": int(threshold),
        "selected_episode_symptom_rule": "all 18 selected-episode fields must be explicit 0/1",
        "available_status_columns": status_columns,
        "current_inattentive_columns": list(ADHD_CURRENT_INATTENTIVE_COLUMNS),
        "current_hyperactive_impulsive_columns": list(
            ADHD_CURRENT_HYPERACTIVE_IMPULSIVE_COLUMNS
        ),
        "past_inattentive_columns": list(ADHD_PAST_INATTENTIVE_COLUMNS),
        "past_hyperactive_impulsive_columns": list(
            ADHD_PAST_HYPERACTIVE_IMPULSIVE_COLUMNS
        ),
        "valid_diagnosis_codes": sorted(ADHD_VALID_DIAGNOSIS_VALUES),
        "valid_selected_episode_symptom_codes": [0, 1],
        "not_administered_code": 555,
        "branch_skip_code": 888,
        "raw_baseline_participants": int(len(frame)),
        "clean_control_pool": int(clean_pool.sum()),
        "sampled_clean_controls": int(selected_clean.sum()),
        "full_adhd_cases": int(full_case.sum()),
        "strict_selected_episode_cases": int(
            (full_case & selected_symptoms_valid).sum()
        ),
        "predominantly_inattentive_cases": int(inattentive_presentation.sum()),
        "hyperactive_impulsive_only_cases": int(
            (
                full_case
                & selected_symptoms_valid
                & inattentive_count.lt(threshold)
                & hyperactive_count.ge(threshold)
            ).sum()
        ),
        "combined_cases": int(
            (
                full_case
                & selected_symptoms_valid
                & inattentive_count.ge(threshold)
                & hyperactive_count.ge(threshold)
            ).sum()
        ),
        "invalid_selected_episode_cases_excluded": int(
            (full_case & ~selected_symptoms_valid).sum()
        ),
        "below_threshold_full_cases_excluded": int(
            (
                full_case
                & selected_symptoms_valid
                & inattentive_count.lt(threshold)
                & hyperactive_count.lt(threshold)
            ).sum()
        ),
        "control_sampling": {
            "seed": cohort_seed,
            "method": "SHA256-stable random sample before family-disjoint splitting",
            "control_to_inattentive_ratio": control_to_inattentive_ratio,
        },
        "class_counts": {str(k): int(counts.get(k, 0)) for k in (0, 1, 2)},
        "materialized_label_columns": ["participant_id", "target"],
    }
    return labels, details


def read_selected(
    root: Path,
    relative: str,
    definitions: dict[str, str],
    session: str,
    cohort_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    table = table_at_session(root / relative, session)
    missing = sorted(set(definitions) - set(table.columns))
    if missing:
        raise ValueError(f"{relative} is missing selected features: {missing}")
    table = table[["participant_id", *definitions]].copy()
    table = table[table["participant_id"].isin(cohort_ids)]
    receipt = {
        "table": relative,
        "sha256": sha256_file(root / relative),
        "features": definitions,
        "cohort_rows_with_any_selected_feature": int(
            table[list(definitions)].notna().any(axis=1).sum()
        ),
    }
    return table, receipt


def merge_selected_sources(
    *,
    root: Path,
    sources: dict[str, dict[str, str]],
    session: str,
    cohort_ids: set[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    merged: pd.DataFrame | None = None
    receipts = []
    for relative, definitions in sources.items():
        table, receipt = read_selected(
            root, relative, definitions, session, cohort_ids
        )
        merged = (
            table
            if merged is None
            else merged.merge(table, on="participant_id", how="outer", validate="one_to_one")
        )
        receipts.append(receipt)
    if merged is None:
        raise ValueError("At least one selected source is required")
    return merged, receipts


def relevant_feature_columns(relative: str, columns: list[str]) -> list[str]:
    """Select complete, interpretable instrument domains without outcome screening."""

    if relative.endswith("nc_y_nihtb.parquet"):
        return sorted(
            column
            for column in columns
            if column.endswith(("__agecorr_score", "__agecor_score"))
        )
    if relative.endswith("nc_y_wisc.parquet"):
        return [
            column
            for column in ("nc_y_wisc__raw_score", "nc_y_wisc__scaled_score")
            if column in columns
        ]
    if relative.endswith("nc_y_lmt.parquet"):
        summaries = {
            "nc_y_lmt__crct_acc",
            "nc_y_lmt__crct_count",
            "nc_y_lmt__crct_rt",
            "nc_y_lmt__to_count",
            "nc_y_lmt__wrong_count",
            "nc_y_lmt__wrong_prcnt",
            "nc_y_lmt__wrong_rt",
            "nc_y_lmt_effncy",
            "nc_y_lmt_rt",
        }
        return sorted(
            column
            for column in columns
            if column in summaries
            or column.startswith("nc_y_lmt__crct__stimtype")
        )
    if relative.endswith("nc_y_ravlt.parquet"):
        return sorted(
            column
            for column in columns
            if column.startswith("nc_y_ravlt__trial")
            or column.startswith("nc_y_ravlt__lstb")
        )
    if relative.endswith("ph_p_sds.parquet"):
        return sorted(column for column in columns if column.startswith("ph_p_sds__"))
    if relative.endswith("ph_y_pa.parquet"):
        return [column for column in ("ph_y_pa_001", "ph_y_pa_002", "ph_y_pa_003") if column in columns]

    if relative.endswith("mh_p_cbcl.parquet"):
        excluded_fragments = ("__attn", "__adhd_", "__sct_")
        raw_items = {
            column
            for column in columns
            if column.startswith("mh_p_cbcl__")
            and not column.endswith(("_nm", "_sum", "_tscore"))
            and not any(fragment in column for fragment in excluded_fragments)
        }
        return sorted(raw_items | (SAFE_CBCL_TSCORES & set(columns)))
    if relative.endswith("mh_y_bisbas.parquet"):
        return sorted(
            column
            for column in columns
            if column.startswith("mh_y_bisbas__")
            and not column.endswith("_nm")
            and "__v01" not in column
        )
    if relative.endswith("mh_y_upps.parquet"):
        return sorted(
            column
            for column in columns
            if column.startswith("mh_y_upps__") and not column.endswith("_nm")
        )
    if relative.endswith("fc_p_fes.parquet"):
        return sorted(
            column
            for column in columns
            if column.startswith("fc_p_fes__")
            and not column.endswith("_nm")
        )
    if relative.endswith("fc_y_pm.parquet"):
        return sorted(
            column
            for column in columns
            if column.startswith("fc_y_pm_")
            and column not in {"fc_y_pm_age", "fc_y_pm_adm"}
            and not column.endswith("_nm")
        )
    if relative.endswith("fc_p_nsc.parquet"):
        return sorted(
            column
            for column in columns
            if column.startswith("fc_p_nsc__ns_") and not column.endswith("_nm")
        )
    if relative.endswith("ab_p_demo.parquet"):
        return sorted(DEMOGRAPHIC_CONTEXT_COLUMNS & set(columns))
    raise ValueError(f"No construct-level selection rule for {relative}")


def merge_relevant_sources(
    *,
    root: Path,
    relatives: tuple[str, ...],
    session: str,
    cohort_ids: set[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    merged: pd.DataFrame | None = None
    receipts: list[dict[str, object]] = []
    for relative in relatives:
        path = root / relative
        table = table_at_session(path, session)
        selected = relevant_feature_columns(relative, list(table.columns))
        if not selected:
            raise ValueError(f"No relevant features selected from {relative}")
        table = table[["participant_id", *selected]]
        table = table[table["participant_id"].isin(cohort_ids)]
        merged = (
            table
            if merged is None
            else merged.merge(table, on="participant_id", how="outer", validate="one_to_one")
        )
        receipts.append(
            {
                "table": relative,
                "sha256": sha256_file(path),
                "selection": "prespecified construct-level rule; independent of labels",
                "features": selected,
                "feature_count": len(selected),
                "cohort_rows_with_any_selected_feature": int(
                    table[selected].notna().any(axis=1).sum()
                ),
            }
        )
    if merged is None:
        raise ValueError("At least one relevant source is required")
    return merged, receipts


def add_sleep_domains(
    *, root: Path, session: str, cohort_ids: set[str], cognition: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    relative = "physical_health/ph_p_sds.parquet"
    raw = table_at_session(root / relative, session)
    raw = raw[raw["participant_id"].isin(cohort_ids)].copy()
    domain_definitions = {}
    selected_item_columns = []
    derived = pd.DataFrame({"participant_id": raw["participant_id"]})
    for output_column, (prefix, description) in SLEEP_DOMAINS.items():
        columns = sorted(column for column in raw.columns if column.startswith(prefix))
        if not columns:
            raise ValueError(f"No sleep items found for {prefix}")
        minimum = ceil(len(columns) / 2)
        score = raw[columns].mean(axis=1, skipna=True)
        score = score.mask(raw[columns].notna().sum(axis=1) < minimum)
        derived[output_column] = score.astype(np.float32)
        selected_item_columns.extend(columns)
        domain_definitions[output_column] = {
            "description": description,
            "aggregation": f"item mean requiring at least {minimum}/{len(columns)} observed items",
            "items": columns,
        }
    cognition = cognition.merge(
        derived, on="participant_id", how="outer", validate="one_to_one"
    )
    receipt = {
        "table": relative,
        "sha256": sha256_file(root / relative),
        "derived_domains": domain_definitions,
        "raw_items_used": selected_item_columns,
        "cohort_rows_with_any_sleep_domain": int(
            derived[list(SLEEP_DOMAINS)].notna().any(axis=1).sum()
        ),
    }
    return cognition, receipt


def add_structural_imaging(
    *, root: Path, session: str, cohort_ids: set[str], imaging: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    receipts = []
    for relative, definition in T1_DTI_TABLES.items():
        table = table_at_session(root / relative, session)
        feature_columns = [column for column in table if column != "participant_id"]
        table = table[table["participant_id"].isin(cohort_ids)]
        imaging = imaging.merge(
            table, on="participant_id", how="outer", validate="one_to_one"
        )
        receipts.append(
            {
                "table": relative,
                "definition": definition,
                "sha256": sha256_file(root / relative),
                "features": len(feature_columns),
                "cohort_rows": int(len(table)),
            }
        )
    return imaging, receipts


def audit_family_disjoint(splits: dict[str, list[str]], root: Path) -> dict[str, object]:
    family = pd.read_parquet(
        root / "genetic/gn_y_genrel.parquet",
        columns=["participant_id", "gn_y_genrel_id__fam"],
    ).drop_duplicates("participant_id")
    family["participant_id"] = family["participant_id"].astype(str)
    family = family.set_index("participant_id")["gn_y_genrel_id__fam"]
    groups = {}
    for split, ids in splits.items():
        groups[split] = {
            str(family.get(participant_id))
            if pd.notna(family.get(participant_id))
            else f"missing_{participant_id}"
            for participant_id in ids
        }
    overlap = {
        "training_validation": len(groups["training"] & groups["validation"]),
        "training_testing": len(groups["training"] & groups["testing"]),
        "validation_testing": len(groups["validation"] & groups["testing"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Family overlap detected: {overlap}")
    return {"overlap_family_counts": overlap}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abcd-root", type=Path, required=True)
    parser.add_argument("--genotype-prefix", type=Path, required=True)
    parser.add_argument("--gene-ranges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session", default="ses-00A")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--cohort-seed", type=int, default=2026)
    parser.add_argument(
        "--endpoint",
        choices=("presentation3",),
        default="presentation3",
    )
    parser.add_argument("--control-to-remitted-ratio", type=float, default=2.0)
    parser.add_argument("--ld-r2", type=float, default=0.8)
    parser.add_argument("--plink", type=Path, required=True)
    args = parser.parse_args()

    root = args.abcd_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if args.endpoint == "course3":
        labels, endpoint = build_labels(
            root,
            args.session,
            args.cohort_seed,
            args.control_to_remitted_ratio,
        )
    else:
        labels, endpoint = build_presentation_labels(
            root,
            args.session,
            args.cohort_seed,
            args.control_to_remitted_ratio,
        )
    labels.to_parquet(output / "labels.parquet", index=False)
    cohort_ids = set(labels["participant_id"])
    splits = split_ids(labels, root, args.split_seed)
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    family_audit = audit_family_disjoint(splits, root)

    imaging, imaging_receipt = build_functional_imaging_modality(
        abcd_root=root, cohort_ids=cohort_ids, output=output
    )
    imaging, structural_receipts = add_structural_imaging(
        root=root,
        session=args.session,
        cohort_ids=cohort_ids,
        imaging=imaging,
    )
    imaging.to_parquet(output / "imaging.parquet", index=False)
    imaging_receipt.update(
        {
            "definition": "baseline rs-fMRI + SST/n-back/MID task-fMRI + T1 regional signal intensity + DTI FA",
            "structural_mri_included": True,
            "diffusion_mri_included": True,
            "sources": [*imaging_receipt["sources"], *structural_receipts],
            "features": int(imaging.shape[1] - 1),
            "participants_with_any_imaging": int(len(imaging)),
            "output_sha256": sha256_file(output / "imaging.parquet"),
        }
    )
    (output / "imaging_processing_receipt.json").write_text(
        json.dumps(imaging_receipt, indent=2) + "\n"
    )

    genetic, genetic_receipt = build_genetic_modality(
        cohort_ids=cohort_ids,
        train_ids=set(splits["training"]),
        abcd_root=root,
        genotype_prefix=args.genotype_prefix.resolve(),
        gene_ranges_path=args.gene_ranges.resolve(),
        plink=args.plink.resolve(),
        output=output,
        geno=0.02,
        maf=0.01,
        ld_window=50,
        ld_step=5,
        ld_r2=args.ld_r2,
    )
    ancestry_columns = [
        column for column in genetic if column.startswith("gn_y_popstruct_pc__")
    ]
    genetic = genetic.drop(columns=ancestry_columns)
    genetic.to_parquet(output / "genetic.parquet", index=False)
    genetic_receipt.update(
        {
            "definition": "direct additive SNP dosages in 14 prespecified ADHD-relevant gene windows; no PRS or ancestry PCs",
            "ancestry_pc_features": 0,
            "excluded_ancestry_pc_features": len(ancestry_columns),
            "polygenic_risk_score_features": 0,
            "output_sha256": sha256_file(output / "genetic.parquet"),
        }
    )
    genetic_receipt.get("inputs", {}).pop("population_structure_sha256", None)
    (output / "genetic_processing_receipt.json").write_text(
        json.dumps(genetic_receipt, indent=2) + "\n"
    )

    cognition, cognition_receipts = merge_relevant_sources(
        root=root,
        relatives=COGNITION_HEALTH_TABLES,
        session=args.session,
        cohort_ids=cohort_ids,
    )
    cognition.to_parquet(output / "cognition_health.parquet", index=False)

    behavior, behavior_receipts = merge_relevant_sources(
        root=root,
        relatives=BEHAVIOR_ENVIRONMENT_TABLES,
        session=args.session,
        cohort_ids=cohort_ids,
    )
    for extra_sources in (ENVIRONMENT_FEATURES, PRENATAL_EXPOSURES):
        extra, receipts = merge_selected_sources(
            root=root,
            sources=extra_sources,
            session=args.session,
            cohort_ids=cohort_ids,
        )
        behavior = behavior.merge(
            extra, on="participant_id", how="outer", validate="one_to_one"
        )
        behavior_receipts.extend(receipts)
    behavior.to_parquet(output / "behavior_environment.parquet", index=False)

    target_columns = {
        *endpoint["available_status_columns"],
        *endpoint["current_inattentive_columns"],
        *endpoint["current_hyperactive_impulsive_columns"],
        *endpoint.get("past_inattentive_columns", []),
        *endpoint.get("past_hyperactive_impulsive_columns", []),
    }
    predictor_columns = (
        set(imaging.columns)
        | set(genetic.columns)
        | set(cognition.columns)
        | set(behavior.columns)
    )
    target_predictor_overlap = sorted(target_columns & predictor_columns)
    if target_predictor_overlap:
        raise RuntimeError(f"Target columns found among predictors: {target_predictor_overlap}")

    modality_frames = {
        "I": ("imaging", imaging),
        "G": ("genetic", genetic),
        "C": ("cognition_health", cognition),
        "B": ("behavior_environment", behavior),
    }
    source_tables = {
        "I": [item["table"] for item in imaging_receipt["sources"]],
        "G": [str(args.genotype_prefix.resolve()) + ".{bed,bim,fam}"],
        "C": list(COGNITION_HEALTH_TABLES),
        "B": [
            *BEHAVIOR_ENVIRONMENT_TABLES,
            *ENVIRONMENT_FEATURES,
            *PRENATAL_EXPOSURES,
        ],
    }
    modalities = []
    for code, (name, frame) in modality_frames.items():
        modalities.append(
            {
                "code": code,
                "name": name,
                "path": f"{name}.parquet",
                "max_missing": 0.80,
                "min_variance": 1e-8,
                "feature_columns": [
                    column for column in frame if column != "participant_id"
                ],
                "source_tables": source_tables[code],
            }
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
            "abcd_root": str(root),
            "session": args.session,
            "task": f"adhd_{args.endpoint}",
            "target_definition": endpoint["name"],
            "class_meanings": endpoint["class_mapping"],
            "adhd_endpoint": endpoint,
            "ordinal_target": args.endpoint == "course3",
            "split_seed": args.split_seed,
            "cohort_seed": args.cohort_seed,
            "split_strategy": "stratified_group_by_genetic_family",
            "family_audit": family_audit,
            "modality_grouping": {
                "I": "rs-fMRI + SST/n-back/MID task-fMRI + T1 regional signal intensity + DTI FA",
                "G": "direct SNP dosages only; no PRS or population-structure PCs",
                "C": "executive control, memory, processing speed, sleep, and physical activity",
                "B": "non-ADHD psychopathology, impulsivity/reward, family/SES/neighborhood, and physical environment",
            },
            "feature_selection": (
                "construct-level prespecification from established instruments; "
                "no outcome-guided univariate feature screening"
            ),
            "excluded_predictor_domains": [
                "parent K-SADS ADHD",
                "CBCL ADHD/attention items and scores",
                "ADHD medication variables",
                "genetic ancestry principal components",
                "externally weighted polygenic risk scores",
            ],
            "target_predictor_column_overlap": target_predictor_overlap,
            "source_target_table": ADHD_TARGET_TABLE,
            "source_target_sha256": sha256_file(root / ADHD_TARGET_TABLE),
            "imaging_processing": imaging_receipt,
            "genetic_processing": genetic_receipt,
            "cognition_health_processing": {
                "sources": cognition_receipts,
                "output_sha256": sha256_file(output / "cognition_health.parquet"),
            },
            "behavior_environment_processing": {
                "sources": behavior_receipts,
                "output_sha256": sha256_file(output / "behavior_environment.parquet"),
            },
            "downstream_preprocessing": (
                "training-only missingness filtering, median imputation, and z-score scaling"
            ),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    indexed_labels = labels.set_index("participant_id")["target"]
    summary = {
        "participants": int(len(labels)),
        "class_counts": {
            str(k): int(v)
            for k, v in labels["target"].value_counts().sort_index().items()
        },
        "majority_accuracy": float(labels["target"].value_counts(normalize=True).max()),
        "split_sizes": {split: len(ids) for split, ids in splits.items()},
        "class_by_split": {
            split: {
                str(k): int(v)
                for k, v in indexed_labels.loc[ids].value_counts().sort_index().items()
            }
            for split, ids in splits.items()
        },
        "family_audit": family_audit,
        "coverage": {
            name: int(frame["participant_id"].isin(cohort_ids).sum())
            for _, (name, frame) in modality_frames.items()
        },
        "raw_dimensions": {
            name: int(frame.shape[1] - 1)
            for _, (name, frame) in modality_frames.items()
        },
        "target_predictor_column_overlap": target_predictor_overlap,
        "endpoint": endpoint,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"manifest={output / 'manifest.json'}")


if __name__ == "__main__":
    main()
