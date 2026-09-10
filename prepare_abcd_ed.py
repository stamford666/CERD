#!/usr/bin/env python3
"""Prepare a reproducible four-modality ABCD eating-disorder experiment.

This is a task preset, not part of the ADNI path. It reads the official ABCD
parquet release in place and writes compact model-ready tables plus a manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


MODALITIES = {
    "I": ("imaging", "imaging/mr_y_smri__t1__gm__dsk.parquet"),
    "G": ("genetic", "genetic/gn_y_popstruct.parquet"),
    "N": ("neurocognition", "neurocognition/nc_y_nihtb.parquet"),
    "P": ("physical_health", "physical_health/ph_y_anthr.parquet"),
}
TARGET_TABLE = "mental_health/mh_p_ksads__ed.parquet"
TARGET_METADATA = "mental_health/mh_p_ksads__ed.json"
DEFAULT_BED_OUTCOME_SESSIONS = ("ses-00A", "ses-01A", "ses-02A")


def numeric_baseline(path: Path, session: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "session_id" in frame:
        frame = frame[frame["session_id"].astype(str) == session]
    numeric = list(frame.select_dtypes(include="number").columns)
    if not numeric:
        raise ValueError(f"No numeric features in {path}")
    out = frame[["participant_id", *numeric]].copy()
    out["participant_id"] = out["participant_id"].astype(str)
    return out.drop_duplicates("participant_id", keep="first")


def build_labels(root: Path, session: str, target: str) -> pd.DataFrame:
    frame = pd.read_parquet(root / TARGET_TABLE)
    metadata = json.loads((root / TARGET_METADATA).read_text())
    frame = frame[frame["session_id"].astype(str) == session].copy()
    present = [
        key for key, value in metadata.items()
        if key in frame and "Diagnosis:" in value.get("Description", "")
        and " - Present " in value.get("Description", "")
    ]
    past = [
        key for key, value in metadata.items()
        if key in frame and "Diagnosis:" in value.get("Description", "")
        and " - Past " in value.get("Description", "")
    ]
    columns = present if target == "present_any" else present + past
    values = frame[columns].astype(str)
    administered = (values != "555").any(axis=1)
    labels = (values == "1").any(axis=1).astype(int)
    out = pd.DataFrame({
        "participant_id": frame.loc[administered, "participant_id"].astype(str),
        "target": labels.loc[administered].to_numpy(),
    })
    return out.drop_duplicates("participant_id", keep="first")


def build_bed_course_labels(
    root: Path,
    outcome_sessions: tuple[str, ...] = DEFAULT_BED_OUTCOME_SESSIONS,
    required_session: str = "ses-02A",
    require_all_sessions: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a disease-specific BED-vs-assessed-BED-negative endpoint.

    Cases meet the parent K-SADS symptom criteria for probable full BED
    (present or past) at any requested visit.  This is a cumulative phenotype
    observed across the requested visits, not an incident-onset endpoint or a
    clinical diagnosis.  The released ED ``*_dx`` scores
    are deliberately not used because ABCD Release 6.0/6.1 documentation asks
    users to derive ED phenotypes from symptoms. Branch-skipped values (888)
    are treated as criterion-negative; an entirely unadministered module (555)
    at a required visit is excluded.  By default, all requested visits must
    contain an administered module so cases and controls have equal assessment
    opportunity.  Negative participants may have
    other eating disorders or psychiatric conditions; this is a one-vs-rest
    BED endpoint, not an eating-disorder-free control design.
    """

    if not outcome_sessions:
        raise ValueError("At least one BED outcome session is required")
    if required_session not in outcome_sessions:
        raise ValueError("The required BED follow-up must be in outcome_sessions")

    frame = pd.read_parquet(root / TARGET_TABLE)
    metadata = json.loads((root / TARGET_METADATA).read_text())
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["session_id"] = frame["session_id"].astype(str)
    required_by_period = {
        period: [
            f"mh_p_ksads__ed__binge__{period}_sx",
            f"mh_p_ksads__ed__binge__charac__{period}_sx",
            f"mh_p_ksads__ed__binge__distrs__{period}_sx",
            f"mh_p_ksads__ed__binge__wkly3mo__{period}_sx",
        ]
        for period in ("pres", "past")
    }
    compensatory_by_period = {
        period: [
            f"mh_p_ksads__ed__compbehav__{period}_sx",
            f"mh_p_ksads__ed__compbehav__wkly3mo__{period}_sx",
        ]
        for period in ("pres", "past")
    }
    anorexia_by_period = {
        period: [
            f"mh_p_ksads__ed__emac__{period}_sx",
            f"mh_p_ksads__ed__fear__obese__{period}_sx",
            f"mh_p_ksads__ed__slfwrth__{period}_sx",
        ]
        for period in ("pres", "past")
    }
    symptom_columns = sorted(
        {
            column
            for collection in (
                required_by_period,
                compensatory_by_period,
                anorexia_by_period,
            )
            for columns in collection.values()
            for column in columns
        }
    )
    missing_columns = sorted(set(symptom_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"BED symptom columns missing from K-SADS table: {missing_columns}")

    administered_ids: dict[str, set[str]] = {}
    for session in outcome_sessions:
        at_session = frame[frame["session_id"] == session].copy()
        session_values = at_session[symptom_columns].apply(pd.to_numeric, errors="coerce")
        # Missing values must not be mistaken for evidence that the module was
        # administered.  Valid ABCD symptom responses are 0/1, while 888 is a
        # valid branch skip following an administered screening question.
        administered = session_values.isin((0, 1, 888)).any(axis=1)
        administered_ids[session] = set(
            at_session.loc[administered, "participant_id"]
        )
    if require_all_sessions:
        eligible_ids = set.intersection(
            *(administered_ids[session] for session in outcome_sessions)
        )
    else:
        eligible_ids = administered_ids[required_session]

    window = frame[
        frame["session_id"].isin(outcome_sessions)
        & frame["participant_id"].isin(eligible_ids)
    ].copy()
    period_cases = []
    for period in ("pres", "past"):
        required_met = (
            window[required_by_period[period]].apply(pd.to_numeric, errors="coerce") == 1
        ).all(axis=1)
        recurrent_compensation = (
            window[compensatory_by_period[period]].apply(pd.to_numeric, errors="coerce") == 1
        ).all(axis=1)
        anorexia_pattern = (
            window[anorexia_by_period[period]].apply(pd.to_numeric, errors="coerce") == 1
        ).all(axis=1)
        period_cases.append(required_met & ~recurrent_compensation & ~anorexia_pattern)
    window["bed_case"] = period_cases[0] | period_cases[1]
    status = (
        window.groupby("participant_id", sort=True)[["bed_case"]]
        .max()
        .reindex(sorted(eligible_ids), fill_value=False)
    )
    labels = pd.DataFrame(
        {
            "participant_id": status.index.astype(str),
            "target": status["bed_case"].astype(int).to_numpy(),
        }
    )
    details: dict[str, object] = {
        "endpoint_type": "cumulative probable BED phenotype",
        "incident_onset_endpoint": False,
        "clinical_diagnosis": False,
        "outcome_sessions": list(outcome_sessions),
        "required_followup_session": required_session,
        "require_all_outcome_sessions": bool(require_all_sessions),
        "administered_by_session": {
            session: len(administered_ids[session]) for session in outcome_sessions
        },
        "phenotype": "symptom-derived probable full BED; released diagnosis scores not used",
        "control_definition": (
            "assessed BED-negative; other eating or psychiatric disorders are not excluded"
        ),
        "required_symptom_columns": required_by_period,
        "compensatory_exclusion_columns": compensatory_by_period,
        "anorexia_exclusion_columns": anorexia_by_period,
        "branch_skipped_888": "criterion-negative",
        "not_administered_555": (
            "excluded when the complete module is unadministered at any required outcome visit"
            if require_all_sessions
            else "excluded when the complete required follow-up module is unadministered"
        ),
        "eligible_participants": int(len(status)),
    }
    return labels, details


def split_ids(labels: pd.DataFrame, root: Path, seed: int) -> dict[str, list[str]]:
    family = pd.read_parquet(
        root / "genetic/gn_y_genrel.parquet",
        columns=["participant_id", "gn_y_genrel_id__fam"],
    ).drop_duplicates("participant_id")
    family["participant_id"] = family["participant_id"].astype(str)
    framed = labels.merge(family, on="participant_id", how="left")
    # Participants without a recorded family become their own group.
    missing = framed["gn_y_genrel_id__fam"].isna()
    framed.loc[missing, "gn_y_genrel_id__fam"] = "missing_" + framed.loc[missing, "participant_id"]
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    remaining_idx, testing_idx = next(outer.split(framed, framed["target"], framed["gn_y_genrel_id__fam"]))
    remaining = framed.iloc[remaining_idx].reset_index(drop=True)
    testing = framed.iloc[testing_idx]
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1)
    train_idx, validation_idx = next(inner.split(remaining, remaining["target"], remaining["gn_y_genrel_id__fam"]))
    train = remaining.iloc[train_idx]
    validation = remaining.iloc[validation_idx]
    return {
        "training": train["participant_id"].tolist(),
        "validation": validation["participant_id"].tolist(),
        "testing": testing["participant_id"].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abcd-root", required=True)
    parser.add_argument("--output-dir", default="./data/abcd_ed")
    parser.add_argument("--session", default="ses-00A")
    parser.add_argument("--target", choices=["present_any", "lifetime_any"], default="lifetime_any")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-per-class", type=int, default=None,
                        help="Keep this many participants per class for a fast integration run.")
    args = parser.parse_args()
    root = Path(args.abcd_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    labels = build_labels(root, args.session, args.target)
    if args.smoke_per_class is not None:
        labels = pd.concat(
            [group.sample(min(len(group), args.smoke_per_class), random_state=args.seed)
             for _, group in labels.groupby("target")],
            ignore_index=True,
        )
    labels.to_parquet(output / "labels.parquet", index=False)
    modality_specs = []
    coverage = {}
    for code, (name, relative_path) in MODALITIES.items():
        table = numeric_baseline(root / relative_path, args.session)
        if args.smoke_per_class is not None:
            table = table[table["participant_id"].isin(labels["participant_id"])]
        table.to_parquet(output / f"{name}.parquet", index=False)
        coverage[name] = int(table["participant_id"].isin(labels["participant_id"]).sum())
        modality_specs.append({"code": code, "name": name, "path": f"{name}.parquet"})

    splits = split_ids(labels, root, args.seed)
    (output / "splits.json").write_text(json.dumps(splits, indent=2))
    manifest = {
        "dataset": "abcd",
        "id_column": "participant_id",
        "label": {"path": "labels.parquet", "column": "target", "class_values": [0, 1]},
        "splits": "splits.json",
        "modalities": modality_specs,
        "provenance": {
            "abcd_root": str(root), "session": args.session,
            "target_definition": args.target, "split_seed": args.seed,
            "split_strategy": "stratified_group_by_genetic_family",
            "source_target_table": TARGET_TABLE,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = {
        "participants": int(len(labels)),
        "positive": int(labels["target"].sum()),
        "negative": int((labels["target"] == 0).sum()),
        "coverage": coverage,
        "split_sizes": {key: len(value) for key, value in splits.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"manifest={output / 'manifest.json'}")


if __name__ == "__main__":
    main()
