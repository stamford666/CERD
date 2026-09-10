#!/usr/bin/env python3
"""Create a reproducible, ADNI-like mild missingness manifest for ABCD."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RATIOS = {
    "training": 1071 / 1480,
    "validation": 227 / 318,
    "testing": 219 / 318,
}

# Distribution among newly masked, naturally complete IGCB samples. Most cases
# miss one modality; a small tail misses two or three modalities. Natural ABCD
# missingness is retained separately.
MIXED_PATTERN_WEIGHTS = {
    "G": 0.35,
    "I": 0.18,
    "C": 0.10,
    "B": 0.09,
    "IG": 0.08,
    "IC": 0.04,
    "IB": 0.03,
    "GC": 0.03,
    "GB": 0.03,
    "CB": 0.02,
    "IGC": 0.02,
    "IGB": 0.01,
    "ICB": 0.01,
    "GCB": 0.01,
}

# Equal allocation across every non-empty proper subset of IGCB.  This is used
# for the controlled MCAR-like benchmark: selected participants lose a random
# one-, two-, or three-modality subset, but never all four modalities.
UNIFORM_PATTERN_WEIGHTS = {
    pattern: 1.0
    for pattern in (
        "I", "G", "C", "B",
        "IG", "IC", "IB", "GC", "GB", "CB",
        "IGC", "IGB", "ICB", "GCB",
    )
}


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def stable_key(seed: int, split: str, group: str, participant_id: str) -> str:
    raw = f"{seed}|{split}|{group}|{participant_id}".encode()
    return hashlib.sha256(raw).hexdigest()


def apportion_patterns(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate an exact sample count with the largest-remainder method."""
    denominator = sum(weights.values())
    raw = {pattern: total * weight / denominator for pattern, weight in weights.items()}
    counts = {pattern: int(np.floor(value)) for pattern, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(weights, key=lambda pattern: (-(raw[pattern] - counts[pattern]), pattern))
    for pattern in order[:remaining]:
        counts[pattern] += 1
    return {pattern: count for pattern, count in counts.items() if count > 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default="./data/abcd_cog3/manifest.json")
    parser.add_argument("--output-dir", default="./data/abcd_cog3_adni_missing")
    parser.add_argument("--drop-code", default="G")
    parser.add_argument(
        "--pattern-mode", choices=["single", "mixed", "uniform"], default="single",
        help=(
            "single: original one-modality mask; mixed: weighted IGCB masks; "
            "uniform: equal allocation over all one-/two-/three-modality masks"
        ),
    )
    parser.add_argument(
        "--target-missing-ratio",
        type=float,
        default=None,
        help="override the ADNI-derived split ratios with one overall incomplete-sample ratio",
    )
    parser.add_argument(
        "--selection-scope",
        choices=["class", "split"],
        default="class",
        help="class preserves the legacy label-stratified design; split is label-independent",
    )
    parser.add_argument(
        "--apply-before-preprocessing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "apply the frozen keep mask before fitting training-only feature filters, "
            "imputation values, and scalers; the legacy default is enabled only for "
            "uniform masking"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.target_missing_ratio is not None and not 0.0 < args.target_missing_ratio < 1.0:
        raise ValueError("--target-missing-ratio must be strictly between 0 and 1")

    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    source_base = source_manifest_path.parent
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source_manifest_path.read_text())
    id_col = manifest["id_column"]
    splits = json.loads(resolve(source_base, manifest["splits"]).read_text())
    labels = None
    if args.selection_scope == "class":
        labels = pd.read_parquet(resolve(source_base, manifest["label"]["path"]))
        labels[id_col] = labels[id_col].astype(str)
        labels = labels.drop_duplicates(id_col).set_index(id_col)

    modality_ids: dict[str, set[str]] = {}
    for spec in manifest["modalities"]:
        frame = pd.read_parquet(resolve(source_base, spec["path"]), columns=[id_col])
        modality_ids[spec["code"].upper()] = set(frame[id_col].astype(str))
    codes = [spec["code"].upper() for spec in manifest["modalities"]]
    drop_code = args.drop_code.upper()
    if drop_code not in codes:
        raise ValueError(f"Drop code {drop_code!r} not found in manifest codes {codes}")
    if args.pattern_mode in {"mixed", "uniform"} and set(codes) != set("IGCB"):
        raise ValueError(f"Mixed/uniform masking requires IGCB modalities, got {codes}")

    all_ids = list(
        dict.fromkeys(
            str(participant_id)
            for split_ids in splits.values()
            for participant_id in split_ids
        )
    )
    natural = pd.DataFrame(
        {code: [participant_id in modality_ids[code] for participant_id in all_ids] for code in codes},
        index=all_ids,
        dtype=bool,
    )
    keep = pd.DataFrame(True, index=all_ids, columns=codes, dtype=bool)
    label_column = manifest["label"]["column"]
    hidden_by_split: dict[str, int] = {}
    patterns_by_split: dict[str, Counter] = {}

    target_complete_ratios = {
        split: (
            1.0 - args.target_missing_ratio
            if args.target_missing_ratio is not None
            else DEFAULT_RATIOS[split]
        )
        for split in DEFAULT_RATIOS
    }

    for split, ratio in target_complete_ratios.items():
        split_ids = [str(x) for x in splits[split]]
        hidden = 0
        pattern_counter = Counter()
        if args.selection_scope == "class":
            assert labels is not None
            groups = [
                (
                    str(label),
                    [
                        x
                        for x in split_ids
                        if int(labels.at[x, label_column]) == int(label)
                    ],
                )
                for label in manifest["label"]["class_values"]
            ]
        else:
            groups = [("all", split_ids)]
        for group, group_ids in groups:
            naturally_full = [x for x in group_ids if bool(natural.loc[x].all())]
            desired_full = round(ratio * len(group_ids))
            hide_count = max(0, len(naturally_full) - desired_full)
            selected = sorted(
                naturally_full,
                key=lambda participant_id: stable_key(args.seed, split, group, participant_id),
            )[:hide_count]
            if args.pattern_mode == "single":
                keep.loc[selected, drop_code] = False
                pattern_counter[drop_code] += hide_count
            else:
                pattern_weights = (
                    MIXED_PATTERN_WEIGHTS
                    if args.pattern_mode == "mixed"
                    else UNIFORM_PATTERN_WEIGHTS
                )
                allocation = apportion_patterns(hide_count, pattern_weights)
                assignments = [
                    pattern for pattern, count in sorted(allocation.items()) for _ in range(count)
                ]
                assignments.sort(
                    key=lambda pattern: stable_key(args.seed + 17, split, group, pattern)
                )
                selected = sorted(
                    selected,
                    key=lambda participant_id: stable_key(
                        args.seed + 31, split, group, participant_id
                    ),
                )
                for participant_id, pattern in zip(selected, assignments):
                    for code in pattern:
                        keep.loc[participant_id, code] = False
                    pattern_counter[pattern] += 1
            hidden += hide_count
        hidden_by_split[split] = hidden
        patterns_by_split[split] = pattern_counter

    mask = keep.reset_index(names=id_col)
    mask.to_parquet(output / "missingness.parquet", index=False)

    derived = copy.deepcopy(manifest)
    derived["label"]["path"] = os.path.relpath(
        resolve(source_base, manifest["label"]["path"]), output
    )
    derived["splits"] = os.path.relpath(resolve(source_base, manifest["splits"]), output)
    for source_spec, derived_spec in zip(manifest["modalities"], derived["modalities"]):
        derived_spec["path"] = os.path.relpath(resolve(source_base, source_spec["path"]), output)
    apply_before_preprocessing = (
        args.pattern_mode == "uniform"
        if args.apply_before_preprocessing is None
        else args.apply_before_preprocessing
    )
    derived["missingness"] = {
        "path": "missingness.parquet",
        "mode": "keep_mask",
        "seed": args.seed,
        "apply_before_preprocessing": apply_before_preprocessing,
        "design": (
            "fixed incomplete-sample ratio; "
            + (
                "uniform random IGCB one/pair/triple masks"
                if args.pattern_mode == "uniform"
                else (
                    "weighted IGCB one/pair/triple masks"
                    if args.pattern_mode == "mixed"
                    else f"newly hide {drop_code} only"
                )
            )
            if args.target_missing_ratio is not None
            else (
                "ADNI IGCB split-specific complete-sample ratios; mild mixed single/pair/triple masks"
                if args.pattern_mode == "mixed"
                else "ADNI IGCB split-specific complete-sample ratios; newly hide G only"
            )
        ),
        "target_complete_ratios": target_complete_ratios,
    }
    derived.setdefault("provenance", {})["missingness_design"] = {
        "source_manifest": str(source_manifest_path),
        "seed": args.seed,
        "newly_hidden_modality": (
            drop_code if args.pattern_mode == "single" else None
        ),
        "pattern_mode": args.pattern_mode,
        "mixed_pattern_weights": (
            MIXED_PATTERN_WEIGHTS
            if args.pattern_mode == "mixed"
            else (UNIFORM_PATTERN_WEIGHTS if args.pattern_mode == "uniform" else {})
        ),
        "selection_scope": args.selection_scope,
        "selection": (
            "within split, SHA256-stable label-independent MCAR among naturally complete samples"
            if args.selection_scope == "split"
            else "within split and class, SHA256-stable MCAR among naturally complete samples"
        ),
        "target_missing_ratio": args.target_missing_ratio,
        "adni_reference_complete_ratios": DEFAULT_RATIOS,
    }
    (output / "manifest.json").write_text(json.dumps(derived, indent=2))

    final_observed = natural & keep
    if bool((final_observed.sum(axis=1) < 1).any()):
        raise RuntimeError("Generated mask contains an all-modality-missing participant")
    summary = {
        "seed": args.seed,
        "drop_code": drop_code,
        "pattern_mode": args.pattern_mode,
        "selection_scope": args.selection_scope,
        "target_missing_ratio": args.target_missing_ratio,
        "splits": {},
    }
    for split, ids in splits.items():
        ids = [str(x) for x in ids]
        observed_count = final_observed.loc[ids].sum(axis=1)
        split_summary = {
            "n": len(ids),
            "complete": int((observed_count == len(codes)).sum()),
            "complete_ratio": float((observed_count == len(codes)).mean()),
            "newly_masked_samples": hidden_by_split[split],
            "newly_masked_patterns": dict(sorted(patterns_by_split[split].items())),
            "observed_count_distribution": {
                str(k): int(v) for k, v in sorted(Counter(observed_count.tolist()).items())
            },
        }
        if args.pattern_mode == "single":
            split_summary[f"newly_hidden_{drop_code.lower()}"] = hidden_by_split[split]
        summary["splits"][split] = split_summary
    (output / "missingness_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
