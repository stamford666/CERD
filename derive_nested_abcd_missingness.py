#!/usr/bin/env python3
"""Derive a lower missingness ratio as a strict subset of a parent keep mask."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def stable_key(seed: int, split: str, participant_id: str) -> str:
    return hashlib.sha256(f"{seed}|{split}|all|{participant_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-missing-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    parent_path = args.parent_manifest.resolve()
    source_base = source_path.parent
    parent_base = parent_path.parent
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text())
    parent = json.loads(parent_path.read_text())
    id_col = source["id_column"]
    codes = [spec["code"].upper() for spec in source["modalities"]]
    splits = json.loads(resolve(source_base, source["splits"]).read_text())
    all_ids = [str(pid) for ids in splits.values() for pid in ids]

    natural = pd.DataFrame(True, index=all_ids, columns=codes, dtype=bool)
    for spec in source["modalities"]:
        code = spec["code"].upper()
        frame = pd.read_parquet(resolve(source_base, spec["path"]), columns=[id_col])
        present = set(frame[id_col].astype(str))
        natural[code] = [pid in present for pid in natural.index]

    parent_mask = pd.read_parquet(resolve(parent_base, parent["missingness"]["path"]))
    parent_mask[id_col] = parent_mask[id_col].astype(str)
    parent_keep = parent_mask.set_index(id_col)[codes].astype(bool).loc[all_ids]
    child_keep = pd.DataFrame(True, index=all_ids, columns=codes, dtype=bool)

    summary = {
        "seed": args.seed,
        "target_missing_ratio": args.target_missing_ratio,
        "parent_manifest": str(parent_path),
        "nesting": "every artificial missing cell in the child is also missing in the parent",
        "splits": {},
    }
    for split, raw_ids in splits.items():
        ids = [str(pid) for pid in raw_ids]
        naturally_full = [pid for pid in ids if bool(natural.loc[pid].all())]
        parent_masked = [pid for pid in naturally_full if not bool(parent_keep.loc[pid].all())]
        desired_complete = round((1.0 - args.target_missing_ratio) * len(ids))
        hide_count = len(naturally_full) - desired_complete
        if not 0 <= hide_count <= len(parent_masked):
            raise ValueError(
                f"Cannot nest split={split}: need {hide_count} masked samples from "
                f"parent pool of {len(parent_masked)}"
            )
        selected = sorted(parent_masked, key=lambda pid: stable_key(args.seed, split, pid))[:hide_count]
        child_keep.loc[selected] = parent_keep.loc[selected]
        final_observed = natural.loc[ids] & child_keep.loc[ids]
        observed_count = final_observed.sum(axis=1)
        patterns = Counter(
            "".join(code for code in codes if not bool(child_keep.at[pid, code]))
            for pid in selected
        )
        summary["splits"][split] = {
            "n": len(ids),
            "complete": int((observed_count == len(codes)).sum()),
            "complete_ratio": float((observed_count == len(codes)).mean()),
            "newly_masked_samples": len(selected),
            "newly_masked_patterns": dict(sorted(patterns.items())),
            "observed_count_distribution": {
                str(k): int(v) for k, v in sorted(Counter(observed_count.tolist()).items())
            },
        }

    child_keep.reset_index(names=id_col).to_parquet(output / "missingness.parquet", index=False)
    derived = copy.deepcopy(parent)
    derived["label"]["path"] = os.path.relpath(resolve(parent_base, parent["label"]["path"]), output)
    derived["splits"] = os.path.relpath(resolve(parent_base, parent["splits"]), output)
    for spec in derived["modalities"]:
        parent_spec = next(x for x in parent["modalities"] if x["code"] == spec["code"])
        spec["path"] = os.path.relpath(resolve(parent_base, parent_spec["path"]), output)
    derived["missingness"] = {
        "path": "missingness.parquet",
        "mode": "keep_mask",
        "seed": args.seed,
        "apply_before_preprocessing": True,
        "design": "fixed incomplete-sample ratio nested within the parent random mixed-modality mask",
        "target_complete_ratios": {split: 1.0 - args.target_missing_ratio for split in splits},
    }
    derived.setdefault("provenance", {})["missingness_design"] = {
        "source_manifest": str(source_path),
        "parent_manifest": str(parent_path),
        "seed": args.seed,
        "selection_scope": "split",
        "target_missing_ratio": args.target_missing_ratio,
        "nested_within_parent": True,
        "selection": "SHA256-stable subset of parent artificially masked naturally complete participants",
    }
    (output / "manifest.json").write_text(json.dumps(derived, indent=2) + "\n")
    (output / "missingness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    parent_aligned = parent_keep.loc[child_keep.index, child_keep.columns]
    violations = int(((~child_keep) & parent_aligned).sum().sum())
    if violations:
        raise RuntimeError(f"Nested mask audit failed with {violations} cell violations")
    print(json.dumps(summary, indent=2))
    print("nesting_violations=0")


if __name__ == "__main__":
    main()
