#!/usr/bin/env python3
"""Create a train+validation fixed-epoch refit view of a frozen ABCD manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(base: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    source_base = source_path.parent
    source = json.loads(source_path.read_text())
    split_path = Path(resolve_path(source_base, source["splits"]))
    split = json.loads(split_path.read_text())
    train = list(map(str, split["training"]))
    validation = list(map(str, split["validation"]))
    testing = list(map(str, split["testing"]))
    if set(train) & set(validation) or set(train) & set(testing) or set(validation) & set(testing):
        raise ValueError("Source split is not disjoint")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    development = train + validation
    # The shared ABCD loader requires every split key to be non-empty even in
    # validation-only mode. Reserve one development participant in an unused
    # runner-test slot; the loader never iterates it in this protocol.
    runner_dummy_test = [development[-1]]
    refit_split = {
        "training": development[:-1],
        "validation": testing,
        "testing": runner_dummy_test,
    }
    split_output = output_dir / "splits.json"
    split_output.write_text(json.dumps(refit_split, indent=2) + "\n")

    manifest = dict(source)
    manifest["label"] = dict(source["label"])
    manifest["label"]["path"] = resolve_path(source_base, source["label"]["path"])
    manifest["splits"] = "splits.json"
    manifest["modalities"] = []
    for modality in source["modalities"]:
        item = dict(modality)
        item["path"] = resolve_path(source_base, modality["path"])
        manifest["modalities"].append(item)
    if "missingness" in source:
        manifest["missingness"] = dict(source["missingness"])
        manifest["missingness"]["path"] = resolve_path(
            source_base, source["missingness"]["path"]
        )
    manifest["provenance"] = dict(source.get("provenance", {}))
    manifest["provenance"]["refit_protocol"] = {
        "role": "fixed_epoch_train_plus_validation_refit",
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "training_ids": len(refit_split["training"]),
        "held_test_ids_exposed_as_final_validation": len(refit_split["validation"]),
        "unused_development_ids_in_runner_testing_slot": 1,
        "held_test_used_during_training": False,
        "held_test_evaluation_count": 1,
    }
    output_path = output_dir / "manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(output_path),
                "manifest_sha256": sha256_file(output_path),
                "split_sha256": sha256_file(split_output),
                "sizes": {key: len(value) for key, value in refit_split.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
