"""Manifest-driven ABCD adapter.

The adapter deliberately lives outside every baseline.  ADNI continues to use
the original, local ``data.py`` implementation; only ``--data abcd`` reaches
this module.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "abcd_manifest.json"


def _manifest_path(args: Any) -> Path:
    raw = getattr(args, "dataset_manifest", None)
    return Path(raw).expanduser().resolve() if raw else DEFAULT_MANIFEST


def _read_manifest(args: Any) -> tuple[dict[str, Any], Path]:
    path = _manifest_path(args)
    if not path.is_file():
        raise FileNotFoundError(
            f"ABCD manifest not found: {path}. Pass --dataset-manifest PATH; "
            "see data/abcd_adhd_manifest.example.json."
        )
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dataset", "abcd").lower() != "abcd":
        raise ValueError(f"Expected an ABCD manifest, got {manifest.get('dataset')!r}")
    return manifest, path.parent


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_abcd_modalities(args: Any) -> dict[str, int]:
    manifest, _ = _read_manifest(args)
    by_code = {item["code"].upper(): item for item in manifest["modalities"]}
    requested = list(dict.fromkeys(str(args.modality).upper()))
    unknown = [code for code in requested if code not in by_code]
    if unknown:
        raise ValueError(
            f"Unknown ABCD modality code(s) {unknown}; manifest provides {sorted(by_code)}"
        )
    return {by_code[code]["name"]: idx for idx, code in enumerate(requested)}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


def _feature_columns(frame: pd.DataFrame, spec: dict[str, Any], id_column: str) -> list[str]:
    explicit = spec.get("feature_columns")
    if explicit:
        missing = sorted(set(explicit) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing feature columns in {spec['path']}: {missing[:10]}")
        return list(explicit)
    excluded = {id_column, *spec.get("exclude_columns", [])}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def _select_session(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    session_value = spec.get("session_value")
    if session_value is None:
        return frame
    session_column = spec.get("session_column", "session_id")
    if session_column not in frame:
        raise ValueError(f"{spec['path']} has no session column {session_column!r}")
    selected = frame[frame[session_column].astype(str) == str(session_value)]
    if selected.empty:
        raise ValueError(f"{spec['path']} has no rows for {session_column}={session_value!r}")
    return selected


def _combination_indices(codes: str) -> dict[str, int]:
    codes = "".join(sorted(set(codes.upper())))
    result = {codes: 0}
    index = 1
    for size in range(len(codes) - 1, 0, -1):
        for subset in combinations(codes, size):
            result["".join(subset)] = index
            index += 1
    return result


def validate_abcd_manifest(args: Any) -> dict[str, Any]:
    manifest, base = _read_manifest(args)
    errors: list[str] = []
    for key in ("id_column", "label", "splits", "modalities"):
        if key not in manifest:
            errors.append(f"missing top-level key: {key}")
    if errors:
        raise ValueError("Invalid ABCD manifest: " + "; ".join(errors))
    referenced = [manifest["label"]["path"], manifest["splits"]]
    referenced += [item["path"] for item in manifest["modalities"]]
    if manifest.get("missingness"):
        referenced.append(manifest["missingness"]["path"])
    missing = [str(_resolve(base, item)) for item in referenced if not _resolve(base, item).is_file()]
    if missing:
        raise FileNotFoundError("ABCD manifest references missing files:\n" + "\n".join(missing))
    codes = [item["code"].upper() for item in manifest["modalities"]]
    names = [item["name"] for item in manifest["modalities"]]
    if len(codes) != len(set(codes)) or len(names) != len(set(names)):
        raise ValueError("ABCD modality codes and names must be unique")
    return manifest


def resolved_abcd_manifest_path(args: Any) -> Path:
    """Return the exact manifest path used by the ABCD adapter."""

    return _manifest_path(args)


def canonical_abcd_subject_ids(args: Any) -> np.ndarray:
    """Return label-table participant IDs in the adapter's canonical row order."""

    manifest = validate_abcd_manifest(args)
    _, base = _read_manifest(args)
    id_col = manifest["id_column"]
    label_spec = manifest["label"]
    labels_df = _select_session(
        _read_frame(_resolve(base, label_spec["path"])), label_spec
    )
    if id_col not in labels_df or label_spec["column"] not in labels_df:
        raise ValueError(
            f"Label table must contain {id_col!r} and {label_spec['column']!r}"
        )
    labels_df = (
        labels_df[[id_col, label_spec["column"]]]
        .dropna()
        .drop_duplicates(id_col)
    )
    return labels_df[id_col].astype(str).to_numpy(dtype=np.str_)


def load_abcd_data(args: Any, modality_dict: dict[str, int], patch_embeddings_cls: Any):
    """Return the same tuple as each baseline's existing ADNI loader."""
    manifest = validate_abcd_manifest(args)
    _, base = _read_manifest(args)
    id_col = manifest["id_column"]
    label_spec = manifest["label"]
    labels_df = _select_session(_read_frame(_resolve(base, label_spec["path"])), label_spec)
    if id_col not in labels_df or label_spec["column"] not in labels_df:
        raise ValueError(f"Label table must contain {id_col!r} and {label_spec['column']!r}")
    labels_df = labels_df[[id_col, label_spec["column"]]].dropna().drop_duplicates(id_col)
    labels_df[id_col] = labels_df[id_col].astype(str)
    raw_labels = labels_df[label_spec["column"]]
    class_values = label_spec.get("class_values")
    if class_values is None:
        class_values = sorted(raw_labels.unique().tolist())
    class_to_idx = {str(value): idx for idx, value in enumerate(class_values)}
    labels = raw_labels.astype(str).map(class_to_idx)
    if labels.isna().any():
        unexpected = sorted(raw_labels[labels.isna()].astype(str).unique())
        raise ValueError(f"Labels outside class_values: {unexpected}")
    labels = labels.to_numpy(dtype=np.int64)
    ids = labels_df[id_col].tolist()
    id_to_idx = {subject_id: idx for idx, subject_id in enumerate(ids)}

    with _resolve(base, manifest["splits"]).open(encoding="utf-8") as handle:
        split_ids = json.load(handle)
    def indices(key: str) -> list[int]:
        missing = [str(x) for x in split_ids[key] if str(x) not in id_to_idx]
        if missing:
            raise ValueError(f"{key} split contains IDs absent from label table: {missing[:10]}")
        return [id_to_idx[str(x)] for x in split_ids[key]]
    train_ids, valid_ids, test_ids = (indices("training"), indices("validation"), indices("testing"))
    split_sets = [set(train_ids), set(valid_ids), set(test_ids)]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("ABCD train/validation/test splits overlap")

    requested_codes = str(args.modality).upper()
    specs = {item["name"]: item for item in manifest["modalities"]}
    data_dict: dict[str, np.ndarray] = {}
    encoder_dict: dict[str, Any] = {}
    input_dims: dict[str, int] = {}
    observed = np.zeros((len(ids), len(modality_dict)), dtype=bool)
    combinations_per_subject = [""] * len(ids)
    device = getattr(args, "torch_device", args.device)

    for name, column_idx in modality_dict.items():
        spec = specs[name]
        frame = _select_session(_read_frame(_resolve(base, spec["path"])), spec)
        modality_id_col = spec.get("id_column", id_col)
        if modality_id_col not in frame:
            raise ValueError(f"{spec['path']} has no ID column {modality_id_col!r}")
        frame[modality_id_col] = frame[modality_id_col].astype(str)
        frame = frame.drop_duplicates(modality_id_col, keep="first").set_index(modality_id_col)
        features = _feature_columns(frame.reset_index(), spec, modality_id_col)
        if not features:
            raise ValueError(f"No numeric features selected for ABCD modality {name!r}")
        numeric = frame[features].apply(pd.to_numeric, errors="coerce")
        train_subjects = [ids[i] for i in train_ids if ids[i] in numeric.index]
        if not train_subjects:
            raise ValueError(f"ABCD modality {name!r} has no training subjects")
        train_frame = numeric.loc[train_subjects]
        max_missing = float(spec.get("max_missing", 1.0))
        min_variance = float(spec.get("min_variance", 0.0))
        keep = train_frame.isna().mean(axis=0) <= max_missing
        numeric = numeric.loc[:, keep]
        train_frame = train_frame.loc[:, keep]
        fill_for_variance = train_frame.median(axis=0).fillna(0.0)
        variances = train_frame.fillna(fill_for_variance).var(axis=0)
        keep = variances > min_variance
        numeric = numeric.loc[:, keep]
        train_frame = train_frame.loc[:, keep]
        features = numeric.columns.tolist()
        if not features:
            raise ValueError(f"All ABCD features were filtered for modality {name!r}")
        available_ids = [subject_id for subject_id in ids if subject_id in numeric.index]
        fill = train_frame.median(axis=0).fillna(0.0)
        scaler = StandardScaler().fit(train_frame.fillna(fill).to_numpy(dtype=np.float32))
        values = scaler.transform(numeric.loc[available_ids].fillna(fill).to_numpy(dtype=np.float32))
        array = np.full((len(ids), len(features)), -2.0, dtype=np.float32)
        row_indices = [id_to_idx[x] for x in available_ids]
        array[row_indices] = values.astype(np.float32)
        observed[row_indices, column_idx] = True
        code = spec["code"].upper()
        for row_idx in row_indices:
            combinations_per_subject[row_idx] += code
        data_dict[name] = array
        input_dims[name] = len(features)
        encoder_dict[name] = patch_embeddings_cls(len(features), args.num_patches, args.hidden_dim).to(device)

    missingness_spec = manifest.get("missingness")
    if missingness_spec:
        missingness = _read_frame(_resolve(base, missingness_spec["path"]))
        if id_col not in missingness:
            raise ValueError(f"Missingness table must contain {id_col!r}")
        missingness[id_col] = missingness[id_col].astype(str)
        missingness = missingness.drop_duplicates(id_col).set_index(id_col).reindex(ids)
        if missingness.index.has_duplicates or missingness.isna().all(axis=1).any():
            raise ValueError("Missingness table must contain exactly one row for every labeled participant")
        column_map = missingness_spec.get("columns", {})
        for name, column_idx in modality_dict.items():
            code = specs[name]["code"].upper()
            column = column_map.get(code, code)
            if column not in missingness:
                raise ValueError(f"Missingness table has no keep-mask column for modality {code!r}")
            values = missingness[column]
            if values.isna().any():
                raise ValueError(f"Missingness keep-mask column {column!r} contains missing values")
            if pd.api.types.is_bool_dtype(values):
                keep_mask = values.to_numpy(dtype=bool)
            elif pd.api.types.is_numeric_dtype(values):
                if not values.isin([0, 1]).all():
                    raise ValueError(f"Missingness keep-mask column {column!r} must contain only 0/1")
                keep_mask = values.to_numpy(dtype=bool)
            else:
                normalized_values = values.astype(str).str.lower()
                if not normalized_values.isin(["true", "false", "0", "1"]).all():
                    raise ValueError(f"Missingness keep-mask column {column!r} is not boolean")
                keep_mask = normalized_values.isin(["true", "1"]).to_numpy(dtype=bool)
            force_missing = ~keep_mask
            data_dict[name][force_missing] = -2.0
            observed[force_missing, column_idx] = False

    combo_map = _combination_indices(requested_codes)
    ordered_modalities = sorted(modality_dict.items(), key=lambda item: item[1])
    ordered_codes = [specs[name]["code"].upper() for name, _ in ordered_modalities]
    normalized = [
        "".join(sorted(code for code, is_observed in zip(ordered_codes, row) if is_observed))
        for row in observed
    ]
    data_dict["modality_comb"] = np.asarray([combo_map.get(value, -1) for value in normalized])
    keep = np.flatnonzero(data_dict["modality_comb"] >= 0)
    keep_set = set(keep.tolist())
    train_ids = [i for i in train_ids if i in keep_set]
    valid_ids = [i for i in valid_ids if i in keep_set]
    test_ids = [i for i in test_ids if i in keep_set]
    if not train_ids or not valid_ids or not test_ids:
        raise ValueError("At least one ABCD split is empty after removing subjects with no selected modality")

    return (
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids,
        len(class_values), input_dims, {}, {}, observed, 0,
    )
