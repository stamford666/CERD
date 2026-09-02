import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cerd.datasets.abcd import load_abcd_data, resolve_abcd_modalities


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODALITIES = {
    "S": "structural_mri",
    "R": "resting_fmri",
    "D": "diffusion_mri",
    "G": "genetic",
    "N": "neurocognition",
    "P": "physical_health",
    "M": "mental_health",
    "E": "environment",
}


def test_committed_abcd_interfaces_describe_independent_binary_reference_task():
    manifest = json.loads(
        (ROOT / "data" / "abcd_adhd_manifest.example.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (ROOT / "configs" / "abcd_adhd_mofe.json").read_text(encoding="utf-8")
    )

    assert manifest["label"]["column"] == "target"
    assert manifest["label"]["class_values"] == [0, 1]
    assert {
        item["code"]: item["name"] for item in manifest["modalities"]
    } == EXPECTED_MODALITIES
    assert manifest["provenance"]["predictor_session"] == "ses-00A"
    assert manifest["provenance"]["outcome_sessions"] == ["ses-00A"]
    assert manifest["provenance"]["clinical_diagnosis"] is False
    assert manifest["provenance"]["informant"] == "parent"

    referenced_paths = [manifest["label"]["path"], manifest["splits"]]
    referenced_paths.extend(item["path"] for item in manifest["modalities"])
    assert all(not Path(value).is_absolute() for value in referenced_paths)

    assert config["data"] == "abcd"
    assert "independent public binary reference benchmark" in config["scope"]
    assert "not the frozen three-class dev946" in config["scope"]
    assert config["task"] == "baseline_parent_ksads_full_adhd_present_or_past"
    assert config["experiment_tag"] == "abcd_adhd"
    assert config["variant"] == "mofe"
    assert config["modality"] == "SRDGNPME"
    assert config["dual_boundary_rank_loss_weight"] == 0.0


def test_abcd_loader_accepts_binary_eight_modality_manifest(tmp_path):
    participant_ids = [f"subject-{index}" for index in range(6)]
    labels = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    labels.to_csv(tmp_path / "labels.csv", index=False)
    splits = {
        "training": participant_ids[:2],
        "validation": participant_ids[2:4],
        "testing": participant_ids[4:],
    }
    (tmp_path / "splits.json").write_text(json.dumps(splits), encoding="utf-8")

    modality_specs = []
    for offset, (code, name) in enumerate(EXPECTED_MODALITIES.items()):
        path = f"{name}.csv"
        pd.DataFrame(
            {
                "participant_id": participant_ids,
                "feature": np.arange(6, dtype=np.float32) + offset,
            }
        ).to_csv(tmp_path / path, index=False)
        modality_specs.append(
            {
                "code": code,
                "name": name,
                "path": path,
                "max_missing": 0.8,
                "min_variance": 1e-8,
            }
        )

    manifest = {
        "dataset": "abcd",
        "id_column": "participant_id",
        "label": {
            "path": "labels.csv",
            "column": "target",
            "class_values": [0, 1],
        },
        "splits": "splits.json",
        "modalities": modality_specs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = Namespace(
        dataset_manifest=str(manifest_path),
        modality="SRDGNPME",
        device="cpu",
        torch_device=torch.device("cpu"),
        num_patches=2,
        hidden_dim=4,
    )

    modality_dict = resolve_abcd_modalities(args)
    loaded = load_abcd_data(
        args,
        modality_dict,
        lambda *_: torch.nn.Identity(),
    )
    (
        data_dict,
        _,
        loaded_labels,
        train_ids,
        valid_ids,
        test_ids,
        num_classes,
    ) = loaded[:7]

    assert list(modality_dict) == list(EXPECTED_MODALITIES.values())
    assert num_classes == 2
    assert loaded_labels.tolist() == labels["target"].tolist()
    assert (train_ids, valid_ids, test_ids) == ([0, 1], [2, 3], [4, 5])
    assert data_dict["modality_comb"].tolist() == [0] * 6
