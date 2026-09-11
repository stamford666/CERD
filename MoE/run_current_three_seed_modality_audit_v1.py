#!/usr/bin/env python3
"""Three-seed CERD strict-removal modality faithfulness audit.

The audit replays the frozen validation-selected checkpoints.  For each seed,
it evaluates the ordinary test input and then, on originally complete test
participants, zeros one modality token block, marks that modality unavailable,
and disables completion.  Seed-level metrics are aggregated by arithmetic
mean and sample standard deviation; probabilities are never ensembled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import baseline_runner
from data import create_loaders, get_modality_combinations, load_and_preprocess_data


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METRICS = ("accuracy", "macro_f1", "macro_auroc")
DISPLAY = {
    "adni": ("MRI", "Genetics", "Clinical", "Biospecimen"),
    "abcd": ("Imaging", "Genetics", "Cognition/health", "Behavior/environment"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def members(dataset: str) -> list[tuple[int, Path, Path]]:
    if dataset == "adni":
        checkpoint_root = HERE / "fair_two_dataset_tuning_20260910" / "runs" / "adni" / "p16_d030_lr125" / "checkpoints" / "adni"
        reference_root = HERE / "fair_two_dataset_tuning_20260910" / "formal" / "predictions" / "adni"
        seeds = (0, 1, 2)
    else:
        checkpoint_root = Path(
            os.environ.get(
                "ABCD_MODALITY_AUDIT_CHECKPOINT_ROOT",
                str(HERE / "abcd_current_binary_cerd_v1" / "validation" / "p16d030e8k2" / "checkpoints" / "abcd"),
            )
        )
        reference_root = Path(
            os.environ.get(
                "ABCD_MODALITY_AUDIT_REFERENCE_ROOT",
                str(HERE / "abcd_current_binary_cerd_v1" / "formal" / "predictions" / "abcd"),
            )
        )
        seeds = (31, 32, 33)
    result = []
    for seed in seeds:
        checkpoint = checkpoint_root / f"our_moe_seed{seed}.pt"
        reference = reference_root / f"our_moe_seed{seed}.csv"
        checkpoint.resolve(strict=True)
        reference.resolve(strict=True)
        result.append((seed, checkpoint, reference))
    return result


def runtime_args(payload: dict[str, Any], device: torch.device) -> argparse.Namespace:
    args = argparse.Namespace(**dict(payload["args"]))
    require(args.model == "our_moe" and str(args.modality).upper() == "IGCB", "checkpoint is not four-modality CERD")
    if args.data == "abcd" and not Path(args.dataset_manifest).is_absolute():
        args.dataset_manifest = str((ROOT / args.dataset_manifest).resolve(strict=True))
    baseline_runner.effective_hyperparameters(args)
    args.torch_device = device
    args.num_workers = 0
    args.pin_memory = False
    args.n_full_modalities = 4
    return args


def combination_index(codes: list[str], observed: np.ndarray) -> int:
    mapping = get_modality_combinations("".join(codes))
    key = "".join(sorted(code for code, keep in zip(codes, observed.tolist()) if keep))
    require(key in mapping, f"unknown observed combination: {key}")
    return int(mapping[key])


def strict_removal_forward(
    model: torch.nn.Module,
    tokens: list[torch.Tensor],
    hidden_index: int,
    codes: list[str],
) -> np.ndarray:
    batch_size = int(tokens[0].shape[0])
    observed_np = np.ones(4, dtype=bool)
    observed_np[hidden_index] = False
    observed = torch.as_tensor(observed_np, device=tokens[0].device).unsqueeze(0).expand(batch_size, -1)
    combination = torch.full(
        (batch_size,),
        combination_index(codes, observed_np),
        device=tokens[0].device,
        dtype=torch.long,
    )
    masked_tokens = [torch.zeros_like(token) if index == hidden_index else token for index, token in enumerate(tokens)]
    previous = bool(model.disable_completion)
    model.disable_completion = True
    try:
        output = model(
            *masked_tokens,
            observed_mask=observed,
            expert_indices=combination,
            return_importance=False,
            return_recon_loss=False,
        )
    finally:
        model.disable_completion = previous
    return torch.softmax(output["logits"], dim=1).detach().to(dtype=torch.float64, device="cpu").numpy()


def predictions_from_probability(
    probability: np.ndarray, decision_threshold: float | None
) -> np.ndarray:
    if decision_threshold is None:
        return probability.argmax(axis=1)
    return baseline_runner.binary_predictions_at_threshold(
        probability, decision_threshold
    )


def metric_triplet(
    truth: np.ndarray,
    probability: np.ndarray,
    decision_threshold: float | None,
) -> dict[str, float]:
    bundle = baseline_runner.metric_bundle(
        truth,
        predictions_from_probability(probability, decision_threshold),
        probability,
        probability.shape[1],
    )
    return {metric: 100.0 * float(bundle[metric]) for metric in METRICS}


def run_seed(dataset: str, seed: int, checkpoint: Path, reference: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require({"model", "encoders", "args"}.issubset(payload), "incomplete checkpoint")
    args = runtime_args(payload, device)
    require(args.data == dataset and int(args.seed) == seed, "checkpoint identity mismatch")
    baseline_runner.seed_everything(seed)
    modality_dict = baseline_runner.resolve_modality_dict(args)
    modality_names = [name for name, _ in sorted(modality_dict.items(), key=lambda item: item[1])]
    require(len(modality_names) == 4, "expected four modalities")
    codes = list(str(args.modality).upper())
    (
        data_dict,
        encoders,
        all_labels,
        train_ids,
        valid_ids,
        test_ids,
        num_classes,
        input_dims,
        transforms,
        masks,
        observed_array,
        full_modality_index,
    ) = load_and_preprocess_data(args, modality_dict)
    test_loader = create_loaders(
        data_dict,
        observed_array,
        all_labels,
        train_ids,
        valid_ids,
        test_ids,
        args.batch_size,
        0,
        False,
        input_dims,
        transforms,
        masks,
        args.preprocessed,
        args.use_common_ids,
    )[3]
    require(test_loader is not None and num_classes >= 2, "invalid test loader")
    if num_classes == 2:
        decision_protocol = payload.get("decision_protocol")
        require(
            isinstance(decision_protocol, dict)
            and decision_protocol.get("implementation_revision")
            == "validation_absolute_threshold_v1",
            "binary checkpoint lacks its frozen validation threshold",
        )
        decision_threshold = float(decision_protocol["threshold"])
    else:
        decision_threshold = None
    model = baseline_runner.build_model(args, 4, num_classes, full_modality_index).to(device)
    model.load_state_dict(payload["model"], strict=True)
    require(set(encoders) == set(payload["encoders"]), "encoder set mismatch")
    for name, encoder in encoders.items():
        encoder.load_state_dict(payload["encoders"][name], strict=True)
        encoder.to(device).eval()
    model.eval()

    all_probability, all_truth, all_allocation, all_observed = [], [], [], []
    complete_probability, complete_truth = [], []
    removed_probability: list[list[np.ndarray]] = [[], [], [], []]
    with torch.inference_mode():
        for batch in test_loader:
            samples, truth, combination, observed = batch
            combination = combination.to(device)
            observed = observed.to(device)
            tokens, observed_mask = baseline_runner.encode_batch(args, samples, observed, encoders, modality_dict, device)
            output = model(
                *tokens,
                observed_mask=observed_mask,
                expert_indices=combination,
                return_importance=False,
                return_recon_loss=False,
            )
            probability = torch.softmax(output["logits"], dim=1).detach().to(dtype=torch.float64, device="cpu").numpy()
            allocation = output["w"].detach().to(dtype=torch.float64, device="cpu").numpy()
            truth_np = truth.detach().cpu().numpy().astype(np.int64, copy=False)
            observed_np = observed_mask.detach().cpu().numpy().astype(bool, copy=False)
            all_probability.append(probability)
            all_truth.append(truth_np)
            all_allocation.append(allocation)
            all_observed.append(observed_np)
            complete = observed_mask.all(dim=1)
            if complete.any():
                selected_tokens = [token[complete] for token in tokens]
                complete_probability.append(probability[complete.detach().cpu().numpy()])
                complete_truth.append(truth_np[complete.detach().cpu().numpy()])
                for hidden_index in range(4):
                    removed_probability[hidden_index].append(strict_removal_forward(model, selected_tokens, hidden_index, codes))
            if bool(getattr(args, "clear_eval_gate_cache", False)):
                baseline_runner.discard_evaluation_gate_loss(model)

    probability = np.concatenate(all_probability)
    truth = np.concatenate(all_truth)
    allocation = np.concatenate(all_allocation)
    observed = np.concatenate(all_observed)
    complete_probability_np = np.concatenate(complete_probability)
    complete_truth_np = np.concatenate(complete_truth)
    complete_mask = observed.all(axis=1)
    require(int(complete_mask.sum()) == len(complete_truth_np), "complete-case alignment mismatch")
    reference_frame = pd.read_csv(reference)
    columns = [f"probability_class_{index}" for index in range(num_classes)]
    require(len(reference_frame) == len(probability) and all(column in reference_frame for column in columns), "reference schema mismatch")
    reference_probability = reference_frame[columns].to_numpy(dtype=np.float64)
    replay_error = float(np.max(np.abs(probability - reference_probability)))
    require(replay_error <= 2e-3, f"checkpoint replay mismatch: {replay_error}")
    replay_prediction = predictions_from_probability(probability, decision_threshold)
    reference_prediction = predictions_from_probability(
        reference_probability, decision_threshold
    )
    require(
        "prediction" in reference_frame
        and np.array_equal(
            reference_prediction,
            reference_frame["prediction"].to_numpy(dtype=np.int64),
        ),
        "stored prediction does not match its frozen decision rule",
    )
    replay_prediction_mismatches = int(
        np.sum(replay_prediction != reference_prediction)
    )

    full_complete = metric_triplet(
        complete_truth_np, complete_probability_np, decision_threshold
    )
    removed_probability_np = [np.concatenate(chunks) for chunks in removed_probability]
    original_prediction = predictions_from_probability(
        complete_probability_np, decision_threshold
    )
    row_index = np.arange(len(original_prediction))
    original_confidence = complete_probability_np[row_index, original_prediction]
    confidence_decreases = np.stack(
        [
            np.maximum(
                original_confidence
                - probability_removed[row_index, original_prediction],
                0.0,
            )
            for probability_removed in removed_probability_np
        ],
        axis=1,
    )
    decrease_sum = confidence_decreases.sum(axis=1, keepdims=True)
    relevance = np.divide(
        confidence_decreases,
        decrease_sum,
        out=np.full_like(confidence_decreases, 0.25),
        where=decrease_sum > 1e-12,
    )
    rows = []
    for index, name in enumerate(DISPLAY[dataset]):
        probability_removed = removed_probability_np[index]
        removed = metric_triplet(
            complete_truth_np, probability_removed, decision_threshold
        )
        rows.append(
            {
                "modality": name,
                "allocation_percent": 100.0 * float(allocation[:, index].mean()),
                "counterfactual_relevance_percent": 100.0
                * float(relevance[:, index].mean()),
                "counterfactual_flip_rate_percent": 100.0
                * float(
                    np.mean(
                        predictions_from_probability(
                            probability_removed, decision_threshold
                        )
                        != original_prediction
                    )
                ),
                "strict_removal_metrics": removed,
                "decrease": {metric: full_complete[metric] - removed[metric] for metric in METRICS},
            }
        )
    flip_rates = np.asarray(
        [row["counterfactual_flip_rate_percent"] for row in rows],
        dtype=np.float64,
    )
    decision_change_shares = (
        100.0 * flip_rates / flip_rates.sum()
        if flip_rates.sum() > 1e-12
        else np.full(4, 25.0)
    )
    for row, share in zip(rows, decision_change_shares):
        row["decision_change_relevance_percent"] = float(share)
    return {
        "seed": seed,
        "checkpoint_sha256": sha256(checkpoint),
        "reference_sha256": sha256(reference),
        "maximum_reference_replay_error": replay_error,
        "reference_prediction_replay_mismatches": replay_prediction_mismatches,
        "test_rows": int(len(truth)),
        "complete_test_rows": int(complete_mask.sum()),
        "decision_threshold": decision_threshold,
        "full_test_metrics": metric_triplet(
            truth, probability, decision_threshold
        ),
        "full_complete_case_metrics": full_complete,
        "modalities": rows,
    }


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1))}


def aggregate(dataset: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    per_seed_drop_shares = []
    for record in records:
        positive_drops = np.maximum(
            np.asarray(
                [
                    row["decrease"]["accuracy"]
                    for row in record["modalities"]
                ],
                dtype=np.float64,
            ),
            0.0,
        )
        per_seed_drop_shares.append(
            100.0 * positive_drops / positive_drops.sum()
            if positive_drops.sum() > 1e-12
            else np.full(4, 25.0)
        )
    rows = []
    for index, name in enumerate(DISPLAY[dataset]):
        rows.append(
            {
                "modality": name,
                "allocation_percent": mean_sd([record["modalities"][index]["allocation_percent"] for record in records]),
                "counterfactual_relevance_percent": mean_sd(
                    [
                        record["modalities"][index][
                            "counterfactual_relevance_percent"
                        ]
                        for record in records
                    ]
                ),
                "counterfactual_flip_rate_percent": mean_sd(
                    [
                        record["modalities"][index][
                            "counterfactual_flip_rate_percent"
                        ]
                        for record in records
                    ]
                ),
                "decision_change_relevance_percent": mean_sd(
                    [
                        record["modalities"][index][
                            "decision_change_relevance_percent"
                        ]
                        for record in records
                    ]
                ),
                "accuracy_drop_share_percent": mean_sd(
                    [shares[index] for shares in per_seed_drop_shares]
                ),
                "decrease": {
                    metric: mean_sd([record["modalities"][index]["decrease"][metric] for record in records])
                    for metric in METRICS
                },
            }
        )
    relevance_vector = np.asarray(
        [row["decision_change_relevance_percent"]["mean"] for row in rows]
    )
    drop_share_vector = np.asarray(
        [row["accuracy_drop_share_percent"]["mean"] for row in rows]
    )
    relevance_rank = np.argsort(np.argsort(relevance_vector)).astype(np.float64)
    drop_rank = np.argsort(np.argsort(drop_share_vector)).astype(np.float64)
    return {
        "dataset": dataset.upper(),
        "aggregation": "arithmetic mean and sample standard deviation across three independently trained seeds; no probability ensemble",
        "intervention": "originally complete test participants; selected input block zeroed, marked unavailable, and conditional completion disabled",
        "allocation": "mean normalized CERD modality decision allocation over the full test cohort; four shares sum to 100% within each seed",
        "counterfactual_relevance": "label-free positive decrease in the original predicted-class probability after strict modality removal, normalized across four removals per participant; zero-total rows receive a uniform share",
        "decision_change_relevance": "label-free strict-removal prediction-change rates normalized within each independently trained seed; four modality shares sum to 100%",
        "accuracy_drop_share": "positive complete-case accuracy decreases normalized within each independently trained seed before arithmetic aggregation; four shares sum to 100%",
        "faithfulness_correspondence": {
            "pearson_r": float(
                np.corrcoef(relevance_vector, drop_share_vector)[0, 1]
            ),
            "spearman_rho": float(
                np.corrcoef(relevance_rank, drop_rank)[0, 1]
            ),
            "scope": "four modality-level mean shares; descriptive because n=4",
        },
        "seed_records": records,
        "modalities": rows,
        "status": "PASS",
    }


def plot(payload: dict[str, Any], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cerd-modality-audit-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(4)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), sharey=True)
    for axis, dataset, title in zip(
        axes, ("adni", "abcd"), ("ADNI", "ABCD")
    ):
        rows = payload[dataset]["modalities"]
        relevance = np.asarray(
            [row["decision_change_relevance_percent"]["mean"] for row in rows]
        )
        normalized_drop = np.asarray(
            [row["accuracy_drop_share_percent"]["mean"] for row in rows]
        )
        axis.bar(
            x - width / 2,
            relevance,
            width,
            color="#355C7D",
            edgecolor="white",
            linewidth=0.45,
            label="Decision-change relevance",
        )
        axis.bar(
            x + width / 2,
            normalized_drop,
            width,
            color="#D17C2F",
            edgecolor="white",
            linewidth=0.45,
            label="Accuracy-drop share",
        )
        axis.set_title(title, pad=3.0, fontsize=8.7)
        axis.set_xticks(x, ("I", "G", "C", "B"))
        axis.grid(axis="y", color="#e4e4e4", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Normalized contribution (%)")
    axes[1].legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
    )
    fig.tight_layout(pad=0.45, w_pad=1.4)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema": "cerd-three-seed-modality-audit-v2"}
    original_cwd = Path.cwd()
    os.chdir(HERE)
    try:
        for dataset in ("adni", "abcd"):
            records = []
            for seed, checkpoint, reference in members(dataset):
                print(f"[{dataset.upper()}] seed={seed}", flush=True)
                records.append(run_seed(dataset, seed, checkpoint, reference, device))
            payload[dataset] = aggregate(dataset, records)
    finally:
        os.chdir(original_cwd)
    output_json = args.output_dir / "cerd_three_seed_modality_audit_v2.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(payload, args.output_dir / "cerd_two_dataset_modality_drop_v1")
    print(output_json, flush=True)


if __name__ == "__main__":
    main()
