# CERD: Missingness-aware multimodal classification

CERD combines conditional missing-modality generation, sparse mixture-of-experts fusion, and reliability/evidence-weighted prediction for incomplete multimodal data. This repository releases the **main method only**; baseline implementations, protected data, checkpoints, and participant-level predictions are not included.

## Results

Learned entries are test mean ± population SD over three complete 50-epoch runs (seeds 0/1/2). All metrics except MCC are percentages. Model/profile/checkpoint selection and binary thresholds use validation data only.

### ABCD ADHD

Baseline parent K-SADS full ADHD present-or-past versus assessed available-field-negative controls. The family-disjoint cohort has 11,671 participants; train/validation/test = 8,325/1,679/1,667 with 1,145/228/245 positives. This is an algorithmic research endpoint, not a clinical diagnosis. ADHD-AUPRC is primary; no-skill AP is the 14.70% test prevalence.

| Method | Acc | BalAcc | Macro-F1 | W-F1 | ADHD-F1 | Sens. | Spec. | AUROC | ADHD-AUPRC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always negative | 85.30 | 50.00 | 46.03 | 78.54 | 0.00 | 0.00 | 100.00 | 50.00 | 14.70 | 0.000 |
| Flex-MoE (official-code-derived adaptation) | 86.40 ± 0.53 | 68.21 ± 0.89 | 70.00 ± 0.17 | 85.66 ± 0.21 | 47.81 ± 0.72 | 42.45 ± 2.91 | 93.98 ± 1.12 | 83.35 ± 0.26 | 50.32 ± 1.09 | 0.407 ± 0.001 |
| I²MoE (official-code adapter) | 86.32 ± 0.27 | 66.87 ± 1.70 | 68.92 ± 1.20 | 85.34 ± 0.30 | 45.66 ± 2.52 | 39.32 ± 4.34 | 94.42 ± 0.97 | 83.58 ± 0.03 | 50.73 ± 0.32 | 0.389 ± 0.017 |
| **MoE++-corrected (I²MoE-code adapter)** | 86.60 ± 0.46 | **69.23 ± 1.44** | **70.83 ± 0.69** | **85.97 ± 0.20** | **49.39 ± 1.60** | 44.63 ± 4.02 | 93.83 ± 1.18 | **84.26 ± 0.08** | **52.36 ± 0.80** | **0.423 ± 0.010** |
| AnyMod (reimplementation) | 85.46 ± 0.42 | 68.73 ± 1.60 | 69.57 ± 1.04 | 85.09 ± 0.38 | 47.58 ± 2.09 | **45.03 ± 3.95** | 92.43 ± 0.92 | 81.95 ± 0.80 | 46.35 ± 0.82 | 0.394 ± 0.020 |
| AGDiC-inspired | 86.02 ± 0.37 | 68.61 ± 0.63 | 69.97 ± 0.13 | 85.47 ± 0.15 | 48.01 ± 0.51 | 43.95 ± 2.04 | 93.27 ± 0.78 | 82.06 ± 0.54 | 49.70 ± 0.86 | 0.403 ± 0.001 |
| ACADiff-inspired (fair masked denoising) | 86.18 ± 0.25 | 68.14 ± 1.04 | 69.76 ± 0.53 | 85.49 ± 0.03 | 47.48 ± 1.26 | 42.59 ± 2.87 | 93.69 ± 0.78 | 81.99 ± 0.29 | 48.25 ± 1.44 | 0.401 ± 0.007 |
| **CERD (ours)** | **86.90 ± 0.20** | 67.66 ± 1.74 | 69.97 ± 1.27 | 85.89 ± 0.32 | 47.42 ± 2.63 | 40.41 ± 4.33 | **94.91 ± 0.88** | 84.06 ± 0.28 | 51.21 ± 0.63 | 0.412 ± 0.017 |

CERD is first in Accuracy and Specificity and second in ADHD-AUPRC/AUROC. In the frozen confirmatory analysis, no CERD-vs-baseline AUPRC comparison survived Holm correction; CERD versus MoE++-corrected was −1.15 percentage points (95% family/seed bootstrap CI −4.09 to +1.81). We therefore do **not** claim significant superiority on this endpoint.

### ADNI

Three-class IGCB task: 2,116 participants; train/validation/test = 1,480/318/318. Binary-only sensitivity/specificity are not forced onto this multiclass task.

| Method | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Flex-MoE (official-code-derived adaptation) | 62.16 ± 1.04 | 60.30 ± 1.79 | 59.97 ± 1.40 | 61.03 ± 0.99 | 76.98 ± 0.78 | 61.42 ± 1.13 |
| I²MoE (official-code adapter) | 64.05 ± 1.86 | 62.77 ± 2.80 | 61.83 ± 1.54 | 61.91 ± 1.36 | 79.35 ± 1.33 | 66.67 ± 1.65 |
| MoE++-corrected (I²MoE-code adapter) | 59.75 ± 0.44 | 59.55 ± 1.41 | 58.85 ± 1.68 | 59.07 ± 1.49 | 79.23 ± 1.14 | 65.63 ± 1.83 |
| AnyMod (reimplementation) | 64.78 ± 3.78 | 63.95 ± 3.12 | 63.55 ± 3.50 | 64.24 ± 3.59 | 79.77 ± 1.73 | 66.20 ± 2.21 |
| AGDiC-inspired | 58.60 ± 2.75 | 55.49 ± 1.65 | 56.54 ± 1.94 | 58.13 ± 2.44 | 75.10 ± 1.11 | 59.58 ± 2.45 |
| ACADiff-inspired (fair masked denoising) | 55.87 ± 0.39 | 53.69 ± 0.88 | 52.14 ± 0.98 | 53.07 ± 0.96 | 71.10 ± 1.01 | 55.75 ± 0.51 |
| **CERD (ours)** | **64.88 ± 1.19** | **64.23 ± 0.41** | **63.75 ± 1.02** | **64.26 ± 1.27** | **80.29 ± 0.71** | **67.37 ± 1.22** |

## Method

```text
multimodal features + observed-modality mask
                  │
       modality-specific encoders
                  │
 conditional cross-attention generation
       for unavailable modalities
                  │
 observed + generated modality tokens
                  │
       Transformer + sparse MoE
                  │
 joint / unimodal / pairwise predictions
                  │
       reliability × evidence fusion
                  ▼
              prediction
```

Training jointly uses classification, sparse-router balancing, conditional reconstruction, branch supervision, artificial modality dropout, and full-to-reduced-view self-distillation. The MoFe objective is valid for binary ABCD and three-class ADNI; dual-boundary ranking is rejected for binary targets.

## Baselines and protocol

The repository does not contain baseline code. Reported names deliberately retain implementation qualifiers:

| Method | Distinguishing mechanism |
|---|---|
| [Flex-MoE](https://papers.nips.cc/paper_files/paper/2024/hash/b2f2af5403042b1344f4e93b35fb67d9-Abstract-Conference.html) | missing-modality bank and sparse routing |
| [I²MoE](https://proceedings.mlr.press/v267/xin25c.html) | uniqueness/synergy/redundancy interaction experts |
| [MoE++-corrected](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7efe88bb4138d602e56637cfcf713654-Abstract-Conference.html) | learned, constant, copy, and zero experts |
| [AnyMod](https://papers.miccai.org/miccai-2024/814-Paper1760.html) | modality queries, task anchors, Transformer fusion |
| [AGDiC-inspired](https://papers.miccai.org/miccai-2025/0059-Paper0965.html) | recovery plus adaptive graph relations |
| [ACADiff-inspired](https://arxiv.org/abs/2603.09931) | fair masked latent denoising/completion |

For ABCD-ADHD, all seven methods shared four validation-only imbalance profiles (`neutral`, `mild`, `moderate`, `strong`), selected by three-seed mean ADHD-AUPRC. The frozen profiles were Flex-MoE `neutral`, I²MoE `strong`, MoE++ `neutral`, AnyMod `neutral`, AGDiC `moderate`, ACADiff `mild`, and CERD `mild`. CERD additionally disclosed six validation-only structure candidates; `base+mild` won. The seven configurations were frozen before the single formal 7×3 test batch. Checkpoints maximize validation ADHD-AUPRC, and decision thresholds maximize validation Macro-F1. Confirmatory inference used 10,000 whole-family paired randomizations, 10,000 family/seed bootstraps, and Holm correction across six comparisons.

## Run the main method

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/laekov/fastmoe.git
cd fastmoe && USE_NCCL=0 python setup.py install && cd ..
pip install -e .

# Validation only. Change false to true only after selection is frozen.
scripts/train_abcd.sh /path/to/abcd_adhd_manifest.json 0 false outputs/abcd_adhd
scripts/train_adni.sh /path/to/adni 0 false outputs/adni
```

ABCD uses eight baseline modalities (`SRDGNPME`): structural MRI, resting fMRI, diffusion MRI, genetic ancestry/population structure, neurocognition, physical health, non-target mental health, and environment. Target-defining K-SADS ADHD fields and direct/aggregate ADHD proxies must be excluded. Preprocessing is fitted on training participants only. Start from [`data/abcd_adhd_manifest.example.json`](data/abcd_adhd_manifest.example.json); configuration records are under [`configs/`](configs/).

## Layout and limitations

```text
cerd/model.py       encoders, generators, reliability/evidence fusion
cerd/moe.py         sparse MoE/router
cerd/losses.py      CERD objectives
cerd/metrics.py     metrics and validation-only binary thresholding
cerd/datasets/      manifest-driven ABCD adapter
train.py            validation-locked training and optional formal test
```

ABCD and ADNI data are controlled-access and are not redistributed. Confirm endpoint definitions, release versions, and class semantics with the relevant data dictionaries before publication. See [`data/README.md`](data/README.md) and [`NOTICE.md`](NOTICE.md).
