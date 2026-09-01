# CERD: Missingness-aware multimodal classification

CERD combines subject-conditioned missing-modality generation, provenance-aware sparse mixture-of-experts fusion, and reliability/confidence-weighted prediction for incomplete multimodal data. This repository is a **core-method reference implementation**: it exposes the central architecture and a generic trainer, but it is not the frozen internal campaign runner and does not support exact numerical reproduction of the reported tables. Baseline implementations, protected data, checkpoints, and participant-level predictions are not included.

## Results

The two historical benchmark sections below report test mean ± population SD over three complete 50-epoch runs (seeds 0/1/2). All metrics except MCC are percentages. Model/profile/checkpoint selection and binary thresholds use validation data only. The later Method-revision study is separate, validation-only, and uses the same six metrics for ABCD and ADNI.

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

### Method revision: validation-only common-six report

This is a separate development study and must not be combined with the test
tables above. In particular, its ABCD endpoint is **not** the preceding 11,671-
participant binary ABCD-ADHD benchmark. It is a 946-participant development
cohort using ses-00A IGCB features to predict the three strict ses-01A ADHD
presentations (inattentive, hyperactive/impulsive, and combined) with five
family-disjoint folds. Its ADNI side performs five-fold validation only within
the 1,480-participant training/development cohort from the preceding 2,116-
participant historical benchmark; the historical 318-participant validation
and 318-participant test partitions are not part of this Method-revision
report. Both Method-revision datasets are evaluated with exactly Accuracy,
Balanced Accuracy, Macro-F1, Weighted-F1, Macro-AUROC, and Macro-AUPRC. Every
arm keeps sparse MoE fusion. Initial arms use the 16-expert/top-4 anchor; the
later capacity screen changes only expert count and top-k while preserving a
25% active-expert ratio. The full revision adds stochastic
observed-subset reconstruction, normalized token-level reconstruction, and
detached normalized-entropy branch confidence. Macro-F1 is primary.

The table reports arithmetic means of the two pooled out-of-fold seed results.
It is descriptive and has no associated significance claim.

| Dataset | Arm | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| ABCD | Legacy mean, seeds 9/10 | 53.01 | 44.28 | 43.43 | 53.93 | 57.33 | 39.31 |
| ABCD | Full revision mean, seeds 9/10 | 53.59 | 44.31 | 43.63 | 54.37 | 57.70 | 39.94 |
| ADNI | Legacy mean, seeds 26/27 | 57.60 | 57.20 | 57.46 | 57.52 | 74.94 | 58.77 |
| ADNI | Full revision mean, seeds 26/27 | 58.38 | 58.53 | 58.60 | 58.51 | 75.68 | 58.89 |

The fixed new-seed confirmation did **not** satisfy its predeclared shared
claim rule. ABCD's pooled Macro-F1 difference (full minus legacy) was +0.63
percentage points at seed 9 but −0.24 points at seed 10; the two-seed mean
difference was only +0.20 points. ADNI's corresponding differences were +0.23
and +2.05 points. This seed sensitivity is why the ABCD result is described as
unstable validation evidence rather than a confirmed improvement.

A later adaptive ABCD-only screen compared six validation configurations at
seed 11. Its screen-selected `t4` candidate changed context dropout, token-loss
weight, and the entropy transform together; relative to its anchor it improved
Macro-F1 by 1.00 points. In the frozen seed-12 rerun on the same participants
and folds, however, all three confirmation checks failed:

| t4 minus anchor | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Seed 11 screen | +0.42 | +1.28 | +1.00 | +0.45 | −0.22 | +0.68 |
| Seed 12 confirmation | −1.90 | +0.89 | −0.17 | −1.44 | −0.54 | −0.66 |

This is a new-training-seed check, not an independent-data confirmation. The
decision was `NOT_CONFIRMED`; `t4` was not adopted, and its seed-11 result is
not attributed to any one of its three simultaneous parameter changes.

A subsequent seed-13 ABCD robustness screen returned to the retained full-
revision anchor. It held context dropout at 0.25 and compared a 2×3 local grid:
detached entropy versus detached exponential-entropy confidence, crossed with
token-loss weights 0.05, 0.075, and 0.10. The gate and sparse MoE remained
fixed. All values below are pooled out-of-fold percentages on the same five
family-disjoint development folds; they are not test results.

| Confidence | Token weight | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entropy (anchor) | 0.050 | 54.55 | 44.16 | 44.16 | 54.71 | 59.03 | 40.97 |
| Exp-entropy | 0.050 | 53.81 | 44.10 | 43.82 | 54.23 | 59.02 | 41.15 |
| Entropy | 0.075 | 54.44 | 43.75 | 43.73 | 54.60 | 59.02 | 40.83 |
| Exp-entropy | 0.075 | 52.11 | 44.23 | 43.38 | 52.95 | 58.88 | 40.88 |
| Entropy | 0.100 | 52.75 | 44.69 | 43.80 | 53.61 | 59.17 | 40.43 |
| Exp-entropy | 0.100 | 52.75 | 44.78 | 43.81 | 53.55 | 59.29 | 40.66 |

Promotion required all four conditions relative to the anchor: Macro-F1 at
least +0.30 points, Accuracy no worse than −0.50 points, Macro-AUROC no worse
than −0.50 points, and Macro-F1 wins in at least three of five folds. Every
non-anchor candidate reduced pooled Macro-F1, so none was eligible. The frozen
decision was `NO_PROMOTION_RETAIN_ANCHOR`; no new-seed confirmation was
launched from this screen.

The seed-14 one-factor balance screen next kept the Method, output gate, and
16-expert/top-4 anchor fixed while changing only class weighting, sampling, or
training-only logit adjustment. None of the five alternatives qualified:

| Seed-14 arm | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | 52.75 | 46.30 | 44.83 | 53.77 | 59.90 | 41.16 |
| Class-weight power 0.3125 | 52.64 | 46.01 | 44.47 | 53.63 | 60.41 | 40.90 |
| Class-weight power 0.375 | 51.59 | 47.02 | 44.47 | 53.08 | 60.39 | 40.87 |
| Sampler power 0.5625 | 51.59 | 45.92 | 43.98 | 52.96 | 60.11 | 41.07 |
| Sampler power 0.625 | 53.49 | 45.32 | 44.45 | 54.16 | 59.66 | 40.32 |
| Logit-adjust tau 0.025 | 52.64 | 46.10 | 44.58 | 53.59 | 60.17 | 40.98 |

The decision was `NO_PROMOTION_RETAIN_ANCHOR`, so the conditional seed-15
confirmation was skipped.

The final capacity lineage changed only sparse-MoE capacity. Seed 16 compared
the retained 16-expert/top-4 anchor with an 8-expert/top-2 compact variant;
both retained one router/fusion layer, the output gate, every Method/balance
setting, and a 25% active-expert ratio:

| Seed-16 arm | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| 16 experts / top-4 anchor | 50.63 | 44.25 | 42.41 | 52.27 | 57.82 | 39.22 |
| 8 experts / top-2 compact | 55.18 | 46.04 | 45.19 | 55.81 | 60.70 | 41.15 |
| Compact minus anchor | +4.55 | +1.79 | +2.77 | +3.55 | +2.87 | +1.93 |

The compact arm met every promotion condition and was rerun at seed 17:

| Seed-17 arm | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| 16 experts / top-4 anchor | 53.07 | 45.07 | 44.04 | 53.96 | 59.20 | 40.07 |
| 8 experts / top-2 compact | 54.02 | 44.55 | 44.21 | 54.27 | 59.73 | 40.94 |
| Compact minus anchor | +0.95 | −0.52 | +0.17 | +0.31 | +0.53 | +0.87 |

Seed 17 strictly improved Macro-F1 and passed both −0.50-point guardrails;
the seed-16/17 mean Macro-F1 delta was +1.47 points. All confirmation checks
passed and the decision was `CONFIRMED`.

For completeness, the following is the descriptive arithmetic mean of the two
pooled out-of-fold seed results:

| Seeds 16/17 mean | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| 16 experts / top-4 anchor | 51.85 | 44.66 | 43.23 | 53.11 | 58.51 | 39.65 |
| 8 experts / top-2 compact | 54.60 | 45.29 | 44.70 | 55.04 | 60.21 | 41.05 |
| Compact minus anchor | +2.75 | +0.63 | +1.47 | +1.93 | +1.70 | +1.40 |

These remain same-cohort, new-training-seed validation results, not
independent-data confirmation, external generalization, or a significance
claim.

The generator output gate is retained. Removing it changed pooled Macro-F1 by
−0.77 points on ABCD and +0.08 points on ADNI, failing the predeclared rule that
required non-degradation on both datasets. MoE is also retained in every arm;
this study is not a dense-versus-MoE comparison.

The public [Method narrative](docs/METHOD.md), [validation protocol](docs/METHOD_REVISION_VALIDATION_PROTOCOL.md),
and [de-identified common-six result](results/method_revision_moe_common6.json)
record the reported arms and aggregate values. They contain no participant-level
predictions, IDs, split lists, checkpoints, local paths, or fold-level arrays.

These public artifacts are not an exact numerical reproduction package. The
JSON files under [`configs/`](configs/) are de-identified descriptive parameter
records and are not consumed by `train.py`. Protected manifests, fold
assignments, fitted preprocessing assets, participant-level outputs,
checkpoints, run receipts, and the internal campaign orchestration are not
released.

The reported campaigns also contain training details outside this core-method
reference: an LRPA rank-4 patch adapter; the ABCD-specific presentation-axis
objective, class-weighted quality branch-auxiliary objective, and fixed
no-recompute dropped-combination policy; and ADNI legacy image imputation plus
final-epoch refitting. The public trainer instead uses validation-selected
checkpointing. Consequently, the released code can be used to inspect and
exercise the core Method switches, but its generic commands must not be
presented as reproducing the reported numbers exactly.

## Method

```text
multimodal features + observed-modality mask
                  │
       modality-specific encoders
                  │
 subject-conditioned latent completion
                  │
 observed/generated provenance marking
                  │
 Transformer + sparse MoE (retained)
                  │
 joint / unimodal / pairwise predictions
                  │
 input reliability × entropy confidence
                  ▼
              prediction
```

The revised Method follows one path: reconstruct missing information, preserve
its provenance, estimate input reliability and predictive confidence, and then
combine diagnostic branches. Training groups the objectives into task,
reconstruction, sparse-router balancing, and robustness terms. The detailed
formulation and its validation boundary are documented in
[`docs/METHOD.md`](docs/METHOD.md).

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

ABCD uses eight baseline modalities (`SRDGNPME`): structural MRI, resting fMRI, diffusion MRI, genetic ancestry/population structure, neurocognition, physical health, non-target mental health, and environment. Target-defining K-SADS ADHD fields and direct/aggregate ADHD proxies must be excluded. Preprocessing is fitted on training participants only. Start from [`data/abcd_adhd_manifest.example.json`](data/abcd_adhd_manifest.example.json); de-identified descriptive parameter records are under [`configs/`](configs/) and are not executable configuration files for `train.py`.

## Layout and limitations

```text
cerd/model.py       encoders, generators, reliability/evidence fusion
cerd/moe.py         sparse MoE/router
cerd/losses.py      CERD objectives
cerd/metrics.py     metrics and validation-only binary thresholding
cerd/datasets/      manifest-driven ABCD adapter
train.py            generic validation-locked trainer and optional formal test
docs/               Method narrative and validation protocol
results/            de-identified aggregate results only
```

ABCD and ADNI data are controlled-access and are not redistributed. Confirm endpoint definitions, release versions, and class semantics with the relevant data dictionaries before publication. The Method-revision artifact is validation-only and must not be described as an independent test, external generalization, or a statistically significant result. See [`data/README.md`](data/README.md) and [`NOTICE.md`](NOTICE.md).
