# ABCD ADHD presentation3 + SNP + fixed 15% missingness

This artifact reports the rebuilt ABCD three-class experiment with four modalities: imaging (rs/task-fMRI, T1, DTI), candidate-window SNP dosages, cognition/health, and behavior/environment. No CatBoost, class offset, probability ensemble, ancestry PCs, or external PRS is used.

## Endpoint and cohort

- Class 0: low-symptom non-ADHD comparator.
- Class 1: full ADHD with predominantly inattentive presentation.
- Class 2: full ADHD with a hyperactive/impulsive component (hyperactive/impulsive or combined presentation).
- Cohort: 2,868 participants with class counts 1,270 / 635 / 963.
- Family-disjoint train/validation/test: 2,041 / 414 / 413.
- Fixed missingness: approximately 15% incomplete participants in each split; mostly one or two missing modalities, rare three, never all four.

The detailed Chinese data and biological audit is in [`../docs/ABCD_PRESENTATION3_SNP_V4_ZH.md`](../docs/ABCD_PRESENTATION3_SNP_V4_ZH.md).

## Aggregation rule

Each method was independently trained and evaluated with seeds 31, 32, and 33. Every reported entry is the arithmetic mean and sample standard deviation of the three seed-level metrics. It is not a probability ensemble. Checkpoints were selected by validation Macro-F1, then passed strict validation replay before one-time test evaluation.

## Validation results

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| CERD | 60.47 ± 0.74 | **56.27 ± 0.92** | **75.29 ± 0.18** |
| Flex-MoE | 60.39 ± 1.47 | 53.35 ± 0.86 | 74.15 ± 0.73 |
| I2MoE | **60.55 ± 1.64** | 55.42 ± 0.94 | 74.66 ± 1.04 |
| MoE++ | 59.98 ± 1.70 | 53.04 ± 0.42 | 74.00 ± 0.98 |
| AnyMod | 57.49 ± 2.66 | 52.78 ± 1.26 | 70.63 ± 2.56 |
| AGDiC | 56.60 ± 2.65 | 51.55 ± 2.29 | 70.99 ± 1.55 |
| ACADiff | 57.65 ± 1.75 | 50.95 ± 1.48 | 69.03 ± 0.66 |

## Test results

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| CERD | **59.24 ± 1.65** | **53.62 ± 1.40** | **74.43 ± 1.03** |
| Flex-MoE | 57.71 ± 1.14 | 50.07 ± 1.03 | 72.58 ± 1.25 |
| I2MoE | 55.29 ± 1.33 | 49.12 ± 1.20 | 73.38 ± 1.05 |
| MoE++ | 57.06 ± 0.98 | 48.54 ± 0.78 | 73.44 ± 1.65 |
| AnyMod | 54.48 ± 4.44 | 48.46 ± 4.23 | 69.60 ± 3.16 |
| AGDiC | 54.80 ± 0.92 | 48.58 ± 0.99 | 70.70 ± 0.75 |
| ACADiff | 54.32 ± 1.22 | 47.00 ± 0.71 | 68.31 ± 0.63 |

CERD has the highest three-seed test mean on all three reported metrics. Its numerical margins over the strongest baseline for each metric are +1.53 percentage points in Accuracy, +3.55 in Macro-F1, and +0.99 in Macro-AUROC. No formal paired significance test has been run for this artifact, so the claim is limited to a three-seed numerical comparison.

## CERD seed-level test metrics

| Seed | Best epoch | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---:|---:|---:|---:|---:|
| 31 | 8 | 57.3850 | 52.0586 | 73.3712 |
| 32 | 10 | 59.8063 | 54.0452 | 74.4939 |
| 33 | 8 | 60.5327 | 54.7527 | 75.4376 |
| Mean ± SD | — | 59.2413 ± 1.6481 | 53.6189 ± 1.3967 | 74.4342 ± 1.0345 |

The machine-readable receipt is [`abcd_adhd_presentation3_snp_missing15_v4.json`](abcd_adhd_presentation3_snp_missing15_v4.json). It contains aggregate and per-seed metrics but no participant identifiers, predictions, raw data, or checkpoints.
