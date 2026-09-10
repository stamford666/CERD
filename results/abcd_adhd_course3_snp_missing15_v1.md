# ABCD ADHD clinical-course + SNP + fixed 15% missingness

This artifact reports the new clinically interpretable ABCD endpoint using the same four-modality input and fixed missing-modality protocol as the matched baselines. It is a separate task from the earlier presentation3 experiment and does not overwrite that result.

## Endpoint and cohort

| Class | Clinical meaning | Total | Train / validation / test |
|---:|---|---:|---:|
| 0 | No current, past, partial-remission, or unspecified ADHD and no more than three current symptoms in either nine-item domain | 1,776 | 1,254 / 272 / 250 |
| 1 | Past full ADHD or partial remission, without current full or unspecified ADHD | 888 | 654 / 115 / 119 |
| 2 | Current full ADHD with at least six current symptoms in either nine-item domain | 863 | 604 / 119 / 140 |
| **Total** | — | **3,527** | **2,512 / 506 / 509** |

Splitting is stratified by label and grouped by genetic family. Family overlap between every pair of splits is zero. This endpoint represents three baseline clinical states; it must not be described as a neurodegenerative progression analogous to ADNI.

The four modalities are imaging (`I=785`), direct candidate-window SNP dosages (`G=1,183`), cognition/health (`C=90`), and behavior/environment (`B=221`, of which 219 columns survive training-only preprocessing). Their exact source tables, task-fMRI contrasts, instruments, environmental variables, and processing steps are documented in [`../docs/ABCD_COURSE3_MODALITIES_ZH.md`](../docs/ABCD_COURSE3_MODALITIES_ZH.md).

## Protocol

- The same manifest, family split, preprocessing and fixed missingness table are used by every method.
- Approximately 15% of participants in each split have one to three modalities hidden; no participant has all four modalities hidden.
- Seeds 31/32/33 are independently trained models. Reported values are arithmetic means and sample standard deviations of seed-level metrics, not a probability ensemble.
- Checkpoints are selected by validation Macro-F1 and replayed before formal test evaluation.
- Prediction uses raw softmax argmax. There is no class-logit offset, threshold tuning, CatBoost replacement, PRS, ancestry PC, or test-time calibration.

## Matched test results

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **61.43 ± 0.30** | 51.14 ± 0.72 | **74.88 ± 0.34** |
| Flex-MoE | 60.84 ± 1.67 | 52.52 ± 1.11 | 73.96 ± 0.88 |
| I2MoE | 58.28 ± 1.50 | 51.86 ± 0.65 | 74.83 ± 0.89 |
| MoE++ | 58.22 ± 1.39 | 49.81 ± 0.88 | 74.20 ± 0.99 |
| AnyMod | 60.77 ± 1.39 | **53.17 ± 1.19** | 73.64 ± 1.10 |
| AGDiC | 60.64 ± 1.59 | 53.16 ± 1.12 | 72.69 ± 0.44 |
| ACADiff | 56.91 ± 0.45 | 47.48 ± 1.01 | 69.75 ± 0.76 |

CERD has the highest three-seed mean Accuracy and Macro-AUROC, but not the highest Macro-F1. Its Accuracy margin over Flex-MoE is 0.59 percentage points and its Macro-AUROC margin over I2MoE is 0.04 points; neither should be called statistically significant without a formal paired analysis. AnyMod has the highest Macro-F1, 2.03 points above CERD.

The remaining difficulty is concentrated in the past/remission class. Its clinical definition is clearer than the former presentation boundary, but historical ADHD and current ADHD can still share genetics, brain characteristics and behavioral history. The result therefore separates clinical states imperfectly even though the class rules themselves are mutually exclusive.

## CERD seed-level results

| Seed | Best epoch | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---:|---:|---:|---:|---:|
| 31 | 7 | 61.4931 | 51.1509 | 75.2405 |
| 32 | 19 | 61.1002 | 51.8582 | 74.5747 |
| 33 | 14 | 61.6896 | 50.4184 | 74.8128 |
| **Mean ± SD** | — | **61.4276 ± 0.3001** | **51.1425 ± 0.7199** | **74.8760 ± 0.3374** |

The participant-free machine-readable receipt is [`abcd_adhd_course3_snp_missing15_v1.json`](abcd_adhd_course3_snp_missing15_v1.json). Raw data, IDs, predictions and checkpoints are not published.
