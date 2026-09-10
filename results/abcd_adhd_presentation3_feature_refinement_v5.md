# ABCD presentation3 feature-refinement audit

This report adds the experiments run after the first presentation3 release. It does not overwrite the v4 artifact: it records the corrected structural MRI representation, the candidate-gene compression study, the added task-performance variables, and the resulting CERD evaluations.

## Task and evaluation

The endpoint is unchanged: low-symptom non-ADHD comparator (class 0), full predominantly inattentive ADHD (class 1), and full ADHD with a hyperactive/impulsive component (class 2). The cohort contains 2,868 participants (1,270 / 635 / 963), split into 2,041 training, 414 validation, and 413 test participants with no genetic-family overlap. The fixed label-independent missingness manifest leaves approximately 15% of each split incomplete; it never removes all four modalities.

Every number below is the arithmetic mean and sample standard deviation of independently trained seeds 31, 32, and 33. No probability ensemble, CatBoost, external PRS, ancestry PCs, or class-logit offsets are used.

## What changed in the data

- Imaging now uses the actual Desikan cortical volumes normalized by intracranial volume, cortical thickness, bilateral subcortical volumes normalized by intracranial volume, intracranial volume, rs/task-fMRI, and DTI FA. The 71 T1 signal-intensity variables previously described as gray-matter volume were removed. The corrected imaging modality has 877 features.
- The primary genetic variant compresses 1,178 retained SNP dosages into one training-fitted principal component for each of 14 prespecified candidate-gene windows. This produces 14 genetic features without external PRS, ancestry PCs, or label-supervised representation learning.
- Cognition/health adds 19 task-performance summaries from SST, n-back, and MID, giving 109 features.
- A secondary behavior/environment variant adds 24 sex, developmental/perinatal, and broad family-history variables. It deliberately excludes ADHD symptoms, diagnoses, medication, and CBCL attention variables.

## Three-seed validation comparison

| Representation | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| Correct structure + 8 PCs/gene + task behavior | 61.11 ± 1.58 | 56.08 ± 0.79 | **75.36 ± 0.59** |
| Correct structure + 1 PC/gene + task behavior | **61.35 ± 0.84** | 56.79 ± 0.70 | 74.95 ± 0.68 |
| Above + etiologic context | 60.55 ± 2.28 | **57.05 ± 1.56** | 74.92 ± 1.28 |
| Correct structure + 1,178 direct SNPs + original non-imaging variables | 60.63 ± 1.05 | 56.79 ± 1.38 | 74.87 ± 1.09 |

The corrected variants converge to a narrow validation range rather than producing a large gain from a single feature block. One component per gene gives the best validation accuracy and similar Macro-F1 while sharply reducing the genetic input from 1,178 SNPs to 14 features. Adding broader etiologic context shifts the balance toward Macro-F1 but increases seed variability. Direct SNP dosages do not improve the three-seed mean, indicating that retaining every dosage mainly restores dimensionality rather than robust predictive information.

## Formal seed-specific checkpoint evaluation

The one-PC-per-gene representation was evaluated using the checkpoint selected by validation Macro-F1 within each seed.

| Seed | Best epoch | Test Accuracy (%) | Test Macro-F1 (%) | Test Macro-AUROC (%) |
|---:|---:|---:|---:|---:|
| 31 | 24 | 58.84 | 54.08 | 74.61 |
| 32 | 14 | 55.93 | 51.75 | 72.96 |
| 33 | 8 | 58.35 | 52.15 | 73.31 |
| Mean ± SD | — | **57.71 ± 1.56** | **52.66 ± 1.25** | **73.63 ± 0.87** |

The validation mean is 61.35% Accuracy, 56.79% Macro-F1, and 74.95% Macro-AUROC. The lower test Accuracy and Macro-F1 show that the validation gain did not fully transfer under seed-specific early stopping.

## Fixed-epoch train-plus-validation refit

A secondary refit fixes training at 14 epochs, the median of the three selected epochs (24, 14, and 8), and trains on the development pool before evaluating the original 413-participant test set. Owing to the shared runner contract, one development participant occupies an unused test slot and the original test metrics are stored under its validation key.

| Seed | Epochs | Test Accuracy (%) | Test Macro-F1 (%) | Test Macro-AUROC (%) |
|---:|---:|---:|---:|---:|
| 31 | 14 | 57.87 | 55.55 | 74.85 |
| 32 | 14 | 61.26 | 56.76 | 76.25 |
| 33 | 14 | 57.14 | 54.17 | 75.05 |
| Mean ± SD | — | **58.76 ± 2.20** | **55.49 ± 1.29** | **75.38 ± 0.76** |

This refit is the latest CERD-only estimate for the corrected representation. It improves Macro-F1 and Macro-AUROC relative to the seed-specific-checkpoint run, but matched baselines have not yet been refit with this representation and training scope. Therefore this artifact reports the CERD result without making a superiority claim or reusing the v4 baseline rows as if they were matched.

The aggregate machine-readable record is [`abcd_adhd_presentation3_feature_refinement_v5.json`](abcd_adhd_presentation3_feature_refinement_v5.json).
