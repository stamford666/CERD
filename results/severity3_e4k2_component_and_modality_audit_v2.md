# CERD component and modality audit

All values are arithmetic means ± sample standard deviations across three
independently trained seeds. Test probabilities are not ensembled. ABCD uses
the same family-disjoint 3,000-participant severity3 cohort and fixed 15%
missingness manifest as the main comparison.

## ABCD component controls

Every one-factor control retains the full four-expert/top-2 configuration except
for the named change. The reliability control fixes every usable modality
reliability to one while retaining entropy confidence and learned branch priors;
it is therefore a single-factor control rather than a uniform replacement of
the complete fusion rule.

| Configuration | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| w/o conditional completion | 64.41 ± 2.77 | 53.34 ± 2.43 | 75.37 ± 3.08 |
| w/o provenance embeddings | 65.03 ± 2.31 | 53.07 ± 1.09 | 77.14 ± 1.75 |
| w/o expert diversity (E=1) | 64.95 ± 1.76 | 52.82 ± 2.22 | 77.79 ± 0.47 |
| Dense FFN | 66.43 ± 2.35 | 53.81 ± 0.77 | 77.30 ± 0.79 |
| w/o multigranular decomposition | 63.63 ± 1.99 | 53.73 ± 0.65 | 76.73 ± 0.63 |
| w/o modality reliability | 66.04 ± 3.28 | 53.91 ± 1.88 | 77.44 ± 1.38 |
| **Full CERD** | **66.82 ± 2.14** | **55.18 ± 1.16** | **77.90 ± 0.55** |

The controls distinguish complementary functions. Replacing all source-centered
readouts by the global branch causes the largest overall accuracy loss (3.19
points), showing that the decomposition contributes information beyond one
pooled decision. Removing conditional completion has the strongest availability-
specific effect: on the 64 incomplete test participants, accuracy, Macro-F1,
and AUROC decrease from 64.58/55.06/80.13 to 58.33/45.87/75.08. Provenance and
modality reliability primarily improve class-balanced prediction, lowering
Macro-F1 by 2.10 and 1.27 points when removed.

The expert controls isolate sparse specialization from parameter count. Full
CERD exceeds the one-expert control by 1.87 Accuracy and 2.36 Macro-F1 points.
It also exceeds the capacity-aligned Dense FFN on all three overall metrics;
on incomplete participants the margins expand to 1.56 Accuracy, 4.13 Macro-F1,
and 2.66 AUROC points. The main benefit of sparse experts is therefore retained
specialization when the available source subset changes, rather than nominal
model size alone.

The participant-free seed records and availability strata are stored in
[`abcd_severity3_e4k2_component_suite_v2.json`](abcd_severity3_e4k2_component_suite_v2.json).

## ADNI component controls

| Configuration | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| w/o conditional completion | 63.73 ± 0.79 | 62.82 ± 1.58 | 79.48 ± 0.41 |
| w/o provenance embeddings | 63.52 ± 1.57 | 63.41 ± 1.95 | 80.50 ± 0.70 |
| w/o expert diversity (E=1) | 64.05 ± 0.65 | 63.70 ± 0.59 | **81.13 ± 1.02** |
| w/o multigranular decomposition | 62.47 ± 0.79 | 62.36 ± 1.00 | 79.72 ± 1.32 |
| w/o reliability-aware weights | 64.05 ± 1.49 | 63.88 ± 0.21 | 80.95 ± 0.19 |
| **Full CERD** | **65.72 ± 1.09** | **64.56 ± 2.09** | 81.07 ± 0.55 |

ADNI provides the same component ordering in Accuracy and Macro-F1: the full
model exceeds every removal, and the largest loss again follows removal of the
multigranular readout. The one-expert model has a 0.06-point higher mean AUROC,
but loses 1.67 Accuracy and 0.86 Macro-F1 points, separating ranking parity from
the benefit of sparse specialization at the operating decision.

## Frozen-checkpoint modality intervention

For each originally complete test participant, one source is zeroed, marked
unavailable, and excluded from conditional completion. Prediction-change
relevance is the label-free class-flip rate normalized across the four sources;
positive accuracy decreases are normalized independently. Both four-entry
vectors sum to 100% within each seed.

| Dataset | Modality | Prediction-change share (%) | Accuracy-drop share (%) | Accuracy decrease (points) |
|---|---|---:|---:|---:|
| ADNI | MRI | 35.44 ± 3.79 | 44.03 ± 3.10 | 43.07 ± 1.47 |
| ADNI | Genetics | 17.55 ± 3.82 | 9.19 ± 10.52 | 9.28 ± 12.36 |
| ADNI | Clinical | 10.44 ± 2.25 | 6.61 ± 3.80 | 6.70 ± 4.34 |
| ADNI | Biospecimen | 36.57 ± 1.90 | 40.17 ± 9.75 | 38.81 ± 5.94 |
| ABCD | Imaging | 19.21 ± 9.62 | 14.04 ± 23.72 | 9.98 ± 18.99 |
| ABCD | Genetics | 30.90 ± 1.04 | 57.05 ± 14.71 | 28.75 ± 2.24 |
| ABCD | Cognition/health | 23.88 ± 6.43 | 12.79 ± 7.73 | 5.95 ± 2.75 |
| ABCD | Behavior/environment | 26.01 ± 3.09 | 16.12 ± 2.98 | 8.42 ± 2.22 |

Prediction-change and accuracy-drop shares have Pearson correlations of 0.979
on ADNI and 0.825 on ABCD; Spearman rho is 0.8 on both. The audit recovers the
expected cohort-specific structure: structural MRI and biospecimens dominate
ADNI, whereas QC/LD-pruned SNP dosage is the strongest functional dependency
for the ABCD ADHD-severity decision. The latter result is obtained by removing
the model's actual genetic input, not from a marginal association or an external
PRS.

The full frozen-checkpoint replay receipt is
[`cerd_three_seed_modality_audit_severity3_e4k2_v2.json`](cerd_three_seed_modality_audit_severity3_e4k2_v2.json).
