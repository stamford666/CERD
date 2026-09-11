# ABCD current-ADHD binary matched experiment

All values are arithmetic mean ± sample standard deviation over independently
trained seeds 31/32/33. Probabilities are not averaged across seeds.

## Endpoint and cohort

- Class 0: no current, past, partial-remission, or unspecified parent K-SADS
  ADHD diagnosis and at most three current symptoms in each nine-item domain.
- Class 1: current full parent K-SADS ADHD and at least six explicitly coded
  current symptoms in at least one nine-item domain.
- Past-only, remission-only, unspecified, and symptom-only participants are
  excluded rather than forced into either class.
- Cohort: 2,120 participants (1,272 controls; 848 cases).
- Family-disjoint train/validation/test: 1,520/297/303, with class counts
  917/603, 177/120, and 178/125.
- A fixed label-independent manifest leaves 85% complete participants per
  split and never hides all four modalities.

## Full-test matched comparison

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| CERD | 78.44 ± 0.76 | 77.81 ± 0.35 | 85.12 ± 0.05 |
| Flex-MoE | 78.22 ± 0.33 | 76.56 ± 0.67 | **86.23 ± 0.33** |
| I2MoE | 78.33 ± 1.01 | 76.63 ± 1.28 | 86.08 ± 0.25 |
| MoE++-corrected | 77.12 ± 0.69 | 76.34 ± 0.47 | 85.78 ± 0.21 |
| AnyMod | 79.43 ± 1.66 | 78.51 ± 1.26 | 84.07 ± 1.62 |
| AGDiC-inspired | **80.31 ± 1.01** | **79.21 ± 0.69** | 85.38 ± 0.52 |
| ACADiff-inspired | 78.11 ± 2.10 | 76.80 ± 1.94 | 82.59 ± 1.44 |

CERD does not rank first on the full test cohort: AGDiC-inspired is highest in
Accuracy and Macro-F1, and Flex-MoE is highest in Macro-AUROC. This result must
not be described as a universal or significant CERD advantage.

## Originally incomplete test participants (n=45)

The same frozen models are evaluated on the participants whose formal test
inputs contain at least one unavailable modality.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **78.52 ± 4.63** | **76.70 ± 3.52** | **85.70 ± 1.30** |
| Flex-MoE | **78.52 ± 1.28** | 75.65 ± 1.10 | 82.00 ± 0.59 |
| I2MoE | 77.04 ± 3.39 | 73.32 ± 4.58 | 84.52 ± 1.35 |
| MoE++-corrected | 73.33 ± 2.22 | 70.72 ± 1.37 | 79.93 ± 2.17 |
| AnyMod | 77.04 ± 3.39 | 75.18 ± 1.70 | 83.63 ± 0.39 |
| AGDiC-inspired | 75.56 ± 4.62 | 72.42 ± 4.28 | 80.37 ± 2.73 |
| ACADiff-inspired | 76.30 ± 4.63 | 73.10 ± 5.16 | 79.70 ± 1.03 |

CERD ties the highest Accuracy and gives the highest Macro-F1 and AUC in the
incomplete stratum. Because this stratum contains 45 participants, its larger
seed variation must be reported with the mean.

## CERD seed receipt

| Seed | Best epoch | Accuracy | Macro-F1 | Macro-AUROC |
|---:|---:|---:|---:|---:|
| 31 | 11 | 78.88 | 77.99 | 85.17 |
| 32 | 10 | 77.56 | 77.40 | 85.10 |
| 33 | 10 | 78.88 | 78.05 | 85.08 |

The selected configuration uses 16 tokens per modality, hidden width 128,
eight experts with top-2 routing, dropout 0.30, and the common CERD objectives.
It was selected by the highest three-seed mean validation Macro-F1 among eight
predeclared candidates. Binary probability thresholds were chosen from each
seed's validation set and frozen before strict validation replay and one-time
test evaluation.

## Component and modality audits

The full-test Dense FFN control is slightly higher than sparse CERD, so it is
not valid evidence for an overall sparse-MoE gain. On incomplete participants,
however, CERD exceeds Dense FFN by 2.22 Accuracy, 3.66 Macro-F1, and 1.33 AUC
points; it also exceeds E=1 by 4.45 Accuracy and 3.95 Macro-F1 points. Full
component values are in
[`abcd_current_binary_component_ablation_v1.md`](abcd_current_binary_component_ablation_v1.md).

Modality relevance is now defined from the model's own strict-removal decision
changes, normalized across the four modalities within each seed. Its ordering
is compared with the independently normalized positive Accuracy decreases.
The four-point descriptive correspondence is Pearson r=0.979 on ADNI and
r=0.947 on ABCD; ABCD Spearman rho is 1.0. The audit receipt is
[`cerd_three_seed_modality_audit_v2.json`](cerd_three_seed_modality_audit_v2.json).
