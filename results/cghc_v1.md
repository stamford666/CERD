# CGHC-v1 aggregate results

> **Evidence boundary:** retrospective fixed-candidate paired evidence. This is not a pristine new confirmatory test, and no modality result is a causal or disease-etiology estimate.

CGHC-v1 retains conditional generation and sparse-MoE neural members, while its final cross-model hierarchical median consensus is not an MoE.

## Common-six results

Values are percentages. Every CGHC-v1 value is strictly higher than every listed three-seed baseline ensemble on the same metric and dataset.

| Dataset | Method | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| ADNI | CGHC-v1 | 67.30 | 67.90 | 66.82 | 66.71 | 83.06 | 72.39 |
| ADNI | Flex-MoE | 62.89 | 61.21 | 60.50 | 61.30 | 79.63 | 66.07 |
| ADNI | I2MoE | 66.98 | 65.99 | 64.17 | 63.81 | 81.94 | 69.96 |
| ADNI | MoE++ (corrected) | 63.52 | 63.42 | 62.85 | 63.27 | 80.96 | 67.57 |
| ADNI | AnyMod | 64.15 | 63.09 | 62.43 | 63.33 | 81.97 | 70.83 |
| ADNI | AGDiC | 58.81 | 55.64 | 56.74 | 58.23 | 76.56 | 62.20 |
| ADNI | ACADiff | 56.60 | 54.20 | 52.04 | 53.25 | 71.74 | 56.78 |
| ABCD | CGHC-v1 | 62.01 | 55.50 | 56.16 | 61.20 | 75.68 | 58.99 |
| ABCD | Flex-MoE | 57.77 | 54.26 | 54.53 | 58.78 | 74.69 | 57.69 |
| ABCD | I2MoE | 57.54 | 54.88 | 55.36 | 59.13 | 74.26 | 56.89 |
| ABCD | MoE++ (corrected) | 59.83 | 54.56 | 54.13 | 59.15 | 74.32 | 57.52 |
| ABCD | AnyMod | 59.60 | 54.19 | 54.78 | 59.51 | 73.43 | 56.64 |
| ABCD | AGDiC | 56.79 | 53.55 | 53.73 | 57.76 | 71.81 | 54.94 |
| ABCD | ACADiff | 57.25 | 54.19 | 54.67 | 58.61 | 73.17 | 56.53 |

- [Common-six performance figure](../figures/cghc_common6.svg)

## Paired evidence against Flex-MoE

Holm adjustment is across all six comparators within each dataset and endpoint. Positive lower bounds are one-sided 95% cluster-bootstrap bounds.

| Dataset | Endpoint | Difference (pp) | Raw p | Holm p | Lower bound (pp) | Holm significant and lower positive |
|---|---|---:|---:|---:|---:|:---:|
| ADNI | Macro-F1 | +6.32 | 0.00776 | 0.03104 | +2.12 | yes |
| ADNI | Accuracy | +4.40 | 0.05224 | 0.20896 | +0.31 | no |
| ABCD | Macro-F1 | +1.63 | 0.06336 | 0.25343 | -0.13 | no |
| ABCD | Accuracy | +4.24 | 0.00006 | 0.00018 | +2.67 | yes |

The supported dataset-specific statements are ABCD Accuracy and ADNI Macro-F1. Other rows remain descriptive even when their point difference is positive.

## Conditional-generation component controls

These controls come from a separate exact-500 five-seed, five-fold development-OOF campaign. They describe the conditional-generative component and are not matched ablations of the final heterogeneous consensus.

| Dataset | Full minus no completion: Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| ADNI | +0.20 | +0.23 | +1.34 | +1.08 | -1.40 | -2.79 |
| ABCD | +3.07 | +1.44 | +1.94 | +2.17 | +0.63 | -0.01 |

Completion improves ABCD hard classification and five of six metrics in this matched campaign; its ABCD Macro-AUPRC difference is essentially zero. On ADNI it improves the hard classification metrics, including Macro-F1, but not both ranking metrics. The evidence therefore supports a task-dependent benefit rather than an unconditional all-metric claim.

## Modality–disease association

The public interpretation replays frozen validation checkpoints and reports participant-level member-averaged decision allocation with cluster-bootstrap intervals. ADNI compares dementia with cognitively normal; ABCD compares multiple diagnosis domains with no diagnosis. These are fitted-model associations, not disease causes or causal effects.

- [Modality allocation by disease stratum](../figures/cghc_modality_association.svg)
