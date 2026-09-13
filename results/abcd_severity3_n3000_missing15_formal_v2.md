# ABCD current ADHD severity task: selected sparse model

This receipt combines the six matched baselines from the complete formal rerun
with CERD's validation-selected sparse configuration. Two reduced routing
configurations were trained for all 100 epochs without test evaluation. The
configuration with the higher arithmetic three-seed validation Macro-F1 was
frozen (`4 experts, top-2`) and each selected checkpoint was then evaluated on
the test set once. Every row below uses the same family-disjoint split, fixed
approximately 15% missingness mask, class-weight rule, and direct three-seed
mean; probabilities are not ensembled.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **66.82 ± 2.14** | **55.18 ± 1.16** | **77.90 ± 0.55** |
| Flex-MoE | 63.79 ± 2.60 | 51.44 ± 2.28 | 74.69 ± 1.30 |
| I2MoE | 63.24 ± 0.82 | 49.46 ± 1.27 | 74.60 ± 0.92 |
| MoE++ | 64.64 ± 2.22 | 52.06 ± 4.35 | 74.44 ± 1.81 |
| AnyMod | 62.15 ± 2.00 | 49.09 ± 0.61 | 69.17 ± 4.68 |
| AGDiC | 62.69 ± 3.04 | 48.83 ± 0.84 | 71.71 ± 2.66 |
| ACADiff | 63.16 ± 1.18 | 49.05 ± 1.56 | 70.90 ± 0.78 |

CERD leads the strongest baseline by 2.18 Accuracy, 3.12 Macro-F1, and 3.20
Macro-AUROC points. Its seed-level test metrics are 66.36/56.20/78.34,
69.16/55.41/78.06, and 64.95/53.92/77.28 for seeds 31/32/33.

On complete inputs (n=364), CERD obtains 67.22 ± 2.22 Accuracy,
54.83 ± 2.11 Macro-F1, and 77.51 ± 1.33 Macro-AUROC. On incomplete inputs
(n=64), it obtains 64.58 ± 4.77, 55.06 ± 4.59, and 80.13 ± 3.81. These are
descriptive strata containing different participants; causal source effects
are evaluated by the strict-removal audit.

