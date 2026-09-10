# ADNI matched comparison with validation-selected CERD

All rows use the same frozen ADNI train/validation/test split, test cohort of
318 participants, seeds 0/1/2, raw per-seed argmax predictions, and arithmetic
three-seed reporting. The baseline rows are unchanged from the frozen v3
campaign. CERD is updated using a six-candidate validation-only screen; test
labels and test probabilities were unavailable to configuration selection.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **65.72 ± 1.09** | **64.56 ± 2.09** | **81.07 ± 0.55** |
| Flex-MoE | 62.26 ± 1.09 | 60.62 ± 2.12 | 77.75 ± 1.03 |
| I2MoE | 64.05 ± 2.27 | 61.83 ± 1.89 | 79.35 ± 1.63 |
| MoE++ | 59.75 ± 0.54 | 58.85 ± 2.06 | 79.23 ± 1.39 |
| AnyMod | 64.78 ± 4.63 | 63.55 ± 4.28 | 79.77 ± 2.12 |
| AGDiC | 58.60 ± 3.36 | 56.54 ± 2.38 | 75.10 ± 1.36 |
| ACADiff | 55.87 ± 0.48 | 52.14 ± 1.20 | 71.10 ± 1.24 |

CERD has the highest three-seed mean on all three reported metrics. Compared
with the strongest baseline for each metric, its descriptive margins are
0.94 percentage point in Accuracy, 1.01 in Macro-F1, and 1.30 in
Macro-AUROC. The direct CERD seed-level receipt and validation-selection scope
are reported in
[`adni_direct_three_seed_mean_v2.md`](adni_direct_three_seed_mean_v2.md).
