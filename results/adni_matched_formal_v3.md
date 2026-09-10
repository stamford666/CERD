# ADNI matched formal campaign: three-seed arithmetic means

All methods use the same frozen ADNI split and seeds 0/1/2. Values are the
arithmetic mean and sample standard deviation of three independently evaluated
models, not a seed-probability ensemble.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| CERD | 64.47 ± 0.83 | **64.26 ± 1.00** | **80.57 ± 1.14** |
| Flex-MoE | 62.26 ± 1.09 | 60.62 ± 2.12 | 77.75 ± 1.03 |
| I2MoE | 64.05 ± 2.27 | 61.83 ± 1.89 | 79.35 ± 1.63 |
| MoE++ | 59.75 ± 0.54 | 58.85 ± 2.06 | 79.23 ± 1.39 |
| AnyMod | **64.78 ± 4.63** | 63.55 ± 4.28 | 79.77 ± 2.12 |
| AGDiC | 58.60 ± 3.36 | 56.54 ± 2.38 | 75.10 ± 1.36 |
| ACADiff | 55.87 ± 0.48 | 52.14 ± 1.20 | 71.10 ± 1.24 |

This is the matched baseline table associated with the earlier frozen formal
campaign. The later direct CERD run in
[`adni_direct_three_seed_mean_v1.md`](adni_direct_three_seed_mean_v1.md)
obtains 65.30/64.48/80.81, but it is listed separately because it is not one of
the checkpoints in this matched campaign.
