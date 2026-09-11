# ABCD current-ADHD binary matched component controls

arithmetic mean and sample standard deviation across seeds 31/32/33; no probability ensemble. the full configuration was selected on validation only; every control changes only its named component, selects its checkpoint by the same validation rule, and is evaluated once on test after strict replay.

| Configuration | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| Full CERD | 78.44 ± 0.76 | 77.81 ± 0.35 | 85.12 ± 0.05 |
| Dense FFN | 79.10 ± 0.83 | 78.00 ± 0.82 | 85.74 ± 0.34 |
| Single expert (E=1) | 78.33 ± 1.49 | 77.86 ± 1.33 | 86.28 ± 0.39 |
| w/o multigranular decomposition | 78.99 ± 1.25 | 78.35 ± 0.90 | 85.69 ± 0.16 |
| w/o conditional completion | 77.34 ± 2.34 | 76.54 ± 2.22 | 84.92 ± 0.47 |
| w/o provenance encoding | 79.65 ± 1.06 | 78.81 ± 1.33 | 85.16 ± 0.12 |
| Uniform branch weights | 78.22 ± 1.75 | 77.46 ± 1.14 | 85.46 ± 0.43 |

## Originally incomplete test participants (n=45)

| Configuration | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| Full CERD | 78.52 ± 4.63 | 76.70 ± 3.52 | 85.70 ± 1.30 |
| Dense FFN | 76.30 ± 3.39 | 73.04 ± 3.66 | 84.37 ± 1.22 |
| Single expert (E=1) | 74.07 ± 2.57 | 72.75 ± 1.62 | 86.15 ± 1.67 |
| w/o multigranular decomposition | 74.81 ± 5.59 | 72.79 ± 3.92 | 84.74 ± 1.48 |
| w/o conditional completion | 74.07 ± 6.42 | 71.52 ± 4.38 | 82.67 ± 2.78 |
| w/o provenance encoding | 79.26 ± 1.28 | 76.55 ± 2.25 | 85.48 ± 0.93 |
| Uniform branch weights | 74.07 ± 8.41 | 71.50 ± 6.46 | 85.70 ± 0.34 |
