# ABCD current-status three-class task: formal rerun

This directory contains a fresh formal rerun on the frozen 3,000-participant
ABCD cohort with 15% label-independent random missingness. All methods use the
same family-disjoint train/validation/test split, missingness mask, and
class-weight rule (`class_weight_power = 0.75`). Each model is trained for all
100 epochs. The checkpoint with the best validation Macro-F1 is selected, and
the test set is evaluated once. Reported values are the arithmetic mean and
sample standard deviation over independent seeds 31, 32, and 33; no ensemble
is used.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| CERD | **67.37 ± 1.07** | **54.38 ± 0.81** | 74.75 ± 0.92 |
| Flex-MoE | 63.87 ± 3.03 | 51.27 ± 3.15 | 74.09 ± 2.93 |
| I2MoE | 63.25 ± 3.10 | 53.90 ± 1.35 | 74.54 ± 1.01 |
| MoE++ | 64.80 ± 0.62 | 53.58 ± 1.08 | 75.11 ± 0.76 |
| AnyModal | 61.69 ± 4.00 | 53.57 ± 2.25 | 74.95 ± 1.29 |
| AGDiC | 63.79 ± 0.75 | 52.52 ± 1.72 | **75.16 ± 0.53** |
| ACADiff | 63.17 ± 2.69 | 51.68 ± 2.95 | 73.05 ± 1.03 |

CERD has the highest mean Accuracy and Macro-F1. Relative to the strongest
baseline for each metric, its Accuracy margin is 2.57 percentage points over
MoE++, while its Macro-F1 margin is 0.48 points over I2MoE. AGDiC has the
highest AUROC, exceeding CERD by 0.41 points. The small Macro-F1 margin is
primarily associated with the difficult middle class (symptom-positive with
no recorded ADHD diagnosis), so these results do not support claiming a broad
or statistically established advantage on every metric.

Machine-readable aggregate results are in `results.json`; the frozen protocol
and completion receipt are in `run_complete.json`. Per-seed validation-only
records and one-time formal test evaluations are retained under `validation/`
and `formal/`, respectively.
