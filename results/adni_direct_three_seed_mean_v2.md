# ADNI direct CERD result: validation-selected three-seed mean

Each seed is independently trained, selected by validation Macro-F1, replayed
on validation, and evaluated once on the held-out test set. The aggregate is
the arithmetic mean and sample standard deviation of the three seed-level
metrics; model probabilities are never averaged.

| Seed | Best epoch | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---:|---:|---:|---:|---:|
| 0 | 30 | 66.35 | 66.86 | 80.97 |
| 1 | 15 | 66.35 | 64.04 | 81.65 |
| 2 | 23 | 64.47 | 62.78 | 80.57 |
| Mean ± SD | — | **65.72 ± 1.09** | **64.56 ± 2.09** | **81.07 ± 0.55** |

The configuration was selected from six candidates using only the fixed
validation split. The primary rule was three-seed mean Macro-F1, with mean
Accuracy constrained to remain within 0.5 percentage point of the validation
anchor. The selected setting retains the complete CERD architecture and uses
16 patches per modality, dropout 0.30, and learning rate 1.25e-4. Relative to
the preceding direct three-seed result, it changes only ordinary optimization
settings and improves held-out Accuracy, Macro-F1, and Macro-AUROC by
0.42, 0.08, and 0.26 percentage points, respectively. These are descriptive
gains; no significance claim is made from three seeds.
