# Current two-dataset ablation and modality audit

This report matches the current CERD main results: ADNI seeds 0/1/2 and ABCD
clinical-course seeds 31/32/33. Every value is the arithmetic mean and sample
standard deviation of independently trained seed-level metrics. No probability
ensemble is used.

## Component ablation

Each variant is independently trained, selected by validation Macro-F1, and
formally evaluated only after replaying its saved validation checkpoint. The
`w/o Multigranular Decomposition` control uses only the global head and removes
the specialist auxiliary, distillation, reduced-view CE, and ranking terms; it
therefore removes the decision-decomposition stage rather than merely changing
its inference weight.

| Configuration | ADNI Accuracy | ADNI Macro-F1 | ADNI Macro-AUROC | ABCD Accuracy | ABCD Macro-F1 | ABCD Macro-AUROC |
|---|---:|---:|---:|---:|---:|---:|
| w/o Conditional Completion | 63.73 ± 0.79 | 62.82 ± 1.58 | 79.48 ± 0.41 | 60.64 ± 1.97 | 51.52 ± 2.19 | 74.23 ± 1.22 |
| w/o Provenance Embeddings | 63.52 ± 1.57 | 63.41 ± 1.95 | 80.50 ± 0.70 | 60.90 ± 1.53 | 50.67 ± 1.13 | 74.75 ± 0.50 |
| w/o Sparse MoE (Dense FFN) | 64.99 ± 1.19 | 65.27 ± 1.12 | 81.59 ± 0.74 | 60.38 ± 1.15 | 52.70 ± 0.95 | 75.36 ± 0.51 |
| w/o Multigranular Decomposition | 62.47 ± 0.79 | 62.36 ± 1.00 | 79.72 ± 1.32 | 59.99 ± 1.45 | 51.21 ± 1.85 | 74.04 ± 0.48 |
| w/o Reliability-aware Weights | 64.05 ± 1.49 | 63.88 ± 0.21 | 80.95 ± 0.19 | 61.95 ± 1.50 | 52.46 ± 1.48 | 74.99 ± 0.76 |
| **Full CERD** | **65.72 ± 1.09** | **64.56 ± 2.09** | **81.07 ± 0.55** | **61.43 ± 0.30** | **51.14 ± 0.72** | **74.88 ± 0.34** |

Conditional completion, provenance, and multigranular decomposition improve
Accuracy and Macro-AUROC on both datasets. Sparse MoE improves Accuracy over
the aligned dense FFN by 0.73 points on ADNI and 1.05 points on ABCD, while the
dense control is higher on several macro-averaged cells. Reliability-aware
weights improve all ADNI means; the uniform ABCD control has slightly higher
means but substantially larger Accuracy variation. No reduced control dominates
Full CERD across both endpoints.

## Modality allocation and strict removal

The audit replays the three frozen full-model checkpoints per dataset. On every
originally complete test participant, one input block is set to zero, marked
unavailable, and excluded from conditional completion. The table reports the
resulting metric decrease from the same checkpoint. `Allocation` is the CERD
forward-pass modality decision allocation averaged over the full test cohort;
the four shares sum to 100% within each seed.

| Dataset | Modality | Allocation (%) | ΔAccuracy | ΔMacro-F1 | ΔMacro-AUROC |
|---|---|---:|---:|---:|---:|
| ADNI | MRI | 23.67 ± 1.77 | 43.07 ± 1.47 | 52.46 ± 1.44 | 21.51 ± 5.62 |
| ADNI | Genetics | 21.81 ± 1.68 | 9.28 ± 12.36 | 18.14 ± 21.49 | 2.49 ± 3.19 |
| ADNI | Clinical | 28.16 ± 0.52 | 6.70 ± 4.34 | 9.61 ± 7.26 | 1.98 ± 1.72 |
| ADNI | Biospecimen | 26.36 ± 0.44 | 38.81 ± 5.94 | 49.95 ± 4.35 | 17.75 ± 13.65 |
| ABCD | Imaging | 25.23 ± 1.13 | 23.33 ± 20.00 | 25.30 ± 18.58 | 6.55 ± 6.49 |
| ABCD | Genetics | 23.97 ± 0.65 | 25.64 ± 15.40 | 30.69 ± 8.52 | 4.78 ± 4.87 |
| ABCD | Cognition/health | 26.35 ± 0.91 | 6.00 ± 5.60 | 16.69 ± 12.15 | −0.40 ± 0.54 |
| ABCD | Behavior/environment | 24.45 ± 1.08 | 10.70 ± 0.35 | 27.97 ± 1.26 | 19.07 ± 1.29 |

The participant-free machine-readable receipt is
[`cerd_three_seed_modality_audit_v1.json`](cerd_three_seed_modality_audit_v1.json).
It includes checkpoint and reference hashes, replay errors, seed-level metrics,
and aggregate values. The maximum probability replay error is below 3e-8.
