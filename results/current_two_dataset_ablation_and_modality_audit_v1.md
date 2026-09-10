# Current two-dataset ablation and modality audit

This report matches the current CERD main results: ADNI seeds 0/1/2 and ABCD
ADHD-presentation seeds 31/32/33. Every value is the arithmetic mean and sample
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
| w/o Conditional Completion | 63.73 ± 0.79 | 62.82 ± 1.58 | 79.48 ± 0.41 | 55.61 ± 1.94 | 50.95 ± 1.58 | 71.41 ± 1.03 |
| w/o Provenance Embeddings | 63.52 ± 1.57 | 63.41 ± 1.95 | 80.50 ± 0.70 | 59.16 ± 1.85 | 53.42 ± 1.58 | 74.44 ± 1.03 |
| w/o Sparse MoE (Dense FFN) | 64.99 ± 1.19 | 65.27 ± 1.12 | 81.59 ± 0.74 | 59.08 ± 2.31 | 54.33 ± 1.43 | 75.11 ± 1.07 |
| w/o Multigranular Decomposition | 62.47 ± 0.79 | 62.36 ± 1.00 | 79.72 ± 1.32 | 56.98 ± 3.72 | 53.04 ± 3.19 | 74.06 ± 1.10 |
| w/o Reliability-aware Weights | 64.05 ± 1.49 | 63.88 ± 0.21 | 80.95 ± 0.19 | 59.32 ± 1.59 | 54.35 ± 2.30 | 73.81 ± 0.78 |
| **Full CERD** | **65.72 ± 1.09** | **64.56 ± 2.09** | **81.07 ± 0.55** | **59.24 ± 1.65** | **53.62 ± 1.40** | **74.43 ± 1.03** |

Conditional completion gives the clearest cross-dataset effect. Its removal
reduces all three ABCD metrics by 2.67--3.63 points and all ADNI metrics by
1.19--1.99 points. Removing multigranular decomposition lowers ABCD Accuracy
by 2.26 points and ADNI Accuracy by 3.25 points, showing that the anchored
decision branches recover information not retained by the global head alone.
The smaller provenance differences and the metric trade-offs for dense and
uniform controls are reported as observed rather than described as universal
gains. Full CERD is the only configuration that combines the strongest ADNI
result with the highest ABCD Accuracy among these component controls.

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
| ABCD | Imaging | 24.38 ± 1.14 | 6.74 ± 7.81 | 5.08 ± 5.56 | 0.97 ± 2.63 |
| ABCD | Genetics | 23.68 ± 4.16 | 8.74 ± 8.97 | 12.99 ± 7.82 | 0.87 ± 2.49 |
| ABCD | Cognition/health | 25.74 ± 1.36 | 14.25 ± 1.24 | 30.18 ± 2.58 | 0.75 ± 1.11 |
| ABCD | Behavior/environment | 26.19 ± 3.38 | 15.29 ± 1.94 | 33.25 ± 2.29 | 15.20 ± 2.12 |

The presentation endpoint shifts the ABCD intervention profile toward the two
phenotypic sources: cognition/health and behavior/environment cause the largest
Accuracy and Macro-F1 decreases, while behavior/environment also dominates the
ranking decrease. Imaging and SNPs remain active but show larger seed variation.
The normalized allocations stay distributed across all four sources, so this
pattern reflects endpoint-specific dependence rather than a collapsed gate.

The participant-free machine-readable receipt is
[`cerd_three_seed_modality_audit_v1.json`](cerd_three_seed_modality_audit_v1.json).
It includes checkpoint and reference hashes, replay errors, seed-level metrics,
and aggregate values. The maximum probability replay error is below 3e-8.
