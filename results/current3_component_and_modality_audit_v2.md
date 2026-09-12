# Current component controls and modality audit

This note records the experiments added after the matched ABCD main comparison.
All values are arithmetic mean ± sample standard deviation over three independently
trained seeds. Test probabilities are not ensembled.

## ABCD component controls

The current full CERD result is 67.37 ± 1.07 Accuracy, 54.38 ± 0.81 Macro-F1,
and 74.75 ± 0.92 Macro-AUROC. Every control below uses the same 3,000-person
family-disjoint split, fixed 15% participant-level mask, training-only transforms,
100 complete epochs, validation-Macro-F1 checkpoint selection, and one frozen test
evaluation.

| Configuration | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| w/o conditional completion | 66.74 ± 3.46 | **54.46 ± 1.29** | 73.64 ± 0.72 |
| w/o provenance embeddings | 65.97 ± 1.68 | 54.19 ± 0.23 | 74.49 ± 0.68 |
| w/o multigranular decomposition | 61.07 ± 2.84 | 50.14 ± 3.50 | 71.41 ± 3.83 |
| w/o reliability-aware weights | 66.20 ± 3.85 | 53.94 ± 1.40 | 73.66 ± 0.77 |
| Single expert (E = 1) | 63.71 ± 0.71 | 52.14 ± 0.61 | 73.73 ± 0.69 |
| Full CERD | **67.37 ± 1.07** | 54.38 ± 0.81 | **74.75 ± 0.92** |

The strongest component signal is multigranular decomposition: the full model gains
6.30 Accuracy, 4.24 Macro-F1, and 3.34 AUROC points over its global-head-only
counterpart. The E = 1 control loses 3.66 Accuracy and 2.24 Macro-F1 points, directly
supporting the value of expert diversity. Conditional completion and provenance have
smaller aggregate effects; their benefit is therefore not overstated.

The aligned Dense-FFN capacity control obtains 66.74/55.00/75.26 overall. On the 64
incomplete test participants, CERD obtains 66.67/46.33/72.45 and Dense obtains
64.06/44.16/71.80. Thus sparse routing gives +2.61 Accuracy, +2.17 Macro-F1, and
+0.65 AUROC specifically under heterogeneous missing inputs, although Dense has a
small overall Macro-F1/AUROC advantage.

The full nine-arm cumulative and one-factor receipt is
[`abcd_current3_component_suite_v1.json`](abcd_current3_component_suite_v1.json).

## ADNI single-expert control

Replacing the 16-expert top-4 pool by one expert gives 64.05 ± 0.65 Accuracy,
63.70 ± 0.59 Macro-F1, and 81.13 ± 1.02 AUROC. Full CERD improves Accuracy and
Macro-F1 to 65.72 ± 1.09 and 64.56 ± 2.09; the single expert has a 0.06-point higher
mean AUROC. The participant-free seed receipt is
[`adni_single_expert_control_v1.json`](adni_single_expert_control_v1.json).

## Frozen-checkpoint modality audit

Each modality is removed from originally complete test participants while participant
identity and the trained checkpoint remain fixed. Prediction-change rates and positive
Accuracy decreases are normalized within seed, so each four-modality vector sums to
100%. The analysis replays all six reference checkpoints with zero label mismatches;
the maximum probability error is below 3e-8.

| Dataset | Modality | Prediction-change share (%) | Accuracy-drop share (%) | Accuracy decrease (points) |
|---|---|---:|---:|---:|
| ADNI | MRI | 35.44 ± 3.79 | 44.03 ± 3.10 | 43.07 ± 1.47 |
| ADNI | Genetics | 17.55 ± 3.82 | 9.19 ± 10.52 | 9.28 ± 12.36 |
| ADNI | Clinical | 10.44 ± 2.25 | 6.61 ± 3.80 | 6.70 ± 4.34 |
| ADNI | Biospecimen | 36.57 ± 1.90 | 40.17 ± 9.75 | 38.81 ± 5.94 |
| ABCD | Imaging | 15.02 ± 5.02 | 12.06 ± 10.57 | 5.02 ± 3.58 |
| ABCD | Genetics | 22.07 ± 10.83 | 28.10 ± 25.47 | 15.34 ± 17.73 |
| ABCD | Cognition/health | 30.91 ± 4.33 | 27.63 ± 9.26 | 11.78 ± 0.99 |
| ABCD | Behavior/environment | 32.01 ± 3.73 | 32.20 ± 10.40 | 13.79 ± 1.14 |

Prediction-change and Accuracy-drop shares have Pearson correlation 0.979 on ADNI
and 0.874 on ABCD; Spearman rho is 0.8 on both. The published interpretation uses
this intervention-based correspondence. The receipt also preserves the exact internal
decision-evidence decomposition as a separate quantity, without treating that quantity
as a removal-effect estimate.

The complete audit is
[`cerd_three_seed_modality_audit_v3.json`](cerd_three_seed_modality_audit_v3.json).
