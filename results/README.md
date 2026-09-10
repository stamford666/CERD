# Result receipts

Only participant-free aggregate receipts are published here. Raw
data, participant identifiers, per-participant predictions, logs, and model
checkpoints are excluded.

## Current two-dataset analyses

[`current_two_dataset_ablation_and_modality_audit_v1.md`](current_two_dataset_ablation_and_modality_audit_v1.md)
reports matched component ablations and strict leave-one-modality-out results
for both ADNI and the current ABCD presentation task. The accompanying
[`cerd_three_seed_modality_audit_v1.json`](cerd_three_seed_modality_audit_v1.json)
contains the participant-free seed-level replay audit and normalized modality
allocation values. The ABCD component seed records are stored separately in
[`abcd_presentation3_component_ablation_v1.json`](abcd_presentation3_component_ablation_v1.json).

## Archived ABCD clinical-course endpoint

[`abcd_adhd_course3_snp_missing15_v1.md`](abcd_adhd_course3_snp_missing15_v1.md)
and its [JSON receipt](abcd_adhd_course3_snp_missing15_v1.json) report the
alternative low-symptom / past-or-remitted / current-ADHD endpoint.
CERD obtains 61.43 ± 0.30 Accuracy, 51.14 ± 0.72 Macro-F1, and
74.88 ± 0.34 Macro-AUROC. It has the highest numerical mean Accuracy and AUC
in that matched table, while AnyMod has the highest Macro-F1. This artifact is
retained for provenance and is not the current ABCD main task.

## ABCD v4 matched main comparison

[`abcd_adhd_presentation3_snp_missing15_v4.md`](abcd_adhd_presentation3_snp_missing15_v4.md)
and its [JSON receipt](abcd_adhd_presentation3_snp_missing15_v4.json) contain the
main comparison. CERD and all six baselines use the same features, participants,
family-disjoint split, fixed 15% missingness table, seed set, and evaluation
rule. CERD obtains 59.24 ± 1.65 Accuracy, 53.62 ± 1.40 Macro-F1, and
74.43 ± 1.03 Macro-AUROC, the highest three-seed mean in that matched table.
The structural variables are T1 regional signal intensities, not regional
volumes; this naming correction applies equally to every method and does not
change comparison fairness.

## ABCD v5

[`abcd_adhd_presentation3_feature_refinement_v5.md`](abcd_adhd_presentation3_feature_refinement_v5.md)
and its [JSON receipt](abcd_adhd_presentation3_feature_refinement_v5.json)
record the corrected structural-imaging representation, one-PC-per-candidate-
gene genetics, added task-performance variables, seed-level metrics, and both
evaluation protocols.

- validation-selected seed checkpoints: 57.71 ± 1.56 Accuracy,
  52.66 ± 1.25 Macro-F1, 73.63 ± 0.87 Macro-AUROC;
- fixed 14-epoch development-pool refit: 58.76 ± 2.20 Accuracy,
  55.49 ± 1.29 Macro-F1, 75.38 ± 0.76 Macro-AUROC.

V5 is a separate feature-representation study using true structural volumes
and thickness, gene-wise PCA, and additional task-performance variables.
Baselines have not yet been rerun on v5, so the v4 baseline rows are not reused
as if they were matched v5 comparisons.

## ADNI

[`adni_matched_updated_cerd_v4.md`](adni_matched_updated_cerd_v4.md) and its
[JSON receipt](adni_matched_updated_cerd_v4.json) give the current seven-method
comparison. Every row uses the same frozen split, test cohort, seeds 0/1/2,
raw per-seed argmax predictions, and arithmetic three-seed aggregation. CERD
obtains 65.72 ± 1.09 Accuracy, 64.56 ± 2.09 Macro-F1, and 81.07 ± 0.55
Macro-AUROC, the highest mean in all three columns.

The CERD configuration was selected from six candidates on validation data
only. Its seed-level validation and held-out results are in
[`adni_direct_three_seed_mean_v2.md`](adni_direct_three_seed_mean_v2.md) and
the corresponding [JSON receipt](adni_direct_three_seed_mean_v2.json).

Every aggregate is the arithmetic mean of seed-level metrics. Probability
ensembling and post-hoc test calibration are not part of the reporting rule.

The earlier frozen campaign remains archived in
[`adni_matched_formal_v3.md`](adni_matched_formal_v3.md); the updated report
explicitly identifies its unchanged baseline source rather than silently
rewriting that historical receipt. The preceding direct CERD result is likewise
retained as `adni_direct_three_seed_mean_v1` for provenance.
