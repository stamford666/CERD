# Result receipts

Only participant-free aggregate receipts are published here. Raw data,
identifiers, per-participant predictions, logs, and checkpoints are excluded.

## Current ABCD binary task

- [`abcd_current_binary_matched_v1.md`](abcd_current_binary_matched_v1.md):
  endpoint, cohort, full-test and incomplete-subset baseline comparisons,
  seed-level CERD values, and protocol notes.
- [`abcd_current_binary_cerd_v1.json`](abcd_current_binary_cerd_v1.json):
  validation-only configuration ranking plus formal CERD result.
- [`abcd_current_binary_baselines_v1.json`](abcd_current_binary_baselines_v1.json):
  six matched baselines, including full and availability-stratified metrics.
- [`abcd_current_binary_component_ablation_v1.md`](abcd_current_binary_component_ablation_v1.md):
  full-test and incomplete-subset component controls; the JSON counterpart
  preserves seed-level values.
- [`cerd_three_seed_modality_audit_v2.json`](cerd_three_seed_modality_audit_v2.json):
  two-dataset frozen-checkpoint modality intervention and correspondence audit.

All aggregates are arithmetic means and sample standard deviations over three
independently trained seeds, never probability ensembles.

## Current ADNI task

[`adni_matched_updated_cerd_v4.md`](adni_matched_updated_cerd_v4.md) and its
JSON receipt give the current seven-method CN/MCI/AD comparison. CERD obtains
65.72 ± 1.09 Accuracy, 64.56 ± 2.09 Macro-F1, and 81.07 ± 0.55 Macro-AUROC.

## Archived endpoint receipts

Earlier course and presentation-three receipts remain only for provenance.
They are not reused in the current binary tables; the current files are the
five receipts listed at the top of this page.
