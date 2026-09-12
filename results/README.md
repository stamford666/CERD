# Result receipts

Only participant-free aggregate receipts are published. Raw data, identifiers,
per-participant predictions, logs, and checkpoints are excluded.

## Current ABCD three-class task

- [`abcd_current3_n3000_missing15_formal_v1.md`](abcd_current3_n3000_missing15_formal_v1.md): concise formal comparison and protocol.
- [`abcd_current3_n3000_missing15_formal_v1.json`](abcd_current3_n3000_missing15_formal_v1.json): seed-level and aggregate results for CERD and six baselines, including complete and incomplete test strata.
- [`abcd_current3_n3000_missing15_protocol_v1.json`](abcd_current3_n3000_missing15_protocol_v1.json): frozen data-manifest hash, seeds, training schedule, selection rule, and implementation hashes.

The current endpoint has 3,000 participants in low-symptom/no-status,
symptom-positive/no-status, and current-full-ADHD groups. All aggregates are
arithmetic means and sample standard deviations over seeds 31/32/33; no
probability ensemble is used.

## Current component and modality audits

- [`current3_component_and_modality_audit_v2.md`](current3_component_and_modality_audit_v2.md): readable two-dataset component and strict-removal summary.
- [`abcd_current3_component_suite_v1.json`](abcd_current3_component_suite_v1.json): nine ABCD cumulative/one-factor controls, all 100 epochs over seeds 31/32/33.
- [`adni_single_expert_control_v1.json`](adni_single_expert_control_v1.json): ADNI one-expert control over seeds 0/1/2.
- [`cerd_three_seed_modality_audit_v3.json`](cerd_three_seed_modality_audit_v3.json): frozen-checkpoint decision-change, exact decision-evidence, and removal-effect audit.

## Current ADNI task

[`adni_matched_updated_cerd_v4.md`](adni_matched_updated_cerd_v4.md) and its
JSON receipt give the current seven-method CN/MCI/AD comparison. CERD obtains
65.72 ± 1.09 Accuracy, 64.56 ± 2.09 Macro-F1, and 81.07 ± 0.55 Macro-AUROC.

## Archived ABCD endpoints

Earlier binary, clinical-course, and presentation-based receipts remain only
as provenance. They must not be combined with the current three-class task in
tables or analyses.
