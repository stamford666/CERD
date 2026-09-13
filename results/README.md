# Result receipts

Only participant-free aggregate receipts are published. Raw data, identifiers,
per-participant predictions, logs, and checkpoints are excluded.

## Current ABCD three-class task

- [`abcd_severity3_n3000_missing15_formal_v2.md`](abcd_severity3_n3000_missing15_formal_v2.md): concise formal comparison and protocol.
- [`abcd_severity3_n3000_missing15_formal_v2.json`](abcd_severity3_n3000_missing15_formal_v2.json): seed-level and aggregate results for CERD and six baselines, including complete and incomplete test strata.
- [`abcd_severity3_n3000_missing15_protocol_v2.json`](abcd_severity3_n3000_missing15_protocol_v2.json): frozen data-manifest hash, seeds, training schedule, selection rule, and implementation hashes.

The current endpoint has 3,000 participants in disjoint 0--2 symptom,
3--5 symptom, and current-full-ADHD/6--9 symptom strata. It is an ordered
cross-sectional severity task. All aggregates are arithmetic means and sample
standard deviations over seeds 31/32/33; no probability ensemble is used.

## Current component and modality audits

- [`severity3_e4k2_component_and_modality_audit_v2.md`](severity3_e4k2_component_and_modality_audit_v2.md): readable two-dataset component and strict-removal summary.
- [`abcd_severity3_e4k2_component_suite_v2.json`](abcd_severity3_e4k2_component_suite_v2.json): final ABCD one-factor and capacity controls, all 100 epochs over seeds 31/32/33.
- [`adni_single_expert_control_v1.json`](adni_single_expert_control_v1.json): ADNI one-expert control over seeds 0/1/2.
- [`cerd_three_seed_modality_audit_severity3_e4k2_v2.json`](cerd_three_seed_modality_audit_severity3_e4k2_v2.json): frozen-checkpoint decision-change, exact decision-evidence, and removal-effect audit.

## Current ADNI task

[`adni_matched_updated_cerd_v4.md`](adni_matched_updated_cerd_v4.md) and its
JSON receipt give the current seven-method CN/MCI/AD comparison. CERD obtains
65.72 ± 1.09 Accuracy, 64.56 ± 2.09 Macro-F1, and 81.07 ± 0.55 Macro-AUROC.

Earlier exploratory ABCD endpoint receipts are intentionally omitted from this
release so that they cannot be confused with the frozen severity3 task.
