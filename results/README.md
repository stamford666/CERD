# Result receipts

Only participant-free aggregate receipts are published. Raw data, identifiers,
predictions, logs, and checkpoints are excluded.

## Current paper results

- [`abcd_severity3_n3000_missing15_formal_v2.md`](abcd_severity3_n3000_missing15_formal_v2.md):
  current three-class, 3,000-person ABCD comparison and frozen protocol.
- `abcd_severity3_n3000_missing15_formal_v2.json`: seven-method seed-level and
  aggregate ABCD results, including availability strata.
- `abcd_severity3_n3000_missing15_protocol_v2.json`: manifest, code hashes,
  seeds, training schedule, and model-selection rule.
- [`severity3_e4k2_component_and_modality_audit_v2.md`](severity3_e4k2_component_and_modality_audit_v2.md):
  current ADNI/ABCD component controls and strict-removal interpretation.
- `abcd_severity3_e4k2_component_suite_v2.json`: final ABCD one-factor and capacity controls.
- `adni_single_expert_control_v1.json`: ADNI one-expert MoE control.
- `cerd_three_seed_modality_audit_severity3_e4k2_v2.json`: exact decision evidence plus
  frozen-checkpoint prediction-change and accuracy-drop shares.
- [`adni_matched_updated_cerd_v4.md`](adni_matched_updated_cerd_v4.md): current
  ADNI CN/MCI/AD comparison.

Every aggregate is an arithmetic three-seed mean ± sample standard deviation,
not a probability ensemble.

Earlier exploratory endpoint receipts are intentionally omitted so they cannot
be confused with the frozen severity3 task used by the manuscript.
