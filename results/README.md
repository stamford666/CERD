# Result receipts

Only participant-free aggregate receipts are published. Raw data, identifiers,
predictions, logs, and checkpoints are excluded.

## Current paper results

- [`abcd_current3_n3000_missing15_formal_v1.md`](abcd_current3_n3000_missing15_formal_v1.md):
  current three-class, 3,000-person ABCD comparison and frozen protocol.
- `abcd_current3_n3000_missing15_formal_v1.json`: seven-method seed-level and
  aggregate ABCD results, including availability strata.
- `abcd_current3_n3000_missing15_protocol_v1.json`: manifest, code hashes,
  seeds, training schedule, and model-selection rule.
- [`current3_component_and_modality_audit_v2.md`](current3_component_and_modality_audit_v2.md):
  current ADNI/ABCD component controls and strict-removal interpretation.
- `abcd_current3_component_suite_v1.json`: nine ABCD component controls.
- `adni_single_expert_control_v1.json`: ADNI one-expert MoE control.
- `cerd_three_seed_modality_audit_v3.json`: exact decision evidence plus
  frozen-checkpoint prediction-change and accuracy-drop shares.
- [`adni_matched_updated_cerd_v4.md`](adni_matched_updated_cerd_v4.md): current
  ADNI CN/MCI/AD comparison.

Every aggregate is an arithmetic three-seed mean ± sample standard deviation,
not a probability ensemble.

Earlier endpoint receipts remain as archival provenance only and are not used
by the current manuscript tables.
