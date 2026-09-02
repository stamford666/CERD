# Data interfaces

No ABCD or ADNI participant data, identifiers, splits, or derived feature
tables are distributed in this repository. Obtain each dataset through its
official access process and keep it outside version control.

## ABCD binary reference benchmark

This manifest-driven binary workflow is an independent public reference
benchmark. It is not the source of the final method-revision ABCD result, which
uses the frozen three-class presentation endpoint on the dev946 development
cohort with five family-disjoint folds. The controlled dev946 tables, folds, and
campaign runner are not distributed.

Pass a manifest with `--dataset-manifest`. Start from
`abcd_adhd_manifest.example.json`. The manifest points to the binary label table,
immutable train/validation/test split, and eight baseline (`ses-00A`) modality
tables: structural MRI, resting-state fMRI, diffusion MRI, ancestry/population
structure, neurocognition, physical health, mental health, and environment.
Feature filtering, median imputation, and standardization are fitted on training
subjects only.

The independent reference benchmark uses baseline parent K-SADS **full ADHD
present-or-past** as the positive research endpoint. Controls are assessed negative on the
available full, partial-remission, and unspecified ADHD fields. This is an
algorithmic research endpoint, not a clinical diagnosis. Predictor tables must
exclude target-defining K-SADS fields, target-revealing source tables, direct
ADHD proxies, and aggregate scores that could reconstruct those proxies.

Expected split keys are `training`, `validation`, and `testing`. Build these
splits once, stratify by the binary target, and keep genetic-family members in
exactly one split. Sites may occur across partitions, so this is not a
site-held-out evaluation. The loader verifies participant-level split
disjointness; family/site checks belong in the protected data-preparation audit
because those identifiers are not distributed here.

The label table uses `target = 1` for full ADHD present-or-past and `target = 0`
for assessed available-field-negative controls. Do not place participant identifiers,
protected splits, derived tables, predictions, or checkpoints under version
control.

## ADNI

Pass the prepared directory with `--adni-data-root`. The loader expects:

```text
adni/
├── label.csv
├── PTID_splits.json
├── image/UCSFFSX7_06Jan2026.csv
├── genomic/genomic_merged.h5ad
├── clinical/clinical_merged.csv
└── biospecimen/biospecimen_merged.csv
```

`label.csv` is indexed by `PTID` and contains the three-level `DIAGNOSIS`
code. Confirm the semantic class names against the ADNI data dictionary before
publication.
