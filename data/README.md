# Data interfaces

No ABCD or ADNI participant data, identifiers, splits, or derived feature
tables are distributed in this repository. Obtain each dataset through its
official access process and keep it outside version control.

## ABCD

Pass a manifest with `--dataset-manifest`. Start from
`abcd_manifest.example.json`. The manifest points to the binary label table,
immutable train/validation/test split, and eight baseline (`ses-00A`) modality
tables: structural MRI, resting-state fMRI, diffusion MRI, ancestry/population
structure, neurocognition, physical health, mental health, and environment.
Feature filtering, median imputation, and standardization are fitted on training
subjects only.

The endpoint is a cumulative symptom-derived **probable BED** phenotype observed
at any administered visit from baseline through year 2. It is neither a
clinical diagnosis nor an incident-onset endpoint. Eligible participants must
have an administered outcome module at all three visits. Controls are assessed
BED-negative; other eating or psychiatric disorders are not excluded. The
predictor tables must exclude K-SADS eating-disorder fields, direct
eating-disorder instruments/proxies, and their composite scores.

Expected split keys are `training`, `validation`, and `testing`. Build these
splits once, stratify by the binary target, and keep genetic-family members in
exactly one split. Sites may occur across partitions, so this is not a
site-held-out evaluation. The loader verifies participant-level split
disjointness; family/site checks belong in the protected data-preparation audit
because those identifiers are not distributed here.

The released label table uses `target = 1` for probable BED and `target = 0`
for assessed BED-negative controls. Do not place participant identifiers,
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
