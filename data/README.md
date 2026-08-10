# Data interfaces

No ABCD or ADNI participant data, identifiers, splits, or derived feature
tables are distributed in this repository. Obtain each dataset through its
official access process and keep it outside version control.

## ABCD

Pass a manifest with `--dataset-manifest`. Start from
`abcd_manifest.example.json`. The manifest points to the label table, immutable
train/validation/test split, and four modality tables. Feature filtering,
median imputation, and standardization are fitted on training subjects only.

Expected split keys are `training`, `validation`, and `testing`.

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
