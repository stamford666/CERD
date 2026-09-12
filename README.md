# CERD

CERD is an incomplete-multimodal classifier with subject-conditioned token
completion, observed/generated provenance, sparse mixture-of-experts fusion,
and multigranular reliability-aware decision fusion. This branch contains the
executable code and participant-free aggregate result receipts; manuscript
sources are maintained on `stamford666-paper`.

## Current matched results

Every entry is the arithmetic mean ± sample standard deviation over three
independently trained models. No probability ensemble is used.

### ABCD: current-status ADHD three-class endpoint

The frozen cohort contains 3,000 participants and uses three clinically
explicit baseline parent K-SADS groups: low-symptom participants without a
recorded ADHD status; symptom-positive participants without a recorded ADHD
status; and current full ADHD. The family-disjoint train/validation/test split
contains 2,141/430/429 participants. A fixed, label-independent mask makes 15%
of every split incomplete.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **67.37 ± 1.07** | **54.38 ± 0.81** | 74.75 ± 0.92 |
| Flex-MoE | 63.87 ± 3.03 | 51.27 ± 3.15 | 74.09 ± 2.93 |
| I2MoE | 63.25 ± 3.10 | 53.90 ± 1.35 | 74.54 ± 1.01 |
| MoE++ | 64.80 ± 0.62 | 53.58 ± 1.08 | 75.11 ± 0.76 |
| AnyMod | 61.69 ± 4.00 | 53.57 ± 2.25 | 74.95 ± 1.29 |
| AGDiC | 63.79 ± 0.75 | 52.52 ± 1.72 | **75.16 ± 0.53** |
| ACADiff | 63.17 ± 2.69 | 51.68 ± 2.95 | 73.05 ± 1.03 |

CERD has the highest mean Accuracy and Macro-F1. Its Accuracy exceeds the
strongest baseline by 2.57 points; the Macro-F1 margin is 0.48 points. AGDiC
has the highest AUROC by 0.41 points. The complete result, per-seed values,
availability-stratified metrics, and frozen protocol are in
[`results/abcd_current3_n3000_missing15_formal_v1.md`](results/abcd_current3_n3000_missing15_formal_v1.md).

### ADNI: CN/MCI/AD endpoint

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **65.72 ± 1.09** | **64.56 ± 2.09** | **81.07 ± 0.55** |
| Flex-MoE | 62.26 ± 1.09 | 60.62 ± 2.12 | 77.75 ± 1.03 |
| I2MoE | 64.05 ± 2.27 | 61.83 ± 1.89 | 79.35 ± 1.63 |
| MoE++ | 59.75 ± 0.54 | 58.85 ± 2.06 | 79.23 ± 1.39 |
| AnyMod | 64.78 ± 4.63 | 63.55 ± 4.28 | 79.77 ± 2.12 |
| AGDiC | 58.60 ± 3.36 | 56.54 ± 2.38 | 75.10 ± 1.36 |
| ACADiff | 55.87 ± 0.48 | 52.14 ± 1.20 | 71.10 ± 1.24 |

The frozen ADNI receipt is
[`results/adni_matched_updated_cerd_v4.md`](results/adni_matched_updated_cerd_v4.md).

## Component and source-dependence checks

The current three-seed controls use the same evaluation rules as the main tables.
On ABCD, removing multigranular decomposition reduces Accuracy from 67.37 to
61.07 and Macro-F1 from 54.38 to 50.14. Replacing the sparse pool by one expert
gives 63.71 Accuracy and 52.14 Macro-F1. On ADNI, the corresponding single-expert
control gives 64.05 Accuracy and 63.70 Macro-F1, compared with 65.72 and 64.56
for full CERD.

Frozen-checkpoint strict-removal audits normalize both prediction-change rates and
positive Accuracy decreases across the four modalities. Their modality-level Pearson
correlations are 0.979 on ADNI and 0.874 on ABCD (Spearman rho = 0.8 on both).
All reference predictions are reproduced with zero label mismatches before the
interventions are evaluated.

The complete participant-free report is
[`results/current3_component_and_modality_audit_v2.md`](results/current3_component_and_modality_audit_v2.md).

## ABCD labels and modalities

| Class | Definition | Participants |
|---|---|---:|
| 0 | No current, past, partial-remission, or unspecified ADHD status; at most three current symptoms in each nine-item domain | 1,677 |
| 1 | No recorded ADHD status; at least four current symptoms in either nine-item domain | 460 |
| 2 | Current full ADHD; at least six current symptoms in either nine-item domain | 863 |

Class 1 is symptom-positive without a recorded diagnosis, not mild diagnosed
ADHD. All target-defining ADHD fields and CBCL ADHD/attention scores are
excluded from predictors.

The four model modalities are:

- **I — Imaging (785):** rs-fMRI, SST/n-back/MID task-fMRI, regional T1 gray
  matter volume, and DTI FA;
- **G — Genetics (1,184):** direct 0/1/2 SNP dosages after training-only
  missingness/MAF QC and LD pruning in prespecified gene windows; no PCA, PRS,
  or ancestry components;
- **C — Cognition/health (90):** cognitive tasks, sleep, and physical activity;
- **B — Behavior/environment (221 raw; 219 retained):** non-ADHD behavior,
  temperament, family/neighborhood context, socioeconomic measures, and
  residential/prenatal exposures.

The Chinese label-selection and leakage audit is
[`docs/ABCD_CURRENT3_LABEL_SELECTION_N3000_V1_ZH.md`](docs/ABCD_CURRENT3_LABEL_SELECTION_N3000_V1_ZH.md).

## Missingness, preprocessing, and evaluation

ABCD uses a fixed seed-2026 label-independent mask. It hides one, two, or three
modalities, never all four, and is applied before preprocessing. Feature
filtering, median imputation, z-score scaling, SNP missingness/MAF filtering,
and LD pruning use training participants only. ADNI retains its recorded
source-table availability.

For ABCD, every method trains for all 100 epochs under the same split, mask,
and class-weight rule. The checkpoint with the best validation Macro-F1 is
selected and evaluated on the test set once. Seeds 31/32/33 are aggregated by
arithmetic mean and sample standard deviation.

## Repository map

```text
MoE/                         model, sparse routing, unified runner, audits
multimodal_data/             manifest loading and training-only transforms
build_abcd_*.py              ABCD endpoint and feature construction
derive_nested_abcd_missingness.py  fixed incomplete-input manifest
docs/                        detailed method and data notes
results/                     participant-free aggregate receipts
scripts/validate_release.py  release consistency checks
```

Raw cohort data, identifiers, per-participant predictions, logs, and
checkpoints are not distributed.

## Reproduction entry points

Build the selected endpoint with `build_abcd_adhd_current3_snp_v2.py`, create
the fixed mask with `derive_nested_abcd_missingness.py`, and run the matched
formal comparison with:

```bash
DATASET_MANIFEST=/path/to/manifest.json \
I2MOE_OFFICIAL_ROOT=/path/to/I2MoE \
bash MoE/run_abcd_current3_n3000_missing15_formal_v1.sh
```

Run `python scripts/validate_release.py` before publishing.

The ABCD component suite, ADNI single-expert control, and frozen-checkpoint
modality audit can be reproduced with:

```bash
bash MoE/run_abcd_current3_component_suite_v1.sh
bash MoE/run_adni_single_expert_control_v1.sh
python MoE/run_current_three_seed_modality_audit_v1.py
```
