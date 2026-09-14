# CERD

CERD is an incomplete-multimodal classifier with subject-conditioned token
completion, observed/generated provenance, sparse mixture-of-experts fusion,
and multigranular reliability-aware decision fusion. This branch contains the
executable code and participant-free aggregate result receipts; manuscript
sources are maintained on `stamford666-paper`.

## Current matched results

Every entry is the arithmetic mean ± sample standard deviation over three
independently trained models. No probability ensemble is used.

### ABCD: current ADHD severity endpoint

The frozen cohort contains 3,000 participants and uses three disjoint baseline
parent K-SADS severity strata. The severity variable is the larger of the two
nine-item current symptom-domain counts: class 0 has 0--2 symptoms per domain,
no ADHD status, and no current impairment; class 1 has 3--5 symptoms in either
domain, no current full ADHD, and a positive onset/cross-setting condition;
class 2 has 6--9 symptoms, current full ADHD, current impairment, and a positive
onset/cross-setting condition. This is an ordered cross-sectional severity task,
not longitudinal progression. The family-disjoint train/validation/test split
contains 2,134/438/428 participants. A fixed, label-independent mask makes
approximately 15% of every split incomplete.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **66.82 ± 2.14** | **55.18 ± 1.16** | **77.90 ± 0.55** |
| Flex-MoE | 63.79 ± 2.60 | 51.44 ± 2.28 | 74.69 ± 1.30 |
| I2MoE | 63.24 ± 0.82 | 49.46 ± 1.27 | 74.60 ± 0.92 |
| MoE++ | 64.64 ± 2.22 | 52.06 ± 4.35 | 74.44 ± 1.81 |
| AnyMod | 62.15 ± 2.00 | 49.09 ± 0.61 | 69.17 ± 4.68 |
| AGDiC | 62.69 ± 3.04 | 48.83 ± 0.84 | 71.71 ± 2.66 |
| ACADiff | 63.16 ± 1.18 | 49.05 ± 1.56 | 70.90 ± 0.78 |

CERD has the highest mean on all three metrics. Relative to the strongest
baseline in each column, the margins are 2.18 Accuracy, 3.12 Macro-F1, and
3.20 Macro-AUROC points. The complete result, per-seed values,
availability-stratified metrics, and frozen protocol are in
[`results/abcd_severity3_n3000_missing15_formal_v2.md`](results/abcd_severity3_n3000_missing15_formal_v2.md).

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
On ABCD, removing multigranular decomposition reduces Accuracy from 66.82 to
63.63, while removing conditional completion lowers incomplete-subset Macro-F1
from 55.06 to 45.87. A strict reliability control fixes every usable modality
score to one while retaining confidence and branch priors; it lowers overall
Macro-F1 to 53.91. Full CERD also exceeds the capacity-aligned Dense FFN by
0.39/1.37/0.59 Accuracy/Macro-F1/AUROC points and the one-expert control by
1.87/2.36/0.10 points. On the incomplete ABCD stratum, the Dense-FFN margins
expand to 1.56/4.13/2.66 points. On ADNI, the one-expert control gives 64.05
Accuracy and 63.70 Macro-F1, compared with 65.72 and 64.56 for full CERD.

Frozen-checkpoint strict-removal audits normalize both prediction-change rates and
positive Accuracy decreases across the four modalities. Their modality-level Pearson
correlations are 0.979 on ADNI and 0.825 on ABCD (Spearman rho = 0.8 on both).
All reference predictions are reproduced with zero label mismatches before the
interventions are evaluated.

The complete participant-free report is
[`results/severity3_e4k2_component_and_modality_audit_v2.md`](results/severity3_e4k2_component_and_modality_audit_v2.md).

## ABCD labels and modalities

| Class | Definition | Participants |
|---|---|---:|
| 0 | No current/past/remission/unspecified ADHD; 0--2 symptoms per domain; no current impairment | 1,733 |
| 1 | No current full ADHD; 3--5 symptoms in either domain; positive onset/cross-setting condition | 404 |
| 2 | Current full ADHD; 6--9 symptoms; current impairment and onset/cross-setting condition | 863 |

The symptom-count intervals are mutually exclusive. All target-defining ADHD
fields and CBCL ADHD/attention scores are excluded from predictors.

The four model modalities are:

- **I — Imaging (785):** rs-fMRI, SST/n-back/MID task-fMRI, regional T1-weighted
  gray-matter intensity, and DTI FA;
- **G — Genetics (1,172):** direct 0/1/2 SNP dosages after training-only
  missingness/MAF QC and LD pruning in prespecified gene windows; no PCA, PRS,
  or ancestry components;
- **C — Cognition/health (90):** cognitive tasks, sleep, and physical activity;
- **B — Behavior/environment (221 raw; 219 retained):** non-ADHD behavior,
  temperament, family/neighborhood context, socioeconomic measures, and
  residential/prenatal exposures.

The detailed Chinese endpoint and data audit is
[`docs/ABCD_SEVERITY3_LABEL_N3000_V1_ZH.md`](docs/ABCD_SEVERITY3_LABEL_N3000_V1_ZH.md).
The exact Chinese ABCD MRI-space, normalization, and missing-value protocol is
[`docs/ABCD_NORMALIZATION_ZH.md`](docs/ABCD_NORMALIZATION_ZH.md).

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

Build the selected endpoint with `build_abcd_adhd_current3_snp_v2.py --endpoint severity3`, create
the fixed mask with `derive_nested_abcd_missingness.py`, and run the matched
formal comparison with:

```bash
DATASET_MANIFEST=/path/to/manifest.json \
I2MOE_OFFICIAL_ROOT=/path/to/I2MoE \
bash MoE/run_abcd_severity3_n3000_missing15_formal_v2.sh
```

Run `python scripts/validate_release.py` before publishing.

The ABCD component suite, ADNI single-expert control, and frozen-checkpoint
modality audit can be reproduced with:

```bash
bash MoE/run_abcd_severity3_e4k2_ablation_v1.sh
bash MoE/run_adni_single_expert_control_v1.sh
bash MoE/run_abcd_severity3_modality_audit_v1.sh
```
