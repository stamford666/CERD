# CERD

CERD is an incomplete-multimodal classifier with subject-conditioned token
completion, observed/generated provenance, sparse mixture-of-experts fusion,
and multigranular reliability-aware decision fusion. This branch contains the
executable code and participant-free aggregate result receipts; manuscript
sources are maintained on `stamford666-paper`.

## Current matched results

Every entry below is the arithmetic mean ± sample standard deviation of three
independently trained models. No row is a probability ensemble.

### ABCD: strict current-ADHD binary endpoint

The endpoint is a strict low-symptom non-ADHD comparator versus current full
parent K-SADS ADHD. The cohort has 2,120 participants (1,272/848), a
family-disjoint 1,520/297/303 train/validation/test split, and the same fixed
15% incomplete-input manifest for every method.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| CERD | 78.44 ± 0.76 | 77.81 ± 0.35 | 85.12 ± 0.05 |
| Flex-MoE | 78.22 ± 0.33 | 76.56 ± 0.67 | **86.23 ± 0.33** |
| I2MoE | 78.33 ± 1.01 | 76.63 ± 1.28 | 86.08 ± 0.25 |
| MoE++-corrected | 77.12 ± 0.69 | 76.34 ± 0.47 | 85.78 ± 0.21 |
| AnyMod | 79.43 ± 1.66 | 78.51 ± 1.26 | 84.07 ± 1.62 |
| AGDiC-inspired | **80.31 ± 1.01** | **79.21 ± 0.69** | 85.38 ± 0.52 |
| ACADiff-inspired | 78.11 ± 2.10 | 76.80 ± 1.94 | 82.59 ± 1.44 |

CERD is not first on the complete-plus-incomplete aggregate and the repository
does not claim otherwise. On the 45 originally incomplete test participants,
CERD ties the highest Accuracy (78.52%) and gives the highest Macro-F1 (76.70%)
and AUC (85.70%) among these matched methods. Seed values, subset results, and
the frozen protocol are in
[`results/abcd_current_binary_matched_v1.md`](results/abcd_current_binary_matched_v1.md).

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

## ABCD labels and modalities

- Class 0 requires no current, past, partial-remission, or unspecified ADHD and
  at most three current symptoms in each nine-item domain.
- Class 1 requires current full ADHD, all 18 current symptoms explicitly coded
  0/1, and at least six symptoms in one domain.
- Past-only, remission-only, unspecified, and symptom-only participants are
  excluded.

The four model modalities are:

- **I — Imaging (785):** rs-fMRI, SST/n-back/MID task-fMRI, regional T1 gray
  matter volume, and DTI FA;
- **G — Genetics (1,150):** direct 0/1/2 dosages from 14 prespecified
  ADHD-relevant gene windows after training-only variant QC and LD pruning;
- **C — Cognition/health (90):** cognitive tasks, sleep, and physical activity;
- **B — Behavior/environment (219 effective):** non-ADHD behavior,
  temperament/impulsivity, family/neighborhood context, socioeconomic factors,
  and residential/prenatal exposures.

No CatBoost, external teacher, external PRS, ancestry PCs, test-set class
offset, or probability ensemble is used. The detailed biological/data audit is
[`docs/ABCD_CURRENT_ADHD_BINARY_SNP_V1_ZH.md`](docs/ABCD_CURRENT_ADHD_BINARY_SNP_V1_ZH.md).

## Missingness and preprocessing

A fixed label-independent seed-2026 mask makes approximately 15% of each ABCD
split incomplete, usually by hiding one modality, sometimes two, rarely three,
and never all four. The mask is applied before preprocessing. Column filtering,
median imputation, and scaling use training statistics only and operate only
inside modalities that remain observed. A wholly unavailable modality remains
masked for CERD's conditional generator.

ADNI uses its recorded source-table availability; 27.64%, 28.62%, and 31.13%
of train/validation/test participants are incomplete.

## Current ABCD protocol

The validation-selected CERD configuration uses 16 tokens per modality,
hidden width 128, one four-head fusion layer, eight experts with top-2 routing,
dropout 0.30, and rank-4 patch adapters. Seeds 31/32/33 independently select a
checkpoint. Each binary probability threshold is selected from validation
labels, frozen, then replayed before one-time test evaluation.

The full-test Dense FFN control is slightly above sparse CERD, so it is not
reported as evidence of a universal MoE gain. In the incomplete stratum, CERD
exceeds Dense by 2.22 Accuracy, 3.66 Macro-F1, and 1.33 AUC points, and exceeds
E=1 by 4.45 Accuracy and 3.95 Macro-F1 points. See
[`results/abcd_current_binary_component_ablation_v1.md`](results/abcd_current_binary_component_ablation_v1.md).

## Modality faithfulness

Internal branch allocation is a decomposition of the probability mixture, not
an intervention score. The current audit therefore defines label-free model
relevance by the strict-removal prediction-change rate, normalized over the
four modalities within each seed. It is compared with normalized positive
Accuracy decreases from the same removals. Their descriptive Pearson
correspondence is 0.979 on ADNI and 0.947 on ABCD; ABCD's rank order matches
exactly. See [`results/cerd_three_seed_modality_audit_v2.json`](results/cerd_three_seed_modality_audit_v2.json).

## Repository map

```text
MoE/                         model, sparse routing, unified runner, audits
multimodal_data/             manifest loading and training-only transforms
build_abcd_*.py              ABCD endpoint and feature construction
prepare_abcd_missingness.py  fixed missing-input manifest
docs/                        detailed method and data notes
results/                     participant-free aggregate receipts
scripts/validate_release.py  release consistency checks
```

Raw cohort data, identifiers, predictions, and checkpoints are not distributed.

## Reproduction entry points

Build the binary endpoint and modalities with
`build_abcd_adhd_current_binary_snp_v1.py --endpoint current_binary`,
then generate the fixed mask with `prepare_abcd_missingness.py`. The main,
baseline, component, and modality entry points are:

```bash
bash MoE/run_abcd_current_binary_cerd_screen_v1.sh
bash MoE/run_abcd_current_binary_baselines_v1.sh
bash MoE/run_abcd_current_binary_component_ablation_v1.sh

ABCD_MODALITY_AUDIT_CHECKPOINT_ROOT=/path/to/checkpoints/abcd \
ABCD_MODALITY_AUDIT_REFERENCE_ROOT=/path/to/predictions/abcd \
python MoE/run_current_three_seed_modality_audit_v1.py \
  --device 0 --output-dir /path/to/audit-output
```

Run `python scripts/validate_release.py` before publishing.
