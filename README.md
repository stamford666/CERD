# CERD

This repository contains the implementation and aggregate result records for
CERD, a missing-modality multimodal classifier with conditional completion,
sparse mixture-of-experts encoding, and reliability-aware decision fusion.

> **Repository correction (2026-09-10).** This history was rebuilt from the
> runner and preprocessing code that produced the current experiments. The
> previously uploaded legacy `cerd/` package, stale manuscript, old ABCD
> status-3 artifacts, and probability-ensemble ADNI summary are intentionally
> absent. They described a different data representation or aggregation rule.

## Main matched comparisons

All entries are arithmetic means and sample standard deviations across three
independently trained seeds, not probability ensembles.

### ABCD presentation3 v4

The current ABCD source-to-feature definitions are documented in Chinese in
[`docs/ABCD_COURSE3_MODALITIES_ZH.md`](docs/ABCD_COURSE3_MODALITIES_ZH.md).
The document gives the exact rs/task-fMRI contrasts, T1 and DTI quantities,
training-only SNP QC/LD pruning, cognition/health instruments, and every
family/community/address-linked environmental source used for the new
clinical-course rerun.

CERD and all six baselines use the same 2,868 participants, features,
family-disjoint split, fixed 15% missingness manifest, and seeds 31/32/33.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **59.24 ± 1.65** | **53.62 ± 1.40** | **74.43 ± 1.03** |
| Flex-MoE | 57.71 ± 1.14 | 50.07 ± 1.03 | 72.58 ± 1.25 |
| I2MoE | 55.29 ± 1.33 | 49.12 ± 1.20 | 73.38 ± 1.05 |
| MoE++ | 57.06 ± 0.98 | 48.54 ± 0.78 | 73.44 ± 1.65 |
| AnyMod | 54.48 ± 4.44 | 48.46 ± 4.23 | 69.60 ± 3.16 |
| AGDiC | 54.80 ± 0.92 | 48.58 ± 0.99 | 70.70 ± 0.75 |
| ACADiff | 54.32 ± 1.22 | 47.00 ± 0.71 | 68.31 ± 0.63 |

CERD is highest on all three metrics. Its margins over the strongest baseline
for each metric are +1.53 Accuracy, +3.55 Macro-F1, and +0.99 Macro-AUROC
percentage points. Seed-level values are in the
[ABCD matched report](results/abcd_adhd_presentation3_snp_missing15_v4.md).
The 71 structural variables are correctly named as T1 regional signal
intensities; every method receives the same variables, so the naming correction
does not alter comparison fairness.

### ADNI matched comparison

All methods use the same frozen ADNI split and seeds 0/1/2. The CERD row was
selected from six ordinary configurations using validation data only; baseline
rows are unchanged from the frozen formal campaign.

| Method | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| **CERD** | **65.72 ± 1.09** | **64.56 ± 2.09** | **81.07 ± 0.55** |
| Flex-MoE | 62.26 ± 1.09 | 60.62 ± 2.12 | 77.75 ± 1.03 |
| I2MoE | 64.05 ± 2.27 | 61.83 ± 1.89 | 79.35 ± 1.63 |
| MoE++ | 59.75 ± 0.54 | 58.85 ± 2.06 | 79.23 ± 1.39 |
| AnyMod | 64.78 ± 4.63 | 63.55 ± 4.28 | 79.77 ± 2.12 |
| AGDiC | 58.60 ± 3.36 | 56.54 ± 2.38 | 75.10 ± 1.36 |
| ACADiff | 55.87 ± 0.48 | 52.14 ± 1.20 | 71.10 ± 1.24 |

CERD is highest on all three metrics, with margins of 0.94 Accuracy, 1.01
Macro-F1, and 1.30 Macro-AUROC percentage points over the strongest baseline
for each metric. See the
[updated ADNI matched report](results/adni_matched_updated_cerd_v4.md) and the
[CERD seed-level receipt](results/adni_direct_three_seed_mean_v2.md).

## Separate representation results

These rows are not inserted into either matched table because their model or
feature configuration differs from the corresponding frozen campaign.

| Dataset and evaluation | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---|---:|---:|---:|
| ABCD v5, validation-selected checkpoints | 57.71 ± 1.56 | 52.66 ± 1.25 | 73.63 ± 0.87 |
| ABCD v5, fixed 14-epoch development-pool refit | 58.76 ± 2.20 | 55.49 ± 1.29 | 75.38 ± 0.76 |

The v5 feature screen reached 61.35% validation Accuracy, which is not a
held-out test result. Matched baselines have not yet been rerun on v5. Full
seed-level records and scope notes are in [`results/`](results/README.md).

## Current ABCD endpoint and data

The task uses 2,868 ABCD baseline participants and three clinically
interpretable parent K-SADS ADHD classes:

1. low-symptom non-ADHD comparator (1,270);
2. full ADHD, predominantly inattentive presentation (635);
3. full ADHD with a hyperactive/impulsive component, including combined
   presentation (963).

Participants are split 2,041/414/413 into train/validation/test sets with no
genetic-family overlap. A fixed, label-independent mask makes approximately
15% of participants incomplete in every split. It usually removes one or two
of the four modalities, rarely three, and never all four.

The four input modalities are:

- **I — imaging:** rs-fMRI, SST/n-back/MID task-fMRI, cortical volume/ICV,
  cortical thickness, subcortical volume/ICV, ICV, and DTI FA;
- **G — genetics:** 1,178 training-QC/LD-pruned SNP dosages from 14
  prespecified candidate-gene windows, summarized by one training-fitted PC
  per gene (14 final features); no external PRS or ancestry PCs;
- **C — cognition/health:** 90 cognitive, sleep, and physical-activity
  variables plus 19 SST/n-back/MID task-performance summaries;
- **B — behavior/environment:** non-ADHD CBCL domains, temperament and
  impulsivity, family/neighborhood measures, and demographic/prenatal context.

The matched v4 input uses 785 imaging, 1,178 SNP, 90 cognition/health, and 218
effective behavior/environment features. The v5 representation study replaces
the 71 T1 signal-intensity variables with actual regional volume and thickness,
compresses SNPs within genes, and adds task-performance summaries. See the
primary matched-data description in
[`docs/ABCD_PRESENTATION3_SNP_V4_ZH.md`](docs/ABCD_PRESENTATION3_SNP_V4_ZH.md)
and the v5 audit in [`docs/ABCD_DATA_ZH.md`](docs/ABCD_DATA_ZH.md).

## Current CERD implementation

The executable model is `AGMGFlexMoE` in [`MoE/models.py`](MoE/models.py),
trained by [`MoE/baseline_runner.py`](MoE/baseline_runner.py). For each
modality, a feature-to-token encoder creates a configurable number of tokens
with width 128. Conditional
generators complete missing token sets from the observed modalities while a
provenance indicator distinguishes observed from generated tokens. Sparse MoE
Transformer blocks use 16 experts with top-4 routing. Prediction combines one
joint, four modality-centered, and six pair-centered branches using normalized
weights based on modality reliability, branch evidence, and a learned branch
prior.

The matched ABCD main configuration uses eight tokens per modality, four
attention heads, one fusion layer, dropout 0.35, rank-4 patch adapters, and
training-time modality dropout 0.25. The updated ADNI configuration uses 16
tokens, dropout 0.30, and a validation-selected learning rate of 1.25e-4. All
loss terms are recorded in
[`docs/METHOD.md`](docs/METHOD.md). CERD does not use
CatBoost, an external teacher, class-logit offsets, checkpoint ensembling,
ancestry PCs, or external PRS.

## Repository map

```text
MoE/                         current model, MoE layers, runner, and run script
multimodal_data/             manifest loading, training-only transforms, metrics
build_abcd_*.py              endpoint and ABCD representation builders
prepare_abcd_*.py            multimodal/missingness/refit manifest preparation
UnifiedAD/, MoRA/, ...       baseline model adapters used by the unified runner
docs/                        current method and data documentation
results/                     aggregate, participant-free result receipts
scripts/validate_release.py  consistency and accidental-artifact checks
```

Raw ABCD and ADNI data, participant identifiers, predictions, and checkpoints
are not distributed. ABCD access must be obtained separately and the local
paths supplied on the command line.

## Environment and execution

Python dependencies are listed in `requirements.txt`. The sparse MoE layer also
requires FastMoE, whose installation depends on the local PyTorch/CUDA setup.
The optional I2MoE and MoE++ adapters expect an official I2MoE checkout at
`I2MoE/` in the repository root.

Build the matched v4 dataset and its fixed missingness manifest:

```bash
python build_abcd_adhd_presentation3_snp_v4.py \
  --abcd-root /path/to/ABCD/Tabulated \
  --genotype-prefix /path/to/genotypes \
  --gene-ranges /path/to/gene_coordinates_auto.bed \
  --plink /path/to/plink \
  --output-dir data/abcd_adhd_presentation3_snp_v4

python prepare_abcd_missingness.py \
  --source-manifest data/abcd_adhd_presentation3_snp_v4/manifest.json \
  --output-dir data/abcd_adhd_presentation3_snp_v4_random_missing15 \
  --pattern-mode mixed --seed 2026
```

Train one seed of the matched CERD main configuration, then formally replay the
saved validation checkpoint before opening the test split:

```bash
PYTHON_BIN=python bash MoE/run_abcd_presentation3_v4_validation.sh 31 0

python MoE/evaluate_validation_checkpoint.py \
  --source-result MoE/abcd_adhd_presentation3_snp_missing15_v4_validation/abcd/our_moe_seed31.json \
  --output-root MoE/abcd_adhd_presentation3_snp_missing15_v4_formal \
  --device 0
```

Run one seed of the validation-selected ADNI configuration, then apply the
same validation-replay gate before formal evaluation:

```bash
PYTHON_BIN=python bash MoE/run_adni_cerd_v2_validation.sh 0 0

python MoE/evaluate_validation_checkpoint.py \
  --source-result MoE/adni_cerd_v2_validation/adni/our_moe_seed0.json \
  --output-root MoE/adni_cerd_v2_formal \
  --device 0
```

Build the optional v5 representation from the matched presentation-3 v4
manifest:

```bash
python build_abcd_adhd_presentation3_refined_v5.py \
  --source-missing-dir /path/to/abcd_presentation3_v4_missing15 \
  --abcd-root /path/to/ABCD/Tabulated \
  --genotype-root /path/to/genotype_smokescreen \
  --output-dir data/abcd_adhd_presentation3_refined_v5 \
  --components-per-gene 1
```

Create the development-pool refit manifest and reproduce a fixed-epoch run:

```bash
python prepare_abcd_fixed_epoch_refit_manifest.py \
  --source-manifest data/abcd_adhd_presentation3_refined_v5/manifest.json \
  --output-dir data/abcd_adhd_presentation3_refined_v5_refit_dev

DATASET_MANIFEST=data/abcd_adhd_presentation3_refined_v5_refit_dev/manifest.json \
PYTHON_BIN=python bash MoE/run_abcd_fixed_epoch_refit_v1.sh 31 0 14
```

Run the lightweight release checks with:

```bash
python scripts/validate_release.py
```
