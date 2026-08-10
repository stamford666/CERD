# CERD

CERD is a missingness-aware generative reliability mixture-of-experts model for
subject-level multimodal classification with naturally incomplete inputs. This
release contains the **main method only**. Baseline implementations, research
data, checkpoints, predictions, and tuning artifacts are intentionally not
included.

## Tasks

| Dataset | Task | Modalities (IGCB) | Train / Validation / Test |
|---|---|---|---:|
| ABCD | Three-class lifetime K-SADS diagnostic burden: none, one domain, two-or-more domains | imaging, genetic, cognition/health, behavior/environment | 8,430 / 1,692 / 1,745 |
| ADNI | Three-class diagnosis-code classification | MRI, genomic, clinical, biospecimen | 1,480 / 318 / 318 |

The ADNI loader maps the original diagnosis codes 1/2/3 to internal labels
0/1/2. The complete CN/MCI/AD name mapping should be verified against the
original ADNI dictionary before it is stated in a manuscript.

## Method

```text
multimodal features + observed mask
              │
              ▼
 modality-specific tabular patch encoders
        (16 tokens, 128 dimensions)
              │
       conditional cross-attention
       generation for missing tokens
              │
       observed/generated embeddings
              │
       Transformer + sparse top-4 MoE
              │
  joint + unimodal + pairwise predictors
              │
 reliability × evidence branch fusion
              ▼
          class probabilities
```

The common training objective combines classification, router balancing,
conditional reconstruction, diagnostic-branch supervision, artificial modality
dropout, and full-to-reduced-view self-distillation. The release also exposes:

- `DBR`: dual-boundary ranking for ordered three-class targets;
- `MoFe`: more-vs-fewer modality ranking;
- `MaskedBranch-TCL`: correct active branches teach other active branches;
- `TBFD`: trusted active branches teach the final fused prediction;
- parameter-free seed/family hierarchical median consensus.

`core` is identical across datasets. The currently frozen experiments used
`dbr` for ABCD and `mofe`-based variants for ADNI; `unified` enables both DBR
and MoFe when studying one shared objective on both datasets.

## Repository layout

```text
cerd/model.py       model, conditional generators, and reliability fusion
cerd/moe.py         sparse MoE/router implementation
cerd/losses.py      CERD-only training objectives
cerd/data.py        ABCD/ADNI data dispatch and loaders
cerd/datasets/      manifest-driven ABCD adapter
cerd/ensemble.py    hierarchical median consensus
train.py            validation-locked training and optional formal evaluation
hmc.py              probability-file HMC command-line utility
configs/            frozen configuration records
scripts/            three-seed launch examples
tests/              unit tests without research data
```

## Installation

Python 3.10+ and a CUDA-enabled PyTorch environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/laekov/fastmoe.git
cd fastmoe
USE_NCCL=0 python setup.py install
cd ..
pip install -e .
```

FastMoE is a required dependency of the sparse expert backbone. See
[`NOTICE.md`](NOTICE.md) for upstream attribution.

## Data

Data are not distributed. See [`data/README.md`](data/README.md) and the ABCD
manifest template. All feature filtering, imputation, and scaling statistics
are fitted using training subjects only.

## Training

ABCD DBR-MoE, one seed:

```bash
python train.py \
  --data abcd \
  --variant dbr \
  --dataset-manifest /path/to/abcd_manifest.json \
  --seed 0 \
  --device 0
```

ADNI MoFe-MoE, one seed:

```bash
python train.py \
  --data adni \
  --variant mofe \
  --adni-data-root /path/to/adni \
  --adni-image-imputation mean \
  --seed 0 \
  --device 0
```

By default, the runner trains all 50 epochs, selects a checkpoint only by
validation Macro-F1, and does **not** evaluate test data. Add `--evaluate-test`
only after the configuration is frozen. Test prediction is raw-softmax argmax;
there is no threshold, temperature, or class-bias calibration.

The launch scripts run seeds 0/1/2. ADNI variants are `mofe`, `mofe_tcl`, and
`mofe_tbfd`; use `--adni-image-imputation median` for the median-imputation
member.

## Baselines

Baseline source code is not bundled in this first release. The comparison set
uses the same subject splits, modality masks, checkpoint rule, and metrics:

| Method name used in the paper | Main characteristic | Implementation status in the experiments |
|---|---|---|
| Flex-MoE | modality-combination missing bank and sparse expert routing | official-code-derived shared-runner adaptation |
| I2MoE | uniqueness, synergy, and redundancy interaction experts | official-code adapter |
| MoE++ | learned/constant/copy/zero experts with top-2 routing | corrected official-code adapter |
| AnyMod | modality queries, task anchors, and Transformer fusion | reimplementation |
| AGDiC | flow recovery and adaptive graph relations | inspired implementation |
| ACADiff | conditional latent diffusion completion | inspired implementation with fair masked denoising |

The qualifiers above are part of the method names and should not be removed
when reporting these locally adapted results.

## Frozen test results

Values are percentages. Baselines and single CERD rows are mean ± population
standard deviation across three complete 50-epoch runs. HMC is one fixed,
validation-frozen ensemble and therefore has no artificial standard deviation.

### ABCD

| Method | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Flex-MoE | 57.31 ± 0.37 | 54.05 ± 0.94 | 54.19 ± 0.56 | 58.28 ± 0.35 | 73.97 ± 0.24 | 56.93 ± 0.44 |
| I2MoE (official-code adapter) | 57.36 ± 0.69 | 54.52 ± 0.63 | 54.95 ± 0.57 | 58.84 ± 0.53 | 74.11 ± 0.05 | 56.78 ± 0.11 |
| MoE++ (corrected official-code adapter) | 59.52 ± 0.70 | 54.69 ± 0.46 | 54.35 ± 0.59 | 59.13 ± 0.67 | 74.01 ± 0.18 | 57.15 ± 0.33 |
| AnyMod (reimplementation) | 59.14 ± 0.57 | 53.85 ± 0.41 | 54.37 ± 0.26 | 59.10 ± 0.26 | 73.18 ± 0.38 | 56.29 ± 0.38 |
| AGDiC-inspired | 56.10 ± 0.61 | 53.14 ± 0.85 | 53.21 ± 0.72 | 57.04 ± 0.49 | 71.35 ± 0.64 | 54.57 ± 0.59 |
| ACADiff-inspired (fair masked denoising) | 57.40 ± 0.24 | 54.26 ± 0.11 | 54.72 ± 0.16 | 58.73 ± 0.07 | 73.10 ± 0.03 | 56.37 ± 0.07 |
| CERD single DBR-MoE | 60.52 ± 0.87 | 54.53 ± 0.08 | 54.21 ± 0.26 | 59.43 ± 0.17 | 74.56 ± 0.28 | 57.47 ± 0.55 |
| **CERD Het-HMC (9 models)** | **62.12** | **54.84** | 54.48 | **60.12** | **75.68** | **58.99** |

ABCD Het-HMC combines DBR-MoE, CatBoost, and TabM-32. Because baseline code
is intentionally excluded here, this repository alone reproduces the single
DBR-MoE path but not the heterogeneous 62.12% ensemble.

### ADNI

| Method | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Flex-MoE | 62.16 ± 1.04 | 60.30 ± 1.79 | 59.97 ± 1.40 | 61.03 ± 0.99 | 76.98 ± 0.78 | 61.42 ± 1.13 |
| I2MoE (official-code adapter) | 64.05 ± 1.86 | 62.77 ± 2.80 | 61.83 ± 1.54 | 61.91 ± 1.36 | 79.35 ± 1.33 | 66.67 ± 1.65 |
| MoE++ (corrected official-code adapter) | 59.75 ± 0.44 | 59.55 ± 1.41 | 58.85 ± 1.68 | 59.07 ± 1.49 | 79.23 ± 1.14 | 65.63 ± 1.83 |
| AnyMod (reimplementation) | 64.78 ± 3.78 | 63.95 ± 3.12 | 63.55 ± 3.50 | 64.24 ± 3.59 | 79.77 ± 1.73 | 66.20 ± 2.21 |
| AGDiC-inspired | 58.60 ± 2.75 | 55.49 ± 1.65 | 56.54 ± 1.94 | 58.13 ± 2.44 | 75.10 ± 1.11 | 59.58 ± 2.45 |
| ACADiff-inspired (fair masked denoising) | 55.87 ± 0.39 | 53.69 ± 0.88 | 52.14 ± 0.98 | 53.07 ± 0.96 | 71.10 ± 1.01 | 55.75 ± 0.51 |
| CERD single MoFe-MoE | 64.88 ± 1.19 | 64.23 ± 0.41 | 63.75 ± 1.02 | 64.26 ± 1.27 | 80.29 ± 0.71 | 67.37 ± 1.22 |
| **CERD HMC (12 models)** | **67.30** | **67.90** | **66.82** | **66.71** | **83.06** | **72.39** |

The ADNI HMC uses four CERD variants (mean, median, MaskedBranch-TCL, and
TBFD), each with three seeds.

## Reproducibility notes

- Metrics: Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, macro one-vs-rest AUROC, and macro AUPRC.
- Checkpoint selection: validation Macro-F1 only.
- Single-model reporting: three full seeds, population standard deviation.
- HMC: median within seed family, median across families, exactly one final normalization.
- No raw participant data, split IDs, test predictions, or trained weights are committed.
- The recorded table comes from the frozen internal runs. This public refactor
  is source-equivalent at the core-component level, but the protected datasets
  and frozen checkpoints are not redistributed.

## Acknowledgement

CERD builds on the sparse routing foundation of Flex-MoE and FastMoE. Please
cite the corresponding upstream work and see [`NOTICE.md`](NOTICE.md).
