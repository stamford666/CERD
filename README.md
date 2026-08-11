# CERD: Multimodal Probable BED Classification in ABCD

CERD is a missingness-aware generative reliability mixture-of-experts model for
subject-level multimodal classification with naturally incomplete inputs. This
repository releases the **main method only**. Baseline implementations,
protected ABCD data, checkpoints, participant-level predictions, and tuning
artifacts are not distributed.

## Results at a glance

Learned-method entries report mean ± population standard deviation over
three complete runs (seeds 0, 1, and 2). All metrics except MCC are
percentages. BED AP is the primary metric; its no-skill reference is the 2.23%
BED prevalence in the test partition.

| Method | Profile | BED AP | AUROC | BED-F1 | BalAcc | Macro-F1 | Accuracy | Weighted-F1 | Sensitivity | Specificity | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always negative | — | 2.23 | 50.00 | 0.00 | 50.00 | 49.44 | 97.77 | 96.67 | 0.00 | 100.00 | 0.000 |
| Flex-MoE (official-code-derived) | mild | 8.23 ± 0.71 | 77.29 ± 0.68 | 10.11 ± 5.30 | 54.84 ± 3.79 | 54.03 ± 2.42 | 96.01 ± 0.91 | 96.00 ± 0.36 | 11.76 ± 8.66 | 97.93 ± 1.12 | 0.087 ± 0.049 |
| I²MoE (official-code adapter) | strong | 10.34 ± 1.98 | 70.69 ± 2.47 | 14.53 ± 7.17 | 58.50 ± 5.23 | 56.15 ± 3.30 | 95.66 ± 1.14 | 95.91 ± 0.44 | 19.61 ± 11.85 | 97.39 ± 1.43 | 0.134 ± 0.066 |
| MoE++-corrected (I²MoE-code adapter) | mild | 7.30 ± 0.67 | 75.86 ± 1.64 | 6.81 ± 4.94 | 52.72 ± 2.06 | 52.52 ± 2.28 | 96.53 ± 0.79 | 96.20 ± 0.29 | 6.86 ± 5.00 | 98.57 ± 0.92 | 0.053 ± 0.043 |
| AnyMod (reimplementation) | mild | 8.15 ± 0.96 | 73.54 ± 3.70 | 7.39 ± 1.70 | 52.41 ± 0.85 | 52.89 ± 0.72 | 96.86 ± 0.67 | 96.38 ± 0.31 | 5.88 ± 2.40 | 98.93 ± 0.74 | 0.069 ± 0.007 |
| AGDiC-inspired | moderate | 9.04 ± 1.13 | 76.78 ± 0.14 | 11.19 ± 1.08 | 54.47 ± 0.63 | 54.62 ± 0.52 | 96.20 ± 0.11 | 96.13 ± 0.04 | 10.78 ± 1.39 | 98.15 ± 0.14 | 0.093 ± 0.010 |
| ACADiff-inspired | strong | 6.43 ± 1.05 | 71.98 ± 2.20 | 8.58 ± 6.11 | 53.30 ± 2.50 | 53.46 ± 2.96 | 96.73 ± 0.48 | 96.34 ± 0.16 | 7.84 ± 5.55 | 98.75 ± 0.60 | 0.071 ± 0.058 |
| **CERD-MoFe (ours)** | mild | 7.83 ± 0.73 | 74.23 ± 1.10 | 0.00 ± 0.00 | 49.89 ± 0.07 | 49.38 ± 0.03 | 97.56 ± 0.13 | 96.57 ± 0.07 | 0.00 ± 0.00 | 99.78 ± 0.14 | -0.007 ± 0.002 |

The always-negative classifier shows why Accuracy and Weighted-F1 are
misleading for this rare phenotype. Three-seed variation measures training
randomness, not sampling uncertainty; the test set contains only 34 positives.
CERD's validation-selected absolute thresholds did not identify a test BED
case; the zero sensitivity is reported without test-set threshold retuning.

## Task

We use baseline (`ses-00A`) multimodal features to classify a **cumulative,
symptom-derived probable binge-eating disorder (BED) phenotype** observed at
any administered visit from baseline through Year 2.

A participant is positive when the parent K-SADS eating-disorder module
supports probable full BED at baseline, Year 1, or Year 2: binge eating, at
least three associated characteristics, marked distress, and a frequency of at
least once per week for three months. Periods meeting recurrent compensatory
behavior criteria or an anorexia-like symptom pattern are excluded. Released
K-SADS diagnosis-score (`*_dx`) variables are never used.

This endpoint is not a clinician-confirmed diagnosis and not a strict
incident-onset endpoint. Controls are assessed BED-negative; they may have
other eating or psychiatric disorders. Eligibility requires an administered ED
module at all three outcome visits.

| Cohort | Participants | Probable BED | Assessed BED-negative |
|---|---:|---:|---:|
| Full eligible cohort | 10,724 | 257 (2.40%) | 10,467 (97.60%) |
| Training | 7,655 | 194 | 7,461 |
| Validation | 1,541 | 29 | 1,512 |
| Test | 1,528 | 34 | 1,494 |

The split is target-stratified and grouped by genetic family, so no family
appears in more than one partition. Sites may occur across partitions; this is
therefore not a held-out-site evaluation.

## Inputs

Only baseline predictors are used. Counts below are raw numeric features before
training-only filtering.

| Code | Modality | Raw features |
|---|---|---:|
| S | Structural MRI | 71 |
| R | Resting-state fMRI | 68 |
| D | Diffusion MRI | 71 |
| G | Genetic ancestry/population-structure PCs | 32 |
| N | Neurocognition | 147 |
| P | Physical health and development | 58 |
| M | Non-ED mental-health measures | 182 |
| E | Demographic, family, and neighborhood environment | 136 |

All K-SADS eating-disorder fields and direct ED instruments are excluded from
the predictors. We additionally remove four direct CBCL eating/weight proxies
(`Overeating`, `Overweight`, `Doesn't eat well`, and `Vomiting`) and every CBCL
sum, T-score, or count that could algebraically reintroduce those items.
Feature filtering, median imputation, and scaling are fitted on training
participants only. Physical-health predictors include anthropometrics, so this
is a multimodal clinical prediction task rather than an imaging-only claim.

## Method

```text
baseline multimodal features + observed-modality mask
                         │
                         ▼
             modality-specific token encoders
                         │
          conditional cross-attention generation
                 for unavailable modalities
                         │
            observed and generated embeddings
                         │
                Transformer + sparse MoE
                         │
          joint, unimodal, and pairwise heads
                         │
             reliability × evidence fusion
                         ▼
                  BED probability
```

CERD jointly optimizes classification, sparse-router balancing, conditional
reconstruction, branch supervision, artificial modality dropout, and
full-to-reduced-view self-distillation. The binary BED task uses the MoFe
(more-vs-fewer observed modalities) objective. The ordered three-class DBR
objective is not used; explicitly requesting DBR for a binary target is
rejected with an error.

## Baselines

The six comparison implementations are not included in this main-method
release. All are run through one shared data, optimization, checkpoint, and
evaluation interface. The qualifiers below are part of the reported method
names and should not be removed.

| Reported name | Year | Distinguishing mechanism | Local implementation |
|---|---:|---|---|
| [Flex-MoE](https://papers.nips.cc/paper_files/paper/2024/hash/b2f2af5403042b1344f4e93b35fb67d9-Abstract-Conference.html) | 2024 | Missing-modality bank and sparse expert routing | Official-code-derived adaptation |
| [I²MoE](https://proceedings.mlr.press/v267/xin25c.html) | 2025 | Uniqueness, synergy, and redundancy interaction experts | Official-code adapter |
| [MoE++-corrected](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7efe88bb4138d602e56637cfcf713654-Abstract-Conference.html) | 2025 | Learned, constant, copy, and zero experts | Corrected I²MoE-code adapter, not the original LLM implementation |
| [AnyMod](https://papers.miccai.org/miccai-2024/814-Paper1760.html) | 2024 | Modality queries, task anchors, and Transformer fusion | Reimplementation |
| [AGDiC-inspired](https://papers.miccai.org/miccai-2025/0059-Paper0965.html) | 2025 | Flow-based recovery and adaptive graph relations | Inspired token-space implementation |
| [ACADiff-inspired](https://arxiv.org/abs/2603.09931) | 2026 | Conditional latent diffusion completion | Inspired token-space implementation |

## Evaluation protocol

Each method receives the same participants, eight feature tables, natural
modality masks, training-only preprocessing, 50-epoch budget, and three
predeclared imbalance profiles. For each method, one profile is selected using
seed-0 validation BED AP only; that method-specific profile is then frozen and
the method is retrained with seeds 0, 1, and 2.

- Checkpoint: highest validation positive-class average precision (BED AP).
- Decision rule: one absolute BED-probability threshold selected on validation
  Macro-F1 and applied unchanged to test predictions.
- Test isolation: test labels and score distributions do not select the
  profile, checkpoint, or threshold.
- Reporting: mean ± population standard deviation over three complete runs.

Architecture-specific validated/default settings are fixed within each method;
the shared imbalance grid is `mild` (0.35/0.15), `moderate` (0.50/0.25), and
`strong` (0.50/0.50) sampler/class-weight powers.

The fixed batch sizes for Flex-MoE/I²MoE/MoE++/AnyMod/AGDiC/ACADiff/CERD are
32/64/128/128/128/128/64, respectively. I²MoE uses 64 because its interaction
transformer exceeds the available GPU memory at batch 128; this choice is fixed
before validation and used unchanged in formal runs.

## Installation and training

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

One validation-only run:

```bash
python train.py \
  --data abcd \
  --variant mofe \
  --modality SRDGNPME \
  --dataset-manifest /path/to/abcd_bed_manifest.json \
  --train-epochs 50 \
  --warm-up-epochs 5 \
  --batch-size 64 \
  --lr 0.0001 \
  --weight-decay 0 \
  --sampler-power 0.35 \
  --class-weight-power 0.15 \
  --num-layers-pred 2 \
  --more-fewer-rank-loss-weight 0.1 \
  --dual-boundary-rank-loss-weight 0 \
  --seed 0 \
  --device 0 \
  --no-evaluate-test
```

The JSON files under `configs/` are human-readable experiment records; they are
not loaded automatically. Keep the selected sampler/class-weight powers in the
command, `scripts/train_abcd.sh`, and the record synchronized. The three-seed
script interface is
`scripts/train_abcd.sh MANIFEST [DEVICE] [EVALUATE_TEST] [OUTPUT_DIR]`; set its
third positional argument to `true` only after model selection is frozen.
Protected split IDs and participant-level outputs must remain outside version
control.

## Repository layout

```text
cerd/model.py       CERD encoders, conditional generators, and reliability fusion
cerd/moe.py         sparse MoE/router implementation
cerd/losses.py      CERD training objectives
cerd/metrics.py     binary-safe metrics and validation threshold selection
cerd/data.py        dataset dispatch
cerd/datasets/      manifest-driven ABCD adapter
train.py            validation-locked training and optional formal evaluation
configs/            human-readable main-method configuration records
scripts/            three-seed launch examples
tests/              unit tests without research data
```

## Data and release limitation

ABCD data are not distributed; see [`data/README.md`](data/README.md) and the
manifest template. The exact ABCD release identifier is not embedded in the
protected local manifest and must be verified with the data administrator
before publication. The [ABCD Release 6.0 notes](https://docs.abcdstudy.org/latest/documentation/release_notes/6_0.html)
warned that released eating-disorder diagnosis scores were overly restrictive,
so this study derives the research phenotype directly from symptom fields and
does not use released `*_dx` scores. [Release 7.0](https://docs.abcdstudy.org/latest/documentation/release_notes/7_0.html)
later reingested and cleaned diagnostic and symptom data. The endpoint should
therefore be described only as symptom-derived probable BED, and the experiment
should be repeated after the precise source release is confirmed.

Legacy ADNI loaders and ordered three-class objectives remain in the package for
method portability, but no ADNI experiment or result is part of this ABCD BED
release.

## Acknowledgement

CERD builds on the sparse-routing foundations of Flex-MoE and FastMoE. See
[`NOTICE.md`](NOTICE.md) for upstream attribution.
