# CERD: missingness-aware multimodal classification

CERD follows one decision path: reconstruct missing latent information, mark
whether each representation was observed or generated, estimate its trust, and
aggregate valid candidate predictions. Its conditional-generative neural
members retain a sparse mixture-of-experts (MoE) backbone. The frozen CGHC-v1
predictor then combines independently trained member families by hierarchical
median consensus; this final cross-model consensus is not itself an MoE.

## Method at a glance

```text
(1) reconstruct missing latent representations
                         │
(2) attach observed/generated provenance
                         │
       [retained Transformer + sparse-MoE backbone]
                         │
       candidate joint/unimodal/pairwise decision views
                         │
(3) estimate trust:
    input reliability × normalized-entropy confidence
                         │
(4) aggregate valid branches into the final decision
                         │
      fixed hierarchical median across model families
```

The four stages above are CERD's single conceptual structure. The sparse-MoE
layer is retained fusion machinery between provenance encoding and trust
estimation, not a fifth stage or a separate CERD contribution. Joint, unimodal,
and pairwise outputs are candidate decision views rather than independent
method modules. In the final predictor, ABCD combines the conditional-generative
Rank-MoE family with CatBoost and TabM-32 families, while ADNI combines four
conditional-generative MoE variants. Conditional generation and MoE are
therefore retained without forcing the final fusion rule to be another MoE.

The detailed formulation is in the [Method narrative](docs/METHOD.md). The
[evaluation protocol](docs/METHOD_REVISION_VALIDATION_PROTOCOL.md) defines the
common-six reporting boundary, validation controls, and privacy requirements.
The [controlled final-result builder](docs/FINAL_RESULT_BUILDER.md) defines the
private-to-public, receipt-bound construction interface.

## Current CGHC-v1 results

The receipt-bound [CGHC-v1 result report](results/cghc_v1.md) is the current
common-six evaluation of the frozen heterogeneous-consensus predictor. It also
links the matched completion ablation and aggregate, validation-only modality–
disease association analysis. The latter describes model associations and must
not be interpreted as evidence of disease causation.

## Archived core-member development evidence

The receipt-bound block below is retained for provenance, but it predates and
is superseded by the current CGHC-v1 predictor above. Its `FINAL` label applies
only to that earlier development campaign; it is not the current result table.

<!-- FINAL_RESULTS_START -->
> **Release status: FINAL.** Adaptive same cohort development evidence only; both scored cohorts were reused for model and configuration selection, so any Holm adjusted difference is descriptive and confirmatory support is false.

All reported predictive results use exactly six metrics. Values are percentages; no binary-only or task-specific metric is mixed into either dataset.

### Evaluation boundary

| Dataset | Evidence scope | Design | Split | n | Folds | Partition reused | Selection independent |
|---|---|---|---|---:|---:|:---:|:---:|
| ADNI | development cv | five-fold subject-level out-of-fold ensemble evaluation | training1480 out-of-fold development cohort | 1480 | 5 | yes | no |
| ABCD | development cv | five-fold family-disjoint out-of-fold ensemble evaluation | dev946 family-disjoint out-of-fold development cohort | 946 | 5 | yes | no |

ADNI campaign boundary: ADNI validation-318, test-318, and unassigned-910 are outside this campaign: not selected into any arm, not iterated over, not scored, and not included in any fitted statistic.

ABCD campaign boundary: ABCD protected temporal internal holdout 850 is outside this campaign: not selected into any arm, not iterated over, not scored, and not included in any fitted statistic.

### Final common-six results

| Dataset | Method | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| ADNI | CERD | 61.82 | 61.93 | 60.69 | 60.25 | 78.18 | 63.45 |
| ADNI | Comparator | 61.01 | 61.60 | 60.30 | 59.82 | 77.91 | 63.28 |
| ABCD | CERD | 53.81 | 43.58 | 43.20 | 54.27 | 62.29 | 42.63 |
| ABCD | Comparator | 54.33 | 43.40 | 43.14 | 54.56 | 62.34 | 43.05 |

### Statistical comparisons

Each dataset contributes exactly one paired, one-sided Macro-F1 comparison. Swap-test p-values are Holm-adjusted jointly across ADNI and ABCD; significance is derived from adjusted p < 0.05. Confirmatory support additionally requires a positive 95% bootstrap lower bound on a selection-independent, non-reused locked evaluation scope.

| Dataset | Paired comparison | Metric | Difference (pp) | Bootstrap lower (pp) / level | Swap p | Holm adjusted p | Alpha | Significant difference | Confirmatory support | Test / unit / n | Swap draws / RNG | Bootstrap draws / RNG |
|---|---|---|---:|---:|---:|---:|---:|:---:|:---:|---|---|---|
| ADNI | CERD − Comparator | Macro-F1 | +0.39 | -1.22 / 95.0% | 0.3463 | 0.6925 | 0.0500 | no | no | paired swap test / subject / n=1480 | 50000 / 20260905 | 20000 / 20260906 |
| ABCD | CERD − Comparator | Macro-F1 | +0.05 | -2.55 / 95.0% | 0.4864 | 0.6925 | 0.0500 | no | no | paired swap test / family / n=922 | 50000 / 20260905 | 20000 / 20260906 |

### Pre-specified matched ablations

These rows are validation controls for the four-stage decision chain, not eight coequal CERD modules. Each row is one pre-specified matched configuration relative to full CERD; differences are descriptive ablation effects and do not by themselves establish causal necessity. The dense-FFN row is a backbone-sensitivity control, and canonical artifact order is retained for traceability.

| Dataset | Ablation | Accuracy | BalAcc | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| ADNI | Dense FFN instead of sparse MoE | 63.65 | 63.25 | 62.88 | 62.68 | 78.89 | 64.61 |
| ADNI | Without observed/generated provenance | 61.28 | 61.40 | 60.23 | 59.79 | 77.80 | 63.18 |
| ADNI | Uniform instead of reliability-aware branch weights | 60.74 | 61.14 | 60.48 | 59.89 | 78.33 | 64.51 |
| ADNI | Mean instead of gated-attention pooling | 61.82 | 61.99 | 60.86 | 60.39 | 77.96 | 63.30 |
| ADNI | Without stochastic observed-subset context masking | 61.55 | 61.73 | 60.75 | 60.32 | 78.42 | 63.49 |
| ADNI | Without latent completion | 61.62 | 61.69 | 59.36 | 59.17 | 79.58 | 66.25 |
| ADNI | Without more/fewer-modality objective | 62.09 | 62.43 | 61.20 | 60.65 | 78.35 | 63.78 |
| ADNI | Without generator output gate | 61.69 | 61.88 | 60.68 | 60.23 | 78.08 | 63.66 |
| ABCD | Dense FFN instead of sparse MoE | 54.55 | 42.48 | 42.52 | 54.48 | 61.81 | 42.86 |
| ABCD | Without observed/generated provenance | 54.23 | 43.49 | 43.29 | 54.46 | 62.43 | 42.97 |
| ABCD | Uniform instead of reliability-aware branch weights | 52.85 | 44.06 | 43.27 | 53.75 | 62.04 | 43.38 |
| ABCD | Mean instead of gated-attention pooling | 53.81 | 44.09 | 43.63 | 54.31 | 61.93 | 42.97 |
| ABCD | Without stochastic observed-subset context masking | 51.69 | 42.23 | 41.65 | 52.62 | 62.53 | 42.95 |
| ABCD | Without latent completion | 50.74 | 42.14 | 41.26 | 52.10 | 61.66 | 42.64 |
| ABCD | Without more/fewer-modality objective | 52.96 | 42.97 | 42.53 | 53.61 | 62.50 | 43.14 |
| ABCD | Without generator output gate | 52.43 | 42.91 | 42.31 | 53.13 | 62.33 | 42.68 |

### Aggregate interpretability

The explanation artifact contrasts complete and incomplete inputs using modality-level decision allocation and grouped joint/unimodal/pairwise branch mass. These are descriptive routing summaries, not causal feature importance. No participant-level explanation data are released.

| Dataset | Condition design | Complete condition | Incomplete condition | Aggregate source |
|---|---|---|---|---|
| ADNI | natural_disjoint | naturally complete four-modality inputs | naturally incomplete inputs | aggregate out-of-fold checkpoint replay |
| ABCD | natural_disjoint | naturally complete four-modality inputs | naturally incomplete inputs | aggregate out-of-fold checkpoint replay |

- [Common-six performance](figures/common6.svg)
- [Ablation effects](figures/ablations.svg)
- [Decision allocation and branch mass](figures/decision_allocation.svg)

<!-- FINAL_RESULTS_END -->

The [aggregate artifact contract](results/README.md) defines how the tables and
figures are generated. A final release is rejected unless both datasets have
all six metrics, the required ablations, paired statistics, and aggregate
interpretability values. The repository does not ship a placeholder result JSON;
the public-release check accepts only the completed aggregate artifact and its
exactly synchronized generated README and figures.

## Core reference implementation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/laekov/fastmoe.git
cd fastmoe && USE_NCCL=0 python setup.py install && cd ..
pip install -e .

# Independent binary ABCD reference benchmark; it is not the final three-class row.
scripts/train_abcd.sh ABCD_MANIFEST.json 0 false outputs/abcd

# ADNI core-method reference workflow.
scripts/train_adni.sh ADNI_DATA_DIRECTORY 0 false outputs/adni

# Run one matched control on the same base options and seed.
python train.py --data adni --variant mofe --ablation-id no_provenance \
  --data-order-seed 0 --fold-id 0 --split-receipt-sha256 SPLIT_SHA256 \
  --adni-data-root ADNI_DATA_DIRECTORY
```

This is a core-method reference, not the private frozen campaign runner.
The final method-revision ABCD result uses the frozen three-class presentation
endpoint on dev946 with five family-disjoint folds; the manifest-driven binary
ABCD command above is a separate interface example and contributes no value to
the final table.
The CLI value `--variant mofe` is retained only as a legacy compatibility
identifier; it does not name CERD's contribution, and MoE remains the fusion
backbone.
Protected data, fitted preprocessing objects, participant-level outputs,
checkpoints, and internal orchestration are not distributed. The descriptive
JSON files under `configs/` are not consumed by `train.py` and do not imply
exact numerical reproduction.

`--ablation-id` accepts `full` and the eight ordered public IDs in
[`configs/matched_ablations_v1.json`](configs/matched_ablations_v1.json). Every
named arm changes exactly one control profile bit. Fold identity, model and
data-order seeds, split-receipt digest, resolved control profile, and canonical
configuration digest are bound into the run protocol. The detailed no-clobber,
construction-alignment, and random-number guarantees belong to the
[evaluation protocol](docs/METHOD_REVISION_VALIDATION_PROTOCOL.md); they verify
matched controls rather than define additional method modules.

## Repository layout

```text
cerd/       reconstruct/provenance/trust/decision pipeline; retained MoE backbone
configs/    de-identified descriptive parameter records
data/       protected-data interface documentation only
docs/       Method and final evaluation protocol
results/    final aggregate artifact contract; no participant-level data
figures/    aggregate-only generated SVG figures
scripts/    training entry points and public-result renderer
tests/      core implementation and release-contract tests
```

ABCD and ADNI are controlled-access datasets and are not redistributed. See
the [data interface notes](data/README.md) and [release notice](NOTICE.md).
