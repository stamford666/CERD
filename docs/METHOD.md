# Current CERD method and implementation map

This document describes the code path used by the result receipts in this
repository. It is intentionally limited to the model itself and does not treat
ordinary hyperparameter selection as a methodological contribution.

## 1. Inputs and observed-mask semantics

Each sample contains up to four tabular modalities in the fixed order
`I, G, C, B`. The manifest loader returns a Boolean observed-modality mask in
addition to feature tensors. Whole-modality absence is applied before any
statistics are fitted. Column filtering, scalar median imputation, and z-score
standardization are fitted using observed training entries only, then reused
unchanged for validation and test data.

## 2. Modality tokenization

Every modality has its own feature-to-token encoder. It maps the input vector to
patch tokens with hidden width 128 and adds modality and position identity.
The separate encoders avoid requiring the raw feature spaces to have equal
dimensionality. Rank-4 patch adapters provide a low-rank modality-specific
update without duplicating the full encoder.

## 3. Conditional token completion

When modality `m` is absent, a modality-specific conditional generator queries
the available modality tokens and predicts the missing token set. A learned
output gate controls the residual contribution of the generated tokens.
Observed tokens pass through unchanged. Provenance embeddings mark whether a
token is observed or generated, so downstream fusion is not forced to assign
them equal credibility.

Training masks additional observed modalities with probability 0.25 and uses
their real token representations as reconstruction targets. At least one
context modality is retained. The reconstruction objective has weight 0.25;
the generator receives reconstruction gradients but not the primary task
gradient in the released configuration (`generator_task_grad=False`). This
separates completion learning from unrestricted label-driven imputation.

## 4. Sparse MoE fusion

The completed token sequence is processed by a Transformer layer whose
feed-forward block is a sparse mixture of experts. There are 16 experts and one
router; each token is sent to the top four experts. This implements
sample- and token-dependent capacity while keeping expert computation sparse.
The router balance term has weight 0.01.

The implementation is in `AGMGFlexMoE` (`MoE/models.py`) and its FastMoE layer
is in `MoE/moe_module.py`. The executable training/evaluation path is
`MoE/baseline_runner.py --model our_moe`.

## 5. Reliability-aware decision decomposition

CERD forms eleven prediction branches from the same fused representation:

- one joint branch using all usable modality features;
- four modality-centered branches;
- six pair-centered branches for all two-modality combinations.

These branches are decision specialists, not claims that the latent
representations are statistically pure unimodal or pairwise decompositions.
For a missing pattern, branches that require unusable evidence are masked.
Each active branch receives a score formed from modality reliability, detached
predictive evidence (negative entropy), and a learned low-capacity prior. A
softmax over active branches makes the final weights sum to one. The final
prediction is the weighted mixture of branch probabilities.

The branch auxiliary classification loss has weight 0.10. Training also uses a
dropped-view cross-entropy term (0.10), self-distillation from the less-missing
view at temperature 2 (0.15), and a more-versus-fewer-modality ranking term
(0.10). These terms teach prediction stability under missingness; they do not
change the test labels or aggregation rule.

## 6. Released configurations

| Item | Matched ABCD v4 main experiment | ABCD v5 representation study |
|---|---:|---:|
| Hidden width | 128 | 128 |
| Patch tokens per modality | 8 | 16 |
| Attention heads | 4 | 4 |
| Fusion layers | 1 | 1 |
| Prediction layers | 1 | 1 |
| Experts / routers / top-k | 16 / 1 / 4 | 16 / 1 / 4 |
| Dropout | 0.35 | 0.30 |
| Patch-adapter rank | 4 | 4 |
| Batch size | 64 | 64 |
| Optimizer | AdamW | AdamW |
| Learning rate / weight decay | 1e-4 / 1e-2 | 1e-4 / 1e-2 |
| Gradient clip | 5 | 5 |
| Training modality dropout | 0.25 | 0.25 |

The result is the raw softmax argmax from one independently trained model per
seed. Reported means are arithmetic means of seed-level Accuracy, Macro-F1,
and Macro-AUROC. No CatBoost stage, external teacher, class offset,
checkpoint soup, or probability ensemble is part of the current method.
