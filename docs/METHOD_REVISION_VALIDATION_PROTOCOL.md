# Method-revision MoE validation protocol

## Scope and claim boundary

This protocol evaluates a revised CERD Method while retaining the sparse
mixture-of-experts (MoE) backbone in every arm. It is validation-only
development evidence: no test split or sealed holdout was accessed. The
reported values are descriptive pooled out-of-fold metrics, not evidence of
external generalization or statistical significance.

The public repository is a core-method reference implementation, not the
frozen campaign runner, and it does not support exact numerical reproduction
of these values. Its generic trainer and descriptive configuration records are
provided to expose the central Method switches rather than to reconstruct the
controlled internal execution environment.

The public result artifact contains no participant identifiers, split lists,
participant-level predictions, logits, checkpoints, local paths, or execution
logs.

The ABCD Method-revision endpoint is distinct from the 11,671-participant
binary ABCD-ADHD benchmark reported elsewhere in the README. It uses a
946-participant development cohort, ses-00A IGCB features, and a strict
three-class ses-01A ADHD-presentation target: inattentive,
hyperactive/impulsive, or combined. Its five folds are family-disjoint.

## Method question

The experiment tests four changes motivated by the Method review:

1. train latent completion from stochastic, nonempty observed subsets;
2. supervise the generated token sequence in addition to its pooled summary;
3. replace shift-sensitive evidence magnitude with detached normalized-entropy
   confidence; and
4. test whether the generator's post-cross-attention output gate is necessary.

The conceptual path is:

```text
reconstruct missing information
        -> mark observed/generated provenance
        -> estimate input reliability and predictive confidence
        -> combine joint, unimodal, and pairwise decisions
```

## Sparse-MoE anchor and controlled capacity variant

The initial Method-revision and balance-screen arms use the same anchor:

- one sparse fusion layer;
- 16 experts and one router;
- top-4 routing;
- router load-balancing weight 0.01; and
- observed/generated provenance embeddings and reliability-aware branch
  decomposition.

No dense replacement is evaluated. The final capacity screen changes only the
expert count and top-k to 8/top-2 while retaining one router/fusion layer, all
Method and balance settings, and the same 25% active-expert ratio. Thus MoE is
retained throughout, but “same backbone” applies only before this controlled
capacity comparison.

## Arms

| Arm | Stochastic subset | Token loss | Entropy confidence | Output gate |
|---|:---:|:---:|:---:|:---:|
| `c0_legacy` | no | no | no | yes |
| `c1_full` | yes | yes | yes | yes |
| `c2_full_minus_subset` | no | yes | yes | yes |
| `c3_full_minus_token` | yes | no | yes | yes |
| `c4_full_minus_entropy` | yes | yes | no | yes |
| `c5_full_minus_gate` | yes | yes | yes | no |

For stochastic-subset reconstruction, each eligible sample has at least two
physically observed modalities. A nonempty proper subset is retained as
context and the masked, physically observed modalities become reconstruction
targets. The context-drop probability is 0.25. The token objective is
normalized token-level Smooth-L1 with weight 0.05, added to the pooled
reconstruction objective.

For a branch probability vector \(p\) over \(C\) classes, predictive confidence
is

\[
q(p)=\max\!\left(1-\frac{H(p)}{\log C},10^{-3}\right),
\]

and is detached before branch weighting. This makes confidence invariant to a
common shift of all class logits.

The output-gate ablation bypasses the sigmoid multiplication while retaining
the gate parameters and their initialization. This keeps parameter names and
random-number consumption aligned between the gated and ungated arms.

## Dataset-specific fixed training detail

The ADNI runs retain their predeclared generator-only task-gradient path;
ABCD does not use that path. In both datasets the ordinary generator task
gradient is disabled. This dataset-specific distinction was fixed before the
Method-revision runs and is part of the reported internal configuration. A
reproduction attempt would need to match it, but this public release makes no
exact numerical reproduction claim.

## Validation design

The first ablation uses five validation folds with ABCD seed 9 and ADNI seed
26. The fixed-config confirmation repeats only `c0_legacy` and `c1_full` with
new seeds: ABCD seed 10 and ADNI seed 27. No aggregate ablation result was used
to add, remove, or reconfigure an arm after the predeclared grid was launched.
This grid-level statement does not describe within-run checkpoint selection.

The reported ADNI campaign used legacy image imputation and final-epoch
refitting. By contrast, the public generic trainer uses validation-selected
checkpointing and is not the campaign runner. Other campaign-specific details
outside the public reference are the LRPA rank-4 patch adapter and the
ABCD-specific presentation-axis and class-weighted quality branch-auxiliary
objectives with its fixed no-recompute dropped-combination policy.

Both datasets use exactly the same six metrics:

1. Accuracy;
2. Balanced Accuracy;
3. Macro-F1;
4. Weighted-F1;
5. Macro-AUROC; and
6. Macro-AUPRC.

Macro-F1 is the primary metric. Metrics are computed from pooled out-of-fold
predictions. The public artifact also reports arithmetic two-seed means as a
descriptive stability summary; these means are not a substitute for an
independent test or a significance analysis.

## Predeclared decisions

The fixed-config new-seed confirmation supports a shared primary claim only if
`c1_full` improves pooled Macro-F1 and does not reduce pooled Accuracy relative
to `c0_legacy` on both datasets.

The output gate is removed only if the no-gate arm has pooled Macro-F1 at least
as high as `c1_full` on both datasets, strictly higher on at least one dataset,
and pooled Accuracy at least as high on both datasets.

The new-seed confirmation did not satisfy the shared primary-claim rule:
ABCD's Macro-F1 difference changed sign at seed 10. The gate-removal rule also
failed because removing the gate reduced ABCD Macro-F1 and Balanced Accuracy.
The predeclared gate rule therefore retains the output gate. Sparse MoE remains
because every arm uses it; this protocol does not compare MoE with a dense
replacement and makes no dense-versus-MoE selection claim.

## Adaptive ABCD follow-ups

The adaptive follow-ups reuse the same 946-participant ABCD development cohort
and five family-disjoint folds. They provide validation-only tuning evidence,
not independent-data confirmation. Every candidate retains the output gate and
sparse MoE; only the final capacity stage changes expert count and top-k.

At seed 11, a six-configuration screen promoted `t4` for confirmation. The
candidate simultaneously changed context dropout from 0.25 to 0.10, token-loss
weight from 0.05 to 0.10, and confidence from detached entropy to detached
exponential entropy, so its screen result cannot identify a component-specific
effect. A frozen seed-12 rerun compared only `t4` with its anchor. It failed all
three confirmation checks: strict pooled Macro-F1 improvement, Accuracy delta
at least −0.005, and Macro-AUROC delta at least −0.005. The decision was
`NOT_CONFIRMED`, and `t4` was not adopted.

The seed-13 robustness screen therefore returned to the retained anchor and
held context dropout at 0.25. Its local 2×3 grid crossed two detached confidence
transforms (`entropy_detached` and `entropy_exp_detached`) with normalized token-
loss weights 0.05, 0.075, and 0.10. The 0.05 detached-entropy configuration was
the anchor. Promotion required every one of the following relative to that
anchor:

1. pooled Macro-F1 delta at least +0.003;
2. pooled Accuracy delta at least −0.005;
3. pooled Macro-AUROC delta at least −0.005; and
4. a strict Macro-F1 win in at least three of the five folds.

None of the five non-anchor candidates improved pooled Macro-F1, so no
candidate met all four conditions. The frozen decision was
`NO_PROMOTION_RETAIN_ANCHOR`, and the conditional new-seed confirmation was not
launched. This negative screen does not support a positive tuning or robustness
claim.

At seed 14, a six-arm one-factor balance screen kept the full Method and
16-expert/top-4 anchor fixed. It varied class-weight power, sampler power, or
training-only logit adjustment one at a time. No alternative met the same
promotion guardrails, so the decision was `NO_PROMOTION_RETAIN_ANCHOR`; the
conditional seed-15 confirmation was skipped.

At seed 16, the final capacity screen compared that exact anchor with an
8-expert/top-2 compact sparse MoE. Both arms kept a 25% active-expert ratio and
all non-capacity arguments were inherited unchanged. The compact arm met all
four promotion conditions. Its precommitted seed-17 exact10 rerun required all
of: strict compact-minus-anchor Macro-F1 improvement, Accuracy and Macro-AUROC
deltas each at least −0.005, and a seed-16/17 mean Macro-F1 delta at least
0.003. All checks passed and the decision was `CONFIRMED`.

All seed-14/16/17 values are pooled out-of-fold common-six validation metrics
on the same 946-participant family-disjoint development cohort. These stages
do not access test or sealed holdout data and are not independent-data
confirmation or a significance analysis.

## Public artifacts

- [Method narrative](METHOD.md)
- [De-identified common-six result](../results/method_revision_moe_common6.json)
- De-identified descriptive parameter records under [`configs/`](../configs/)

The parameter records are not consumed by the public `train.py` and are not a
complete executable campaign specification. Internal fold assignments,
protected manifests and fitted preprocessing assets, predictions, checkpoints,
run receipts, logs, and campaign orchestration remain private.
