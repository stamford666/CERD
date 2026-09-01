# CERD Method

## Overview

CERD addresses incomplete multimodal classification through one connected
decision process:

```text
reconstruct -> mark provenance -> estimate trust -> make a decision
```

The method has three substantive components: subject-conditioned latent
completion, provenance-aware multimodal representation, and reliability-aware
decision decomposition. A sparse mixture-of-experts (MoE) layer remains the
fusion implementation between representation and decision; it is retained in
all Method-revision experiments rather than treated as a separate add-on.

This repository is a core-method reference implementation, not the frozen
campaign runner. It implements the central Method switches described below but
does not claim exact numerical reproduction of the reported validation values.
Protected data definitions, fixed folds, fitted preprocessing assets, and
internal run orchestration are not released. Campaign-specific LRPA rank-4
patch adaptation, the ABCD presentation-axis and class-weighted quality
branch-auxiliary objectives, the ABCD fixed no-recompute dropped-combination
policy, and ADNI legacy image imputation and final-epoch refitting are also
outside this public reference.

## Problem formulation

For each subject, let \(x_m\) denote modality \(m\) and let \(o_m\in\{0,1\}\)
indicate whether that modality is physically observed. Each modality-specific
encoder maps an observed input to a sequence of latent tokens. Missing inputs
are never presented as if they had been measured: the model separately tracks
whether a latent sequence is observed, generated, or unavailable.

## Subject-conditioned latent completion

When modality \(m\) is missing but at least one other modality is observed, a
conditional cross-attention generator uses the current subject's observed
latent tokens as context and produces a token sequence for \(m\). The reported
revision trains this generator from stochastic observed subsets. For an
eligible training subject, a nonempty proper subset of the physically observed
modalities is sampled as context; the remaining observed modalities become
reconstruction targets. This better matches arbitrary incomplete contexts than
training completion only from fully observed inputs.

The reconstruction objective supervises both levels of the generated output:

- a pooled latent objective preserves subject-level semantic content; and
- normalized token-level Smooth-L1 supervision constrains the full generated
  token sequence.

The token term addresses the ambiguity of producing multiple tokens while
supervising only their mean. In the reported validation configuration, the
observed context-drop probability is 0.25 and the normalized token-loss weight
is 0.05.

After cross-attention and feed-forward refinement, the generator applies a
learned sigmoid output gate. A matched ablation bypassed this multiplication
without changing the constructed parameters or their initialization. The
predeclared rule for removing the gate failed, so the output gate is retained;
this is not a positive empirical-selection claim for the gate itself.

## Provenance-aware multimodal encoding

Observed and generated representations receive different learned provenance
embeddings before fusion. The resulting token streams are processed by a
Transformer with a sparse MoE feed-forward layer. The retained anchor uses one
fusion layer, 16 experts, one router, and top-4 routing. A later controlled
capacity screen also evaluates 8 experts with top-2 routing; both activate 25%
of experts per token. Router balancing is trained explicitly, while the
observed-modality pattern determines which specialization context is relevant.

MoE is not used to erase provenance. Completion supplies a usable latent
representation; provenance embeddings preserve how that representation was
obtained; sparse experts then model heterogeneous multimodal interactions.

## Reliability-aware decision decomposition

CERD separates two notions of trust:

1. **input reliability** describes whether a modality representation is
   observed, generated, or unusable and how trustworthy that input is; and
2. **predictive confidence** describes how concentrated a diagnostic branch's
   class distribution is.

The decision layer contains a joint branch, one branch per usable modality,
and pairwise branches. Pairwise features are formed in the shared latent space
using the two modality vectors, their elementwise product, and their absolute
difference. Branches that require unavailable information are masked out.

For branch probabilities \(p\) over \(C\) classes, the reported predictive
confidence is detached normalized-entropy confidence:

\[
q(p)=\max\!\left(1-\frac{H(p)}{\log C},10^{-3}\right).
\]

Unlike confidence computed from the unnormalized magnitude of class logits,
this quantity is invariant to adding a common constant to every class logit.
Input reliability, predictive confidence, and a low-capacity branch prior
determine the branch mixture weights. The final prediction is a weighted
mixture of branch class probabilities.

Reported branch weights should be described as **branch-associated mixture
mass**, not as causal modality importance. Token attention is an auxiliary
inspection signal rather than a guarantee of faithful attribution.

## Training objective

The training losses are organized by role rather than presented as independent
method modules:

\[
\mathcal L
=\mathcal L_{\mathrm{task}}
+\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+\lambda_{\mathrm{moe}}\mathcal L_{\mathrm{balance}}
+\lambda_{\mathrm{rob}}\mathcal L_{\mathrm{robust}}.
\]

Here, \(\mathcal L_{\mathrm{task}}\) is supervised classification;
\(\mathcal L_{\mathrm{rec}}\) combines pooled and token-level completion;
\(\mathcal L_{\mathrm{balance}}\) regularizes sparse routing; and
\(\mathcal L_{\mathrm{robust}}\) groups branch supervision, artificial
modality-drop consistency, and full-to-reduced-view distillation.

## Validation-evaluated structural configuration

The predeclared full revision uses:

```text
pattern-aware reconstruction        true
observed-context drop probability   0.25
normalized token-loss weight        0.05
branch confidence                    detached normalized entropy
generator output gate               retained
sparse MoE                           retained
```

The Method-revision evidence is validation-only. In the first seed, the full
revision improved pooled Macro-F1 over the legacy arm on both datasets. With
new seeds, ADNI improved again but ABCD's Macro-F1 difference changed sign.
Consequently, the predeclared shared confirmation rule was not met. The
two-seed arithmetic means are reported only as descriptive stability summaries
and do not establish significance or external generalization.

ABCD-only adaptive follow-ups did not replace this configuration. A seed-11
screen candidate that changed three settings together failed its frozen
seed-12 new-training-seed confirmation. A subsequent seed-13 local 2×3 screen
kept context dropout at 0.25, the output gate, and sparse MoE fixed while
crossing two detached entropy transforms with token-loss weights 0.05, 0.075,
and 0.10. Every non-anchor candidate reduced pooled Macro-F1. The decision was
`NO_PROMOTION_RETAIN_ANCHOR`, so no new-seed confirmation was launched from
that screen. These are validation-only outcomes on the same development cohort,
not evidence of external generalization.

A seed-14 one-factor balance screen also retained the anchor, so its
conditional seed-15 confirmation was skipped. The final capacity lineage
changed only sparse-MoE capacity: seed 16 compared the 16-expert/top-4 anchor
with an 8-expert/top-2 compact variant while preserving one router/fusion
layer, the output gate, every Method/balance setting, and the 25% active-expert
ratio. The compact variant passed the screen; its frozen seed-17 paired rerun
then passed all four confirmation checks. The decision was `CONFIRMED`. This
remains same-cohort validation-only new-seed stability evidence, not an
independent-data or significance claim.

See the [validation protocol](METHOD_REVISION_VALIDATION_PROTOCOL.md) and the
[de-identified common-six result](../results/method_revision_moe_common6.json)
for the reported arms, metrics, and decision rules.
