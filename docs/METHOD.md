# CERD Method

## Overview

CERD addresses incomplete multimodal classification through one connected
decision process:

```text
reconstruct -> mark provenance -> estimate trust -> make a decision
```

These four stages are the only conceptual organization of CERD. Marked latent
representations pass through a retained Transformer–sparse-MoE fusion
backbone between provenance encoding and trust estimation. The backbone is an
implementation substrate, not an additional CERD stage or contribution.

This document describes the core method. Endpoint-specific optimization,
controlled campaign configuration, matched-control execution, random-number
alignment, and public reporting rules are validation details documented in the
linked protocol rather than components of the method.

## Problem formulation

For each subject, let \(x_m\) denote modality \(m\) and let \(o_m\in\{0,1\}\)
indicate whether that modality is physically observed. Each modality-specific
encoder maps an observed input to a sequence of latent tokens. Missing inputs
are never presented as if they had been measured: the model separately tracks
whether a latent sequence is observed, generated, or unavailable.

## Stage 1: Reconstruction

When modality \(m\) is missing but at least one other modality is observed, a
conditional cross-attention generator uses the current subject's observed
latent tokens as context and produces a token sequence for \(m\). The final
Method specification trains this generator from stochastic observed subsets.
For an eligible training subject, a nonempty proper subset of the physically
observed modalities is sampled as context; the remaining observed modalities
become reconstruction targets. This better matches arbitrary incomplete
contexts than training completion only from fully observed inputs.

The reconstruction objective supervises both levels of the generated output:

- a pooled latent objective preserves subject-level semantic content; and
- normalized token-level Smooth-L1 supervision constrains the full generated
  token sequence.

The token term addresses the ambiguity of producing multiple tokens while
supervising only their mean. Context-drop probability and the normalized
token-loss weight are frozen before final evaluation and recorded with the
private campaign receipt; this reference implementation exposes both switches.

After cross-attention and feed-forward refinement, a learned sigmoid output
gate modulates the generated tokens. This gate is an implementation detail of
reconstruction, not a separate stage in the decision chain.

## Stage 2: Provenance marking

Observed and generated representations receive different learned provenance
embeddings before fusion. The marked token streams then pass through the
retained Transformer–sparse-MoE fusion backbone. CERD's contribution at this
stage is preserving source identity: the backbone never turns a generated
representation into an apparently observed one. Sparse expert routing is
inherited backbone machinery and is evaluated only as an implementation
sensitivity control.

## Stage 3: Trust estimation

The retained fusion backbone first produces candidate class distributions for
a joint view, each usable unimodal view, and each usable pairwise view.
Pairwise features combine the two modality vectors, their elementwise product,
and their absolute difference in the shared latent space. These candidates are
the objects whose trust is estimated; they are decision views, not independent
method modules.

CERD separates two notions of trust:

1. **input reliability** describes whether a modality representation is
   observed, generated, or unusable and how trustworthy that input is; and
2. **predictive confidence** describes how concentrated a diagnostic branch's
   class distribution is.

For branch probabilities \(p\) over \(C\) classes, the final Method uses
detached normalized-entropy confidence:

\[
q(p)=\max\!\left(1-\frac{H(p)}{\log C},10^{-3}\right).
\]

Unlike confidence computed from the unnormalized magnitude of class logits,
this quantity is invariant to adding a common constant to every class logit.
Input reliability, predictive confidence, and a low-capacity branch prior
together define the trust assigned to each valid candidate prediction.

## Stage 4: Decision aggregation

Candidates that require unavailable information are masked out. Their trust
scores are normalized over the remaining valid set, and the final prediction
is the resulting weighted mixture of class probabilities.

Any reported weights should be described as
**branch-associated mixture mass**, not as causal modality importance. Token
attention is an auxiliary inspection signal rather than a guarantee of
faithful attribution.

## Frozen heterogeneous consensus

The decision above is the output of one conditional-generative neural member.
The final CGHC-v1 predictor adds one fixed cross-model aggregation step. For
each member family, class probabilities are combined coordinate-wise across
three seeds; the family summaries are then combined coordinate-wise and
normalized once onto the probability simplex. This hierarchy prevents a family
with more checkpoints from receiving more weight solely because it has more
members.

ABCD uses three independently trained families: conditional-generative
Rank-MoE, CatBoost, and TabM-32. ADNI uses four conditional-generative neural
variants, each with three seeds. Thus sparse MoE remains part of the method's
conditional-generative neural path, but the final consensus is deliberately a
simple fixed median rather than another learned MoE gate. Any fixed class
decision offset is selected from validation only and changes the hard decision,
not the reported probability vector.

This consensus should not be confused with the within-member branch weights.
Branch weights describe a neural member's decision allocation; cross-model
medians provide robustness across fitted systems. Neither quantity establishes
that a modality causes disease. Modality summaries are reported as descriptive
predictive allocation or observational association, with an explicit non-causal
boundary.

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
The balance term is training regularization for the retained backbone, not a
fifth method stage. The grouped robustness term is an auxiliary constraint
across missing-data views, not an independent decision module.

## Implementation and evaluation scope

The public repository implements the four-stage method and its verification
controls, but it is not the controlled campaign runner. Dataset definitions,
fixed partitions, fitted preprocessing assets, and internal orchestration are
not released. Private campaigns may additionally bind endpoint-specific
optimization and implementation details that are outside this core reference,
including low-rank patch adaptation, ABCD-specific presentation-axis and
class-weighted branch objectives, a fixed dropped-combination policy, and
ADNI-specific image imputation or refitting rules. Such details must be stated
in the corresponding frozen campaign receipt and must not be inferred from the
public trainer. Matched-control mechanics, random-number alignment, the shared
six-metric reporting boundary, and private-to-public artifact rules are kept in
the [evaluation protocol](METHOD_REVISION_VALIDATION_PROTOCOL.md) and
[aggregate artifact contract](../results/README.md).
