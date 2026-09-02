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
fusion implementation between representation and decision. It is retained in
full CERD and every non-backbone ablation; only the explicit dense-backbone
control replaces it.

This repository is a core-method reference implementation, not the frozen
campaign runner. It implements all eight matched-control mechanics described
below but
does not claim exact numerical reproduction of the private campaign results.
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

After cross-attention and feed-forward refinement, the generator applies a
learned sigmoid output gate. Its matched ablation bypasses this multiplication
without changing the constructed parameters or their initialization, keeping
the comparison aligned apart from the named factor.

The `no_stochastic_context` arm first performs the identical stochastic draw
and preserves the identical reconstruction targets, then expands only each
target's context to all other observed modalities. The `no_completion` arm
still constructs the generators and reconstruction projectors in their normal
order, but executes neither missing-latent generation nor reconstruction.

## Provenance-aware multimodal encoding

Observed and generated representations receive different learned provenance
embeddings before fusion. The resulting token streams are processed by a
Transformer with a sparse MoE feed-forward layer. Expert count and top-k are
frozen together with the final training configuration. Router balancing is
trained explicitly, while the observed-modality pattern determines which
specialization context is relevant.

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

For branch probabilities \(p\) over \(C\) classes, the final Method uses
detached normalized-entropy confidence:

\[
q(p)=\max\!\left(1-\frac{H(p)}{\log C},10^{-3}\right).
\]

Unlike confidence computed from the unnormalized magnitude of class logits,
this quantity is invariant to adding a common constant to every class logit.
Input reliability, predictive confidence, and a low-capacity branch prior
determine the branch mixture weights. The final prediction is a weighted
mixture of branch class probabilities.

For the pooling and branch-weight controls, the attention and reliability
modules remain constructed and their normal score paths are evaluated. The
selected pooled feature is replaced by the token mean, or the final valid
branch mixture is replaced by a uniform distribution, respectively.

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

## Final reporting boundary

The public result reports ADNI and ABCD with one shared metric vocabulary:
Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, Macro-AUROC, and
Macro-AUPRC. No earlier tuning lineage or incompatible endpoint table is mixed
into that artifact.
Both final rows use three-class endpoints. In particular, the final ABCD row is
the frozen strict presentation endpoint on dev946 with five family-disjoint
folds. The manifest-driven binary ABCD-ADHD workflow in the public reference
code is an independent benchmark and contributes no value to the final table.

The pre-specified matched ablations cover dense FFN versus sparse MoE, provenance
marking, reliability-aware branch weighting, gated-attention pooling,
stochastic observed-subset context masking, latent completion, the
more/fewer-modality objective, and the generator output gate. Context masking
and completion are tested as separate factors.
The dense arm is the sole arm whose fusion stack contains no sparse expert
layer. The `no_mofe` arm retains the same artificial reduced-view forward pass,
classification, and distillation paths but excludes the more/fewer-modality
rank objective from the optimized sum.

The canonical ID-to-control mapping and its order digest live in
[`configs/matched_ablations_v1.json`](../configs/matched_ablations_v1.json).
Invoke a row with `--ablation-id ID` on an otherwise identical base command.
`--data-order-seed` is stored in checkpoint protocol metadata and drives
loader/sampler generators that are independent of model initialization.
This freezes the RNG scheme for example order only; the public runner neither
records nor claims an epoch-order hash closure. Dense construction consumes a
shadow sparse initialization and restores its RNG tail, so every common
parameter (including post-backbone modules) has the full-model initialization.
At forward time, however, the sparse `NoisyGate` consumes router noise while
the dense control does not. Forward-pass dropout, modality dropout, and
reconstruction sampling can consequently diverge between those two arms. The
public runner does not claim full forward-RNG alignment or reproduction of the
private campaign's fold/job RNG and orchestration receipts.
Aggregate explanation reports modality decision allocation and grouped
joint/unimodal/pairwise branch mass for complete and incomplete inputs. These
quantities are descriptive routing summaries, not causal importance.

See the [evaluation protocol](METHOD_REVISION_VALIDATION_PROTOCOL.md) and the
[aggregate artifact contract](../results/README.md). Until the final campaign
is frozen, the README and figures deliberately render `NOT AVAILABLE` rather
than reusing values from an earlier experiment.
