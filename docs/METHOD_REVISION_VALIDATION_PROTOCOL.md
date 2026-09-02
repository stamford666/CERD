# Final CERD evaluation and public-reporting protocol

## Scope

The public result contains only the final ADNI and ABCD experiments. Both use
the same six metrics:

1. Accuracy;
2. Balanced Accuracy;
3. Macro-F1;
4. Weighted-F1;
5. Macro-AUROC; and
6. Macro-AUPRC.

Both final endpoints are three-class tasks. ABCD uses the frozen strict
presentation endpoint on dev946 with five family-disjoint folds; the
manifest-driven binary ABCD-ADHD workflow in the core reference code is an
independent benchmark and contributes no result to this artifact. Macro-AUROC
uses macro one-vs-rest averaging and Macro-AUPRC averages one-vs-rest average
precision over every fixed class. The independent binary reference applies the
same rule over both fixed labels rather than reporting positive-class AP alone.

Macro-F1 is the primary classification metric. Binary-only quantities,
task-specific sensitivity/specificity, and results from earlier endpoints or
tuning lineages are excluded from the final table.

The sparse mixture-of-experts backbone remains part of full CERD. The public
reference implementation exposes the core Method but is not the protected
campaign runner. Exact data versions, fixed partitions, preprocessing receipts,
fold assignments, checkpoint hashes, and orchestration stay in the controlled
experiment record.

## Freeze and evaluation boundary

For each dataset, record the cohort, endpoint, partition, subject count, fold
count, clustering unit, candidate configuration, comparator configurations,
checkpoint ensemble rule, and statistical plan before final scoring. Selection
data may be used for tuning; any reused partition must be named plainly and
must not be presented as independent confirmation.

All methods in a paired comparison must use the same eligible participants,
target definition, modality inputs, preprocessing boundary, split, and metric
implementation. Missing-class behavior for Macro-AUROC and Macro-AUPRC is
fixed before evaluation. Probability ensembles, if used, are constructed
before computing any of the six pooled metrics.

The public aggregate artifact records the realized evaluation design and split
for both datasets. A result remains `NOT AVAILABLE` until every required field
is frozen and available.

The v3 public contract fixes ADNI to 1,480 subjects and five subject-level OOF
folds, and ABCD to 946 subjects, five family-disjoint OOF folds, and 922 family
resampling units. Both are adaptively reused development cohorts:
`evidence_scope=development_cv`, `partition_reused=true`, and
`selection_independent=false`. Consequently `confirmatory_support` is always
false for this release even if a descriptive paired comparison is significant.
The top-level claim boundary uses the fixed sentence: “Adaptive same cohort
development evidence only; both scored cohorts were reused for model and
configuration selection, so any Holm adjusted difference is descriptive and
confirmatory support is false.”

ADNI validation-318, test-318, and unassigned-910 are outside this campaign:
not selected into any arm, not iterated over, not scored, and not included in
any fitted statistic.
ABCD protected temporal internal holdout 850 is likewise outside this campaign:
not selected into any arm, not iterated over, not scored, and not included in
any fitted statistic.

The artifact also records an evidence scope, whether the scored partition was
reused during development, and whether model/configuration selection was
independent of that partition. A new training seed on the same participants is
same-cohort development replication, not independent-data confirmation.
`development_cv` and `reused_validation` results may be released descriptively,
but cannot support a confirmatory superiority flag. Only a locked evaluation,
locked internal holdout, or external test that is both non-reused and
selection-independent is eligible for that flag.

## Primary results and statistical comparisons

The primary table reports exactly one CERD row and one named comparator for
each dataset. Values are stored as fractions and rendered as percentages.
There is no threshold selection on final labels.
Each row also carries a public configuration ID, configuration SHA-256, and
execution-receipt SHA-256. These digests bind the reported row to its controlled
configuration and receipt without releasing a path. The paired comparison
carries the SHA-256 of its controlled analysis receipt.

Each dataset has exactly one one-sided paired Macro-F1 comparison on the exact
same evaluation observations. Each released comparison records:

- the two named methods, `paired=true`, and the primary Macro-F1 metric;
- the observed difference, with CERD minus comparator orientation;
- the one-sided 95% paired-bootstrap lower bound;
- the raw paired-swap p-value and jointly recomputed Holm-adjusted p-value;
- the fixed test, unit count, draw counts, RNG seeds, and alpha; and
- a separate confirmatory-support flag checked by the release validator.

The paired swap uses a separately initialized NumPy PCG64 stream per dataset,
50,000 draws, RNG seed 20260905, one swap bit per subject/family, an inclusive
one-sided tail, and plus-one Monte Carlo correction. The paired cluster
bootstrap uses a separate PCG64 stream per dataset, 20,000 draws, RNG seed
20260906, and the 0.05 percentile with NumPy's `linear` quantile method. Both
statistics recompute pooled fixed-three-class Macro-F1; they never average
unit-level F1 values. Holm correction is performed jointly across the ADNI and
ABCD primary comparisons. Significance is derived, rather than supplied, from
Holm-adjusted p-value `< 0.05`.

The confirmatory-support flag can be true only when CERD is the correctly
oriented `ours` row, the comparator is a comparator row, the tested metric is
the pre-specified primary Macro-F1, CERD's difference and bootstrap
lower bound are positive, the alternative is `greater`, the adjusted p-value
is below alpha, and the evidence boundary is eligible as defined above.
The validator also recomputes the reported difference directly from the two
public Macro-F1 values.

ABCD inference uses whole families as the resampling unit. ADNI uses subjects.
The bootstrap and paired swap operate on the same declared unit. A superiority
statement is made only for a pre-specified
comparison whose adjusted test meets alpha and whose effect direction is
positive; otherwise the result is described without a superiority claim.

## Pre-specified matched ablations

Each dataset reports all six metrics for every row below. Apart from the named
factor, the data split, seed/ensemble policy, optimization budget, prediction
aggregation, and evaluation code remain matched to full CERD.

| Public ID | Change from full CERD | Question isolated |
|---|---|---|
| `dense_backbone` | replace sparse MoE feed-forward fusion with its dense control | whether sparse expert routing contributes |
| `no_provenance` | remove observed/generated provenance embeddings | whether source marking contributes |
| `uniform_branch_weights` | replace reliability-aware weights with uniform valid-branch weights | whether trust-aware decision weighting contributes |
| `mean_pooling` | replace gated-attention pooling with mean pooling | whether learned token pooling contributes |
| `no_stochastic_context` | disable stochastic observed-subset context masking while retaining completion | whether context masking contributes separately from reconstruction |
| `no_completion` | disable latent completion under the matched missingness policy | whether reconstruction contributes |
| `no_mofe` | disable the more/fewer-modality objective | whether the robustness objective contributes |
| `no_output_gate` | bypass the generator output gate with aligned construction | whether the gate contributes |

The dense control is the only arm that removes MoE; all other rows retain the
same sparse-MoE configuration as full CERD. Ablation values are reported even
when they do not favor the full model.

The public control mapping is frozen in
[`configs/matched_ablations_v1.json`](../configs/matched_ablations_v1.json).
Named runs use `train.py --ablation-id ID --data-order-seed SEED`; fold-based
runs additionally require `--fold-id FOLD --split-receipt-sha256 SHA256`.
Checkpoint protocol metadata records the resolved eight-bit profile, ordered-ID
digest, safe split-receipt digest, and canonical configuration digest. The
output stem binds those values, and a pre-existing output is accepted only
when both its receipt and checkpoint exactly match. The loader/sampler seed is
separate from model RNG so module-construction differences cannot change
example order. The public protocol freezes that RNG scheme but does not record
epoch-order hashes and therefore makes no epoch-order hash-closure claim. For
`no_stochastic_context`, target selection and random draws are preserved and
only the context is expanded. For `no_completion`, generator and projector
modules remain constructed but generation and reconstruction are skipped. For
`no_mofe`, the reduced-view forward remains active while only the named rank
objective is excluded.

The dense arm shadow-constructs each corresponding sparse feed-forward block
to restore the full model's construction-RNG tail; all common parameters,
including alternating dense layers and every post-backbone module, therefore
start identically. This is construction alignment only. Sparse router noise is
drawn during the full model's forward pass and is absent from the dense arm, so
their global forward RNG streams are not promised to remain aligned. The
dedicated loader/sampler generator keeps sample order independent of that
difference.

## Aggregate interpretability

Explanation is generated by replaying the frozen final checkpoints without
changing predictions. For each dataset and both complete and incomplete input
conditions, release only:

- the number of evaluated observations;
- four normalized modality decision-allocation masses; and
- normalized mass grouped across joint, unimodal, and pairwise branches.

The artifact fixes `natural_disjoint` as its condition design and uses a
controlled public aggregate-source label. Complete and incomplete counts must
sum exactly to the evaluation subject count, with each
public cell containing at least 10 subjects. Each allocation sums to one within
absolute tolerance 1e-6. Each vector is the arithmetic mean of normalized
per-subject vectors within its natural condition; no row-level explanation is
accepted by the public builder. These values are descriptive routing/decision
allocation, not causal modality importance or proof of explanation faithfulness.

## Public artifact and privacy barrier

The repository does not ship a placeholder result JSON. The controlled builder
interface and exact private schemas are specified in
[`FINAL_RESULT_BUILDER.md`](FINAL_RESULT_BUILDER.md). After its input and
receipts are approved, create `results/final_results.json`, then render the
README and SVG figures with:

```bash
# The reviewed post-barrier finalizer internally invokes the bound builder and
# writes the FINAL-bound candidate to results/final_results.json.
python scripts/render_release_results.py --input results/final_results.json
python scripts/render_release_results.py --input results/final_results.json --check --require-final
```

The renderer rejects a final artifact missing either dataset, any common-six
metric, any required ablation, paired comparison metadata, or aggregate
interpretability values. It recomputes significance and rejects inconsistent
confirmatory-support claims.

The `--require-final` release gate accepts only the completed final artifact.
Plain text fields reject placeholder tokens, Markdown control characters,
local paths, and participant-like identifiers so a result cannot break the
generated tables or disclose protected information.

Never publish participant identifiers, labels, split membership, predictions,
probabilities, logits, embeddings, per-participant attributions, checkpoints,
protected paths, or run logs. Only validated aggregate JSON, generated tables,
and aggregate SVG figures cross the public-release boundary.
