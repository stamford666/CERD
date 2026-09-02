# Public result artifact

## Current CGHC-v1 result bundle

`cghc_v1.json` is the current model-comparison artifact. It reports the same
six metrics for ADNI and ABCD, six three-seed comparator ensembles, paired
Macro-F1 and secondary Accuracy inference, the retained conditional-generation
boundary, and aggregate-only component interpretation. Its paired evidence is
retrospective fixed-candidate evidence, not a pristine new confirmatory test.

The artifact distinguishes two levels that must not be merged:

- the final CGHC-v1 hierarchical consensus, whose last fusion rule is not an
  MoE; and
- the separate exact-500 development-OOF matched campaign, which supplies
  component controls such as `no_completion` but is not a matched ablation of
  the final heterogeneous consensus.

Public interpretation contains only validation-cohort aggregates and
cluster-bootstrap intervals. Modality allocation and disease-stratum
differences are descriptive fitted-model associations, never estimates of
disease etiology or causal effects. `cghc_v1_receipt.json` binds the JSON and
the two CGHC SVG figures. The write-once builder is
[`scripts/build_cghc_release.py`](../scripts/build_cghc_release.py).

## Earlier matched-component artifact

`final_results.json` is created only after the experiments, comparisons, and
public-release boundary are frozen. The repository does not ship a placeholder
result JSON. The final artifact contains aggregate values only.

The low-level private-to-public construction primitive is documented in
[`docs/FINAL_RESULT_BUILDER.md`](../docs/FINAL_RESULT_BUILDER.md). It consumes
one explicit, hash-bound manifest and shared row-aligned OOF inputs, validates
configuration/execution/analysis/aggregation receipts, and atomically writes
only `final_results.json`. For a formal release it is called only inside the
reviewed post-barrier finalizer, which binds the candidate bytes in its private
statistics FINAL and verifies that the repository copy is identical. The
standalone builder CLI is not release authorization and must not be used to
publish campaign results.

Each dataset must declare its evidence scope, whether the scored partition was
reused during development, and whether selection was independent of it. The
renderer distinguishes a statistically significant difference from
confirmatory superiority. Same-cohort development or reused-validation
results can be reported descriptively, but can never receive the latter flag.

Each dataset contains exactly two common-six rows, ordered as CERD and its one
pre-specified comparator, followed by exactly one paired Macro-F1 comparison.
The public artifact uses exactly these six metrics, stored as fractions in
`[0, 1]`: Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, Macro-AUROC, and
Macro-AUPRC. It also carries the pre-specified matched ablations, paired statistical
comparisons, and aggregate decision-allocation summaries for ADNI and ABCD.
Both final endpoints are three-class tasks. The frozen ABCD result is the
strict presentation endpoint on dev946 with five family-disjoint folds; the
manifest-driven binary ABCD-ADHD workflow is an independent public reference
benchmark and is not a source for this artifact. Macro-AUPRC is the macro mean
of one-vs-rest average precision over every fixed class, including both fixed
labels when the independent binary reference is run.
Complete and incomplete explanation conditions include a plain-language
definition so naturally missing inputs cannot be conflated with a frozen
stress-test mask.

The eight required matched ablations treat stochastic observed-subset context
masking and latent completion as separate factors. ADNI validation-318, test-318, and
unassigned-910 are outside this campaign: not selected into any arm, not
iterated over, not scored, and not included in any fitted statistic.
ABCD protected temporal internal holdout 850 is also outside this campaign and
is not selected, iterated, scored, or included in a fitted statistic.
These ablations are validation controls for the four-stage decision chain, not
eight coequal CERD modules. The dense-FFN row is specifically a
backbone-sensitivity control; canonical artifact order is retained only for
stable validation and rendering.

The final v3 artifact locks ADNI to 1,480 subjects and five subject-level OOF
folds, and ABCD to 946 subjects, five family-disjoint OOF folds, and 922 family
resampling units. Both scored cohorts are adaptive reused-development evidence,
so `partition_reused=true`, `selection_independent=false`, and every
`confirmatory_support` value is false. Each primary row includes a public
configuration ID plus SHA-256 digests for its configuration and controlled
execution receipt; the eight ablation rows carry the same provenance fields,
and each paired comparison includes its analysis-receipt digest. The top level
commits to the exact source manifest and interpretability carries its verified
aggregation-receipt digest. No local receipt path is released.
The top-level claim boundary is fixed to: “Adaptive same cohort development
evidence only; both scored cohorts were reused for model and configuration
selection, so any Holm adjusted difference is descriptive and confirmatory
support is false.”
Interpretability uses `natural_disjoint` only. The ordered modality names are
ADNI: image, genomic, clinical, biospecimen; and ABCD: imaging, genetic,
cognition and health, behavior and environment.
The reported vectors are subject-level arithmetic means of per-subject
normalized vectors; natural complete and incomplete definitions and counts are
frozen by the validator.

For isolated validation and development, the low-level builder can construct an
aggregate without rendering any other public file:

```bash
python scripts/build_final_results.py \
  --manifest /controlled/final-build-manifest.json \
  --output results/final_results.json
```

Formal campaign publication uses the reviewed post-barrier finalizer instead;
before push, the repository JSON must match its FINAL-bound candidate byte for
byte.

Generate the README tables and SVG figures with:

```bash
python scripts/render_release_results.py --input results/final_results.json
```

Use `--check --require-final` in the public-release CI gate to reject malformed,
incomplete, unsynchronized, or placeholder artifacts. The paired swap tests
use whole clusters, NumPy PCG64, 50,000 draws, RNG seed 20260905, an inclusive
tail, and plus-one correction. The one-sided 95% paired-cluster bootstrap lower
bounds use 20,000 draws, PCG64 seed 20260906, and the 0.05 `linear` quantile.
Holm adjustment is recomputed jointly across the ADNI and ABCD primary
Macro-F1 comparisons. Plain text fields reject paths, Markdown control
characters, and placeholder tokens.
Participant identifiers, labels, per-participant predictions, logits, fold
assignments, checkpoints, and local paths must never be added here.
