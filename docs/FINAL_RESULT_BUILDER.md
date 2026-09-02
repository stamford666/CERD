# Controlled final-result builder

`scripts/build_final_results.py` is the low-level private-to-public construction
primitive for the schema-v3 aggregate. It performs no campaign discovery and
has no default input or output. Its successful exit validates content, but does
not authorize a formal release. The formal campaign invokes these functions
inside its reviewed post-barrier finalizer, binds the exact public candidate in
the private statistics FINAL, and exports those same bytes to the repository.
Do not invoke the standalone CLI to publish a formal result.

For isolated validation and development only, the low-level invocation is:

```bash
python scripts/build_final_results.py \
  --manifest /controlled/final-build-manifest.json \
  --output results/final_results.json
```

The formal release procedure must use the reviewed finalizer rather than this
standalone command. The output must be named `final_results.json`. An existing regular artifact is
not replaced unless `--replace` is explicit. The builder writes only that JSON;
README tables and SVGs remain a separate renderer step.

## Two-phase boundary

`compute_private_result(manifest_path)` reads and verifies controlled inputs and
returns an in-memory `PrivateBuildResult`. `validate_public_result(result)` then
applies the public schema and privacy allowlist and returns canonical JSON
bytes. `atomic_write_public_result(...)` writes validated bytes through a
same-directory temporary file, re-reads and validates them, flushes both file
and directory state, and publishes atomically.

No participant identifier, cluster identifier, label, fold, probability,
private path, or receipt content crosses this boundary. The public artifact
contains aggregate common-six values and receipt/configuration digests only.
Its top-level `source_manifest_sha256` commits to the exact private manifest
bytes without exposing that manifest.

## Private manifest v1

The manifest schema is `cerd-private-final-builder-v1` and has exactly
`schema`, `generated_at`, and `datasets`. `datasets` contains exactly `adni`
and `abcd`. Every relative file path is resolved against the manifest
directory, and every referenced file is read and checked against its declared
lowercase SHA-256.

Each dataset declares:

- the exact subject, fold, resampling-unit name, and resampling-unit count;
- one explicit row-aligned NPZ path and SHA-256;
- fixed class columns `[0, 1, 2]` and an ordered, unique seed-ID list;
- primary rows ordered as `full`/ours and `comparator`/comparator;
- the eight canonical ablations in renderer order;
- an analysis-receipt path and SHA-256; and
- subject-level interpretability aggregation semantics plus an aggregate-
  receipt path and SHA-256.

Every primary or ablation row supplies a public-safe configuration ID, the
actual configuration path/SHA-256, and the actual execution-receipt
path/SHA-256. Primary rows additionally supply the public method and role.

```text
dataset:
contract, input_npz, input_sha256, class_columns, seed_ids, primary,
ablations, analysis_receipt_path, analysis_receipt_sha256, interpretability

primary row:
input_id, method, role, configuration_id, configuration_path,
configuration_sha256, execution_receipt_path, execution_receipt_sha256

ablation row:
id, configuration_id, configuration_path, configuration_sha256,
execution_receipt_path, execution_receipt_sha256

interpretability manifest entry:
aggregation_unit, aggregation_method, aggregation_receipt_path,
aggregation_receipt_sha256
```

## Row-aligned OOF v1

The NPZ scalar `format` is `cerd-private-row-aligned-oof-v1`; scalar `dataset`
is `adni` or `abcd`. With `allow_pickle=False`, its remaining arrays are
exactly:

```text
row_ids                       [subjects]
cluster_ids                   [subjects]
fold_ids                      [subjects]
labels                        [subjects]
seed_ids                      [seeds]
class_columns                 [3]
probabilities_full            [seeds, subjects, 3]
probabilities_comparator      [seeds, subjects, 3]
probabilities_<ablation-id>   [seeds, subjects, 3], for all eight ablations
```

All ten arms therefore share one row, target, fold, and cluster alignment.
Each seed-level probability row must be finite, lie in `[0, 1]`, and sum to
one within `1e-6`. The builder takes the arithmetic mean over the manifest seed
order before computing any metric. ADNI must have 1,480 unique subject
clusters and five nonempty folds. ABCD must have 946 rows, 922 family clusters,
and five nonempty folds; a family cannot cross folds.

## Structured receipt bindings

Execution receipts use `cerd-private-execution-receipt-v1`. They bind the
dataset and arm to the verified configuration digest, OOF NPZ digest,
probability-array name, class columns, seed IDs, equal-weight ensemble rule,
cohort/fold contract, resampling contract, and `status=complete`.

```text
schema, dataset, arm_id, configuration_sha256, oof_input_sha256,
probability_array, seed_ids, class_columns, seed_ensemble, subjects, folds,
resampling_unit, resampling_units, status
```

Analysis receipts use `cerd-private-analysis-receipt-v1`. They bind the OOF
digest and ensemble contract to the full-versus-comparator Macro-F1 analysis:
50,000 cluster swaps with PCG64 seed 20260905, 20,000 paired cluster bootstrap
draws with PCG64 seed 20260906, and the 0.05 quantile using NumPy's `linear`
quantile method. The builder recomputes and matches the observed difference,
inclusive plus-one one-sided p-value, and bootstrap lower bound to absolute
tolerance `1e-12`. Holm adjustment is then recomputed jointly across ADNI and
ABCD.

```text
schema, dataset, oof_input_sha256, class_columns, seed_ids, seed_ensemble,
resampling_unit, n_units, comparison

comparison:
metric, ours_input_id, comparator_input_id, alternative, swap_draws,
swap_rng_seed, bootstrap_draws, bootstrap_rng_seed, bootstrap_quantile,
bootstrap_quantile_method, observed_delta, p_value, bootstrap_lower_bound
```

Aggregate interpretability receipts use
`cerd-private-aggregation-receipt-v1`. They bind the same OOF digest to
`natural_disjoint`, subject-level arithmetic means of per-subject normalized
vectors, the frozen cohort size, and exactly two privacy-safe aggregate cells:
natural complete and natural incomplete. Only their counts, four modality
allocation means, and three branch-mass means are accepted. Counts must sum to
the cohort and each cell must contain at least ten subjects; vectors must sum
to one within `1e-6`.

```text
schema, dataset, oof_input_sha256, aggregation_unit, aggregation_method,
condition_design, subjects, conditions

each complete/incomplete condition:
subjects, decision_allocation, branch_mass
```

## Statistical definition

The paired statistic is pooled fixed-three-class Macro-F1 for the full model
minus the comparator. A swap draw uses one Bernoulli bit per subject (ADNI) or
family (ABCD) and swaps every member of that unit together. Its one-sided value
is `(1 + count(permuted_delta >= observed_delta)) / 50001`.

A bootstrap draw samples the declared number of clusters with replacement,
applies identical multiplicities to both methods, and recomputes pooled
fixed-class Macro-F1. It does not average cluster-level F1 values. A bootstrap
sample that omits a class still assigns that fixed class an F1 of zero.

## Deliberate limitation

The public repository has no protected cohort allowlist and cannot independently
recognize participant identity. Structural cohort/fold checks plus the exact
OOF digest and its controlled analysis/aggregation receipt bindings are the
approval boundary. The controlled system must approve those receipts before
invocation. Changing the cohort while regenerating every receipt is a new
controlled analysis, not something this public builder can authorize.
