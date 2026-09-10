# Result receipts

Only participant-free aggregate receipts are published here. Raw
data, participant identifiers, per-participant predictions, logs, and model
checkpoints are excluded.

## ABCD v4 matched main comparison

[`abcd_adhd_presentation3_snp_missing15_v4.md`](abcd_adhd_presentation3_snp_missing15_v4.md)
and its [JSON receipt](abcd_adhd_presentation3_snp_missing15_v4.json) contain the
main comparison. CERD and all six baselines use the same features, participants,
family-disjoint split, fixed 15% missingness table, seed set, and evaluation
rule. CERD obtains 59.24 ± 1.65 Accuracy, 53.62 ± 1.40 Macro-F1, and
74.43 ± 1.03 Macro-AUROC, the highest three-seed mean in that matched table.
The structural variables are T1 regional signal intensities, not regional
volumes; this naming correction applies equally to every method and does not
change comparison fairness.

## ABCD v5

[`abcd_adhd_presentation3_feature_refinement_v5.md`](abcd_adhd_presentation3_feature_refinement_v5.md)
and its [JSON receipt](abcd_adhd_presentation3_feature_refinement_v5.json)
record the corrected structural-imaging representation, one-PC-per-candidate-
gene genetics, added task-performance variables, seed-level metrics, and both
evaluation protocols.

- validation-selected seed checkpoints: 57.71 ± 1.56 Accuracy,
  52.66 ± 1.25 Macro-F1, 73.63 ± 0.87 Macro-AUROC;
- fixed 14-epoch development-pool refit: 58.76 ± 2.20 Accuracy,
  55.49 ± 1.29 Macro-F1, 75.38 ± 0.76 Macro-AUROC.

V5 is a separate feature-representation study using true structural volumes
and thickness, gene-wise PCA, and additional task-performance variables.
Baselines have not yet been rerun on v5, so the v4 baseline rows are not reused
as if they were matched v5 comparisons.

## ADNI

[`adni_direct_three_seed_mean_v1.md`](adni_direct_three_seed_mean_v1.md) and its
[JSON receipt](adni_direct_three_seed_mean_v1.json) report the direct CERD
evaluation using seeds 0/1/2. The result is 65.30 ± 1.79 Accuracy,
64.48 ± 0.99 Macro-F1, and 80.81 ± 0.56 Macro-AUROC.

Every aggregate is the arithmetic mean of seed-level metrics. The previously
shown 68.24% value was obtained by averaging three models' class probabilities
before taking argmax; it answers a different ensemble question and is not part
of the current reporting rule.

The separate [`adni_matched_formal_v3.md`](adni_matched_formal_v3.md) and
[JSON receipt](adni_matched_formal_v3.json) provide the complete seven-method
matched table. In that frozen campaign, CERD records 64.47/64.26/80.57; the
later 65.30 direct run is not silently substituted into the older matched row.
