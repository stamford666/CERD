# Aggregate result figures

The release renderer owns three SVG files:

- `common6.svg`: the six primary metrics for ADNI and ABCD;
- `ablations.svg`: Macro-F1 change from the full CERD configuration; and
- `decision_allocation.svg`: complete/incomplete modality allocation and
  grouped joint/unimodal/pairwise branch mass.

No placeholder SVG is committed. The three figures are generated only after
`results/final_results.json` is approved and passes the final-release gate; no
figure should be assembled from participant-level public files. Decision
allocation is descriptive routing mass, not causal feature importance.
