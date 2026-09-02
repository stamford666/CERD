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

The current CGHC-v1 write-once builder additionally owns:

- `cghc_common6.svg`: CGHC-v1 versus the strongest of six comparator ensembles
  for each common-six metric; and
- `cghc_modality_association.svg`: conditional-generative component decision
  allocation across the three endpoint-specific disease strata.

The CGHC interpretation figure is generated only after both dataset replays
pass probability-replay, privacy, and receipt checks. Its disease-stratum
contrasts are descriptive fitted-model associations, not disease causes or
causal effects.
