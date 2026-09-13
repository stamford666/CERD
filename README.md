# CERD manuscript

This branch contains the current Information Fusion manuscript. Executable
preprocessing and training code is maintained on the repository's `main`
branch.

The current ABCD endpoint is an ordered cross-sectional parent K-SADS severity
task with non-overlapping 0--2, 3--5, and 6--9 symptom intervals plus clinical
status, impairment, and onset/cross-setting conditions. The 3,000-person cohort
uses a family-disjoint 2,134/438/428 split and one fixed label-independent mask
with approximately 15% incomplete participants in every split. CERD obtains
66.82 ± 2.14 Accuracy, 55.18 ± 1.16 Macro-F1, and 77.90 ± 0.55 Macro-AUROC,
leading the matched methods on all three metrics.

Matched ABCD controls keep Full CERD highest on all three reported metrics.
Conditional completion contributes most strongly on incomplete participants,
the multigranular readout gives the largest overall accuracy gain, and CERD
outperforms both one-expert and capacity-aligned Dense-FFN controls. A strict
removal audit identifies direct QC/LD-pruned SNP dosage as the strongest
functional dependency for the ADHD-severity decision.

On ADNI CN/MCI/AD, CERD obtains 65.72 ± 1.09 Accuracy, 64.56 ± 2.09 Macro-F1,
and 81.07 ± 0.55 Macro-AUROC. Every table reports the arithmetic mean and
sample standard deviation of three independently trained seeds; probabilities
are not ensembled.

The compiled 11-page paper is [`cas-sc-sample.pdf`](cas-sc-sample.pdf). The
self-contained submission package is
[`CERD_LaTeX_current.zip`](CERD_LaTeX_current.zip). Aggregate,
participant-free result receipts are retained in [`results/`](results/README.md).
