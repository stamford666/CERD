# CERD manuscript

This branch contains the current Information Fusion manuscript. Executable
preprocessing and training code is maintained on the repository's `main`
branch.

The current ABCD endpoint has three explicit baseline parent K-SADS groups:
low-symptom participants without an ADHD status, symptom-positive participants
without a recorded status, and current full ADHD. The 3,000-person cohort uses
a family-disjoint 2,141/430/429 split and one fixed label-independent mask with
approximately 15% incomplete participants in every split. CERD obtains
67.37 ± 1.07 Accuracy, 54.38 ± 0.81 Macro-F1, and 74.75 ± 0.92 Macro-AUROC,
leading the matched methods in Accuracy and Macro-F1.

On ADNI CN/MCI/AD, CERD obtains 65.72 ± 1.09 Accuracy, 64.56 ± 2.09 Macro-F1,
and 81.07 ± 0.55 Macro-AUROC. Every table reports the arithmetic mean and
sample standard deviation of three independently trained seeds; probabilities
are not ensembled.

The compiled 11-page paper is [`cas-sc-sample.pdf`](cas-sc-sample.pdf). The
self-contained submission package is
[`CERD_LaTeX_current.zip`](CERD_LaTeX_current.zip). Aggregate,
participant-free result receipts are retained in [`results/`](results/README.md).
