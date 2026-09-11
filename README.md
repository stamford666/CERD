# CERD manuscript

This branch contains the current Information Fusion manuscript. Executable
preprocessing and training code is maintained on the repository's `main`
branch.

The ABCD endpoint is binary: strict low-symptom non-ADHD controls versus
current full parent K-SADS ADHD. It contains 2,120 participants (1,272/848),
uses a family-disjoint 1,520/297/303 split, and applies the same label-independent
15% missing-modality manifest to every method. CERD obtains 78.44 ± 0.76
Accuracy, 77.81 ± 0.35 Macro-F1, and 85.12 ± 0.05 Macro-AUROC. On the 45
originally incomplete test participants, it ties the highest Accuracy and has
the highest Macro-F1 and Macro-AUROC among the matched methods.

On ADNI CN/MCI/AD, CERD obtains 65.72 ± 1.09 Accuracy, 64.56 ± 2.09 Macro-F1,
and 81.07 ± 0.55 Macro-AUROC. Every table reports the arithmetic mean and
sample standard deviation of three independently trained seeds; probabilities
are not ensembled.

The compiled 11-page paper is [`cas-sc-sample.pdf`](cas-sc-sample.pdf). The
self-contained submission package is
[`CERD_LaTeX_current.zip`](CERD_LaTeX_current.zip). Aggregate,
participant-free result receipts are retained in [`results/`](results/README.md).
