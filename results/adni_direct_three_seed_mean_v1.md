# ADNI direct CERD result: three-seed arithmetic mean

Each seed is independently trained, checkpoint-selected, and evaluated. The
reported aggregate is the arithmetic mean and sample standard deviation of
the three seed-level metrics, not a probability ensemble.

| Seed | Best epoch | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |
|---:|---:|---:|---:|---:|
| 0 | 31 | 64.78 | 64.33 | 80.35 |
| 1 | 33 | 67.30 | 65.54 | 80.65 |
| 2 | 30 | 63.84 | 63.58 | 81.43 |
| Mean ± SD | — | **65.30 ± 1.79** | **64.48 ± 0.99** | **80.81 ± 0.56** |

The previously calculated 68.24% accuracy came from averaging the three
models' predicted class-probability vectors for every participant and then
taking a single argmax. Ensemble voting can correct different errors across
seeds, so its accuracy need not equal the average of the three accuracies. It
is excluded here to keep CERD and all baselines on the same three-seed-mean
reporting rule.
