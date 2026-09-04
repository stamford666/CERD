# ABCD 25% random-missing ablations

These are three-seed, validation-only component ablations under the fixed ABCD
random-missing protocol. They are descriptive development evidence and do not
replace or modify the repository's existing primary experiment results.

The manifest contains 1,745 participants. A fixed set of 436 participants
(24.9857%) loses a random non-empty proper subset of the four modalities; no
participant loses all four modalities. Every arm uses the same mask. The
row-aligned validation evaluation contains 1,692 participants, including 423
with simulated missing modalities.

| Arm | Seeds | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full CERD | 3 | **61.47** | **55.97** | **55.81** | **60.85** | 75.46 | 59.46 |
| Without latent completion | 3 | 60.64 | 54.89 | 54.49 | 59.78 | 75.45 | 59.43 |
| Dense backbone | 3 | 60.76 | 54.63 | 54.02 | 59.59 | **75.68** | **59.78** |
| Without provenance-aware routing | 3 | 61.11 | 55.25 | 54.97 | 60.28 | 75.52 | 59.50 |
| Uniform branch weights | 3 | 61.17 | 55.35 | 55.11 | 60.38 | 75.54 | 59.51 |

Changes relative to Full CERD, in percentage points:

| Removed/replaced component | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Macro-AUROC | Macro-AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Latent completion | -0.83 | -1.08 | -1.32 | -1.07 | -0.02 | -0.03 |
| Sparse MoE backbone | -0.71 | -1.34 | -1.79 | -1.25 | +0.22 | +0.32 |
| Provenance-aware routing | -0.35 | -0.72 | -0.84 | -0.56 | +0.06 | +0.04 |
| Learned branch weights | -0.30 | -0.61 | -0.70 | -0.46 | +0.08 | +0.05 |

The full model is strongest on Accuracy, Balanced Accuracy, Macro-F1, and
Weighted-F1. The dense control is slightly higher on ranking metrics but loses
1.79 percentage points of Macro-F1. No confirmatory significance claim is made
from this reused development boundary.

The machine-readable artifact is
[`abcd_random_missing25_ablations_v1.json`](abcd_random_missing25_ablations_v1.json).
