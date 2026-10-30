# Normalization Ablation Report

Frozen split. Character n-gram baseline. Validation selects; test evaluated once.

| Configuration | Val macro-F1 | Val accuracy | Test macro-F1 | Status |
|---|---:|---:|---:|---|
| `raw` | 0.4545 | 0.4375 | 0.4 | selected|
| `unicode_whitespace` | 0.4545 | 0.4375 | not selected | |
| `protected_spans` | 0.4545 | 0.4375 | not selected | |
| `repetition` | 0.4545 | 0.4375 | not selected | |
| `variant_dictionary` | 0.4545 | 0.4375 | not selected | |
| `full` | 0.4545 | 0.4375 | not selected | |

Selected: `raw`.

## Interpretation

Preprocessing is a trade-off. Report slice deltas with the overall score; do not select on test results.
