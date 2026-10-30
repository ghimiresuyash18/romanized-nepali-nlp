# Phase 5 Frozen Sentiment Baseline

**Frozen Date**: 2026-10-30  
**Dataset**: `tests/fixtures/romanized_examples.jsonl` (N=106 records)  
**Dataset Hash**: `44aa844b2488a092`  
**Model Architecture**: Character n-gram TF-IDF (`char_wb`, range 2-5) + Logistic Regression (`class_weight='balanced'`)  

## Frozen Metrics (Held-out Test Split, N=15)

- **Accuracy**: 0.5333 (95% CI: [0.3333, 0.7333])
- **Macro-F1**: 0.4444 (95% CI: [0.2444, 0.6556])
- **Weighted-F1**: 0.4667

## Artifact Manifest

- `split_manifest.json`: Frozen train (75), validation (16), and test (15) record IDs.
- `config.json`: Pinned baseline model hyperparameters.
- `metrics.json`: Detailed overall, per-class, and challenge-slice evaluation.
- `predictions.jsonl`: Bitwise predictions on test split with confidence scores.

## Integrity Rule
This test split and its predictions are permanently frozen. External corpora (specifically `Boredoom17/Nepali-Flow-Roman`) MUST NOT be used to tune or contaminate this evaluation benchmark.
