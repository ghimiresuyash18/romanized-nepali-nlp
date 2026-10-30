# Model Card

## Models
Majority, word TF-IDF, and character `char_wb` TF-IDF + logistic regression baselines. Transformer/contextual models deferred.

## Training and evaluation
Frozen split: `experiments/baseline_phase5/split_manifest.json`. Model artifacts: `runs/*/model.pkl`. Reproducible preprocessing ablations: `configs/ablation_phase6.json` and `scripts/run_ablation.py`.

## Intended use
Baseline research and regression testing for Romanized Nepali sentiment.

## Limitations
Small benchmark. Neutral class, negation scope, sarcasm, emoji-only text, OOV English, numeric tokens, and spelling variation remain failure modes. Scores are not production quality.

## Safety
Do not infer personal attributes or make consequential decisions from predictions. Models are pickled; load only trusted artifacts.
