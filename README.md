# Romanized Nepali NLP Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A specialized NLP toolkit engineered specifically for Romanized Nepali (Nepali written using the Latin alphabet, often mixed with English loanwords, slang, code-switching, emojis, and social web tokens).

It provides:
- **Unicode-Safe Tokenization**: Exact code-point character offset tracking across Latin, Devanagari, emoticons, hashtags, mentions, and compound emojis.
- **Configurable Preprocessing**: Normalizes elongated vowels (`ramrooo` -> `ramro`), irregular Unicode whitespaces, zero-width characters, and phonetic variations (`chha` -> `cha`, `vayo` -> `bhayo`) while strictly protecting URLs, handles, and emojis.
- **Dataset Validation & Deduplication**: High-performance exact and near-duplicate detection, label conflict auditing, and field schema validation.
- **Leakage-Safe Data Splitting**: Group-aware, transitive deduplication with union-find and stratification to prevent train/validation/test contamination.
- **Sentiment Baseline**: Classical character n-gram (`char_wb`) TF-IDF and logistic regression baseline with challenge slice evaluation and bootstrap confidence intervals.
- **External Corpus Ingestion**: Structural profiling and quality triage for large scraped web corpora (e.g. YouTube comments).

---

## Table of Contents

- [Installation](#installation)
- [Quickstart: Python API](#quickstart-python-api)
  - [1. Tokenization & Span Offsets](#1-tokenization--span-offsets)
  - [2. Preprocessing & Normalization](#2-preprocessing--normalization)
  - [3. Dataset Loading, Quality Audit & Deduplication](#3-dataset-loading-quality-audit--deduplication)
  - [4. Leakage-Safe Splitting](#4-leakage-safe-splitting)
  - [5. Sentiment Baseline Model](#5-sentiment-baseline-model)
- [Command Line Interface (CLI)](#command-line-interface-cli)
- [Repository Structure](#repository-structure)
- [Architecture & Detailed Documentation](#architecture--detailed-documentation)
- [Licensing](#licensing)
- [Credits & Attributions](#credits--attributions)
- [Citation](#citation)

---

## Installation

Install in editable mode:

```bash
# Clone the repository
git clone https://github.com/<your-username>/romanized-nepali-nlp.git
cd romanized-nepali-nlp

# Library and sentiment baseline
pip install -e .

# Add chart generation and other optional ML tooling
pip install -e ".[ml]"

# With Development extras (pytest, ruff)
pip install -e ".[dev,ml]"
```

Run test suite to verify installation:

```bash
pytest
ruff check .
```

---

## Quickstart: Python API

### 1. Tokenization & Span Offsets

The tokenizer yields `Token` objects with exact `[start:end]` Unicode code-point offsets, guaranteeing `text[start:end] == token.text`:

```python
from romanized_nepali_nlp import tokenize, format_terminal, format_html

text = "namaste sathi! check https://hamro.com/news, #NepalPride 🔥"
tokens = tokenize(text)

# Terminal bracketed view: [text|CATEGORY]
print(format_terminal(tokens))
# [namaste|WORD] [sathi|WORD] [!|PUNCTUATION] [check|WORD] [https://hamro.com/news|URL] [,|PUNCTUATION] [#NepalPride|HASHTAG] [🔥|EMOJI]

# Inspect token attributes
for t in tokens:
    print(f"{t.text:<25} {t.category.value:<12} [{t.start}:{t.end}]")

# Export interactive color-coded HTML visualization
html_str = format_html(tokens, text, title="Token Visualization")
```

### 2. Preprocessing & Normalization

```python
from romanized_nepali_nlp import PreprocessingPipeline, PipelineConfig

config = PipelineConfig(
    lowercase=False,
    reduce_repetitions=True,
    normalize_variants=True,
    variant_policy="conservative"
)
pipeline = PreprocessingPipeline(config)

result = pipeline.preprocess("yo geet raamro chha dheraiii, check https://hamro.com 🔥")

print("Raw       :", result.raw_text)
# Raw       : yo geet raamro chha dheraiii, check https://hamro.com 🔥

print("Normalized:", result.normalized_text)
# Normalized: yo geet ramro cha dherai, check https://hamro.com 🔥

print("Steps     :", result.steps)
# ['protect_special_tokens', 'unicode_nfc', 'reduce_repetitions', 'normalize_variants_conservative', 'normalize_whitespace', 'restore_special_tokens']
```

### 3. Dataset Loading, Quality Audit & Deduplication

```python
from romanized_nepali_nlp import load, compute_stats, detect_duplicates

# Auto-detects CSV, TSV, or JSONL datasets
res = load("tests/fixtures/valid_dataset.jsonl")
print(f"Loaded {len(res.records)} records with {len(res.issues)} issues.")

# Detect exact raw and normalized duplicates + label conflicts
dup_report = detect_duplicates(res.records)
print(f"Raw duplicate groups: {len(dup_report.raw_groups)}")
print(f"Conflicted IDs      : {dup_report.conflicted_ids}")

# Compute aggregate quality statistics
stats = compute_stats(res.records, dup_report=dup_report, issues=res.issues)
print(stats.to_markdown())
```

### 4. Leakage-Safe Splitting

Splits dataset into reproducible train/validation/test partitions with label stratification and transitive duplicate grouping (ensuring identical and near-identical texts never cross partitions):

```python
from romanized_nepali_nlp import SplitConfig, split, check_leakage

config = SplitConfig(train=0.70, validation=0.15, test=0.15, seed=42)
split_result = split(res.records, config=config)

# Pre-training leakage check
leaked = check_leakage(split_result)
assert len(leaked) == 0, f"Leakage detected: {leaked}"

print(f"Train: {len(split_result.train)}, Val: {len(split_result.validation)}, Test: {len(split_result.test)}")
```

### 5. Sentiment Baseline Model

Train a character n-gram (`char_wb`) baseline, predict on new samples, and inspect top predictive features:

```python
from romanized_nepali_nlp import SentimentModel, SentimentModelConfig

# Configure character n-gram TF-IDF + LogisticRegression baseline
cfg = SentimentModelConfig(
    model_type="char",
    analyzer="char_wb",
    ngram_range=(2, 5),
    class_weight="balanced",
    C=1.0,
    seed=42
)

model = SentimentModel(cfg).fit(split_result.train)

# Predict labels and probability distribution
preds = model.predict_texts(["yo video ekdam dami cha", "sarhai naramro service"])
for p in preds:
    print(f"{p.id}: {p.predicted_label} (scores: {p.scores})")

# Inspect top features per class
top_feats = model.get_top_features(n=5)
for cls_name, feats in top_feats.items():
    print(f"Class: {cls_name} -> {feats}")

# Persist and reload
model.save("runs/char_model.pkl")
loaded_model = SentimentModel.load("runs/char_model.pkl")
```

---

## Command Line Interface (CLI)

The package installs the `romanized-nepali` executable with comprehensive subcommands:

### 1. Tokenization

```bash
# Tokenize free text to tabular view
romanized-nepali tokenize "yo ta ramrooo cha @anil_k 🔥"

# Stream line-by-line in brackets format: [word|CATEGORY]
romanized-nepali tokenize "k cha?" --format brackets

# Generate standalone color-coded HTML visualization
romanized-nepali tokenize "nepal jindabad 🇳🇵" --html tokens.html
```

### 2. Preprocessing

```bash
# Normalize and collapse elongated repetitions
romanized-nepali preprocess "yo ta raamro chha dheraiii"

# Display side-by-side / unified diff
romanized-nepali preprocess "raamro chha" --format diff

# Output full JSON provenance metadata
romanized-nepali preprocess "dheraiii maza aayo" --format json
```

### 3. Dataset Management

```bash
# Inspect dataset schema, first rows, and missingness
romanized-nepali dataset preview my_dataset.csv

# Generate dataset quality and duplicate report (Markdown or HTML)
romanized-nepali dataset report my_dataset.jsonl --format markdown --out quality.md

# Split dataset into train/val/test with manifest
romanized-nepali dataset split my_dataset.jsonl --train 0.70 --validation 0.15 --test 0.15 --out-dir data/splits --manifest data/splits/manifest.json
```

### 4. Sentiment Training, Inference & Evaluation

```bash
# Train character baseline model
romanized-nepali train data/splits/train.jsonl --model-type char -o runs/char_run

# Predict on new inputs (file or piped stdin)
romanized-nepali predict data/splits/test.jsonl --model runs/char_run/model.pkl --format table

# Evaluate predictions with challenge slices and bootstrap confidence intervals
romanized-nepali evaluate runs/char_run/val_predictions.jsonl --format markdown --out eval_report.md
```

### 5. External Corpus Ingestion & Triage

```bash
# Ingest raw external corpus, filter non-Latin text, and partition into clean, mixed, and noisy
romanized-nepali corpus data/raw/scraped_comments.jsonl --benchmark data/splits/test.jsonl --output-dir data/processed/corpus
```

### 6. Robustness and Release Workflows

Run the spelling-variation challenge evaluation, phenomenon slice analysis, and
normalization ablation from the repository root:

```bash
python3 scripts/evaluate_spelling_challenge.py
python3 scripts/phenomena_analysis.py
python3 scripts/run_ablation.py
python3 scripts/demo.py
```

Outputs go to `reports/` and `experiments/ablation_phase6/`. The ablation runner
selects preprocessing on validation macro-F1 and evaluates the frozen test set
once. For a release check, run:

```bash
pytest
ruff check .
```

See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md), [docs/data-card.md](docs/data-card.md),
and [docs/model-card.md](docs/model-card.md) before distributing results.

---

## Repository Structure

```text
├── configs/               # Reproducible JSON configurations for baselines & preprocessing
├── docs/                  # Architecture specs, tokenization contracts, annotation guidelines
├── experiments/           # Experiment logs, split manifests, and metrics
├── reports/               # Tokenizer diagnostic reports, galleries, and charts
├── runs/                  # Saved model artifacts, training logs, and validation reports
├── src/
│   └── romanized_nepali_nlp/
│       ├── cli.py             # Unified CLI entry points
│       ├── corpus.py          # External corpus ingestion & triage
│       ├── coverage.py        # Tokenizer coverage diagnostic profiler
│       ├── dataset.py         # CSV/JSONL loaders & schema validation
│       ├── evaluation.py      # Challenge slices, bootstrap CIs & error reporting
│       ├── normalization.py   # Unicode, whitespace, repetitions & protected spans
│       ├── pipeline.py        # Composable preprocessing pipeline
│       ├── quality.py         # Duplicate detection & dataset statistics
│       ├── reporting.py       # Diagnostic charts & HTML galleries
│       ├── resources/         # Curated Romanized spelling variant lexicons
│       ├── sentiment.py       # Majority, word & character n-gram sentiment baselines
│       ├── spans.py           # Token span validation & visualizations
│       ├── splitting.py       # Group-aware, leakage-safe dataset splitting
│       ├── tokenizer.py       # Master Unicode-safe regex scanner
│       ├── types.py           # Core dataclasses & schemas
│       ├── variant_mining.py  # Lexical frequency spelling candidate extraction
│       └── variants.py        # Spelling variation normalizer
└── tests/                 # 160+ unit, integration, and regression tests
    └── fixtures/          # Curated test datasets and expected tokens
```

---

## Architecture & Detailed Documentation

- [docs/project-map.md](docs/project-map.md): End-to-end component flow and system lifecycle
- [docs/api-contract.md](docs/api-contract.md): Core schemas, code-point offsets, and data contracts
- [docs/tokenization.md](docs/tokenization.md): Tokenizer specification, character policy, and regex precedence
- [docs/preprocessing-gallery.md](docs/preprocessing-gallery.md): Pipeline flow diagram and before/after normalization gallery
- [docs/variant-dashboard.md](docs/variant-dashboard.md): Spelling variant registry and review status
- [docs/annotation-guidelines.md](docs/annotation-guidelines.md): Guidelines for sentiment labeling Romanized Nepali
- [docs/baseline-comparison.md](docs/baseline-comparison.md): Empirical comparison of majority, word, and character n-gram baselines
- [docs/evaluation-report.md](docs/evaluation-report.md): Challenge slice robustness and error breakdown
- [docs/data-card.md](docs/data-card.md): Data sources, privacy, labels, bias, and limits
- [docs/model-card.md](docs/model-card.md): Baseline models, metrics, intended use, and failure modes
- [docs/project-status.md](docs/project-status.md): Release-candidate status dashboard
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md): Final verification and human-approval gate

---

## Licensing

This project is licensed under the terms of the [MIT License](LICENSE).

---

## Credits & Attributions

- **Boredoom17/Nepali-Flow-Roman**: External corpus data curated by Aadarsha Chhetri ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Scikit-learn**: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
- **Matplotlib**: Hunter, J. D., *Matplotlib: A 2D Graphics Environment*, Computing in Science & Engineering, 2007.

---

## Citation

If you use this toolkit in your research or application, please cite:

```bibtex
@software{romanized_nepali_nlp_2026,
  title = {{Romanized Nepali NLP Toolkit}: Tokenization, Normalization, and Sentiment Baselines},
  author = {Ghimire, Purushottam and contributors},
  year = {2026},
  url = {https://github.com/purushottamghimire/romanized-nepali-nlp},
  license = {MIT}
}
```

---

## Contributing

Contributions, bug reports, and pull requests are welcome!
Please make sure to run the linters and test suite before submitting pull requests:

```bash
ruff check .
ruff format --check .
pytest
```
