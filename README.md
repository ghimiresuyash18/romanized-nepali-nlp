# Romanized Nepali NLP Toolkit

A Python toolkit focused exclusively on Romanized Nepali text. It provides Unicode-safe tokenization, configurable text normalization, dataset quality verification, and a classical character n-gram sentiment analysis baseline.

## Quickstart

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev,ml]"

# Run tests
pytest

# Run linter
ruff check .
```

## Python API Usage

### Tokenization

```python
from romanized_nepali_nlp import tokenize, format_terminal, format_html

text = "namaste sathi! check https://hamro.com/news, #NepalPride 🔥"
tokens = tokenize(text)

# Terminal bracketed view: [text|CATEGORY]
print(format_terminal(tokens))
# [namaste|WORD] [sathi|WORD] [!|PUNCTUATION] [check|WORD] [https://hamro.com/news|URL] [,|PUNCTUATION] [#NepalPride|HASHTAG] [🔥|EMOJI]

# Export interactive color-coded HTML visualization
html_str = format_html(tokens, text, title="My Tokenization")
```

### Preprocessing Pipeline

```python
from romanized_nepali_nlp import PreprocessingPipeline, PipelineConfig

# Conservative default: preserves raw text, URLs/emojis protected, safe variants normalized
pipeline = PreprocessingPipeline(PipelineConfig())
result = pipeline.preprocess("yo geet raamro chha dheraiii, check https://hamro.com 🔥")

print(result.raw_text)  # Original text untouched
print(result.normalized_text)  # "yo geet ramro cha dherai, check https://hamro.com 🔥"
print(
    result.steps
)  # ['protect_special_tokens', 'normalize_variants_conservative', 'restore_special_tokens']
```

## CLI Usage

The package installs the `romanized-nepali` command line tool:

```bash
# 1. Tokenize directly
romanized-nepali tokenize "yo ta ramrooo cha @anil_k 🔥"

# Output:
# TEXT                      CATEGORY        START  END   
# ------------------------------------------------------------
# 1    yo                        WORD            0      2     
# 2    ta                        WORD            3      5     
# 3    ramrooo                   WORD            6      13    
# 4    cha                       WORD            14     17    
# 5    @anil_k                   MENTION         18     25    
# 6    🔥                        EMOJI           26     27    

# Output formats: brackets, json, markdown
romanized-nepali tokenize "k cha?" --format brackets
# [k|WORD] [cha|WORD] [?|PUNCTUATION]

# Generate standalone HTML visualization
romanized-nepali tokenize "nepal jindabad 🇳🇵" --html tokens.html

# 2. Preprocess & Normalize
romanized-nepali preprocess "yo ta raamro chha dheraiii"
# yo ta ramro cha dherai

# View unified diff
romanized-nepali preprocess "raamro" --format diff
# - raamro
# + ramro

# Output full JSON provenance
romanized-nepali preprocess "dheraiii maza aayo" --format json
```

## Architecture & Documentation

- [docs/project-map.md](docs/project-map.md): End-to-end component flow
- [docs/api-contract.md](docs/api-contract.md): Core schemas, code-point offsets, and data contracts
- [docs/tokenization.md](docs/tokenization.md): Tokenizer specification, character policy, and regex precedence
- [docs/preprocessing-gallery.md](docs/preprocessing-gallery.md): Preprocessing pipeline flow diagram and before/after sample gallery
- [docs/variant-dashboard.md](docs/variant-dashboard.md): Spelling variant registry and review status
- [tests/fixtures/token_category_distribution.md](tests/fixtures/token_category_distribution.md): Token category distribution over 106 fixtures
