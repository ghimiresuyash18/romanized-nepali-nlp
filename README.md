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

```python
from romanized_nepali_nlp import tokenize, format_terminal, format_html

text = "namaste sathi! check https://hamro.com/news, #NepalPride 🔥"
tokens = tokenize(text)

# Terminal bracketed view: [text|CATEGORY]
print(format_terminal(tokens))
# [namaste|WORD] [sathi|WORD] [!|PUNCTUATION] [check|ENGLISH_WORD] [https://hamro.com/news|URL] [,|PUNCTUATION] [#NepalPride|HASHTAG] [🔥|EMOJI]

# Export interactive color-coded HTML visualization
html_str = format_html(tokens, text, title="My Tokenization")
```

## CLI Usage

The package installs the `romanized-nepali` command line tool:

```bash
# Tokenize directly
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

# Inspect span coverage and diagnostics
romanized-nepali tokenize "Check this: https://hamro.com/news" --diagnostics
```

## Architecture & Documentation

- [docs/project-map.md](docs/project-map.md): End-to-end component flow
- [docs/api-contract.md](docs/api-contract.md): Core schemas, code-point offsets, and data contracts
- [docs/tokenization.md](docs/tokenization.md): Tokenizer specification, character policy, and regex precedence
- [tests/fixtures/token_category_distribution.md](tests/fixtures/token_category_distribution.md): Token category distribution over 106 fixtures
