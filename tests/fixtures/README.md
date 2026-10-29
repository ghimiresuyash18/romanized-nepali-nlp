# Romanized Nepali Development Fixtures

This directory contains test fixtures and development mini-corpora used to validate the tokenizer, normalization pipeline, and baseline sentiment classifiers.

## Files

1. **`romanized_examples.jsonl`**:
   - 106 curated examples of Romanized Nepali text.
   - Covers 6 key linguistic categories:
     - `ordinary`: Natural greetings, inquiries, opinions, and everyday comments.
     - `spelling_variation`: Common phonetic spelling alternates (`ramro`, `raamro`, `ramroo`, `chha`, `chaa`, `dheraiii`, `vako`, `xau`).
     - `code_switching`: Romanized Nepali mixed with English words, URLs, @mentions, #hashtags, and numeric/currency expressions.
     - `punctuation_emoji`: Emoticons, repeated punctuation (`???`, `!!!`), Unicode emoji combinations (`🔥`, `❤️`, `😡`, `😂`, `🙏`).
     - `negation_intensifier`: Negation markers (`chaina`, `lagena`, `parena`) and intensifiers (`ekdam`, `sarai`, `achammai`, `bilkul`).
     - `edge_cases`: Empty text, whitespace-only, Latin diacritics (`ā`, `ī`), uppercase, malformed URLs, and isolated symbols.

2. **`expected_tokens.jsonl`**:
   - 32 hand-annotated examples.
   - Exact start/end code-point offsets for each token.
   - Annotates categories: `WORD`, `ENGLISH_WORD`, `URL`, `EMAIL`, `MENTION`, `HASHTAG`, `NUMBER`, `EMOJI`, `PUNCTUATION`.

## Privacy & Provenance

- **Origin**: Synthetic and anonymized sentences designed specifically to evaluate Romanized Nepali orthographic variability.
- **Privacy & PII**: Contains zero personally identifiable information (PII). All user handles (e.g. `@user`, `@anil_k`) and domains (`hamro.com`, `example.com`) are synthetic placeholders.
- **Licensing**: Released under the project's MIT License for open research and reproducibility.
