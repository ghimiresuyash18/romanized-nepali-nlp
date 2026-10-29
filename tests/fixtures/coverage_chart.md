# Fixture Coverage Report

Summary of dataset coverage across categories in `romanized_examples.jsonl`:

| Category | Count | Description / Focus |
|---|---|---|
| `ordinary` | 20 | Greetings, opinions, daily inquiries, standard spelling |
| `spelling_variation` | 18 | Lengthened vowels (`raamro`, `ramroo`), aspirated consonants (`chha`), chat phonetics (`vako`, `xau`) |
| `code_switching` | 17 | English loanwords, URLs, `@mentions`, `#hashtags`, phone numbers, currency |
| `punctuation_emoji` | 17 | Repeated punctuation (`???`, `!!!`), Unicode emojis (`❤️`, `🔥`, `🙏`, `😂`), emoticons (`:)`, `:(`) |
| `negation_intensifier` | 17 | Morphological negations (`chaina`, `lagena`, `parena`), intensifiers (`ekdam`, `sarai`, `achammai`) |
| `edge_cases` | 20 | Empty strings, whitespace-only, Latin diacritics (`ā`, `ī`), uppercase, broken URLs |
| **Total Examples** | **106** | Exceeds >= 100 acceptance criteria |

## Hand-Annotated Ground Truth (`expected_tokens.jsonl`)

- **Total Annotated Records**: 32 (Target: >= 30)
- **Code-Point Offset Integrity**: 100% verified (`text[start:end] == token.text` for all tokens).
- **Categories Verified**: `WORD`, `ENGLISH_WORD`, `URL`, `EMAIL`, `MENTION`, `HASHTAG`, `NUMBER`, `EMOJI`, `PUNCTUATION`.
