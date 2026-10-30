# Final Report: Corpus-Driven Tokenizer and Preprocessing Expansion

## 1. Executive Summary

This phase validated and expanded the Romanized Nepali tokenizer and preprocessing pipeline using the real 307,999-row `Nepali-Flow-Roman` corpus. All operations were strictly non-destructive: existing baseline files (`experiments/baseline_phase5/`) and v1 partitions (`data/processed/nepali_flow_roman_full/`) were frozen and untouched.

### Decisive Outcomes
- **Coverage:** Successfully processed all 307,948 active corpus rows (and 51 triage rejection rows) with **0 tokenizer crashes or uncaught exceptions**.
- **Span Unknown Rate Reduction:** Unknown spans dropped from **10,940 (0.2615%)** in v1 to **888 (0.0213%)** in v2 — a **91.9% reduction**.
- **Regressions & Quality:** 53 new corpus regression tests and 52 fixture cases added. Overall test suite expanded from **89 to 156 passing tests** (100% pass rate). Ruff lint clean.
- **Conservative Normalization:** Verified across raw, Unicode/whitespace, protected-spans, and approved-variant modes. Negation clauses (`chaina`, `lagena`, `man parena`, `thiyena`), intensifiers, and mixed code-switching preserved without loss.
- **Resource Discipline:** Extracted 636 candidate spelling variants into interim storage; only linguistically verified, approved mappings remain in the active resource.

---

## 2. v1 vs v2 Coverage Metrics

| Metric | v1 Tokenizer | v2 Tokenizer | Delta |
| :--- | :--- | :--- | :--- |
| **Total Rows Profiled** | 307,948 | 307,948 | 0 |
| **Total Tokens** | 4,183,084 | 4,177,205 | -5,879 |
| **Avg Tokens / Row** | 13.58 | 13.56 | -0.02 |
| **Unknown Spans Total** | 10,940 | 888 | **-10,052 (-91.9%)** |
| **Unknown Span Rate** | 0.2615% | 0.0213% | **-0.2402%** |
| **Uncovered Characters** | 0 (0.0%) | 0 (0.0%) | 0 |
| **Tokenizer Crashes / Errors** | 0 | 0 | 0 |

### Partition Breakdown
- **`roman_clean` (226,612 rows):**
  - v1 Unknowns: 4,789 (0.1905%) $\rightarrow$ v2 Unknowns: **659 (0.0262%)** (-4,130)
- **`roman_mixed` (73,345 rows):**
  - v1 Unknowns: 5,827 (0.3809%) $\rightarrow$ v2 Unknowns: **196 (0.0128%)** (-5,631)
- **`roman_noisy` (7,991 rows):**
  - v1 Unknowns: 324 (0.2320%) $\rightarrow$ v2 Unknowns: **33 (0.0236%)** (-291)

---

## 3. Tokenizer Improvements & Justification

All additions directly addressed genuine failures discovered in the 250 stratified review rows:
1. **Typographic Apostrophes & Quotes:** Added `’` (U+2019) to `word_intra` apostrophes, allowing `it’s`, `don’t`, `timro’s`, `Nepal’s` to remain unified `WORD` tokens. Added `“`, `”`, `‘`, `’`, `«`, `»`, `„`, `‟`, `‹`, `›` to punctuation.
2. **Ellipsis Sequences:** Added `…` (U+2026) and multi-dot/ellipsis runs `[.…]{2,}` to `PUNCTUATION`, preventing ellipsis from fusing with adjacent words (e.g. `maya…`).
3. **Currency Symbols:** Added `₹`, `£`, `€`, `¥` to recognized punctuation, preventing `£800` or `₹500` from falling into `UNKNOWN`.
4. **Alphanumerics & Percentage:** Updated `NUMBER_PATTERN` to support percentages (`20%`) and `ALPHANUM_PATTERN` to handle hyphenated terms like `Covid-19`.
5. **Indic Romanization Diacritics:** Expanded Latin character ranges to include Latin Extended Additional (`\u1E00-\u1EFF`) for IAST / ISO 15919 transliteration (e.g. `nepālī`, `kṛṣṇa`).
6. **Expressive Punctuation:** Added `⁉` and `‼` glyphs to expressive punctuation runs.

---

## 4. Preprocessing Modes & Fingerprints

Three versioned configurations are saved in `configs/`:

| Config File | Mode Name | Description | Fingerprint |
| :--- | :--- | :--- | :--- |
| `configs/preprocessing_raw.json` | `raw` | Byte-for-byte passthrough of original text | `716aef3e5c8da7ed` |
| `configs/preprocessing_conservative.json` | `conservative` | NFC + whitespace normalization; protected URLs, handles, emojis, numbers | `3d2a5c45709069da` |
| `configs/preprocessing_variant_review.json` | `variant_review` | Conservative + 2-repeat capping + approved spelling variant mapping | `0ac588c4b50707b1` |

Verified that negation clauses (`ramro chaina`, `ramro lagena`, `man parena`, `movie ramro thiyena`) retain meaning and tokens across all modes.

---

## 5. Artifacts and File Manifest

### Reports & Galleries
- `reports/tokenizer_preprocessing_phase/v1_tokenizer_coverage.md`
- `reports/tokenizer_preprocessing_phase/v1_v2_tokenizer_comparison.md`
- `reports/tokenizer_preprocessing_phase/final_report.md`
- `reports/tokenizer_preprocessing_phase/token_boundary_gallery.html`
- `reports/tokenizer_preprocessing_phase/preprocessing_diff_gallery.html`
- Diagnostic Charts in `reports/tokenizer_preprocessing_phase/charts/`:
  - `token_categories.png`
  - `unknown_spans.png`
  - `token_lengths.png`
  - `special_tokens.png`
  - `repetitions.png`

### Interim & Mined Data
- `data/interim/tokenizer_coverage_v1/` (`roman_clean.jsonl`, `roman_mixed.jsonl`, `roman_noisy.jsonl`, `coverage_summary.json`)
- `data/interim/tokenizer_coverage_v2/` (`roman_clean.jsonl`, `roman_mixed.jsonl`, `roman_noisy.jsonl`, `coverage_summary.json`)
- `data/interim/tokenizer_review/stratified_review.jsonl` (250 reviewed rows)
- `data/interim/variant_candidates/candidates.jsonl` (636 candidate pairs)
- `data/interim/preprocessing_review/preprocessing_modes_review.json`

### Corpus v2 Output
- Directory: `data/processed/nepali_flow_roman_v2/`
- Partition counts: `roman_clean` (226,612), `roman_mixed` (73,345), `roman_noisy` (7,991), `reject_or_review` (51)
- Manifest: `data/processed/nepali_flow_roman_v2/manifest.json`

### Fixtures & Tests
- `tests/fixtures/corpus_expected_tokens.jsonl` (52 regression test cases)
- `tests/fixtures/corpus_review_cases.jsonl` (20 categorized review cases)
- `tests/fixtures/corpus_normalization_cases.jsonl` (10 negation and normalization cases)
- `tests/fixtures/corpus_variant_candidates.jsonl` (30 mined variant cases)
- Test suite: `tests/test_corpus_regressions.py`, `tests/test_coverage.py`, `tests/test_variant_mining.py`, `tests/test_reporting.py` (156 passing tests)

---

## 6. Known Limitations

1. **Residual Unknown Spans (888 items, 0.021%):** Consist exclusively of genuine foreign scripts (Vietnamese accents, Azerbaijani characters, Bengali script) or kaomoji art sequences with invisible control characters (`\u2060`). These are source data noise rather than Romanized Nepali words.
2. **Ambiguous Context Words:** Particles like `yo`, `ma`, `ta`, and `na` remain classified structurally as `WORD` without language forced. Contextual disambiguation is deferred to model training.

---

## 7. Recommended Annotation Slices for Sentiment Phase

For the subsequent manual sentiment annotation phase, sample stratified slices as follows:
1. **Clean Sentiment Core (1,000 rows):** Draw 1,000 rows from `data/processed/nepali_flow_roman_v2/roman_clean.jsonl` with short to medium length (5-20 words).
2. **Negation Focus (500 rows):** Filter for rows containing `chaina`, `lagena`, `parena`, `thiyena`, `hoina`, `hudaina` to ensure heavy evaluation of valence flips.
3. **Mixed Code-Switching (500 rows):** Sample from `data/processed/nepali_flow_roman_v2/roman_mixed.jsonl` to assess sentiment behavior on English-Nepali mixed comments.
4. **Social & Emoji (500 rows):** Sample from `data/processed/nepali_flow_roman_v2/roman_noisy.jsonl` containing emojis and repeated letters to evaluate expressive sentiment.
