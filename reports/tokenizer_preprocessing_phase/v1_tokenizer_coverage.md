# Tokenizer Coverage & Profiling Report (v1)

## Executive Summary

- **Total Rows Profiled:** 307,948
- **Total Tokens:** 4,183,084
- **Average Tokens / Row:** 13.58
- **Tokenizer Crashes / Errors:** 0
- **Unknown Spans Total:** 10,940 (0.2615% of tokens)
- **Uncovered Non-Whitespace Characters:** 0 (0.0000%)

---

## Partition-Level Breakdown

| Metric | Overall | Clean | Mixed | Noisy |
| :--- | :--- | :--- | :--- | :--- |
| **Rows** | 307,948 | 226,612 | 73,345 | 7,991 |
| **Tokens** | 4,183,084 | 2,513,520 | 1,529,908 | 139,656 |
| **Avg Tokens/Row** | 13.58 | 11.09 | 20.86 | 17.48 |
| **Unknown Spans** | 10,940 | 4,789 | 5,827 | 324 |
| **Unknown Span Rate** | 0.2615% | 0.1905% | 0.3809% | 0.2320% |
| **Rows w/ Unknowns** | 5,946 | 2,630 | 3,144 | 172 |
| **Repeated Tokens (3+)** | 9,867 | 0 | 0 | 9,867 |
| **Long Tokens (>30c)** | 93 | 53 | 21 | 19 |

---

## Token Category Distribution

| Category | Total Count | Clean | Mixed | Noisy |
| :--- | :--- | :--- | :--- | :--- |
| **WORD** | 3,887,498 | 2,357,388 | 1,400,374 | 129,736 |
| **PUNCTUATION** | 235,482 | 118,789 | 108,714 | 7,979 |
| **NUMBER** | 45,629 | 30,594 | 13,839 | 1,196 |
| **HASHTAG** | 2,122 | 1,192 | 830 | 100 |
| **MENTION** | 784 | 470 | 292 | 22 |
| **EMOJI** | 622 | 294 | 29 | 299 |
| **EMAIL** | 7 | 4 | 3 | 0 |
| **URL** | 0 | 0 | 0 | 0 |
| **UNKNOWN** | 10,940 | 4,789 | 5,827 | 324 |

---

## Token Length Distributions

| Bucket (chars) | Total Tokens | Clean | Mixed | Noisy |
| :--- | :--- | :--- | :--- | :--- |
| **1-3** | 1,649,430 | 952,701 | 645,640 | 51,089 |
| **4-6** | 2,047,864 | 1,282,379 | 695,077 | 70,408 |
| **7-10** | 453,106 | 261,627 | 174,524 | 16,955 |
| **11-20** | 32,403 | 16,660 | 14,577 | 1,166 |
| **21-30** | 188 | 100 | 69 | 19 |
| **31+** | 93 | 53 | 21 | 19 |

---

## Visual Diagnostic Charts

1. `charts/token_categories.png` - Distribution of token categories across partitions.
2. `charts/unknown_spans.png` - Unknown span rates by partition.
3. `charts/token_lengths.png` - Token length distribution histograms.
4. `charts/special_tokens.png` - Special tokens (Hashtag, Mention, Emoji, Number, URL).
5. `charts/repetitions.png` - Repeated character tokens across partitions.
