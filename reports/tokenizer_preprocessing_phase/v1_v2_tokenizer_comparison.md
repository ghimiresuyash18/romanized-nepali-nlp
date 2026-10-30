# Tokenizer v1 vs v2 Comparison Report

## Summary Comparison

| Metric | v1 Tokenizer | v2 Tokenizer | Delta |
| :--- | :--- | :--- | :--- |
| **Total Rows Processed** | 307,948 | 307,948 | 0 |
| **Total Tokens** | 4,183,084 | 4,177,205 | -5,879 |
| **Avg Tokens / Row** | 13.58 | 13.56 | -0.02 |
| **Total Unknown Spans** | 10,940 | 888 | -10,052 (-91.9%) |
| **Unknown Span Rate** | 0.2615% | 0.0213% | -0.2402% |
| **Tokenizer Crashes / Errors** | 0 | 0 | 0 |
| **Uncovered Chars** | 0 | 0 | 0 |

---

## Partition-Level Comparison

### Partition: `roman_clean`
- **Rows:** 226,612
- **v1 Unknown Spans:** 4,789 (0.1905%)
- **v2 Unknown Spans:** 659 (0.0262%)
- **Delta:** -4,130 unknown spans

### Partition: `roman_mixed`
- **Rows:** 73,345
- **v1 Unknown Spans:** 5,827 (0.3809%)
- **v2 Unknown Spans:** 196 (0.0128%)
- **Delta:** -5,631 unknown spans

### Partition: `roman_noisy`
- **Rows:** 7,991
- **v1 Unknown Spans:** 324 (0.2320%)
- **v2 Unknown Spans:** 33 (0.0236%)
- **Delta:** -291 unknown spans

---

## Token Category Changes

| Category | v1 Count | v2 Count | Delta |
| :--- | :--- | :--- | :--- |
| **WORD** | 3,887,498 | 3,888,828 | +1,330 |
| **PUNCTUATION** | 235,482 | 238,389 | +2,907 |
| **NUMBER** | 45,629 | 45,563 | -66 |
| **UNKNOWN** | 10,940 | 888 | -10,052 |
| **HASHTAG** | 2,122 | 2,123 | +1 |
| **MENTION** | 784 | 785 | +1 |
| **EMOJI** | 622 | 622 | +0 |