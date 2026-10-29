# Project Map: Romanized Nepali NLP Toolkit

This document maps the architectural flow from raw Romanized Nepali text to evaluated sentiment predictions.

```mermaid
flowchart TD
    subgraph Input ["Input Layer"]
        RAW["Raw Romanized Nepali Text (CSV / JSONL)"]
    end

    subgraph Preprocessing ["Text Processing Pipeline"]
        TOK["Tokenizer<br/>(Unicode-safe regex / offsets)"]
        NORM["Normalizers<br/>(Unicode NFC/NFKD, Social tokens, Variants)"]
        RES["PreprocessingResult<br/>(raw_text, normalized_text, metadata)"]
        TOK --> NORM --> RES
    end

    subgraph DataManagement ["Dataset & Validation"]
        VAL["Validation & Leakage Prevention<br/>(Deduplication & label conflict audit)"]
        SPLIT["Stratified / Leakage-Safe Splitting<br/>(Train / Validation / Test)"]
        VAL --> SPLIT
    end

    subgraph Modeling ["Feature & Baseline Model"]
        FEAT["Character n-gram TF-IDF<br/>(Spelling-variation tolerant)"]
        CLF["Linear Classifier<br/>(Logistic Regression / LinearSVC)"]
        FEAT --> CLF
    end

    subgraph Evaluation ["Evaluation & Error Analysis"]
        METRICS["Standard Metrics<br/>(Macro-F1, Precision, Recall)"]
        SLICES["Slice Reporting<br/>(Spelling variants, Code-switching, Short texts, Negation, Emojis)"]
        METRICS --> SLICES
    end

    RAW --> TOK
    RAW --> VAL
    RES --> FEAT
    SPLIT --> FEAT
    CLF --> METRICS
```

## Component Overview

1. **Tokenizer (`tokenizer.py`)**: Preserves character offsets in code points and extracts categories (WORD, EMOJI, URL, MENTION, etc.).
2. **Normalization (`normalization.py` & `variants.py`)**: Non-destructive transformations. Always preserves `raw_text` and logs applied steps.
3. **Dataset & Validation (`dataset.py`, `quality.py`, `splitting.py`)**: Validates schemas, detects duplicate/contradictory labels across train/test splits.
4. **Features & Model (`features.py`, `sentiment.py`)**: Character n-grams tolerate orthographic variation (`ramro` / `raamro` / `ramroo`).
5. **Evaluation (`sentiment.py`)**: Produces per-slice performance breakdowns to evaluate robustness on noisy text.
