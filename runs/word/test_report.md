# Model Evaluation & Challenge Slice Report

## 1. Summary Performance

- **Total Evaluated Records**: 15
- **Accuracy**: **0.6667** (95% CI: [0.4000, 0.8667])
- **Macro-F1**: **0.6243** (95% CI: [0.3521, 0.8474])
- **Weighted-F1**: **0.6519**

## 2. Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `negative` | 0.4000 | 0.5000 | **0.4444** | 4 |
| `neutral` | 1.0000 | 0.4000 | **0.5714** | 5 |
| `positive` | 0.7500 | 1.0000 | **0.8571** | 6 |

## 3. Confusion Matrix

| Gold \ Predicted | `negative` | `neutral` | `positive` |
|---|---|---|---|
| **`negative`** | 2 | 0 | 2 |
| **`neutral`** | 3 | 2 | 0 |
| **`positive`** | 0 | 0 | 6 |

## 4. Challenge Slice Robustness

| Slice | Count | Accuracy | Macro-F1 | Status |
|---|---|---|---|---|
| **spelling_variation** | 4 | 1.0000 | **1.0000** | ✅ Evaluated |
| **negation** | 1 | N/A | **N/A** | insufficient_samples (n=1 < 3) |
| **emojis** | 3 | 1.0000 | **1.0000** | ✅ Evaluated |
| **code_switching** | 5 | 0.4000 | **0.2667** | ✅ Evaluated |
| **short_text** | 4 | 1.0000 | **1.0000** | ✅ Evaluated |
| **special_tokens** | 2 | N/A | **N/A** | insufficient_samples (n=2 < 3) |

## 5. Visualizations

### Per-Class F1 Performance
```mermaid
xychart-beta
    title "Per-Class F1 Scores"
    x-axis [negative, neutral, positive]
    y-axis "F1-Score" 0.0 --> 1.0
    bar [0.4444, 0.5714, 0.8571]
```

### Slice Macro-F1 Comparison
```mermaid
xychart-beta
    title "Challenge Slice Robustness (Macro-F1)"
    x-axis [spelling_variation, emojis, code_switching, short_text]
    y-axis "Macro-F1" 0.0 --> 1.0
    bar [1.0, 1.0, 0.2667, 1.0]
```

## 6. Error Analysis Sample

Total misclassified instances: **5**

| ID | Gold | Predicted | Conf | Slices | Text |
|---|---|---|---|---|---|
| `cs_002` | `neutral` | `negative` | 0.37 | code_switching | please malai support garnus na. |
| `cs_006` | `neutral` | `negative` | 0.39 | code_switching, special_tokens | mero 9841000000 ma call garna saknu huncha. |
| `ord_003` | `neutral` | `negative` | 0.37 | none | bhat khanu bhayo ta aaja? |
| `cs_007` | `negative` | `positive` | 0.46 | code_switching, special_tokens | delivery charge Rs 150 dherai mahango bhayo. |
| `neg_010` | `negative` | `positive` | 0.38 | negation | sarai jhur kura garyau yar. |