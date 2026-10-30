# Model Evaluation & Challenge Slice Report

## 1. Summary Performance

- **Total Evaluated Records**: 15
- **Accuracy**: **0.5333** (95% CI: [0.2667, 0.8000])
- **Macro-F1**: **0.4444** (95% CI: [0.2590, 0.5691])
- **Weighted-F1**: **0.4667**

## 2. Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `negative` | 0.3750 | 0.7500 | **0.5000** | 4 |
| `neutral` | 0.0000 | 0.0000 | **0.0000** | 5 |
| `positive` | 0.8333 | 0.8333 | **0.8333** | 6 |

## 3. Confusion Matrix

| Gold \ Predicted | `negative` | `neutral` | `positive` |
|---|---|---|---|
| **`negative`** | 3 | 0 | 1 |
| **`neutral`** | 5 | 0 | 0 |
| **`positive`** | 0 | 1 | 5 |

## 4. Challenge Slice Robustness

| Slice | Count | Accuracy | Macro-F1 | Status |
|---|---|---|---|---|
| **spelling_variation** | 4 | 0.5000 | **0.4444** | ✅ Evaluated |
| **negation** | 1 | N/A | **N/A** | insufficient_samples (n=1 < 3) |
| **emojis** | 3 | 0.6667 | **0.5556** | ✅ Evaluated |
| **code_switching** | 5 | 0.4000 | **0.2667** | ✅ Evaluated |
| **short_text** | 4 | 0.5000 | **0.4444** | ✅ Evaluated |
| **special_tokens** | 2 | N/A | **N/A** | insufficient_samples (n=2 < 3) |

## 5. Visualizations

### Per-Class F1 Performance
```mermaid
xychart-beta
    title "Per-Class F1 Scores"
    x-axis [negative, neutral, positive]
    y-axis "F1-Score" 0.0 --> 1.0
    bar [0.5, 0.0, 0.8333]
```

### Slice Macro-F1 Comparison
```mermaid
xychart-beta
    title "Challenge Slice Robustness (Macro-F1)"
    x-axis [spelling_variation, emojis, code_switching, short_text]
    y-axis "Macro-F1" 0.0 --> 1.0
    bar [0.4444, 0.5556, 0.2667, 0.4444]
```

## 6. Error Analysis Sample

Total misclassified instances: **7**

| ID | Gold | Predicted | Conf | Slices | Text |
|---|---|---|---|---|---|
| `var_015` | `neutral` | `negative` | 0.40 | spelling_variation, short_text | k gardai xau sathi |
| `cs_002` | `neutral` | `negative` | 0.48 | code_switching | please malai support garnus na. |
| `cs_006` | `neutral` | `negative` | 0.41 | code_switching, special_tokens | mero <PHONE> ma call garna saknu huncha. |
| `ord_003` | `neutral` | `negative` | 0.36 | none | bhat khanu bhayo ta aaja? |
| `ord_017` | `neutral` | `negative` | 0.41 | none | kehi bujhina feri bhandinus na. |
| `punc_005` | `positive` | `neutral` | 0.35 | spelling_variation, emojis, short_text | wowww!!! unbelievable performance!!! |
| `cs_007` | `negative` | `positive` | 0.53 | code_switching, special_tokens | delivery charge Rs 150 dherai mahango bhayo. |