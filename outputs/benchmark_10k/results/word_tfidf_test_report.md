# Model Evaluation & Challenge Slice Report

## 1. Summary Performance

- **Total Evaluated Records**: 1495
- **Accuracy**: **0.7204** (95% CI: [0.6990, 0.7418])
- **Macro-F1**: **0.6831** (95% CI: [0.6596, 0.7092])
- **Weighted-F1**: **0.7232**

## 2. Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `negative` | 0.5753 | 0.6141 | **0.5941** | 311 |
| `neutral` | 0.6209 | 0.6501 | **0.6352** | 403 |
| `positive` | 0.8421 | 0.7990 | **0.8200** | 781 |

## 3. Confusion Matrix

| Gold \ Predicted | `negative` | `neutral` | `positive` |
|---|---|---|---|
| **`negative`** | 191 | 67 | 53 |
| **`neutral`** | 77 | 262 | 64 |
| **`positive`** | 64 | 93 | 624 |

## 4. Challenge Slice Robustness

| Slice | Count | Accuracy | Macro-F1 | Status |
|---|---|---|---|---|
| **spelling_variation** | 37 | 0.6486 | **0.5627** | ✅ Evaluated |
| **negation** | 41 | 0.6829 | **0.5278** | ✅ Evaluated |
| **emojis** | 6 | 1.0000 | **1.0000** | ✅ Evaluated |
| **code_switching** | 300 | 0.7967 | **0.6433** | ✅ Evaluated |
| **short_text** | 116 | 0.7931 | **0.6848** | ✅ Evaluated |
| **special_tokens** | 151 | 0.7285 | **0.7210** | ✅ Evaluated |

## 5. Visualizations

### Per-Class F1 Performance
```mermaid
xychart-beta
    title "Per-Class F1 Scores"
    x-axis [negative, neutral, positive]
    y-axis "F1-Score" 0.0 --> 1.0
    bar [0.5941, 0.6352, 0.82]
```

### Slice Macro-F1 Comparison
```mermaid
xychart-beta
    title "Challenge Slice Robustness (Macro-F1)"
    x-axis [spelling_variation, negation, emojis, code_switching, short_text, special_tokens]
    y-axis "Macro-F1" 0.0 --> 1.0
    bar [0.5627, 0.5278, 1.0, 0.6433, 0.6848, 0.721]
```

## 6. Error Analysis Sample

Total misclassified instances: **418**

| ID | Gold | Predicted | Conf | Slices | Text |
|---|---|---|---|---|---|
| `nfr-0267837` | `positive` | `negative` | 0.42 | none | AaThi Nai Dame touching |
| `nfr-0171815` | `positive` | `neutral` | 0.42 | none | Nawaj Ansari ON fire after coming in Nepal |
| `nfr-0076227` | `positive` | `negative` | 0.41 | none | Karma vai le atti diyotw aaja tw astinai ko v |
| `nfr-0030235` | `positive` | `negative` | 0.49 | none | dil ko Chhu Lene wali Kahani Hai. Doston Ham  |
| `nfr-0085917` | `positive` | `neutral` | 0.46 | none | All the boys look like k-drama main lead... A |
| `nfr-0057698` | `positive` | `neutral` | 0.53 | none | Like many Indians... I like Nepal and Russia  |
| `nfr-0155250` | `positive` | `negative` | 0.49 | none | Mahabir pun vneko yo des ko prabidi ko king h |
| `nfr-0115364` | `positive` | `neutral` | 0.40 | none | Da ak bit. Ak.bit. Ak din sun nu hunxa. Mero  |
| `nfr-0063514` | `positive` | `neutral` | 0.46 | none | Khola gang myachiss . Day by day yo dai ko so |
| `nfr-0173983` | `positive` | `neutral` | 0.44 | none | Album ni chaiyo xitoo Yota matra intro ho |
| `nfr-0020916` | `positive` | `neutral` | 0.42 | none | yarr sach me handsome hote hai yarr muzhe lag |
| `nfr-0026898` | `positive` | `negative` | 0.41 | code_switching, special_tokens | Deshko change ko lagi yetiko content banayera |
| `nfr-0058993` | `positive` | `neutral` | 0.49 | negation | Raju dai lok gitko ta raja ho Tara kaiile nad |
| `nfr-0125969` | `positive` | `neutral` | 0.45 | none | maile ne star maker bata thaha payar here KO  |
| `nfr-0251296` | `positive` | `neutral` | 0.51 | none | Bro is also rocking in rap |
| `nfr-0070477` | `positive` | `negative` | 0.50 | none | Tapaile yesko version lai nai change gardinu  |
| `nfr-0148135` | `positive` | `negative` | 0.47 | none | Punm ko voes ramro k grnu vot dine hru le mis |
| `nfr-0273067` | `positive` | `negative` | 0.36 | none | la aba majja auni bhayo |
| `nfr-0149700` | `positive` | `neutral` | 0.57 | none | Aap ka video bahot achchha lagta hai bhai ham |
| `nfr-0046764` | `positive` | `neutral` | 0.36 | none | Aati sudar lagyo ..KTM basi lai jumla ko maul |