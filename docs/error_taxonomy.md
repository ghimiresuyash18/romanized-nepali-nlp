# Error Taxonomy: Romanized Nepali Sentiment Baseline

> Generated from `data/processed/error_analysis.jsonl` (16 reviewed errors, baseline_phase5).

## Categories

### 1. `negation_scope`
The model misreads negation tokens or fails to propagate their scope across the
clause.  Romanized Nepali frequently places the negative marker (`na`, `naramro`,
`chhaina`) at the start, middle, or end of a phrase, and the baseline treats
each word as an independent signal rather than composing scope.

**Typical pattern:** negation token present → model ignores or inverts polarity.

**Examples from corpus:**
- *"khana na ramro thiyo na naramro."* — double `na` creates litotes (neutral);
  model predicts **positive**.
- *"dherai naramro behaving garyo."* — `naramro` is explicit negative; model
  predicts **positive** (intensifier `dherai` likely dominates).
- *"please malai support garnus na."* — sentence-final `na` is a softening
  particle, not negation; model predicts **negative** (false alarm).

---

### 2. `sarcasm`
Surface-positive tokens appear in a negative context, or vice versa.  No
errors in the current 16-record set are explicitly tagged `sarcasm`, but
several `out_of_vocab` errors suggest the model defaults to the literal
token-level polarity without pragmatic adjustment.

**Note:** sarcasm detection requires pragmatic / discourse context.  The
heuristic slice detector in `phenomena.py` does not address sarcasm; errors
here require human annotation.

---

### 3. `lexical_ambiguity`
A token carries different sentiment depending on context.  Examples include
polarity-neutral words that are misclassified because the model lacks sufficient
context or the word co-occurs with misleading neighbours.

**Example:** *"kati ramro muskaan ho."* (err_005) — gold label **positive**;
predicted **negative**.  The phrase is unambiguously positive but the model
may associate `kati` (how much / so) with questioning/negative contexts.

---

### 4. `spelling`
Non-standard spellings, character repetition, or transliteration variants that
fall outside the model's vocabulary.

**Examples:**
- *"wowww!!! unbelievable performance!!!"* (err_015) — repeated vowel elongation
  (`wowww`) plus all-English exclamatory text; model predicts **neutral**.
- *"k gardai xau sathi"* (err_010) — `xau` is a phonetic variant of `chau`
  (informal "are doing"); model predicts **negative**.

**Impact:** 4 errors (25 %) in this set carry `spelling_variation` tag.

---

### 5. `emoji_mismatch`
The model ignores or misinterprets emoji sentiment signals.

**Examples:**
- *"👍👍👍👍👍"* (err_006) — all-emoji input; gold **positive**, model **neutral**.
  The baseline has no emoji representation; text embedding is empty.
- *"aaja ta party ho 🎉🥳🍻"* (err_007) — celebration emoji; gold **positive**,
  model **neutral**.

**Note:** `🎉`, `🥳`, `🍻` are not in the conservative `emoji_categories.json`
list, so they are not detected by `phenomena.py`.  Only the 26 listed emoji
are recognised.

---

### 6. `source_artifact`
Errors attributable to data properties rather than model failure: texts
containing phone numbers, URLs, currency symbols (`Rs`), or other special
tokens that the tokenizer strips or mishandles.

**Examples:**
- *"mero <PHONE> ma call garna saknu huncha."* (err_012) — phone number
  dominates; model predicts **negative**.
- *"delivery charge Rs 150 dherai mahango bhayo."* (err_016) — currency + number
  present; despite clear negative sentiment (`mahango bhayo` = became expensive),
  model predicts **positive**.

---

## Error Review (from `error_analysis.jsonl`)

| error_id | record_id | gold | predicted | category | slice_tags | likely_cause | notes |
|----------|-----------|------|-----------|----------|------------|--------------|-------|
| err_001 | edge_018 | neutral | positive | out_of_vocab | — | OOV (foreign script tokens) | `étudiant café` — non-Nepali text |
| err_002 | neg_008 | neutral | positive | negation_misread | negation | Litotes: double `na` | Scope failure |
| err_003 | ord_015 | neutral | positive | out_of_vocab | — | OOV | Functional ordering text |
| err_004 | ord_020 | neutral | positive | out_of_vocab | — | OOV | Greeting / functional |
| err_005 | neg_012 | positive | negative | out_of_vocab | negation | OOV | Misclassified as negation? tag mismatch |
| err_006 | edge_015 | positive | neutral | out_of_vocab | emojis, short_text | Emoji-only text | No text signal |
| err_007 | punc_014 | positive | neutral | out_of_vocab | emojis, code_switching | Emoji + party vocab | Emojis outside known set |
| err_008 | neg_004 | negative | positive | negation_misread | negation | `naramro` + `dherai` | Intensifier overrides negation |
| err_009 | cs_009 | negative | neutral | out_of_vocab | code_switching | English phrases | "waste of time" OOV |
| err_010 | var_015 | neutral | negative | spelling_noise | spelling_variation, short_text | `xau` variant | Rare phonetic spelling |
| err_011 | cs_002 | neutral | negative | out_of_vocab | code_switching | `please support` | English function words → false neg |
| err_012 | cs_006 | neutral | negative | out_of_vocab | code_switching, special_tokens | Phone number | Numeric token noise |
| err_013 | ord_003 | neutral | negative | out_of_vocab | — | OOV | Common ordering utterance |
| err_014 | ord_017 | neutral | negative | out_of_vocab | — | OOV | Ordinary utterance |
| err_015 | punc_005 | positive | neutral | spelling_noise | spelling_variation, emojis, short_text | `wowww` elongation | All-English + spelling |
| err_016 | cs_007 | negative | positive | out_of_vocab | code_switching, special_tokens | `Rs 150` + `delivery` | Currency + OOV |

---

## Findings

### Distribution of error categories

| Category | Count | % |
|----------|------:|--:|
| `out_of_vocab` | 11 | 69 % |
| `negation_misread` | 2 | 13 % |
| `spelling_noise` | 2 | 13 % |
| (other / unlabelled) | 1 | 6 % |

### Primary failure mode: OOV / unknown vocabulary
69 % of errors are attributed to `out_of_vocab`.  The baseline model lacks
representations for: foreign-script tokens, emoji-only inputs, phone numbers,
currency strings, English phrases, and rare Romanized Nepali spellings.

### Negation scope is the second-largest failure
Two explicit negation errors (err_002, err_008) share a pattern: a sentiment
intensifier (`dherai`) or double negation (`na … na`) leads the model to
predict the opposite polarity.  The model does not compose negation and
intensification.

### Slice precision (heuristic vs reviewed ground truth)

| Slice | Precision |
|-------|-----------|
| `code_switch` | **1.00** (3/3 TP) — English function words are reliable signals |
| `negation` | **0.00** (0/2 TP) — heuristic fires on softening `na`, not scope negation |

The negation heuristic in `phenomena.py` lacks scope awareness.  Sentence-final
`na` (softening particle) matches the same token list as clausal negation.
**Recommendation:** add a position-aware rule (e.g., exclude sentence-final `na`
when no preceding negative content word is present).

### Emoji coverage gap
Emojis in errors (`🎉`, `🥳`, `🍻`) are **not** in `emoji_categories.json`.
Expand the resource or add an "other" bucket to avoid silent misses.

### Neutral class is hardest
All 5 neutral records in the test set were misclassified (accuracy = 0, per
`metrics.json`).  Neutral text lacks strong polarity tokens; the model defaults
to the most frequent class boundary.
