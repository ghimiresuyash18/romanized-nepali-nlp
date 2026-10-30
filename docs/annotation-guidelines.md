# Annotation Guidelines: Romanized Nepali Sentiment Analysis

Version: `v1.0.0-frozen`  
Benchmark status: **Frozen**  
Task: Classify Romanized Nepali social and web text into one of three sentiment classes: `positive`, `negative`, or `neutral`.

---

## 1. Class Definitions & Canonical Examples

| Class | Definition | Romanized Nepali Examples |
|---|---|---|
| **`positive`** | Clear expressions of approval, happiness, satisfaction, praise, gratitude, optimism, love, or encouragement. | - `ramro chha yo geet` (Good song)<br/>- `yo thau sachhai sundar raicha` (This place is truly beautiful)<br/>- `dhanyabad hajurko madat ko lagi` (Thank you for your help)<br/>- `dherai khusi lagyo sathi` (Very happy my friend) |
| **`negative`** | Clear expressions of sadness, anger, frustration, complaint, criticism, disappointment, insult, or pain. | - `sarai naramro film` (Very bad movie)<br/>- `mero phone harayo k garne hola` (Lost phone, what to do)<br/>- `kathmandu ma dherai dhulo cha` (Too much dust in Kathmandu)<br/>- `kasto hawa manche` (What an awful/useless person) |
| **`neutral`** | Purely informational text, facts, announcements, routine scheduling, casual questions without emotional charge, or objective descriptions. | - `aaja ma ghar janchhu` (Today I go home)<br/>- `namaste sathi haru k cha khabar?` (Hello friends what's the news?)<br/>- `aaja college bida cha re` (College is reportedly closed today)<br/>- `chhito aaunus samaya bhai sakyo` (Come fast it is already time) |

---

## 2. Linguistic Edge Cases

### 2.1 Negation
- Negation shifts or reverses polarity:
  - `ramro` (positive) -> `ramro chhaina` / `naramro` -> **`negative`**.
  - `khusi chu` (positive) -> `khusi chhaina` -> **`negative`**.
  - `dukhdaayi chaina` (not sorrowful) -> **`neutral`** or **`positive`** depending on context.

### 2.2 Intensifiers
- Intensifiers amplify existing polarity without changing class:
  - `sarai ramro`, `ekdum babbal`, `dherai dammi` -> **`positive`**.
  - `ekdam bakwaas`, `sarai jhyau`, `ati dherai dhulo` -> **`negative`**.

### 2.3 Profanity & Slang
- Romanized slurs or derogatory slang (`muji`, `radi`, `hawa`, `kukur`) are **`negative`** when targeted at people or products.
- Playful banter among close peers without hostility: label **`neutral`** if no negative intent is detected, or flag low confidence.

### 2.4 Emojis & Emoticons
- Emojis indicate strong emotion and often disambiguate text:
  - Smiling faces (`:)`, `😊`, `🔥`, `❤️`) support **`positive`**.
  - Crying/angry faces (`:(`, `😭`, `😡`, `🤮`) support **`negative`**.
  - Neutral icons (`📍`, `📅`, `🚗`) remain **`neutral`**.

### 2.5 Questions
- Pure information-seeking questions (`bhat khanu bhayo ta?`, `kata tira parcha?`) -> **`neutral`**.
- Rhetorical questions expressing frustration (`kin yeti hawa bhako?`) -> **`negative`**.
- Enthusiastic rhetorical questions (`kasto babbal geet haina ra?`) -> **`positive`**.

### 2.6 Code-Switching (English + Romanized Nepali)
- Code-switched sentences retain overall sentiment regardless of language:
  - `movie ekdum super hit lagyo` -> **`positive`**.
  - `service completely worst thiyo` -> **`negative`**.
  - `meeting at 4pm ma aau hai` -> **`neutral`**.

---

## 3. Mixed Sentiment & Sarcasm Rules

### 3.1 Mixed Sentiment Rule (Recency / Concession Rule)
- When a text contains both positive and negative clauses:
  - If linked by contrastive conjunctions (`tara`, `but`, `bhaneta pani`), the **final concluding clause** dictates the dominant sentiment:
    - *Example*: `acting ramro thiyo tara story ekdam bakwaas` -> **`negative`** (concludes on bad story).
    - *Example*: `budget dherai lagyo tara movie babbal banyo` -> **`positive`** (concludes on great movie).
  - If equal balanced pros and cons without dominance: label **`neutral`** with confidence `<= 0.70`.

### 3.2 Sarcasm / Irony Rule
- Sarcasm occurs when positive wording conveys negative sentiment:
  - *Example*: `wah kasto ramro service 3 ghanta parkhaiyo 👏` -> **`negative`**.
  - When sarcastic intent is clear from context, emoji, or contradiction, assign the **intended (underlying) sentiment** (usually `negative`), not literal words.
  - Set confidence to `< 0.8` to document annotator hesitation.

---

## 4. Excluded / Uncertain Examples & Confidence Field

### 4.1 Excluded Examples
- Exclude if text is:
  1. Gibberish / keyboard mash (`asdfghjk`, `qwerty123`).
  2. Pure advertisements / spam with no user reaction.
  3. Languages other than Nepali/English (e.g. Hindi, Bhojpuri).
  4. Non-text symbols only.

### 4.2 Annotator Confidence Field
- Every annotation must supply `confidence` between `0.0` and `1.0`:
  - `1.0`: Completely clear, unambiguous text.
  - `0.8`: Typical confident label with minor noise.
  - `0.6`: Ambiguous, mixed, or sarcastic text.
  - `< 0.5`: High uncertainty (adjudication required).

---

## 5. Frozen Release Specification
- Version: `v1.0.0-frozen`
- Changes require major version bump (`v2.0.0`) and full re-annotation of benchmark test sets.
