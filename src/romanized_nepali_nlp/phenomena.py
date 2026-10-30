"""Heuristic metadata module for sentiment-relevant linguistic phenomena.

Detects negation, intensifiers, emojis, and code-switching in Romanized Nepali text.

**All heuristics are approximate.**  Slice membership is assigned by surface-form
pattern matching without syntactic parsing.  False positives and false negatives
are expected, especially for:
- Negation tokens that appear in non-negating contexts (e.g. names, loanwords).
- Code-switch tokens that overlap with rare Romanized Nepali spellings.
- Emojis whose sentiment is highly context-dependent (e.g. 👏 used sarcastically).

This module NEVER modifies the input text; it only returns annotation metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Load emoji categories from the bundled JSON resource
# ---------------------------------------------------------------------------

_RESOURCE_DIR = Path(__file__).parent / "resources"
_EMOJI_CATEGORIES_PATH = _RESOURCE_DIR / "emoji_categories.json"

with _EMOJI_CATEGORIES_PATH.open(encoding="utf-8") as _fh:
    _EMOJI_DATA: dict = json.load(_fh)

# Map each emoji character → category name
_EMOJI_TO_CATEGORY: Dict[str, str] = {}
for _cat, _emojis in _EMOJI_DATA["categories"].items():
    for _e in _emojis:
        _EMOJI_TO_CATEGORY[_e] = _cat

# ---------------------------------------------------------------------------
# Token lists (word-boundary matched, case-insensitive)
# ---------------------------------------------------------------------------

#: Romanized Nepali negation tokens.
NEGATION_TOKENS: List[str] = [
    "na",
    "nai",
    "chhaina",
    "hoina",
    "hudaina",
    "bhayena",
    "nabhayeko",
    "naramro",
    "nabujhne",
    "nagarnu",
    "nalagne",
]

#: Romanized Nepali / mixed intensifier tokens.
INTENSIFIER_TOKENS: List[str] = [
    "dherai",
    "ekdam",
    "ati",
    "sab_bhandaa",
    "alikati",
    "thulo",
    "sano",
    "bahut",
    "dherei",
]

#: English function words used as code-switch indicators.
#: A token matching any of these (when the surrounding text is otherwise
#: Romanized Nepali) is treated as a code-switch signal.
CODE_SWITCH_FUNCTION_WORDS: List[str] = [
    "the", "a", "an",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might", "shall",
    "must", "ought",
    "for", "and", "nor", "but", "or", "yet", "so",
    "please", "support", "call", "delivery", "charge",
]

# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------

def _build_token_pattern(tokens: List[str]) -> re.Pattern[str]:
    """Return a word-boundary pattern for a list of literal tokens."""
    alternation = "|".join(re.escape(t) for t in tokens)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_NEGATION_RE = _build_token_pattern(NEGATION_TOKENS)
_INTENSIFIER_RE = _build_token_pattern(INTENSIFIER_TOKENS)
_CODE_SWITCH_RE = _build_token_pattern(CODE_SWITCH_FUNCTION_WORDS)

# Match any single Unicode emoji or multi-codepoint sequence in the lookup map.
# Build a pattern from the known emojis so we only match what we categorise.
_KNOWN_EMOJIS_RE = re.compile(
    "|".join(re.escape(e) for e in _EMOJI_TO_CATEGORY),
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_phenomena(text: str) -> dict:
    """Detect sentiment-relevant phenomena in *text*. Returns metadata only; does not modify text.

    Parameters
    ----------
    text:
        Raw input string (Romanized Nepali, possibly mixed with English or emoji).

    Returns
    -------
    dict
        A metadata dictionary with the following keys:

        ``has_negation`` *(bool)*
            True if at least one negation token was found.

        ``negation_tokens`` *(list[str])*
            Distinct negation surface-forms found (lower-cased, deduplicated,
            in order of first occurrence).

        ``has_intensifier`` *(bool)*
            True if at least one intensifier token was found.

        ``intensifier_tokens`` *(list[str])*
            Distinct intensifier surface-forms found (lower-cased, deduplicated).

        ``emoji_categories`` *(list[str])*
            Deduplicated list of emoji category names present (e.g. ``"face_positive"``).
            Empty list if no known emoji found.

        ``emoji_list`` *(list[str])*
            Every emoji character found, in order of occurrence (may contain duplicates).

        ``has_code_switch`` *(bool)*
            True if at least one English function/content word from the indicator
            list was found.

        ``code_switch_tokens`` *(list[str])*
            Distinct code-switch indicator tokens found (lower-cased, deduplicated).

    Notes
    -----
    * All heuristics are **approximate** — slice membership should be treated as
      a signal, not ground truth.
    * The function never alters *text*; all matches are read-only.
    * Negation and intensifier detection is word-boundary–based and
      case-insensitive, so ``NA``, ``Na``, and ``na`` all match.
    * Code-switch detection uses only the bundled function-word list; it does
      not call external APIs or language models.
    """
    # --- Negation ---
    neg_matches = [m.group().lower() for m in _NEGATION_RE.finditer(text)]
    neg_unique = list(dict.fromkeys(neg_matches))

    # --- Intensifier ---
    int_matches = [m.group().lower() for m in _INTENSIFIER_RE.finditer(text)]
    int_unique = list(dict.fromkeys(int_matches))

    # --- Emojis ---
    emoji_list: List[str] = _KNOWN_EMOJIS_RE.findall(text)
    cats_seen: List[str] = []
    seen_cats: set = set()
    for em in emoji_list:
        cat = _EMOJI_TO_CATEGORY.get(em)
        if cat and cat not in seen_cats:
            cats_seen.append(cat)
            seen_cats.add(cat)

    # --- Code-switch ---
    cs_matches = [m.group().lower() for m in _CODE_SWITCH_RE.finditer(text)]
    cs_unique = list(dict.fromkeys(cs_matches))

    return {
        "has_negation": bool(neg_unique),
        "negation_tokens": neg_unique,
        "has_intensifier": bool(int_unique),
        "intensifier_tokens": int_unique,
        "emoji_categories": cats_seen,
        "emoji_list": emoji_list,
        "has_code_switch": bool(cs_unique),
        "code_switch_tokens": cs_unique,
    }


__all__ = [
    "detect_phenomena",
    "NEGATION_TOKENS",
    "INTENSIFIER_TOKENS",
    "CODE_SWITCH_FUNCTION_WORDS",
]
