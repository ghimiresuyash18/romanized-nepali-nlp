"""Shared emoticon lexicon and sentiment feature helpers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Dict, List, Tuple

_MARKERS = {
    "positive": "POSITIVE_EMOTICON",
    "negative": "NEGATIVE_EMOTICON",
    "neutral": "NEUTRAL_EMOTICON",
}


@lru_cache(maxsize=1)
def load_emoticon_lexicon() -> Dict[str, List[str]]:
    """Load the packaged emoticon sentiment lexicon."""
    resource = resources.files("romanized_nepali_nlp").joinpath("resources/emoticons.json")
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def emoticon_pattern() -> str:
    """Return a longest-first regex for known emoticons."""
    values = {value for items in load_emoticon_lexicon().values() for value in items}
    return "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))


@lru_cache(maxsize=1)
def emoticon_sentiments() -> Dict[str, str]:
    """Return each emoticon's sentiment category; first category wins overlaps."""
    result: Dict[str, str] = {}
    for category in ("positive", "negative", "neutral"):
        for value in load_emoticon_lexicon()[category]:
            result.setdefault(value, category)
    return result


def find_emoticons(text: str) -> List[Tuple[str, str]]:
    """Find known emoticons and their lexicon sentiment categories."""
    pattern = re.compile(emoticon_pattern())
    sentiments = emoticon_sentiments()
    return [(match.group(), sentiments[match.group()]) for match in pattern.finditer(text)]


def append_emoticon_features(text: str) -> str:
    """Append explicit sentiment markers so emoticons become model features."""
    matches = find_emoticons(text)
    if not matches:
        return text
    markers = " ".join(_MARKERS[category] for _, category in matches)
    return f"{text} {markers}"
