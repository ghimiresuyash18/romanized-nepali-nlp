"""Spelling variant and phonetic normalization using curated linguistic resources."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_RESOURCE_PATH = (
    Path(__file__).parent.parent.parent / "resources" / "romanized_variants.json"
)


class VariantNormalizer:
    """Normalize Romanized Nepali spelling variations using versioned lexicons."""

    def __init__(
        self,
        resource_path: Optional[Path] = None,
        policy: str = "conservative",
        min_confidence: float = 0.90,
    ) -> None:
        self.policy = policy
        self.min_confidence = min_confidence
        self.path = resource_path or DEFAULT_RESOURCE_PATH
        self.mappings: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}
        self.variant_details: Dict[str, Dict[str, Any]] = {}

        self.load_resource()

    def load_resource(self) -> None:
        """Load and filter spelling variants according to the configured policy."""
        if not self.path.exists():
            return

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.version = data.get("version", "1.0.0")

        # Policy filtering: conservative vs aggressive
        for item in data.get("variants", []):
            if item.get("review_status") != "approved":
                continue

            conf = float(item.get("confidence", 0.0))
            if conf < self.min_confidence:
                continue

            category = item.get("category", "")
            surface = item["surface_form"].lower()
            canonical = item["canonical_form"]

            if self.policy == "conservative":
                # Only apply safe phonetic and elongation normalizations
                if category in {"safe", "elongation"}:
                    self.mappings[surface] = canonical
                    self.variant_details[surface] = item
            else:
                # Aggressive includes slang, abbreviations, and informal spellings
                self.mappings[surface] = canonical
                self.variant_details[surface] = item

    def normalize_word(self, word: str) -> Tuple[str, bool]:
        """Normalize a single token word, preserving casing if capitalized."""
        lower_word = word.lower()
        if lower_word in self.mappings:
            canonical = self.mappings[lower_word]
            # Match title case or uppercase if word had it
            if word.isupper():
                return canonical.upper(), True
            if word.istitle():
                return canonical.capitalize(), True
            return canonical, True
        return word, False

    def normalize_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Normalize all recognized words in free text while preserving punctuation."""
        if not text or not self.mappings:
            return text, []

        applied: List[Dict[str, Any]] = []

        # Tokenize by word boundaries without disturbing punctuation/spacing
        def replace_match(match: re.Match[str]) -> str:
            w = match.group(0)
            norm_w, changed = self.normalize_word(w)
            if changed:
                applied.append(
                    {
                        "original": w,
                        "canonical": norm_w,
                        "details": self.variant_details.get(w.lower(), {}),
                    }
                )
                return norm_w
            return w

        # Match Latin words including diacritics
        pattern = re.compile(r"\b[a-zA-Z\u00C0-\u024F\u0300-\u036F]+\b")
        result = pattern.sub(replace_match, text)

        return result, applied
