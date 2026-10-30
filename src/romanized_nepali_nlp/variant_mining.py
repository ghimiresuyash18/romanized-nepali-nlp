"""Spelling variant extraction and mining from Romanized Nepali corpora."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from romanized_nepali_nlp.tokenizer import Tokenizer
from romanized_nepali_nlp.types import TokenCategory

WORD_CHARS = re.compile(r"^[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]+$")
REPEATED_CHARS = re.compile(r"([a-zA-Z])\1{2,}")


@dataclass
class VariantCandidate:
    """A mined spelling variant candidate."""

    surface_form: str
    canonical_form: str
    frequency: int
    source_partition: str
    confidence: float
    category: str
    provenance: str
    review_status: str = "pending_review"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VariantMiner:
    """Mines spelling variant candidates from tokenized corpus partitions."""

    def __init__(self, tokenizer: Optional[Tokenizer] = None) -> None:
        self.tokenizer = tokenizer or Tokenizer()

    def count_vocabulary(
        self,
        jsonl_path: Path,
        max_rows: Optional[int] = None,
    ) -> Counter[str]:
        """Extract lowercase word vocabulary frequencies from a JSONL partition."""
        vocab: Counter[str] = Counter()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_rows and idx >= max_rows:
                    break
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    text = data.get("text", "")
                    tokens = self.tokenizer.tokenize(text)
                    for t in tokens:
                        if t.category == TokenCategory.WORD and WORD_CHARS.match(t.text):
                            vocab[t.text.lower()] += 1
                except Exception:
                    continue
        return vocab

    def mine_candidates(
        self,
        vocab: Counter[str],
        source_partition: str = "roman_clean",
        min_canonical_freq: int = 10,
        known_approved: Optional[Set[str]] = None,
    ) -> List[VariantCandidate]:
        """Mine candidate variants from vocabulary frequencies."""
        approved = known_approved or set()
        candidates: List[VariantCandidate] = []

        # Find words with repetitions (e.g. ramrooo -> ramro)
        for word, count in vocab.items():
            if len(word) < 4:
                continue
            if REPEATED_CHARS.search(word):
                reduced = REPEATED_CHARS.sub(r"\1", word)
                if reduced != word and vocab.get(reduced, 0) >= min_canonical_freq:
                    candidates.append(
                        VariantCandidate(
                            surface_form=word,
                            canonical_form=reduced,
                            frequency=count,
                            source_partition=source_partition,
                            confidence=0.95,
                            category="elongation",
                            provenance="corpus_clean_mixed",
                            review_status="approved" if word in approved else "pending_review",
                            notes="Mined from character elongation run",
                        )
                    )

        # Find common phonetic substitutions (chha/cha, vayo/bhayo, xa/cha)
        phonetic_pairs = [
            (r"^chh", "ch", 0.95, "safe"),
            (r"^v", "bh", 0.92, "safe"),
            (r"^x", "ch", 0.94, "safe"),
            (r"ee", "i", 0.90, "safe"),
            (r"oo", "u", 0.90, "safe"),
        ]

        for word, count in vocab.items():
            if len(word) < 3:
                continue
            for pattern, repl, conf, cat in phonetic_pairs:
                if re.search(pattern, word):
                    target = re.sub(pattern, repl, word)
                    if target != word and vocab.get(target, 0) >= min_canonical_freq:
                        if count < vocab[target]:
                            candidates.append(
                                VariantCandidate(
                                    surface_form=word,
                                    canonical_form=target,
                                    frequency=count,
                                    source_partition=source_partition,
                                    confidence=conf,
                                    category=cat,
                                    provenance="corpus_clean_mixed",
                                    review_status=(
                                    "approved" if word in approved else "pending_review"
                                ),
                                    notes=f"Phonetic candidate: {pattern} -> {repl}",
                                )
                            )

        # Sort by frequency descending
        candidates.sort(key=lambda c: c.frequency, reverse=True)
        return candidates


def save_candidates(candidates: List[VariantCandidate], output_path: Path) -> None:
    """Save variant candidates to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
