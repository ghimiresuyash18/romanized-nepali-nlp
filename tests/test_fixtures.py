"""Tests verifying fixture coverage, provenance contracts, and token slice integrity."""

import json
from collections import Counter
from pathlib import Path

from romanized_nepali_nlp.types import TokenCategory

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_romanized_examples_coverage():
    examples_path = FIXTURES_DIR / "romanized_examples.jsonl"
    assert examples_path.exists(), "romanized_examples.jsonl missing"

    records = []
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Acceptance criteria: At least 100 examples covering all planned edge-case categories
    assert len(records) >= 100, f"Expected >= 100 examples, found {len(records)}"

    category_counts = Counter(r["category"] for r in records)
    required_categories = {
        "ordinary",
        "spelling_variation",
        "code_switching",
        "punctuation_emoji",
        "negation_intensifier",
        "edge_cases",
    }

    assert required_categories.issubset(category_counts.keys())
    for cat in required_categories:
        count = category_counts[cat]
        assert count >= 15, f"Category '{cat}' underrepresented: {count}"

    # Verify required fields
    for r in records:
        assert "id" in r
        assert "text" in r
        assert "category" in r
        assert "expected_label" in r
        assert r["expected_label"] in {"positive", "negative", "neutral"}


def test_expected_tokens_annotations():
    tokens_path = FIXTURES_DIR / "expected_tokens.jsonl"
    assert tokens_path.exists(), "expected_tokens.jsonl missing"

    annotated = []
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotated.append(json.loads(line))

    # Acceptance criteria: At least 30 examples contain reviewed expected annotations
    assert len(annotated) >= 30, f"Expected >= 30 annotated examples, got {len(annotated)}"

    valid_categories = {cat.value for cat in TokenCategory}

    # Verify slice exactness: text[start:end] == token["text"]
    for row in annotated:
        text = row["text"]
        for t in row["tokens"]:
            start = t["start"]
            end = t["end"]
            token_text = t["text"]
            cat = t["category"]

            assert cat in valid_categories, f"Unknown category {cat}"
            assert text[start:end] == token_text, (
                f"Offset mismatch in {row['id']}: expected {token_text!r}, "
                f"got {text[start:end]!r} at [{start}:{end}]"
            )
