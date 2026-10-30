"""Regression test suite for corpus-derived tokenizer cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from romanized_nepali_nlp.spans import validate_token_spans
from romanized_nepali_nlp.tokenizer import Tokenizer
from romanized_nepali_nlp.types import TokenCategory

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTED_TOKENS_PATH = FIXTURES_DIR / "corpus_expected_tokens.jsonl"


def load_corpus_cases():
    cases = []
    with open(EXPECTED_TOKENS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                cases.append(json.loads(line_str))
    return cases


CORPUS_CASES = load_corpus_cases()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer()


@pytest.mark.parametrize("case", CORPUS_CASES, ids=lambda c: c["id"])
def test_corpus_expected_tokens(tokenizer: Tokenizer, case: dict):
    """Test that corpus-derived cases produce the exact expected tokens and categories."""
    text = case["text"]
    expected = case["expected"]

    tokens = tokenizer.tokenize(text)
    validate_token_spans(tokens, text)

    assert len(tokens) == len(expected), (
        f"Length mismatch for '{text}': got {[t.text for t in tokens]}, "
        f"expected {[e['text'] for e in expected]}"
    )

    for i, (tok, exp) in enumerate(zip(tokens, expected)):
        assert tok.text == exp["text"], (
            f"Token {i} text mismatch: got {tok.text!r}, want {exp['text']!r}"
        )
        assert tok.category.value == exp["category"], (
            f"Token {i} ({tok.text!r}) category mismatch: "
            f"got {tok.category.value}, want {exp['category']}"
        )


def test_core_prd_examples(tokenizer: Tokenizer):
    """Verify exact core examples explicitly listed in PRD Section 3 and Phase 3."""
    # 1. Repeated letters + emoji
    toks = tokenizer.tokenize("ramrooooo 😍")
    validate_token_spans(toks, "ramrooooo 😍")
    assert [t.text for t in toks] == ["ramrooooo", "😍"]
    assert toks[0].category == TokenCategory.WORD
    assert toks[1].category == TokenCategory.EMOJI

    # 2. Negation clause
    toks = tokenizer.tokenize("yo kura ramro lagena")
    assert [t.text for t in toks] == ["yo", "kura", "ramro", "lagena"]
    assert all(t.category == TokenCategory.WORD for t in toks)

    # 3. Currency and numbers
    toks = tokenizer.tokenize("delivery charge Rs 150 dherai mahango bhayo")
    expected_words = ["delivery", "charge", "Rs", "150", "dherai", "mahango", "bhayo"]
    assert [t.text for t in toks] == expected_words
    assert toks[3].category == TokenCategory.NUMBER
    assert toks[2].category == TokenCategory.WORD

    # 4. Code-switching
    toks = tokenizer.tokenize("movie ekdam ramro cha")
    assert [t.text for t in toks] == ["movie", "ekdam", "ramro", "cha"]

    # 5. Social mention
    toks = tokenizer.tokenize("@username")
    assert len(toks) == 1
    assert toks[0].text == "@username"
    assert toks[0].category == TokenCategory.MENTION

    # 6. Social hashtag
    toks = tokenizer.tokenize("#nepal")
    assert len(toks) == 1
    assert toks[0].text == "#nepal"
    assert toks[0].category == TokenCategory.HASHTAG

    # 7. URL
    toks = tokenizer.tokenize("https://example.com")
    assert len(toks) == 1
    assert toks[0].text == "https://example.com"
    assert toks[0].category == TokenCategory.URL

    # 8. Ambiguous particle ta
    toks = tokenizer.tokenize("bhat khanu bhayo ta?")
    assert [t.text for t in toks] == ["bhat", "khanu", "bhayo", "ta", "?"]
    assert toks[3].text == "ta"
    assert toks[3].category == TokenCategory.WORD
