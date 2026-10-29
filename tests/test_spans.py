"""Tests for token span validation, span diagnostics, and visualizations."""

import pytest

from romanized_nepali_nlp.spans import (
    SpanValidationError,
    format_html,
    format_markdown,
    format_terminal,
    format_tokens_json,
    get_diagnostics,
    get_uncovered_spans,
    validate_token_spans,
)
from romanized_nepali_nlp.types import Token, TokenCategory


def test_valid_spans():
    text = "mero saathi ramro cha"
    tokens = [
        Token(text="mero", category=TokenCategory.WORD, start=0, end=4),
        Token(text="saathi", category=TokenCategory.WORD, start=5, end=11),
        Token(text="ramro", category=TokenCategory.WORD, start=12, end=17),
        Token(text="cha", category=TokenCategory.WORD, start=18, end=21),
    ]
    validate_token_spans(tokens, text)


def test_reject_overlapping_spans():
    text = "mero saathi"
    tokens = [
        Token(text="mero", category=TokenCategory.WORD, start=0, end=4),
        Token(text="ro", category=TokenCategory.WORD, start=2, end=4),  # overlap!
    ]
    with pytest.raises(SpanValidationError, match="overlaps with previous token"):
        validate_token_spans(tokens, text)


def test_reject_backward_span():
    text = "mero saathi"
    tokens = [
        Token(text="mero", category=TokenCategory.WORD, start=4, end=2),  # start >= end
    ]
    with pytest.raises(SpanValidationError, match="backward or zero-length"):
        validate_token_spans(tokens, text)


def test_reject_out_of_bounds_span():
    text = "ramro"
    tokens = [
        Token(text="ramro", category=TokenCategory.WORD, start=0, end=10),
    ]
    with pytest.raises(SpanValidationError, match="exceeds text length"):
        validate_token_spans(tokens, text)


def test_reject_slice_text_mismatch():
    text = "ramro cha"
    tokens = [
        Token(text="naramro", category=TokenCategory.WORD, start=0, end=5),
    ]
    with pytest.raises(SpanValidationError, match="text mismatch"):
        validate_token_spans(tokens, text)


def test_uncovered_spans_and_diagnostics():
    text = "k cha @user ???"
    tokens = [
        Token(text="k", category=TokenCategory.WORD, start=0, end=1),
        Token(text="cha", category=TokenCategory.WORD, start=2, end=5),
        Token(text="@user", category=TokenCategory.MENTION, start=6, end=11),
    ]
    uncovered = get_uncovered_spans(tokens, text)
    # Gaps should include spaces and trailing ' ???'
    assert any("???" in gap[2] for gap in uncovered)

    diag = get_diagnostics(tokens, text)
    assert diag["token_count"] == 3
    assert diag["uncovered_non_whitespace_count"] == 1
    assert diag["uncovered_non_whitespace"][0]["text"] == "???"


def test_format_terminal_and_markdown():
    tokens = [
        Token(text="k", category=TokenCategory.WORD, start=0, end=1),
        Token(text="?", category=TokenCategory.PUNCTUATION, start=1, end=2),
    ]
    term = format_terminal(tokens)
    assert term == "[k|WORD] [?|PUNCTUATION]"

    md = format_markdown(tokens)
    assert "| `k` | `WORD` |" in md
    assert "| `?` | `PUNCTUATION` |" in md

    j = format_tokens_json(tokens)
    assert '"text": "k"' in j


def test_format_html():
    text = "namaste 🙏"
    tokens = [
        Token(text="namaste", category=TokenCategory.WORD, start=0, end=7),
        Token(text="🙏", category=TokenCategory.EMOJI, start=8, end=9),
    ]
    html_output = format_html(tokens, text, title="Demo Test")
    assert "<!DOCTYPE html>" in html_output
    assert "namaste" in html_output
    assert "🙏" in html_output
    assert "WORD" in html_output
    assert "EMOJI" in html_output
