"""Comprehensive unit tests for Romanized Nepali tokenizer and special token recognition."""

import json
from pathlib import Path

from romanized_nepali_nlp.tokenizer import Tokenizer, tokenize
from romanized_nepali_nlp.types import TokenCategory

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_repeated_letters_and_casing():
    text = "yo ta ramrooo cha RaAmRo"
    tokens = tokenize(text)
    words = [t.text for t in tokens]
    assert words == ["yo", "ta", "ramrooo", "cha", "RaAmRo"]
    # Check normalized text preserves lowercase
    assert tokens[2].normalized_text == "ramrooo"
    assert tokens[4].normalized_text == "raamro"
    assert tokens[4].text == "RaAmRo"


def test_accented_latin_characters():
    text = "nepālī mānchhe"
    tokens = tokenize(text)
    assert len(tokens) == 2
    assert tokens[0].text == "nepālī"
    assert tokens[0].category == TokenCategory.WORD
    assert tokens[1].text == "mānchhe"


def test_url_and_trailing_punctuation():
    text = "Visit https://hamro.com/news, or email info@hamro.np!"
    tokens = tokenize(text)

    # URL should strip trailing comma
    url_tok = tokens[1]
    assert url_tok.text == "https://hamro.com/news"
    assert url_tok.category == TokenCategory.URL

    comma_tok = tokens[2]
    assert comma_tok.text == ","
    assert comma_tok.category == TokenCategory.PUNCTUATION

    # Email
    email_tok = tokens[5]
    assert email_tok.text == "info@hamro.np"
    assert email_tok.category == TokenCategory.EMAIL

    # Trailing exclamation
    excl_tok = tokens[6]
    assert excl_tok.text == "!"
    assert excl_tok.category == TokenCategory.PUNCTUATION


def test_mentions_and_hashtags():
    text = "@anil_k post garyo #NepalPride"
    tokens = tokenize(text)

    assert tokens[0].text == "@anil_k"
    assert tokens[0].category == TokenCategory.MENTION

    assert tokens[3].text == "#NepalPride"
    assert tokens[3].category == TokenCategory.HASHTAG
    assert tokens[3].metadata.get("inner_word") == "NepalPride"


def test_punctuation_runs_and_emojis():
    text = "kina yesto??? wowww!!! 🔥❤️"
    tokens = tokenize(text)

    punc_run1 = [t for t in tokens if t.text == "???"][0]
    assert punc_run1.category == TokenCategory.PUNCTUATION

    punc_run2 = [t for t in tokens if t.text == "!!!"][0]
    assert punc_run2.category == TokenCategory.PUNCTUATION

    emoji_tokens = [t for t in tokens if t.category == TokenCategory.EMOJI]
    assert len(emoji_tokens) >= 1


def test_hyphens_and_apostrophes_configuration():
    text = "day-to-day don't"

    # Default: keep hyphen and apostrophe inside word
    tok_default = tokenize(text)
    assert [t.text for t in tok_default] == ["day-to-day", "don't"]

    # Without hyphens
    tok_no_hyphen = Tokenizer(allow_hyphens=False).tokenize(text)
    assert [t.text for t in tok_no_hyphen] == ["day", "-", "to", "-", "day", "don't"]

    # Without apostrophes
    tok_no_apostrophe = Tokenizer(allow_apostrophes=False).tokenize(text)
    assert [t.text for t in tok_no_apostrophe] == ["day-to-day", "don", "'", "t"]


def test_accuracy_against_expected_tokens_fixture():
    fixture_path = FIXTURES_DIR / "expected_tokens.jsonl"
    with open(fixture_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    total_expected = 0
    matched = 0

    for case in cases:
        text = case["text"]
        expected_tokens = case["tokens"]
        actual_tokens = tokenize(text)

        total_expected += len(expected_tokens)

        # Map by (start, end)
        actual_by_span = {(t.start, t.end): t for t in actual_tokens}

        for exp in expected_tokens:
            span = (exp["start"], exp["end"])
            if span in actual_by_span:
                act = actual_by_span[span]
                if act.text == exp["text"] and act.category.value == exp["category"]:
                    matched += 1

    accuracy = matched / total_expected
    # Acceptance criterion: Reviewed fixture accuracy >= 95%
    assert accuracy >= 0.95, f"Tokenizer accuracy {accuracy:.2%} is below 95% threshold"
