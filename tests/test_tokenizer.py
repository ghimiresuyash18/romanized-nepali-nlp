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

    url_tok = tokens[1]
    assert url_tok.text == "https://hamro.com/news"
    assert url_tok.category == TokenCategory.URL

    comma_tok = tokens[2]
    assert comma_tok.text == ","
    assert comma_tok.category == TokenCategory.PUNCTUATION

    email_tok = next(t for t in tokens if t.category == TokenCategory.EMAIL)
    assert email_tok.text == "info@hamro.np"

    excl_tok = tokens[-1]
    assert excl_tok.text == "!"
    assert excl_tok.category == TokenCategory.PUNCTUATION


def test_url_trailing_brackets():
    """Fix 3: closing parens/brackets attached to URLs must be stripped."""
    for text, expected_url in [
        ("(visit https://hamro.com)", "https://hamro.com"),
        ("[link](https://hamro.com)", "https://hamro.com"),
    ]:
        url_tok = next(t for t in tokenize(text) if t.category == TokenCategory.URL)
        assert url_tok.text == expected_url, f"URL not stripped in: {text!r}"


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

    punc_run1 = next(t for t in tokens if t.text == "???")
    assert punc_run1.category == TokenCategory.PUNCTUATION

    punc_run2 = next(t for t in tokens if t.text == "!!!")
    assert punc_run2.category == TokenCategory.PUNCTUATION

    emoji_tokens = [t for t in tokens if t.category == TokenCategory.EMOJI]
    assert len(emoji_tokens) >= 1


def test_alphanumeric_tokens():
    """Fix 1: letter+digit tokens must not explode into UNKNOWN characters."""
    for text, expected_texts in [
        ("iphone15", ["iphone15"]),
        ("SAVE20", ["SAVE20"]),
        ("covid19 update", ["covid19", "update"]),
        ("pulsar220", ["pulsar220"]),
    ]:
        tokens = tokenize(text)
        actual = [t.text for t in tokens]
        assert actual == expected_texts, f"Alphanumeric mismatch for {text!r}: {actual}"
        # All should be WORD (not UNKNOWN)
        for t in tokens:
            assert t.category != TokenCategory.UNKNOWN, f"UNKNOWN token {t.text!r} in {text!r}"


def test_compound_emojis():
    """Fix 2: skin-tone modifiers and ZWJ family sequences must be single tokens."""
    text = "dami 👍🏻 family 👨‍👩‍👦 heart ❤️"
    tokens = tokenize(text)
    emoji_texts = [t.text for t in tokens if t.category == TokenCategory.EMOJI]

    # 👍🏻 (thumbs + skin tone) should be one token, not two
    assert "👍🏻" in emoji_texts, f"Skin-tone emoji split: {emoji_texts}"
    # ZWJ family should not produce ZWJ-only tokens
    zwj_only_tokens = [t for t in tokens if t.text == "\u200d"]
    assert not zwj_only_tokens, f"ZWJ leaked into separate token: {zwj_only_tokens}"


def test_emoticons():
    """Fix 5: ASCII emoticons must survive as single PUNCTUATION tokens."""
    text = "thikai cha :), bholi :D malai cha :("
    tokens = tokenize(text)
    token_texts = [t.text for t in tokens]

    for emo in [":)", ":D", ":("]:
        assert emo in token_texts, f"Emoticon {emo!r} was split. Got: {token_texts}"

    # None of the emoticon characters should become a stray letter token
    stray_d = [t for t in tokens if t.text == "D"]
    assert not stray_d, "':D' split into stray 'D' token"


def test_devanagari_grouping():
    """Fix 6: Devanagari runs must not generate one UNKNOWN token per codepoint."""
    text = "नमस्ते sathi"
    tokens = tokenize(text)
    unknown_tokens = [t for t in tokens if t.category == TokenCategory.UNKNOWN]
    # Should produce at most 1-2 UNKNOWN tokens (one run), not 10
    assert len(unknown_tokens) <= 2, (
        f"Devanagari fragmented into {len(unknown_tokens)} UNKNOWN tokens: "
        f"{[t.text for t in unknown_tokens]}"
    )


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

        actual_by_span = {(t.start, t.end): t for t in actual_tokens}

        for exp in expected_tokens:
            span = (exp["start"], exp["end"])
            if span not in actual_by_span:
                continue
            act = actual_by_span[span]
            if act.text != exp["text"]:
                continue
            # Accept WORD in place of ENGLISH_WORD: audit removed the distinction.
            exp_cat = exp["category"]
            act_cat = act.category.value
            if act_cat == exp_cat or (exp_cat == "ENGLISH_WORD" and act_cat == "WORD"):
                matched += 1

    accuracy = matched / total_expected
    assert accuracy >= 0.95, f"Tokenizer accuracy {accuracy:.2%} is below 95% threshold"
