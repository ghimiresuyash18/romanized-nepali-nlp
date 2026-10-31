"""Tests for emoticon recognition and sentiment feature markers."""

from romanized_nepali_nlp.emoticons import (
    append_emoticon_features,
    find_emoticons,
    load_emoticon_lexicon,
)
from romanized_nepali_nlp.tokenizer import tokenize
from romanized_nepali_nlp.types import TokenCategory


def test_lexicon_is_large_and_categories_are_disjoint():
    lexicon = load_emoticon_lexicon()
    assert len(lexicon["positive"]) >= 50
    assert len(lexicon["negative"]) >= 50
    assert len(lexicon["neutral"]) >= 10

    categories = [set(values) for values in lexicon.values()]
    assert not categories[0] & categories[1]
    assert not categories[0] & categories[2]
    assert not categories[1] & categories[2]


def test_common_emoticons_remain_single_tokens():
    text = ":), -_-, :(, :D, T_T, ¯\\_(ツ)_/¯"
    tokens = tokenize(text)
    assert [token.text for token in tokens if token.category == TokenCategory.PUNCTUATION] == [
        ":)",
        ",",
        "-_-",
        ",",
        ":(",
        ",",
        ":D",
        ",",
        "T_T",
        ",",
        "¯\\_(ツ)_/¯",
    ]


def test_emoticons_add_explicit_sentiment_features():
    text = "ramro :) tara naramro :("
    assert find_emoticons(text) == [(":)", "positive"), (":(", "negative")]
    prepared = append_emoticon_features(text)
    assert "POSITIVE_EMOTICON" in prepared
    assert "NEGATIVE_EMOTICON" in prepared


def test_neutral_emoticon_gets_neutral_feature():
    prepared = append_emoticon_features("¯\\_(ツ)_/¯")
    assert prepared.endswith("NEUTRAL_EMOTICON")
