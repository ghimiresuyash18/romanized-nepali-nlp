"""Unit tests for core data types and schema contracts."""

import json

from romanized_nepali_nlp.types import (
    PreprocessingResult,
    ProtectedSpan,
    Token,
    TokenCategory,
)


def test_token_serialization_and_offsets():
    text = "mero saathi ramro cha 😊 #nepal"
    # Test token with unicode & emoji offset
    emoji_token = Token(
        text="😊",
        category=TokenCategory.EMOJI,
        start=22,
        end=23,
        normalized_text="😊",
        metadata={"sub_category": "positive_face"},
    )
    assert text[emoji_token.start : emoji_token.end] == "😊"

    token_dict = emoji_token.to_dict()
    assert token_dict["category"] == "EMOJI"
    assert token_dict["start"] == 22

    # Roundtrip serialization
    json_str = json.dumps(token_dict)
    reloaded_dict = json.loads(json_str)
    restored_token = Token.from_dict(reloaded_dict)

    assert restored_token == emoji_token
    assert restored_token.category == TokenCategory.EMOJI


def test_preprocessing_result_serialization():
    span = ProtectedSpan(start=10, end=27, category="URL", text="https://example.com")
    prep = PreprocessingResult(
        raw_text="Visit us: https://example.com today!",
        normalized_text="visit us: https://example.com today!",
        steps=["unicode_nfc", "lowercase"],
        warnings=[],
        protected_spans=[span],
        metadata={"normalized_by": "default_v1"},
    )
    d = prep.to_dict()
    restored = PreprocessingResult.from_dict(d)
    assert restored.raw_text == prep.raw_text
    assert restored.normalized_text == prep.normalized_text
    assert len(restored.protected_spans) == 1
    assert restored.protected_spans[0].text == "https://example.com"
