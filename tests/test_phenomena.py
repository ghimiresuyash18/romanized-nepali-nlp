from romanized_nepali_nlp.phenomena import detect_phenomena


def test_detect_phenomena_metadata_and_boundaries():
    text = "please malai support garnus na dherai ramro 🔥"
    result = detect_phenomena(text)

    assert result["has_negation"]
    assert result["negation_tokens"] == ["na"]
    assert result["has_intensifier"]
    assert result["intensifier_tokens"] == ["dherai"]
    assert result["emoji_categories"] == ["fire_strength"]
    assert result["emoji_list"] == ["🔥"]
    assert result["has_code_switch"]
    assert result["code_switch_tokens"] == ["please", "support"]
    assert text == "please malai support garnus na dherai ramro 🔥"


def test_detect_phenomena_uses_word_boundaries():
    result = detect_phenomena("banana kathmandu")

    assert not result["has_negation"]
    assert not result["has_intensifier"]
    assert not result["has_code_switch"]
    assert result["emoji_list"] == []
