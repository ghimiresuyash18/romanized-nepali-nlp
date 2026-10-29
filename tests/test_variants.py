"""Unit tests for Romanized Nepali spelling variants and dictionary normalization."""

from romanized_nepali_nlp.variants import VariantNormalizer


def test_variant_normalizer_conservative_policy():
    normalizer = VariantNormalizer(policy="conservative")

    # Safe and elongation mappings should apply
    text = "yo geet raamro chha dheraiii"
    normalized, applied = normalizer.normalize_text(text)

    assert normalized == "yo geet ramro cha dherai"
    assert len(applied) == 3

    # Check details preserved
    applied_map = {item["original"]: item["canonical"] for item in applied}
    assert applied_map["raamro"] == "ramro"
    assert applied_map["chha"] == "cha"
    assert applied_map["dheraiii"] == "dherai"


def test_variant_normalizer_aggressive_policy():
    normalizer_cons = VariantNormalizer(policy="conservative")
    normalizer_agg = VariantNormalizer(policy="aggressive")

    # 'tw' (chat slang for 'ta') and 'tmi' (abbreviation)
    slang_text = "tw kasto chhas tmi"

    # In conservative, 'tw' and 'tmi' are not in safe/elongation
    norm_cons, _ = normalizer_cons.normalize_text(slang_text)
    assert "tmi" in norm_cons

    # In aggressive, 'tw' -> 'ta' and 'tmi' -> 'timi'
    norm_agg, applied = normalizer_agg.normalize_text(slang_text)
    assert "timi" in norm_agg
    assert "ta" in norm_agg


def test_casing_preservation_on_variants():
    normalizer = VariantNormalizer(policy="conservative")

    # TitleCase
    res_title, _ = normalizer.normalize_word("Raamro")
    assert res_title == "Ramro"

    # UPPERCASE
    res_upper, _ = normalizer.normalize_word("CHHA")
    assert res_upper == "CHA"


def test_unregistered_word_untouched():
    normalizer = VariantNormalizer(policy="conservative")
    word, changed = normalizer.normalize_word("anautho")
    assert changed is False
    assert word == "anautho"
