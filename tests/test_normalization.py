"""Unit tests for Unicode, whitespace, casing, and protected-span normalization."""

from romanized_nepali_nlp.normalization import (
    generate_diff,
    mask_protected_spans,
    normalize_casing,
    normalize_unicode,
    normalize_whitespace,
    reduce_repetitions,
    restore_protected_spans,
)


def test_unicode_nfc_and_decomposed_warning():
    # Composed vs decomposed 'ā' (U+0101 vs 'a' + U+0304)
    decomposed = "nep" + "a\u0304" + "l\u012b"
    normalized, steps, warnings = normalize_unicode(decomposed, form="NFC")

    assert normalized == "nepālī"
    assert "unicode_nfc" in steps
    assert any("decomposed sequences detected" in w for w in warnings)

    # Re-normalizing already NFC text should produce no steps or warnings
    nfc_again, steps2, warnings2 = normalize_unicode(normalized, form="NFC")
    assert nfc_again == normalized
    assert len(steps2) == 0
    assert len(warnings2) == 0


def test_irregular_spaces_and_zero_width():
    # Text with non-breaking space and zero-width space
    raw = "ramro\u00a0cha\u200bhai"
    cleaned, steps, _ = normalize_unicode(raw)
    assert cleaned == "ramro cha hai"
    assert "normalize_irregular_spaces" in steps
    assert "normalize_zero_width_space" in steps


def test_whitespace_collapsing_and_newlines():
    text = "  pahilo   line   \n\n   dosro   line  "

    # Preserving newlines
    preserved, changed = normalize_whitespace(text, preserve_newlines=True)
    assert changed is True
    assert preserved == "pahilo line\n\ndosro line"

    # Flattening newlines
    flattened, changed = normalize_whitespace(text, preserve_newlines=False)
    assert changed is True
    assert flattened == "pahilo line dosro line"


def test_casing_normalization():
    text = "RaAmRo Kura"
    # Default False
    unchanged, c1 = normalize_casing(text, lowercase=False)
    assert unchanged == text
    assert c1 is False

    # Lowercase True
    lowered, c2 = normalize_casing(text, lowercase=True)
    assert lowered == "raamro kura"
    assert c2 is True


def test_bounded_repetition_reduction():
    text = "ramrooo dheraiii geet"
    reduced, changed = reduce_repetitions(text, max_repeats=2)
    assert changed is True
    assert reduced == "ramroo dheraii geet"

    # No repeats over limit
    unchanged, c = reduce_repetitions("ramro cha", max_repeats=2)
    assert c is False
    assert unchanged == "ramro cha"


def test_protected_spans_preserve_mode():
    text = "Visit https://hamro.com/news, @anil_k for #Nepal 🇳🇵"
    masked, protected_spans, p_map = mask_protected_spans(text)

    # Placeholders should be present in masked text
    assert "<__PROTECTED_URL_0__>" in masked
    assert "<__PROTECTED_MENTION_1__>" in masked
    assert "<__PROTECTED_HASHTAG_2__>" in masked
    assert "<__PROTECTED_EMOJI_3__>" in masked
    assert len(protected_spans) == 4

    # Restoration in preserve mode recovers original text
    restored = restore_protected_spans(masked, p_map, modes={"URL": "preserve"})
    assert restored == text


def test_protected_spans_replace_and_remove_modes():
    text = "Call 9841000000 or email info@hamro.np"
    modes = {"NUMBER": "replace", "EMAIL": "remove"}
    masked, _, p_map = mask_protected_spans(text, modes=modes)

    restored = restore_protected_spans(masked, p_map, modes=modes)
    assert "<NUMBER>" in restored
    assert "9841000000" not in restored
    assert "info@hamro.np" not in restored


def test_generate_diff():
    diff = generate_diff("raamro", "ramro")
    assert "- raamro" in diff
    assert "+ ramro" in diff
