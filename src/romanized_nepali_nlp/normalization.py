"""Unicode, whitespace, casing, and protected-span normalization utilities."""

from __future__ import annotations

import difflib
import re
import unicodedata
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from romanized_nepali_nlp.tokenizer import Tokenizer
from romanized_nepali_nlp.types import ProtectedSpan, TokenCategory

# Default categories considered special/protected
DEFAULT_PROTECTED_CATEGORIES: Set[str] = {
    TokenCategory.URL.value,
    TokenCategory.EMAIL.value,
    TokenCategory.MENTION.value,
    TokenCategory.HASHTAG.value,
    TokenCategory.EMOJI.value,
    TokenCategory.NUMBER.value,
}

# Non-breaking spaces and irregular space characters to convert to standard space
IRREGULAR_SPACES = (
    "\u00a0"  # No-break space
    "\u202f"  # Narrow no-break space
    "\u2000"  # En quad
    "\u2001"  # Em quad
    "\u2002"  # En space
    "\u2003"  # Em space
    "\u2004"  # Three-per-em space
    "\u2005"  # Four-per-em space
    "\u2006"  # Six-per-em space
    "\u2007"  # Figure space
    "\u2008"  # Punctuation space
    "\u2009"  # Thin space
    "\u200a"  # Hair space
)

# Harmless invisible characters to remove outside protected spans
ZERO_WIDTH_SPACE = "\u200b"
ZERO_WIDTH_STRIP_CHARS = (
    "\u200c"  # Zero-width non-joiner
    "\ufeff"  # Byte order mark / zero-width no-break space
)

IRREGULAR_SPACES_REGEX = re.compile(f"[{IRREGULAR_SPACES}]")
ZERO_WIDTH_STRIP_REGEX = re.compile(f"[{ZERO_WIDTH_STRIP_CHARS}]")
DEFAULT_REPETITION_PATTERN = re.compile(r"([a-zA-Z\u00C0-\u024F\u0300-\u036F\u1E00-\u1EFF])\1{2,}")


def normalize_unicode(
    text: str,
    form: str = "NFC",
    strip_zero_width: bool = True,
) -> Tuple[str, List[str], List[str]]:
    """Normalize Unicode representation to NFC or specified form.

    Returns (normalized_text, applied_steps, warnings).
    """
    steps: List[str] = []
    warnings: List[str] = []

    if not text:
        return text, steps, warnings

    # Detect if text had decomposed combining characters
    if not unicodedata.is_normalized(form, text):
        warnings.append(f"Input was not in {form} form; decomposed sequences detected")

    normalized = unicodedata.normalize(form, text)
    if normalized != text:
        steps.append(f"unicode_{form.lower()}")

    # Normalize irregular/non-breaking spaces to standard space
    if IRREGULAR_SPACES_REGEX.search(normalized):
        normalized = IRREGULAR_SPACES_REGEX.sub(" ", normalized)
        steps.append("normalize_irregular_spaces")

    # Normalize zero-width space to standard space and strip non-semantic BOM/non-joiner
    if strip_zero_width:
        if ZERO_WIDTH_SPACE in normalized:
            normalized = normalized.replace(ZERO_WIDTH_SPACE, " ")
            steps.append("normalize_zero_width_space")
        if ZERO_WIDTH_STRIP_REGEX.search(normalized):
            normalized = ZERO_WIDTH_STRIP_REGEX.sub("", normalized)
            steps.append("strip_zero_width")

    return normalized, steps, warnings


def normalize_whitespace(
    text: str,
    preserve_newlines: bool = True,
) -> Tuple[str, bool]:
    """Collapse consecutive spaces while optionally retaining newline boundaries."""
    if not text:
        return text, False

    if preserve_newlines:
        # Collapse multiple horizontal spaces/tabs on each line
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        # Strip trailing/leading blank lines but allow single blank lines
        result = "\n".join(lines).strip()
    else:
        result = re.sub(r"\s+", " ", text).strip()

    changed = result != text
    return result, changed


def normalize_casing(text: str, lowercase: bool = False) -> Tuple[str, bool]:
    """Optionally lowercase text if requested (disabled by default)."""
    if lowercase and text:
        lowered = text.lower()
        return lowered, lowered != text
    return text, False


def reduce_repetitions(
    text: str,
    max_repeats: int = 2,
) -> Tuple[str, bool]:
    """Bound repeated character sequences (e.g. 'ramrooo' -> 'ramroo').

    Restricted to alphabetic letters to preserve numbers and repeated symbols.
    """
    if not text or max_repeats < 1:
        return text, False

    # Match consecutive runs of letters only (excluding digits and punctuation)
    pattern = (
        DEFAULT_REPETITION_PATTERN
        if max_repeats == 2
        else re.compile(
            r"([a-zA-Z\u00C0-\u024F\u0300-\u036F\u1E00-\u1EFF])\1{" + str(max_repeats) + r",}"
        )
    )
    replacement = r"\1" * max_repeats
    result = pattern.sub(replacement, text)
    return result, result != text


def mask_protected_spans(
    text: str,
    tokenizer: Optional[Tokenizer] = None,
    modes: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[ProtectedSpan], Dict[str, Tuple[str, str]]]:
    """Detect special tokens and mask them with collision-resistant placeholders.

    Modes per category:
    - 'preserve': keep original token upon restoration (default).
    - 'replace': leave placeholder e.g. <URL_0> in output.
    - 'remove': remove token entirely from output.

    Returns:
    - masked_text: text with placeholders
    - protected_spans: list of ProtectedSpan metadata
    - placeholder_map: dict of placeholder -> (original_text, category)
    """
    if not text:
        return text, [], {}

    tok = tokenizer or Tokenizer(validate_spans=False)
    tokens = tok.tokenize(text)

    active_modes = modes or {}
    protected_spans: List[ProtectedSpan] = []
    placeholder_map: Dict[str, Tuple[str, str]] = {}

    # Identify tokens to protect (URLs, Emails, Mentions, Hashtags, Numbers, Emojis)
    spans_to_mask = []
    for t in tokens:
        cat_val = t.category.value
        mode = active_modes.get(cat_val, "preserve")
        if cat_val in DEFAULT_PROTECTED_CATEGORIES or cat_val in active_modes:
            if mode != "ignore":
                spans_to_mask.append((t.start, t.end, cat_val, t.text, mode))

    if not spans_to_mask:
        return text, [], {}

    # Sort in ascending order
    spans_to_mask.sort(key=lambda s: s[0])

    # Build masked string with unique per-run digit nonce to prevent collisions with user text
    # and avoid being mangled by reduce_repetitions (which only targets letters).
    nonce = str(uuid.uuid4().int)[:10]
    parts: List[str] = []
    last_idx = 0

    for i, (start, end, cat, original_text, _mode) in enumerate(spans_to_mask):
        placeholder = f"<__PROTECTED_{nonce}_{cat}_{i}__>"
        parts.append(text[last_idx:start])
        parts.append(placeholder)
        last_idx = end

        protected_spans.append(
            ProtectedSpan(
                start=start,
                end=end,
                category=cat,
                text=original_text,
                raw_start=start,
                raw_end=end,
            )
        )
        placeholder_map[placeholder] = (original_text, cat, start, end)

    parts.append(text[last_idx:])
    masked_text = "".join(parts)

    return masked_text, protected_spans, placeholder_map


def restore_protected_spans(
    text: str,
    placeholder_map: Dict[str, Any],
    modes: Optional[Dict[str, str]] = None,
    return_spans: bool = False,
) -> str | Tuple[str, List[ProtectedSpan]]:
    """Restore placeholders according to configured modes (preserve, replace, remove).

    If return_spans=True, returns (restored_text, normalized_protected_spans) where
    span start/end offsets accurately reference positions in the restored text.
    """
    if not placeholder_map:
        return (text, []) if return_spans else text

    active_modes = modes or {}
    result = text
    restored_spans: List[ProtectedSpan] = []

    for placeholder, val in placeholder_map.items():
        original_text = val[0]
        cat = val[1]
        raw_start = val[2] if len(val) > 2 else None
        raw_end = val[3] if len(val) > 3 else None

        mode = active_modes.get(cat, "preserve")
        lower_result = result.lower()
        start_pos = lower_result.find(placeholder.lower())
        if start_pos != -1:
            end_pos = start_pos + len(placeholder)
            if mode == "preserve":
                result = result[:start_pos] + original_text + result[end_pos:]
                restored_spans.append(
                    ProtectedSpan(
                        start=start_pos,
                        end=start_pos + len(original_text),
                        category=cat,
                        text=original_text,
                        raw_start=raw_start,
                        raw_end=raw_end,
                    )
                )
            elif mode == "replace":
                replacement = f"<{cat}>"
                result = result[:start_pos] + replacement + result[end_pos:]
                restored_spans.append(
                    ProtectedSpan(
                        start=start_pos,
                        end=start_pos + len(replacement),
                        category=cat,
                        text=replacement,
                        raw_start=raw_start,
                        raw_end=raw_end,
                    )
                )
            elif mode == "remove":
                # Remove placeholder and adjacent space to prevent orphan double spaces
                if start_pos > 0 and result[start_pos - 1] in " \t":
                    result = result[: start_pos - 1] + result[end_pos:]
                elif end_pos < len(result) and result[end_pos] in " \t":
                    result = result[:start_pos] + result[end_pos + 1 :]
                else:
                    result = result[:start_pos] + result[end_pos:]

    restored_spans.sort(key=lambda s: s.start)

    if return_spans:
        return result, restored_spans
    return result


def generate_diff(raw_text: str, normalized_text: str) -> str:
    """Generate human-readable side-by-side or unified diff string."""
    raw_lines = raw_text.splitlines(keepends=True)
    norm_lines = normalized_text.splitlines(keepends=True)
    if not raw_lines and raw_text:
        raw_lines = [raw_text]
    if not norm_lines and normalized_text:
        norm_lines = [normalized_text]
    diff_lines = list(difflib.ndiff(raw_lines, norm_lines))
    return "".join(diff_lines)
