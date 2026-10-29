"""Unicode-safe tokenizer for Romanized Nepali text, social tokens, and punctuation."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from romanized_nepali_nlp.spans import validate_token_spans
from romanized_nepali_nlp.types import Token, TokenCategory

# --- Regex patterns in precedence order ---

URL_PATTERN = r"https?://[^\s<>\"]+|www\.[^\s<>\"]+"
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
MENTION_PATTERN = r"@[a-zA-Z0-9_]+"
HASHTAG_PATTERN = r"#[a-zA-Z0-9_]+"
NUMBER_PATTERN = r"\b\d+(?:[.,]\d+)*%?\b"

# Fix 2: compound emojis – match base + skin-tone modifiers + ZWJ sequences as one token.
EMOJI_PATTERN = (
    r"(?:"
    r"[\U00010000-\U0010ffff]|[\u2600-\u27ff\u2300-\u23ff\u2b50\u2b55]"
    r")"
    r"(?:"
    r"\ufe0f?"           # optional variation selector-16
    r"(?:\u200d"         # optional ZWJ + following glyph
    r"(?:[\U00010000-\U0010ffff]|[\u2600-\u27ff\u2300-\u23ff\u2b50\u2b55])"
    r"\ufe0f?"
    r")*"
    r"[\U0001f3fb-\U0001f3ff]?"  # optional skin-tone modifier
    r")"
)

# Fix 5: common ASCII emoticons before generic punctuation so :D / :( survive intact.
EMOTICON_PATTERN = r"(?:[:;=8B][-']?[)D(\[\]/\\OpP]|>?:\s*\()"

PUNCTUATION_RUN_PATTERN = (
    r"\?+|!+|(?:\?+!+|!+\?+)+|\.{2,}|"
    r"[.,;:!_?()\[\]{}\"\'\-\/\\—–@#$%^&*+=<>~`|]"
)

# Fix 1: alphanumeric tokens (iphone15, SAVE20, covid19).
# Must require at least one digit to avoid stealing pure-letter words from WORD.
ALPHANUM_PATTERN = r"[a-zA-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F\u0300-\u036F]*[0-9][a-zA-Z0-9]*"

# Fix 6: group contiguous non-Latin, non-space characters (e.g. Devanagari runs) as
# one UNKNOWN token instead of one per codepoint.
UNKNOWN_PATTERN = r"[^\s]+"


class Tokenizer:
    """Configurable scanner for Romanized Nepali, code-switching, and social tokens."""

    def __init__(
        self,
        allow_apostrophes: bool = True,
        allow_hyphens: bool = True,
        extract_hashtag_word: bool = True,
        # Fix 4: classify_english removed; all Latin words are WORD.
        # shortcut: English vs Nepali classification dropped; revisit if a
        # large labelled English word-list is available.
        validate_spans: bool = False,  # Fix 7: off by default (regex guarantees offsets)
    ) -> None:
        self.allow_apostrophes = allow_apostrophes
        self.allow_hyphens = allow_hyphens
        self.extract_hashtag_word = extract_hashtag_word
        self.validate_spans = validate_spans

        # Build dynamic word regex based on punctuation configuration
        word_intra = ""
        if allow_apostrophes and allow_hyphens:
            word_intra = r"[\'\-]"
        elif allow_apostrophes:
            word_intra = r"\'"
        elif allow_hyphens:
            word_intra = r"\-"

        # Pure-Latin words (no digits) – diacritics included
        latin_chars = r"a-zA-Z\u00C0-\u024F\u0300-\u036F"
        if word_intra:
            pure_word = f"[{latin_chars}]+(?:{word_intra}[{latin_chars}]+)*"
        else:
            pure_word = f"[{latin_chars}]+"

        # Master compiled regex; order is the precedence list.
        # ALPHANUM before WORD: it requires a digit, so pure-letter words (day-to-day)
        # are safe — ALPHANUM won't match them. iphone15/SAVE20 are caught whole.
        self.master_regex = re.compile(
            f"(?P<EMAIL>{EMAIL_PATTERN})|"
            f"(?P<URL>{URL_PATTERN})|"
            f"(?P<MENTION>{MENTION_PATTERN})|"
            f"(?P<HASHTAG>{HASHTAG_PATTERN})|"
            f"(?P<NUMBER>{NUMBER_PATTERN})|"
            f"(?P<EMOJI>{EMOJI_PATTERN})|"
            f"(?P<EMOTICON>{EMOTICON_PATTERN})|"
            f"(?P<ALPHANUM>{ALPHANUM_PATTERN})|"
            f"(?P<WORD>{pure_word})|"
            f"(?P<PUNCTUATION>{PUNCTUATION_RUN_PATTERN})|"
            f"(?P<UNKNOWN>{UNKNOWN_PATTERN})"
        )

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize input text into a sequence of Token objects."""
        if not text:
            return []

        tokens: List[Token] = []

        for match in self.master_regex.finditer(text):
            group_name = match.lastgroup or "UNKNOWN"
            raw_text = match.group()
            start, end = match.span()
            metadata: Dict[str, Any] = {}

            # Fix 3: strip trailing brackets/parens from URLs too
            if group_name == "URL":
                trailing = re.search(r"[.,!?;:()\[\]{}]+$", raw_text)
                if trailing:
                    cut = len(trailing.group())
                    punc_text = raw_text[-cut:]
                    raw_text = raw_text[:-cut]
                    end = end - cut
                    tokens.append(
                        Token(
                            text=raw_text,
                            category=TokenCategory.URL,
                            start=start,
                            end=end,
                            normalized_text=raw_text,
                        )
                    )
                    tokens.append(
                        Token(
                            text=punc_text,
                            category=TokenCategory.PUNCTUATION,
                            start=end,
                            end=end + cut,
                            normalized_text=punc_text,
                        )
                    )
                    continue

            # EMOTICON and ALPHANUM map to existing TokenCategory values
            if group_name == "EMOTICON":
                category = TokenCategory.PUNCTUATION
            elif group_name == "ALPHANUM":
                category = TokenCategory.WORD
            else:
                category = TokenCategory(group_name)

            if category == TokenCategory.HASHTAG and self.extract_hashtag_word:
                metadata["inner_word"] = raw_text.lstrip("#")

            tokens.append(
                Token(
                    text=raw_text,
                    category=category,
                    start=start,
                    end=end,
                    normalized_text=raw_text.lower(),
                    metadata=metadata,
                )
            )

        if self.validate_spans:
            validate_token_spans(tokens, text)

        return tokens


# Default singleton (validate_spans=False for throughput)
_DEFAULT_TOKENIZER = Tokenizer()


def tokenize(
    text: str,
    allow_apostrophes: bool = True,
    allow_hyphens: bool = True,
    extract_hashtag_word: bool = True,
) -> List[Token]:
    """Tokenize Romanized Nepali text."""
    if allow_apostrophes and allow_hyphens and extract_hashtag_word:
        return _DEFAULT_TOKENIZER.tokenize(text)
    return Tokenizer(
        allow_apostrophes=allow_apostrophes,
        allow_hyphens=allow_hyphens,
        extract_hashtag_word=extract_hashtag_word,
    ).tokenize(text)
