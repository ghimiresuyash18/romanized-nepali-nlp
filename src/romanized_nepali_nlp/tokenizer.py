"""Unicode-safe tokenizer for Romanized Nepali text, social tokens, and punctuation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from romanized_nepali_nlp.spans import validate_token_spans
from romanized_nepali_nlp.types import Token, TokenCategory

# Default frequent English loanwords/tokens observed in code-switched Romanized Nepali
DEFAULT_ENGLISH_WORDS: Set[str] = {
    "project",
    "cool",
    "bro",
    "please",
    "support",
    "check",
    "out",
    "for",
    "info",
    "video",
    "call",
    "now",
    "rs",
    "mail",
    "ok",
    "match",
    "wowww",
    "mobile",
    "service",
    "link",
    "phone",
    "update",
    "crash",
    "save",
    "discount",
    "performance",
    "worst",
    "pathetic",
    "amazing",
    "best",
    "purchase",
    "waste",
    "of",
    "time",
    "money",
    "ever",
    "charge",
    "total",
    "unbelievable",
    "customer",
    "experience",
    "smooth",
    "login",
    "contact",
    "news",
    "details",
    "trending",
}

# Regex components
URL_PATTERN = r"https?://[^\s<>\"]+|www\.[^\s<>\"]+"
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
MENTION_PATTERN = r"@[a-zA-Z0-9_]+"
HASHTAG_PATTERN = r"#[a-zA-Z0-9_]+"
NUMBER_PATTERN = r"\b\d+(?:[.,]\d+)*%?\b"
EMOJI_PATTERN = (
    r"[\U00010000-\U0010ffff]|"
    r"[\u2600-\u27ff\u2300-\u23ff\u2b50\u2b55\u200d\ufe0f]+"
)
PUNCTUATION_RUN_PATTERN = (
    r"\?+|!+|(?:\?+!+|!+\?+)+|\.{2,}|"
    r"[.,;:!_?()\[\]{}\"\'\-\/\\—–@#$%^&*+=<>~`|]"
)


class Tokenizer:
    """Configurable scanner for Romanized Nepali, code-switching, and social tokens."""

    def __init__(
        self,
        allow_apostrophes: bool = True,
        allow_hyphens: bool = True,
        extract_hashtag_word: bool = True,
        classify_english: bool = True,
        custom_english_words: Optional[Set[str]] = None,
        validate_spans: bool = True,
    ) -> None:
        self.allow_apostrophes = allow_apostrophes
        self.allow_hyphens = allow_hyphens
        self.extract_hashtag_word = extract_hashtag_word
        self.classify_english = classify_english
        self.validate_spans = validate_spans

        self.english_words = set(DEFAULT_ENGLISH_WORDS)
        if custom_english_words:
            self.english_words.update(w.lower() for w in custom_english_words)

        # Build dynamic word regex based on punctuation configuration
        word_intra = ""
        if allow_apostrophes and allow_hyphens:
            word_intra = r"[\'\-]"
        elif allow_apostrophes:
            word_intra = r"\'"
        elif allow_hyphens:
            word_intra = r"\-"

        # Include basic Latin + Latin Extended-A/B for Roman transliteration diacritics
        latin_chars = r"a-zA-Z\u00C0-\u024F\u0300-\u036F"
        if word_intra:
            self.word_pattern = f"[{latin_chars}]+(?:{word_intra}[{latin_chars}]+)*"
        else:
            self.word_pattern = f"[{latin_chars}]+"

        # Master compiled regex with named groups
        self.master_regex = re.compile(
            f"(?P<EMAIL>{EMAIL_PATTERN})|"
            f"(?P<URL>{URL_PATTERN})|"
            f"(?P<MENTION>{MENTION_PATTERN})|"
            f"(?P<HASHTAG>{HASHTAG_PATTERN})|"
            f"(?P<NUMBER>{NUMBER_PATTERN})|"
            f"(?P<EMOJI>{EMOJI_PATTERN})|"
            f"(?P<WORD>{self.word_pattern})|"
            f"(?P<PUNCTUATION>{PUNCTUATION_RUN_PATTERN})|"
            f"(?P<UNKNOWN>[^\\s])"
        )

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize input text into a sequence of verified Token objects."""
        if not text:
            return []

        tokens: List[Token] = []

        for match in self.master_regex.finditer(text):
            group_name = match.lastgroup or "UNKNOWN"
            raw_text = match.group()
            start, end = match.span()
            metadata: Dict[str, Any] = {}

            # Handle trailing punctuation on URLs (e.g., 'https://hamro.com,')
            if group_name == "URL":
                trailing = re.search(r"[.,!?;:]+$", raw_text)
                if trailing:
                    cut = len(trailing.group())
                    punc_start = end - cut
                    punc_text = raw_text[-cut:]
                    raw_text = raw_text[:-cut]
                    end = punc_start

                    # Emit URL token
                    tokens.append(
                        Token(
                            text=raw_text,
                            category=TokenCategory.URL,
                            start=start,
                            end=end,
                            normalized_text=raw_text,
                            metadata=metadata,
                        )
                    )
                    # Emit trailing punctuation token
                    tokens.append(
                        Token(
                            text=punc_text,
                            category=TokenCategory.PUNCTUATION,
                            start=punc_start,
                            end=punc_start + cut,
                            normalized_text=punc_text,
                            metadata={},
                        )
                    )
                    continue

            category = TokenCategory(group_name)

            if category == TokenCategory.HASHTAG and self.extract_hashtag_word:
                metadata["inner_word"] = raw_text.lstrip("#")

            if category == TokenCategory.WORD and self.classify_english:
                if raw_text.lower() in self.english_words:
                    category = TokenCategory.ENGLISH_WORD

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


# Default singleton instance for quick usage
_DEFAULT_TOKENIZER = Tokenizer()


def tokenize(
    text: str,
    allow_apostrophes: bool = True,
    allow_hyphens: bool = True,
    extract_hashtag_word: bool = True,
    classify_english: bool = True,
) -> List[Token]:
    """Tokenize Romanized Nepali text using default or customized scanner options."""
    if allow_apostrophes and allow_hyphens and extract_hashtag_word and classify_english:
        return _DEFAULT_TOKENIZER.tokenize(text)

    custom_tok = Tokenizer(
        allow_apostrophes=allow_apostrophes,
        allow_hyphens=allow_hyphens,
        extract_hashtag_word=extract_hashtag_word,
        classify_english=classify_english,
    )
    return custom_tok.tokenize(text)
