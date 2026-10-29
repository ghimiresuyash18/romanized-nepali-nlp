"""Romanized Nepali NLP Toolkit.

Focused toolkit for Romanized Nepali tokenization, normalization, and sentiment analysis.
"""

from romanized_nepali_nlp.spans import (
    SpanValidationError,
    format_html,
    format_markdown,
    format_terminal,
    format_tokens_json,
    get_diagnostics,
    get_uncovered_spans,
    validate_token_spans,
)
from romanized_nepali_nlp.tokenizer import Tokenizer, tokenize
from romanized_nepali_nlp.types import (
    DatasetRecord,
    Prediction,
    PreprocessingResult,
    ProtectedSpan,
    Severity,
    Token,
    TokenCategory,
    ValidationIssue,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetRecord",
    "Prediction",
    "PreprocessingResult",
    "ProtectedSpan",
    "Severity",
    "SpanValidationError",
    "Token",
    "TokenCategory",
    "Tokenizer",
    "ValidationIssue",
    "format_html",
    "format_markdown",
    "format_terminal",
    "format_tokens_json",
    "get_diagnostics",
    "get_uncovered_spans",
    "tokenize",
    "validate_token_spans",
]
