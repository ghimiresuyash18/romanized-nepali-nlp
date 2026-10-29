"""Romanized Nepali NLP Toolkit.

Focused toolkit for Romanized Nepali tokenization, normalization, and sentiment analysis.
"""

from romanized_nepali_nlp.normalization import (
    generate_diff,
    mask_protected_spans,
    normalize_casing,
    normalize_unicode,
    normalize_whitespace,
    reduce_repetitions,
    restore_protected_spans,
)
from romanized_nepali_nlp.pipeline import PipelineConfig, PreprocessingPipeline
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
from romanized_nepali_nlp.variants import VariantNormalizer

__version__ = "0.1.0"

__all__ = [
    "DatasetRecord",
    "PipelineConfig",
    "Prediction",
    "PreprocessingPipeline",
    "PreprocessingResult",
    "ProtectedSpan",
    "Severity",
    "SpanValidationError",
    "Token",
    "TokenCategory",
    "Tokenizer",
    "ValidationIssue",
    "VariantNormalizer",
    "format_html",
    "format_markdown",
    "format_terminal",
    "format_tokens_json",
    "generate_diff",
    "get_diagnostics",
    "get_uncovered_spans",
    "mask_protected_spans",
    "normalize_casing",
    "normalize_unicode",
    "normalize_whitespace",
    "reduce_repetitions",
    "restore_protected_spans",
    "tokenize",
    "validate_token_spans",
]
