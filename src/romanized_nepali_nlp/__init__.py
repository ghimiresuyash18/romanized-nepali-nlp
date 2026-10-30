"""Romanized Nepali NLP Toolkit.

Focused toolkit for Romanized Nepali tokenization, normalization, and sentiment analysis.
"""

from romanized_nepali_nlp.corpus import (
    CorpusManifest,
    RowProfile,
    audit_corpus_against_benchmark,
    compute_latin_ratio,
    compute_non_ascii_ratio,
    process_corpus_file,
    profile_and_triage_row,
)
from romanized_nepali_nlp.dataset import (
    ColumnMap,
    DatasetRecord,
    LoadResult,
    ValidationIssue,
    ingestion_summary,
    load,
    load_csv,
    load_jsonl,
    preview,
)
from romanized_nepali_nlp.evaluation import (
    ErrorRecord,
    EvaluationReport,
    SliceMetric,
    evaluate_predictions,
)
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
from romanized_nepali_nlp.quality import (
    DatasetStats,
    DuplicateGroup,
    DuplicateReport,
    compute_stats,
    detect_duplicates,
    report_html,
)
from romanized_nepali_nlp.sentiment import (
    Prediction,
    SentimentModel,
    SentimentModelConfig,
    compute_learning_curve,
)
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
from romanized_nepali_nlp.splitting import (
    SplitConfig,
    SplitResult,
    check_leakage,
    split,
    split_balance_report,
)
from romanized_nepali_nlp.tokenizer import Tokenizer, tokenize
from romanized_nepali_nlp.types import (
    PreprocessingResult,
    ProtectedSpan,
    Token,
    TokenCategory,
)
from romanized_nepali_nlp.variants import VariantNormalizer

__version__ = "0.1.0"

__all__ = [
    "ColumnMap",
    "DatasetRecord",
    "DatasetStats",
    "DuplicateGroup",
    "DuplicateReport",
    "LoadResult",
    "PipelineConfig",
    "PreprocessingPipeline",
    "PreprocessingResult",
    "ProtectedSpan",
    "SpanValidationError",
    "SplitConfig",
    "SplitResult",
    "Token",
    "TokenCategory",
    "Tokenizer",
    "ValidationIssue",
    "VariantNormalizer",
    "check_leakage",
    "compute_stats",
    "detect_duplicates",
    "format_html",
    "format_markdown",
    "format_terminal",
    "format_tokens_json",
    "generate_diff",
    "get_diagnostics",
    "get_uncovered_spans",
    "ingestion_summary",
    "load",
    "load_csv",
    "load_jsonl",
    "mask_protected_spans",
    "normalize_casing",
    "normalize_unicode",
    "normalize_whitespace",
    "preview",
    "reduce_repetitions",
    "report_html",
    "restore_protected_spans",
    "split",
    "split_balance_report",
    "tokenize",
    "validate_token_spans",
    "SentimentModel",
    "SentimentModelConfig",
    "Prediction",
    "compute_learning_curve",
    "evaluate_predictions",
    "EvaluationReport",
    "SliceMetric",
    "ErrorRecord",
    "CorpusManifest",
    "RowProfile",
    "audit_corpus_against_benchmark",
    "compute_latin_ratio",
    "compute_non_ascii_ratio",
    "process_corpus_file",
    "profile_and_triage_row",
]
