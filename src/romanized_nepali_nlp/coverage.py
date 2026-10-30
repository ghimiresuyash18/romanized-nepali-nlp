"""Corpus tokenizer coverage profiler and diagnostic reporter."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from romanized_nepali_nlp.spans import get_uncovered_spans
from romanized_nepali_nlp.tokenizer import Tokenizer
from romanized_nepali_nlp.types import Token, TokenCategory

REPEATED_CHAR_REGEX = re.compile(r"([a-zA-Z])\1{2,}")


@dataclass
class PartitionStats:
    """Summary statistics for a single corpus partition."""

    partition: str
    total_rows: int = 0
    total_tokens: int = 0
    total_chars: int = 0
    unknown_spans: int = 0
    rows_with_unknown_spans: int = 0
    uncovered_chars: int = 0
    rows_with_uncovered_chars: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    repeated_tokens_count: int = 0
    long_tokens_count: int = 0  # length > 30
    token_length_buckets: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        avg_tokens = round(self.total_tokens / self.total_rows, 2) if self.total_rows else 0.0
        avg_chars = round(self.total_chars / self.total_rows, 2) if self.total_rows else 0.0
        unknown_rate = (
            round(self.unknown_spans / self.total_tokens, 6) if self.total_tokens else 0.0
        )
        uncovered_rate = (
            round(self.uncovered_chars / self.total_chars, 6) if self.total_chars else 0.0
        )
        return {
            "partition": self.partition,
            "total_rows": self.total_rows,
            "total_tokens": self.total_tokens,
            "total_chars": self.total_chars,
            "avg_tokens_per_row": avg_tokens,
            "avg_chars_per_row": avg_chars,
            "unknown_spans": self.unknown_spans,
            "rows_with_unknown_spans": self.rows_with_unknown_spans,
            "unknown_span_rate": unknown_rate,
            "uncovered_chars": self.uncovered_chars,
            "rows_with_uncovered_chars": self.rows_with_uncovered_chars,
            "uncovered_char_rate": uncovered_rate,
            "categories": self.categories,
            "repeated_tokens_count": self.repeated_tokens_count,
            "long_tokens_count": self.long_tokens_count,
            "token_length_buckets": self.token_length_buckets,
            "error_count": self.error_count,
        }


def profile_row(
    row_id: str,
    text: str,
    tokenizer: Optional[Tokenizer] = None,
) -> Tuple[Dict[str, Any], List[Token], int, List[str]]:
    """Tokenize and profile a single row."""
    tok = tokenizer or Tokenizer()
    tokens = tok.tokenize(text)

    cat_counts: Dict[str, int] = Counter()
    unknown_count = 0
    unknown_examples: List[str] = []

    for t in tokens:
        cat_counts[t.category.value] += 1
        if t.category == TokenCategory.UNKNOWN:
            unknown_count += 1
            if len(unknown_examples) < 5:
                unknown_examples.append(t.text)

    # Check uncovered non-whitespace characters
    uncovered = get_uncovered_spans(tokens, text)
    uncovered_chars = sum(len(t.strip()) for _, _, t in uncovered if t.strip())

    row_record = {
        "id": row_id,
        "token_count": len(tokens),
        "unknown_span_count": unknown_count,
        "categories": dict(cat_counts),
        "status": "ok",
    }
    return row_record, tokens, uncovered_chars, unknown_examples


def run_partition_coverage(
    input_path: Path,
    output_path: Path,
    errors_path: Path,
    partition_name: str,
    tokenizer: Optional[Tokenizer] = None,
    max_rows: Optional[int] = None,
) -> PartitionStats:
    """Run tokenizer coverage over a single JSONL partition file."""
    stats = PartitionStats(partition=partition_name)
    tok = tokenizer or Tokenizer()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    cat_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()

    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(output_path, "w", encoding="utf-8") as out_f, \
         open(errors_path, "a", encoding="utf-8") as err_f:

        for line_idx, line in enumerate(in_f):
            if max_rows and line_idx >= max_rows:
                break
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
                row_id = data.get("id", f"row_{line_idx}")
                text = data.get("text", "")
            except Exception as e:
                stats.error_count += 1
                err_record = {
                    "partition": partition_name,
                    "line": line_idx,
                    "error": str(e),
                }
                err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                continue

            try:
                row_record, tokens, uncovered_chars, _ = profile_row(row_id, text, tok)
                out_f.write(json.dumps(row_record, ensure_ascii=False) + "\n")

                stats.total_rows += 1
                stats.total_tokens += len(tokens)
                stats.total_chars += len(text)

                for cat, count in row_record["categories"].items():
                    cat_counter[cat] += count

                if row_record["unknown_span_count"] > 0:
                    stats.unknown_spans += row_record["unknown_span_count"]
                    stats.rows_with_unknown_spans += 1

                if uncovered_chars > 0:
                    stats.uncovered_chars += uncovered_chars
                    stats.rows_with_uncovered_chars += 1

                for t in tokens:
                    t_len = len(t.text)
                    if t_len <= 3:
                        bucket_counter["1-3"] += 1
                    elif t_len <= 6:
                        bucket_counter["4-6"] += 1
                    elif t_len <= 10:
                        bucket_counter["7-10"] += 1
                    elif t_len <= 20:
                        bucket_counter["11-20"] += 1
                    elif t_len <= 30:
                        bucket_counter["21-30"] += 1
                    else:
                        bucket_counter["31+"] += 1
                        stats.long_tokens_count += 1

                    if t.category == TokenCategory.WORD and REPEATED_CHAR_REGEX.search(t.text):
                        stats.repeated_tokens_count += 1

            except Exception as e:
                stats.error_count += 1
                err_record = {
                    "partition": partition_name,
                    "id": row_id,
                    "line": line_idx,
                    "error": str(e),
                }
                err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")

    stats.categories = dict(cat_counter)
    stats.token_length_buckets = dict(bucket_counter)
    return stats


def run_full_coverage(
    partitions: Dict[str, Path],
    output_dir: Path,
    tokenizer: Optional[Tokenizer] = None,
    max_rows_per_partition: Optional[int] = None,
) -> Dict[str, Any]:
    """Run coverage profiling across multiple partitions and output summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "tokenizer_errors.jsonl"
    if errors_path.exists():
        errors_path.unlink()

    summaries: Dict[str, Any] = {}
    for part_name, in_path in partitions.items():
        out_path = output_dir / f"{part_name}.jsonl"
        stats = run_partition_coverage(
            input_path=in_path,
            output_path=out_path,
            errors_path=errors_path,
            partition_name=part_name,
            tokenizer=tokenizer,
            max_rows=max_rows_per_partition,
        )
        summaries[part_name] = stats.to_dict()

    # Aggregate overall stats
    total_rows = sum(s["total_rows"] for s in summaries.values())
    total_tokens = sum(s["total_tokens"] for s in summaries.values())
    total_chars = sum(s["total_chars"] for s in summaries.values())
    unknown_spans = sum(s["unknown_spans"] for s in summaries.values())
    uncovered_chars = sum(s["uncovered_chars"] for s in summaries.values())
    errors = sum(s["error_count"] for s in summaries.values())

    overall_categories: Counter[str] = Counter()
    for s in summaries.values():
        for cat, count in s["categories"].items():
            overall_categories[cat] += count

    overall = {
        "total_rows": total_rows,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "avg_tokens_per_row": round(total_tokens / total_rows, 2) if total_rows else 0.0,
        "unknown_spans": unknown_spans,
        "unknown_span_rate": round(unknown_spans / total_tokens, 6) if total_tokens else 0.0,
        "uncovered_chars": uncovered_chars,
        "uncovered_char_rate": round(uncovered_chars / total_chars, 6) if total_chars else 0.0,
        "error_count": errors,
        "categories": dict(overall_categories),
        "partitions": summaries,
    }

    summary_file = output_dir / "coverage_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    return overall
