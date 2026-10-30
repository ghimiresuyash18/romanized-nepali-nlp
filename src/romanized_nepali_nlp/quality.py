"""Duplicate detection, dataset statistics, and quality reports.

Tasks 14 and 15: exact/near-duplicate detection, conflicting labels,
statistics generation, and JSON/Markdown quality reports.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from romanized_nepali_nlp.dataset import DatasetRecord, ValidationIssue

# ---------------------------------------------------------------------------
# Duplicate detection (Task 14)
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _text_hash(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string, truncated to 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_for_dedup(text: str) -> str:
    """Cheap normalisation for near-dedup: lowercase + collapse whitespace.

    shortcut: no character n-gram similarity; add difflib/Levenshtein when
    the dataset exceeds ~50 k records and char-set collisions become common.
    """
    return _WHITESPACE_RE.sub(" ", text.lower().strip())


@dataclass
class DuplicateGroup:
    """A group of records that share identical raw or normalised text."""

    group_id: str  # hash of the canonical key
    key_type: str  # "raw" | "normalized"
    record_ids: List[str]
    labels: List[Optional[str]]
    sources: List[Optional[str]]
    has_label_conflict: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "key_type": self.key_type,
            "size": len(self.record_ids),
            "record_ids": self.record_ids,
            "labels": self.labels,
            "sources": self.sources,
            "has_label_conflict": self.has_label_conflict,
        }


@dataclass
class DuplicateReport:
    """Summary of exact and near-duplicate detection across a record set."""

    raw_groups: List[DuplicateGroup]
    normalized_groups: List[DuplicateGroup]
    # Record IDs with a conflict (different labels among duplicates)
    conflicted_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_duplicate_groups": len(self.raw_groups),
            "raw_duplicate_records": sum(g.size for g in self.raw_groups),
            "normalized_duplicate_groups": len(self.normalized_groups),
            "normalized_duplicate_records": sum(g.size for g in self.normalized_groups),
            "conflicted_record_ids": self.conflicted_ids,
            "raw_groups": [g.to_dict() for g in self.raw_groups],
            "normalized_groups": [g.to_dict() for g in self.normalized_groups],
        }


def detect_duplicates(records: Sequence[DatasetRecord]) -> DuplicateReport:
    """Detect exact raw and normalised duplicate groups among records.

    Groups are deterministic: same input order yields the same group IDs.
    Does NOT delete records; returns groups for human review.
    """
    # --- raw exact duplicates ---
    raw_buckets: Dict[str, List[DatasetRecord]] = defaultdict(list)
    for rec in records:
        raw_buckets[_text_hash(rec.text)].append(rec)

    raw_groups = []
    for bucket_hash, bucket_recs in raw_buckets.items():
        if len(bucket_recs) < 2:
            continue
        labels = [r.label for r in bucket_recs]
        conflict = len({lbl for lbl in labels if lbl is not None}) > 1
        raw_groups.append(
            DuplicateGroup(
                group_id=bucket_hash,
                key_type="raw",
                record_ids=[r.id for r in bucket_recs],
                labels=labels,
                sources=[r.source for r in bucket_recs],
                has_label_conflict=conflict,
            )
        )

    # --- normalised duplicates (excludes already-found raw dups) ---
    # Only records with unique raw text can form a normalised-only dup group
    raw_dup_ids = {rid for g in raw_groups for rid in g.record_ids}
    norm_buckets: Dict[str, List[DatasetRecord]] = defaultdict(list)
    for rec in records:
        norm_buckets[_text_hash(_normalize_for_dedup(rec.text))].append(rec)

    norm_groups = []
    for bucket_hash, bucket_recs in norm_buckets.items():
        # Only report as a normalized group if the raw texts differ for at least 2 members
        if len(bucket_recs) < 2:
            continue
        raw_texts = {r.text for r in bucket_recs}
        if len(raw_texts) == 1:
            continue  # already captured as raw duplicate
        # Also skip if all members are already in raw dup groups
        non_raw_dup = [r for r in bucket_recs if r.id not in raw_dup_ids]
        if len(non_raw_dup) < 2 and len(bucket_recs) - len(non_raw_dup) == len(bucket_recs):
            continue
        labels = [r.label for r in bucket_recs]
        conflict = len({lbl for lbl in labels if lbl is not None}) > 1
        norm_groups.append(
            DuplicateGroup(
                group_id=bucket_hash,
                key_type="normalized",
                record_ids=[r.id for r in bucket_recs],
                labels=labels,
                sources=[r.source for r in bucket_recs],
                has_label_conflict=conflict,
            )
        )

    conflicted_ids = sorted(
        {rid for g in raw_groups + norm_groups if g.has_label_conflict for rid in g.record_ids}
    )

    return DuplicateReport(
        raw_groups=raw_groups,
        normalized_groups=norm_groups,
        conflicted_ids=conflicted_ids,
    )


# ---------------------------------------------------------------------------
# Dataset statistics and quality report (Task 15)
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Aggregate statistics and quality flags for a collection of records."""

    total: int
    by_label: Dict[str, int]
    by_source: Dict[str, int]
    by_split: Dict[str, int]
    missing_label: int
    missing_text: int
    label_conflict_count: int
    raw_duplicate_groups: int
    normalized_duplicate_groups: int
    text_length_chars: Dict[str, float]  # min, max, mean, median
    issues_by_category: Dict[str, int]
    # Token-category slice counts from the tokenizer
    token_slices: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "by_label": self.by_label,
            "by_source": self.by_source,
            "by_split": self.by_split,
            "missing_label": self.missing_label,
            "missing_text": self.missing_text,
            "label_conflict_count": self.label_conflict_count,
            "raw_duplicate_groups": self.raw_duplicate_groups,
            "normalized_duplicate_groups": self.normalized_duplicate_groups,
            "text_length_chars": self.text_length_chars,
            "issues_by_category": self.issues_by_category,
            "token_slices": self.token_slices,
        }

    def to_markdown(self) -> str:
        """Render stats as a Markdown report section."""
        lines = ["# Dataset Quality Report", ""]
        lines.append(f"**Total records:** {self.total}")
        lines.append("")

        lines.append("## Label Distribution")
        lines.append("| Label | Count |")
        lines.append("|---|---|")
        for label, count in sorted(self.by_label.items()):
            lines.append(f"| {label} | {count} |")
        lines.append(f"| *(missing)* | {self.missing_label} |")
        lines.append("")

        lines.append("## Source Distribution")
        lines.append("| Source | Count |")
        lines.append("|---|---|")
        for src, count in sorted(self.by_source.items()):
            lines.append(f"| {src or '(none)'} | {count} |")
        lines.append("")

        if self.by_split:
            lines.append("## Split Distribution")
            lines.append("| Split | Count |")
            lines.append("|---|---|")
            for spl, count in sorted(self.by_split.items()):
                lines.append(f"| {spl} | {count} |")
            lines.append("")

        lines.append("## Text Length (characters)")
        tl = self.text_length_chars
        lines.append(f"- Min: {tl.get('min', 0):.0f}")
        lines.append(f"- Max: {tl.get('max', 0):.0f}")
        lines.append(f"- Mean: {tl.get('mean', 0):.1f}")
        lines.append(f"- Median: {tl.get('median', 0):.1f}")
        lines.append("")

        lines.append("## Quality Flags")
        lines.append(f"- Missing labels: {self.missing_label}")
        lines.append(f"- Missing text: {self.missing_text}")
        lines.append(f"- Label conflicts (within duplicate groups): {self.label_conflict_count}")
        lines.append(f"- Raw duplicate groups: {self.raw_duplicate_groups}")
        lines.append(f"- Normalised duplicate groups: {self.normalized_duplicate_groups}")
        lines.append("")

        if self.issues_by_category:
            lines.append("## Issues by Category")
            lines.append("| Category | Count |")
            lines.append("|---|---|")
            for cat, count in sorted(self.issues_by_category.items()):
                lines.append(f"| {cat} | {count} |")
            lines.append("")

        if self.token_slices:
            lines.append("## Token Slice Counts")
            lines.append("| Slice | Count |")
            lines.append("|---|---|")
            for sl, count in sorted(self.token_slices.items()):
                lines.append(f"| {sl} | {count} |")
            lines.append("")

        return "\n".join(lines)


def _lengths_stats(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    mid = n // 2
    if n % 2 == 0:
        median = (lengths_sorted[mid - 1] + lengths_sorted[mid]) / 2
    else:
        median = float(lengths_sorted[mid])
    return {
        "min": float(lengths_sorted[0]),
        "max": float(lengths_sorted[-1]),
        "mean": round(sum(lengths_sorted) / n, 2),
        "median": round(median, 2),
    }


def compute_stats(
    records: Sequence[DatasetRecord],
    dup_report: Optional[DuplicateReport] = None,
    issues: Optional[Sequence[ValidationIssue]] = None,
    tokenize_slices: bool = True,
) -> DatasetStats:
    """Compute aggregate statistics over a record collection.

    Args:
        records: The loaded (valid) records.
        dup_report: Pre-computed duplicate report; computed internally if None.
        issues: Validation issues from the load phase.
        tokenize_slices: If True, count token categories per record.
    """
    if dup_report is None:
        dup_report = detect_duplicates(records)

    by_label: Dict[str, int] = defaultdict(int)
    by_source: Dict[str, int] = defaultdict(int)
    by_split: Dict[str, int] = defaultdict(int)
    missing_label = 0
    missing_text = 0
    lengths: List[int] = []
    token_slices: Dict[str, int] = defaultdict(int)

    if tokenize_slices:
        # Lazy import to avoid circular dependency; pipeline is in the same package
        from romanized_nepali_nlp.tokenizer import tokenize as _tokenize

    for rec in records:
        if rec.label:
            by_label[rec.label] += 1
        else:
            missing_label += 1
        by_source[rec.source or "(none)"] += 1
        if rec.split:
            by_split[rec.split] += 1
        if not rec.text or not rec.text.strip():
            missing_text += 1
        else:
            lengths.append(len(rec.text))

        if tokenize_slices:
            for tok in _tokenize(rec.text):
                token_slices[tok.category.value] += 1

    issues_by_cat: Dict[str, int] = defaultdict(int)
    for issue in issues or []:
        issues_by_cat[issue.category] += 1

    return DatasetStats(
        total=len(records),
        by_label=dict(by_label),
        by_source=dict(by_source),
        by_split=dict(by_split),
        missing_label=missing_label,
        missing_text=missing_text,
        label_conflict_count=len(dup_report.conflicted_ids),
        raw_duplicate_groups=len(dup_report.raw_groups),
        normalized_duplicate_groups=len(dup_report.normalized_groups),
        text_length_chars=_lengths_stats(lengths),
        issues_by_category=dict(issues_by_cat),
        token_slices=dict(token_slices),
    )


def report_html(stats: DatasetStats, title: str = "Dataset Quality Report") -> str:
    """Render a self-contained HTML quality report."""
    import html as _html

    md_body = _html.escape(stats.to_markdown())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{_html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 2rem; background: #f8fafc; color: #0f172a; }}
    pre {{ white-space: pre-wrap; background: #fff; padding: 1rem;
           border: 1px solid #cbd5e1; border-radius: 6px; }}
  </style>
</head>
<body>
  <pre>{md_body}</pre>
</body>
</html>
"""
