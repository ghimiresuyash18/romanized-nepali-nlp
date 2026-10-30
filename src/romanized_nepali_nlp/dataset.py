"""CSV/JSONL dataset loaders, schema validation, and DatasetRecord types.

Tasks 12 and 13: loading, column mapping, row-level errors, and label validation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ALLOWED_LABELS: frozenset[str] = frozenset({"positive", "negative", "neutral"})
MAX_TEXT_LENGTH: int = 10_000  # characters; longer texts are flagged as invalid


# ---------------------------------------------------------------------------
# Core record and issue types (restored from the removed speculative types,
# now with loaders and validation that actually use them)
# ---------------------------------------------------------------------------


@dataclass
class DatasetRecord:
    """A single record in a sentiment dataset."""

    id: str
    text: str
    label: Optional[str] = None  # must be in ALLOWED_LABELS when present
    source: Optional[str] = None
    split: Optional[str] = None  # train / validation / test
    source_file: Optional[str] = None
    row_number: Optional[int] = None
    confidence: Optional[float] = None
    annotator: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetRecord:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            label=data.get("label"),
            source=data.get("source"),
            split=data.get("split"),
            source_file=data.get("source_file"),
            row_number=data.get("row_number"),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            annotator=data.get("annotator"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ValidationIssue:
    """A validation problem detected in a dataset record."""

    severity: str  # "error" | "warning"
    message: str
    row_id: Optional[str] = None
    source_file: Optional[str] = None
    row_number: Optional[int] = None
    field: Optional[str] = None
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoadResult:
    """Output of a dataset load operation."""

    records: List[DatasetRecord]
    issues: List[ValidationIssue]
    source_file: str
    total_rows: int  # raw rows attempted (excludes header)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "total_rows": self.total_rows,
            "loaded": len(self.records),
            "quarantined": self.total_rows - len(self.records),
            "warnings": len([i for i in self.issues if i.severity == "warning"]),
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Column mapping config (Task 12, subtask 3)
# ---------------------------------------------------------------------------


@dataclass
class ColumnMap:
    """Maps source file columns to DatasetRecord fields."""

    text: str = "text"
    label: Optional[str] = "label"
    id: Optional[str] = "id"
    source: Optional[str] = "source"
    confidence: Optional[str] = None
    annotator: Optional[str] = None

    # Optional label remapping: {"1": "positive", "0": "negative"}
    label_map: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Label mapping registry (Task 13, subtask 4)
# ---------------------------------------------------------------------------

LABEL_MAPS: Dict[str, Dict[str, str]] = {
    # numeric sentiment scores (common in Reddit-scraped datasets)
    "numeric_3class": {"1": "positive", "0": "neutral", "-1": "negative"},
    # binary positive/negative
    "binary": {"1": "positive", "0": "negative"},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_label(raw: str, label_map: Dict[str, str]) -> str:
    """Apply label_map then lower-strip; return the canonical label string."""
    mapped = label_map.get(raw, raw)
    return mapped.strip().lower()


def _validate_record(
    rec: DatasetRecord,
    issues: List[ValidationIssue],
) -> bool:
    """Validate a single record; append issues and return True if usable."""
    ok = True

    if not rec.text or not rec.text.strip():
        issues.append(
            ValidationIssue(
                severity="error",
                message="text field is blank or empty",
                row_id=rec.id,
                source_file=rec.source_file,
                row_number=rec.row_number,
                field="text",
                category="blank_text",
            )
        )
        ok = False

    if rec.text and len(rec.text) > MAX_TEXT_LENGTH:
        issues.append(
            ValidationIssue(
                severity="warning",
                message=f"text length {len(rec.text)} exceeds {MAX_TEXT_LENGTH} characters",
                row_id=rec.id,
                source_file=rec.source_file,
                row_number=rec.row_number,
                field="text",
                category="long_text",
            )
        )

    if rec.label is not None and rec.label not in ALLOWED_LABELS:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"label {rec.label!r} not in {sorted(ALLOWED_LABELS)}",
                row_id=rec.id,
                source_file=rec.source_file,
                row_number=rec.row_number,
                field="label",
                category="invalid_label",
            )
        )
        ok = False

    return ok


def _row_to_record(
    row: Dict[str, Any],
    col: ColumnMap,
    source_file: str,
    row_number: int,
) -> Tuple[Optional[DatasetRecord], Optional[ValidationIssue]]:
    """Convert a raw dict row to a DatasetRecord; return (record, None) or (None, issue)."""
    if col.text not in row:
        return None, ValidationIssue(
            severity="error",
            message=f"missing required column '{col.text}'",
            source_file=source_file,
            row_number=row_number,
            field=col.text,
            category="missing_column",
        )

    raw_label = row.get(col.label, "") if col.label else None
    if raw_label is not None:
        raw_label = _coerce_label(str(raw_label), col.label_map) if raw_label else None

    # Build a stable ID from row number if no ID column is configured
    if col.id and col.id in row:
        record_id = str(row[col.id])
    else:
        record_id = f"{Path(source_file).stem}_row{row_number}"

    # Collect unknown columns into metadata
    known_cols = {col.text, col.label, col.id, col.source, col.confidence, col.annotator} - {None}
    extra = {k: v for k, v in row.items() if k not in known_cols}

    confidence_val = None
    if col.confidence and col.confidence in row:
        try:
            confidence_val = float(row[col.confidence])
        except (TypeError, ValueError):
            pass

    rec = DatasetRecord(
        id=record_id,
        text=str(row[col.text]),
        label=raw_label if raw_label else None,
        source=str(row[col.source]) if col.source and col.source in row else None,
        source_file=source_file,
        row_number=row_number,
        confidence=confidence_val,
        annotator=str(row[col.annotator]) if col.annotator and col.annotator in row else None,
        metadata=extra,
    )
    return rec, None


# ---------------------------------------------------------------------------
# Public loaders (Task 12)
# ---------------------------------------------------------------------------


def load_jsonl(
    path: Path | str,
    col: Optional[ColumnMap] = None,
    validate: bool = True,
    source_name: Optional[str] = None,
) -> LoadResult:
    """Load a JSONL file (one JSON object per line) into DatasetRecord objects.

    Args:
        path: Path to the JSONL file.
        col: Column mapping; defaults to standard field names.
        validate: Run schema and label validation on each record.
        source_name: Override the source label (defaults to filename stem).

    Returns:
        LoadResult with records and any validation issues.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as UTF-8.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    col = col or ColumnMap()
    source_file = source_name or path.name
    records: List[DatasetRecord] = []
    issues: List[ValidationIssue] = []
    total_rows = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=f"malformed JSON: {exc}",
                            source_file=source_file,
                            row_number=line_num,
                            category="malformed_row",
                        )
                    )
                    continue

                if not isinstance(row, dict):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message="expected a JSON object, got a non-dict value",
                            source_file=source_file,
                            row_number=line_num,
                            category="malformed_row",
                        )
                    )
                    continue

                rec, err = _row_to_record(row, col, source_file, line_num)
                if err:
                    issues.append(err)
                    continue

                if rec.source is None:
                    rec.source = source_name or path.stem

                if validate:
                    if _validate_record(rec, issues):
                        records.append(rec)
                    # quarantine: do not add invalid record, issue already appended
                else:
                    records.append(rec)

    except UnicodeDecodeError as exc:
        raise ValueError(f"Encoding error reading {path}: {exc}") from exc

    return LoadResult(
        records=records,
        issues=issues,
        source_file=source_file,
        total_rows=total_rows,
    )


def load_csv(
    path: Path | str,
    col: Optional[ColumnMap] = None,
    validate: bool = True,
    source_name: Optional[str] = None,
    delimiter: str = ",",
    quotechar: str = '"',
) -> LoadResult:
    """Load a UTF-8 CSV file into DatasetRecord objects.

    Args:
        path: Path to the CSV file.
        col: Column mapping; defaults to standard field names.
        validate: Run schema and label validation on each record.
        source_name: Override the source label.
        delimiter: CSV delimiter character (default: comma).
        quotechar: CSV quote character (default: double-quote).

    Returns:
        LoadResult with records and any validation issues.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as UTF-8.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    col = col or ColumnMap()
    source_file = source_name or path.name
    records: List[DatasetRecord] = []
    issues: List[ValidationIssue] = []
    total_rows = 0

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
            if reader.fieldnames is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message="CSV file appears empty or has no header",
                        source_file=source_file,
                        category="malformed_row",
                    )
                )
                return LoadResult(records=[], issues=issues, source_file=source_file, total_rows=0)

            for data_row_num, row in enumerate(reader, start=2):  # row 1 is header
                total_rows += 1
                rec, err = _row_to_record(dict(row), col, source_file, data_row_num)
                if err:
                    issues.append(err)
                    continue

                if rec.source is None:
                    rec.source = source_name or path.stem

                if validate:
                    if _validate_record(rec, issues):
                        records.append(rec)
                else:
                    records.append(rec)

    except UnicodeDecodeError as exc:
        raise ValueError(f"Encoding error reading {path}: {exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"CSV parse error in {path}: {exc}") from exc

    return LoadResult(
        records=records,
        issues=issues,
        source_file=source_file,
        total_rows=total_rows,
    )


def load(
    path: Path | str,
    col: Optional[ColumnMap] = None,
    validate: bool = True,
    source_name: Optional[str] = None,
    **kwargs: Any,
) -> LoadResult:
    """Auto-detect CSV or JSONL by extension and load accordingly."""
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return load_jsonl(path, col=col, validate=validate, source_name=source_name)
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        return load_csv(
            path, col=col, validate=validate, source_name=source_name, delimiter=delimiter
        )
    raise ValueError(
        f"Unrecognised extension '{path.suffix}'. Supported: .jsonl, .ndjson, .csv, .tsv"
    )


# ---------------------------------------------------------------------------
# Preview (Task 12, subtask 6)
# ---------------------------------------------------------------------------


def preview(
    result: LoadResult,
    n: int = 5,
) -> str:
    """Return a human-readable preview of a LoadResult as a plain-text card."""
    lines: List[str] = []
    lines.append(f"=== Dataset Preview: {result.source_file} ===")
    lines.append(f"Rows attempted : {result.total_rows}")
    lines.append(f"Loaded         : {len(result.records)}")
    error_count = len([i for i in result.issues if i.severity == "error"])
    warn_count = len([i for i in result.issues if i.severity == "warning"])
    lines.append(f"Errors         : {error_count}")
    lines.append(f"Warnings       : {warn_count}")

    if result.records:
        # Field missingness (Task 12, subtask 6)
        fields = ("label", "source", "confidence", "annotator")
        lines.append("")
        lines.append("--- Missingness ---")
        for fld in fields:
            missing = sum(1 for r in result.records if getattr(r, fld) is None)
            pct = 100 * missing / len(result.records)
            lines.append(f"  {fld:<12}: {missing}/{len(result.records)} missing ({pct:.0f}%)")

        # Schema
        lines.append("")
        lines.append("--- Schema ---")
        first = result.records[0].to_dict()
        for k, v in first.items():
            lines.append(f"  {k}: {type(v).__name__}")

        # First n rows
        lines.append("")
        lines.append(f"--- First {min(n, len(result.records))} Records ---")
        for rec in result.records[:n]:
            label_str = rec.label or "(none)"
            text_snippet = rec.text[:80].replace("\n", " ")
            lines.append(f"  [{rec.id}] label={label_str!r} text={text_snippet!r}")

    if result.issues:
        lines.append("")
        lines.append("--- First 5 Issues ---")
        for issue in result.issues[:5]:
            lines.append(
                f"  [{issue.severity.upper()}] row={issue.row_number} "
                f"field={issue.field} cat={issue.category}: {issue.message}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ingestion summary card (Task 12, subtask 7 / Task 15 partial)
# ---------------------------------------------------------------------------


def ingestion_summary(results: Sequence[LoadResult]) -> Dict[str, Any]:
    """Return a machine-readable ingestion summary card across one or more loads."""
    total_rows = sum(r.total_rows for r in results)
    total_loaded = sum(len(r.records) for r in results)
    total_errors = sum(len([i for i in r.issues if i.severity == "error"]) for r in results)
    total_warnings = sum(len([i for i in r.issues if i.severity == "warning"]) for r in results)

    by_source = []
    for r in results:
        by_source.append(
            {
                "source_file": r.source_file,
                "rows_attempted": r.total_rows,
                "loaded": len(r.records),
                "quarantined": r.total_rows - len(r.records),
                "errors": len([i for i in r.issues if i.severity == "error"]),
                "warnings": len([i for i in r.issues if i.severity == "warning"]),
            }
        )

    return {
        "total_rows_attempted": total_rows,
        "total_loaded": total_loaded,
        "total_quarantined": total_rows - total_loaded,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "sources": by_source,
    }
