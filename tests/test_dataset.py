"""Tests for dataset loading, validation, quality reporting, and splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from romanized_nepali_nlp.dataset import (
    ColumnMap,
    DatasetRecord,
    ingestion_summary,
    load,
    load_csv,
    load_jsonl,
    preview,
)
from romanized_nepali_nlp.quality import (
    compute_stats,
    detect_duplicates,
    report_html,
)
from romanized_nepali_nlp.splitting import (
    SplitConfig,
    SplitResult,
    check_leakage,
    split,
    split_balance_report,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALID_JSONL = FIXTURES_DIR / "valid_dataset.jsonl"
VALID_CSV = FIXTURES_DIR / "valid_dataset.csv"
MALFORMED_JSONL = FIXTURES_DIR / "malformed_dataset.jsonl"


# ---------------------------------------------------------------------------
# Task 12 & 13: Loading & Validation
# ---------------------------------------------------------------------------


def test_load_jsonl_valid():
    res = load_jsonl(VALID_JSONL)
    assert res.total_rows == 10
    assert len(res.records) == 10
    assert len(res.issues) == 0
    assert res.records[0].id == "rec1"
    assert res.records[0].text == "ramro chha yo geet"
    assert res.records[0].label == "positive"
    assert res.records[0].source == "youtube"
    assert res.records[0].row_number == 1


def test_load_csv_valid():
    res = load_csv(VALID_CSV)
    assert res.total_rows == 10
    assert len(res.records) == 10
    assert len(res.issues) == 0
    assert res.records[0].id == "rec1"
    assert res.records[0].label == "positive"
    assert res.records[3].id == "rec4"
    assert "Dherai" in res.records[3].text


def test_load_auto_detect():
    res_jsonl = load(VALID_JSONL)
    assert len(res_jsonl.records) == 10
    res_csv = load(VALID_CSV)
    assert len(res_csv.records) == 10

    with pytest.raises(ValueError, match="Unrecognised extension"):
        load("dataset.parquet")


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_jsonl("non_existent_file.jsonl")


def test_load_malformed_jsonl_and_quarantine():
    res = load_jsonl(MALFORMED_JSONL)
    assert res.total_rows == 5
    # Only m1 is valid; m2 (bad json), m3 (blank text), m4 (invalid label), m5 (no text)
    assert len(res.records) == 1
    assert res.records[0].id == "m1"
    assert len(res.issues) == 4

    categories = {issue.category for issue in res.issues}
    assert "malformed_row" in categories
    assert "blank_text" in categories
    assert "invalid_label" in categories
    assert "missing_column" in categories

    # Invariant: accepted + error issues == total_rows
    error_issues = [i for i in res.issues if i.severity == "error"]
    assert len(res.records) + len(error_issues) == res.total_rows


def test_label_map_applied(tmp_path: Path):
    file_path = tmp_path / "custom.jsonl"
    file_path.write_text(
        json.dumps({"id": "1", "text": "mitho chha", "score": "1"})
        + "\n"
        + json.dumps({"id": "2", "text": "bekar", "score": "-1"})
        + "\n",
        encoding="utf-8",
    )
    col = ColumnMap(
        text="text",
        label="score",
        id="id",
        label_map={"1": "positive", "-1": "negative"},
    )
    res = load_jsonl(file_path, col=col)
    assert len(res.records) == 2
    assert res.records[0].label == "positive"
    assert res.records[1].label == "negative"


def test_preview_output():
    res = load_jsonl(VALID_JSONL)
    text = preview(res, n=3)
    assert "=== Dataset Preview: valid_dataset.jsonl ===" in text
    assert "Rows attempted : 10" in text
    assert "Loaded         : 10" in text
    assert "--- First 3 Records ---" in text


def test_ingestion_summary():
    res1 = load_jsonl(VALID_JSONL)
    res2 = load_jsonl(MALFORMED_JSONL)
    summary = ingestion_summary([res1, res2])
    assert summary["total_rows_attempted"] == 15
    assert summary["total_loaded"] == 11
    assert summary["total_quarantined"] == 4
    assert summary["total_errors"] == 4
    assert len(summary["sources"]) == 2


# ---------------------------------------------------------------------------
# Task 14: Duplicate Detection
# ---------------------------------------------------------------------------


def test_detect_duplicates_raw_and_conflicts():
    res = load_jsonl(VALID_JSONL)
    dup_report = detect_duplicates(res.records)

    # rec1 and rec2 have exact same text
    # rec5 and rec6 have exact same text but different labels (positive vs negative)
    assert len(dup_report.raw_groups) == 2

    # Conflicted IDs must include rec5 and rec6
    assert "rec5" in dup_report.conflicted_ids
    assert "rec6" in dup_report.conflicted_ids
    assert "rec1" not in dup_report.conflicted_ids

    # Normalized duplicates: rec3 and rec4 (whitespace/casing difference)
    assert len(dup_report.normalized_groups) == 1
    norm_grp = dup_report.normalized_groups[0]
    assert set(norm_grp.record_ids) == {"rec3", "rec4"}
    assert not norm_grp.has_label_conflict


# ---------------------------------------------------------------------------
# Task 15: Quality Reports and Statistics
# ---------------------------------------------------------------------------


def test_compute_stats():
    res = load_jsonl(VALID_JSONL)
    stats = compute_stats(res.records, issues=res.issues)
    assert stats.total == 10
    assert stats.by_label["positive"] == 4
    assert stats.by_label["negative"] == 4
    assert stats.by_label["neutral"] == 2
    assert stats.label_conflict_count == 2
    assert stats.raw_duplicate_groups == 2
    assert stats.normalized_duplicate_groups == 1
    assert "mean" in stats.text_length_chars
    assert "WORD" in stats.token_slices

    # Markdown rendering
    md = stats.to_markdown()
    assert "## Label Distribution" in md
    assert "## Source Distribution" in md
    assert "## Quality Flags" in md

    # HTML rendering
    html = report_html(stats)
    assert "<!DOCTYPE html>" in html
    assert "Dataset Quality Report" in html


# ---------------------------------------------------------------------------
# Task 16: Reproducible, Leakage-Safe Splitting
# ---------------------------------------------------------------------------


def test_split_reproducible():
    res = load_jsonl(VALID_JSONL)
    cfg1 = SplitConfig(train=0.7, validation=0.15, test=0.15, seed=42)
    s1 = split(res.records, cfg1)

    cfg2 = SplitConfig(train=0.7, validation=0.15, test=0.15, seed=42)
    s2 = split(res.records, cfg2)

    assert [r.id for r in s1.train] == [r.id for r in s2.train]
    assert [r.id for r in s1.validation] == [r.id for r in s2.validation]
    assert [r.id for r in s1.test] == [r.id for r in s2.test]
    assert s1.dataset_hash == s2.dataset_hash


def test_split_group_aware():
    res = load_jsonl(VALID_JSONL)
    cfg = SplitConfig(train=0.7, validation=0.15, test=0.15, seed=42, group_by_duplicates=True)
    res_split = split(res.records, cfg)

    # Check that duplicates land in the exact same split
    def get_split(rid: str) -> str:
        for name in ("train", "validation", "test"):
            if any(r.id == rid for r in res_split.by_name(name)):
                return name
        return "none"

    assert get_split("rec1") == get_split("rec2")
    assert get_split("rec3") == get_split("rec4")
    assert get_split("rec5") == get_split("rec6")


def test_split_transitive_duplicate_groups():
    # Chain of duplicates: d1==d2 (raw), d2==d3 (normalized)
    records = [
        DatasetRecord(id="d1", text="ramro film", label="positive"),
        DatasetRecord(id="d2", text="ramro film", label="positive"),
        DatasetRecord(id="d3", text="ramro   film", label="positive"),
        DatasetRecord(id="u1", text="unique one", label="negative"),
        DatasetRecord(id="u2", text="unique two", label="negative"),
    ]
    cfg = SplitConfig(train=0.5, validation=0.25, test=0.25, seed=42, group_by_duplicates=True)
    res_split = split(records, cfg)

    def get_split(rid: str) -> str:
        for name in ("train", "validation", "test"):
            if any(r.id == rid for r in res_split.by_name(name)):
                return name
        return "none"

    assert get_split("d1") == get_split("d2") == get_split("d3")
    # Verify input records were not mutated in place
    assert records[0].split is None


def test_check_leakage():
    res = load_jsonl(VALID_JSONL)
    res_split = split(res.records)
    leaked = check_leakage(res_split)
    assert leaked == []

    # Inject deliberate leakage
    corrupted = SplitResult(
        train=list(res_split.train) + [res_split.test[0]],
        validation=list(res_split.validation),
        test=list(res_split.test),
        config=res_split.config,
        dataset_hash=res_split.dataset_hash,
    )
    leaked_ids = check_leakage(corrupted)
    assert leaked_ids == [res_split.test[0].id]


def test_split_config_invalid():
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        SplitConfig(train=0.5, validation=0.5, test=0.5)


def test_split_manifest_and_balance_report(tmp_path: Path):
    res = load_jsonl(VALID_JSONL)
    res_split = split(res.records)

    manifest_path = tmp_path / "manifest.json"
    res_split.save_manifest(manifest_path)
    assert manifest_path.exists()

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["dataset_hash"] == res_split.dataset_hash
    assert "counts" in manifest_data
    assert "label_distribution" in manifest_data
    assert "record_ids" in manifest_data

    report = split_balance_report(res_split)
    assert "## Split Balance Report" in report
    assert "| Label | Train | Validation | Test |" in report


# ---------------------------------------------------------------------------
# CLI Dataset Commands
# ---------------------------------------------------------------------------


def test_cli_dataset_commands(tmp_path: Path):
    import subprocess
    import sys

    # preview
    cmd_preview = [
        sys.executable,
        "-m",
        "romanized_nepali_nlp.cli",
        "dataset",
        "preview",
        str(VALID_JSONL),
    ]
    p1 = subprocess.run(cmd_preview, capture_output=True, text=True)
    assert p1.returncode == 0
    assert "Dataset Preview" in p1.stdout

    # report (json)
    cmd_report = [
        sys.executable,
        "-m",
        "romanized_nepali_nlp.cli",
        "dataset",
        "report",
        str(VALID_JSONL),
        "--format",
        "json",
    ]
    p2 = subprocess.run(cmd_report, capture_output=True, text=True)
    assert p2.returncode == 0
    stats_data = json.loads(p2.stdout)
    assert stats_data["total"] == 10

    # split
    out_dir = tmp_path / "splits"
    manifest_file = tmp_path / "split_manifest.json"
    cmd_split = [
        sys.executable,
        "-m",
        "romanized_nepali_nlp.cli",
        "dataset",
        "split",
        str(VALID_JSONL),
        "--out-dir",
        str(out_dir),
        "--manifest",
        str(manifest_file),
    ]
    p3 = subprocess.run(cmd_split, capture_output=True, text=True)
    assert p3.returncode == 0
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "validation.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
    assert manifest_file.exists()
