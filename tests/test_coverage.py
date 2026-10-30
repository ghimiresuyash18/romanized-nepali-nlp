"""Unit tests for corpus tokenizer coverage profiling."""

from __future__ import annotations

import json
from pathlib import Path

from romanized_nepali_nlp.coverage import (
    profile_row,
    run_full_coverage,
    run_partition_coverage,
)
from romanized_nepali_nlp.tokenizer import Tokenizer


def test_profile_row_basic():
    tok = Tokenizer()
    row, tokens, uncovered, unk = profile_row("r1", "yo ramro cha 😍", tok)

    assert row["id"] == "r1"
    assert row["token_count"] == 4
    assert row["unknown_span_count"] == 0
    assert row["status"] == "ok"
    assert row["categories"]["WORD"] == 3
    assert row["categories"]["EMOJI"] == 1
    assert uncovered == 0
    assert len(unk) == 0


def test_profile_row_with_unknown():
    tok = Tokenizer()
    row, tokens, uncovered, unk = profile_row("r2", "nepali \u0600\u0601 noise", tok)

    assert row["id"] == "r2"
    assert row["unknown_span_count"] >= 1
    assert row["categories"].get("UNKNOWN", 0) >= 1
    assert len(unk) >= 1


def test_run_partition_coverage(tmp_path: Path):
    in_file = tmp_path / "in.jsonl"
    out_file = tmp_path / "out.jsonl"
    err_file = tmp_path / "err.jsonl"

    data = [
        {"id": "c1", "text": "ramro cha"},
        {"id": "c2", "text": "yo kura thik chaina"},
        {"id": "c3", "text": "malformed invalid"},
    ]
    with open(in_file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
        f.write("invalid json line\n")

    stats = run_partition_coverage(
        input_path=in_file,
        output_path=out_file,
        errors_path=err_file,
        partition_name="test_part",
    )

    assert stats.total_rows == 3
    assert stats.error_count == 1
    assert out_file.exists()
    assert err_file.exists()

    # Verify output rows
    with open(out_file) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 3
    assert lines[0]["id"] == "c1"
    assert lines[0]["status"] == "ok"


def test_run_full_coverage(tmp_path: Path):
    part_a = tmp_path / "part_a.jsonl"
    part_b = tmp_path / "part_b.jsonl"

    with open(part_a, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a1", "text": "ramro kura"}) + "\n")
    with open(part_b, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "b1", "text": "mitho khana #nepal"}) + "\n")

    parts = {"part_a": part_a, "part_b": part_b}
    out_dir = tmp_path / "out_coverage"

    summary = run_full_coverage(parts, out_dir)

    assert summary["total_rows"] == 2
    assert summary["total_tokens"] == 5
    assert (out_dir / "part_a.jsonl").exists()
    assert (out_dir / "part_b.jsonl").exists()
    assert (out_dir / "coverage_summary.json").exists()
