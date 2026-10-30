"""Tests for external corpus ingestion, profiling, quality triage, and leakage audit (Phase 5.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from romanized_nepali_nlp.cli import main
from romanized_nepali_nlp.corpus import (
    CorpusManifest,
    audit_corpus_against_benchmark,
    compute_latin_ratio,
    compute_non_ascii_ratio,
    process_corpus_file,
    profile_and_triage_row,
)
from romanized_nepali_nlp.dataset import DatasetRecord

FIXTURES_DIR = Path(__file__).parent / "fixtures"
NFR_SAMPLE = FIXTURES_DIR / "nfr_sample.jsonl"


def test_latin_ratio_and_non_ascii():
    assert compute_latin_ratio("ramro geet cha") == 1.0
    assert compute_latin_ratio("राम्रो गीत छ") == 0.0
    assert compute_latin_ratio("12345 !!!") == 0.0
    assert compute_latin_ratio("ramro राम्रो") == pytest.approx(0.625, abs=0.01)

    assert compute_non_ascii_ratio("hello") == 0.0
    assert compute_non_ascii_ratio("hello 🔥") > 0.0


def test_profile_and_triage_row():
    # Clean Romanized Nepali
    clean_p = profile_and_triage_row(
        {"text": "yo geet dherai ramro lagyo malai", "source": "youtube"},
        index=1,
    )
    assert clean_p.partition == "roman_clean"
    assert clean_p.latin_ratio == 1.0

    # Mixed code-switching
    mixed_p = profile_and_triage_row(
        {"text": "please subscribe to our channel sathi haru", "source": "youtube"},
        index=2,
    )
    assert mixed_p.partition == "roman_mixed"

    # Noisy repetition / emoji
    noisy_p = profile_and_triage_row(
        {"text": "dammiiiiii film thiyooooo 🔥🔥🔥🔥🔥", "source": "youtube"},
        index=3,
    )
    assert noisy_p.partition == "roman_noisy"

    # Reject / review (Devanagari)
    dev_p = profile_and_triage_row(
        {"text": "यो भिडियो एकदम राम्रो छ धन्यवाद", "source": "youtube"},
        index=4,
    )
    assert dev_p.partition == "reject_or_review"

    # Reject / review (Empty)
    empty_p = profile_and_triage_row({"text": "   "}, index=5)
    assert empty_p.partition == "reject_or_review"


def test_audit_corpus_against_benchmark():
    benchmark_records = [
        DatasetRecord(id="bm_001", text="yo ta sarhai ramro", label="positive"),
        DatasetRecord(id="bm_002", text="aaja ma ghar janchhu", label="neutral"),
    ]

    profiles = [
        profile_and_triage_row({"text": "naya video aayo"}, index=1),
        profile_and_triage_row({"text": "yo ta sarhai ramro"}, index=2),  # exact duplicate
    ]

    safe, quarantined = audit_corpus_against_benchmark(profiles, benchmark_records)

    assert len(safe) == 1
    assert len(quarantined) == 1
    assert quarantined[0]["matched_benchmark_id"] == "bm_001"
    assert profiles[1].partition == "reject_or_review"
    assert "Benchmark overlap" in profiles[1].triage_reason


def test_process_corpus_file_and_cli(tmp_path):
    out_dir = tmp_path / "nfr_processed"
    manifest = CorpusManifest(
        dataset="Boredoom17/Nepali-Flow-Roman",
        revision="test-commit",
        downloaded_at="2026-10-30T11:20:00Z",
        rows_downloaded=23,
        source_license_field="CC BY 4.0",
        upstream_attribution="Aadarsha Chhetri. (2026). Nepali-Flow-Roman.",
        local_sha256="test-hash",
        processing_version="0.1.0",
        notes=["Test snapshot"],
    )

    benchmark_records = [
        DatasetRecord(id="bm_001", text="yo ta sarhai ramro", label="positive"),
    ]

    summary = process_corpus_file(
        NFR_SAMPLE,
        output_dir=out_dir,
        benchmark_records=benchmark_records,
        manifest=manifest,
    )

    assert summary["total_rows"] == 23
    assert summary["leakage_quarantined"] == 1
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "profile.jsonl").exists()
    assert (out_dir / "roman_clean.jsonl").exists()
    assert (out_dir / "roman_mixed.jsonl").exists()
    assert (out_dir / "roman_noisy.jsonl").exists()
    assert (out_dir / "reject_or_review.jsonl").exists()
    assert (out_dir / "leakage_quarantine.jsonl").exists()
    assert (out_dir / "profile_report.md").exists()

    # CLI test
    cli_out = tmp_path / "cli_processed"
    code = main(
        [
            "corpus",
            str(NFR_SAMPLE),
            "-o",
            str(cli_out),
        ]
    )
    assert code == 0
    assert (cli_out / "profile.jsonl").exists()
    assert (cli_out / "profile_report.md").exists()


def test_process_empty_corpus_file(tmp_path: Path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    out_dir = tmp_path / "empty_out"
    summary = process_corpus_file(empty_file, output_dir=out_dir)
    assert summary["total_rows"] == 0
    assert (out_dir / "profile_report.md").exists()
