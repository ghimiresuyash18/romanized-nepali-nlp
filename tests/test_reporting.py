"""Unit tests for reporting and visualization utilities."""

from __future__ import annotations

from pathlib import Path

from romanized_nepali_nlp.reporting import (
    generate_charts,
    generate_html_token_gallery,
    generate_preprocessing_diff_gallery,
    generate_v1_coverage_report,
    generate_v1_v2_comparison_report,
    mask_pii,
)


def test_mask_pii():
    text = "call 9841234567 or email info@test.com or ask @anilk"
    masked = mask_pii(text)
    assert "9841234567" not in masked
    assert "info@test.com" not in masked
    assert "@anilk" not in masked
    assert "[PHONE_MASKED]" in masked
    assert "[EMAIL_MASKED]" in masked
    assert "@user_masked" in masked


def test_generate_v1_coverage_report(tmp_path: Path):
    summary = {
        "total_rows": 100,
        "total_tokens": 1000,
        "avg_tokens_per_row": 10.0,
        "unknown_spans": 5,
        "unknown_span_rate": 0.005,
        "uncovered_chars": 0,
        "uncovered_char_rate": 0.0,
        "error_count": 0,
        "categories": {"WORD": 900, "PUNCTUATION": 95, "UNKNOWN": 5},
        "partitions": {
            "roman_clean": {
                "total_rows": 100,
                "total_tokens": 1000,
                "avg_tokens_per_row": 10.0,
                "unknown_spans": 5,
                "unknown_span_rate": 0.005,
                "rows_with_unknown_spans": 3,
                "repeated_tokens_count": 2,
                "long_tokens_count": 1,
                "categories": {"WORD": 900, "PUNCTUATION": 95, "UNKNOWN": 5},
                "token_length_buckets": {"1-3": 300, "4-6": 500, "7-10": 200},
            }
        },
    }
    out_file = tmp_path / "v1_report.md"
    content = generate_v1_coverage_report(summary, out_file)
    assert out_file.exists()
    assert "# Tokenizer Coverage & Profiling Report (v1)" in content
    assert "1,000" in content


def test_generate_charts(tmp_path: Path):
    summary = {
        "partitions": {
            "roman_clean": {
                "unknown_span_rate": 0.002,
                "repeated_tokens_count": 10,
                "categories": {"WORD": 800, "PUNCTUATION": 100},
                "token_length_buckets": {"1-3": 200, "4-6": 400},
            },
            "roman_mixed": {
                "unknown_span_rate": 0.004,
                "repeated_tokens_count": 5,
                "categories": {"WORD": 300, "PUNCTUATION": 50},
                "token_length_buckets": {"1-3": 100, "4-6": 200},
            },
            "roman_noisy": {
                "unknown_span_rate": 0.003,
                "repeated_tokens_count": 25,
                "categories": {"WORD": 100, "PUNCTUATION": 20},
                "token_length_buckets": {"1-3": 50, "4-6": 50},
            },
        }
    }
    charts_dir = tmp_path / "charts"
    generate_charts(summary, charts_dir)

    assert (charts_dir / "token_categories.png").exists()
    assert (charts_dir / "unknown_spans.png").exists()
    assert (charts_dir / "token_lengths.png").exists()
    assert (charts_dir / "special_tokens.png").exists()
    assert (charts_dir / "repetitions.png").exists()


def test_generate_html_token_gallery(tmp_path: Path):
    samples = [{
        "id": "s1",
        "partition": "clean",
        "note": "sample",
        "raw_text": "ramro cha",
        "tokens": [
            {"text": "ramro", "category": "WORD"},
            {"text": "cha", "category": "WORD"},
        ],
    }]
    out_html = tmp_path / "gallery.html"
    generate_html_token_gallery(samples, out_html, "Test Gallery")

    assert out_html.exists()
    html_text = out_html.read_text(encoding="utf-8")
    assert "Test Gallery" in html_text
    assert "ramro" in html_text


def test_generate_preprocessing_diff_gallery(tmp_path: Path):
    cases = [{
        "id": "c1",
        "description": "test case",
        "raw": "ramro cha",
        "unicode_whitespace": "ramro cha",
        "protected_spans": "ramro cha",
        "variant_normalized": "ramro cha",
    }]
    out_html = tmp_path / "diff_gallery.html"
    generate_preprocessing_diff_gallery(cases, out_html)

    assert out_html.exists()
    html_text = out_html.read_text(encoding="utf-8")
    assert "Preprocessing Modes Before / After Gallery" in html_text


def test_generate_v1_v2_comparison_report(tmp_path: Path):
    v1 = {
        "total_rows": 100,
        "total_tokens": 1000,
        "avg_tokens_per_row": 10.0,
        "unknown_spans": 50,
        "unknown_span_rate": 0.05,
        "uncovered_chars": 0,
        "error_count": 0,
        "categories": {"WORD": 900, "UNKNOWN": 50},
        "partitions": {},
    }
    v2 = {
        "total_rows": 100,
        "total_tokens": 1000,
        "avg_tokens_per_row": 10.0,
        "unknown_spans": 5,
        "unknown_span_rate": 0.005,
        "uncovered_chars": 0,
        "error_count": 0,
        "categories": {"WORD": 945, "UNKNOWN": 5},
        "partitions": {},
    }
    out_file = tmp_path / "comp.md"
    report = generate_v1_v2_comparison_report(v1, v2, out_file)

    assert out_file.exists()
    assert "# Tokenizer v1 vs v2 Comparison Report" in report
    assert "-45" in report
