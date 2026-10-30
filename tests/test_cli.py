"""Subprocess and integration tests for the CLI."""

import json
import subprocess
import sys


def run_cli(*args, input_text=None):
    cmd = [sys.executable, "-m", "romanized_nepali_nlp.cli", *args]
    proc = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
    )
    return proc


def test_cli_help():
    res = run_cli("--help")
    assert res.returncode == 0
    assert "Romanized Nepali NLP Toolkit CLI" in res.stdout


def test_cli_tokenize_positional():
    res = run_cli("tokenize", "ramro cha bro!")
    assert res.returncode == 0
    assert "ramro" in res.stdout
    assert "WORD" in res.stdout
    assert "bro" in res.stdout


def test_cli_tokenize_format_json():
    res = run_cli("tokenize", "kasto ramro 😊", "--format", "json")
    assert res.returncode == 0
    tokens = json.loads(res.stdout)
    assert len(tokens) == 3
    assert tokens[0]["text"] == "kasto"
    assert tokens[2]["text"] == "😊"
    assert tokens[2]["category"] == "EMOJI"


def test_cli_tokenize_format_brackets():
    res = run_cli("tokenize", "k cha?", "--format", "brackets")
    assert res.returncode == 0
    assert res.stdout.strip() == "[k|WORD] [cha|WORD] [?|PUNCTUATION]"


def test_cli_tokenize_stdin():
    res = run_cli("tokenize", "-", input_text="nepal jindabad 🇳🇵")
    assert res.returncode == 0
    assert "nepal" in res.stdout
    assert "jindabad" in res.stdout

    # Test streaming lines with brackets
    res_stream = run_cli(
        "tokenize", "-", "--format", "brackets", input_text="pahilo line\ndosro line"
    )
    assert res_stream.returncode == 0
    lines = res_stream.stdout.strip().splitlines()
    assert len(lines) == 2
    assert lines[0] == "[pahilo|WORD] [line|WORD]"
    assert lines[1] == "[dosro|WORD] [line|WORD]"


def test_cli_tokenize_diagnostics():
    res = run_cli("tokenize", "hello @user ???", "--diagnostics")
    assert res.returncode == 0
    assert "--- Diagnostics ---" in res.stdout
    assert "coverage_ratio" in res.stdout


def test_cli_tokenize_html_export(tmp_path):
    html_file = tmp_path / "tokens.html"
    res = run_cli("tokenize", "dherai ramro thau", "--html", str(html_file))
    assert res.returncode == 0
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "dherai" in content


def test_cli_preprocess_positional():
    res = run_cli("preprocess", "yo ta raamro chha")
    assert res.returncode == 0
    assert res.stdout.strip() == "yo ta ramro cha"


def test_cli_preprocess_format_json():
    res = run_cli("preprocess", "dheraiii maza aayo", "--format", "json")
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["raw_text"] == "dheraiii maza aayo"
    assert data["normalized_text"] == "dherai maza aayo"
    assert "normalize_variants_conservative" in data["steps"]


def test_cli_preprocess_format_diff():
    res = run_cli("preprocess", "raamro", "--format", "diff")
    assert res.returncode == 0
    assert "- raamro" in res.stdout
    assert "+ ramro" in res.stdout


def test_cli_preprocess_stdin():
    res = run_cli("preprocess", "-", input_text="  kasto   chha   ")
    assert res.returncode == 0
    assert res.stdout.strip() == "kasto cha"
