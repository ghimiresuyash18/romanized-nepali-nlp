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
    assert "ENGLISH_WORD" in res.stdout


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
