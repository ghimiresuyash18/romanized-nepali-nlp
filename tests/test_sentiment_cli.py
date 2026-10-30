"""Integration tests for sentiment CLI commands: train, predict, evaluate (Task 20)."""

from __future__ import annotations

import json
from pathlib import Path

from romanized_nepali_nlp.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ROMANIZED_EXAMPLES = FIXTURES_DIR / "romanized_examples.jsonl"


def test_cli_train_majority(tmp_path):
    run_dir = tmp_path / "majority_run"
    code = main(
        [
            "train",
            str(ROMANIZED_EXAMPLES),
            "--label-col",
            "expected_label",
            "--model-type",
            "majority",
            "-o",
            str(run_dir),
        ]
    )
    assert code == 0
    assert (run_dir / "model.pkl").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "train_log.txt").exists()


def test_cli_train_word_and_predict(tmp_path):
    run_dir = tmp_path / "word_run"
    train_code = main(
        [
            "train",
            str(ROMANIZED_EXAMPLES),
            "--label-col",
            "expected_label",
            "--model-type",
            "word",
            "-o",
            str(run_dir),
        ]
    )
    assert train_code == 0
    assert (run_dir / "model.pkl").exists()
    assert (run_dir / "val_report.json").exists()

    # Predict on test data
    pred_out = tmp_path / "preds.jsonl"
    pred_code = main(
        [
            "predict",
            str(ROMANIZED_EXAMPLES),
            "--model",
            str(run_dir / "model.pkl"),
            "--label-col",
            "expected_label",
            "-o",
            str(pred_out),
        ]
    )
    assert pred_code == 0
    assert pred_out.exists()

    with open(pred_out, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) > 0
    assert "predicted_label" in lines[0]
    assert "scores" in lines[0]


def test_cli_train_char_with_config(tmp_path):
    cfg_file = tmp_path / "custom_config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "model_type": "char",
                "analyzer": "char_wb",
                "ngram_range": [2, 4],
                "C": 0.5,
                "seed": 123,
            }
        )
    )

    run_dir = tmp_path / "char_run"
    train_code = main(
        [
            "train",
            str(ROMANIZED_EXAMPLES),
            "--label-col",
            "expected_label",
            "--config",
            str(cfg_file),
            "-o",
            str(run_dir),
        ]
    )
    assert train_code == 0
    assert (run_dir / "model.pkl").exists()
    assert (run_dir / "top_features.json").exists()

    # Test evaluate command
    eval_out = tmp_path / "eval_report.md"
    eval_code = main(
        [
            "evaluate",
            str(run_dir / "val_predictions.jsonl"),
            "--dataset",
            str(ROMANIZED_EXAMPLES),
            "--format",
            "markdown",
            "-o",
            str(eval_out),
            "--export-errors",
            str(tmp_path / "errors.jsonl"),
        ]
    )
    assert eval_code == 0
    assert eval_out.exists()
    assert (tmp_path / "errors.jsonl").exists()


def test_cli_predict_missing_file_returns_error(tmp_path):
    code = main(["predict", str(tmp_path / "non_existent.csv"), "--model", "some_model.pkl"])
    assert code == 1
