"""Unit tests for evaluation reports, challenge slices, and error analysis (Task 21)."""

from __future__ import annotations

import pytest

from romanized_nepali_nlp.dataset import DatasetRecord
from romanized_nepali_nlp.evaluation import (
    evaluate_predictions,
    is_code_switching,
    is_emoji,
    is_negation,
    is_short_text,
    is_special_token,
    is_spelling_variation,
)
from romanized_nepali_nlp.sentiment import Prediction


def test_slice_predicates():
    # Spelling variation
    assert is_spelling_variation(DatasetRecord(id="1", text="raamrooo cha"))
    assert not is_spelling_variation(DatasetRecord(id="2", text="ramro cha"))
    assert is_spelling_variation(
        DatasetRecord(id="3", text="ramro cha", metadata={"category": "spelling_variation"})
    )

    # Negation
    assert is_negation(DatasetRecord(id="4", text="yo ramro chhaina"))
    assert is_negation(DatasetRecord(id="5", text="hudaina kaam"))
    assert not is_negation(DatasetRecord(id="6", text="ramro cha"))

    # Emoji
    assert is_emoji(DatasetRecord(id="7", text="dherai ramro :)"))
    assert is_emoji(DatasetRecord(id="8", text="yo kasto ho 🔥"))
    assert not is_emoji(DatasetRecord(id="9", text="sadhaiko jasto"))

    # Code switching
    assert is_code_switching(DatasetRecord(id="10", text="sound quality ekdam worst thiyo"))
    assert not is_code_switching(DatasetRecord(id="11", text="bhat khanu bhayo"))

    # Short text
    assert is_short_text(DatasetRecord(id="12", text="thik cha"))
    assert not is_short_text(
        DatasetRecord(id="13", text="aaja bihana dekhi dherai kaam garera thakai lagyo yar")
    )

    # Special tokens
    assert is_special_token(DatasetRecord(id="14", text="check this out https://nepal.gov.np"))
    assert is_special_token(DatasetRecord(id="15", text="hello @rambahadur"))
    assert not is_special_token(DatasetRecord(id="16", text="namaste sathi"))


def test_evaluate_predictions_metrics_and_slices():
    records = [
        DatasetRecord(id="r1", text="yo geet dherai ramro cha", label="positive"),
        DatasetRecord(id="r2", text="yo movie naramro chaina tara kasto jhyau", label="negative"),
        DatasetRecord(id="r3", text="aaja ma ghar janchhu", label="neutral"),
        DatasetRecord(id="r4", text="ekdam babbal performance 🔥", label="positive"),
        DatasetRecord(id="r5", text="worst service scam company", label="negative"),
        DatasetRecord(id="r6", text="bholi meeting 10 baje", label="neutral"),
        DatasetRecord(id="r7", text="raamrooo lagyo malai", label="positive"),
        DatasetRecord(id="r8", text="kaam hudaina", label="negative"),
        DatasetRecord(id="r9", text="bhat khanu bhayo", label="neutral"),
        DatasetRecord(id="r10", text="dhanyabad", label="positive"),
    ]

    preds = [
        Prediction(
            id="r1",
            gold_label="positive",
            predicted_label="positive",
            scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
        ),
        Prediction(
            id="r2",
            gold_label="negative",
            predicted_label="negative",
            scores={"positive": 0.1, "negative": 0.8, "neutral": 0.1},
        ),
        Prediction(
            id="r3",
            gold_label="neutral",
            predicted_label="neutral",
            scores={"positive": 0.1, "negative": 0.1, "neutral": 0.8},
        ),
        Prediction(
            id="r4",
            gold_label="positive",
            predicted_label="positive",
            scores={"positive": 0.85, "negative": 0.05, "neutral": 0.1},
        ),
        Prediction(
            id="r5",
            gold_label="negative",
            predicted_label="positive",
            scores={"positive": 0.6, "negative": 0.3, "neutral": 0.1},
        ),  # error
        Prediction(
            id="r6",
            gold_label="neutral",
            predicted_label="neutral",
            scores={"positive": 0.05, "negative": 0.05, "neutral": 0.9},
        ),
        Prediction(
            id="r7",
            gold_label="positive",
            predicted_label="positive",
            scores={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
        ),
        Prediction(
            id="r8",
            gold_label="negative",
            predicted_label="negative",
            scores={"positive": 0.1, "negative": 0.7, "neutral": 0.2},
        ),
        Prediction(
            id="r9",
            gold_label="neutral",
            predicted_label="negative",
            scores={"positive": 0.1, "negative": 0.5, "neutral": 0.4},
        ),  # error
        Prediction(
            id="r10",
            gold_label="positive",
            predicted_label="positive",
            scores={"positive": 0.95, "negative": 0.02, "neutral": 0.03},
        ),
    ]

    report = evaluate_predictions(preds, records=records, n_bootstraps=100, min_slice_samples=2)

    assert report.total_records == 10
    assert report.accuracy == pytest.approx(0.80, abs=1e-2)
    assert 0.0 <= report.macro_f1 <= 1.0
    assert 0.0 <= report.weighted_f1 <= 1.0

    # Per-class metrics
    for c in ("positive", "negative", "neutral"):
        assert c in report.per_class
        assert "precision" in report.per_class[c]
        assert "recall" in report.per_class[c]
        assert "f1" in report.per_class[c]
        assert report.per_class[c]["support"] > 0

    # Confusion matrix
    assert report.confusion_matrix["positive"]["positive"] == 4
    assert report.confusion_matrix["negative"]["positive"] == 1

    # Bootstrap CI
    assert "accuracy" in report.bootstrap_ci
    ci_low, ci_high = report.bootstrap_ci["accuracy"]
    assert 0.0 <= ci_low <= ci_high <= 1.0

    # Slices
    slice_names = {s.name for s in report.slices}
    assert "spelling_variation" in slice_names
    assert "negation" in slice_names
    assert "short_text" in slice_names

    # Errors
    assert len(report.errors) == 2
    err_ids = {e.id for e in report.errors}
    assert "r5" in err_ids
    assert "r9" in err_ids


def test_markdown_and_html_generation():
    preds = [
        Prediction(
            id="1",
            gold_label="positive",
            predicted_label="positive",
            scores={"positive": 1.0, "negative": 0.0, "neutral": 0.0},
        ),
        Prediction(
            id="2",
            gold_label="negative",
            predicted_label="negative",
            scores={"positive": 0.0, "negative": 1.0, "neutral": 0.0},
        ),
        Prediction(
            id="3",
            gold_label="neutral",
            predicted_label="neutral",
            scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
        ),
    ]
    report = evaluate_predictions(preds, n_bootstraps=0)

    md = report.to_markdown()
    assert "# Model Evaluation & Challenge Slice Report" in md
    assert "```mermaid" in md
    assert "Confusion Matrix" in md

    html = report.to_html()
    assert "<!DOCTYPE html>" in html
    assert "Model Evaluation & Challenge Slice Report" in html
