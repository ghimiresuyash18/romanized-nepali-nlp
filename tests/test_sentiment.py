"""Unit and integration tests for sentiment baselines (Tasks 18 and 19)."""

from __future__ import annotations

from pathlib import Path

import pytest

from romanized_nepali_nlp.dataset import ColumnMap, DatasetRecord, load_jsonl
from romanized_nepali_nlp.sentiment import (
    SentimentModel,
    SentimentModelConfig,
    compute_learning_curve,
)
from romanized_nepali_nlp.splitting import SplitConfig, split

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ROMANIZED_EXAMPLES = FIXTURES_DIR / "romanized_examples.jsonl"


@pytest.fixture
def sample_dataset():
    res = load_jsonl(ROMANIZED_EXAMPLES, col=ColumnMap(label="expected_label"))
    return res.records


@pytest.fixture
def train_val_splits(sample_dataset):
    split_res = split(sample_dataset, SplitConfig(seed=42))
    return split_res.train, split_res.validation, split_res.test


# ---------------------------------------------------------------------------
# Task 18: Majority and Word Baselines
# ---------------------------------------------------------------------------


def test_majority_baseline(train_val_splits):
    train_recs, val_recs, _ = train_val_splits
    cfg = SentimentModelConfig(model_type="majority")
    model = SentimentModel(cfg).fit(train_recs)

    assert model.majority_label_ is not None
    assert sum(model.class_priors_.values()) == pytest.approx(1.0)

    preds = model.predict_records(val_recs)
    assert len(preds) == len(val_recs)
    for p in preds:
        assert p.predicted_label == model.majority_label_
        assert p.scores[model.majority_label_] == pytest.approx(
            model.class_priors_[model.majority_label_]
        )


def test_word_baseline_training_and_predictions(train_val_splits):
    train_recs, val_recs, _ = train_val_splits
    cfg = SentimentModelConfig(model_type="word", ngram_range=(1, 2))
    model = SentimentModel(cfg).fit(train_recs)

    assert model.vectorizer_ is not None
    assert model.classifier_ is not None
    assert model.vectorizer_.analyzer == "word"

    preds = model.predict_records(val_recs)
    assert len(preds) == len(val_recs)
    for p in preds:
        assert p.predicted_label in {"positive", "negative", "neutral"}
        assert sum(p.scores.values()) == pytest.approx(1.0, abs=1e-4)


def test_learning_curve(train_val_splits):
    train_recs, val_recs, _ = train_val_splits
    curve = compute_learning_curve(
        train_recs,
        val_recs,
        config=SentimentModelConfig(model_type="char"),
        train_fractions=[0.3, 0.6, 1.0],
    )
    assert len(curve) == 3
    assert curve[0]["train_samples"] <= curve[1]["train_samples"] <= curve[2]["train_samples"]
    for pt in curve:
        assert 0.0 <= pt["train_macro_f1"] <= 1.0
        assert 0.0 <= pt["val_macro_f1"] <= 1.0


# ---------------------------------------------------------------------------
# Task 19: Character TF-IDF Baseline & Robustness
# ---------------------------------------------------------------------------


def test_char_baseline_analyzer_options(train_val_splits):
    train_recs, val_recs, _ = train_val_splits

    # char_wb analyzer
    cfg_wb = SentimentModelConfig(model_type="char", analyzer="char_wb", ngram_range=(2, 4))
    model_wb = SentimentModel(cfg_wb).fit(train_recs)
    preds_wb = model_wb.predict_records(val_recs)
    assert len(preds_wb) == len(val_recs)

    # char analyzer
    cfg_char = SentimentModelConfig(model_type="char", analyzer="char", ngram_range=(2, 4))
    model_char = SentimentModel(cfg_char).fit(train_recs)
    preds_char = model_char.predict_records(val_recs)
    assert len(preds_char) == len(val_recs)


def test_char_baseline_min_df_and_feature_cap(train_val_splits):
    train_recs, _, _ = train_val_splits
    cfg = SentimentModelConfig(
        model_type="char",
        analyzer="char_wb",
        ngram_range=(2, 3),
        min_df=2,
        max_features=50,
    )
    model = SentimentModel(cfg).fit(train_recs)
    assert model.vectorizer_ is not None
    feature_count = len(model.vectorizer_.get_feature_names_out())
    assert feature_count <= 50


def test_char_baseline_save_and_load_identical_predictions(train_val_splits, tmp_path):
    train_recs, val_recs, _ = train_val_splits
    cfg = SentimentModelConfig(model_type="char", analyzer="char_wb", ngram_range=(2, 4))
    model = SentimentModel(cfg).fit(train_recs)

    preds_before = model.predict_records(val_recs)

    save_path = tmp_path / "char_model.pkl"
    model.save(save_path)
    assert save_path.exists()

    loaded_model = SentimentModel.load(save_path)
    preds_after = loaded_model.predict_records(val_recs)

    assert len(preds_before) == len(preds_after)
    for b, a in zip(preds_before, preds_after):
        assert b.id == a.id
        assert b.predicted_label == a.predicted_label
        for cls in b.scores:
            assert b.scores[cls] == pytest.approx(a.scores[cls], abs=1e-6)


def test_top_features_inspection(train_val_splits):
    train_recs, _, _ = train_val_splits
    cfg = SentimentModelConfig(model_type="char", analyzer="char_wb", ngram_range=(2, 4))
    model = SentimentModel(cfg).fit(train_recs)

    top_feats = model.get_top_features(n=5)
    for cls in ("positive", "negative", "neutral"):
        assert cls in top_feats
        assert len(top_feats[cls]) <= 5
        for feat, score in top_feats[cls]:
            assert isinstance(feat, str)
            assert isinstance(score, float)
            assert not feat.startswith("http")
            assert not feat.startswith("@")


def test_spelling_variation_predictions(train_val_splits):
    """Inspect spelling-variant predictions on pairs (Task 19, subtask 8)."""
    train_recs, _, _ = train_val_splits
    cfg = SentimentModelConfig(model_type="char", analyzer="char_wb", ngram_range=(2, 4))
    model = SentimentModel(cfg).fit(train_recs)

    variant_pairs = [
        ("yo ta ramro cha", "yo ta raamro cha"),
        ("yo ta ramro cha", "yo ta ramroo cha"),
        ("yo ta ramro cha", "yo ta ramrooo cha"),
        ("sarai naramro film", "sarai naaramro film"),
    ]
    for orig, var in variant_pairs:
        preds = model.predict_texts([orig, var], ids=["orig", "var"])
        assert preds[0].predicted_label == preds[1].predicted_label


def test_binary_top_features_inspection():
    binary_records = [
        DatasetRecord(id="b1", text="ramro ramro cha", label="positive"),
        DatasetRecord(id="b2", text="dherai ramro film", label="positive"),
        DatasetRecord(id="b3", text="naramro kura", label="negative"),
        DatasetRecord(id="b4", text="ekdam kharab cha", label="negative"),
    ]
    cfg = SentimentModelConfig(model_type="char", analyzer="char_wb", ngram_range=(2, 3), min_df=1)
    model = SentimentModel(cfg).fit(binary_records)
    top_feats = model.get_top_features(n=3)
    assert set(top_feats.keys()) == {"negative", "positive"}
    assert len(top_feats["positive"]) > 0
    assert len(top_feats["negative"]) > 0
