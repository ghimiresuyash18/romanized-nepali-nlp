"""Unit and integration tests for PreprocessingPipeline and PipelineConfig."""

import json
from pathlib import Path

from romanized_nepali_nlp.pipeline import PipelineConfig, PreprocessingPipeline

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_pipeline_preserves_raw_text():
    config = PipelineConfig(lowercase=True, reduce_repetitions=True)
    pipeline = PreprocessingPipeline(config=config)

    raw = "  Yo RAAMROOO cha!! Visit https://hamro.com  "
    result = pipeline.preprocess(raw)

    # Raw text must never be mutated
    assert result.raw_text == raw
    assert result.normalized_text != raw
    assert "https://hamro.com" in result.normalized_text
    assert len(result.steps) > 0
    assert result.metadata["pipeline_fingerprint"] == config.fingerprint()

    # Protected span offsets must accurately slice normalized text
    assert len(result.protected_spans) == 1
    span = result.protected_spans[0]
    assert result.normalized_text[span.start : span.end] == "https://hamro.com"
    assert result.raw_text[span.raw_start : span.raw_end] == "https://hamro.com"


def test_pipeline_invalid_failure_policy_raises():
    import pytest

    with pytest.raises(ValueError, match="Invalid failure_policy"):
        PipelineConfig(failure_policy="fail-fast")


def test_pipeline_idempotence():
    # Applying the pipeline a second time to its own output should not alter normalized text
    pipeline = PreprocessingPipeline()
    raw = "yo kasto raamro chha sathi!  Visit https://hamro.com"

    first_run = pipeline.preprocess(raw)
    second_run = pipeline.preprocess(first_run.normalized_text)

    assert second_run.normalized_text == first_run.normalized_text


def test_pipeline_config_serialization_and_fingerprint(tmp_path):
    config = PipelineConfig(
        lowercase=True,
        variant_policy="aggressive",
        reduce_repetitions=True,
    )
    fp1 = config.fingerprint()

    # Save to JSON
    cfg_file = tmp_path / "pipeline_config.json"
    config.save(cfg_file)
    assert cfg_file.exists()

    # Reload
    reloaded = PipelineConfig.load(cfg_file)
    assert reloaded == config
    assert reloaded.fingerprint() == fp1


def test_pipeline_batch_processing():
    pipeline = PreprocessingPipeline()
    texts = [
        "namaste sathi",
        "yo ta raamro chha",
        "dheraiii maza aayo",
    ]
    results = pipeline.preprocess_batch(texts)
    assert len(results) == 3
    assert results[1].normalized_text == "yo ta ramro cha"
    assert results[2].normalized_text == "dherai maza aayo"


def test_pipeline_failure_policy(monkeypatch):
    from unittest.mock import MagicMock

    import pytest

    # 1. fail_fast raises exception
    p_fail = PreprocessingPipeline(PipelineConfig(failure_policy="fail_fast"))
    monkeypatch.setattr(
        p_fail,
        "tokenizer",
        MagicMock(tokenize=MagicMock(side_effect=RuntimeError("tokenizer crash"))),
    )
    with pytest.raises(RuntimeError, match="tokenizer crash"):
        p_fail.preprocess("test text")

    # 2. skip_with_warning falls back and logs warning
    p_skip = PreprocessingPipeline(PipelineConfig(failure_policy="skip_with_warning"))
    monkeypatch.setattr(
        p_skip,
        "tokenizer",
        MagicMock(tokenize=MagicMock(side_effect=RuntimeError("tokenizer crash"))),
    )
    res_skip = p_skip.preprocess("test text")
    assert res_skip.normalized_text == "test text"
    assert any("tokenizer crash" in w for w in res_skip.warnings)
    assert res_skip.steps == ["fallback_keep_original"]

    # 3. keep_original falls back with raw error in warnings
    p_orig = PreprocessingPipeline(PipelineConfig(failure_policy="keep_original"))
    monkeypatch.setattr(
        p_orig,
        "tokenizer",
        MagicMock(tokenize=MagicMock(side_effect=RuntimeError("tokenizer crash"))),
    )
    res_orig = p_orig.preprocess("test text")
    assert res_orig.normalized_text == "test text"
    assert res_orig.warnings == ["tokenizer crash"]
    assert res_orig.steps == ["fallback_keep_original"]


def test_pipeline_over_fixture_corpus():
    examples_file = FIXTURES_DIR / "romanized_examples.jsonl"
    pipeline = PreprocessingPipeline()

    records = []
    with open(examples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) >= 100

    # Batch process all 106 fixture texts
    raw_texts = [r["text"] for r in records]
    results = pipeline.preprocess_batch(raw_texts)

    assert len(results) == len(raw_texts)
    for r, res in zip(records, results):
        assert res.raw_text == r["text"]
        assert isinstance(res.normalized_text, str)


def test_pipeline_repetition_reduction_with_protected_tokens():
    cfg = PipelineConfig(reduce_repetitions=True)
    pipe = PreprocessingPipeline(cfg)
    for _ in range(50):
        res = pipe.preprocess("ramrooo dheraiii https://example.com/test @user 🔥")
        assert "<__PROTECTED_" not in res.normalized_text
        assert "https://example.com/test" in res.normalized_text
        assert "@user" in res.normalized_text
        assert "🔥" in res.normalized_text
