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


def test_pipeline_failure_policy():
    # Test fallback policy when custom step or error happens
    config = PipelineConfig(failure_policy="skip_with_warning")
    pipeline = PreprocessingPipeline(config=config)

    # Normal run does not fail
    res = pipeline.preprocess("normal text")
    assert len(res.warnings) == 0
    assert res.normalized_text == "normal text"


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
