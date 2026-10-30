# Data

This repository includes small, public-safe datasets for running the examples. It does **not** include the full scraped corpus.

## Included

- `processed/spelling_challenge.jsonl`: 150 spelling-variation challenge examples.
- `processed/spelling_challenge_manifest.json`: challenge-set metadata and SHA-256 hash.
- `processed/baseline_phase5/benchmark.jsonl`: 15-record public benchmark.
- `processed/error_analysis.jsonl`: reviewed error metadata.
- `../tests/fixtures/romanized_examples.jsonl`: 106 labeled examples used by tests and the baseline demo.

Run the included workflows from the repository root:

```bash
python3 scripts/evaluate_spelling_challenge.py
python3 scripts/phenomena_analysis.py
python3 scripts/run_ablation.py
python3 scripts/demo.py
```

## Full corpus

`data/raw/` and most of `data/interim/` are intentionally not published. They contain large or intermediate corpus material whose source terms, privacy status, and redistribution rights must be checked separately. Place an authorized local JSONL corpus under `data/raw/`, then run:

```bash
romanized-nepali corpus data/raw/your_corpus.jsonl \
  --benchmark data/processed/baseline_phase5/benchmark.jsonl \
  --output-dir data/processed/your_corpus
```

Do not commit restricted raw text, personal identifiers, or generated intermediate files. See `docs/data-card.md` and `docs/source_manifest_template.md` before adding data.
