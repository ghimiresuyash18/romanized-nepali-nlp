# Data Card

## Scope
Romanized Nepali sentiment toolkit and small reviewed benchmark fixtures.

## Sources
- `tests/fixtures/romanized_examples.jsonl`: synthetic/permitted examples used for tests and baseline demonstrations.
- `data/processed/baseline_phase5/`: frozen benchmark split and reviewed error records.
- `data/processed/spelling_challenge.jsonl`: Task 22 challenge examples; synthetic variations, not raw source text.

## Labels
`positive`, `negative`, `neutral`. Annotation guidance: `docs/annotation-guidelines.md`.

## Privacy and licensing
Do not distribute restricted raw text. Preserve provenance and terms from `docs/source_manifest_template.md`. Public repository fixtures contain no intentionally published phone-number or user-identifier source data. Dataset terms remain separate from MIT code licensing.

## Known bias and limits
Romanized Nepali spelling, dialect, code-switching, sarcasm, emojis, and neutral utterances are unevenly represented. Heuristic phenomenon labels are approximate metadata, not ground truth. Human fluent-speaker review remains required before external release of new challenge examples.

## Intended use
Research, preprocessing inspection, and baseline comparison. Not high-stakes decision making.
