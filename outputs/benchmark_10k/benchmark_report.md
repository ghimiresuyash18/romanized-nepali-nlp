# 10,000-Row Romanized Nepali Sentiment Benchmark

## Executive result

This benchmark evaluates three sentiment baselines on a new 10,000-record sample drawn from the locally published `Boredoom17/Nepali-Flow-Roman` corpus. The source corpus contains 307,999 web-collected YouTube comments and is documented as CC BY 4.0 in `data/processed/nepali_flow_roman_v2/manifest.json`.

The source records were unlabeled. Therefore, this is a **silver-label benchmark**, not a human-gold benchmark. Labels were generated independently by `gpt-6-luna` using strict structured output, with one label per record. The label source is recorded in every labeled record and in the summary JSON.

## Dataset construction

- Source corpus: 307,999 records.
- Sample size: 10,000 records.
- Sampling: deterministic, seed 42, proportional across the three usable corpus partitions.
- `roman_clean`: 7,359 records.
- `roman_mixed`: 2,382 records.
- `roman_noisy`: 259 records.
- `reject_or_review`: excluded from this benchmark.
- Labels: `positive`, `negative`, `neutral`.
- Silver-label distribution:
  - positive: 5,224
  - negative: 2,078
  - neutral: 2,698
- Mean labeler confidence: 0.8888.
- Minimum reported labeler confidence: 0.25.

The label instructions required `neutral` for factual, unclear, non-sentimental, or insufficiently understandable text. The 50-record pilot produced valid structured output with no errors. The full run produced 500 batches and 10,000 labels; two ID mismatches were retried and corrected before evaluation.

## Evaluation protocol

The labeled sample was split deterministically with seed 42:

| Split | Records |
|---|---:|
| Train | 7,003 |
| Validation | 1,502 |
| Test | 1,495 |

The models were trained only on the training split. The reported metrics come from the held-out test split. Bootstrap confidence intervals use 500 resamples.

## Results

| Model | Accuracy | Macro-F1 | Weighted-F1 | 95% accuracy CI | 95% macro-F1 CI |
|---|---:|---:|---:|---:|---:|
| Majority | 0.5224 | 0.2288 | 0.3585 | 0.4980–0.5445 | 0.2216–0.2350 |
| Word TF-IDF + logistic regression | 0.7204 | 0.6831 | 0.7232 | 0.6990–0.7418 | 0.6596–0.7092 |
| Character TF-IDF + logistic regression | **0.7505** | **0.7191** | **0.7546** | 0.7274–0.7732 | 0.6948–0.7430 |

Character TF-IDF is the strongest of the three baselines on this sample. Compared with Word TF-IDF, it improves:

- Accuracy by 3.01 percentage points.
- Macro-F1 by 3.60 percentage points.
- Weighted-F1 by 3.14 percentage points.

This supports the project’s hypothesis that character features are useful for noisy Romanized Nepali spelling variation. It does not prove that character TF-IDF is universally superior, because the labels are silver labels and the sample comes from one source corpus.

## Frozen-model transfer check

For reference, the previously bundled models trained on the original 106-record fixture were also evaluated on the same new test split without retraining:

| Existing model | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen word model | 0.4642 | 0.3958 |
| Frozen character model | 0.4094 | 0.3959 |

The large gap between frozen models and models trained on 7,003 sample records shows that the original 106-record fixture is useful as a regression test but is not sufficient for broad-domain generalization.

## Interpretation and limitations

This is a substantial engineering benchmark, but it is not equivalent to manually annotated research data. The same LLM produced the reference labels for all three models, so the scores measure agreement with that silver-labeling policy. They do not establish agreement with human annotators.

Potential label risks include:

- Romanized Nepali and code-switched text can be ambiguous.
- Some source comments are not strongly sentiment-bearing.
- The independent labeler can misunderstand spelling variation or cultural context.
- Model training and evaluation labels share the same annotation policy.
- The source corpus contains mixed and noisy material, not only clean Romanized Nepali.
- The emoji slice is very small in this sample and should not support a strong conclusion.

For a publication-quality result, manually annotate and adjudicate a held-out gold test set, ideally with at least two human annotators and agreement statistics. The current benchmark should be described as a **10k silver-label baseline study**, not as a human-ground-truth benchmark.

## Reproduction

The reusable runner is `benchmark_10k.py` in this output directory. The generated labeled records and per-model reports are also included here.

The high-level commands used were:

```bash
python3 scripts/benchmark_10k.py \
  --corpus data/processed/nepali_flow_roman_v2 \
  --output-dir .work/benchmark_10k \
  --size 10000 \
  --seed 42 \
  prepare

python3 scripts/benchmark_10k.py \
  --corpus data/processed/nepali_flow_roman_v2 \
  --output-dir .work/benchmark_10k \
  parse

python3 scripts/benchmark_10k.py \
  --corpus data/processed/nepali_flow_roman_v2 \
  --output-dir .work/benchmark_10k \
  evaluate
```

The labeling step used the independent `gpt-6-luna` annotator with strict JSON output. The raw labeling batches are retained under `.work/benchmark_10k/` for auditability.
