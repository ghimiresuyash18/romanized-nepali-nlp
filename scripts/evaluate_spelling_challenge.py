"""Evaluate all 3 sentiment models on the spelling challenge set.

Outputs:
  reports/spelling_challenge/model_results.json
  Prints a markdown table to stdout.

Usage:
  python scripts/evaluate_spelling_challenge.py
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHALLENGE_PATH = ROOT / "data/processed/spelling_challenge.jsonl"
RUNS_DIR = ROOT / "runs"
RESULTS_PATH = ROOT / "reports/spelling_challenge/model_results.json"

MODEL_NAMES = ["char", "word", "majority"]
CATEGORIES = [
    "vowel_variation",
    "consonant_variation",
    "repeated_chars",
    "omitted_letters",
    "abbreviations",
    "spacing_variation",
]
SENTIMENTS = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model(name: str) -> dict[str, Any]:
    path = RUNS_DIR / name / "model.pkl"
    with path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301


def preprocess(text: str, model_bundle: dict[str, Any]) -> str:
    """Apply pipeline preprocessing when the model was trained with it."""
    cfg = model_bundle.get("config", {})
    if not cfg.get("preprocess", False):
        return text
    try:
        from romanized_nepali_nlp.pipeline import PipelineConfig, PreprocessingPipeline

        pcfg = cfg.get("pipeline_config") or {}
        pipeline = PreprocessingPipeline(PipelineConfig(**pcfg) if pcfg else PipelineConfig())
        result = pipeline.run(text)
        return result.normalized_text
    except Exception:  # noqa: BLE001
        return text


def predict(text: str, bundle: dict[str, Any]) -> str:
    """Return predicted sentiment label for a single text."""
    model_type = bundle["config"]["model_type"]
    if model_type == "majority":
        return str(bundle["majority_label"])
    processed = preprocess(text, bundle)
    vec = bundle["vectorizer"]
    clf = bundle["classifier"]
    x = vec.transform([processed])
    label = clf.predict(x)[0]
    return str(label)


def macro_f1(
    gold: list[str],
    pred: list[str],
    labels: list[str],
) -> float:
    """Compute macro-averaged F1 across given labels."""
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1s.append(2 * precision * recall / denom if denom > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load challenge set
    records = []
    with CHALLENGE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} challenge examples.")

    # Load models
    bundles: dict[str, dict[str, Any]] = {}
    for name in MODEL_NAMES:
        try:
            bundles[name] = load_model(name)
            print(f"Loaded model: {name}")
        except FileNotFoundError:
            print(f"WARNING: model '{name}' not found at {RUNS_DIR / name / 'model.pkl'}")

    if not bundles:
        print("ERROR: no models loaded; aborting.")
        sys.exit(1)

    # Run predictions
    # predictions[model_name] = list of predicted labels (aligned with records)
    predictions: dict[str, list[str]] = {name: [] for name in bundles}
    for rec in records:
        for name, bundle in bundles.items():
            predictions[name].append(predict(rec["text"], bundle))

    gold_labels = [r["sentiment"] for r in records]

    # ── Per-category macro-F1 per model ───────────────────────────────────
    cat_f1: dict[str, dict[str, float]] = defaultdict(dict)
    for name in bundles:
        for cat in CATEGORIES:
            idx = [i for i, r in enumerate(records) if r["variation_category"] == cat]
            if not idx:
                cat_f1[cat][name] = 0.0
                continue
            g = [gold_labels[i] for i in idx]
            p = [predictions[name][i] for i in idx]
            cat_f1[cat][name] = round(macro_f1(g, p, SENTIMENTS), 4)

    # ── Per-sentiment accuracy per model ──────────────────────────────────
    sent_acc: dict[str, dict[str, float]] = defaultdict(dict)
    for name in bundles:
        for sent in SENTIMENTS:
            idx = [i for i, r in enumerate(records) if r["sentiment"] == sent]
            if not idx:
                sent_acc[sent][name] = 0.0
                continue
            g = [gold_labels[i] for i in idx]
            p = [predictions[name][i] for i in idx]
            sent_acc[sent][name] = round(accuracy(g, p), 4)

    # ── Overall macro-F1 ──────────────────────────────────────────────────
    overall_f1: dict[str, float] = {}
    for name in bundles:
        overall_f1[name] = round(macro_f1(gold_labels, predictions[name], SENTIMENTS), 4)

    # ── Build results dict ────────────────────────────────────────────────
    results = {
        "models_evaluated": list(bundles.keys()),
        "total_examples": len(records),
        "overall_macro_f1": overall_f1,
        "macro_f1_per_category": {cat: dict(cat_f1[cat]) for cat in CATEGORIES},
        "accuracy_per_sentiment": {sent: dict(sent_acc[sent]) for sent in SENTIMENTS},
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written → {RESULTS_PATH}\n")

    # ── Print markdown table ──────────────────────────────────────────────
    model_cols = list(bundles.keys())
    col_header = " | ".join(f"{m:>8}" for m in model_cols)
    sep_row = " | ".join("-" * 8 for _ in model_cols)

    print("## Macro-F1 per Variation Category\n")
    print(f"| {'Category':<20} | {col_header} |")
    print(f"| {'-'*20} | {sep_row} |")
    for cat in CATEGORIES:
        vals = " | ".join(f"{cat_f1[cat].get(m, 0.0):8.4f}" for m in model_cols)
        print(f"| {cat:<20} | {vals} |")
    overall_row = " | ".join(f"{overall_f1.get(m, 0.0):8.4f}" for m in model_cols)
    print(f"| {'**Overall**':<20} | {overall_row} |")

    print("\n## Accuracy per Sentiment\n")
    print(f"| {'Sentiment':<10} | {col_header} |")
    print(f"| {'-'*10} | {sep_row} |")
    for sent in SENTIMENTS:
        vals = " | ".join(f"{sent_acc[sent].get(m, 0.0):8.4f}" for m in model_cols)
        print(f"| {sent:<10} | {vals} |")


if __name__ == "__main__":
    main()
