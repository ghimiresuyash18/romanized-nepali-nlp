"""Run fixed preprocessing ablations on the frozen Phase 5 split.

Usage: python scripts/run_ablation.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romanized_nepali_nlp.dataset import DatasetRecord  # noqa: E402
from romanized_nepali_nlp.evaluation import evaluate_predictions  # noqa: E402
from romanized_nepali_nlp.pipeline import PipelineConfig  # noqa: E402
from romanized_nepali_nlp.sentiment import SentimentModel, SentimentModelConfig  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/romanized_examples.jsonl"
MANIFEST = ROOT / "experiments/baseline_phase5/split_manifest.json"
OUT = ROOT / "experiments/ablation_phase6"

# One factor at a time, plus the complete candidate. Validation selects; test runs once.
FACTORS = {
    "raw": dict(preprocess=False),
    "unicode_whitespace": dict(unicode_form="NFC", strip_zero_width=True, collapse_whitespace=True),
    "protected_spans": dict(
        unicode_form="NFC", strip_zero_width=True, collapse_whitespace=True, protected=True
    ),
    "repetition": dict(
        unicode_form="NFC",
        strip_zero_width=True,
        collapse_whitespace=True,
        protected=True,
        reduce_repetitions=True,
    ),
    "variant_dictionary": dict(
        unicode_form="NFC",
        strip_zero_width=True,
        collapse_whitespace=True,
        protected=True,
        normalize_variants=True,
    ),
    "full": dict(
        unicode_form="NFC",
        strip_zero_width=True,
        collapse_whitespace=True,
        protected=True,
        reduce_repetitions=True,
        normalize_variants=True,
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def records_by_split() -> dict[str, list[DatasetRecord]]:
    rows = {r["id"]: r for r in load_jsonl(FIXTURE)}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result: dict[str, list[DatasetRecord]] = {}
    for split, ids in manifest["record_ids"].items():
        result[split] = [
            DatasetRecord(
                id=rid, text=rows[rid]["text"], label=rows[rid]["expected_label"], split=split
            )
            for rid in ids
        ]
    return result


def model_config(factor: dict) -> SentimentModelConfig:
    if not factor.get("preprocess", True):
        return SentimentModelConfig(model_type="char", preprocess=False, seed=42)
    protected = factor.get("protected", False)
    p = PipelineConfig(
        unicode_form=factor.get("unicode_form", ""),
        strip_zero_width=factor.get("strip_zero_width", False),
        collapse_whitespace=factor.get("collapse_whitespace", False),
        reduce_repetitions=factor.get("reduce_repetitions", False),
        normalize_variants=factor.get("normalize_variants", False),
        protected_modes=(
            {
                "URL": "preserve",
                "EMAIL": "preserve",
                "MENTION": "preserve",
                "HASHTAG": "preserve",
                "EMOJI": "preserve",
                "NUMBER": "preserve",
            }
            if protected
            else {}
        ),
        failure_policy="skip_with_warning",
    )
    return SentimentModelConfig(
        model_type="char", preprocess=True, pipeline_config=p.to_dict(), seed=42
    )


def metric(predictions: list, records: list[DatasetRecord]) -> dict:
    return evaluate_predictions(
        predictions, records=records, n_bootstraps=0, min_slice_samples=1
    ).to_dict()


def prediction_hash(predictions: list) -> str:
    payload = "\n".join(json.dumps(p.to_dict(), sort_keys=True) for p in predictions)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_heatmap(results: list[dict]) -> None:
    names = [r["name"] for r in results]
    vals = [r.get("validation", {}).get("macro_f1") for r in results]
    width, height, cell = 720, 100 + 34 * len(names), 34
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font:14px sans-serif}.title{font-weight:bold}</style>",
        '<text x="20" y="24" class="title">Validation macro-F1 by '
        "preprocessing configuration</text>",
    ]
    for i, (name, val) in enumerate(zip(names, vals)):
        y = 45 + i * cell
        shade = "#d9f2d9" if val is not None else "#f5cccc"
        lines += [
            f'<rect x="20" y="{y}" width="680" height="28" fill="{shade}" stroke="#aaa"/>',
            f'<text x="30" y="{y + 19}">{name}</text>',
            f'<text x="550" y="{y + 19}">{"n/a" if val is None else f"{val:.4f}"}</text>',
        ]
    lines.append("</svg>")
    (OUT / "macro_f1_heatmap.svg").write_text("\n".join(lines), encoding="utf-8")


def write_report(results: list[dict], selected: str) -> None:
    lines = [
        "# Normalization Ablation Report",
        "",
        "Frozen split. Character n-gram baseline. Validation selects; test evaluated once.",
        "",
        "| Configuration | Val macro-F1 | Val accuracy | Test macro-F1 | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        v, t = r.get("validation", {}), r.get("test", {})
        lines.append(
            f"| `{r['name']}` | {v.get('macro_f1', 'n/a')} | "
            f"{v.get('accuracy', 'n/a')} | {t.get('macro_f1', 'not selected')} | "
            f"{'selected' if r['name'] == selected else ''}|"
        )
    lines += [
        "",
        f"Selected: `{selected}`.",
        "",
        "## Interpretation",
        "",
        "Preprocessing is a trade-off. Report slice deltas with the overall score; "
        "do not select on test results.",
        "",
    ]
    (OUT / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_slice_deltas(results: list[dict]) -> None:
    baseline = results[0].get("validation", {})
    base = {s["name"]: s.get("macro_f1") for s in baseline.get("slices", [])}
    lines = [
        "# Validation Slice Delta Chart",
        "",
        "Macro-F1 delta versus `raw`; `n/a` means no evaluable slice metric.",
        "",
        "| Slice | " + " | ".join(r["name"] for r in results) + " |",
        "|---|" + "---:|" * len(results),
    ]
    slice_names = sorted(
        {s["name"] for r in results for s in r.get("validation", {}).get("slices", [])}
    )
    for name in slice_names:
        cells = []
        for r in results:
            current = {
                s["name"]: s.get("macro_f1") for s in r.get("validation", {}).get("slices", [])
            }.get(name)
            old = base.get(name)
            cells.append("n/a" if current is None or old is None else f"{current - old:+.4f}")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    (OUT / "slice_deltas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    splits = records_by_split()
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for name, factor in FACTORS.items():
        started = time.perf_counter()
        row = {"name": name, "config": factor, "failure": None}
        try:
            model = SentimentModel(model_config(factor)).fit(splits["train"])
            val_preds = model.predict_records(splits["validation"])
            row["validation"] = metric(val_preds, splits["validation"])
            row["validation_prediction_hash"] = prediction_hash(val_preds)
            row["runtime_seconds"] = round(time.perf_counter() - started, 4)
            results.append(row)
        except Exception as exc:  # preserve failed configurations in the ledger
            row["failure"] = f"{type(exc).__name__}: {exc}"
            row["runtime_seconds"] = round(time.perf_counter() - started, 4)
            results.append(row)
    valid = [r for r in results if r.get("failure") is None]
    if not valid:
        raise RuntimeError("No ablation configuration completed")
    selected = max(valid, key=lambda r: r["validation"]["macro_f1"])["name"]
    for row in results:
        if row["name"] == selected:
            factor = FACTORS[selected]
            model = SentimentModel(model_config(factor)).fit(splits["train"])
            test_preds = model.predict_records(splits["test"])
            row["test"] = metric(test_preds, splits["test"])
            row["test_prediction_hash"] = prediction_hash(test_preds)
            with (OUT / "selected_test_predictions.jsonl").open("w", encoding="utf-8") as fh:
                for p in test_preds:
                    fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    payload = {
        "seed": 42,
        "split_manifest": str(MANIFEST.relative_to(ROOT)),
        "selected_by": "validation_macro_f1",
        "selected": selected,
        "runs": results,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_heatmap(results)
    write_report(results, selected)
    write_slice_deltas(results)
    print(f"Selected {selected}; wrote {OUT}")


if __name__ == "__main__":
    main()
