"""Phenomena analysis script.

Loads the benchmark set, runs detect_phenomena on every record, prints slice
membership counts and class balance, compares with error_analysis.jsonl ground
truth where available, and saves a summary to reports/phenomena/slice_counts.md.

Usage
-----
    python scripts/phenomena_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Allow running directly without installing the package
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from romanized_nepali_nlp.phenomena import detect_phenomena  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
BENCHMARK_PATH = ROOT / "data" / "processed" / "baseline_phase5" / "benchmark.jsonl"
ERROR_PATH = ROOT / "data" / "processed" / "error_analysis.jsonl"
OUT_PATH = ROOT / "reports" / "phenomena" / "slice_counts.md"

# The benchmark file contains only text + id (no gold labels in this split).
# Gold labels come from the error_analysis predictions where available, or we
# mark them as "unknown".
GOLD_FROM_ERRORS: dict[str, str] = {}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:  # noqa: C901 – complexity acceptable for a one-off script
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    benchmark = load_jsonl(BENCHMARK_PATH)
    errors = load_jsonl(ERROR_PATH)

    # Build gold-label lookup from error records (record_id → gold_label)
    for e in errors:
        rid = e.get("record_id") or e.get("id", "")
        gold = e.get("gold_label", "")
        if rid and gold:
            GOLD_FROM_ERRORS[rid] = gold

    # ------------------------------------------------------------------
    # 2. Run detect_phenomena on every benchmark record
    # ------------------------------------------------------------------
    annotated: list[dict] = []
    for rec in benchmark:
        rid = rec.get("id", "")
        text = rec.get("text", "")
        phenomena = detect_phenomena(text)
        gold = GOLD_FROM_ERRORS.get(rid, "unknown")
        annotated.append(
            {
                "id": rid,
                "text": text,
                "gold_label": gold,
                "phenomena": phenomena,
            }
        )

    # ------------------------------------------------------------------
    # 3. Build slice membership counts
    # ------------------------------------------------------------------
    slices = {
        "negation": [a for a in annotated if a["phenomena"]["has_negation"]],
        "intensifier": [a for a in annotated if a["phenomena"]["has_intensifier"]],
        "emoji": [a for a in annotated if a["phenomena"]["emoji_list"]],
        "code_switch": [a for a in annotated if a["phenomena"]["has_code_switch"]],
    }

    # ------------------------------------------------------------------
    # 4. Print results
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phenomena Analysis — Benchmark Set")
    print(f"Total records: {len(annotated)}")
    print("=" * 60)

    slice_summaries: list[dict] = []
    for slice_name, members in slices.items():
        label_counts: Counter = Counter(m["gold_label"] for m in members)
        total = len(members)
        print(f"\n[{slice_name}]  n={total}")
        for label, cnt in sorted(label_counts.items()):
            print(f"  {label}: {cnt}")
        slice_summaries.append(
            {
                "name": slice_name,
                "count": total,
                "label_counts": dict(sorted(label_counts.items())),
            }
        )

    # ------------------------------------------------------------------
    # 5. Compare heuristic vs error_analysis slice_tags (precision proxy)
    # ------------------------------------------------------------------
    # Build a set of ids tagged per slice in error_analysis
    gt_negation_ids: set[str] = set()
    gt_code_switch_ids: set[str] = set()
    for e in errors:
        rid = e.get("record_id") or e.get("id", "")
        tags = e.get("slice_tags", [])
        if "negation" in tags:
            gt_negation_ids.add(rid)
        if "code_switching" in tags:
            gt_code_switch_ids.add(rid)

    # Heuristic positive sets (ids only, restricted to records in error_analysis)
    error_ids_in_benchmark = {e.get("record_id") or e.get("id", "") for e in errors}

    def _precision(heuristic_slice: list[dict], gt_ids: set[str]) -> tuple[int, int, float | None]:
        """Return (tp, fp, precision) for records that appear in both sets."""
        relevant = [a for a in heuristic_slice if a["id"] in error_ids_in_benchmark]
        if not relevant:
            return 0, 0, None
        tp = sum(1 for a in relevant if a["id"] in gt_ids)
        fp = len(relevant) - tp
        prec = tp / len(relevant) if relevant else None
        return tp, fp, prec

    print("\n" + "=" * 60)
    print("Precision vs error_analysis.jsonl ground truth")
    print("(Only records present in error_analysis are evaluated)")
    print("=" * 60)

    neg_tp, neg_fp, neg_prec = _precision(slices["negation"], gt_negation_ids)
    cs_tp, cs_fp, cs_prec = _precision(slices["code_switch"], gt_code_switch_ids)

    if neg_prec is not None:
        print(f"\nnegation    — TP={neg_tp}  FP={neg_fp}  precision={neg_prec:.2f}")
    else:
        print("\nnegation    — no overlap with error_analysis")
    if cs_prec is not None:
        print(f"code_switch — TP={cs_tp}  FP={cs_fp}  precision={cs_prec:.2f}")
    else:
        print("code_switch — no overlap with error_analysis")

    # ------------------------------------------------------------------
    # 6. Save markdown report
    # ------------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(annotated, slice_summaries, neg_tp, neg_fp, neg_prec, cs_tp, cs_fp, cs_prec)
    print(f"\nReport saved → {OUT_PATH}")


def _write_markdown(
    annotated: list[dict],
    slice_summaries: list[dict],
    neg_tp: int,
    neg_fp: int,
    neg_prec: float | None,
    cs_tp: int,
    cs_fp: int,
    cs_prec: float | None,
) -> None:
    lines: list[str] = []
    lines += [
        "# Phenomena Slice Counts — Romanized Nepali Benchmark",
        "",
        "> **Note:** All slice labels are produced by heuristic pattern matching,",
        "> not ground truth annotation. Treat counts as approximate signals.",
        "",
        f"**Total benchmark records:** {len(annotated)}",
        "",
        "---",
        "",
        "## Slice Membership Counts",
        "",
        "| Slice | Count | negative | neutral | positive | unknown |",
        "|-------|------:|---------:|--------:|---------:|--------:|",
    ]

    for s in slice_summaries:
        lc = s["label_counts"]
        lines.append(
            f"| {s['name']} | {s['count']} "
            f"| {lc.get('negative', 0)} "
            f"| {lc.get('neutral', 0)} "
            f"| {lc.get('positive', 0)} "
            f"| {lc.get('unknown', 0)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Record Phenomena",
        "",
        "| id | text (truncated) | negation | intensifier | emoji_cats | code_switch |",
        "|----|-----------------|:--------:|:-----------:|:----------:|:-----------:|",
    ]
    for a in annotated:
        ph = a["phenomena"]
        text_short = a["text"][:40].replace("|", "\\|") + ("…" if len(a["text"]) > 40 else "")
        neg = "✓" if ph["has_negation"] else ""
        intensifier = "✓" if ph["has_intensifier"] else ""
        emojis = ", ".join(ph["emoji_categories"]) if ph["emoji_categories"] else ""
        cs = "✓" if ph["has_code_switch"] else ""
        lines.append(f"| {a['id']} | {text_short} | {neg} | {intensifier} | {emojis} | {cs} |")

    lines += [
        "",
        "---",
        "",
        "## Precision vs `error_analysis.jsonl` Ground Truth",
        "",
        "> Precision is computed only for records present in `error_analysis.jsonl`.",
        "> It measures what fraction of heuristic positives match the reviewed slice tags.",
        "",
        "| Slice | TP | FP | Precision |",
        "|-------|---:|---:|----------:|",
    ]

    neg_prec_str = f"{neg_prec:.2f}" if neg_prec is not None else "N/A"
    cs_prec_str = f"{cs_prec:.2f}" if cs_prec is not None else "N/A"
    lines.append(f"| negation | {neg_tp} | {neg_fp} | {neg_prec_str} |")
    lines.append(f"| code_switch | {cs_tp} | {cs_fp} | {cs_prec_str} |")

    lines += [
        "",
        "---",
        "",
        "## Heuristic Rules Summary",
        "",
        "| Phenomenon | Method |",
        "|-----------|--------|",
        (
            "| negation | Word-boundary regex on token list:"
            " `na, nai, chhaina, hoina, hudaina, bhayena,"
            " nabhayeko, naramro, nabujhne, nagarnu, nalagne` |"
        ),
        (
            "| intensifier | Word-boundary regex on token list:"
            " `dherai, ekdam, ati, sab_bhandaa, alikati, thulo, sano, bahut, dherei` |"
        ),
        (
            "| emoji | Character lookup against `emoji_categories.json`;"
            " multi-codepoint sequences matched in order |"
        ),
        (
            "| code_switch | Word-boundary regex on English function/content word list;"
            " see `phenomena.py` `CODE_SWITCH_FUNCTION_WORDS` |"
        ),
        "",
        "*Generated by `scripts/phenomena_analysis.py`.*",
    ]

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
