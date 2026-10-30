"""Evaluation reports, challenge slices, bootstrap confidence intervals, and error analysis.

Task 21: Comprehensive evaluation module for Romanized Nepali sentiment models.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from romanized_nepali_nlp.dataset import ALLOWED_LABELS, DatasetRecord
from romanized_nepali_nlp.sentiment import Prediction

NEGATION_PATTERNS = re.compile(
    r"\b(chaina|chhain|chhaina|haina|hudaina|bhayena|naramro|nadekheko|nakhaye|gardina|bhandina|xaina|xain|sakina|thiyena)\b",
    re.IGNORECASE,
)
EMOJI_PATTERNS = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u27bf]|(:\)|:\(|;-\)|;\)|:D|:P|<3)",
    re.UNICODE,
)
ENGLISH_LOANWORDS = re.compile(
    r"\b(movie|film|song|geet|sound|system|quality|service|staff|food|college|office|meeting|weekend|party|mobile|phone|update|download|exam|library|content|fraud|company|scam|worst|best|super|totally|satisfied|chill)\b",
    re.IGNORECASE,
)
SPECIAL_TOKEN_PATTERNS = re.compile(
    r"(https?://\S+|www\.\S+|@\w+|#\w+|\b\d+\b)",
    re.IGNORECASE,
)
REPETITION_PATTERNS = re.compile(r"([a-zA-Z])\1{2,}")


# ---------------------------------------------------------------------------
# Slice predicates
# ---------------------------------------------------------------------------


def is_spelling_variation(record: DatasetRecord) -> bool:
    """True if text has character repetitions or metadata category indicates spelling variation."""
    cat = (record.metadata.get("category") or "").lower()
    if cat == "spelling_variation":
        return True
    return bool(REPETITION_PATTERNS.search(record.text))


def is_negation(record: DatasetRecord) -> bool:
    """True if text contains negative markers or negation category."""
    cat = (record.metadata.get("category") or "").lower()
    if cat == "negation_intensifier":
        return True
    return bool(NEGATION_PATTERNS.search(record.text))


def is_emoji(record: DatasetRecord) -> bool:
    """True if text contains emojis, emoticons, or punctuation_emoji category."""
    cat = (record.metadata.get("category") or "").lower()
    if cat == "punctuation_emoji":
        return True
    return bool(EMOJI_PATTERNS.search(record.text))


def is_code_switching(record: DatasetRecord) -> bool:
    """True if text has English loan words or code_switching category."""
    cat = (record.metadata.get("category") or "").lower()
    if cat == "code_switching":
        return True
    return bool(ENGLISH_LOANWORDS.search(record.text))


def is_short_text(record: DatasetRecord) -> bool:
    """True if word count <= 3 or char length <= 20."""
    words = record.text.strip().split()
    return len(words) <= 3 or len(record.text.strip()) <= 20


def is_special_token(record: DatasetRecord) -> bool:
    """True if text contains URLs, mentions, hashtags, or numeric tokens."""
    return bool(SPECIAL_TOKEN_PATTERNS.search(record.text))


CHALLENGE_SLICES: Dict[str, Callable[[DatasetRecord], bool]] = {
    "spelling_variation": is_spelling_variation,
    "negation": is_negation,
    "emojis": is_emoji,
    "code_switching": is_code_switching,
    "short_text": is_short_text,
    "special_tokens": is_special_token,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SliceMetric:
    """Evaluation metrics for a single challenge slice."""

    name: str
    count: int
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorRecord:
    """Detailed record for misclassified samples."""

    id: str
    text: str
    gold_label: str
    predicted_label: str
    confidence: float
    scores: Dict[str, float]
    text_length: int
    slice_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Comprehensive sentiment model evaluation report."""

    total_records: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    classes: List[str]
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[str, Dict[str, int]]
    bootstrap_ci: Dict[str, Tuple[float, float]]
    slices: List[SliceMetric]
    errors: List[ErrorRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "classes": self.classes,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "bootstrap_ci": {
                k: [round(v[0], 4), round(v[1], 4)] for k, v in self.bootstrap_ci.items()
            },
            "slices": [s.to_dict() for s in self.slices],
            "error_count": len(self.errors),
            "errors": [e.to_dict() for e in self.errors],
        }

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_markdown(self) -> str:
        """Render markdown report with tables and Mermaid visualizations."""
        return format_markdown_report(self)

    def to_html(self) -> str:
        """Render self-contained HTML report."""
        return format_html_report(self)


# ---------------------------------------------------------------------------
# Evaluator function
# ---------------------------------------------------------------------------


def evaluate_predictions(
    predictions: Sequence[Prediction],
    records: Optional[Sequence[DatasetRecord]] = None,
    n_bootstraps: int = 500,
    min_slice_samples: int = 3,
    seed: int = 42,
) -> EvaluationReport:
    """Evaluate predictions against gold labels with challenge slices and bootstrap CI."""
    labeled_preds = [p for p in predictions if p.gold_label]
    if not labeled_preds:
        raise ValueError("No gold labels present in predictions to evaluate.")

    classes = sorted(list(ALLOWED_LABELS))
    gold_labels = [p.gold_label for p in labeled_preds]
    pred_labels = [p.predicted_label for p in labeled_preds]

    # Core metrics
    acc = float(accuracy_score(gold_labels, pred_labels))
    mac_f1 = float(f1_score(gold_labels, pred_labels, average="macro", zero_division=0))
    wt_f1 = float(f1_score(gold_labels, pred_labels, average="weighted", zero_division=0))

    # Per-class metrics
    p, r, f1, sup = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=classes, zero_division=0
    )
    per_class: Dict[str, Dict[str, float]] = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(gold_labels, pred_labels, labels=classes)
    cm_dict: Dict[str, Dict[str, int]] = {}
    for i, true_cls in enumerate(classes):
        cm_dict[true_cls] = {}
        for j, pred_cls in enumerate(classes):
            cm_dict[true_cls][pred_cls] = int(cm[i, j])

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    n = len(gold_labels)
    boot_acc: List[float] = []
    boot_macro: List[float] = []
    if n >= 10 and n_bootstraps > 0:
        for _ in range(n_bootstraps):
            indices = rng.integers(0, n, size=n)
            sample_gold = [gold_labels[i] for i in indices]
            sample_pred = [pred_labels[i] for i in indices]
            boot_acc.append(float(accuracy_score(sample_gold, sample_pred)))
            boot_macro.append(
                float(f1_score(sample_gold, sample_pred, average="macro", zero_division=0))
            )
        ci_acc = (float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5)))
        ci_macro = (float(np.percentile(boot_macro, 2.5)), float(np.percentile(boot_macro, 97.5)))
    else:
        ci_acc = (acc, acc)
        ci_macro = (mac_f1, mac_f1)

    bootstrap_ci = {"accuracy": ci_acc, "macro_f1": ci_macro}

    # Map records by ID for slice analysis
    rec_by_id = {r.id: r for r in records} if records else {}

    # Challenge slices
    slice_metrics: List[SliceMetric] = []
    for slice_name, slice_fn in CHALLENGE_SLICES.items():
        slice_preds: List[Prediction] = []
        for p in labeled_preds:
            rec = rec_by_id.get(p.id) or DatasetRecord(id=p.id, text=p.raw_text or "")
            if slice_fn(rec):
                slice_preds.append(p)

        cnt = len(slice_preds)
        if cnt < min_slice_samples:
            slice_metrics.append(
                SliceMetric(
                    name=slice_name,
                    count=cnt,
                    note=f"insufficient_samples (n={cnt} < {min_slice_samples})",
                )
            )
        else:
            s_gold = [p.gold_label for p in slice_preds]
            s_pred = [p.predicted_label for p in slice_preds]
            s_acc = float(accuracy_score(s_gold, s_pred))
            s_f1 = float(f1_score(s_gold, s_pred, average="macro", zero_division=0))
            slice_metrics.append(
                SliceMetric(
                    name=slice_name,
                    count=cnt,
                    accuracy=round(s_acc, 4),
                    macro_f1=round(s_f1, 4),
                )
            )

    # Errors list (false positives & false negatives)
    errors: List[ErrorRecord] = []
    for p in labeled_preds:
        if p.gold_label != p.predicted_label:
            rec = rec_by_id.get(p.id) or DatasetRecord(id=p.id, text=p.raw_text or "")
            tags = [s_name for s_name, fn in CHALLENGE_SLICES.items() if fn(rec)]
            confidence = p.scores.get(p.predicted_label, 0.0)
            errors.append(
                ErrorRecord(
                    id=p.id,
                    text=p.raw_text or rec.text,
                    gold_label=p.gold_label,  # type: ignore[arg-type]
                    predicted_label=p.predicted_label,
                    confidence=round(confidence, 4),
                    scores=p.scores,
                    text_length=len(p.raw_text or rec.text),
                    slice_tags=tags,
                )
            )

    return EvaluationReport(
        total_records=n,
        accuracy=round(acc, 4),
        macro_f1=round(mac_f1, 4),
        weighted_f1=round(wt_f1, 4),
        classes=classes,
        per_class=per_class,
        confusion_matrix=cm_dict,
        bootstrap_ci=bootstrap_ci,
        slices=slice_metrics,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_markdown_report(report: EvaluationReport) -> str:
    """Format EvaluationReport into rich GitHub Flavored Markdown."""
    lines = [
        "# Model Evaluation & Challenge Slice Report",
        "",
        "## 1. Summary Performance",
        "",
        f"- **Total Evaluated Records**: {report.total_records}",
        (
            f"- **Accuracy**: **{report.accuracy:.4f}** "
            f"(95% CI: [{report.bootstrap_ci['accuracy'][0]:.4f}, "
            f"{report.bootstrap_ci['accuracy'][1]:.4f}])"
        ),
        (
            f"- **Macro-F1**: **{report.macro_f1:.4f}** "
            f"(95% CI: [{report.bootstrap_ci['macro_f1'][0]:.4f}, "
            f"{report.bootstrap_ci['macro_f1'][1]:.4f}])"
        ),
        f"- **Weighted-F1**: **{report.weighted_f1:.4f}**",
        "",
        "## 2. Per-Class Breakdown",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "|---|---|---|---|---|",
    ]
    for c in report.classes:
        st = report.per_class[c]
        p_val = f"{st['precision']:.4f}"
        r_val = f"{st['recall']:.4f}"
        f1_val = f"{st['f1']:.4f}"
        lines.append(f"| `{c}` | {p_val} | {r_val} | **{f1_val}** | {st['support']} |")

    # Confusion matrix
    lines.extend(
        [
            "",
            "## 3. Confusion Matrix",
            "",
            "| Gold \\ Predicted | " + " | ".join(f"`{c}`" for c in report.classes) + " |",
            "|" + "---|" * (len(report.classes) + 1),
        ]
    )
    for gold_c in report.classes:
        row_str = f"| **`{gold_c}`** | "
        row_str += " | ".join(str(report.confusion_matrix[gold_c][p_c]) for p_c in report.classes)
        row_str += " |"
        lines.append(row_str)

    # Challenge slices
    lines.extend(
        [
            "",
            "## 4. Challenge Slice Robustness",
            "",
            "| Slice | Count | Accuracy | Macro-F1 | Status |",
            "|---|---|---|---|---|",
        ]
    )
    for s in report.slices:
        acc_s = f"{s.accuracy:.4f}" if s.accuracy is not None else "N/A"
        f1_s = f"{s.macro_f1:.4f}" if s.macro_f1 is not None else "N/A"
        status = s.note if s.note else "✅ Evaluated"
        lines.append(f"| **{s.name}** | {s.count} | {acc_s} | **{f1_s}** | {status} |")

    # Mermaid Visualizations
    lines.extend(
        [
            "",
            "## 5. Visualizations",
            "",
            "### Per-Class F1 Performance",
            "```mermaid",
            "xychart-beta",
            '    title "Per-Class F1 Scores"',
            f"    x-axis [{', '.join(f'{c}' for c in report.classes)}]",
            '    y-axis "F1-Score" 0.0 --> 1.0',
            f"    bar [{', '.join(str(report.per_class[c]['f1']) for c in report.classes)}]",
            "```",
            "",
            "### Slice Macro-F1 Comparison",
            "```mermaid",
            "xychart-beta",
            '    title "Challenge Slice Robustness (Macro-F1)"',
            f"    x-axis [{', '.join(s.name for s in report.slices if s.macro_f1 is not None)}]",
            '    y-axis "Macro-F1" 0.0 --> 1.0',
            (
                "    bar ["
                + ", ".join(str(s.macro_f1) for s in report.slices if s.macro_f1 is not None)
                + "]"
            ),
            "```",
            "",
            "## 6. Error Analysis Sample",
            "",
            f"Total misclassified instances: **{len(report.errors)}**",
            "",
            "| ID | Gold | Predicted | Conf | Slices | Text |",
            "|---|---|---|---|---|---|",
        ]
    )
    for err in report.errors[:20]:
        clean_text = err.text.replace("\n", " ").replace("|", "\\|")[:45]
        slice_tags = ", ".join(err.slice_tags) if err.slice_tags else "none"
        lines.append(
            f"| `{err.id}` | `{err.gold_label}` | `{err.predicted_label}` | "
            f"{err.confidence:.2f} | {slice_tags} | {clean_text} |"
        )

    return "\n".join(lines)


def format_html_report(report: EvaluationReport) -> str:
    """Format EvaluationReport into self-contained HTML."""
    rows_class = "".join(
        f"<tr><td><code>{c}</code></td><td>{report.per_class[c]['precision']:.4f}</td>"
        f"<td>{report.per_class[c]['recall']:.4f}</td>"
        f"<td><b>{report.per_class[c]['f1']:.4f}</b></td>"
        f"<td>{report.per_class[c]['support']}</td></tr>"
        for c in report.classes
    )
    rows_slice = "".join(
        f"<tr><td><b>{s.name}</b></td><td>{s.count}</td>"
        f"<td>{s.accuracy if s.accuracy is not None else 'N/A'}</td>"
        f"<td><b>{s.macro_f1 if s.macro_f1 is not None else 'N/A'}</b></td>"
        f"<td>{s.note or 'OK'}</td></tr>"
        for s in report.slices
    )
    style_block = (
        "body { font-family: sans-serif; margin: 2rem; background: #fafafa; }\n"
        "table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }\n"
        "th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }\n"
        "th { background-color: #f2f2f2; }\n"
    )
    return (
        '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        f"<title>Evaluation Report</title>\n<style>{style_block}</style>\n</head>\n<body>\n"
        "<h1>Model Evaluation & Challenge Slice Report</h1>\n"
        f"<p>Total Records: <b>{report.total_records}</b> | "
        f"Accuracy: <b>{report.accuracy:.4f}</b> | "
        f"Macro-F1: <b>{report.macro_f1:.4f}</b></p>\n"
        "<h2>Per-Class Metrics</h2>\n"
        "<table>\n<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>\n"
        f"{rows_class}\n</table>\n"
        "<h2>Challenge Slices</h2>\n"
        "<table>\n<tr><th>Slice</th><th>Count</th><th>Accuracy</th><th>Macro-F1</th><th>Status</th></tr>\n"
        f"{rows_slice}\n</table>\n"
        "</body>\n</html>\n"
    )
