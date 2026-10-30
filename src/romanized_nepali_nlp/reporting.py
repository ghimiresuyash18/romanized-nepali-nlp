"""Reporting, chart generation, and visualization for tokenizer and preprocessing validation."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib

    # Set non-interactive Agg backend before importing pyplot
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

PHONE_REGEX = re.compile(r"\b(?:\+?977[- ]?)?(?:98\d{8}|97\d{8}|\d{7,10})\b")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
HANDLE_REGEX = re.compile(r"@[a-zA-Z0-9_]+")


def mask_pii(text: str) -> str:
    """Mask phone numbers, emails, and user handles to protect privacy."""
    text = EMAIL_REGEX.sub("[EMAIL_MASKED]", text)
    text = HANDLE_REGEX.sub("@user_masked", text)
    text = PHONE_REGEX.sub("[PHONE_MASKED]", text)
    return text


def _row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def generate_v1_coverage_report(summary: Dict[str, Any], output_path: Path) -> str:
    """Generate comprehensive v1 coverage markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = summary.get("partitions", {})
    c = parts.get("roman_clean", {})
    m = parts.get("roman_mixed", {})
    n = parts.get("roman_noisy", {})

    tot_rows = summary.get("total_rows", 0)
    tot_toks = summary.get("total_tokens", 0)
    avg_toks = summary.get("avg_tokens_per_row", 0.0)
    errs = summary.get("error_count", 0)
    unk_spans = summary.get("unknown_spans", 0)
    unk_rate = summary.get("unknown_span_rate", 0.0) * 100
    unc_chars = summary.get("uncovered_chars", 0)
    unc_rate = summary.get("uncovered_char_rate", 0.0) * 100

    cats = summary.get("categories", {})
    c_cats = c.get("categories", {})
    m_cats = m.get("categories", {})
    n_cats = n.get("categories", {})

    c_b = c.get("token_length_buckets", {})
    m_b = m.get("token_length_buckets", {})
    n_b = n.get("token_length_buckets", {})

    lines = [
        "# Tokenizer Coverage & Profiling Report (v1)",
        "",
        "## Executive Summary",
        "",
        f"- **Total Rows Profiled:** {tot_rows:,}",
        f"- **Total Tokens:** {tot_toks:,}",
        f"- **Average Tokens / Row:** {avg_toks:.2f}",
        f"- **Tokenizer Crashes / Errors:** {errs}",
        f"- **Unknown Spans Total:** {unk_spans:,} ({unk_rate:.4f}% of tokens)",
        f"- **Uncovered Non-Whitespace Characters:** {unc_chars} ({unc_rate:.4f}%)",
        "",
        "---",
        "",
        "## Partition-Level Breakdown",
        "",
        _row("Metric", "Overall", "Clean", "Mixed", "Noisy"),
        _row(":---", ":---", ":---", ":---", ":---"),
        _row("**Rows**", f"{tot_rows:,}", f"{c.get('total_rows', 0):,}",
             f"{m.get('total_rows', 0):,}", f"{n.get('total_rows', 0):,}"),
        _row("**Tokens**", f"{tot_toks:,}", f"{c.get('total_tokens', 0):,}",
             f"{m.get('total_tokens', 0):,}", f"{n.get('total_tokens', 0):,}"),
        _row("**Avg Tokens/Row**", f"{avg_toks:.2f}",
             f"{c.get('avg_tokens_per_row', 0.0):.2f}",
             f"{m.get('avg_tokens_per_row', 0.0):.2f}",
             f"{n.get('avg_tokens_per_row', 0.0):.2f}"),
        _row("**Unknown Spans**", f"{unk_spans:,}", f"{c.get('unknown_spans', 0):,}",
             f"{m.get('unknown_spans', 0):,}", f"{n.get('unknown_spans', 0):,}"),
        _row("**Unknown Span Rate**", f"{unk_rate:.4f}%",
             f"{c.get('unknown_span_rate', 0.0)*100:.4f}%",
             f"{m.get('unknown_span_rate', 0.0)*100:.4f}%",
             f"{n.get('unknown_span_rate', 0.0)*100:.4f}%"),
        _row("**Rows w/ Unknowns**",
             f"{sum(p.get('rows_with_unknown_spans', 0) for p in parts.values()):,}",
             f"{c.get('rows_with_unknown_spans', 0):,}",
             f"{m.get('rows_with_unknown_spans', 0):,}",
             f"{n.get('rows_with_unknown_spans', 0):,}"),
        _row("**Repeated Tokens (3+)**",
             f"{sum(p.get('repeated_tokens_count', 0) for p in parts.values()):,}",
             f"{c.get('repeated_tokens_count', 0):,}",
             f"{m.get('repeated_tokens_count', 0):,}",
             f"{n.get('repeated_tokens_count', 0):,}"),
        _row("**Long Tokens (>30c)**",
             f"{sum(p.get('long_tokens_count', 0) for p in parts.values()):,}",
             f"{c.get('long_tokens_count', 0):,}",
             f"{m.get('long_tokens_count', 0):,}",
             f"{n.get('long_tokens_count', 0):,}"),
        "",
        "---",
        "",
        "## Token Category Distribution",
        "",
        _row("Category", "Total Count", "Clean", "Mixed", "Noisy"),
        _row(":---", ":---", ":---", ":---", ":---"),
    ]

    all_cats = [
        "WORD", "PUNCTUATION", "NUMBER", "HASHTAG",
        "MENTION", "EMOJI", "EMAIL", "URL", "UNKNOWN",
    ]
    for cat_name in all_cats:
        lines.append(_row(
            f"**{cat_name}**",
            f"{cats.get(cat_name, 0):,}",
            f"{c_cats.get(cat_name, 0):,}",
            f"{m_cats.get(cat_name, 0):,}",
            f"{n_cats.get(cat_name, 0):,}",
        ))

    lines.extend([
        "",
        "---",
        "",
        "## Token Length Distributions",
        "",
        _row("Bucket (chars)", "Total Tokens", "Clean", "Mixed", "Noisy"),
        _row(":---", ":---", ":---", ":---", ":---"),
    ])

    for b_key in ["1-3", "4-6", "7-10", "11-20", "21-30", "31+"]:
        c_cnt = c_b.get(b_key, 0)
        m_cnt = m_b.get(b_key, 0)
        n_cnt = n_b.get(b_key, 0)
        tot_cnt = c_cnt + m_cnt + n_cnt
        lines.append(_row(
            f"**{b_key}**",
            f"{tot_cnt:,}",
            f"{c_cnt:,}",
            f"{m_cnt:,}",
            f"{n_cnt:,}",
        ))

    lines.extend([
        "",
        "---",
        "",
        "## Visual Diagnostic Charts",
        "",
        "1. `charts/token_categories.png` - Distribution of token categories across partitions.",
        "2. `charts/unknown_spans.png` - Unknown span rates by partition.",
        "3. `charts/token_lengths.png` - Token length distribution histograms.",
        "4. `charts/special_tokens.png` - Special tokens (Hashtag, Mention, Emoji, Number, URL).",
        "5. `charts/repetitions.png` - Repeated character tokens across partitions.",
        "",
    ])

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    return report


def generate_charts(summary: Dict[str, Any], charts_dir: Path) -> None:
    """Generate diagnostic PNG charts."""
    if plt is None:
        raise ImportError(
            "matplotlib is required for chart generation. Install with: pip install '.[ml]'"
        )

    charts_dir.mkdir(parents=True, exist_ok=True)
    parts = summary.get("partitions", {})

    # 1. Token categories chart
    cats = ["WORD", "PUNCTUATION", "NUMBER", "UNKNOWN", "HASHTAG", "MENTION", "EMOJI"]
    partitions = ["roman_clean", "roman_mixed", "roman_noisy"]
    cat_counts = {
        p: [parts.get(p, {}).get("categories", {}).get(c, 0) for c in cats]
        for p in partitions
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(cats))
    width = 0.25
    for i, p in enumerate(partitions):
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, cat_counts[p], width=width, label=p)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(cats, rotation=25)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Token Categories by Partition (v1)")
    ax.legend()
    fig.savefig(charts_dir / "token_categories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Unknown spans chart
    unknown_rates = [parts.get(p, {}).get("unknown_span_rate", 0.0) * 100 for p in partitions]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(partitions, unknown_rates, color=["#4285F4", "#FBBC05", "#EA4335"])
    ax.set_ylabel("Unknown Span Rate (%)")
    ax.set_title("Unknown Span Rate by Partition")
    for b in bars:
        yval = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2, yval + 0.01, f"{yval:.4f}%", ha="center", va="bottom"
        )
    fig.savefig(charts_dir / "unknown_spans.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Token lengths chart
    buckets = ["1-3", "4-6", "7-10", "11-20", "21-30", "31+"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, p in enumerate(partitions):
        b_counts = [parts.get(p, {}).get("token_length_buckets", {}).get(b, 0) for b in buckets]
        offsets = [xi + i * width for xi in range(len(buckets))]
        ax.bar(offsets, b_counts, width=width, label=p)
    ax.set_xticks([xi + width for xi in range(len(buckets))])
    ax.set_xticklabels(buckets)
    ax.set_ylabel("Token Count")
    ax.set_title("Token Length Distribution by Partition")
    ax.legend()
    fig.savefig(charts_dir / "token_lengths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Special tokens chart
    specials = ["HASHTAG", "MENTION", "EMOJI", "NUMBER", "URL", "EMAIL"]
    spec_counts = {
        p: [parts.get(p, {}).get("categories", {}).get(s, 0) for s in specials]
        for p in partitions
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, p in enumerate(partitions):
        offsets = [xi + i * width for xi in range(len(specials))]
        ax.bar(offsets, spec_counts[p], width=width, label=p)
    ax.set_xticks([xi + width for xi in range(len(specials))])
    ax.set_xticklabels(specials)
    ax.set_ylabel("Count")
    ax.set_title("Special Tokens by Partition")
    ax.legend()
    fig.savefig(charts_dir / "special_tokens.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Repetitions chart
    rep_counts = [parts.get(p, {}).get("repeated_tokens_count", 0) for p in partitions]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(partitions, rep_counts, color=["#34A853", "#4285F4", "#EA4335"])
    ax.set_ylabel("Repeated Tokens Count")
    ax.set_title("Repeated Character Tokens (3+ identical) by Partition")
    for b in bars:
        yval = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2, yval + 50, f"{int(yval):,}", ha="center", va="bottom"
        )
    fig.savefig(charts_dir / "repetitions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_html_token_gallery(
    samples: List[Dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    """Generate standalone HTML gallery displaying tokens and boundaries with colored tags."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cat_colors = {
        "WORD": "#1a73e8",
        "PUNCTUATION": "#5f6368",
        "NUMBER": "#188038",
        "EMOJI": "#f29900",
        "HASHTAG": "#a142f4",
        "MENTION": "#d93025",
        "URL": "#12b5cb",
        "EMAIL": "#e37400",
        "UNKNOWN": "#d93025",
    }

    cards_html: List[str] = []
    for s in samples:
        row_id = html.escape(str(s.get("id", "")))
        partition = html.escape(str(s.get("partition", "")))
        note = html.escape(str(s.get("note", "")))
        masked_text = html.escape(mask_pii(s.get("raw_text", s.get("text", ""))))

        token_spans: List[str] = []
        for t in s.get("tokens", []):
            tok_text = html.escape(t.get("text", ""))
            tok_cat = t.get("category", "UNKNOWN")
            color = cat_colors.get(tok_cat, "#5f6368")
            token_spans.append(
                f'<span style="display:inline-block; margin:2px; padding:2px 6px; '
                f'border-radius:4px; background:#f1f3f4; border:1px solid #dadce0; '
                f'font-family:monospace; font-size:13px;">'
                f'{tok_text}'
                f'<sub style="font-size:9px; color:{color}; font-weight:bold; '
                f'margin-left:3px;">{tok_cat}</sub>'
                f'</span>'
            )

        card = (
            f'<div style="margin-bottom:16px; padding:12px; border:1px solid #e0e0e0; '
            f'border-radius:8px; background:#fff;">\n'
            f'  <div style="font-size:12px; color:#70757a; margin-bottom:6px;">\n'
            f'    <strong>ID:</strong> {row_id} | <strong>Partition:</strong> {partition} '
            f'{"| <em>" + note + "</em>" if note else ""}\n'
            f'  </div>\n'
            f'  <div style="font-size:13px; color:#202124; margin-bottom:8px; '
            f'background:#f8f9fa; padding:6px 10px; border-radius:4px;">\n'
            f'    <strong>Raw:</strong> {masked_text}\n'
            f'  </div>\n'
            f'  <div style="line-height:1.8;">\n'
            f'    {" ".join(token_spans)}\n'
            f'  </div>\n'
            f'</div>\n'
        )
        cards_html.append(card)

    doc = (
        f'<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        f'<title>{html.escape(title)}</title>\n'
        f'<style>\n'
        f'body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; '
        f'background:#f8f9fa; margin:20px; }}\n'
        f'.container {{ max-width:960px; margin:0 auto; }}\n'
        f'h1 {{ font-size:22px; color:#202124; }}\n'
        f'</style>\n</head>\n<body>\n<div class="container">\n'
        f'<h1>{html.escape(title)}</h1>\n'
        f'<p style="color:#5f6368; font-size:14px;">'
        f'Total reviewed examples: {len(samples)}. PII masked for privacy.</p>\n'
        f'{"".join(cards_html)}\n'
        f'</div>\n</body>\n</html>'
    )
    output_path.write_text(doc, encoding="utf-8")


def generate_preprocessing_diff_gallery(cases: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate HTML gallery comparing raw text across preprocessing modes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_html: List[str] = []
    for c in cases:
        row_id = html.escape(str(c.get("id", "")))
        desc = html.escape(str(c.get("description", "")))
        raw = html.escape(mask_pii(c.get("raw", "")))
        unicode_norm = html.escape(mask_pii(c.get("unicode_whitespace", "")))
        protected = html.escape(mask_pii(c.get("protected_spans", "")))
        variant_norm = html.escape(mask_pii(c.get("variant_normalized", "")))

        row = (
            f'<tr style="border-bottom:1px solid #e0e0e0;">\n'
            f'  <td style="padding:8px; font-family:monospace; font-size:12px; '
            f'vertical-align:top;">{row_id}<br><small style="color:#70757a;">{desc}</small></td>\n'
            f'  <td style="padding:8px; font-size:13px; vertical-align:top; '
            f'background:#fafafa;">{raw}</td>\n'
            f'  <td style="padding:8px; font-size:13px; vertical-align:top;">{unicode_norm}</td>\n'
            f'  <td style="padding:8px; font-size:13px; vertical-align:top;">{protected}</td>\n'
            f'  <td style="padding:8px; font-size:13px; vertical-align:top; '
            f'background:#f0f8ff;">{variant_norm}</td>\n'
            f'</tr>\n'
        )
        rows_html.append(row)

    table = (
        f'<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        f'<title>Preprocessing Modes Diff Gallery</title>\n'
        f'<style>\n'
        f'body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; '
        f'background:#f8f9fa; margin:20px; }}\n'
        f'table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; '
        f'overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.1); }}\n'
        f'th {{ background:#f1f3f4; padding:10px; font-size:13px; text-align:left; '
        f'color:#202124; }}\n'
        f'h1 {{ font-size:22px; color:#202124; }}\n'
        f'</style>\n</head>\n<body>\n'
        f'<div style="max-width:1100px; margin:0 auto;">\n'
        f'<h1>Preprocessing Modes Before / After Gallery</h1>\n'
        f'<p style="color:#5f6368; font-size:14px;">Comparing Raw, Unicode/Whitespace, '
        f'Protected Span, and Approved Variant normalization modes.</p>\n'
        f'<table>\n<thead>\n<tr>\n'
        f'  <th style="width:15%;">Case / ID</th>\n'
        f'  <th style="width:22%;">1. Raw Text</th>\n'
        f'  <th style="width:21%;">2. Unicode / Whitespace</th>\n'
        f'  <th style="width:21%;">3. Protected Spans</th>\n'
        f'  <th style="width:21%;">4. Approved Variants</th>\n'
        f'</tr>\n</thead>\n<tbody>\n'
        f'{"".join(rows_html)}\n'
        f'</tbody>\n</table>\n</div>\n</body>\n</html>'
    )
    output_path.write_text(table, encoding="utf-8")


def generate_v1_v2_comparison_report(
    v1: Dict[str, Any],
    v2: Dict[str, Any],
    output_path: Path,
) -> str:
    """Generate Markdown report comparing v1 and v2 tokenizer coverage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    v1_u = v1.get("unknown_spans", 0)
    v2_u = v2.get("unknown_spans", 0)
    u_diff = v2_u - v1_u
    u_pct = (u_diff / v1_u * 100) if v1_u else 0.0

    v1_r = v1.get("unknown_span_rate", 0.0) * 100
    v2_r = v2.get("unknown_span_rate", 0.0) * 100

    v1_tot_tok = v1.get("total_tokens", 0)
    v2_tot_tok = v2.get("total_tokens", 0)
    avg_diff = v2.get("avg_tokens_per_row", 0.0) - v1.get("avg_tokens_per_row", 0.0)

    lines = [
        "# Tokenizer v1 vs v2 Comparison Report",
        "",
        "## Summary Comparison",
        "",
        _row("Metric", "v1 Tokenizer", "v2 Tokenizer", "Delta"),
        _row(":---", ":---", ":---", ":---"),
        _row("**Total Rows Processed**", f"{v1.get('total_rows', 0):,}",
             f"{v2.get('total_rows', 0):,}", "0"),
        _row("**Total Tokens**", f"{v1_tot_tok:,}", f"{v2_tot_tok:,}",
             f"{v2_tot_tok - v1_tot_tok:+,}"),
        _row("**Avg Tokens / Row**", f"{v1.get('avg_tokens_per_row', 0.0):.2f}",
             f"{v2.get('avg_tokens_per_row', 0.0):.2f}", f"{avg_diff:+.2f}"),
        _row("**Total Unknown Spans**", f"{v1_u:,}", f"{v2_u:,}",
             f"{u_diff:+,} ({u_pct:+.1f}%)"),
        _row("**Unknown Span Rate**", f"{v1_r:.4f}%", f"{v2_r:.4f}%",
             f"{v2_r - v1_r:+.4f}%"),
        _row("**Tokenizer Crashes / Errors**", f"{v1.get('error_count', 0)}",
             f"{v2.get('error_count', 0)}", "0"),
        _row("**Uncovered Chars**", f"{v1.get('uncovered_chars', 0)}",
             f"{v2.get('uncovered_chars', 0)}", "0"),
        "",
        "---",
        "",
        "## Partition-Level Comparison",
        "",
    ]

    for p in ["roman_clean", "roman_mixed", "roman_noisy"]:
        v1_p = v1.get("partitions", {}).get(p, {})
        v2_p = v2.get("partitions", {}).get(p, {})
        v1_p_u = v1_p.get("unknown_spans", 0)
        v2_p_u = v2_p.get("unknown_spans", 0)
        v1_p_r = v1_p.get("unknown_span_rate", 0.0) * 100
        v2_p_r = v2_p.get("unknown_span_rate", 0.0) * 100
        lines.extend([
            f"### Partition: `{p}`",
            f"- **Rows:** {v1_p.get('total_rows', 0):,}",
            f"- **v1 Unknown Spans:** {v1_p_u:,} ({v1_p_r:.4f}%)",
            f"- **v2 Unknown Spans:** {v2_p_u:,} ({v2_p_r:.4f}%)",
            f"- **Delta:** {v2_p_u - v1_p_u:+,} unknown spans",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Token Category Changes",
        "",
        _row("Category", "v1 Count", "v2 Count", "Delta"),
        _row(":---", ":---", ":---", ":---"),
    ])

    for cat in ["WORD", "PUNCTUATION", "NUMBER", "UNKNOWN", "HASHTAG", "MENTION", "EMOJI"]:
        v1_c = v1.get("categories", {}).get(cat, 0)
        v2_c = v2.get("categories", {}).get(cat, 0)
        lines.append(_row(
            f"**{cat}**",
            f"{v1_c:,}",
            f"{v2_c:,}",
            f"{v2_c - v1_c:+,}",
        ))

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    return report
