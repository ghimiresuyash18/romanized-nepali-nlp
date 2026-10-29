"""Token span utilities, validation, and visualization."""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Tuple

from romanized_nepali_nlp.types import Token, TokenCategory


class SpanValidationError(ValueError):
    """Raised when token spans violate boundary, ordering, or overlap constraints."""


def validate_token_spans(tokens: List[Token], text: str) -> None:
    """Validate that token spans are strictly within bounds, non-backward, and non-overlapping.

    Raises SpanValidationError on any violation.
    """
    text_len = len(text)
    last_end = 0

    for i, token in enumerate(tokens):
        if not isinstance(token.category, TokenCategory):
            try:
                TokenCategory(token.category)
            except Exception as e:
                raise SpanValidationError(f"Token {i} has invalid category {token.category}: {e}")

        if token.start < 0:
            raise SpanValidationError(
                f"Token {i} ({token.text!r}) has negative start offset: {token.start}"
            )
        if token.end > text_len:
            raise SpanValidationError(
                f"Token {i} ({token.text!r}) end offset {token.end} exceeds text length {text_len}"
            )
        if token.start >= token.end:
            raise SpanValidationError(
                f"Token {i} ({token.text!r}) backward or zero-length span: "
                f"[{token.start}:{token.end}]"
            )
        if token.start < last_end:
            raise SpanValidationError(
                f"Token {i} ({token.text!r}) at [{token.start}:{token.end}] "
                f"overlaps with previous token ending at {last_end}"
            )

        # Confirm slice content matches token.text
        slice_text = text[token.start : token.end]
        if slice_text != token.text:
            raise SpanValidationError(
                f"Token {i} text mismatch: token.text={token.text!r} but "
                f"text[{token.start}:{token.end}]={slice_text!r}"
            )

        last_end = token.end


def get_uncovered_spans(tokens: List[Token], text: str) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, uncovered_substring) for gaps between tokens."""
    uncovered: List[Tuple[int, int, str]] = []
    last_pos = 0

    for token in tokens:
        if token.start > last_pos:
            gap_text = text[last_pos : token.start]
            uncovered.append((last_pos, token.start, gap_text))
        last_pos = max(last_pos, token.end)

    if last_pos < len(text):
        gap_text = text[last_pos:]
        uncovered.append((last_pos, len(text), gap_text))

    return uncovered


def get_diagnostics(tokens: List[Token], text: str) -> Dict[str, Any]:
    """Calculate coverage diagnostics and highlight non-whitespace uncovered characters."""
    uncovered = get_uncovered_spans(tokens, text)
    non_ws_gaps = [{"start": s, "end": e, "text": t.strip()} for s, e, t in uncovered if t.strip()]
    total_token_chars = sum(len(t.text) for t in tokens)
    coverage_ratio = total_token_chars / len(text) if len(text) > 0 else 1.0

    return {
        "text_length": len(text),
        "token_count": len(tokens),
        "total_token_chars": total_token_chars,
        "coverage_ratio": round(coverage_ratio, 4),
        "uncovered_gaps_count": len(uncovered),
        "uncovered_non_whitespace_count": len(non_ws_gaps),
        "uncovered_non_whitespace": non_ws_gaps,
    }


def format_terminal(tokens: List[Token], text: str | None = None) -> str:
    """Format tokens as readable bracketed text: [text|CATEGORY]."""
    return " ".join(f"[{t.text}|{t.category.value}]" for t in tokens)


def format_tokens_json(tokens: List[Token], indent: int = 2) -> str:
    """Serialize tokens to formatted JSON string."""
    return json.dumps([t.to_dict() for t in tokens], ensure_ascii=False, indent=indent)


# Category color palette for HTML visualization
CATEGORY_COLORS: Dict[str, str] = {
    TokenCategory.WORD.value: "#2563eb",  # Blue
    TokenCategory.ENGLISH_WORD.value: "#0284c7",  # Light blue
    TokenCategory.URL.value: "#7c3aed",  # Purple
    TokenCategory.EMAIL.value: "#9333ea",  # Violet
    TokenCategory.MENTION.value: "#059669",  # Green
    TokenCategory.HASHTAG.value: "#10b981",  # Emerald
    TokenCategory.NUMBER.value: "#d97706",  # Amber
    TokenCategory.EMOJI.value: "#e11d48",  # Rose
    TokenCategory.PUNCTUATION.value: "#64748b",  # Slate
    TokenCategory.UNKNOWN.value: "#dc2626",  # Red
}


def format_html(
    tokens: List[Token],
    text: str,
    title: str = "Tokenization Visualizer",
) -> str:
    """Generate interactive, accessible HTML visualization with color-coded token badges."""
    validate_token_spans(tokens, text)

    rendered_parts: List[str] = []
    last_idx = 0

    for t in tokens:
        if t.start > last_idx:
            # Render gap / whitespace
            gap = html.escape(text[last_idx : t.start])
            rendered_parts.append(f"<span class='gap'>{gap}</span>")

        color = CATEGORY_COLORS.get(t.category.value, "#475569")
        escaped_text = html.escape(t.text)
        meta_tooltip = f"Category: {t.category.value}&#10;Span: [{t.start}:{t.end}]"
        if t.metadata:
            meta_tooltip += f"&#10;Meta: {json.dumps(t.metadata)}"

        style = f"background-color: {color}22; border-color: {color}; color: {color};"
        badge = (
            f"<span class='token' style='{style}' title='{meta_tooltip}'>"
            f"<span class='text'>{escaped_text}</span>"
            f"<sub class='cat'>{t.category.value}</sub>"
            f"</span>"
        )
        rendered_parts.append(badge)
        last_idx = t.end

    if last_idx < len(text):
        gap = html.escape(text[last_idx:])
        rendered_parts.append(f"<span class='gap'>{gap}</span>")

    rendered_body = "".join(rendered_parts)

    legend_items = "".join(
        f"<span class='legend-item' style='background: {col}22; border: 1px solid {col}; "
        f"color: {col};'>{cat}</span>"
        for cat, col in CATEGORY_COLORS.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 2rem;
      background: #f8fafc;
      color: #0f172a;
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
      padding: 0.75rem;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
    }}
    .legend-item {{
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .viewer {{
      line-height: 2.4;
      font-size: 1.1rem;
      padding: 1.5rem;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
    }}
    .token {{
      display: inline-flex;
      align-items: baseline;
      border-width: 1px;
      border-style: solid;
      border-radius: 4px;
      padding: 0.1rem 0.35rem;
      margin: 0 0.15rem;
      cursor: help;
    }}
    .token .text {{ font-weight: 500; }}
    .token .cat {{
      font-size: 0.65rem;
      font-weight: 700;
      margin-left: 0.3rem;
      text-transform: uppercase;
      opacity: 0.85;
    }}
    .gap {{ white-space: pre-wrap; color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="legend">{legend_items}</div>
  <div class="viewer">{rendered_body}</div>
</body>
</html>
"""


def format_markdown(tokens: List[Token]) -> str:
    """Format tokens as a Markdown table."""
    lines = [
        "| # | Token Text | Category | Start | End | Normalized |",
        "|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(tokens, 1):
        norm = t.normalized_text if t.normalized_text is not None else "-"
        lines.append(
            f"| {i} | `{t.text}` | `{t.category.value}` | {t.start} | {t.end} | `{norm}` |"
        )
    return "\n".join(lines)
