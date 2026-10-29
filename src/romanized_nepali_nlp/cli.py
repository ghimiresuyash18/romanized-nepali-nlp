"""Command Line Interface (CLI) for Romanized Nepali NLP toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from romanized_nepali_nlp.normalization import generate_diff
from romanized_nepali_nlp.pipeline import PipelineConfig, PreprocessingPipeline
from romanized_nepali_nlp.spans import (
    format_html,
    format_markdown,
    format_terminal,
    format_tokens_json,
    get_diagnostics,
)
from romanized_nepali_nlp.tokenizer import Tokenizer


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="romanized-nepali",
        description="Romanized Nepali NLP Toolkit CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Tokenize subcommand
    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize Romanized Nepali text")
    tokenize_parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Input text to tokenize (or pass '-' to read from stdin)",
    )
    tokenize_parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to a file to read input text from",
    )
    tokenize_parser.add_argument(
        "--format",
        choices=["table", "brackets", "json", "markdown"],
        default="table",
        help="Output format (default: table)",
    )
    tokenize_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print span coverage and uncovered non-whitespace characters",
    )
    tokenize_parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Save an interactive HTML token visualization to the specified filepath",
    )
    tokenize_parser.add_argument(
        "--no-hyphens",
        action="store_true",
        help="Split intra-word hyphens into separate tokens",
    )
    tokenize_parser.add_argument(
        "--no-apostrophes",
        action="store_true",
        help="Split intra-word apostrophes into separate tokens",
    )

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Normalize and preprocess Romanized Nepali text"
    )
    preprocess_parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Input text to preprocess (or pass '-' to read from stdin)",
    )
    preprocess_parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to a file to read input text from",
    )
    preprocess_parser.add_argument(
        "--format",
        choices=["text", "json", "diff"],
        default="text",
        help="Output format (default: text)",
    )
    preprocess_parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert text to lowercase (disabled by default)",
    )
    preprocess_parser.add_argument(
        "--reduce-repetitions",
        action="store_true",
        help="Bound elongated repeated characters",
    )
    preprocess_parser.add_argument(
        "--no-variants",
        action="store_true",
        help="Disable spelling variant normalization",
    )
    preprocess_parser.add_argument(
        "--variant-policy",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Variant normalization policy (default: conservative)",
    )

    return parser


def read_input_text(args: argparse.Namespace) -> str:
    """Read text from file, positional argument, or stdin."""
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            sys.stderr.write(f"Error: File not found: {args.file}\n")
            sys.exit(1)
        return file_path.read_text(encoding="utf-8")

    if args.text == "-" or (args.text is None and not sys.stdin.isatty()):
        return sys.stdin.read()

    if args.text is not None:
        return args.text

    sys.stderr.write("Error: No input text provided. Provide text, --file, or pipe via stdin.\n")
    sys.exit(1)


def run_tokenize(args: argparse.Namespace) -> int:
    text = read_input_text(args)

    tokenizer = Tokenizer(
        allow_apostrophes=not args.no_apostrophes,
        allow_hyphens=not args.no_hyphens,
    )

    tokens = tokenizer.tokenize(text)

    # Export HTML if requested
    if args.html:
        html_content = format_html(tokens, text, title="Romanized Nepali Tokenizer Visualization")
        out_path = Path(args.html)
        out_path.write_text(html_content, encoding="utf-8")
        sys.stdout.write(f"HTML visualization saved to: {out_path.resolve()}\n")

    # Format output
    if args.format == "brackets":
        sys.stdout.write(format_terminal(tokens) + "\n")
    elif args.format == "json":
        sys.stdout.write(format_tokens_json(tokens) + "\n")
    elif args.format == "markdown":
        sys.stdout.write(format_markdown(tokens) + "\n")
    else:
        # Table format
        sys.stdout.write(f"{'#':<4} {'TEXT':<25} {'CATEGORY':<15} {'START':<6} {'END':<6}\n")
        sys.stdout.write("-" * 60 + "\n")
        for i, t in enumerate(tokens, 1):
            sys.stdout.write(
                f"{i:<4} {t.text:<25} {t.category.value:<15} {t.start:<6} {t.end:<6}\n"
            )

    # Diagnostics if requested
    if args.diagnostics:
        diag = get_diagnostics(tokens, text)
        sys.stdout.write("\n--- Diagnostics ---\n")
        sys.stdout.write(json.dumps(diag, indent=2) + "\n")

    return 0


def run_preprocess(args: argparse.Namespace) -> int:
    text = read_input_text(args)

    config = PipelineConfig(
        lowercase=args.lowercase,
        reduce_repetitions=args.reduce_repetitions,
        normalize_variants=not args.no_variants,
        variant_policy=args.variant_policy,
    )
    pipeline = PreprocessingPipeline(config=config)
    result = pipeline.preprocess(text)

    if args.format == "json":
        sys.stdout.write(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n")
    elif args.format == "diff":
        diff_view = generate_diff(result.raw_text, result.normalized_text)
        sys.stdout.write(diff_view + "\n")
    else:
        sys.stdout.write(result.normalized_text + "\n")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "tokenize":
        return run_tokenize(args)
    if args.subcommand == "preprocess":
        return run_preprocess(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
