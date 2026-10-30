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

    _add_dataset_subparser(subparsers)
    _add_sentiment_subparsers(subparsers)
    _add_corpus_subparser(subparsers)

    return parser


def _add_corpus_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register corpus ingestion, profiling, and quality triage subcommand (Phase 5.5)."""
    cp = subparsers.add_parser("corpus", help="External corpus ingestion, profiling, and triage")
    cp.add_argument("file", help="Path to raw corpus file (JSONL, CSV, or text)")
    cp.add_argument(
        "--benchmark",
        default=None,
        help="Path to benchmark dataset to audit against duplicate leakage",
    )
    cp.add_argument(
        "-o",
        "--output-dir",
        default="data/processed/nfr",
        help="Output directory for profile and partition files (default: data/processed/nfr)",
    )
    cp.add_argument(
        "--dataset-name",
        default="Boredoom17/Nepali-Flow-Roman",
        help="Corpus identifier (default: Boredoom17/Nepali-Flow-Roman)",
    )
    cp.add_argument("--revision", default="pinned-snapshot", help="Upstream commit or tag")
    cp.add_argument("--license", default="CC BY 4.0", help="Source license field")
    cp.add_argument(
        "--attribution",
        default="Aadarsha Chhetri. (2026). Nepali-Flow-Roman.",
        help="Upstream attribution string",
    )


def _add_sentiment_subparsers(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register train, predict, and evaluate subcommands (Tasks 18-21)."""
    # --- train ---
    tr = subparsers.add_parser("train", help="Train a sentiment baseline model")
    tr.add_argument("file", help="Path to CSV or JSONL training dataset file")
    tr.add_argument("--text-col", default="text", help="Column name for text field")
    tr.add_argument("--label-col", default="label", help="Column name for label field")
    tr.add_argument("--id-col", default="id", help="Column name for ID field")
    tr.add_argument(
        "--model-type",
        choices=["majority", "word", "char"],
        default="char",
        help="Baseline model type (default: char)",
    )
    tr.add_argument(
        "--analyzer",
        choices=["char", "char_wb", "word"],
        default="char_wb",
        help="Feature analyzer (default: char_wb)",
    )
    tr.add_argument("--min-n", type=int, default=2, help="Minimum n-gram range (default: 2)")
    tr.add_argument("--max-n", type=int, default=5, help="Maximum n-gram range (default: 5)")
    tr.add_argument("--min-df", type=int, default=1, help="Minimum document frequency (default: 1)")
    tr.add_argument("--max-features", type=int, default=None, help="Maximum features cap")
    tr.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting (default: balanced)",
    )
    tr.add_argument(
        "--c", type=float, default=1.0, help="Inverse regularization strength (default: 1.0)"
    )
    tr.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    tr.add_argument("--no-preprocess", action="store_true", help="Skip text preprocessing pipeline")
    tr.add_argument("--split-manifest", default=None, help="Path to split manifest JSON file")
    tr.add_argument("--config", default=None, help="Path to JSON configuration file")
    tr.add_argument("-o", "--output-dir", default="runs/latest", help="Output run directory")

    # --- predict ---
    pr = subparsers.add_parser("predict", help="Predict sentiment labels and scores")
    pr.add_argument("file", help="Path to CSV, JSONL, or text file (or '-' for stdin)")
    pr.add_argument("--model", required=True, help="Path to model artifact or run directory")
    pr.add_argument("--text-col", default="text", help="Column name for text field")
    pr.add_argument("--id-col", default="id", help="Column name for ID field")
    pr.add_argument(
        "--label-col", default="label", help="Column name for optional gold label field"
    )
    pr.add_argument(
        "--format",
        choices=["jsonl", "csv", "table"],
        default="jsonl",
        help="Prediction output format (default: jsonl)",
    )
    pr.add_argument("-o", "--output", default=None, help="Write output to file instead of stdout")

    # --- evaluate ---
    ev = subparsers.add_parser("evaluate", help="Evaluate predictions against gold labels")
    ev.add_argument("predictions", help="Path to predictions JSONL file")
    ev.add_argument(
        "--dataset", default=None, help="Optional dataset file for metadata / slice extraction"
    )
    ev.add_argument(
        "--format",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Report format (default: markdown)",
    )
    ev.add_argument("-o", "--out", default=None, help="Write report to file instead of stdout")
    ev.add_argument(
        "--export-errors", default=None, help="Export misclassified errors to JSONL file"
    )


def _add_dataset_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'dataset' subcommand and its sub-subcommands."""
    ds_parser = subparsers.add_parser(
        "dataset", help="Dataset loading, validation, quality reporting, and splitting"
    )
    ds_sub = ds_parser.add_subparsers(dest="ds_command", help="Dataset operations")

    # --- preview ---
    pr = ds_sub.add_parser(
        "preview", help="Preview a dataset file (schema, first rows, missingness)"
    )
    pr.add_argument("file", help="Path to CSV or JSONL dataset file")
    pr.add_argument("--text-col", default="text", help="Column name for text field")
    pr.add_argument("--label-col", default="label", help="Column name for label field")
    pr.add_argument("--id-col", default="id", help="Column name for record ID")
    pr.add_argument("--delimiter", default=",", help="CSV delimiter (default: comma)")
    pr.add_argument("--n", type=int, default=5, help="Number of sample rows to show")

    # --- report ---
    rp = ds_sub.add_parser("report", help="Generate a quality report (JSON or Markdown)")
    rp.add_argument("file", help="Path to CSV or JSONL dataset file")
    rp.add_argument("--text-col", default="text")
    rp.add_argument("--label-col", default="label")
    rp.add_argument("--id-col", default="id")
    rp.add_argument("--delimiter", default=",")
    rp.add_argument("--format", choices=["json", "markdown", "html"], default="markdown")
    rp.add_argument("--out", default=None, help="Write report to this file instead of stdout")
    rp.add_argument(
        "--no-token-slices",
        action="store_true",
        help="Skip tokenizing each record (faster for large datasets)",
    )

    # --- split ---
    sp = ds_sub.add_parser("split", help="Split a dataset into train/validation/test")
    sp.add_argument("file", help="Path to CSV or JSONL dataset file")
    sp.add_argument("--text-col", default="text")
    sp.add_argument("--label-col", default="label")
    sp.add_argument("--id-col", default="id")
    sp.add_argument("--delimiter", default=",")
    sp.add_argument("--train", type=float, default=0.70)
    sp.add_argument("--validation", type=float, default=0.15)
    sp.add_argument("--test", type=float, default=0.15)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--no-stratify", action="store_true", help="Disable label stratification")
    sp.add_argument("--no-group-dups", action="store_true", help="Disable duplicate grouping")
    sp.add_argument("--out-dir", default=".", help="Directory to write split JSONL files")
    sp.add_argument("--manifest", default=None, help="Write split manifest to this JSON file")


def _make_col_map(args: argparse.Namespace):
    from romanized_nepali_nlp.dataset import ColumnMap

    return ColumnMap(
        text=args.text_col,
        label=args.label_col,
        id=args.id_col,
    )


def run_dataset_preview(args: argparse.Namespace) -> int:
    from romanized_nepali_nlp.dataset import load, preview

    col = _make_col_map(args)
    result = load(Path(args.file), col=col, delimiter=args.delimiter)
    sys.stdout.write(preview(result, n=args.n) + "\n")
    return 0


def run_dataset_report(args: argparse.Namespace) -> int:
    from romanized_nepali_nlp.dataset import load
    from romanized_nepali_nlp.quality import compute_stats, report_html

    col = _make_col_map(args)
    result = load(Path(args.file), col=col, delimiter=args.delimiter)
    stats = compute_stats(
        result.records, issues=result.issues, tokenize_slices=not args.no_token_slices
    )
    if args.format == "json":
        text = json.dumps(stats.to_dict(), ensure_ascii=False, indent=2)
    elif args.format == "html":
        text = report_html(stats)
    else:
        text = stats.to_markdown()

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        sys.stdout.write(f"Report written to {args.out}\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


def run_dataset_split(args: argparse.Namespace) -> int:
    from romanized_nepali_nlp.dataset import load
    from romanized_nepali_nlp.splitting import (
        SplitConfig,
        check_leakage,
        split,
        split_balance_report,
    )

    col = _make_col_map(args)
    result = load(Path(args.file), col=col, delimiter=args.delimiter)
    cfg = SplitConfig(
        train=args.train,
        validation=args.validation,
        test=args.test,
        seed=args.seed,
        stratify_by_label=not args.no_stratify,
        group_by_duplicates=not args.no_group_dups,
    )
    split_result = split(result.records, config=cfg)
    leaked = check_leakage(split_result)
    if leaked:
        sys.stderr.write(f"ERROR: Leakage detected for {len(leaked)} record IDs: {leaked[:5]}...\n")
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        recs = split_result.by_name(split_name)
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        sys.stdout.write(f"Wrote {len(recs)} records to {out_path}\n")
    if args.manifest:
        split_result.save_manifest(args.manifest)
        sys.stdout.write(f"Manifest written to {args.manifest}\n")
    sys.stdout.write("\n" + split_balance_report(split_result) + "\n")
    return 0


def run_dataset(args: argparse.Namespace) -> int:
    ds_cmd = getattr(args, "ds_command", None)
    if not ds_cmd:
        sys.stderr.write("Usage: romanized-nepali dataset [preview|report|split] ...\n")
        return 1
    if ds_cmd == "preview":
        return run_dataset_preview(args)
    if ds_cmd == "report":
        return run_dataset_report(args)
    if ds_cmd == "split":
        return run_dataset_split(args)
    return 0


def iter_input_lines(args: argparse.Namespace):
    """Stream input lines one by one to avoid OOM on large files."""
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            sys.stderr.write(f"Error: File not found: {args.file}\n")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line
        return

    if args.text == "-" or (args.text is None and not sys.stdin.isatty()):
        for line in sys.stdin:
            yield line
        return

    if args.text is not None:
        yield args.text
        return

    sys.stderr.write("Error: No input text provided. Provide text, --file, or pipe via stdin.\n")
    sys.exit(1)


def read_input_text(args: argparse.Namespace) -> str:
    """Read full text when required for whole-document formats (JSON, diff, HTML)."""
    return "".join(iter_input_lines(args))


def run_tokenize(args: argparse.Namespace) -> int:
    # Stream line-by-line for brackets and table to avoid memory exhaustion on large inputs
    is_stream = (
        args.file is not None or args.text == "-" or (args.text is None and not sys.stdin.isatty())
    )
    can_stream = is_stream and not args.html and not args.diagnostics
    if can_stream and args.format in ("brackets", "table"):
        tokenizer = Tokenizer(
            allow_apostrophes=not args.no_apostrophes,
            allow_hyphens=not args.no_hyphens,
        )
        if args.format == "table":
            sys.stdout.write(f"{'#':<4} {'TEXT':<25} {'CATEGORY':<15} {'START':<6} {'END':<6}\n")
            sys.stdout.write("-" * 60 + "\n")
        idx = 1
        for line in iter_input_lines(args):
            line_clean = line.rstrip("\r\n")
            line_tokens = tokenizer.tokenize(line_clean)
            if args.format == "brackets":
                sys.stdout.write(format_terminal(line_tokens) + "\n")
            else:
                for t in line_tokens:
                    sys.stdout.write(
                        f"{idx:<4} {t.text:<25} {t.category.value:<15} {t.start:<6} {t.end:<6}\n"
                    )
                    idx += 1
        return 0

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
    config = PipelineConfig(
        lowercase=args.lowercase,
        reduce_repetitions=args.reduce_repetitions,
        normalize_variants=not args.no_variants,
        variant_policy=args.variant_policy,
    )
    pipeline = PreprocessingPipeline(config=config)

    # Stream line-by-line in plain text format to avoid memory exhaustion on large files
    is_stream = (
        args.file is not None or args.text == "-" or (args.text is None and not sys.stdin.isatty())
    )
    if args.format == "text" and is_stream:
        for line in iter_input_lines(args):
            line_clean = line.rstrip("\r\n")
            res = pipeline.preprocess(line_clean)
            sys.stdout.write(res.normalized_text + "\n")
        return 0

    text = read_input_text(args)
    result = pipeline.preprocess(text)

    if args.format == "json":
        sys.stdout.write(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n")
    elif args.format == "diff":
        diff_view = generate_diff(result.raw_text, result.normalized_text)
        sys.stdout.write(diff_view)
        if not diff_view.endswith("\n"):
            sys.stdout.write("\n")
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
    if args.subcommand == "dataset":
        return run_dataset(args)
    if args.subcommand == "train":
        return run_train(args)
    if args.subcommand == "predict":
        return run_predict(args)
    if args.subcommand == "evaluate":
        return run_evaluate(args)
    if args.subcommand == "corpus":
        return run_corpus(args)

    return 0


def _resolve_model_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        cand = p / "model.pkl"
        if cand.exists():
            return cand
    return p


def run_train(args: argparse.Namespace) -> int:
    """Train sentiment model and write run directory artifacts (Task 20)."""
    from romanized_nepali_nlp.dataset import ColumnMap, DatasetRecord, load
    from romanized_nepali_nlp.evaluation import evaluate_predictions
    from romanized_nepali_nlp.sentiment import SentimentModel, SentimentModelConfig
    from romanized_nepali_nlp.splitting import SplitConfig, split

    try:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Config loading / resolution
        cfg_dict = {}
        if args.config:
            cfg_path = Path(args.config)
            if not cfg_path.exists():
                sys.stderr.write(f"Error: Config file not found: {cfg_path}\n")
                return 1
            cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))

        model_type = cfg_dict.get("model_type", args.model_type)
        analyzer = cfg_dict.get("analyzer", args.analyzer)
        min_n = cfg_dict.get("min_n", args.min_n)
        max_n = cfg_dict.get("max_n", args.max_n)
        min_df = cfg_dict.get("min_df", args.min_df)
        max_features = cfg_dict.get("max_features", args.max_features)
        class_weight = cfg_dict.get(
            "class_weight", None if args.class_weight == "none" else args.class_weight
        )
        c_val = float(cfg_dict.get("C", args.c))
        seed_val = int(cfg_dict.get("seed", args.seed))
        preprocess_val = (
            not args.no_preprocess if "preprocess" not in cfg_dict else cfg_dict["preprocess"]
        )

        cfg = SentimentModelConfig(
            model_type=model_type,
            analyzer=analyzer,
            ngram_range=(min_n, max_n),
            min_df=min_df,
            max_features=max_features,
            class_weight=class_weight,
            C=c_val,
            seed=seed_val,
            preprocess=preprocess_val,
        )

        # 2. Data loading
        col = ColumnMap(text=args.text_col, label=args.label_col, id=args.id_col)
        load_res = load(args.file, col=col)
        if not load_res.records:
            sys.stderr.write(f"Error: No valid records loaded from {args.file}\n")
            return 1

        train_records: List[DatasetRecord] = []
        val_records: List[DatasetRecord] = []

        if args.split_manifest:
            manifest_path = Path(args.split_manifest)
            if not manifest_path.exists():
                sys.stderr.write(f"Error: Manifest file not found: {manifest_path}\n")
                return 1
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_ids = set(manifest.get("record_ids", {}).get("train", []))
            val_ids = set(manifest.get("record_ids", {}).get("validation", []))
            for r in load_res.records:
                if r.id in train_ids:
                    train_records.append(r)
                elif r.id in val_ids:
                    val_records.append(r)
        else:
            split_res = split(load_res.records, SplitConfig(seed=seed_val))
            train_records = split_res.train
            val_records = split_res.validation

        sys.stdout.write(
            f"Training {cfg.model_type} baseline on {len(train_records)} records "
            f"(seed={cfg.seed}, preprocess={cfg.preprocess})...\n"
        )

        # 3. Model training
        model = SentimentModel(cfg).fit(train_records)

        # 4. Save artifacts
        model_out = run_dir / "model.pkl"
        model.save(model_out)

        run_manifest = {
            "config": cfg.to_dict(),
            "source_dataset": str(args.file),
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "classes": model.classes_,
            "training_counts": model.training_counts_,
        }
        (run_dir / "config.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

        top_feats = model.get_top_features(n=15)
        if top_feats:
            (run_dir / "top_features.json").write_text(
                json.dumps(top_feats, indent=2), encoding="utf-8"
            )

        log_lines = [
            f"Run Directory: {run_dir.resolve()}",
            f"Model Type: {cfg.model_type}",
            f"Train Count: {len(train_records)}",
            f"Classes: {model.classes_}",
        ]

        if val_records:
            val_preds = model.predict_records(val_records)
            with open(run_dir / "val_predictions.jsonl", "w", encoding="utf-8") as vf:
                for p in val_preds:
                    vf.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

            val_eval = evaluate_predictions(val_preds, records=val_records)
            val_eval.save_json(run_dir / "val_report.json")
            (run_dir / "val_report.md").write_text(val_eval.to_markdown(), encoding="utf-8")
            (run_dir / "val_report.html").write_text(val_eval.to_html(), encoding="utf-8")

            log_lines.append(f"Val Accuracy: {val_eval.accuracy:.4f}")
            log_lines.append(f"Val Macro-F1: {val_eval.macro_f1:.4f}")
            v_acc = f"{val_eval.accuracy:.4f}"
            v_f1 = f"{val_eval.macro_f1:.4f}"
            sys.stdout.write(f"Validation results: Accuracy={v_acc}, Macro-F1={v_f1}\n")

        (run_dir / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        sys.stdout.write(f"✓ Run artifacts successfully written to: {run_dir.resolve()}\n")
        return 0

    except Exception as e:
        sys.stderr.write(f"Error: Training run failed: {e}\n")
        return 1


def run_predict(args: argparse.Namespace) -> int:
    """Predict sentiment labels and scores using saved model (Task 20)."""
    from romanized_nepali_nlp.dataset import ColumnMap, DatasetRecord, load
    from romanized_nepali_nlp.sentiment import SentimentModel

    try:
        model_path = _resolve_model_path(args.model)
        if not model_path.exists():
            sys.stderr.write(f"Error: Model artifact not found: {model_path}\n")
            return 1

        model = SentimentModel.load(model_path)
        records: List[DatasetRecord] = []

        if args.file == "-":
            # Read from stdin
            lines = sys.stdin.read().splitlines()
            records = [
                DatasetRecord(id=f"input_{i}", text=line)
                for i, line in enumerate(lines)
                if line.strip()
            ]
        else:
            p = Path(args.file)
            if not p.exists():
                sys.stderr.write(f"Error: File not found: {args.file}\n")
                return 1
            if p.suffix.lower() in {".csv", ".jsonl"}:
                col = ColumnMap(text=args.text_col, id=args.id_col, label=args.label_col)
                res = load(p, col=col, validate=False)
                records = res.records
            else:
                lines = p.read_text(encoding="utf-8").splitlines()
                records = [
                    DatasetRecord(id=f"{p.stem}_{i}", text=text_line)
                    for i, text_line in enumerate(lines)
                    if text_line.strip()
                ]

        if not records:
            sys.stderr.write("Warning: No records found to predict.\n")
            return 0

        predictions = model.predict_records(records)

        # Output formatting
        out_handle = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        try:
            if args.format == "jsonl":
                for pred in predictions:
                    out_handle.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
            elif args.format == "csv":
                import csv

                writer = csv.writer(out_handle)
                classes = model.classes_
                writer.writerow(
                    ["id", "predicted_label", "confidence"]
                    + [f"score_{c}" for c in classes]
                    + ["text"]
                )
                for p in predictions:
                    conf = p.scores.get(p.predicted_label, 0.0)
                    scores = [p.scores.get(c, 0.0) for c in classes]
                    writer.writerow(
                        [p.id, p.predicted_label, f"{conf:.4f}"]
                        + [f"{s:.4f}" for s in scores]
                        + [p.raw_text or ""]
                    )
            else:  # table
                header = f"{'ID':<15} {'Predicted':<12} {'Conf':<8} Text"
                out_handle.write(header + "\n" + "-" * 60 + "\n")
                for p in predictions:
                    conf = p.scores.get(p.predicted_label, 0.0)
                    txt = (p.raw_text or "").replace("\n", " ")[:30]
                    out_handle.write(f"{p.id:<15} {p.predicted_label:<12} {conf:<8.2f} {txt}\n")
        finally:
            if args.output:
                out_handle.close()
                sys.stdout.write(f"Wrote {len(predictions)} predictions to {args.output}\n")
        return 0

    except Exception as e:
        sys.stderr.write(f"Error: Prediction failed: {e}\n")
        return 1


def run_evaluate(args: argparse.Namespace) -> int:
    """Evaluate saved predictions against gold labels (Tasks 20-21)."""
    from romanized_nepali_nlp.dataset import ColumnMap, load
    from romanized_nepali_nlp.evaluation import evaluate_predictions
    from romanized_nepali_nlp.sentiment import Prediction

    try:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            sys.stderr.write(f"Error: Predictions file not found: {pred_path}\n")
            return 1

        preds: List[Prediction] = []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                preds.append(
                    Prediction(
                        id=str(d["id"]),
                        predicted_label=str(d["predicted_label"]),
                        scores=d.get("scores", {}),
                        gold_label=d.get("gold_label"),
                        raw_text=d.get("raw_text") or d.get("text"),
                    )
                )

        dataset_records = None
        if args.dataset:
            ds_path = Path(args.dataset)
            if ds_path.exists():
                col = ColumnMap(text="text", id="id", label="label")
                res = load(ds_path, col=col, validate=False)
                dataset_records = res.records

        report = evaluate_predictions(preds, records=dataset_records)

        # Output formatting
        if args.format == "json":
            out_str = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
        elif args.format == "html":
            out_str = report.to_html()
        else:
            out_str = report.to_markdown() + "\n"

        if args.out:
            Path(args.out).write_text(out_str, encoding="utf-8")
            sys.stdout.write(f"Report written to {args.out}\n")
        else:
            sys.stdout.write(out_str)

        if args.export_errors:
            err_path = Path(args.export_errors)
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w", encoding="utf-8") as ef:
                for err in report.errors:
                    ef.write(json.dumps(err.to_dict(), ensure_ascii=False) + "\n")
            sys.stdout.write(f"Exported {len(report.errors)} misclassified errors to {err_path}\n")

        return 0

    except Exception as e:
        sys.stderr.write(f"Error: Evaluation failed: {e}\n")
        return 1


def run_corpus(args: argparse.Namespace) -> int:
    """Ingest, profile, audit, and triage external corpus records (Phase 5.5)."""
    from romanized_nepali_nlp.corpus import CorpusManifest, process_corpus_file
    from romanized_nepali_nlp.dataset import ColumnMap, load

    try:
        from datetime import datetime, timezone

        manifest = CorpusManifest(
            dataset=args.dataset_name,
            revision=args.revision,
            downloaded_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            rows_downloaded=0,
            source_license_field=args.license,
            upstream_attribution=args.attribution,
            local_sha256="",
            processing_version="0.1.0",
            notes=["Unlabeled external corpus", "YouTube comments in Latin script"],
        )
        bench_records = None
        if args.benchmark:
            b_path = Path(args.benchmark)
            if b_path.exists():
                b_res = load(
                    b_path,
                    col=ColumnMap(text="text", id="id", label="label"),
                    validate=False,
                )
                bench_records = b_res.records

        sys.stdout.write(f"Profiling and partitioning external corpus: {args.file}...\n")
        summary = process_corpus_file(
            args.file,
            args.output_dir,
            benchmark_records=bench_records,
            manifest=manifest,
        )
        sys.stdout.write(
            f"✓ Ingested {summary['total_rows']} rows. "
            f"Partitions: {summary['partitions']}. "
            f"Leakage quarantined: {summary['leakage_quarantined']}\n"
        )
        sys.stdout.write(f"Outputs written to: {Path(args.output_dir).resolve()}\n")
        return 0

    except Exception as e:
        sys.stderr.write(f"Error: Corpus processing failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
