"""
benchmarks/multilingual/main.py

CLI entry point for the NayanaOCRBench multilingual benchmark.

    python -m benchmarks.multilingual [options]

Examples:
    # Run all models, all 22 languages, full dataset
    python -m benchmarks.multilingual

    # Run specific models and languages
    python -m benchmarks.multilingual --models glmocr,qwen --languages en,hi,kn

    # Quick iteration — 5 pages per language, skip eval
    python -m benchmarks.multilingual --models glmocr --max-per-language 5 --no-eval

    # Custom output directory
    python -m benchmarks.multilingual --models nanonets --output-dir results/multilingual_run01

    # Provide eval repo path explicitly
    python -m benchmarks.multilingual --eval-repo-path /path/to/OmniDocBench
"""

import argparse
import sys
from pathlib import Path

from benchmarks.multilingual.dataset import ALL_LANGUAGES
from benchmarks.multilingual.runner import run_multilingual
from benchmarks.registry import list_models


def main():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.multilingual",
        description="Run NayanaOCRBench multilingual benchmark on OmniDocs text extraction models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(f"Comma-separated model keys to benchmark (default: all). Available: {', '.join(list_models())}"),
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="",
        help=(f"Comma-separated language codes to evaluate (default: all). Available: {', '.join(ALL_LANGUAGES)}"),
    )
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=0,
        help="Max pages to load per language (0 = all, default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for .md files and summary.json (default: results/multilingual/<run_id>/).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip the official OmniDocBench evaluation step (inference only).",
    )
    parser.add_argument(
        "--eval-repo-path",
        type=str,
        default="",
        help=(
            "Path to the cloned OmniDocBench eval repo. "
            "Cloned automatically to benchmarks/multilingual/eval_repo/ if not provided."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        help="Print available model keys and exit.",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        default=False,
        help="Print available language codes and exit.",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:", list_models())
        sys.exit(0)

    if args.list_languages:
        print("Available languages:", ALL_LANGUAGES)
        sys.exit(0)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else list_models()
    languages = (
        [lang.strip() for lang in args.languages.split(",") if lang.strip()]
        if args.languages
        else None  # None = all languages, resolved inside run_multilingual
    )

    run_multilingual(
        model_keys=model_keys,
        languages=languages,
        max_per_language=args.max_per_language if args.max_per_language > 0 else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        run_eval=not args.no_eval,
        eval_repo_path=Path(args.eval_repo_path) if args.eval_repo_path else None,
    )


if __name__ == "__main__":
    main()
