"""
benchmarks/omnidocbench/__main__.py

CLI entry point:
    python -m benchmarks.omnidocbench [options]

Examples:
    # Run all models, full dataset, with official eval
    python -m benchmarks.omnidocbench

    # Run specific models
    python -m benchmarks.omnidocbench --models glmocr,deepseek

    # Quick iteration — 10 pages, skip eval
    python -m benchmarks.omnidocbench --models qwen --max-samples 10 --no-eval

    # Custom output directory
    python -m benchmarks.omnidocbench --models nanonets --output-dir results/run_01

    # Provide eval repo and ground-truth JSON explicitly
    python -m benchmarks.omnidocbench \\
        --eval-repo-path /path/to/OmniDocBench \\
        --omnidocbench-json /path/to/OmniDocBench.json
"""

import argparse
import sys
from pathlib import Path

from benchmarks.omnidocbench.runner import run_omnidocbench
from benchmarks.registry import list_models


def main():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.omnidocbench",
        description="Run OmniDocBench locally on OmniDocs text extraction models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(f"Comma-separated model keys to benchmark (default: all). Available: {', '.join(list_models())}"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max pages to load from OmniDocBench (0 = full dataset, default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for .md files and summary.json (default: results/omnidocbench/<run_id>/).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip the official OmniDocBench evaluation step (inference only).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip inference; run eval on existing .md files in --output-dir.",
    )
    parser.add_argument(
        "--eval-repo-path",
        type=str,
        default="",
        help=(
            "Path to the cloned OmniDocBench eval repo. "
            "If absent, it will be cloned automatically to benchmarks/omnidocbench/eval_repo/."
        ),
    )
    parser.add_argument(
        "--omnidocbench-json",
        type=str,
        default="",
        help=("Path to OmniDocBench.json ground-truth file. Resolved automatically from HF cache if not provided."),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        help="Print available model keys and exit.",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:", list_models())
        sys.exit(0)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else list_models()

    run_omnidocbench(
        model_keys=model_keys,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        run_eval=not args.no_eval,
        eval_only=args.eval_only,
        eval_repo_path=Path(args.eval_repo_path) if args.eval_repo_path else None,
        omnidocbench_json=Path(args.omnidocbench_json) if args.omnidocbench_json else None,
    )


if __name__ == "__main__":
    main()
