"""
benchmarks/olmocrbench/__main__.py
"""

import argparse
import sys
from pathlib import Path

from benchmarks.olmocrbench.dataset import OLM_SPLITS, SPLIT_LABELS
from benchmarks.olmocrbench.runner import run_olmocrbench_bench
from benchmarks.registry import list_models


def main():
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.olmocrbench",
        description="Run olmocrbench-bench locally on OmniDocs text extraction models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(f"Comma-separated model keys (default: all). Available: {', '.join(list_models())}"),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help=f"Comma-separated split names (default: all 7). Available: {', '.join(OLM_SPLITS)}",
    )
    parser.add_argument("--max-per-split", type=int, default=0, help="Max test cases per split (0 = all).")
    parser.add_argument(
        "--output-dir", type=str, default="", help="Output directory (default: results/olmocrbench/<run_id>/)."
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        default=False,
        help=(
            "Skip inference; reload saved raw results from --output-dir and re-aggregate. "
            "Useful when scorer logic changes but inference is already done."
        ),
    )
    parser.add_argument(
        "--list-info",
        action="store_true",
        default=False,
        help="Print available models and splits, then exit.",
    )

    args = parser.parse_args()

    if args.list_info:
        print("\nAvailable models:", list_models())
        print("\nAvailable splits:")
        for s in OLM_SPLITS:
            print(f"  {s:<20} → leaderboard column: {SPLIT_LABELS[s]}")
        sys.exit(0)

    if args.rescore and not args.output_dir:
        print("--rescore requires --output-dir pointing to a previous run.")
        sys.exit(1)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else list_models()
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else None

    run_olmocrbench_bench(
        model_keys=model_keys,
        splits=split_names,
        max_per_split=args.max_per_split if args.max_per_split > 0 else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        rescore=args.rescore,
    )


if __name__ == "__main__":
    main()
