"""
benchmarks/olmocrbench/runner.py

Local inference + scoring loop for olmocrbench-bench.

For each model:
  1. Instantiate the extractor via the registry.
  2. For each OlmTestCase: render the PDF page → call extractor.extract() → score.
  3. Aggregate per-split and per-check-type pass rates.
  4. Print a leaderboard-style report and write results JSON.

Usage:
    python -m benchmarks.olmocrbench --models glmocr,qwen
    python -m benchmarks.olmocrbench --splits arxiv_math,table_tests --max-per-split 20
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.base import OlmResult, OlmTestCase
from benchmarks.olmocrbench.dataset import OLM_SPLITS, SPLIT_LABELS
from benchmarks.olmocrbench.scorer import score_case
from benchmarks.registry import MODEL_REGISTRY, get_extractor

# ---------------------------------------------------------------------------
# PDF → PIL image
# ---------------------------------------------------------------------------


def _pdf_page_to_image(pdf_bytes: bytes, page_num: int):
    """
    Render the first page of a single-page benchmark PDF to a PIL image.
    page_num refers to the original source document page, not this PDF.
    """
    from pdf2image import convert_from_bytes

    pages = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
    if not pages:
        raise ValueError("pdf2image returned no pages")
    return pages[0].convert("RGB")


# ---------------------------------------------------------------------------
# Ground-truth extractor (for logging)
# ---------------------------------------------------------------------------


def _extract_gt(payload: dict, check_type: str) -> str:
    if check_type in ("text_present", "text_absent"):
        return payload.get("text", "")
    if check_type == "reading_order":
        before = payload.get("before", payload.get("text_before", ""))
        after = payload.get("after", payload.get("text_after", ""))
        return f"BEFORE: {before!r}  →  AFTER: {after!r}"
    if check_type == "table":
        cell = payload.get("cell", "")
        parts = [f"cell={cell!r}"]
        for direction in ("up", "down", "left", "right"):
            val = payload.get(direction)
            if val is not None and str(val).strip():
                parts.append(f"{direction}={val!r}")
        if payload.get("top_heading"):
            parts.append(f"top_heading={payload['top_heading']!r}")
        if payload.get("left_heading"):
            parts.append(f"left_heading={payload['left_heading']!r}")
        return "  ".join(parts)
    if check_type == "math":
        return payload.get("latex", payload.get("math", payload.get("text", "")))
    return ""


# ---------------------------------------------------------------------------
# Core inference + scoring loop for one model
# ---------------------------------------------------------------------------


def run_cases(
    extractor,
    cases: List[OlmTestCase],
    model_key: str,
) -> List[OlmResult]:
    """
    Render each PDF page, run extractor.extract(), score the result.
    Returns a list of OlmResult objects.
    """
    results: List[OlmResult] = []

    for case in cases:
        # Render PDF page before starting timer — matches original pipeline
        try:
            img = _pdf_page_to_image(case.pdf_bytes, case.page_num)
        except Exception as exc:
            results.append(
                OlmResult(
                    case_id=case.case_id,
                    split=case.split,
                    check_type=case.check_type,
                    model=model_key,
                    passed=False,
                    latency_s=0.0,
                    failed=True,
                    error=f"PDF render failed: {exc}",
                )
            )
            print(f"  ✗ [{case.split:<15}] [{case.check_type:<14}] PDF render failed: {exc}")
            continue

        t0 = time.perf_counter()
        try:
            out = extractor.extract(img, output_format="markdown")
            predicted = (getattr(out, "plain_text", None) or out.content or "").strip()
            latency = time.perf_counter() - t0

            passed = score_case(predicted, case.check_type, case.payload, case_id=case.case_id)
            gt = _extract_gt(case.payload, case.check_type)

            results.append(
                OlmResult(
                    case_id=case.case_id,
                    split=case.split,
                    check_type=case.check_type,
                    # Use model_name from output if available, fall back to registry key
                    model=getattr(out, "model_name", None) or model_key,
                    passed=passed,
                    latency_s=latency,
                    gt=gt,
                    predicted=predicted,  # full output, no truncation
                )
            )

            tick = "✓" if passed else "✗"
            pred_preview = predicted[:200].replace("\n", "↵") + ("…" if len(predicted) > 200 else "")
            print(f"  {tick} [{case.split:<15}] [{case.check_type:<14}] {latency:.2f}s")
            print(f"      GT : {gt!r}")
            print(f"      OUT: {pred_preview!r}")

        except Exception as exc:
            latency = time.perf_counter() - t0
            results.append(
                OlmResult(
                    case_id=case.case_id,
                    split=case.split,
                    check_type=case.check_type,
                    model=model_key,
                    passed=False,
                    latency_s=latency,
                    failed=True,
                    error=str(exc),
                )
            )
            print(f"  ✗ [{case.split:<15}] [{case.check_type:<14}] FAILED: {exc}")
            traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: List[OlmResult], model_key: str) -> dict:
    """Compute per-split and per-check-type pass rates."""
    total = len(results)
    failed = [r for r in results if r.failed]
    scorable = [r for r in results if not r.failed]

    by_split: Dict[str, List[bool]] = defaultdict(list)
    by_check: Dict[str, List[bool]] = defaultdict(list)
    for r in scorable:
        by_split[r.split].append(r.passed)
        by_check[r.check_type].append(r.passed)

    all_passed = [r.passed for r in scorable]
    overall = sum(all_passed) / len(all_passed) if all_passed else 0.0

    lats = sorted(r.latency_s for r in results)
    n = len(lats)
    p50 = lats[int(0.50 * n)] if n else None
    p95 = lats[min(int(0.95 * n), n - 1)] if n else None

    return {
        "model": model_key,
        "overall": overall,
        "samples_run": total,
        "samples_failed": len(failed),
        "failure_rate": len(failed) / total if total else 0.0,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "by_split": {split: (sum(v) / len(v) if v else None) for split, v in by_split.items()},
        "by_check": {ct: (sum(v) / len(v) if v else None) for ct, v in by_check.items()},
        "n_by_split": {split: len(v) for split, v in by_split.items()},
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(all_metrics: List[dict], splits: List[str]) -> None:
    div = "=" * 120
    col_labels = [SPLIT_LABELS.get(s, s[:7]) for s in splits]
    header = (
        f"\n{'Model':<18}" + "".join(f"{lbl:>9}" for lbl in col_labels) + f"{'Overall':>10}  {'p50(s)':>7} {'Fail%':>6}"
    )

    print(f"\n{div}")
    print("olmocrbench-BENCH RESULTS  (binary pass/fail unit tests, % passed)")
    print(div + header)
    print("-" * 120)

    def _pct(v):
        return f"{v * 100:6.1f}%" if v is not None else "    n/a"

    for m in all_metrics:
        row = f"{m['model']:<18}"
        for split in splits:
            row += f"{_pct(m['by_split'].get(split)):>9}"
        row += f"{_pct(m['overall']):>10}"
        row += f"  {m['latency_p50_s'] or 0:>7.2f}"
        row += f"  {(m['failure_rate'] or 0) * 100:>5.1f}%"
        print(row)

        counts = "  ".join(f"{SPLIT_LABELS.get(s, s[:7])}={m['n_by_split'].get(s, 0)}" for s in splits)
        print(f"  {'':18}  ({counts})")

    print(div)
    print("↑ = higher is better  |  Each cell = % of unit tests passed in that split")
    print()

    all_checks = sorted({ct for m in all_metrics for ct in m["by_check"]})
    if all_checks:
        print(f"\n{div}")
        print("PER-CHECK-TYPE BREAKDOWN")
        print(div)
        check_header = f"\n{'Model':<18}" + "".join(f"{ct[:12]:>14}" for ct in all_checks)
        print(check_header)
        print("-" * (18 + 14 * len(all_checks)))
        for m in all_metrics:
            row = f"{m['model']:<18}"
            for ct in all_checks:
                row += f"{_pct(m['by_check'].get(ct)):>14}"
            print(row)
        print(div)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_olmocrbench_bench(
    model_keys: List[str],
    splits: Optional[List[str]] = None,
    max_per_split: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run olmocrbench-bench locally for the given models.

    Args:
        model_keys:    List of model keys from the registry.
        splits:        Which splits to evaluate (None = all 7).
        max_per_split: Limit test cases per split (None = all).
        output_dir:    Where to write raw_results.json and summary.json.
                       Defaults to results/olmocrbench/<run_id>/.

    Returns:
        A dict with all_metrics and raw_results.
    """
    import datetime

    from benchmarks.olmocrbench.dataset import load_olmocrbench_bench

    # Validate
    unknown_models = [m for m in model_keys if m not in MODEL_REGISTRY]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Available: {list(MODEL_REGISTRY.keys())}")

    if splits is None:
        splits = OLM_SPLITS
    unknown_splits = [s for s in splits if s not in OLM_SPLITS]
    if unknown_splits:
        raise ValueError(f"Unknown splits: {unknown_splits}. Available: {OLM_SPLITS}")

    # Resolve output directory
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("results") / "olmocrbench" / run_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load dataset once, shared across all models
    print(f"\nLoading olmocrbench-bench  splits={splits}  max_per_split={max_per_split or 'all'}")
    cases = load_olmocrbench_bench(splits, max_per_split=max_per_split)
    if not cases:
        print("No test cases loaded. Exiting.")
        sys.exit(1)
    print(f"Total test cases: {len(cases)}")

    all_metrics: List[dict] = []
    all_raw: Dict[str, list] = {}

    for model_key in model_keys:
        print(f"\n{'=' * 60}")
        print(f"  Running: {model_key}")
        print(f"{'=' * 60}")

        try:
            extractor = get_extractor(model_key)
        except Exception as exc:
            print(f"  [ERROR] Failed to load {model_key}: {exc}")
            # Record full failure for every case
            failed_results = [
                OlmResult(
                    case_id=c.case_id,
                    split=c.split,
                    check_type=c.check_type,
                    model=model_key,
                    passed=False,
                    latency_s=0.0,
                    failed=True,
                    error=str(exc),
                )
                for c in cases
            ]
            metrics = aggregate(failed_results, model_key)
            all_metrics.append(metrics)
            all_raw[model_key] = [asdict(r) for r in failed_results]
            continue

        results = run_cases(extractor, cases, model_key)
        metrics = aggregate(results, model_key)
        all_metrics.append(metrics)
        all_raw[model_key] = [asdict(r) for r in results]

        passed_n = sum(1 for r in results if r.passed and not r.failed)
        print(
            f"\n  Done: {passed_n}/{len(results)} passed"
            f"  overall={metrics['overall'] * 100:.1f}%"
            f"  p50={metrics.get('latency_p50_s') or 0:.2f}s"
        )

        # Free GPU memory before loading the next model
        del extractor
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # Print leaderboard-style report
    print_report(all_metrics, splits)

    # Write results to disk
    payload = {
        "run_id": run_id,
        "benchmark": "olmocrbench-bench",
        "splits": splits,
        "num_cases": len(cases),
        "models": model_keys,
        "metrics": all_metrics,
        "raw_results": all_raw,
    }

    raw_path = output_dir / "raw_results.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFull results saved to: {raw_path}")

    summary = {
        "run_id": run_id,
        "benchmark": "olmocrbench-bench",
        "splits": splits,
        "models": model_keys,
        "metrics": all_metrics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to:      {summary_path}")

    return payload
