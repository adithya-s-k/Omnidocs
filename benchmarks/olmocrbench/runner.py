"""
benchmarks/olmocrbench/runner.py
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

_STATE_FILE = "run_state.json"
_RAW_DIR = "raw_results"  # one JSON per model, written after inference completes


def _load_state(output_dir: Path) -> dict:
    p = output_dir / _STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(output_dir: Path, state: dict) -> None:
    (output_dir / _STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _save_raw_results(output_dir: Path, model_key: str, results: List[OlmResult]) -> None:
    raw_dir = output_dir / _RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = raw_dir / f"{model_key}.json"
    p.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")


def _load_raw_results(output_dir: Path, model_key: str) -> Optional[List[OlmResult]]:
    p = output_dir / _RAW_DIR / f"{model_key}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [OlmResult(**d) for d in data]
    except Exception:
        return None


def _pdf_page_to_image(pdf_bytes: bytes, page_num: int):
    from pdf2image import convert_from_bytes

    pages = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
    if not pages:
        raise ValueError("pdf2image returned no pages")
    return pages[0].convert("RGB")


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
        return "  ".join(parts)
    if check_type == "math":
        return payload.get("latex", payload.get("math", payload.get("text", "")))
    return ""


def run_cases(
    extractor,
    cases: List[OlmTestCase],
    model_key: str,
) -> List[OlmResult]:
    results: List[OlmResult] = []

    for case in cases:
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
                    model=getattr(out, "model_name", None) or model_key,
                    passed=passed,
                    latency_s=latency,
                    gt=gt,
                    predicted=predicted,
                )
            )
            tick = "✓" if passed else "✗"
            print(f"  {tick} [{case.split:<15}] [{case.check_type:<14}] {latency:.2f}s")

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


def aggregate(results: List[OlmResult], model_key: str) -> dict:
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

    all_checks = sorted({ct for m in all_metrics for ct in m["by_check"]})
    if all_checks:
        print(f"\n{div}\nPER-CHECK-TYPE BREAKDOWN\n{div}")
        check_header = f"\n{'Model':<18}" + "".join(f"{ct[:12]:>14}" for ct in all_checks)
        print(check_header)
        print("-" * (18 + 14 * len(all_checks)))
        for m in all_metrics:
            row = f"{m['model']:<18}"
            for ct in all_checks:
                row += f"{_pct(m['by_check'].get(ct)):>14}"
            print(row)
        print(div)


def run_olmocrbench_bench(
    model_keys: List[str],
    splits: Optional[List[str]] = None,
    max_per_split: Optional[int] = None,
    output_dir: Optional[Path] = None,
    rescore: bool = False,
) -> dict:
    """
    Run olmocrbench-bench locally.

    Args:
        model_keys:    Model keys from the registry.
        splits:        Which splits to evaluate (None = all 7).
        max_per_split: Limit test cases per split (None = all).
        output_dir:    Where to write results. Defaults to results/olmocrbench/<run_id>/.
        rescore:       Re-run scoring on raw results already saved to disk,
                       without re-running inference. Useful if the scorer
                       logic changes but inference is already done.
    """
    import datetime

    from benchmarks.olmocrbench.dataset import load_olmocr_bench

    unknown_models = [m for m in model_keys if m not in MODEL_REGISTRY]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Available: {list(MODEL_REGISTRY.keys())}")

    if splits is None:
        splits = OLM_SPLITS
    unknown_splits = [s for s in splits if s not in OLM_SPLITS]
    if unknown_splits:
        raise ValueError(f"Unknown splits: {unknown_splits}. Available: {OLM_SPLITS}")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("results") / "olmocrbench" / run_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    if rescore:
        print("Mode: RESCORE ONLY (reloading saved raw results, skipping inference)")

    state = _load_state(output_dir)
    completed_models: list = state.get("completed_models", [])

    all_metrics: List[dict] = []
    all_raw: Dict[str, list] = {}

    if rescore:
        # Load saved raw results and re-aggregate/re-score
        for model_key in model_keys:
            saved = _load_raw_results(output_dir, model_key)
            if saved is None:
                print(f"  [SKIP] {model_key}: no saved raw results at {output_dir / _RAW_DIR / model_key}.json")
                continue
            print(f"  Rescoring {model_key} ({len(saved)} cases from disk)")
            metrics = aggregate(saved, model_key)
            all_metrics.append(metrics)
            all_raw[model_key] = [asdict(r) for r in saved]

    else:
        # Load dataset once
        print(f"\nLoading olmocrbench-bench  splits={splits}  max_per_split={max_per_split or 'all'}")
        cases = load_olmocr_bench(splits, max_per_split=max_per_split)
        if not cases:
            print("No test cases loaded. Exiting.")
            sys.exit(1)
        print(f"Total test cases: {len(cases)}")

        for model_key in model_keys:
            if model_key in completed_models:
                print(f"\n  ↷ {model_key}: already completed (run_state.json), loading saved results")
                saved = _load_raw_results(output_dir, model_key)
                if saved:
                    metrics = aggregate(saved, model_key)
                    all_metrics.append(metrics)
                    all_raw[model_key] = [asdict(r) for r in saved]
                continue

            print(f"\n{'=' * 60}")
            print(f"  Running: {model_key}")
            print(f"{'=' * 60}")

            try:
                extractor = get_extractor(model_key)
            except Exception as exc:
                print(f"  [ERROR] Failed to load {model_key}: {exc}")
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
                _save_state(output_dir, {"completed_models": completed_models})
                continue

            results = run_cases(extractor, cases, model_key)
            metrics = aggregate(results, model_key)
            all_metrics.append(metrics)
            all_raw[model_key] = [asdict(r) for r in results]

            # Persist raw results and state before moving to next model
            _save_raw_results(output_dir, model_key, results)
            completed_models.append(model_key)
            _save_state(output_dir, {"completed_models": completed_models})

            passed_n = sum(1 for r in results if r.passed and not r.failed)
            print(
                f"\n  Done: {passed_n}/{len(results)} passed"
                f"  overall={metrics['overall'] * 100:.1f}%"
                f"  p50={metrics.get('latency_p50_s') or 0:.2f}s"
            )

            del extractor
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    print_report(all_metrics, splits)

    results_json = {
        "run_id": run_id if not rescore else state.get("run_id", output_dir.name),
        "benchmark": "olmocrbench",
        "execution": "local",
        "models": model_keys,
        "splits": splits,
        "inference": {
            m["model"]: {"cases_run": m["samples_run"], "cases_failed": m["samples_failed"]} for m in all_metrics
        },
        "eval_scores": {
            m["model"]: {
                "overall": m["overall"],
                "by_split": m["by_split"],
                "by_check": m["by_check"],
                "latency_p50_s": m["latency_p50_s"],
                "latency_p95_s": m["latency_p95_s"],
            }
            for m in all_metrics
        },
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")
    return results_json
