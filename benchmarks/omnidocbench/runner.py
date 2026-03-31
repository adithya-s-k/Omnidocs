"""
benchmarks/omnidocbench/runner.py

Local inference loop for OmniDocBench.

For each model:
  1. Instantiate the extractor via the registry (or accept one directly).
  2. Iterate over all PageSamples, call extractor.extract(image).
  3. Write one .md file per page to results/omnidocbench/<run_id>/<model>/.
  4. After all pages, delegate to evaluator.py to run the official scoring.

Usage:
    python -m benchmarks.omnidocbench --models glmocr,qwen --max-samples 50
    python -m benchmarks.omnidocbench --models deepseek --no-eval
"""

from __future__ import annotations

import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.base import PageResult, PageSample
from benchmarks.registry import MODEL_REGISTRY, get_extractor

# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------


def run_inference(
    extractor,
    samples: List[PageSample],
    model_key: str,
) -> List[PageResult]:
    """
    Run a single extractor over all samples.
    Returns a list of PageResult objects.
    """
    from PIL import Image

    results: List[PageResult] = []

    for s in samples:
        # Open image before starting the timer — matches original pipeline exactly
        img = Image.open(io.BytesIO(s.image_bytes)).convert("RGB")
        t0 = time.perf_counter()
        try:
            out = extractor.extract(img, output_format="markdown")
            markdown = (getattr(out, "plain_text", None) or out.content or "").strip()
            latency = time.perf_counter() - t0
            results.append(
                PageResult(
                    image_name=s.image_name,
                    # Use model_name from output if available, fall back to registry key
                    model=getattr(out, "model_name", None) or model_key,
                    markdown=markdown,
                    latency_s=latency,
                )
            )
            print(f"  ✓ {s.image_name:<60} {latency:.2f}s  {len(markdown)} chars")
        except Exception as exc:
            latency = time.perf_counter() - t0
            results.append(
                PageResult(
                    image_name=s.image_name,
                    model=model_key,
                    markdown="",
                    latency_s=latency,
                    failed=True,
                    error=str(exc),
                )
            )
            print(f"  ✗ {s.image_name:<60} FAILED: {exc}")
            traceback.print_exc()

    return results


def write_md_files(
    results: List[PageResult],
    model_dir: Path,
) -> tuple[int, int]:
    """
    Write one .md file per page result into model_dir.
    Returns (written_count, failed_count).
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0

    for r in results:
        stem = Path(r.image_name).stem
        md_path = model_dir / f"{stem}.md"

        if r.failed or not r.markdown:
            # Write empty file so the eval pipeline knows the page was attempted
            md_path.write_text("", encoding="utf-8")
            failed += 1
        else:
            md_path.write_text(r.markdown, encoding="utf-8")
            written += 1

    return written, failed


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------


def run_omnidocbench(
    model_keys: List[str],
    max_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
    run_eval: bool = True,
    eval_repo_path: Optional[Path] = None,
    omnidocbench_json: Optional[Path] = None,
) -> dict:
    """
    Run OmniDocBench locally for the given models.

    Args:
        model_keys:        List of model keys from the registry.
        max_samples:       Limit pages loaded (None = full 1355 pages).
        output_dir:        Where to write .md files and summary.json.
                           Defaults to results/omnidocbench/<run_id>/.
        run_eval:          Whether to run the official pdf_validation.py
                           evaluation after inference. Requires the eval
                           repo to be cloned (handled automatically).
        eval_repo_path:    Path to the cloned OmniDocBench eval repo.
                           If None, defaults to benchmarks/omnidocbench/eval_repo/.
        omnidocbench_json: Path to OmniDocBench.json ground-truth file.
                           If None, auto-resolved from HF cache.

    Returns:
        A summary dict with per-model metrics (if eval ran) or page counts.
    """
    import datetime

    from benchmarks.omnidocbench.dataset import load_omnidocbench

    # Validate model keys
    unknown = [m for m in model_keys if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")

    # Resolve output directory
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("results") / "omnidocbench" / run_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load dataset
    print("\nLoading OmniDocBench dataset...")
    samples = load_omnidocbench(max_samples=max_samples)
    if not samples:
        print("No samples loaded. Exiting.")
        sys.exit(1)

    # Run inference for each model sequentially
    inference_summary: Dict[str, dict] = {}

    for model_key in model_keys:
        print(f"\n{'=' * 60}")
        print(f"  Running inference: {model_key}")
        print(f"{'=' * 60}")

        try:
            extractor = get_extractor(model_key)
        except Exception as exc:
            print(f"  [ERROR] Failed to load {model_key}: {exc}")
            inference_summary[model_key] = {"written": 0, "failed": len(samples), "load_error": str(exc)}
            continue

        results = run_inference(extractor, samples, model_key)

        model_dir = output_dir / model_key
        written, failed = write_md_files(results, model_dir)
        inference_summary[model_key] = {"written": written, "failed": failed}
        print(f"\n  Done: {written} written, {failed} failed → {model_dir}")

        # Free model memory before loading the next one
        del extractor
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # Print inference summary
    print(f"\n{'=' * 60}")
    print(f"INFERENCE COMPLETE — results in: {output_dir}")
    print(f"{'=' * 60}")
    for model_key, s in inference_summary.items():
        print(f"  {model_key:<18} {s['written']} pages written, {s['failed']} failed")

    # Run official evaluation
    eval_scores: Dict[str, dict] = {}
    if run_eval:
        from benchmarks.omnidocbench.evaluator import run_evaluation

        eval_scores = run_evaluation(
            run_output_dir=output_dir,
            model_keys=list(inference_summary.keys()),
            eval_repo_path=eval_repo_path,
            omnidocbench_json=omnidocbench_json,
        )

    # Write summary.json
    results_json = {
        "run_id": run_id,
        "benchmark": "omnidocbench",
        "execution": "local",
        "models": model_keys,
        "inference": inference_summary,
        "eval_scores": eval_scores,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")

    return results_json
